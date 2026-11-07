"""Matched task-reader math for ordinary frozen and end-to-end DINOv2 controls."""

import math

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


def new_head(task, seed):
    # CPU construction is isolated from encoder construction and global sampler RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if task == 'FSOD':
            from runs.shared_supervised_readout import make_fsod_head
            from runs.interp_tiles_fsod_transfer import TiledRoIPool

            head = make_fsod_head()
            head.roi_heads.box_roi_pool = TiledRoIPool()
        elif task == 'Marine':
            head = nn.Conv2d(384, 12, 1)
        elif task == 'NKSID':
            head = nn.Linear(384, 8)
        else:
            head = nn.Conv2d(384, 50, 1)
            with torch.no_grad():
                head.bias[:10].fill_(math.log(.01 / .99))
    return head


def configure_encoder(encoder, arm):
    encoder.train(arm == 'e2e').requires_grad_(arm == 'e2e')
    for name, parameter in encoder.named_parameters():
        if name.endswith('mask_token'):
            parameter.requires_grad_(False)


def body_optimizer(encoder):
    decay, no_decay = [], []
    for name, parameter in encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        target = decay if parameter.ndim > 1 and not name.endswith(('cls_token', 'pos_embed')) else no_decay
        target.append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': .05}, {'params': no_decay, 'weight_decay': 0.}],
                              lr=2e-5, betas=(.9, .999), eps=1e-8)


def body_learning_rate(completed_updates, total_updates):
    step = completed_updates + 1
    if step <= 200:
        return 2e-5 * step / 200
    fraction = (step - 200) / (total_updates - 200)
    return 2e-5 * (.01 + .99 * (1 + math.cos(math.pi * fraction)) / 2)


def head_optimizer(task, head):
    if task == 'FSOD':
        return torch.optim.SGD(head.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
    return torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8)


def arm_features(encoder, images, task, arm):
    if arm == 'frozen':
        with torch.no_grad():
            patches = encoder(images)['patch_tokens']
    else:
        patches = encoder(images)['patch_tokens']
    if task == 'NKSID':
        return patches.mean(1)
    grid = 32 if task in ('FSOD', 'Marine') else 16
    return patches.transpose(1, 2).reshape(len(images), 384, grid, grid).contiguous()


def task_loss(task, head, features, targets):
    if task == 'FSOD':
        return sum(head(features, targets).values())
    if task == 'UATD':
        from sonar_encoder.training.uatd_head import dense_loss

        return dense_loss(head(features), targets)
    logits = head(features)
    if task == 'Marine':
        logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
    return F.cross_entropy(logits, targets.long())


def normalized_input(row, normalization, task):
    from sonar_encoder.training.model import prepare_images

    with Image.open(row['image']) as source:
        gray = np.array(source.convert('L'), dtype=np.uint8)
    sample = {'intensity': torch.from_numpy(gray).float().unsqueeze(0) / 255,
              'valid': torch.ones(gray.shape, dtype=torch.bool)}
    image = prepare_images([sample], normalization, None)
    if task in ('FSOD', 'Marine'):
        image = F.interpolate(image, size=(448, 448), mode='bilinear', align_corners=False, antialias=True)
    return image[0]


def make_targets(task, rows):
    if task == 'UATD':
        return [{'boxes': torch.tensor(row['boxes'], dtype=torch.float32).reshape(-1, 4),
                 'labels': torch.tensor(row['labels'], dtype=torch.int64)} for row in rows]
    if task == 'FSOD':
        from runs.cfc_frozen_detector import scale_boxes

        targets = []
        for row in rows:
            boxes = torch.tensor([a['bbox'] for a in row['annotations']], dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            targets.append({'boxes': scale_boxes(boxes, (row['height'], row['width']), (224, 224)),
                            'labels': torch.tensor([a['category_id'] for a in row['annotations']], dtype=torch.int64)})
        return targets
    if task == 'NKSID':
        return torch.tensor([row['classID'] for row in rows], dtype=torch.int64)
    masks = []
    for row in rows:
        with Image.open(row['mask']) as source:
            masks.append(torch.from_numpy(np.array(source, dtype=np.uint8)))
    return torch.stack(masks)


def batch_targets(task, targets, indices, device):
    if task == 'FSOD':
        return [{key: value.to(device) for key, value in targets[i].items()} for i in indices]
    if task == 'UATD':
        return tuple(value[indices].to(device) for value in targets)
    return targets[indices].to(device)


def decode(task, head, features, rows):
    if task == 'FSOD':
        from runs.shared_supervised_readout import prediction_rows

        return prediction_rows(head(features), rows)
    logits = head(features)
    assert torch.isfinite(logits).all(), 'Nonfinite task logits'
    if task == 'UATD':
        from sonar_encoder.training.uatd_head import decode_detections

        return decode_detections(logits)
    if task == 'Marine':
        return F.interpolate(logits, size=(480, 320), mode='bilinear', align_corners=False).argmax(1).byte().cpu().numpy()
    return logits.cpu().numpy()


def original_scores(task, predictions, targets, coco=None):
    if task == 'FSOD':
        from runs.shared_supervised_json_readout import canonical_evaluate
        from runs.shared_supervised_readout import independent_ap50, prediction_list

        values = prediction_list(predictions)
        canonical, independent = canonical_evaluate(coco, values), independent_ap50(coco, values)
        for category in map(str, range(1, 11)):
            assert abs(canonical['per_class'][category]['AP50'] - independent['per_class'][category]['AP50']) < 1e-10
            for key in ('gt_support', 'matched_gt', 'true_positives', 'false_positives', 'ignored_detections'):
                assert canonical['per_class'][category][key] == independent['per_class'][category][key]
        return canonical, independent
    if task == 'UATD':
        from sonar_encoder.training.uatd_head import detection_metrics
        from runs.recompute_uatd_area_results import dense_metrics

        canonical = detection_metrics(predictions, targets)
        numpy_predictions = [{key: value.numpy() for key, value in row.items()} for row in predictions]
        independent = dense_metrics(numpy_predictions, [row['boxes'].numpy() for row in targets],
                                    [row['labels'].numpy() for row in targets])
        for key in canonical:
            np.testing.assert_allclose(canonical[key], independent[key], rtol=0, atol=1e-12)
        return canonical, independent
    from sonar_encoder.training.readout import confusion_matrix, metrics_from_confusion

    labels = predictions.argmax(1) if task == 'NKSID' else predictions
    classes = 8 if task == 'NKSID' else 12
    canonical = metrics_from_confusion(confusion_matrix(targets, torch.as_tensor(labels), classes),
                                       'nksid' if task == 'NKSID' else 'marine')
    matrix = np.bincount(targets.numpy().astype(np.int64).ravel() * classes + labels.ravel(),
                         minlength=classes**2).reshape(classes, classes)
    assert canonical['confusion_matrix'] == matrix.tolist()
    tp, actual, guessed = matrix.diagonal().astype(np.float64), matrix.sum(1), matrix.sum(0)
    independent = {'confusion_matrix': matrix.tolist(), 'support': actual.tolist()}
    if task == 'NKSID':
        f1 = 2 * tp / np.maximum(actual + guessed, 1)
        independent.update(macro_f1=float(f1.mean()), accuracy=float(tp.sum() / actual.sum()),
                           per_class_f1=f1.tolist(), per_class_precision=(tp / np.maximum(guessed, 1)).tolist(),
                           per_class_recall=(tp / np.maximum(actual, 1)).tolist())
    else:
        iou = tp / np.maximum(actual + guessed - tp, 1)
        independent.update(miou12=float(iou.mean()), debris_miou10=float(iou[1:11].mean()), per_class_iou=iou.tolist())
    for key in independent:
        np.testing.assert_allclose(canonical[key], independent[key], rtol=0, atol=1e-12)
    return canonical, independent
