"""Finite multisource training and taskwise adaptation of the complete shared GSR."""
import argparse
import contextlib
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import time
import traceback
import types

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

from runs import gsr_shared_complete as previous
from runs import shared_supervised_readout as detection_reader
from runs import downstream_vit_finetuning_readout as reader
from runs import selective_reencoding as original
from runs import gsr_detail_readout as dense

BASE = original.BASE
RUN = BASE/'gsr_multidata_expansion_v1'
SOURCE = ('FSOD', 'Marine', 'Xing_detection', 'Xing_segmentation', 'LSOD', 'CFC22')
TARGETS = ('FSOD', 'Marine', 'UATD', 'NKSID', 'Xing_detection', 'Xing_segmentation', 'LSOD', 'HighResolution', 'Diver', 'CFC26_Mini', 'CFC22')
write = previous.write


def guard(deadline):
    values = {'RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > deadline:
        raise TimeoutError('Expanded GSR fixed deadline')
    if values['RSS_bytes'] > 48*1024**3 or values['GPU_reserved_bytes'] > 80*1024**3:
        raise RuntimeError('Expanded GSR resource cap '+str(values))
    return values


class GSR(previous.GSR):
    def __init__(self, config):
        super().__init__(config)
        self.encoder_images = self.encoder_calls = 0

    def global_field(self, images):
        self.encoder_images += len(images); self.encoder_calls += 1
        return super().global_field(images)

    def field(self, images, head, kind, policy='learned', actions=None):
        g = self.global_field(images)
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        if policy == 'global':
            return base
        if actions is None:
            if policy == 'uncertainty':
                if kind == 'detection':
                    probability = head.rpn.head([base])[0][0].sigmoid().clamp(1e-6, 1-1e-6)
                    entropy = -(probability*probability.log()+(1-probability)*(1-probability).log()).mean(1, keepdim=True)
                elif kind == 'classification':
                    patches = F.adaptive_avg_pool2d(base, (2, 2)).flatten(2).transpose(1, 2)
                    probability = head(patches).softmax(-1).clamp_min(1e-6)
                    actions = (-(probability*probability.log()).sum(-1)).argmax(1)
                    entropy = None
                else:
                    logits = head(base)[:, :10] if kind == 'uatd' else head(base)
                    if kind == 'segmentation':
                        probability = logits.softmax(1).clamp_min(1e-6)
                        entropy = -(probability*probability.log()).sum(1, keepdim=True)
                    elif kind == 'point_regression':
                        probability = logits.flatten(2).softmax(-1).reshape_as(logits).clamp_min(1e-6)
                        entropy = -probability*probability.log()
                    else:
                        probability = logits.sigmoid().clamp(1e-6, 1-1e-6)
                        entropy = -(probability*probability.log()+(1-probability)*(1-probability).log()).mean(1, keepdim=True)
                if entropy is not None:
                    actions = F.adaptive_avg_pool2d(entropy, (2, 2)).flatten(1).argmax(1)
            else:
                actions = self.selector(g.detach()).argmax(1)
        active = torch.where(actions >= 0)[0]
        if len(active):
            large = F.interpolate(images[active], (448, 448), mode='bilinear', align_corners=False, antialias=True)
            tiles = torch.stack([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in original.QUADS], 1)
            local = self.global_field(tiles[torch.arange(len(active), device=images.device), actions[active]])
            base = base.clone()
            base[active] = self.corrected(g[active], local, actions[active])
        return base


def new_head(task, spec):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        if task == 'UATD':
            return reader.new_head('UATD', 0)
        if spec['kind'] == 'detection':
            head = reader.new_head('FSOD', 0)
            if spec['classes'] != 10:
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                head.roi_heads.box_predictor = FastRCNNPredictor(1024, spec['classes']+1)
            return head
        if spec['kind'] == 'classification':
            return nn.Linear(384, spec['classes'])
        return nn.Conv2d(384, 1 if spec['kind'] == 'point_regression' else spec['classes'], 1)


def targets_for(task, spec, split):
    rows = spec['rows'][split]
    if task in ('FSOD', 'Marine', 'UATD', 'NKSID'):
        values = reader.make_targets(task, rows)
        if task == 'UATD':
            return dense.box_targets32(values, 'cpu')
        return values
    if spec['kind'] == 'detection':
        values = []
        for row in rows:
            if 'annotations' in row:
                boxes = torch.tensor([a['bbox'] for a in row['annotations']], dtype=torch.float32).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]
                labels = [a['category_id'] for a in row['annotations']]
            else:
                boxes = torch.tensor([a['bbox'] for a in row['boxes_xyxy']], dtype=torch.float32).reshape(-1, 4)
                labels = [a['class_id'] for a in row['boxes_xyxy']]
            boxes = boxes*torch.tensor([224/row['width'], 224/row['height']]*2)
            values.append({'boxes': boxes, 'labels': torch.tensor(labels, dtype=torch.int64)})
        return values
    if spec['kind'] == 'point_regression':
        return torch.tensor([[r['point_xy'][0]/r['width'], r['point_xy'][1]/r['height']] for r in rows], dtype=torch.float32)
    values = []
    for row in rows:
        if task == 'Xing_segmentation':
            with Image.open(row['original_mask']) as image:
                rgb = np.array(image.convert('RGB'), dtype=np.uint8)
            mask = np.full(rgb.shape[:2], 255, np.uint8)
            for i, color in enumerate(((0, 0, 0), (51, 221, 255), (250, 50, 83))):
                mask[(rgb == color).all(2)] = i
            if (mask == 255).any():
                raise ValueError('Unknown Xing annotation palette color')
        else:
            layers = []
            for name in ('obstacle', 'ground'):
                image = Image.new('L', (row['width'], row['height']), 0)
                draw = ImageDraw.Draw(image)
                for polygon in row['polygons']:
                    if polygon['label'] == name:
                        draw.polygon([tuple(p) for p in polygon['points']], fill=1)
                layers.append(np.array(image, dtype=bool))
            mask = np.full(layers[0].shape, 255, np.uint8)
            mask[layers[0] & ~layers[1]] = 0
            mask[layers[1] & ~layers[0]] = 1
        values.append(torch.from_numpy(mask))
    return values


def batch_targets(task, spec, labels, ids):
    if spec['kind'] == 'detection':
        return [{k: v.cuda() for k, v in labels[i].items()} for i in ids]
    if task == 'UATD':
        return tuple(v[ids].cuda() for v in labels)
    if spec['kind'] == 'segmentation' and task != 'Marine':
        return [labels[i].cuda() for i in ids]
    return labels[ids].cuda()


def point_prediction(head, field):
    probability = head(field).flatten(1).softmax(1)
    y, x = torch.meshgrid((torch.arange(32, device=field.device)+.5)/32,
                          (torch.arange(32, device=field.device)+.5)/32, indexing='ij')
    coordinates = torch.stack((x, y), -1).reshape(1024, 2)
    return probability@coordinates


def task_loss(task, spec, head, field, target):
    kind = spec['kind']
    if kind == 'detection':
        return sum(head(field, target).values())
    if task == 'UATD':
        return dense.dense_loss32(head(field), target, target[2].sum())
    if kind == 'classification':
        return F.cross_entropy(head(field.mean((2, 3))), target)
    if kind == 'point_regression':
        return F.smooth_l1_loss(point_prediction(head, field), target)
    if task == 'Marine':
        return reader.task_loss('Marine', head, field, target)
    logits = head(field)
    return torch.stack([F.cross_entropy(F.interpolate(logits[i:i+1], tuple(y.shape), mode='bilinear', align_corners=False),
                                        y[None].long(), ignore_index=255) for i, y in enumerate(target)]).mean()


def coco_labels(spec, split):
    images, annotations = [], []
    for row in spec['rows'][split]:
        images.append({'id': row['image_id'], 'width': row['width'], 'height': row['height']})
        for original_box in row['annotations'] if 'annotations' in row else row['boxes_xyxy']:
            if 'category_id' in original_box:
                box, cls = original_box['bbox'], original_box['category_id']
                crowd = original_box['iscrowd']; area = original_box['area']
            else:
                a, b, c, d = original_box['bbox']; box = [a, b, c-a, d-b]
                cls, crowd = original_box['class_id'], 0; area = box[2]*box[3]
            annotations.append({'id': len(annotations)+1, 'image_id': row['image_id'], 'bbox': box,
                'category_id': cls, 'area': area, 'iscrowd': crowd})
    return {'info': {}, 'images': images, 'annotations': annotations,
            'categories': [{'id': i, 'name': spec['class_names'][str(i)] if 'class_names' in spec else str(i)} for i in range(1, spec['classes']+1)]}


def scores(task, spec, prediction, labels, split):
    if spec['kind'] == 'detection':
        coco = coco_labels(spec, split)
        values = detection_reader.prediction_list(prediction)
        # Preserve existing scorer mathematics; this task supplies its actual class list.
        namespace = {**detection_reader.canonical_evaluate.__globals__, 'CATEGORIES': list(range(1, spec['classes']+1))}
        canonical = types.FunctionType(detection_reader.canonical_evaluate.__code__, namespace)
        independent = types.FunctionType(detection_reader.independent_ap50.__code__, namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            result = canonical(coco, values)
        other = independent(coco, values)
        for category in result['per_class']:
            result['per_class'][category]['ordering']['image_ids'] = [int(x) for x in result['per_class'][category]['ordering']['image_ids']]
            if abs(result['per_class'][category]['AP50']-other['per_class'][category]['AP50']) > 1e-10:
                raise ValueError('Independent native AP50 disagreement')
        return result, other
    if task == 'UATD':
        original_targets = reader.make_targets('UATD', spec['rows'][split])
        return reader.original_scores('UATD', prediction, original_targets)
    if spec['kind'] == 'classification':
        return reader.original_scores('NKSID', prediction, labels)
    if spec['kind'] == 'point_regression':
        sizes = np.array([[r['width'], r['height']] for r in spec['rows'][split]], dtype=np.float64)
        true = np.array([r['point_xy'] for r in spec['rows'][split]], dtype=np.float64)
        error = np.linalg.norm(prediction.astype(np.float64)*sizes-true, axis=1)
        return {'mean_error_pixels': float(error.mean()), 'median_error_pixels': float(np.median(error)),
                'normalized_mean_error': float(np.linalg.norm(prediction.astype(np.float64)-true/sizes, axis=1).mean()), 'count': len(error)}, {'per_image_error_pixels': error.tolist()}
    classes = spec['classes']; matrix = np.zeros((classes, classes), np.int64)
    for guess, truth in zip(prediction, labels):
        truth = truth.numpy(); valid = truth != 255
        matrix += np.bincount(truth[valid].astype(np.int64)*classes+guess[valid], minlength=classes**2).reshape(classes, classes)
    tp, actual, guessed = matrix.diagonal().astype(np.float64), matrix.sum(1), matrix.sum(0)
    iou = tp/np.maximum(actual+guessed-tp, 1)
    result = {'miou': float(iou.mean()), 'per_class_iou': iou.tolist(), 'confusion_matrix': matrix.tolist(), 'support': actual.tolist()}
    if task == 'Marine':
        result.update(miou12=float(iou.mean()), debris_miou10=float(iou[1:11].mean()))
    return result, {'confusion_matrix': matrix.tolist(), 'support': actual.tolist()}


@torch.no_grad()
def evaluate(task, spec, model, head, images, labels, batch, deadline, policy, split='validation'):
    model.eval(); head.eval(); rows = spec['rows'][split]; pieces = []
    random_actions = np.random.RandomState(0).randint(0, 4, len(rows))
    for start in range(0, len(rows), batch):
        end = min(start+batch, len(rows)); actions = None
        if policy == 'random':
            actions = torch.as_tensor(random_actions[start:end], device='cuda')
        elif policy.startswith('fixed'):
            actions = torch.full((end-start,), int(policy[-1]), device='cuda', dtype=torch.long)
        field = model.field(images[start:end], head, 'uatd' if task == 'UATD' else spec['kind'], policy, actions)
        if spec['kind'] == 'detection':
            pieces.append(reader.decode('FSOD', head, field, rows[start:end]))
        elif task == 'UATD':
            pieces.extend(dense.decode32(head(field)))
        elif spec['kind'] == 'classification':
            pieces.append(head(field.mean((2, 3))).cpu().numpy())
        elif spec['kind'] == 'point_regression':
            pieces.append(point_prediction(head, field).cpu().numpy())
        else:
            logits = head(field)
            for i, target in enumerate(labels[start:end]):
                pieces.append(F.interpolate(logits[i:i+1], tuple(target.shape), mode='bilinear', align_corners=False).argmax(1)[0].byte().cpu().numpy())
        guard(deadline)
    prediction = pieces if task == 'UATD' or spec['kind'] == 'segmentation' else np.concatenate(pieces)
    result, independent = scores(task, spec, prediction, labels, split)
    return result, independent, prediction


def source_training(model, p, tasks):
    deadline = min(p['deadline_unix'], time.time()+10800)
    model.encoder.eval().requires_grad_(False); model.selector.requires_grad_(False)
    images = {t: torch.from_numpy(np.load(tasks[t]['views']['train'], mmap_mode='r', allow_pickle=False)).cuda() for t in SOURCE}
    labels = {t: targets_for(t, tasks[t], 'train') for t in SOURCE}
    heads = nn.ModuleDict({t: new_head(t, tasks[t]).cuda() for t in SOURCE})
    optimizers = {t: reader.head_optimizer('FSOD' if tasks[t]['kind'] == 'detection' else 'Marine', heads[t]) for t in SOURCE}
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=.001, weight_decay=.0001)
    plans = np.load(RUN/'source_plan.npz', allow_pickle=False)
    begun = time.monotonic()
    with (RUN/'source_training.jsonl').open('w') as log:
        for update in range(6000):
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = previous.shared.native.module_lr(update, 6000, .001)
            values = {}
            for task in SOURCE:
                head = heads[task]; head.train(); opt = optimizers[task]; opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = (.005 if update < 4000 else .0005 if update < 5500 else .00005) if tasks[task]['kind'] == 'detection' else previous.shared.native.module_lr(update, 6000, .001)
                ids = plans[task][update]; actions = plans[task+'_actions'][update]; value_sum = 0.
                for start in range(0, len(ids), 4):
                    indices = ids[start:start+4].tolist()
                    action = torch.as_tensor(actions[start:start+4], device='cuda')
                    field = model.field(images[task][indices], head, tasks[task]['kind'], actions=action)
                    target = batch_targets(task, tasks[task], labels[task], indices)
                    value = task_loss(task, tasks[task], head, field, target)*len(indices)/len(ids)/len(SOURCE)
                    value.backward(); value_sum += float(value.detach())
                values[task] = value_sum
                opt.step()
            norm = torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); guard(deadline)
            if update == 0:
                write(RUN/'first_source_update.json', {'updates': 1, 'source_tasks': list(SOURCE),
                    'original_DINO_trainable': 0, 'shared_fusion_gradient_norm': float(norm),
                    'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
                    'source_macro': p['source_macro_per_task'], 'field': list(field.shape)})
            if (update+1) % 100 == 0:
                row = {'updates': update+1, 'losses': values, 'wall_seconds': time.monotonic()-begun,
                       'encoder_images': model.encoder_images, **guard(deadline)}
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
    write(RUN/'source_report.json', {'status': 'COMPLETE_EXPANDED_FROZEN_BACKBONE_SOURCE6000',
        'visits_per_task': p['source_visits'], 'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'wall_seconds': time.monotonic()-begun, 'original_DINO_updates': 0, **guard(deadline)})
    model.eval(); model.fusion.requires_grad_(False)
    banks, gain_targets, fit_ids, held_ids = {}, {}, {}, {}
    for task in SOURCE:
        subset = plans[task+'_gain_subset']; fields = []; gains = []
        heads[task].train(tasks[task]['kind'] == 'detection')
        with torch.no_grad():
            for number, i in enumerate(subset):
                x = images[task][int(i):int(i)+1]; g = model.global_field(x)
                large = F.interpolate(x, (448, 448), mode='bilinear', align_corners=False, antialias=True)
                crops = torch.cat([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in original.QUADS])
                local = model.global_field(crops); target = batch_targets(task, tasks[task], labels[task], [int(i)])
                losses = []
                for j in range(-1, 4):
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(10000+int(i))
                        action = torch.tensor([j], device='cuda')
                        field = model.corrected(g, local[max(j, 0):max(j, 0)+1], action)
                        losses.append(float(task_loss(task, tasks[task], heads[task], field, target)))
                fields.append(g.cpu()); gains.append([losses[0]-v for v in losses[1:]])
                if number % 100 == 0:
                    guard(deadline)
            raw = torch.tensor(gains, device='cuda')
            np.save(RUN/(task+'_signed_source_gains.npy'), raw.cpu().numpy())
            banks[task] = torch.cat(fields).cuda()
            order = plans[task+'_selector_order']; n = int(.8*len(order))
            fit_ids[task], held_ids[task] = torch.from_numpy(order[:n]).cuda(), torch.from_numpy(order[n:]).cuda()
            centered = raw-raw.mean(1, keepdim=True)
            gain_targets[task] = centered/centered[fit_ids[task]].std().clamp_min(1e-6)
        del images[task], labels[task]
    del heads, optimizers, optimizer
    model.selector.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.selector.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(31)
    steps = math.ceil(max(len(ids) for ids in fit_ids.values())/64)
    for epoch in range(100):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for task in SOURCE:
                ids = fit_ids[task][torch.randint(len(fit_ids[task]), (64,), generator=generator).cuda()]
                guess = model.selector(banks[task][ids]); centered = guess-guess.mean(1, keepdim=True)
                (F.mse_loss(centered, gain_targets[task][ids])/len(SOURCE)).backward()
            optimizer.step()
        guard(deadline)
    held = {}
    with torch.no_grad():
        for task in SOURCE:
            raw = torch.from_numpy(np.load(RUN/(task+'_signed_source_gains.npy'))).cuda()
            ids = held_ids[task]; choice = model.selector(banks[task][ids]).argmax(1)
            picked = raw[ids, choice]
            held[task] = {'fit': len(fit_ids[task]), 'held': len(ids), 'selected_gain': float(picked.mean()),
                'uniform_gain': float(raw[ids].mean()), 'oracle_gain': float(raw[ids].max(1).values.mean())}
    write(RUN/'selector_report.json', {'status': 'COMPLETE_FIXED_SOURCE_ONLY_SELECTOR', 'epochs': 100,
        'held_diagnostics': held, 'heldout_target_labels_used': False, 'source_wall_seconds': time.monotonic()-begun,
        'all_source_encoder_images': model.encoder_images, **guard(deadline)})
    model.fusion.requires_grad_(True); model.selector.requires_grad_(False)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def metric_key(task, spec):
    return 'map50' if task == 'UATD' else 'AP50' if spec['kind'] == 'detection' else 'macro_f1' if spec['kind'] == 'classification' else 'mean_error_pixels' if spec['kind'] == 'point_regression' else 'miou'


def fit(task, arm, initial, p, spec):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    schedule = p['target_schedules'][task]; folder = RUN/arm/task; folder.mkdir(parents=True)
    deadline = min(p['deadline_unix'], time.time()+p['fit_caps'][task])
    model = GSR(p['encoder'])
    if arm == 'shared_GSR':
        model.load_state_dict(initial, strict=True)
    reader.configure_encoder(model.encoder, 'e2e'); model.selector.requires_grad_(False)
    model.fusion.requires_grad_(arm == 'shared_GSR'); model.cuda()
    head = new_head(task, spec).cuda()
    body_opt = reader.body_optimizer(model.encoder)
    head_opt = reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    fusion_opt = torch.optim.AdamW(model.fusion.parameters(), lr=.0001, weight_decay=.0001) if arm == 'shared_GSR' else None
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r', allow_pickle=False)).cuda() for s in ('train', 'validation')}
    labels = {s: targets_for(task, spec, s) for s in ('train', 'validation')}
    generator = torch.Generator().manual_seed(0); routing = np.random.RandomState(700)
    key = metric_key(task, spec); direction = -1 if spec['kind'] == 'point_regression' else 1
    policy = 'learned' if arm == 'shared_GSR' else 'global'; best = -float('inf'); updates = 0; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs']+1):
            model.train(); model.selector.eval(); head.train()
            if spec['kind'] == 'detection':
                milestones = (8, 11) if task == 'CFC22' else (40, 55)
                head_opt.param_groups[0]['lr'] = .005*.1**sum(epoch > m for m in milestones)
            order = torch.randperm(len(spec['rows']['train']), generator=generator); total = 0.
            for ids in order.split(schedule['batch']):
                guard(deadline); body_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
                if fusion_opt is not None:
                    fusion_opt.zero_grad(set_to_none=True)
                    fusion_opt.param_groups[0]['lr'] = previous.shared.native.module_lr(updates, schedule['updates'], .0001)
                for group in body_opt.param_groups:
                    group['lr'] = reader.body_learning_rate(updates, schedule['updates'])
                actions = None
                if arm == 'shared_GSR' and routing.rand() < .5:
                    actions = torch.as_tensor(routing.randint(0, 4, len(ids)), device='cuda')
                field = model.field(images['train'][ids.cuda()], head, 'uatd' if task == 'UATD' else spec['kind'], policy, actions)
                target = batch_targets(task, spec, labels['train'], ids.tolist())
                value = task_loss(task, spec, head, field, target); value.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                if fusion_opt is not None:
                    torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True); fusion_opt.step()
                body_opt.step(); head_opt.step(); updates += 1; total += float(value.detach())*len(ids)
                if updates == 1:
                    write(folder/'first_counted_update.json', {'updates': 1, 'input': [len(ids), 1, 224, 224],
                        'field': list(field.shape), 'first_ids': ids.tolist(), 'body_gradient_norm': float(norm),
                        'task_kind': spec['kind'], 'complete_shared_source_modules_retained': arm == 'shared_GSR'})
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order), 'wall_seconds': time.monotonic()-begun}
            every = 2 if task == 'CFC22' else 1 if spec['kind'] in ('segmentation', 'classification', 'point_regression') else 5
            if epoch % every == 0:
                result, independent, pred = evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], deadline, policy)
                entry[key] = result[key]
                payload = {'prediction': pred, 'epoch': epoch, 'task': task, 'arm': arm, 'policy': policy}
                if direction*result[key] > best:
                    best, best_epoch, best_result = direction*result[key], epoch, result
                    torch.save(payload, folder/'best_predictions.pt')
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = result, independent
                    torch.save(payload, folder/'terminal_predictions.pt')
                print(json.dumps({'arm': arm, 'task': task, **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    policies = {policy: terminal}
    if arm == 'shared_GSR':
        for other in ('global', 'random', 'fixed0', 'fixed1', 'fixed2', 'fixed3', 'uncertainty'):
            result, independent, pred = evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], deadline, other)
            policies[other] = result
            torch.save({'prediction': pred, 'task': task, 'epoch': schedule['epochs'], 'policy': other}, folder/(other+'_terminal_predictions.pt'))
            write(folder/(other+'_independent.json'), independent)
    held_result = None
    held_split = 'official_validation' if task == 'HighResolution' else 'test' if task in ('LSOD', 'Diver', 'CFC26_Mini') else None
    if held_split is not None:
        held_rows = spec['rows'][held_split]
        held_images = []
        for row in held_rows:
            with Image.open(row['image']) as image:
                intensity = np.log1p(np.array(image.convert('L'), dtype=np.uint8).astype(np.float32)/255.)
            x = F.interpolate(torch.from_numpy(intensity)[None, None], (224, 224), mode='bilinear', align_corners=False, antialias=True)
            held_images.append(((x[0]-spec['normalization']['mean'])/spec['normalization']['std']).cuda())
        held_labels = targets_for(task, spec, held_split)
        held_result, independent, pred = evaluate(task, spec, model, head, torch.stack(held_images), held_labels, schedule['batch'], deadline, policy, held_split)
        torch.save({'prediction': pred, 'epoch': schedule['epochs'], 'task': task, 'policy': policy, 'split': held_split}, folder/'heldout_terminal_predictions.pt')
        write(folder/'heldout_independent.json', independent)
    write(folder/'results.json', {'status': 'COMPLETE_FIXED_EXPANDED_GSR_TARGET', 'task': task, 'kind': spec['kind'],
        'arm': arm, 'role': spec['role'], 'epochs': schedule['epochs'], 'updates': updates, 'metric': key,
        'higher_is_better': direction == 1, 'terminal': terminal, 'terminal_independent': terminal_independent,
        'best': best_result, 'best_epoch': best_epoch, 'terminal_policy_metrics': policies,
        'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'source_selector_fixed': True, 'heldout_terminal_only': held_result, 'heldout_split': held_split,
        'new_weight_files': 0, 'wall_seconds': time.monotonic()-begun, **guard(deadline)})


@torch.no_grad()
def source_vectors(model, images):
    g = model.global_field(images)
    base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
    actions = model.selector(g).argmax(1)
    large = F.interpolate(images, (448, 448), mode='bilinear', align_corners=False, antialias=True)
    tiles = torch.stack([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in original.QUADS], 1)
    local = model.global_field(tiles[torch.arange(len(images), device='cuda'), actions])
    field = model.corrected(g, local, actions)
    return base.mean((2, 3)).cpu().numpy(), field.mean((2, 3)).cpu().numpy()


@torch.no_grad()
def diagnostics(model, p):
    from sonar_encoder.training.model import prepare_images
    from runs.canoe_frozen_retrieval import expected_top1
    from runs.uxo_frozen_temporal_predictability import fit_affine
    model.eval()
    folder = RUN/'diagnostics'; folder.mkdir()
    begun = time.monotonic(); deadline = min(p['deadline_unix'], time.time()+3600)
    assets = BASE/'canoe_protocol_assets_v1/bounded_input_v1'
    rows = [json.loads(line) for line in (assets/'file_manifest.jsonl').read_text().splitlines()]
    old = json.loads((BASE/'canoe_frozen_retrieval_v1/extraction.json').read_text())['states']['t0']['row_ids']
    vectors = {'global': {}, 'GSR': {}}
    for role in ('gallery', 'query'):
        selected = [r for r in rows if r['role'] == role]
        if [r['row_id'] for r in selected] != old[role]:
            raise ValueError('CANOE original positive-matrix row order changed')
        base_pieces, gsr_pieces = [], []
        for start in range(0, len(selected), 16):
            samples = []
            for row in selected[start:start+16]:
                with Image.open(assets/row['relative_png_path']) as image:
                    raw = np.array(image, dtype=np.uint8)
                intensity = torch.from_numpy(raw[2:].copy()).float()[None]/255
                samples.append({'intensity': intensity, 'valid': torch.ones(intensity.shape[-2:], dtype=torch.bool)})
            images = prepare_images(samples, {'mean': .025567574079009666, 'std': .03710258238121667}, None).cuda()
            a, b = source_vectors(model, images); base_pieces.append(a); gsr_pieces.append(b); guard(deadline)
        for name, pieces in (('global', base_pieces), ('GSR', gsr_pieces)):
            vectors[name][role] = np.concatenate(pieces)
            np.save(folder/(name+'_CANOE_'+role+'.npy'), vectors[name][role])
    positives = np.load(BASE/'canoe_frozen_retrieval_v1/positives.npy', allow_pickle=False)
    retrieval = {}
    for name, value in vectors.items():
        entries = expected_top1(value['query'], value['gallery'], positives)
        retrieval[name] = {'expected_R_at_1': float(np.mean([r['score'] for r in entries])), 'per_query': entries}
    write(folder/'CANOE_results.json', {'status': 'COMPLETE_SOURCE_FROZEN_REVISIT_DIAGNOSTIC', 'states': retrieval,
        'scope': 'Same-site/same-day existing1311/1136 revisits, exact inherited pose-positive matrix and header-exclusion; no GT or source fitting, no cross-site/pixel-correspondence/SOTA claim.',
        'wall_seconds': time.monotonic()-begun, **guard(deadline)})
    norm = {'mean': .025372744258233756, 'std': .036703725756021106}
    records = json.loads((BASE/'uxo_frozen_temporal_predictability_v1/records.json').read_text())
    arrays = {name: {role: np.lib.format.open_memmap(folder/(name+'_UXO_'+role+'.npy'), mode='w+', dtype=np.float32, shape=(n, 384))
                    for role, n in (('fitting', 48163), ('development', 11607))} for name in ('global', 'GSR')}
    for record in records:
        with np.load(record['archive_path'], allow_pickle=False) as archive:
            for start in range(0, record['frames'], 16):
                members = record['members'][start:start+16]; samples = []
                for key, shape, crc, usize, csize in members:
                    intensity = torch.from_numpy(archive[key].astype(np.float32))[None]/255
                    samples.append({'intensity': intensity, 'valid': torch.ones(intensity.shape[-2:], dtype=torch.bool)})
                images = prepare_images(samples, norm, None).cuda()
                a, b = source_vectors(model, images)
                index = record['cache_start']+start
                for name, values in (('global', a), ('GSR', b)):
                    arrays[name][record['role']][index:index+len(members)] = values
                guard(deadline)
        print(json.dumps({'diagnostic': 'UXO', 'record': record['record_id']}), flush=True)
    for by_role in arrays.values():
        for array in by_role.values():
            array.flush()
    teacher = {role: np.load(BASE/'uxo_frozen_temporal_predictability_v1'/file, mmap_mode='r') for role, file in (('fitting', 'fit.npy'), ('development', 'dev.npy'))}
    result = {}
    for name in arrays:
        result[name] = {}
        for lag in (1, 8):
            pairs = []
            for r in records:
                if r['role'] != 'fitting':
                    continue
                start, n = r['cache_start'], r['frames']-8
                pairs.append((np.asarray(arrays[name]['fitting'][start:start+n], np.float64), np.asarray(teacher['fitting'][start+lag:start+lag+n], np.float64)))
            fitted = fit_affine(pairs); per_record = []
            for r in records:
                if r['role'] != 'development':
                    continue
                start, n = r['cache_start'], r['frames']-8
                x = np.asarray(arrays[name]['development'][start:start+n], np.float64)
                y = np.asarray(teacher['development'][start+lag:start+lag+n], np.float64)
                error = np.square(x@fitted['A']+fitted['b']-y).mean(1)
                per_record.append({'record_id': r['record_id'], 'pairs': n, 'MSE': float(error.mean())})
            result[name][str(lag)] = {'macro_record_MSE': float(np.mean([r['MSE'] for r in per_record])), 'per_record': per_record,
                'lambda': fitted['lambda'], 'normal_equation_maxabs': fitted['normal_equation_maxabs']}
    write(folder/'UXO_results.json', {'status': 'COMPLETE_FIXED_TEACHER_TEMPORAL_PROXY', 'states': result,
        'scope': 'Original59770frames/60fit14dev within74train records, same natural lag1/8 common anchors. Fit384D equal-record FP64 ridge from each current input representation to inherited frozenE0 future targets; no optical/object GT or physical/causal/independent-capture claim. No source-label training or neural predictor. No new shuffled experiment or affine weight files.',
        'source_encoder_images_with_diagnostics': model.encoder_images, 'wall_seconds': time.monotonic()-begun, **guard(deadline)})


def worker(arm):
    original.runtime()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    p = json.loads((RUN/'protocol.json').read_text())
    tasks = json.loads(Path(p['data_manifest']).read_text())['tasks']
    initial = None
    if arm == 'shared_GSR':
        model = GSR(p['encoder']).cuda(); initial = source_training(model, p, tasks)
        diagnostics(model, p)
        del model; gc.collect()
    for task in TARGETS:
        if arm == 'original_DINO' and task in ('FSOD', 'Marine'):
            continue
        fit(task, arm, initial, p, tasks[task]); gc.collect()


def pipeline():
    data = json.loads((RUN/'model_inputs.json').read_text())
    if json.loads((RUN/'cpu_prepare_receipt.json').read_text())['status'] != 'COMPLETE_CPU_INPUTS_MODEL_EXECUTION_PROTOCOL_PENDING':
        raise RuntimeError('Expanded CPU input preparation is incomplete')
    old_queue = json.loads((BASE/'gsr_shared_complete_v1/queue_state.json').read_text())
    baseline_wait = next(r for r in old_queue['actual_waits'] if r['arm'] == 'original_DINO')
    if baseline_wait['actual_wait_exit'] != 0:
        raise RuntimeError('Previous original-DINO worker did not release normally')
    if (RUN/'protocol.json').exists():
        raise FileExistsError('Expanded GSR protocol already exists; no duplicate queue')
    plan = json.loads((RUN/'execution_plan.json').read_text())
    parent = json.loads((BASE/'gsr_shared_complete_v1/protocol.json').read_text())
    prior = json.loads((BASE/'downstream_vit_finetuning_v1/writing_handoff.json').read_text())
    started = data['created_unix']
    p = {**plan, 'status': 'FROZEN_EXPANDED_COMPLETE_GSR_BEFORE_MODEL_FORWARD',
        'started_unix': started, 'model_freeze_unix': time.time(), 'deadline_unix': started+36*3600,
        'data_manifest': str(RUN/'model_inputs.json'), 'encoder': parent['encoder'],
        'source_entry': str(Path(__file__).resolve()), 'diagram': str(BASE/'gsr_method_diagrams_v1/18_multidata_complete_gsr.png'),
        'source_original_DINO_frozen': True, 'source_loss': 'Equal mean of six task losses; per-source mean example loss, CFC16/micro4 versus other source macro4. Raw6000 terminal, no EMA/selection.',
        'source_gain_supervision': 'Predetermined max2048 training images per task; five counterfactual losses on same task head, FSOD/CFC/Xing/LSOD RPN sampling RNG matched per image; signed gains retained. Shared task-balanced centered contrast regression, fixed80/20 selector fit/diagnostic partition,100epochs. No non-source label access.',
        'source_export': 'One full DINO+fusion+selector CPU RAM snapshot after source fit; no trained weights on disk. Identical initial state copied for every new GSR target. Current source fit is new multisource training, not old checkpoint recovery.',
        'target_optimization': 'Original DINO fullFT AdamW2e-5/warmup200/cosine to1%, wd.05 matrices and0 CLS/PE/1D; unusedmask excluded. Fusion AdamW1e-4/cosine, source selector fixed. Task heads SGD.005/.9/.0005 for detection, AdamW.001/.0001 otherwise. FSOD/Xing/LSOD/mini LR after40/55, CFC starts9/12. No augmentation, FP32/noAMP/TF32off/source False policy.',
        'target_routing': 'Half training minibatches source-learned/half uniform random; terminal learned primary and scheduled strictbest auxiliary. Same-terminal global/random/fixed0..3/uncertainty diagnostic policies, no winning-policy choice. Source selector descriptor drift under downstream fullFT remains explicit.',
        'target_interfaces': 'One common224 global view plus one selected224 quadrant of its AA448 upsample, shared DINO, full384x32 field. F/CFC/Xing/LSOD RPNROI nativeGT->canvas224/stride7/15360anchors/ROI1/7/15percell; classes1..K plusBG0, original boxes/emptyimages. UATD unchangedConv384to50 directly32, existing box_targets32/dense_loss32/decode32, original normalized nativeGT/AP50. NKSID mean32->Linear384to8. Native semantic logits allclasses, Xing literal3RGBcodes, HighRes2positive polygon classes/unannotated+conflicts ignored255. Diver1channel32 softargmax normalizedXY/SmoothL1, nativepoint error, no fabricated boxes/masks.',
        'validation': 'F/Xing/LSOD/mini/UATD every5epochs; CFC every2; seg/classification/point everyepoch. Terminal primary; earliest strict best auxiliary, no weights retained. Same terminal only on LSOD476test/Diver374test/CFCmini75test/HighRes39publisherval; these do not choose epochs/policies or source training.',
        'source_diagnostics': 'Before downstream adaptation, fixed source snapshot CANOE1311gallery1136query pose-positive expectedR1 (header excluded, UATDnorm), and UXO59770frames/60fit14dev/lag1,8 FP64 equal-record affine into inherited frozenE0 future targets, paired source-GSR/global descriptors. No neural predictor/GT invention/weight files; both share exact same raw inputs. Additional3600s diagnostic cap.',
        'fit_caps': {'FSOD': 5000, 'Marine': 2400, 'UATD': 6500, 'NKSID': 1200,
                     'Xing_detection': 7200, 'Xing_segmentation': 2400, 'LSOD': 7200,
                     'HighResolution': 2400, 'Diver': 1800, 'CFC26_Mini': 1800, 'CFC22': 28800},
        'physical_GPUs': {'shared_GSR': 2, 'original_DINO': 3},
        'source_release_file': str(RUN/'previous_unit_release.json'),
        'dispatch': 'GPU3 starts independent nine new baselines after old baseline actualwait0. GPU2 source starts only after ROOT writes actual previous-GSR foreground exit/worker release. No old run interruption, one owned worker/card. CPU prep time and release wait included in shared36h clock. Per-fit caps are subordinate to shared deadline.',
        'SWiSE_frozen': {'FSOD': .902005317706368, 'Marine': .6059200833586624, 'UATD': .2517123460319482},
        'SWiSE_fullFT': {t: prior['tables'][t]['arms']['e2e']['terminal']['mean'] for t in ('FSOD', 'Marine')},
        'strongest_prior_completed_pool': {**parent['strongest_prior_completed_pool'], 'NKSID': prior['tables']['NKSID']['arms']['e2e']['terminal']['mean']},
        'no_new_smoke_SHA_review_weight_files': True,
        'known_readiness_limits': data['not_ready'], 'normalization_order': 'New datasets: nativeL/255 log1p -> AA224 -> affine using native train-only FP64 stats; inherited F/M/U/N views keep their old exact path. Both arms within each task identical. No resampling of GT or geometry claims.'}
    write(RUN/'protocol.json', p)
    plans = {}
    for i, task in enumerate(SOURCE):
        n = data['tasks'][task]['counts']['train']; macro = p['source_macro_per_task'][task]
        rng = np.random.RandomState(13+i)
        plans[task] = np.concatenate([rng.permutation(n) for _ in range(math.ceil(6000*macro/n))])[:6000*macro].reshape(6000, macro)
        plans[task+'_actions'] = np.random.RandomState(23+i).randint(-1, 4, (6000, macro))
        plans[task+'_gain_subset'] = np.random.RandomState(71+i).permutation(n)[:2048]
        plans[task+'_selector_order'] = np.random.RandomState(91+i).permutation(len(plans[task+'_gain_subset']))
    np.savez(RUN/'source_plan.npz', **plans)
    (RUN/'logs').mkdir(exist_ok=True)
    active, waits, launched_source = {}, [], False

    def launch(arm, gpu):
        free, util = map(int, subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
        pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
        owned, disappeared = [], []
        for pid in pids:
            try:
                if Path('/proc/'+pid).stat().st_uid == os.getuid():
                    owned.append(int(pid))
            except FileNotFoundError:
                disappeared.append(int(pid))
        if free < 32768 or owned:
            raise RuntimeError('Expanded physicalGPU admission '+str((gpu, free, owned)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log = (RUN/'logs'/(arm+'.log')).open('w')
        child = subprocess.Popen([original.PYTHON, '-B', '-u', '-m', 'runs.gsr_multidata_training', arm], cwd=original.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        cpus = sorted(os.sched_getaffinity(0)); offset = 0 if gpu == 2 else 4
        os.sched_setaffinity(child.pid, cpus[offset:offset+4])
        active[arm] = (child, log, gpu)
        write(RUN/('admission_GPU'+str(gpu)+'.json'), {'physical_GPU': gpu, 'free_MiB': free, 'utilization_percent': util,
            'same_UID_GPU_PIDs': owned, 'disappeared_processes': disappeared, 'idle_claim': False})
        write(RUN/(arm+'_launch_receipt.json'), {'supervisor_PID': os.getpid(), 'worker_PID': child.pid,
            'physical_GPU': gpu, 'launched_unix': time.time(), 'new_weight_files': 0})

    try:
        launch('original_DINO', 3)
        while active or not launched_source:
            if time.time() > p['deadline_unix']:
                raise TimeoutError('Expanded GSR shared36h deadline')
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > 48*1024**3:
                raise RuntimeError('Expanded GSR48GiB disk cap')
            if not launched_source and Path(p['source_release_file']).exists():
                release = json.loads(Path(p['source_release_file']).read_text())
                if release['actual_foreground_exit'] != 0:
                    raise RuntimeError('Previous fullGSR release was not successful; independent baseline evidence retained')
                launch('shared_GSR', 2); launched_source = True
            for arm, (child, log, gpu) in list(active.items()):
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del active[arm]
                waits.append({'arm': arm, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code})
                if code != 0:
                    raise RuntimeError(arm+' worker exit '+str(code))
            write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in active.items()},
                'actual_waits': waits, 'source_launched': launched_source, 'source_wait': None if launched_source else 'previous actual foreground release pending'})
            time.sleep(2)
        results = {}
        for task in TARGETS:
            gsr = json.loads((RUN/'shared_GSR'/task/'results.json').read_text())
            baseline = BASE/'gsr_shared_complete_v1/original_DINO'/task/'results.json' if task in ('FSOD', 'Marine') else RUN/'original_DINO'/task/'results.json'
            dino = json.loads(baseline.read_text()); key = metric_key(task, data['tasks'][task])
            baseline_key = 'miou12' if task == 'Marine' else key
            value, reference = gsr['terminal'][key], dino['terminal'][baseline_key]
            results[task] = {'GSR': gsr, 'original_DINO': dino, 'original_evidence': str(baseline),
                'baseline_reused_without_new_fit': task in ('FSOD', 'Marine'), 'terminal_difference': value-reference,
                'higher_is_better': gsr['higher_is_better'], 'SWiSE_frozen': p['SWiSE_frozen'].get(task),
                'SWiSE_fullFT': p['SWiSE_fullFT'].get(task), 'strongest_prior_completed_pool': p['strongest_prior_completed_pool'].get(task),
                'comparison_scope': 'Exact new within-task heads/input/GT/schedules, except F/M compatible inherited same-recipe baseline. Historical SWiSE/strongest complete paths may differ. Single seed, no all-task/significance/SOTA acceptance.'}
        write(RUN/'results.json', {'status': 'COMPLETE_FINITE_EXPANDED_SHARED_GSR', 'tasks': results,
            'diagnostics': {'CANOE': str(RUN/'diagnostics/CANOE_results.json'), 'UXO': str(RUN/'diagnostics/UXO_results.json')},
            'new_target_updates': p['planned_new_target_updates'], 'new_weight_files': 0})
        write(RUN/'final_closure.json', {'status': 'COMPLETE_FIXED_SCOPE', 'actual_waits': waits,
            'elapsed_from_CPU_plan_seconds': time.time()-started, 'new_weight_files': 0, 'resource_scope': 'Process reports/guards, not continuous full-machine peaks.'})
        write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        for arm, (child, log, gpu) in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            code = child.wait(); log.close()
            waits.append({'arm': arm, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code})
        write(RUN/'failure.json', {'status': 'STOPPED_NO_AUTOMATIC_RETRY', 'traceback': traceback.format_exc(),
            'actual_waits': waits, 'elapsed_from_CPU_plan_seconds': time.time()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pipeline', 'shared_GSR', 'original_DINO'))
    args = parser.parse_args()
    pipeline() if args.phase == 'pipeline' else worker(args.phase)
