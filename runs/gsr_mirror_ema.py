"""Frozen original DINO, conditional attention and dense task readouts."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align

from runs import gsr_attention_fixed as attention
from runs.cfc_frozen_detector import FrozenDetectorHead

native = attention.native
halo = attention.halo
core = attention.core
reader = attention.reader
RUN = core.old.BASE/'gsr_mirror_ema_v1'
PARENT = core.old.BASE/'gsr_detail_readout_v1'
native.RUN = RUN
halo.RUN = RUN
attention.RUN = RUN
core.RUN = RUN
core.reuse.RUN = RUN
AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))


class ParameterEMA:
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.shadow = [p.detach().clone() for p in self.parameters]

    @torch.no_grad()
    def update(self):
        for average, parameter in zip(self.shadow, self.parameters):
            average.lerp_(parameter, .001)

    @torch.no_grad()
    def install(self):
        backup = [p.detach().clone() for p in self.parameters]
        for parameter, average in zip(self.parameters, self.shadow):
            parameter.copy_(average)
        return backup

    @torch.no_grad()
    def restore(self, backup):
        for parameter, raw in zip(self.parameters, backup):
            parameter.copy_(raw)


@torch.no_grad()
def extract_mirror(teacher, spec):
    output = {}; mirror = None; started = time.monotonic()
    for split, rows in spec['rows'].items():
        global_tokens = torch.empty((len(rows), 1025, 384), device='cuda')
        local_tokens = torch.empty((len(rows), 4, 1025, 384), device='cuda')
        if split == 'train':
            mirror = torch.empty_like(global_tokens)
        for start in range(0, len(rows), 8):
            global_images = []; local_images = []
            for row in rows[start:start+8]:
                image = (native.read_intensity(row).log1p()-spec['normalization']['mean'])/spec['normalization']['std']
                global_images.append(F.interpolate(image[None], (448, 448), mode='bilinear', align_corners=False, antialias=True)[0])
                large = F.interpolate(image[None], (768, 768), mode='bilinear', align_corners=False, antialias=True)[0]
                local_images.append(torch.stack([large[..., y:y+448, x:x+448] for y, x in ((0, 0), (0, 320), (320, 0), (320, 320))]))
            image = torch.stack(global_images).cuda(); n = len(image)
            global_tokens[start:start+n] = teacher.encoder.backbone.prepare_tokens_with_masks(image)
            if split == 'train':
                mirror[start:start+n] = teacher.encoder.backbone.prepare_tokens_with_masks(image.flip(-1))
            image = torch.stack(local_images).flatten(0, 1).cuda()
            local_tokens[start:start+n] = teacher.encoder.backbone.prepare_tokens_with_masks(image).reshape(n, 4, 1025, 384)
            if start % 256 == 0:
                print(json.dumps({'extract': teacher.task, 'split': split, 'rows': start+n}), flush=True)
                resources()
        output[split] = (global_tokens, local_tokens)
    core.old.write(RUN/teacher.task/'extraction.json', {'published_rasters_decoded': sum(len(v) for v in spec['rows'].values()),
        'global_original_views': sum(len(v) for v in spec['rows'].values()), 'global_mirror_views': len(spec['rows']['train']),
        'local_original_views': 4*sum(len(v) for v in spec['rows'].values()), 'transformer_calls_during_extraction': 0,
        'mirror_operator': 'Horizontalreflection ofnormalizedglobalpixels beforefrozenpatchprojection/officialPE; never permutationofprecomputedtokens.',
        'wall_seconds': time.monotonic()-started, 'resources': resources()})
    return output, mirror


def augmented_targets(task, targets, dense, ids, flips):
    if task == 'Marine':
        masks = dense[ids]
        mask = torch.from_numpy(flips).cuda()[:, None, None]
        return torch.where(mask, masks.flip(-1), masks)
    rows = []
    scale = 224. if task == 'FSOD' else 1.
    for index, flip in zip(ids.tolist(), flips):
        boxes = targets[index]['boxes'].clone()
        if flip:
            boxes[:, [0, 2]] = scale-boxes[:, [2, 0]]
        rows.append({'boxes': boxes, 'labels': targets[index]['labels']})
    return [{k: v.cuda() for k, v in row.items()} for row in rows] if task == 'FSOD' else box_targets32(rows, 'cuda')


def slice_targets(task, targets, start, count):
    if task == 'FSOD':
        return targets[start:start+count]
    if task == 'UATD':
        return tuple(v[start:start+count] for v in targets)
    return targets[start:start+count]


class FractionalAnchors(AnchorGenerator):
    def __init__(self):
        super().__init__(((9, 18, 36, 72, 143),), ((.5, 1., 2.),))
        self.first_geometry = {}

    def forward(self, image_list, feature_maps):
        sizes = [f.shape[-2:] for f in feature_maps]; image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0]/size[0], dtype=dtype, device=device),
                    torch.tensor(image_size[1]/size[1], dtype=dtype, device=device)] for size in sizes]
        self.set_cell_anchors(dtype, device)
        anchors = torch.cat(self.grid_anchors(sizes, strides))
        if not self.first_geometry:
            self.first_geometry = {'image_list_hw': list(image_size), 'field_hw': list(sizes[0]),
                'stride_y_x': [float(v) for v in strides[0]], 'anchor_count': len(anchors), 'origin': 0}
        return [anchors for _ in image_list.image_sizes]


class DenseRoIPool(nn.Module):
    def __init__(self):
        super().__init__(); self.first_geometry = {}

    def forward(self, features, boxes, image_shapes):
        if not self.first_geometry:
            self.first_geometry = {'field': list(features['0'].shape), 'spatial_scale': 2/7,
                                   'sampling_ratio': 2, 'aligned': True, 'output_size': [7, 7]}
        return roi_align(features['0'], boxes, (7, 7), spatial_scale=2/7, sampling_ratio=2, aligned=True)


class DenseDetector(FrozenDetectorHead):
    def __init__(self):
        super().__init__()
        self.roi_heads.box_predictor = FastRCNNPredictor(1024, 11)
        self.rpn.anchor_generator = FractionalAnchors()
        self.roi_heads.box_roi_pool = DenseRoIPool()
        self.delta = nn.Conv2d(384, 1536, 1)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def head_field(self, features):
        return F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)+F.pixel_shuffle(self.delta(features), 2)

    def forward(self, features, targets=None):
        return super().forward(self.head_field(features), targets)


class DenseSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = nn.Conv2d(384, 12, 1)
        self.delta = nn.Conv2d(384, 192, 1)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        self.register_buffer('loss_weights', torch.ones(12))

    def forward(self, features):
        coarse = F.interpolate(self.coarse(features), scale_factor=4, mode='bilinear', align_corners=False)
        return coarse+F.pixel_shuffle(self.delta(features), 4)


def new_head(task):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        if task == 'FSOD':
            return DenseDetector()
        if task == 'Marine':
            return DenseSegmentation()
        return reader.new_head('UATD', 0)


class DetailGlobal(attention.AttentionGlobal):
    def __init__(self, task):
        super().__init__(task)
        self.head = new_head(task)

    def field(self, tokens, bypasses, local=False, context=None, actions=None):
        self.attention_context.update(layers=bypasses, context=context, actions=actions)
        values = []
        for i, block in enumerate(self.encoder.backbone.blocks):
            tokens = block(tokens)
            if i in (2, 5, 8, 11):
                values.append(self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, 32, 32))
        return self.neck(torch.stack(values, 1))


class DetailLocal(attention.AttentionLocal):
    def __init__(self, teacher):
        super().__init__(teacher); self.mode = 'mirror_ema'
        if self.task == 'FSOD':
            self.local_head.rpn.anchor_generator.first_geometry = {}
            self.local_head.roi_heads.box_roi_pool.first_geometry = {}

    def context16(self, g):
        return F.avg_pool2d(g, 2)


def box_targets32(targets, device):
    heat = torch.zeros(len(targets), 10, 32, 32, device=device)
    regression = torch.zeros(len(targets), 10, 4, 32, 32, device=device)
    for i, row in enumerate(targets):
        boxes = row['boxes'].to(device); labels = row['labels'].to(device)
        center = (boxes[:, :2]+boxes[:, 2:])*16; cells = center.floor().long()
        size = (boxes[:, 2:]-boxes[:, :2])*32
        heat[i, labels, cells[:, 1], cells[:, 0]] = 1
        regression[i, labels, :, cells[:, 1], cells[:, 0]] = torch.cat((center-cells, size.log()), 1)
    return heat, regression, heat.bool()


def dense_loss32(output, targets, denominator):
    heat, regression, positive = targets
    logits = output[:, :10]; probability = logits.sigmoid()
    ptarget = torch.where(positive, probability, 1-probability)
    focal = (torch.where(positive, .25, .75)*(1-ptarget).square()*F.binary_cross_entropy_with_logits(logits, heat, reduction='none')).sum()
    raw = output[:, 10:].reshape(-1, 10, 4, 32, 32)
    prediction = torch.cat((raw[:, :, :2].sigmoid(), raw[:, :, 2:]), 2)
    regression = (F.smooth_l1_loss(prediction, regression, reduction='none')*positive[:, :, None]).sum()/4
    return (focal+regression)/denominator


def task_loss(task, head, field, target):
    if task == 'FSOD':
        return reader.task_loss(task, head, field, target)
    if task == 'UATD':
        return dense_loss32(head(field), target, target[2].sum())
    logits = F.interpolate(head(field), target.shape[-2:], mode='bilinear', align_corners=False)
    truth = target.long(); probability = logits.softmax(1); onehot = F.one_hot(truth, 12).permute(0, 3, 1, 2)
    support = onehot.sum((0, 2, 3))
    dice = 1-(2*(probability*onehot).sum((0, 2, 3))+1)/(probability.sum((0, 2, 3))+support+1)
    return F.cross_entropy(logits, truth, weight=head.loss_weights)+.5*dice[support > 0].mean()


def crop_targets(task, targets):
    if task != 'UATD':
        return halo.crop_targets(task, targets)
    output = []
    for j in range(4):
        origin = torch.tensor([j%2, j//2], dtype=torch.float32)*halo.ROI_STEP; rows = []
        for row in targets:
            boxes = row['boxes'].clone()
            boxes[:, :2] = torch.maximum(boxes[:, :2], origin)
            boxes[:, 2:] = torch.minimum(boxes[:, 2:], origin+halo.ROI_SIZE)
            keep = (boxes[:, 2:] > boxes[:, :2]).all(1)
            rows.append({'boxes': (boxes[keep]-origin.repeat(2))/halo.ROI_SIZE, 'labels': row['labels'][keep]})
        output.append(box_targets32(rows, 'cuda'))
    return output


def decode32(output):
    from sonar_encoder.training.uatd_head import box_iou
    output = output.detach().float().cpu(); probability = output[:, :10].sigmoid().numpy()
    regression = output[:, 10:].reshape(-1, 10, 4, 32, 32)
    offsets = regression[:, :, :2].sigmoid().numpy(); sizes = (regression[:, :, 2:].exp()/32).numpy(); predictions = []
    for i in range(len(output)):
        boxes_out = []; scores_out = []; labels_out = []
        for label in range(10):
            scores = probability[i, label].reshape(-1); cells = np.flatnonzero(scores >= .05)
            cells = cells[np.argsort(-scores[cells], kind='stable')[:100]]; rows, cols = cells//32, cells%32
            center = (np.stack((cols, rows), 1)+offsets[i, label, :, rows, cols])/32
            wh = sizes[i, label, :, rows, cols]
            boxes = np.concatenate((center-wh/2, center+wh/2), 1).clip(0, 1).astype(np.float32)
            order = np.flatnonzero((boxes[:, 2:] > boxes[:, :2]).all(1))
            while len(order):
                current = order[0]; boxes_out.append(boxes[current]); scores_out.append(scores[cells[current]]); labels_out.append(label)
                remaining = order[1:]; order = remaining[box_iou(boxes[current], boxes[remaining]) <= .5]
        scores = np.array(scores_out, dtype=np.float32); keep = np.argsort(-scores, kind='stable')[:100]
        predictions.append({'boxes': torch.from_numpy(np.array(boxes_out, dtype=np.float32).reshape(-1, 4)[keep]),
                            'scores': torch.from_numpy(scores[keep]), 'labels': torch.tensor(labels_out, dtype=torch.long)[keep]})
    return predictions


def decode(task, head, field, rows):
    return decode32(head(field)) if task == 'UATD' else reader.decode(task, head, field, rows)


@torch.no_grad()
def predict_teacher(teacher, tokens, rows, batch):
    teacher.eval(); pieces = []
    for start in range(0, len(rows), batch):
        g = teacher.field(tokens[start:start+batch], teacher.bypasses)
        pieces.append(decode(teacher.task, teacher.head, g, rows[start:start+batch]))
    return [row for piece in pieces for row in piece] if teacher.task == 'UATD' else np.concatenate(pieces)


@torch.no_grad()
def global_bank(teacher, raw):
    output = {}
    for split, (tokens, fine_tokens) in raw.items():
        g = torch.empty((len(tokens), 384, 32, 32), device='cuda')
        for start in range(0, len(tokens), 8):
            g[start:start+8] = teacher.field(tokens[start:start+8], teacher.bypasses)
        output[split] = (g, fine_tokens)
    return output


def local_box_loss(model, fine, ids, actions, cropped, denominator):
    indices = ids.tolist(); regions = actions.tolist()
    if model.task == 'FSOD':
        target = [{k: v.cuda() for k, v in cropped[j][i].items()} for i, j in zip(indices, regions)]
        return task_loss('FSOD', model.local_head, fine, target)
    target = tuple(torch.stack([cropped[j][k][i] for i, j in zip(indices, regions)]) for k in range(3))
    return dense_loss32(model.local_head(fine), target, denominator)


@torch.no_grad()
def predict_global(teacher, g, spec):
    pieces = []
    for start in range(0, len(g), spec['micro']):
        sl = slice(start, start+spec['micro'])
        pieces.append(decode(teacher.task, teacher.head, g[sl], spec['rows']['validation'][sl]))
    return [row for piece in pieces for row in piece] if teacher.task == 'UATD' else np.concatenate(pieces)


@torch.no_grad()
def detections(task, head, field):
    return decode32(head(field)) if task == 'UATD' else halo.detections(task, head, field)


@torch.no_grad()
def detection_banks(model, spec, bank):
    output = {}
    for split, (g, fine) in bank.items():
        global_rows = []; regions = [[] for _ in range(4)]
        for start in range(0, len(g), spec['micro']):
            sl = slice(start, start+spec['micro'])
            global_rows.extend(detections(model.task, model.head, g[sl]))
            for j in range(4):
                regions[j].extend(halo.map_region(row, j) for row in detections(model.task, model.local_head, fine[sl, j]))
        output[split] = (global_rows, regions)
    return output


@torch.no_grad()
def predict(model, spec, data, actions, detector_predictions=None, fixed_merge=False):
    if model.task != 'UATD':
        return halo.predict(model, spec, data, actions, detector_predictions, fixed_merge)
    if bool((np.asarray(actions) < 0).all()):
        return predict_global(model.teacher, data[0], spec)
    global_rows, regions = detector_predictions
    return [global_rows[i] if j < 0 else halo.merge_detections(model, global_rows[i], regions[j][i], int(j), not fixed_merge)
            for i, j in enumerate(actions)]


@torch.no_grad()
def entropy_actions(model, g):
    pieces = []
    for start in range(0, len(g), 16):
        field = g[start:start+16]
        if model.task == 'Marine':
            logits = model.head(field); entropy = -(logits.softmax(1)*logits.log_softmax(1)).sum(1)
        else:
            logits = model.head(field)[:, :10] if model.task == 'UATD' else model.head.rpn.head([model.head.head_field(field)])[0][0]
            entropy = -(logits.sigmoid()*F.logsigmoid(logits)+(1-logits.sigmoid())*F.logsigmoid(-logits)).mean(1)
        scores = []
        for j in range(4):
            actions = torch.full((len(field),), j, device='cuda', dtype=torch.long)
            roi = F.grid_sample(entropy[:, None], halo.crop_grid(actions, 32, 32), mode='bilinear', padding_mode='border', align_corners=False)
            scores.append(roi.mean((1, 2, 3)))
        pieces.append(torch.stack(scores, 1).argmax(1).cpu().numpy())
    return np.concatenate(pieces)


@torch.no_grad()
def evaluate(model, spec, bank, targets, predictions, selector, sr, updates):
    g = bank['validation'][0]; n = len(g)
    scores = torch.cat([selector(model.context16(g[i:i+64])) for i in range(0, n, 64)]).cpu().numpy()*sr['gain_scale']+sr['gain_mean']
    top = scores.argmax(1); keep = scores.max(1) > 0; entropy = entropy_actions(model, g)
    policies = {'global_only': np.full(n, -1, np.int64), 'learned': np.where(keep, top, -1), 'learned_forced_top1': top,
        'random_budgetmatched': np.where(keep, np.random.default_rng(2).integers(0, 4, n), -1),
        'fixed_budgetmatched': np.where(keep, sr['fixed_region_fit_only'], -1), 'entropy_budgetmatched': np.where(keep, entropy, -1),
        'entropy_forced_top1': entropy, 'merge_control_budgetmatched': np.where(keep, top, -1)}
    metrics = {}; folder = RUN/model.task/model.mode
    for policy, actions in policies.items():
        prediction = predict(model, spec, bank['validation'], actions,
                             predictions['validation'] if model.task != 'Marine' else None,
                             fixed_merge=policy == 'merge_control_budgetmatched')
        metrics[policy] = core.reuse.score_prediction(model.task, prediction, targets['validation'], spec, folder/(policy+'_prediction'))
        print(json.dumps({'score': model.task+'/'+policy, 'value': metrics[policy][spec['metric']]}), flush=True)
    np.savez(folder/'actions.npz', **policies)
    report = {'task': model.task, 'mode': model.mode, 'metrics': metrics, 'local_updates': updates,
        'selection_fraction': float(keep.mean()), 'original_DINO_updates': 0, 'global_parameter_updates': 0,
        'terminal_only': True, 'new_weights_saved': False, 'resources': resources()}
    core.old.write(folder/'results.json', report)
    return report


@torch.no_grad()
def online_cost(model, spec, selector, sr):
    samples = [native.read_intensity(row) for row in spec['rows']['validation'][:32]]; cells = []
    for policy in ('global_only', 'learned'):
        load = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
        baseline = torch.cuda.memory_reserved(); allocated_baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(); records = []
        for visit in range(146):
            i = visit % 32 if visit < 50 else (visit-50) % 32
            torch.cuda.synchronize(); begun = time.perf_counter()
            image = (samples[i].log1p()-spec['normalization']['mean'])/spec['normalization']['std']
            g = model.teacher.global_field(F.interpolate(image[None], (448, 448), mode='bilinear', align_corners=False, antialias=True).cuda()); action = -1
            if model.task != 'Marine':
                global_prediction = detections(model.task, model.head, g)[0]; prediction = global_prediction
            if policy == 'learned':
                scores = selector(model.context16(g))*sr['gain_scale']+sr['gain_mean']
                if float(scores.max()) > 0:
                    action = int(scores.argmax(1)); actions = torch.tensor([action], device='cuda')
                    large = F.interpolate(image[None], (768, 768), mode='bilinear', align_corners=False, antialias=True)
                    y, x = action//2*320, action%2*320
                    tokens = model.teacher.encoder.backbone.prepare_tokens_with_masks(large[..., y:y+448, x:x+448].cuda())
                    fine = model.fine_field(tokens, g, actions)
                    if model.task == 'Marine':
                        model.marine_log_probability(g, fine, actions).argmax(1).byte().cpu().numpy()
                    else:
                        local = halo.map_region(detections(model.task, model.local_head, fine)[0], action)
                        prediction = halo.merge_detections(model, global_prediction, local, action)
            if model.task == 'Marine' and action < 0:
                decode(model.task, model.head, g, spec['rows']['validation'][i:i+1])
            elif model.task != 'Marine':
                halo.serialize_detections(model.task, [prediction], spec['rows']['validation'][i:i+1])
            torch.cuda.synchronize(); elapsed = time.perf_counter()-begun
            if visit >= 50:
                records.append({'row_id': spec['row_ids']['validation'][i], 'round': (visit-50)//32, 'action': action, 'seconds': elapsed})
        values = [r['seconds'] for r in records]
        cells.append({'policy': policy, 'median_seconds': float(np.median(values)), 'p95_seconds': float(np.percentile(values, 95)),
            'raw': records, 'GPU_load': load, 'baseline_reserved': baseline, 'baseline_allocated': allocated_baseline,
            'peak_reserved': torch.cuda.max_memory_reserved(), 'peak_allocated': torch.cuda.max_memory_allocated(),
            'incremental_peak_allocated': torch.cuda.max_memory_allocated()-allocated_baseline})
    core.old.write(RUN/model.task/model.mode/'cost.json', {'cells': cells,
        'boundary': 'Decodednative normalization/resize/H2D/actualglobalencoder/selector/optionallocalencoder/densereadout/head/calibration/NMS/nativepostproc. Decode/setup excluded.',
        'memory_scope': 'Trainingbanks resident, not isolateddeployment memory; loads recorded, oneownedworker/card.'})


def workflow(task):
    native.ACTIVE_TASK = task
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    spec = core.reuse.protocol()['tasks'][task]
    targets = {split: reader.make_targets(task, rows) for split, rows in spec['rows'].items()}
    dense = {split: box_targets32(value, 'cuda') for split, value in targets.items()} if task == 'UATD' else (
        {split: value.cuda() for split, value in targets.items()} if task == 'Marine' else {})
    teacher = DetailGlobal(task).cuda()
    if task == 'Marine':
        counts = torch.bincount(dense['train'].long().flatten(), minlength=12)
        weights = counts.float().rsqrt(); weights /= (weights*counts).sum()/counts.sum()
        teacher.head.loss_weights.copy_(weights)
        core.old.write(RUN/task/'training_class_weights.json', {'exact_train_pixel_counts': counts.tolist(), 'FP32_loss_weights': weights.tolist(),
            'scope': 'TrainingGT only; weightedloss, not weightedmetric or inputnormrefit.'})
    raw, mirror = extract_mirror(teacher, spec)
    global_report = train_global(teacher, spec, raw, mirror, targets, dense)
    del mirror
    if task == 'FSOD':
        core.old.write(RUN/task/'global/decoder_geometry.json', {'anchors': teacher.head.rpn.anchor_generator.first_geometry,
                       'roi': teacher.head.roi_heads.box_roi_pool.first_geometry, 'normal_forward_only': True})
    data = global_bank(teacher, raw); del raw
    prediction = predict_global(teacher, data['validation'][0], spec)
    global_report['metrics'] = core.reuse.score_prediction(task, prediction, targets['validation'], spec, RUN/task/'global/terminal_prediction')
    core.old.write(RUN/task/'global/results.json', global_report)
    cropped = crop_targets(task, dense['train'] if task == 'Marine' else targets['train'])
    model = DetailLocal(teacher).cuda(); updates = train_local(model, spec, data, targets, dense, cropped)
    if task == 'FSOD':
        core.old.write(RUN/task/model.mode/'decoder_geometry.json', {'anchors': model.local_head.rpn.anchor_generator.first_geometry,
                       'roi': model.local_head.roi_heads.box_roi_pool.first_geometry, 'normal_forward_only': True})
    bank = halo.packed_bank(model, data)
    predictions = {} if task == 'Marine' else detection_banks(model, spec, bank)
    if task != 'Marine':
        halo.train_calibrator(model, predictions['train'], halo.normalized_targets(task, targets['train']))
    gains = halo.gain_labels(model, spec, bank, dense, targets, predictions)
    selector, sr = core.reuse.train_selector(task, model.mode, model.context16(bank['train'][0]), gains)
    report = evaluate(model, spec, bank, targets, predictions, selector, sr, updates)
    online_cost(model, spec, selector, sr)
    previous_result = json.loads((PARENT/task/'results.json').read_text())
    previous = {'learned': previous_result['local_stage']['metrics']['learned'][spec['metric']],
                'global_only': previous_result['global_stage']['metrics'][spec['metric']], 'scope': 'Completedsingleheadseed0; not3seedmean.'}
    reference = max(core.reuse.protocol()['references'][task]['baseline_terminal_three_seed_mean'], previous['learned'], previous['global_only'], global_report['metrics'][spec['metric']])
    core.old.write(RUN/task/'results.json', {'status': 'COMPLETE_MIRROR_EMA', 'task': task,
        'global_stage': global_report, 'local_stage': report, 'new_weights_saved': False,
        'prior_completed_terminal_comparisons': previous, 'strongest_completed_comparison_value': reference,
        'learned_minus_strongest': report['metrics']['learned'][spec['metric']]-reference, 'resources': resources()})

module_lr = native.module_lr
resources = native.resources
quick_metric = native.quick_metric

def train_global(teacher, spec, raw, mirror, targets, dense):
    folder = RUN/teacher.task/'global'; folder.mkdir()
    head_opt = reader.head_optimizer(teacher.task, teacher.head)
    extra = [p for name, p in teacher.named_parameters() if p.requires_grad and not name.startswith('head.')]
    opt = torch.optim.AdamW(extra, lr=.001, weight_decay=.0001)
    ema = ParameterEMA([p for p in teacher.parameters() if p.requires_grad])
    flips = np.load(RUN/teacher.task/'global_flip_plan.npy')
    generator = torch.Generator().manual_seed(0); updates = 0; best = -1.; best_epoch = 0
    total_updates = spec['epochs']*math.ceil(len(spec['rows']['train'])/spec['batch']); begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, spec['epochs']+1):
            teacher.train(); teacher.encoder.eval(); order = torch.randperm(len(spec['rows']['train']), generator=generator)
            if teacher.task == 'FSOD':
                for group in head_opt.param_groups:
                    group['lr'] = .005 if epoch <= 40 else (.0005 if epoch <= 55 else .00005)
            total_loss = 0.
            for macro in order.split(spec['batch']):
                head_opt.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:
                    group['lr'] = module_lr(updates, total_updates, .001)
                flip = flips[epoch-1, macro.numpy()]
                augmented = augmented_targets(teacher.task, targets['train'], dense['train'] if teacher.task == 'Marine' else None, macro, flip)
                denominator = augmented[2].sum() if teacher.task == 'UATD' else len(macro)
                value = 0.
                for start in range(0, len(macro), spec['micro']):
                    ids = macro[start:start+spec['micro']]; mask = torch.from_numpy(flip[start:start+spec['micro']]).cuda()
                    tokens = torch.where(mask[:, None, None], mirror[ids], raw['train'][0][ids])
                    g = teacher.field(tokens, teacher.bypasses)
                    target = slice_targets(teacher.task, augmented, start, len(ids))
                    weight = target[2].sum()/denominator if teacher.task == 'UATD' else len(ids)/denominator
                    loss = task_loss(teacher.task, teacher.head, g, target)*weight
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite real global training loss')
                    loss.backward(); value += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(extra, 1., error_if_nonfinite=True)
                head_opt.step(); opt.step(); ema.update(); updates += 1; total_loss += value*len(macro)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {'original_body_trainable': 0,
                        'original_body_has_grad': any(p.grad is not None for p in teacher.encoder.parameters()),
                        'all12_new_bypass_grad_norms': [float(sum(p.grad.square().sum() for p in b.parameters() if p.grad is not None).sqrt()) for b in teacher.bypasses],
                        'global_grid': list(g.shape), 'macro': len(macro), 'micro': spec['micro'], 'loss': value})
                if updates % 100 == 0:
                    resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': total_loss/len(order), 'wall_seconds': time.monotonic()-begun}
            if epoch % 5 == 0:
                backup = ema.install()
                pred = predict_teacher(teacher, raw['validation'][0], spec['rows']['validation'], spec['micro'])
                score = quick_metric(teacher.task, spec, pred, dense['validation'] if teacher.task == 'Marine' else targets['validation'])
                entry['global_metric'] = score
                entry['parameter_state'] = 'EMA_decay.999_allupdates'
                if epoch != spec['epochs']:
                    ema.restore(backup)
                if score > best:
                    best, best_epoch = score, epoch
                print(json.dumps({'task': teacher.task, 'phase': 'global', **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    teacher.eval().requires_grad_(False)
    for p in teacher.parameters():
        p.grad = None
    report = {'task': teacher.task, 'mode': 'global', 'epochs': spec['epochs'], 'updates': updates,
        'scheduled_terminal_metric': score, 'terminal_parameter_state': 'EMA_decay.999_allupdates', 'scheduled_best': best, 'best_epoch': best_epoch,
        'wall_seconds': time.monotonic()-begun, 'original_DINO_trainable': 0,
        'new_weights_saved': False, 'resources': resources()}
    core.old.write(folder/'train_report.json', report)
    return report



def train_local(model, spec, data, targets, dense, cropped):
    folder = RUN/model.task/model.mode; folder.mkdir()
    head_opt = reader.head_optimizer(model.task, model.local_head)
    for group in head_opt.param_groups:
        group['lr'] = .0005 if model.task == 'FSOD' else .0001
    optimizer = torch.optim.AdamW([{'params': model.bypasses.parameters(), 'lr': .0001},
                                  {'params': model.bridge.parameters(), 'lr': .001}], weight_decay=.0001)
    ema = ParameterEMA([p for p in model.parameters() if p.requires_grad])
    actions = torch.from_numpy(np.load(RUN/model.task/'local_actions.npy')).cuda()
    counts = torch.stack([value[2].flatten(1).sum(1) for value in cropped], 1) if model.task == 'UATD' else None
    if model.task == 'Marine':
        weights = model.local_head.loss_weights
        core.old.write(folder/'local_class_weights.json', {'inverse_sqrt_weights': weights.tolist(), 'source': str(RUN/model.task/'training_class_weights.json'),
            'scope': 'Inheritedtraining-only12classweights; no inputnormrefit or evaluationweighting.'})
    generator = torch.Generator().manual_seed(0); updates = 0; begun = time.monotonic()
    total = 30*math.ceil(len(spec['rows']['train'])/spec['batch'])
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, 31):
            model.train(); model.teacher.encoder.eval()
            order = torch.randperm(len(spec['rows']['train']), generator=generator); loss_sum = 0.
            for macro in order.split(spec['batch']):
                a = actions[epoch-1, macro]
                denominator = counts[macro, a].sum().clamp_min(1) if model.task == 'UATD' else len(macro)
                optimizer.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
                optimizer.param_groups[0]['lr'] = native.module_lr(updates, total, .0001)
                value = 0.
                for ids in macro.split(spec['micro']):
                    a = actions[epoch-1, ids]; g = data['train'][0][ids]
                    fine = model.fine_field(data['train'][1][ids, a], g, a)
                    if model.task == 'Marine':
                        labels = torch.stack([cropped[int(j)][int(i)] for i, j in zip(ids, a)]).long()
                        logits = F.interpolate(model.local_head(fine), (280, 187), mode='bilinear', align_corners=False)
                        probability = logits.softmax(1); onehot = F.one_hot(labels, 12).permute(0, 3, 1, 2)
                        support = onehot.sum((0, 2, 3))
                        dice = 1-(2*(probability*onehot).sum((0, 2, 3))+1)/(probability.sum((0, 2, 3))+support+1)
                        loss = F.cross_entropy(logits, labels, weight=weights)+.5*dice[support > 0].mean()
                    else:
                        loss = local_box_loss(model, fine, ids, a, cropped, denominator)
                    if model.task != 'UATD':
                        loss = loss*len(ids)/len(macro)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite attention-local training loss')
                    loss.backward(); value += float(loss.detach())
                parameters = [p for group in optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); head_opt.step(); ema.update(); updates += 1; loss_sum += value*len(macro)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {'original_DINO_trainable': sum(p.numel() for p in model.teacher.encoder.parameters() if p.requires_grad),
                        'global_trainable': sum(p.numel() for p in model.teacher.parameters() if p.requires_grad),
                        'global_has_grad': any(p.grad is not None for p in model.teacher.parameters()),
                        'fine_shape': list(fine.shape), 'new_adapter_grad_norms': [float(sum(p.grad.square().sum() for p in layer.parameters() if p.grad is not None).sqrt()) for layer in model.bypasses],
                        'local_adapter_parameters': sum(p.numel() for p in model.bypasses.parameters()),
                        'local_head_parameters': sum(p.numel() for p in model.local_head.parameters()), 'pixel_gate_parameters': 0,
                        'loss': value, 'macro': len(macro), 'micro': spec['micro']})
                if updates % 100 == 0:
                    native.resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': loss_sum/len(order), 'wall_seconds': time.monotonic()-begun}
            log.write(json.dumps(entry)+'\n'); log.flush()
            if epoch % 5 == 0:
                print(json.dumps({'task': model.task, 'phase': 'attention_local', **entry}), flush=True)
    ema.install()
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    core.old.write(folder/'train_report.json', {'epochs': 30, 'updates': updates, 'wall_seconds': time.monotonic()-begun,
        'original_DINO_updates': 0, 'global_parameter_updates': 0, 'new_weights_saved': False, 'resources': native.resources()})
    return updates


def pipeline():
    parent = json.loads((PARENT/'protocol.json').read_text())
    prior = {task: {'result_path': str(PARENT/task/'results.json'),
                    'scope': 'Samepredeclaredparentcondition/task/metric, completed singleheadseed0 value whenavailable. Not a3seedmean; no output-dependent training/stopping.'}
             for task in parent['tasks']}
    RUN.mkdir(); (RUN/'logs').mkdir(); begun = time.time()
    protocol = {'start_unix': begun, 'deadline_unix': begun+28800, 'budget_scope': 'Separatefixednewscientificmethod8h coordinator/workerwall; previousunitcosts preserved, not a continuationreset. PrelaunchLLMdrafting/serviceCPU isnotmeasured inthisclock.',
        'encoder': parent['encoder'], 'tasks': parent['tasks'], 'references': parent['references'],
        'prior_completed_terminal_comparisons': prior,
        'SWiSE_references': parent['SWiSE_references'],
        'source_utf8': Path(__file__).read_text(), 'reused_native_source_utf8': Path(native.__file__).read_text(), 'reused_halo_source_utf8': Path(halo.__file__).read_text(), 'reused_attention_source_utf8': Path(attention.__file__).read_text(),
        'parent_fixed_run': str(PARENT),
        'principle_diagram': str(core.old.BASE/'gsr_method_diagrams_v1/08_mirror_ema.svg'),
        'authorization': 'User authorizes rapid frozen-original-weight structural optimization on all3tasks, onlyphysicalGPU2/3, maxoneownedworker/card. No newsmoke/SHA/review/trainedweightfiles.',
        'training_regularization': 'Globalhorizontalmirror p.5/private23 perimageepoch, cachedtokens computedfrommirroredpixels/officialPE, neverfliptokenorder alone. DerivedGT coordinates/masks reflected, originalGTfiles unchanged. Localstage remainsoriginalview. EMA.999 ofproductionglobal/localtrainablemodules/head aftereveryupdate, originalDINO excluded; scheduledglobaleval usesEMA andterminalEMA fixed, localterminalEMA fixed. Calibration10/selector100 retaintheirrawstandardoptimizers, notEMA. No best-epoch switching.',
        'global_stage': 'Denseheadrevision: FSODlearnedresidual32to64featureupsampling, sameRPN/ROIclassifier/anchorsizes butfloatstride3.5/origin0/61440anchors/ROI2over7; Marinecoarse1x1pluszero-upPixelShuffle4classresidual to128logits; UATDbodyglobal32/no16pool/heatmap32. 12externalrank8 alpha8 unmergedQKVLoRAdeltas (scale1), globalFiLMexcluded, officialDINOqkvweightsfixed andofficialblockforwardretainedviaoutputhooks. Fourlevelneck, directpublished448, originalhead60FSOD/100Marine/UATD. OriginalDINO fixed. Same headoptimizers/order/schedules; modulesAdamW.001/wd.0001/clip1/200warmupcosine.',
        'local_stage': 'Separatecopiedlocalattentiondeltas/head, fixedglobalmodel,30epochs randomonecornerROI/normalmacro. Directpublished768 four448crops atorigins0/320, overlap128pixels. NormalizedROIspan7/12, origins0/5/12. Local32field retainsdensity via localhead; UATDlocal32, no16pool. Targets/decoder32 andallbodyfields32 match. Localencodingdensity is12/7 ofglobal448; not the2xnonoverlappingcase.',
        'local_context': 'Localrank8featuresFiLMconditionedbyglobalmean384+actualROIcenter2;12attentionmodules. 64dim4head crossattention,1024localquerytokens/256globaltokens plus fullframe positions, zero-up residual. Global-conditioned localattentiondeltas; legacyattribute namebypasses nowholdsQKVdeltas, notConvpass.',
        'local_optimizer': 'BYPASSESAdamWpeak.0001/200warmupcosine; bridgeAdamW.001; noMarinepixelgate; copiedlocalheadSGD.0005FSOD orAdamW.0001others. Allwd as originalnative localstage.',
        'Marine_loss': 'Globalnative480x320 andlocal280x187cropCE weightedbyalltrainnativeclasspixelcounts inverse-sqrt/weightedmean normalization, plus.5softDice overpresentcropGTclasses (smooth1). SameweightedCE+.5presentclassDiceinbothphases; no mixedglobalNLL/pixelgate optimization. Maininference .5*taper probabilityfusion; alternate1*taper. Originalnative480x320 all12evaluationGT unchanged.',
        'detector_loss': 'FSODoriginalcroppedGT4losses unchanged, UATD32targetencoding: center*32, logwh*32, focalalpha.25/.75 gamma2 andSmoothL1beta1/4P unchanged; full-imagePpositive, emptycropauxmax(P,1). OriginalcroppedGT losses only, fullframe evaluationGT unchanged. FSOD sumoriginal4RPN/ROI losses, UATD focal+regression with macropositiveauxdenominator max(P,1), emptycropsretainnegativefocal. No nondifferentiableNMS loss claim.',
        'detector_merge': 'Globalboxes/scores fixed; maplocal224 ornormalizedROI to fullframe using origin(0or5/12) andspan7/12. Calibrate localscores, concatenate, classwiseIoU.5NMS/top100/score>=.05. FP32coordinate roundtrip is declared, not a bitwiseactive-path identity claim.',
        'calibrator': '10epochsAdamW.001/wd.0001/B512/private0.19inputs: scorelogit(eps1e-5),10classonehot,boxcenter/size,maxsameclassglobalIoU/globalconfidence,cornerIDxy.32GELUhidden,zero-output init. Train-only localbranch one-to-one scoreorderedIoU.5/fullframeGT binarylabels. Globalconfidenceunchanged before NMS.',
        'gain_labels': 'Signedtrain-only mergedprediction utility: Marineexistingclass/boundaryweightedNLL reduction; detection classweights inverse-sqrt alltrainGTcounts, weighted2TP/(GT+predictedcount), matchingIoU.5/score.05. Correctemptyimage utility1, falsepositiveempty0. Proxy isnot AP orphysicalGT.',
        'entropy_definition': 'Meanentropy fromactualglobalheadmaps sampledat32x32insideeachfullhaloROI, not oldquartercores. Marineclasssoftmax128; UATDclassBernoulli32; FSDRPNobjectnessBernoulli64. B16chunks; no GT.',
        'selector': 'Existingselector100epochs/private7, parentinner80/20seed37 unchanged. Global/localheads fittedalltrain; innercheck is selector-only, not independentof taskfitting. Retainnegativegains, skipifmaxpredictedgain<=0.',
        'policies': 'global_only,learned,forcedtop1,random/fixed/entropybudgetmatched,entropyforced,mergecontrolsamelearnedactions. Marinecontrol1*taper versusmain.5*taper; detectorcontroluncalibratedlocalscores. No extraheadfit or posthocprimaryswitch.',
        'endpoints': 'Terminal fullmethod only after globalfullschedule/local30/calibration10whereapplicable/selector100. Global scheduledbest isauxiliary, not fullmethodbest. OriginalnativeGT/canonicalandindependentNumPy scoring fromsavedpredictions.',
        'cost': '50warmup+first32fixeddevimages3rounds/B1, actualglobalencoder+optionalonefineencoder+localhead/merge/selector. Decodedraster through nativepostproc; decode excluded, trainingbanksresident, external/cardloadrecorded.',
        'prior_work': 'LoRA borrowedmatrixfactorization https://arxiv.org/abs/2106.09685 (abstractverified; language-modelpaper, not sonarvalidation). Conditionalposition/globalFiLM andhalo taskreadout are thiscandidate, no noveltycertification.',
        'storage': 'No newweights/featurefiles. GPU-RAMbanks only; destructivelyreuse localpatchembeddingstorage for finalfine32features afteralllocaltraining. Requiredpred/gainlabels/logs/meta only.',
        'caps': {'global_seconds': 28800, 'workflow_seconds': 14400, 'RSS_bytes': 48*1024**3, 'new_disk_bytes': 4*1024**3},
        'GPU_caps_bytes': parent['GPU_caps_bytes'], 'required_free_MiB': parent['required_free_MiB'],
        'CPU_affinity': {'2': AVAILABLE_CPUS[:4], '3': AVAILABLE_CPUS[4:8]},
        'schedule': 'CPUstagedqueue now. FSODGPU2 onlyafterparentFSODactualworkerwait0 andnormaladmission; no furtherparentGPU2phases. NewUATDGPU3 onlyafteractualparentforegroundwait0. NewMarineGPU2 afternewFSOD. Acrossbothunits maxoneownedworker/card,2total.',
        'release_receipts': 'ROOT atomicallypublishes release_FSOD_GPU2.json from actualparentFSODwait0; avoids concurrentlyreading mutableparentqueueJSON. Parentactualforegroundwait.json is likewisepublished atomically byROOT after retainedsession89992 actuallyreturns0. Neither receipt grants protocol/math changes.',
        'stop': 'Realerror/nonfinite/hardresources/deadlines stopentireunit. No score-basedstop/automaticretry/batchprecisioncapchanges.',
        'total_global_updates': 43000, 'total_local_updates': 16440,
        'acceptance': 'All3learnedterminals exceedqualifiedSWiSE and arewithin.01 of max(strongestcompatiblecompletedlegacymean,completedparentlearned/globalterminals,ownnewglobalbaseline). Prior singleheadseed0 observations remain separately labeled; no significance/worldSOTA/noveltyclaim.',
        'train_geometry_source': parent['train_geometry_source'],
        'comparison_limit': 'Versusdetailreadout this changesglobalmirrortraining andEMA together, not an isolatedmoduleeffect. SameoriginalGT/splits/metrics/headgeometry asdetailparent. Mirroraugmentation andEMA changedtogether; no isolatedcause, so differences frompriorrun do not isolate a cause. Completepaths/pretraining/budgets differ fromstrongspecialists.'}
    for task in protocol['tasks']:
        folder = RUN/task; folder.mkdir()
        n = len(protocol['tasks'][task]['rows']['train'])
        np.save(folder/'global_flip_plan.npy', np.random.default_rng(23).random((protocol['tasks'][task]['epochs'], n)) < .5)
        for name in ('local_actions.npy', 'selector_split.npz'):
            (folder/name).write_bytes((PARENT/task/name).read_bytes())
    core.old.write(RUN/'protocol.json', protocol)
    active = {}; waits = []; first_started = False; uatd_started = False
    def launch(task, gpu):
        admission = core.old.admission(gpu)
        if admission['free_MiB'] < protocol['required_free_MiB'][task]:
            raise RuntimeError(task+' fine-bank memory admission insufficient '+str(admission))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'XFORMERS_DISABLED': '1'}
        cpus = ','.join(map(str, protocol['CPU_affinity'][str(gpu)]))
        log = (RUN/'logs'/f'{task}.log').open('w')
        child = subprocess.Popen(['/usr/bin/taskset', '-c', cpus, core.old.PYTHON, '-B', '-u', '-m',
            'runs.gsr_mirror_ema', task], cwd=core.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[task] = (child, log, gpu, time.time(), admission)
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'observed_unix': time.time(),
            'workers': {name: value[0].pid for name, value in active.items()}, 'new_weight_files': 0,
            'resource_staging': protocol['schedule']})
        core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {name: value[0].pid for name, value in active.items()}, 'actual_waits': waits})
    try:
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'observed_unix': time.time(),
            'workers': {}, 'status': 'CPU_STAGED_NO_NEW_MODEL', 'resource_staging': protocol['schedule']})
        core.old.write(RUN/'queue_state.json', {'status': 'CPU_STAGED_WAIT_PARENT_FSOD_ACTUAL_EXIT0', 'active': {}, 'actual_waits': []})
        while active or not first_started or not uatd_started:
            if (PARENT/'failure_closure.json').exists():
                raise RuntimeError('Parentfixedunit stopped; no automaticparentrepair or successorcontinuation')
            release = RUN/'release_FSOD_GPU2.json'
            if not first_started and release.exists():
                finished = json.loads(release.read_text())
                if finished['task'] != 'FSOD' or finished['actual_wait_exit'] != 0:
                    raise RuntimeError('ParentFSOD actualexitnonzero; stagedGPU2 launchcancelled')
                launch('FSOD', 2); first_started = True
            if not uatd_started and (PARENT/'actual_foreground_wait.json').exists():
                wait = json.loads((PARENT/'actual_foreground_wait.json').read_text())
                if wait['actual_foreground_exit'] != 0:
                    raise RuntimeError('Parentactualforegroundnonzero; stagedGPU3 launchcancelled')
                core.old.write(RUN/'parent_actual_foreground_wait.json', wait)
                launch('UATD', 3); uatd_started = True
            if time.time() > protocol['deadline_unix']:
                raise TimeoutError('Fixed8h dense-prediction deadline')
            for task, (child, log, gpu, started, admission) in list(active.items()):
                if time.time()-started > protocol['caps']['workflow_seconds']:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del active[task]
                waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code,
                              'wall_seconds': time.time()-started, 'admission': admission})
                if code != 0:
                    raise RuntimeError(task+' actual worker exit '+str(code))
                if task == 'FSOD':
                    launch('Marine', 2)
                core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in active.items()}, 'actual_waits': waits})
            time.sleep(1)
        tasks = {task: json.loads((RUN/task/'results.json').read_text()) for task in protocol['tasks']}
        core.old.write(RUN/'results.json', {'status': 'COMPLETE_FIXED_MIRROR_EMA', 'tasks': tasks})
        size = sum(path.stat().st_size for path in RUN.rglob('*') if path.is_file())
        if size > protocol['caps']['new_disk_bytes']:
            raise RuntimeError('Results-only4GiB cap exceeded')
        closure = {'status': 'COMPLETE_SCOPE_GOAL_REQUIRES_RESULT_DECISION', 'actual_waits': waits,
                   'wall_seconds': time.time()-begun, 'new_entity_bytes': size, 'new_weights_saved': False}
        core.old.write(RUN/'wave_closure.json', closure); core.old.write(RUN/'writing_handoff.json', closure)
        core.old.write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        failure = traceback.format_exc()
        for task, (child, log, gpu, started, admission) in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            code = child.wait(); log.close()
            waits.append({'task': task, 'PID': child.pid, 'actual_wait_exit': code, 'cancelled_after_failure': True})
        core.old.write(RUN/'failure_closure.json', {'status': 'STOPPED_NO_AUTOMATIC_RETRY', 'traceback': failure, 'actual_waits': waits})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['pipeline', 'Marine', 'FSOD', 'UATD'])
    args = parser.parse_args(); core.old.runtime()
    if args.phase == 'pipeline':
        pipeline()
    else:
        workflow(args.phase)
