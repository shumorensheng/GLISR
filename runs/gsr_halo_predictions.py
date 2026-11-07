"""Fixed original DINO with global-guided, fine regional predictions."""
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
from torchvision.ops import batched_nms, box_iou

from runs import gsr_native_evidence_indexfix as native
from sonar_encoder.training.uatd_head import box_targets, decode_detections

core = native.core
reader = native.reader
RUN = core.old.BASE/'gsr_halo_predictions_v1'
COORDINATOR_PARENT = core.old.BASE/'gsr_dense_predictions_v1'
ROI_SIZE = 7/12
ROI_STEP = 5/12
PARENT = core.old.BASE/'gsr_native_evidence_indexfix_v1'
native.RUN = RUN
core.RUN = RUN
core.reuse.RUN = RUN
AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))


def crop_grid(actions, height, width):
    y, x = torch.meshgrid((torch.arange(height, device=actions.device)+.5)/height,
                          (torch.arange(width, device=actions.device)+.5)/width, indexing='ij')
    local = torch.stack((x, y), -1)[None]
    origin = torch.stack((actions%2, actions//2), 1).float()*ROI_STEP
    return 2*(local*ROI_SIZE+origin[:, None, None])-1


@torch.no_grad()
def extract(teacher, spec):
    output = {}; begun = time.monotonic(); count = 0
    for split, rows in spec['rows'].items():
        global_tokens = torch.empty((len(rows), 1025, 384), device='cuda')
        local_tokens = torch.empty((len(rows), 4, 1025, 384), device='cuda')
        for start in range(0, len(rows), 8):
            global_images = []; local_images = []
            for row in rows[start:start+8]:
                image = (native.read_intensity(row).log1p()-spec['normalization']['mean'])/spec['normalization']['std']
                global_images.append(F.interpolate(image[None], (448, 448), mode='bilinear', align_corners=False, antialias=True)[0])
                large = F.interpolate(image[None], (768, 768), mode='bilinear', align_corners=False, antialias=True)[0]
                local_images.append(torch.stack([large[..., y:y+448, x:x+448] for y, x in ((0, 0), (0, 320), (320, 0), (320, 320))]))
            image = torch.stack(global_images).cuda(); n = len(image)
            global_tokens[start:start+n] = teacher.encoder.backbone.prepare_tokens_with_masks(image)
            image = torch.stack(local_images).flatten(0, 1).cuda()
            local_tokens[start:start+n] = teacher.encoder.backbone.prepare_tokens_with_masks(image).reshape(n, 4, 1025, 384)
            count += n
            if start % 256 == 0:
                print(json.dumps({'extract': teacher.task, 'split': split, 'rows': start+n}), flush=True)
                native.resources()
        output[split] = (global_tokens, local_tokens)
    core.old.write(RUN/teacher.task/'extraction.json', {'published_rasters_decoded': count, 'frozen_patch_global_views': count,
        'frozen_patch_local_views': 4*count, 'full_transformer_calls_during_extract': 0,
        'local_view_size': 768, 'local_crop_size': 448, 'local_stride': 320, 'GPU_RAM_only': True,
        'wall_seconds': time.monotonic()-begun, 'resources': native.resources()})
    return output


def crop_targets(task, targets):
    cropped = []
    for j in range(4):
        if task == 'Marine':
            actions = torch.full((len(targets),), j, device=targets.device, dtype=torch.long)
            sampled = F.grid_sample(targets[:, None].float(), crop_grid(actions, 280, 187), mode='nearest',
                                    padding_mode='border', align_corners=False)
            cropped.append(sampled[:, 0].byte())
        else:
            scale = 224. if task == 'FSOD' else 1.
            origin = torch.tensor([j%2, j//2], dtype=torch.float32)*ROI_STEP*scale
            items = []
            for row in targets:
                boxes = row['boxes'].clone()
                boxes[:, :2] = torch.maximum(boxes[:, :2], origin)
                boxes[:, 2:] = torch.minimum(boxes[:, 2:], origin+ROI_SIZE*scale)
                keep = (boxes[:, 2:] > boxes[:, :2]).all(1)
                items.append({'boxes': (boxes[keep]-origin.repeat(2))/ROI_SIZE, 'labels': row['labels'][keep]})
            cropped.append(items if task == 'FSOD' else box_targets(items, 'cuda'))
    return cropped


class FineContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(384, 64, 1)
        self.global_projection = nn.Conv2d(384, 64, 1)
        self.position = nn.Linear(2, 64)
        self.norm = nn.LayerNorm(64)
        self.output = nn.Conv2d(64, 384, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        y, x = torch.meshgrid((torch.arange(32)+.5)/32, (torch.arange(32)+.5)/32, indexing='ij')
        local = torch.stack([torch.stack((x*7/6+j%2*5/6-1, y*7/6+j//2*5/6-1), -1).reshape(1024, 2) for j in range(4)])
        y, x = torch.meshgrid((torch.arange(16)+.5)/8-1, (torch.arange(16)+.5)/8-1, indexing='ij')
        self.register_buffer('local_xy', local)
        self.register_buffer('global_xy', torch.stack((x, y), -1).reshape(256, 2))

    def forward(self, fine, g, actions):
        g = g if g.shape[-1] == 16 else F.avg_pool2d(g, 2)
        q = self.local(fine).flatten(2).transpose(1, 2)+self.position(self.local_xy[actions])
        v = self.global_projection(g).flatten(2).transpose(1, 2)
        k = v+self.position(self.global_xy)[None]
        heads = lambda value: value.reshape(len(fine), -1, 4, 16).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0.)
        mixed = F.gelu(self.norm(q+mixed.transpose(1, 2).reshape(len(fine), 1024, 64)))
        return fine+self.output(mixed.transpose(1, 2).reshape(len(fine), 64, 32, 32))


class DenseLocal(nn.Module):
    context16 = core.reuse.Model.context16

    def __init__(self, teacher):
        super().__init__(); self.task = teacher.task; self.mode = 'halo_local'
        self.teacher = teacher; self.head = teacher.head
        self.local_head = copy.deepcopy(teacher.head).requires_grad_(True)
        self.bypasses = copy.deepcopy(teacher.bypasses).requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(102)
            self.bridge = FineContext()
            if self.task == 'Marine':
                self.pixel_gate = nn.Sequential(nn.Conv2d(768, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))
                nn.init.zeros_(self.pixel_gate[-1].weight); nn.init.constant_(self.pixel_gate[-1].bias, -2.)
            else:
                self.calibrator = nn.Sequential(nn.Linear(19, 32), nn.GELU(), nn.Linear(32, 1))
                nn.init.zeros_(self.calibrator[-1].weight); nn.init.zeros_(self.calibrator[-1].bias)
        y = torch.sin(math.pi*(torch.arange(32)+.5)/32)
        x = torch.sin(math.pi*(torch.arange(32)+.5)/32)
        self.register_buffer('taper', (y[:, None]*x[None])[None, None])
        y, x = torch.meshgrid((torch.arange(480)+.5)/480, (torch.arange(320)+.5)/320, indexing='ij')
        self.register_buffer('native_xy', torch.stack((x, y), -1)[None])

    def fine_field(self, tokens, g, actions):
        fine = self.teacher.field(tokens, self.bypasses, True, g, actions)
        return self.bridge(fine, g, actions)

    def marine_log_probability(self, g, fine, actions, fixed_merge=False):
        global_log = F.log_softmax(F.interpolate(self.head(g), (480, 320), mode='bilinear', align_corners=False), 1)
        base = F.grid_sample(g, crop_grid(actions, 32, 32), mode='bilinear', padding_mode='border', align_corners=False)
        gate = self.pixel_gate(torch.cat((fine, base), 1))
        origin = torch.stack((actions%2, actions//2), 1).float()*ROI_STEP
        grid = 2*(self.native_xy-origin[:, None, None])/ROI_SIZE-1
        valid = ((grid >= -1) & (grid < 1)).all(-1)[:, None]
        gate = F.grid_sample(gate, grid, mode='bilinear', padding_mode='border', align_corners=False)
        window = F.grid_sample(self.taper.expand(len(g), -1, -1, -1), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        log_alpha = window.log()+(math.log(.5) if fixed_merge else F.logsigmoid(gate))
        log_alpha = torch.where(valid, log_alpha, -torch.inf)
        local_logits = F.grid_sample(self.local_head(fine), grid, mode='bilinear', padding_mode='border', align_corners=False)
        local_log = F.log_softmax(local_logits, 1)
        return torch.logaddexp(global_log+torch.log1p(-log_alpha.exp()), local_log+log_alpha)


def train_local(model, spec, data, targets, dense, cropped):
    folder = RUN/model.task/model.mode; folder.mkdir()
    head_opt = reader.head_optimizer(model.task, model.local_head)
    for group in head_opt.param_groups:
        group['lr'] = .0005 if model.task == 'FSOD' else .0001
    extra = list(model.bridge.parameters())+(list(model.pixel_gate.parameters()) if model.task == 'Marine' else [])
    optimizer = torch.optim.AdamW([{'params': model.bypasses.parameters(), 'lr': .0001},
                                  {'params': extra, 'lr': .001}], weight_decay=.0001)
    actions = torch.from_numpy(np.load(RUN/model.task/'local_actions.npy')).cuda()
    counts = torch.stack([value[2].flatten(1).sum(1) for value in cropped], 1) if model.task == 'UATD' else None
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
                    pooled = F.avg_pool2d(fine, 2)
                    if model.task == 'Marine':
                        labels = torch.stack([cropped[int(j)][int(i)] for i, j in zip(ids, a)])
                        logits = F.interpolate(model.local_head(fine), (280, 187), mode='bilinear', align_corners=False)
                        loss = F.cross_entropy(logits, labels.long())
                    else:
                        loss = native.local_aux(model, fine, pooled, ids, a, cropped, denominator)
                    if model.task != 'UATD':
                        loss = loss*len(ids)/len(macro)
                    if model.task == 'Marine':
                        truth = dense['train'][ids].long(); mixed = model.marine_log_probability(g, fine, a)
                        probability = mixed.exp(); label = F.one_hot(truth, 12).permute(0, 3, 1, 2)
                        axes = (0, 2, 3); support = label.sum(axes)
                        dice = 1-(2*(probability*label).sum(axes)+1)/(probability.sum(axes)+support+1)
                        loss = .25*loss+F.nll_loss(mixed, truth)+.25*dice[support > 0].mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite dense-regional training loss')
                    loss.backward(); value += float(loss.detach())
                parameters = [p for group in optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); head_opt.step(); updates += 1; loss_sum += value*len(macro)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {'original_DINO_trainable': 0,
                        'global_trainable': sum(p.numel() for p in model.teacher.parameters() if p.requires_grad),
                        'global_has_grad': any(p.grad is not None for p in model.teacher.parameters()),
                        'fine_shape': list(fine.shape), 'task_local_head_grid': 16 if model.task == 'UATD' else 32,
                        'parameter_counts': {'fixed_original_DINO': sum(p.numel() for p in model.teacher.encoder.parameters()),
                            'fixed_global_path': sum(p.numel() for p in model.teacher.parameters()),
                            'local_bypasses': sum(p.numel() for p in model.bypasses.parameters()),
                            'local_head': sum(p.numel() for p in model.local_head.parameters()),
                            'bridge': sum(p.numel() for p in model.bridge.parameters()),
                            'gate_or_calibrator': sum(p.numel() for p in (model.pixel_gate if model.task == 'Marine' else model.calibrator).parameters())},
                        'new_module_grad_norm': float(sum(p.grad.square().sum() for p in parameters if p.grad is not None).sqrt()),
                        'loss': value, 'macro': len(macro), 'micro': spec['micro']})
                if updates % 100 == 0:
                    native.resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': loss_sum/len(order), 'wall_seconds': time.monotonic()-begun}
            log.write(json.dumps(entry)+'\n'); log.flush()
            if epoch % 5 == 0:
                print(json.dumps({'task': model.task, 'phase': 'dense_local', **entry}), flush=True)
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    report = {'epochs': 30, 'updates': updates, 'wall_seconds': time.monotonic()-begun,
              'original_DINO_updates': 0, 'global_parameter_updates': 0, 'new_weights_saved': False, 'resources': native.resources()}
    core.old.write(folder/'train_report.json', report)
    return updates


@torch.no_grad()
def packed_bank(model, data):
    output = {}
    for split, (g, tokens) in data.items():
        for start in range(0, len(g), 8):
            end = min(start+8, len(g))
            for j in range(4):
                actions = torch.full((end-start,), j, dtype=torch.long, device='cuda')
                fine = model.fine_field(tokens[start:end, j], g[start:end], actions)
                # Destructively reuse token storage after all local training; no second full-size bank.
                tokens[start:end, j, 1:] = fine.flatten(2).transpose(1, 2)
            if start % 256 == 0:
                native.resources()
        output[split] = (g, tokens[:, :, 1:].transpose(2, 3).view(len(g), 4, 384, 32, 32))
    return output


@torch.no_grad()
def detections(task, head, field):
    if task == 'UATD':
        return decode_detections(head(field))
    return [{'boxes': row['boxes'].detach().cpu()/224, 'scores': row['scores'].detach().cpu(),
             'labels': row['labels'].detach().cpu()-1} for row in head(field)]


def map_region(prediction, region):
    origin = torch.tensor([region%2, region//2]*2, dtype=torch.float32)*ROI_STEP
    return {**prediction, 'boxes': prediction['boxes']*ROI_SIZE+origin}


def calibration_input(global_prediction, local_prediction, region):
    boxes = local_prediction['boxes']; labels = local_prediction['labels']; scores = local_prediction['scores']
    overlap = box_iou(boxes, global_prediction['boxes'])
    overlap = overlap*(labels[:, None] == global_prediction['labels'][None])
    # An empty global detection set is part of the intended empty-image population.
    if overlap.shape[1]:
        best, index = overlap.max(1); score = global_prediction['scores'][index]*(best > 0)
    else:
        best = torch.zeros(len(boxes)); score = torch.zeros(len(boxes))
    center = (boxes[:, :2]+boxes[:, 2:])/2; size = boxes[:, 2:]-boxes[:, :2]
    xy = torch.tensor([region%2, region//2], dtype=torch.float32).expand(len(boxes), -1)
    return torch.cat((scores.clamp(1e-5, 1-1e-5).logit()[:, None], F.one_hot(labels, 10).float(),
                      center, size, best[:, None], score[:, None], xy), 1)


def matched(prediction, target):
    overlap = box_iou(prediction['boxes'], target['boxes'])
    overlap *= prediction['labels'][:, None] == target['labels'][None]
    good = torch.zeros(len(prediction['boxes']), dtype=torch.bool)
    for i in prediction['scores'].argsort(descending=True, stable=True):
        if not overlap.shape[1]:
            break
        best, j = overlap[i].max(0)
        if best >= .5:
            good[i] = True; overlap[:, j] = 0
    return good


@torch.no_grad()
def merge_detections(model, global_prediction, local_prediction, region, calibrated=True):
    features = calibration_input(global_prediction, local_prediction, region)
    local_scores = local_prediction['scores']
    if calibrated:
        delta = model.calibrator(features.cuda()).flatten().cpu()
        local_scores = (features[:, 0]+delta).sigmoid()
    boxes = torch.cat((global_prediction['boxes'], local_prediction['boxes']))
    scores = torch.cat((global_prediction['scores'], local_scores))
    labels = torch.cat((global_prediction['labels'], local_prediction['labels']))
    valid = scores >= .05
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    keep = batched_nms(boxes, scores, labels, .5)[:100]
    return {'boxes': boxes[keep], 'scores': scores[keep], 'labels': labels[keep]}


@torch.no_grad()
def detection_banks(model, spec, bank):
    output = {}
    for split, (g, fine) in bank.items():
        global_rows = []; regions = [[] for _ in range(4)]
        for start in range(0, len(g), spec['micro']):
            sl = slice(start, start+spec['micro'])
            global_rows.extend(detections(model.task, model.head, g[sl]))
            for j in range(4):
                f = F.avg_pool2d(fine[sl, j], 2) if model.task == 'UATD' else fine[sl, j]
                regions[j].extend(map_region(row, j) for row in detections(model.task, model.local_head, f))
        output[split] = (global_rows, regions)
    return output


def normalized_targets(task, targets):
    return [{'boxes': row['boxes']/224 if task == 'FSOD' else row['boxes'],
             'labels': row['labels']-1 if task == 'FSOD' else row['labels']} for row in targets]


def train_calibrator(model, predictions, targets):
    global_rows, regions = predictions; features = []; labels = []
    for j in range(4):
        for global_row, local_row, truth in zip(global_rows, regions[j], targets):
            features.append(calibration_input(global_row, local_row, j)); labels.append(matched(local_row, truth))
    x = torch.cat(features).cuda(); y = torch.cat(labels).float().cuda()
    model.calibrator.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(model.calibrator.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(0); updates = 0
    for epoch in range(10):
        for ids in torch.randperm(len(x), generator=generator).split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(x[ids, 0]+model.calibrator(x[ids]).flatten(), y[ids])
            loss.backward(); optimizer.step(); updates += 1
        native.resources()
    model.calibrator.eval().requires_grad_(False)
    for p in model.calibrator.parameters():
        p.grad = None
    core.old.write(RUN/model.task/model.mode/'calibration.json', {'epochs': 10, 'updates': updates,
        'proposal_examples': len(x), 'positive_examples': int(y.sum()),
        'labels': 'Train-only local-branch classwise score-ordered one-to-one matching to original fullframeGT atIoU.5; cropfragments/duplicates can be negatives.',
        'global_scores_changed': False, 'no_development_GT': True, 'new_weights_saved': False})


def detection_utility(prediction, target, weights):
    if not len(prediction['boxes']) and not len(target['boxes']):
        return 1.
    good = matched(prediction, target)
    tp = weights[prediction['labels'][good]].sum()
    return float(2*tp/(weights[target['labels']].sum()+weights[prediction['labels']].sum()))


@torch.no_grad()
def gain_labels(model, spec, bank, dense, targets, predictions):
    g, fine = bank['train']; gains = np.empty((len(g), 4), np.float32)
    if model.task == 'Marine':
        fit = np.load(RUN/model.task/'selector_split.npz')['fit']
        count = torch.bincount(dense['train'][fit].long().flatten(), minlength=12).float()
        weights = count.clamp_min(1).rsqrt(); weights /= (weights*count).sum()/count.sum()
        for start in range(0, len(g), 16):
            end = min(start+16, len(g)); truth = dense['train'][start:end].long()
            w = core.reuse.marine.boundary_weights(truth, weights)
            baseline = F.log_softmax(F.interpolate(model.head(g[start:end]), (480, 320), mode='bilinear', align_corners=False), 1)
            loss0 = (F.nll_loss(baseline, truth, reduction='none')*w).sum((1, 2))/w.sum((1, 2))
            for j in range(4):
                actions = torch.full((end-start,), j, device='cuda', dtype=torch.long)
                logp = model.marine_log_probability(g[start:end], fine[start:end, j], actions)
                loss = (F.nll_loss(logp, truth, reduction='none')*w).sum((1, 2))/w.sum((1, 2))
                gains[start:end, j] = (loss0-loss).cpu().numpy()
            native.resources()
    else:
        truth = normalized_targets(model.task, targets['train'])
        count = torch.bincount(torch.cat([row['labels'] for row in truth]), minlength=10).float()
        weights = count.rsqrt(); weights /= weights.mean()
        global_rows, regions = predictions['train']
        for i, (global_row, target) in enumerate(zip(global_rows, truth)):
            baseline = detection_utility(global_row, target, weights)
            for j in range(4):
                merged = merge_detections(model, global_row, regions[j][i], j)
                gains[i, j] = detection_utility(merged, target, weights)-baseline
            if (i+1) % 200 == 0:
                print(json.dumps({'task': model.task, 'gain_labels': i+1}), flush=True)
                native.resources()
    return gains


def serialize_detections(task, predictions, rows):
    if task == 'UATD':
        return predictions
    from runs.shared_supervised_readout import prediction_rows
    values = [{'boxes': row['boxes']*224, 'scores': row['scores'], 'labels': row['labels']+1} for row in predictions]
    return prediction_rows(values, rows)


@torch.no_grad()
def predict(model, spec, data, actions, detector_predictions=None, fixed_merge=False):
    g, fine = data; pieces = []
    if bool((np.asarray(actions) < 0).all()):
        return core.reuse.predict(model.teacher, spec, (g, None), actions)
    if model.task != 'Marine':
        global_rows, regions = detector_predictions
        for i, j in enumerate(actions):
            pieces.append(global_rows[i] if j < 0 else merge_detections(model, global_rows[i], regions[j][i], int(j), not fixed_merge))
        return serialize_detections(model.task, pieces, spec['rows']['validation'])
    for start in range(0, len(g), spec['micro']):
        end = min(start+spec['micro'], len(g)); a = torch.tensor(actions[start:end], device='cuda')
        global_log = F.log_softmax(F.interpolate(model.head(g[start:end]), (480, 320), mode='bilinear', align_corners=False), 1)
        selected = a >= 0
        if bool(selected.any()):
            local = fine[start:end][torch.arange(len(a), device='cuda'), a.clamp_min(0)]
            mixed = model.marine_log_probability(g[start:end], local, a.clamp_min(0), fixed_merge)
            global_log = torch.where(selected[:, None, None, None], mixed, global_log)
        pieces.append(global_log.argmax(1).byte().cpu().numpy())
    return np.concatenate(pieces)


@torch.no_grad()
def evaluate(model, spec, bank, targets, predictions, selector, sr, updates):
    g = bank['validation'][0]; n = len(g)
    scores = torch.cat([selector(model.context16(g[i:i+64])) for i in range(0, n, 64)]).cpu().numpy()*sr['gain_scale']+sr['gain_mean']
    top = scores.argmax(1); keep = scores.max(1) > 0
    entropy = core.reuse.entropy_actions(model, g)
    policies = {'global_only': np.full(n, -1, np.int64), 'learned': np.where(keep, top, -1),
        'learned_forced_top1': top, 'random_budgetmatched': np.where(keep, np.random.default_rng(2).integers(0, 4, n), -1),
        'fixed_budgetmatched': np.where(keep, sr['fixed_region_fit_only'], -1),
        'entropy_budgetmatched': np.where(keep, entropy, -1), 'entropy_forced_top1': entropy,
        'fusion_fixed_budgetmatched': np.where(keep, top, -1)}
    metrics = {}; folder = RUN/model.task/model.mode
    for policy, actions in policies.items():
        pred = predict(model, spec, bank['validation'], actions,
                       predictions['validation'] if model.task != 'Marine' else None,
                       fixed_merge=policy == 'fusion_fixed_budgetmatched')
        metrics[policy] = core.reuse.score_prediction(model.task, pred, targets['validation'], spec, folder/(policy+'_prediction'))
        print(json.dumps({'score': model.task+'/'+policy, 'value': metrics[policy][spec['metric']]}), flush=True)
    np.savez(folder/'actions.npz', **policies)
    report = {'task': model.task, 'mode': model.mode, 'metrics': metrics, 'local_updates': updates,
        'selection_fraction': float(keep.mean()), 'original_DINO_updates': 0, 'global_parameter_updates': 0,
        'terminal_only': True, 'new_weights_saved': False, 'resources': native.resources()}
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
            normalized = (samples[i].log1p()-spec['normalization']['mean'])/spec['normalization']['std']
            image = F.interpolate(normalized[None], (448, 448), mode='bilinear', align_corners=False, antialias=True).cuda()
            g = model.teacher.global_field(image); action = -1
            if model.task != 'Marine':
                global_prediction = detections(model.task, model.head, g)[0]; prediction = global_prediction
            if policy == 'learned':
                scores = selector(model.context16(g))*sr['gain_scale']+sr['gain_mean']
                if float(scores.max()) > 0:
                    action = int(scores.argmax(1)); a = torch.tensor([action], device='cuda')
                    large = F.interpolate(normalized[None], (768, 768), mode='bilinear', align_corners=False, antialias=True)
                    y, x = action//2*320, action%2*320
                    tokens = model.teacher.encoder.backbone.prepare_tokens_with_masks(large[..., y:y+448, x:x+448].cuda())
                    fine = model.fine_field(tokens, g, a)
                    if model.task == 'Marine':
                        model.marine_log_probability(g, fine, a).argmax(1).byte().cpu().numpy()
                    else:
                        field = F.avg_pool2d(fine, 2) if model.task == 'UATD' else fine
                        local_prediction = map_region(detections(model.task, model.local_head, field)[0], action)
                        prediction = merge_detections(model, global_prediction, local_prediction, action)
            if model.task == 'Marine' and action < 0:
                reader.decode(model.task, model.head, g, spec['rows']['validation'][i:i+1])
            elif model.task != 'Marine':
                serialize_detections(model.task, [prediction], spec['rows']['validation'][i:i+1])
            torch.cuda.synchronize(); elapsed = time.perf_counter()-begun
            if visit >= 50:
                records.append({'row_id': spec['row_ids']['validation'][i], 'round': (visit-50)//32, 'action': action, 'seconds': elapsed})
        values = [row['seconds'] for row in records]
        cells.append({'policy': policy, 'median_seconds': float(np.median(values)), 'p95_seconds': float(np.percentile(values, 95)),
            'raw': records, 'GPU_load': load, 'baseline_reserved': baseline, 'peak_reserved': torch.cuda.max_memory_reserved(),
            'baseline_allocated': allocated_baseline, 'peak_allocated': torch.cuda.max_memory_allocated(),
            'incremental_peak_allocated': torch.cuda.max_memory_allocated()-allocated_baseline})
    core.old.write(RUN/model.task/model.mode/'cost.json', {'cells': cells,
        'boundary': 'Decoded published raster, original normalization, resize/H2D, actualglobalencoder, selector, optionalactualfineencoder/bridge/localhead/calibration/merge/nativepostprocessing. Decode/setup excluded.',
        'memory_scope': 'Trainingbanks resident; not isolateddeployment memory. One ownedworker percard; othercard training may be active.'})


def workflow(task):
    native.ACTIVE_TASK = task
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    spec = core.reuse.protocol()['tasks'][task]
    targets = {split: reader.make_targets(task, rows) for split, rows in spec['rows'].items()}
    dense = {split: box_targets(value, 'cuda') for split, value in targets.items()} if task == 'UATD' else (
        {split: value.cuda() for split, value in targets.items()} if task == 'Marine' else {})
    teacher = native.GlobalModel(task).cuda(); raw = extract(teacher, spec)
    global_report = native.train_global(teacher, spec, raw, targets, dense)
    data = native.global_bank(teacher, raw); del raw
    pred = core.reuse.predict(teacher, spec, (data['validation'][0], None), np.full(len(spec['rows']['validation']), -1, np.int64))
    global_report['metrics'] = core.reuse.score_prediction(task, pred, targets['validation'], spec, RUN/task/'global/terminal_prediction')
    core.old.write(RUN/task/'global/results.json', global_report)
    cropped = crop_targets(task, dense['train'] if task == 'Marine' else targets['train'])
    model = DenseLocal(teacher).cuda(); updates = train_local(model, spec, data, targets, dense, cropped)
    bank = packed_bank(model, data)
    predictions = {} if task == 'Marine' else detection_banks(model, spec, bank)
    if task != 'Marine':
        train_calibrator(model, predictions['train'], normalized_targets(task, targets['train']))
    gains = gain_labels(model, spec, bank, dense, targets, predictions)
    selector, sr = core.reuse.train_selector(task, model.mode, model.context16(bank['train'][0]), gains)
    report = evaluate(model, spec, bank, targets, predictions, selector, sr, updates)
    online_cost(model, spec, selector, sr)
    parent_result = json.loads((PARENT/task/'results.json').read_text())
    previous = {'learned': parent_result['modes'][1]['metrics']['learned'][spec['metric']],
                'global_only': parent_result['global_only_metric_before'], 'scope': 'Completed singleheadseed0; not a3seedmean.'}
    comparison = max(core.reuse.protocol()['references'][task]['baseline_terminal_three_seed_mean'],
                     previous['learned'], previous['global_only'], global_report['metrics'][spec['metric']])
    core.old.write(RUN/task/'results.json', {'status': 'COMPLETE_HALO_REGIONAL_PREDICTIONS', 'task': task,
        'global_stage': global_report, 'local_stage': report, 'new_weights_saved': False,
        'strongest_compatible_reference': core.reuse.protocol()['references'][task],
        'prior_completed_terminal_comparisons': previous, 'strongest_completed_comparison_value': comparison,
        'learned_minus_strongest': report['metrics']['learned'][spec['metric']]-comparison, 'resources': native.resources()})


def pipeline():
    parent = json.loads((PARENT/'protocol.json').read_text())
    coordinator_parent = json.loads((COORDINATOR_PARENT/'protocol.json').read_text())
    prior = {task: {'result_path': str(PARENT/task/'results.json'),
                    'scope': 'Samepredeclaredparentcondition/task/metric, completed singleheadseed0 value whenavailable. Not a3seedmean; no output-dependent training/stopping.'}
             for task in parent['tasks']}
    RUN.mkdir(); (RUN/'logs').mkdir(); begun = time.time()
    protocol = {'start_unix': begun, 'deadline_unix': coordinator_parent['deadline_unix'], 'original_start_unix': coordinator_parent['start_unix'], 'inherited_pre_model_stop': str(COORDINATOR_PARENT/'actual_pre_model_stop.json'), 'inherited_CPU_staged_elapsed_seconds': begun-coordinator_parent['start_unix'],
        'encoder': parent['encoder'], 'tasks': parent['tasks'], 'references': parent['references'],
        'prior_completed_terminal_comparisons': prior,
        'SWiSE_references': parent['SWiSE_references'],
        'source_utf8': Path(__file__).read_text(), 'reused_native_source_utf8': Path(native.__file__).read_text(),
        'parent_fixed_run': str(PARENT),
        'principle_diagram': str(core.old.BASE/'gsr_method_diagrams_v1/05_halo_regional_predictions.svg'),
        'authorization': 'User authorizes rapid frozen-original-weight structural optimization on all3tasks, onlyphysicalGPU2/3, maxoneownedworker/card. No newsmoke/SHA/review/trainedweightfiles.',
        'global_stage': 'Same all12 convolutional bypasses and fourlevelneck, directpublished448, originalhead60FSOD/100Marine/UATD. OriginalDINO fixed. Same headoptimizers/order/schedules; modulesAdamW.001/wd.0001/clip1/200warmupcosine.',
        'local_stage': 'Separatecopiedlocalbypasses/head, fixedglobalmodel,30epochs randomonecornerROI/normalmacro. Directpublished768 four448crops atorigins0/320, overlap128pixels. NormalizedROIspan7/12, origins0/5/12. Local32field retainsdensity via localhead; UATDpool16 coversspan7/12 oftheframe, not scattereddown8. Localencodingdensity is12/7 ofglobal448; not the2xnonoverlappingcase.',
        'local_context': '64dim4head crossattention,1024localquerytokens/256globaltokens plus fullframe positions, zero-up residual. Global-conditioned localbypasses as before.',
        'local_optimizer': 'BYPASSESAdamWpeak.0001/200warmupcosine; bridge/MarinepixelgateAdamW.001; copiedlocalheadSGD.0005FSOD orAdamW.0001others. Allwd as originalnative localstage.',
        'Marine_loss': 'Fullnative mixedprobabilityNLL +.25cropCE +.25softDice(meanoverGTpresentclassesinmacro,smooth1). Two-branch local/global probability mixture in logdomain, sigmoidpixelgateinitialbias-2 times separablesinewindow. Native480x320 all12GT unchanged. Localauxlabels280x187 are nearestsampledfromnativeGT at exactROIcoordinates; logits/gate are bilinearlywarped backtofullnativegrid with explicitROImask.',
        'detector_loss': 'OriginalcroppedGT losses only, fullframe evaluationGT unchanged. FSOD sumoriginal4RPN/ROI losses, UATD focal+regression with macropositiveauxdenominator max(P,1), emptycropsretainnegativefocal. No nondifferentiableNMS loss claim.',
        'detector_merge': 'Globalboxes/scores fixed; maplocal224 ornormalizedROI to fullframe using origin(0or5/12) andspan7/12. Calibrate localscores, concatenate, classwiseIoU.5NMS/top100/score>=.05. FP32coordinate roundtrip is declared, not a bitwiseactive-path identity claim.',
        'calibrator': '10epochsAdamW.001/wd.0001/B512/private0.19inputs: scorelogit(eps1e-5),10classonehot,boxcenter/size,maxsameclassglobalIoU/globalconfidence,cornerIDxy.32GELUhidden,zero-output init. Train-only localbranch one-to-one scoreorderedIoU.5/fullframeGT binarylabels. Globalconfidenceunchanged before NMS.',
        'gain_labels': 'Signedtrain-only mergedprediction utility: Marineexistingclass/boundaryweightedNLL reduction; detection classweights inverse-sqrt alltrainGTcounts, weighted2TP/(GT+predictedcount), matchingIoU.5/score.05. Correctemptyimage utility1, falsepositiveempty0. Proxy isnot AP orphysicalGT.',
        'selector': 'Existingselector100epochs/private7, parentinner80/20seed37 unchanged. Global/localheads fittedalltrain; innercheck is selector-only, not independentof taskfitting. Retainnegativegains, skipifmaxpredictedgain<=0.',
        'policies': 'global_only,learned,forcedtop1,random/fixed/entropybudgetmatched,entropyforced,fusionfixedsamelearnedactions. Fixedfusion is Marine.5taper or uncalibratedlocaldetector scores; no extra headfit.',
        'endpoints': 'Terminal fullmethod only after globalfullschedule/local30/calibration10whereapplicable/selector100. Global scheduledbest isauxiliary, not fullmethodbest. OriginalnativeGT/canonicalandindependentNumPy scoring fromsavedpredictions.',
        'cost': '50warmup+first32fixeddevimages3rounds/B1, actualglobalencoder+optionalonefineencoder+localhead/merge/selector. Decodedraster through nativepostproc; decode excluded, trainingbanksresident, external/cardloadrecorded.',
        'storage': 'No newweights/featurefiles. GPU-RAMbanks only; destructivelyreuse localpatchembeddingstorage for finalfine32features afteralllocaltraining. Requiredpred/gainlabels/logs/meta only.',
        'caps': {'global_seconds': 28800, 'workflow_seconds': 14400, 'RSS_bytes': 48*1024**3, 'new_disk_bytes': 4*1024**3},
        'GPU_caps_bytes': parent['GPU_caps_bytes'], 'required_free_MiB': parent['required_free_MiB'],
        'CPU_affinity': {'2': AVAILABLE_CPUS[:4], '3': AVAILABLE_CPUS[4:8]},
        'schedule': 'CPUstagedqueue now. MarineGPU2 onlyafterparentFSODactualworkerwait0 andnormaladmission; no furtherparentGPU2phases. NewUATDGPU3 onlyafteractualparentforegroundwait0. NewFSODGPU2 afternewMarine. Acrossbothunits maxoneownedworker/card,2total.',
        'release_receipts': 'ROOT atomicallypublishes release_Marine_GPU2.json from actualparentFSODwait0; avoids concurrentlyreading mutableparentqueueJSON. Parentactualforegroundwait.json is likewisepublished atomically byROOT after retainedsession75777 actuallyreturns0. Neither receipt grants protocol/math changes.',
        'stop': 'Realerror/nonfinite/hardresources/deadlines stopentireunit. No score-basedstop/automaticretry/batchprecisioncapchanges.',
        'total_global_updates': 43000, 'total_local_updates': 16440,
        'acceptance': 'All3learnedterminals exceedqualifiedSWiSE and arewithin.01 of max(strongestcompatiblecompletedlegacymean,completedparentlearned/globalterminals,ownnewglobalbaseline). Prior singleheadseed0 observations remain separately labeled; no significance/worldSOTA/noveltyclaim.',
        'train_geometry_source': str(COORDINATOR_PARENT/'train_crop_geometry.json'),
        'comparison_limit': 'The oldnonoverlappingdenseproposal hadzero modelworkers andnoAP; no trainedquarter-vs-haloeffect is claimed. This changesROIcoverage/localoutputdensity/contextbridge/trainingloss/calibration together, so differences frompriorrun do not isolate a cause. Completepaths/pretraining/budgets differ fromstrongspecialists.'}
    for task in protocol['tasks']:
        folder = RUN/task; folder.mkdir()
        for name in ('local_actions.npy', 'selector_split.npz'):
            (folder/name).write_bytes((PARENT/task/name).read_bytes())
    core.old.write(RUN/'protocol.json', protocol)
    active = {}; waits = []; marine_started = False; uatd_started = False
    def launch(task, gpu):
        admission = core.old.admission(gpu)
        if admission['free_MiB'] < protocol['required_free_MiB'][task]:
            raise RuntimeError(task+' fine-bank memory admission insufficient '+str(admission))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'XFORMERS_DISABLED': '1'}
        cpus = ','.join(map(str, protocol['CPU_affinity'][str(gpu)]))
        log = (RUN/'logs'/f'{task}.log').open('w')
        child = subprocess.Popen(['/usr/bin/taskset', '-c', cpus, core.old.PYTHON, '-B', '-u', '-m',
            'runs.gsr_halo_predictions', task], cwd=core.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[task] = (child, log, gpu, time.time(), admission)
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'observed_unix': time.time(),
            'workers': {name: value[0].pid for name, value in active.items()}, 'new_weight_files': 0,
            'resource_staging': protocol['schedule']})
        core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {name: value[0].pid for name, value in active.items()}, 'actual_waits': waits})
    try:
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'observed_unix': time.time(),
            'workers': {}, 'status': 'CPU_STAGED_NO_NEW_MODEL', 'resource_staging': protocol['schedule']})
        core.old.write(RUN/'queue_state.json', {'status': 'CPU_STAGED_WAIT_PARENT_FSOD_ACTUAL_EXIT0', 'active': {}, 'actual_waits': []})
        while active or not marine_started or not uatd_started:
            if (PARENT/'failure_closure.json').exists():
                raise RuntimeError('Parentfixedunit stopped; no automaticparentrepair or successorcontinuation')
            release = RUN/'release_Marine_GPU2.json'
            if not marine_started and release.exists():
                finished = json.loads(release.read_text())
                if finished['task'] != 'FSOD' or finished['actual_wait_exit'] != 0:
                    raise RuntimeError('ParentFSOD actualexitnonzero; stagedGPU2 launchcancelled')
                launch('Marine', 2); marine_started = True
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
                if task == 'Marine':
                    launch('FSOD', 2)
                core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in active.items()}, 'actual_waits': waits})
            time.sleep(1)
        tasks = {task: json.loads((RUN/task/'results.json').read_text()) for task in protocol['tasks']}
        core.old.write(RUN/'results.json', {'status': 'COMPLETE_FIXED_HALO_PREDICTIONS', 'tasks': tasks})
        size = sum(path.stat().st_size for path in RUN.rglob('*') if path.is_file())
        if size > protocol['caps']['new_disk_bytes']:
            raise RuntimeError('Results-only4GiB cap exceeded')
        closure = {'status': 'COMPLETE_SCOPE_GOAL_REQUIRES_RESULT_DECISION', 'actual_waits': waits,
                   'wall_seconds': time.time()-begun, 'lineage_wall_seconds': time.time()-protocol['original_start_unix'], 'inherited_CPU_staged_elapsed_seconds': protocol['inherited_CPU_staged_elapsed_seconds'], 'new_entity_bytes': size, 'new_weights_saved': False}
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
