"""Native fine views with a preserved global path; original DINO weights fixed."""
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
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from runs import gsr_frozen_bypass as previous
from runs import downstream_vit_finetuning_readout as reader
from sonar_encoder.training.model import build_encoder
from sonar_encoder.training.uatd_head import box_targets

core = previous.core
BASE = core.old.BASE
RUN = BASE/'gsr_native_evidence_indexfix_v1'
PREVIOUS = BASE/'gsr_frozen_bypass_v1'
core.RUN = RUN
core.reuse.RUN = RUN
AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))
ACTIVE_TASK = ''


def resources():
    import resource
    p = core.reuse.protocol()
    value = {'RSS_HWM_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             'GPU_reserved_peak_bytes': torch.cuda.max_memory_reserved()}
    if value['RSS_HWM_bytes'] > 48*1024**3 or value['GPU_reserved_peak_bytes'] > p['GPU_caps_bytes'][ACTIVE_TASK]:
        raise RuntimeError('Native-evidence resource cap '+str(value))
    if time.time() > p['deadline_unix']:
        raise TimeoutError('Original8h native-evidence deadline')
    return value


core.reuse.resources = resources


def read_intensity(row):
    with Image.open(row['image']) as image:
        x = torch.from_numpy(np.array(image.convert('L'), dtype=np.uint8)).float()[None]/255
    return x


class GlobalModel(nn.Module):
    def __init__(self, task):
        super().__init__(); self.task = task
        cfg = core.reuse.protocol()['encoder']
        self.encoder = build_encoder(Path(cfg['code_dir']), Path(cfg['weights_path']), 'R010').eval().requires_grad_(False)
        self.head = reader.new_head(task, 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            self.neck = core.SpatialNeck()
            self.bypasses = nn.ModuleList([previous.SpatialBypass() for _ in range(12)])
        for bypass in self.bypasses:
            bypass.condition.requires_grad_(False)

    def field(self, tokens, bypasses, local=False, context=None, actions=None):
        values = []
        for i, block in enumerate(self.encoder.backbone.blocks):
            delta = bypasses[i](block.norm1(tokens), context, actions)
            tokens = block(tokens)+delta
            if i in (2, 5, 8, 11):
                f = self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, 32, 32)
                values.append(F.avg_pool2d(f, 2) if self.task == 'UATD' and not local else f)
        return self.neck(torch.stack(values, 1))

    def global_field(self, images):
        tokens = self.encoder.backbone.prepare_tokens_with_masks(images)
        return self.field(tokens, self.bypasses)


class EvidencePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Conv2d(384, 32, 1)
        self.key = nn.Conv2d(384, 32, 1)
        nn.init.zeros_(self.query.weight); nn.init.zeros_(self.query.bias)

    def forward(self, fine, base):
        query = self.query(base)[:, :, :, :, None]
        keys = self.key(fine).unfold(2, 2, 2).unfold(3, 2, 2).flatten(-2)
        weights = ((query*keys).sum(1)/math.sqrt(32)).softmax(-1)
        values = fine.unfold(2, 2, 2).unfold(3, 2, 2).flatten(-2)
        return (values*weights[:, None]).sum(-1)


class EvidenceFusion(core.old.Fusion):
    def forward(self, local, base, global_field, action):
        n = len(local)
        q = (self.local(local)+self.base(base)).flatten(2).transpose(1, 2)
        q = q+self.position(self.local_xy[action.clamp_min(0)])
        v = self.context(global_field).flatten(2).transpose(1, 2)
        k = v+self.position(self.global_xy)[None]
        heads = lambda x: x.reshape(n, 256, 4, 16).transpose(1, 2)
        attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0.)
        u = F.gelu(self.norm(q+attended.transpose(1, 2).reshape(n, 256, 64))).transpose(1, 2).reshape(n, 64, 16, 16)
        return (local-base+self.output(u))*self.gate(u).sigmoid()*self.taper


class LocalModel(nn.Module):
    context16 = core.reuse.Model.context16
    corrected = core.reuse.Model.corrected

    def __init__(self, teacher):
        super().__init__(); self.task = teacher.task; self.mode = 'native_local'
        self.teacher = teacher
        self.head = teacher.head
        self.local_head = copy.deepcopy(teacher.head).requires_grad_(True)
        self.bypasses = copy.deepcopy(teacher.bypasses).requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(102)
            self.pool = EvidencePool()
            self.fusion = EvidenceFusion()

    def fine_field(self, tokens, g, actions):
        return self.teacher.field(tokens, self.bypasses, True, g, actions)

    def pooled(self, fine, g, actions, average=False):
        if average:
            return F.avg_pool2d(fine, 2)
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False) if self.task == 'UATD' else g
        crops = torch.stack([base[..., r:r+16, c:c+16] for r, c in core.old.QUADS], 1)
        crop = crops[torch.arange(len(g), device=g.device), actions]
        return self.pool(fine, crop)


@torch.no_grad()
def extract(teacher, spec):
    result = {}; count = 0; started = time.monotonic()
    for split, rows in spec['rows'].items():
        global_tokens = torch.empty((len(rows), 1025, 384), device='cuda')
        local_tokens = torch.empty((len(rows), 4, 1025, 384), device='cuda')
        for start in range(0, len(rows), 8):
            global_images = []; fine_images = []
            for row in rows[start:start+8]:
                native = (read_intensity(row).log1p()-spec['normalization']['mean'])/spec['normalization']['std']
                global_images.append(F.interpolate(native[None], (448, 448), mode='bilinear', align_corners=False, antialias=True)[0])
                large = F.interpolate(native[None], (896, 896), mode='bilinear', align_corners=False, antialias=True)[0]
                fine_images.append(torch.stack([large[..., r:r+448, c:c+448] for r, c in ((0, 0), (0, 448), (448, 0), (448, 448))]))
            x = torch.stack(global_images).cuda(); n = len(x)
            global_tokens[start:start+n] = teacher.encoder.backbone.prepare_tokens_with_masks(x)
            fine = torch.stack(fine_images).flatten(0, 1).cuda()
            local_tokens[start:start+n] = teacher.encoder.backbone.prepare_tokens_with_masks(fine).reshape(n, 4, 1025, 384)
            count += n
            if start % 256 == 0:
                print(json.dumps({'extract': teacher.task, 'split': split, 'rows': start+n}), flush=True)
                resources()
        result[split] = (global_tokens, local_tokens)
    core.old.write(RUN/teacher.task/'extraction.json', {'published_rasters_decoded': count,
        'frozen_patch_projection_global_views': count, 'frozen_patch_projection_fine_views': 4*count,
        'full_transformer_blocks_run_during_extraction': 0, 'GPU_RAM_only': True,
        'wall_seconds': time.monotonic()-started, 'resources': resources()})
    return result


def module_lr(completed, total, peak):
    step = completed+1
    if step <= 200:
        return peak*step/200
    return peak*(.01+.99*(1+math.cos(math.pi*(step-200)/(total-200)))/2)


@torch.no_grad()
def predict_teacher(teacher, tokens, rows, batch):
    teacher.eval(); pieces = []
    for start in range(0, len(rows), batch):
        g = teacher.field(tokens[start:start+batch], teacher.bypasses)
        pieces.append(reader.decode(teacher.task, teacher.head, g, rows[start:start+batch]))
    return [x for part in pieces for x in part] if teacher.task == 'UATD' else np.concatenate(pieces)


def quick_metric(task, spec, predictions, targets):
    if task == 'FSOD':
        from runs.shared_supervised_json_readout import canonical_evaluate
        from runs.shared_supervised_readout import prediction_list
        return canonical_evaluate(json.loads(Path(spec['GT_paths']['validation']).read_text()), prediction_list(predictions))['AP50']
    if task == 'UATD':
        from sonar_encoder.training.uatd_head import detection_metrics
        return detection_metrics(predictions, targets)['map50']
    from sonar_encoder.training.readout import confusion_matrix, metrics_from_confusion
    return metrics_from_confusion(confusion_matrix(targets.cpu(), torch.as_tensor(predictions), 12), 'marine')['miou12']


def train_global(teacher, spec, raw, targets, dense):
    folder = RUN/teacher.task/'global'; folder.mkdir()
    head_opt = reader.head_optimizer(teacher.task, teacher.head)
    extra = [p for name, p in teacher.named_parameters() if p.requires_grad and not name.startswith('head.')]
    opt = torch.optim.AdamW(extra, lr=.001, weight_decay=.0001)
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
                value = 0.; denominator = dense['train'][2][macro].sum() if teacher.task == 'UATD' else len(macro)
                for ids in macro.split(spec['micro']):
                    g = teacher.field(raw['train'][0][ids], teacher.bypasses)
                    target = core.reuse.targets_batch(teacher.task, targets['train'], dense.get('train'), ids)
                    weight = target[2].sum()/denominator if teacher.task == 'UATD' else len(ids)/denominator
                    loss = reader.task_loss(teacher.task, teacher.head, g, target)*weight
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite real global training loss')
                    loss.backward(); value += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(extra, 1., error_if_nonfinite=True)
                head_opt.step(); opt.step(); updates += 1; total_loss += value*len(macro)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {'original_body_trainable': 0,
                        'original_body_has_grad': any(p.grad is not None for p in teacher.encoder.parameters()),
                        'all12_new_bypass_grad_norms': [float(sum(p.grad.square().sum() for p in b.parameters() if p.grad is not None).sqrt()) for b in teacher.bypasses],
                        'global_grid': list(g.shape), 'macro': len(macro), 'micro': spec['micro'], 'loss': value})
                if updates % 100 == 0:
                    resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': total_loss/len(order), 'wall_seconds': time.monotonic()-begun}
            if epoch % 5 == 0:
                pred = predict_teacher(teacher, raw['validation'][0], spec['rows']['validation'], spec['micro'])
                score = quick_metric(teacher.task, spec, pred, dense['validation'] if teacher.task == 'Marine' else targets['validation'])
                entry['global_metric'] = score
                if score > best:
                    best, best_epoch = score, epoch
                print(json.dumps({'task': teacher.task, 'phase': 'global', **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    teacher.eval().requires_grad_(False)
    for p in teacher.parameters():
        p.grad = None
    report = {'task': teacher.task, 'mode': 'global', 'epochs': spec['epochs'], 'updates': updates,
        'scheduled_terminal_metric': score, 'scheduled_best': best, 'best_epoch': best_epoch,
        'wall_seconds': time.monotonic()-begun, 'original_DINO_trainable': 0,
        'new_weights_saved': False, 'resources': resources()}
    core.old.write(folder/'train_report.json', report)
    return report


@torch.no_grad()
def global_bank(teacher, raw):
    output = {}
    grid = 16 if teacher.task == 'UATD' else 32
    for split, (tokens, fine_tokens) in raw.items():
        g = torch.empty((len(tokens), 384, grid, grid), device='cuda')
        for start in range(0, len(tokens), 8):
            g[start:start+8] = teacher.field(tokens[start:start+8], teacher.bypasses)
        output[split] = (g, fine_tokens)
    return output


def local_aux(model, fine, pooled, ids, actions, cropped, denominator):
    indices = ids.cpu().tolist(); regions = actions.cpu().tolist()
    if model.task == 'Marine':
        labels = torch.stack([cropped[j][i] for i, j in zip(indices, regions)])
        return F.cross_entropy(F.interpolate(model.local_head(fine), (240, 160), mode='bilinear', align_corners=False), labels.long())
    if model.task == 'FSOD':
        target = [{k: v.cuda() for k, v in cropped[j][i].items()} for i, j in zip(indices, regions)]
        return reader.task_loss('FSOD', model.local_head, fine, target)
    heat, reg, pos = [torch.stack([cropped[j][k][i] for i, j in zip(indices, regions)]) for k in range(3)]
    output = model.local_head(pooled); logits = output[:, :10]; prob = logits.sigmoid()
    ptarget = torch.where(pos, prob, 1-prob)
    focal = (torch.where(pos, .25, .75)*(1-ptarget).square()*F.binary_cross_entropy_with_logits(logits, heat, reduction='none')).sum()
    raw = output[:, 10:].reshape(-1, 10, 4, 16, 16)
    pred = torch.cat((raw[:, :, :2].sigmoid(), raw[:, :, 2:]), 2)
    regression = (F.smooth_l1_loss(pred, reg, reduction='none')*pos[:, :, None]).sum()/4
    return (focal+regression)/denominator


@torch.no_grad()
def predict_local(model, spec, data, actions):
    model.eval(); g, tokens = data; pieces = []
    for start in range(0, len(g), spec['micro']):
        sl = slice(start, start+spec['micro']); a = actions[sl]
        fine = model.fine_field(tokens[sl][torch.arange(len(a), device=tokens.device), a], g[sl], a)
        pooled = model.pooled(fine, g[sl], a)
        field = model.corrected(g[sl], pooled, a)
        pieces.append(reader.decode(model.task, model.head, field, spec['rows']['validation'][sl]))
    return [x for part in pieces for x in part] if model.task == 'UATD' else np.concatenate(pieces)


def train_local(model, spec, data, targets, dense, cropped):
    folder = RUN/model.task/model.mode; folder.mkdir()
    aux_opt = reader.head_optimizer(model.task, model.local_head)
    for group in aux_opt.param_groups:
        group['lr'] = .0005 if model.task == 'FSOD' else .0001
    opt = torch.optim.AdamW([{'params': model.bypasses.parameters(), 'lr': .0001},
        {'params': list(model.pool.parameters())+list(model.fusion.parameters()), 'lr': .001}], weight_decay=.0001)
    generator = torch.Generator().manual_seed(0); actions = torch.from_numpy(np.load(RUN/model.task/'local_actions.npy')).cuda()
    val_actions = torch.from_numpy(np.load(RUN/model.task/'validation_actions.npy')).cuda()
    counts = torch.stack([x[2].flatten(1).sum(1) for x in cropped], 1) if model.task == 'UATD' else None
    updates = 0; total_updates = 30*math.ceil(len(spec['rows']['train'])/spec['batch']); best = -1.; best_epoch = 0; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, 31):
            model.train(); model.teacher.encoder.eval(); order = torch.randperm(len(spec['rows']['train']), generator=generator); total_loss = 0.
            for macro in order.split(spec['batch']):
                macro_actions = actions[epoch-1, macro]
                aux_denominator = counts[macro, macro_actions].sum().clamp_min(1) if model.task == 'UATD' else len(macro)
                main_denominator = dense['train'][2][macro].sum() if model.task == 'UATD' else len(macro)
                aux_opt.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = module_lr(updates, total_updates, .0001)
                value = 0.
                for ids in macro.split(spec['micro']):
                    a = actions[epoch-1, ids]; g = data['train'][0][ids]
                    fine = model.fine_field(data['train'][1][ids, a], g, a)
                    pooled = model.pooled(fine, g, a)
                    field = model.corrected(g, pooled, a)
                    target = core.reuse.targets_batch(model.task, targets['train'], dense.get('train'), ids)
                    weight = target[2].sum()/main_denominator if model.task == 'UATD' else len(ids)/main_denominator
                    main = reader.task_loss(model.task, model.head, field, target)*weight
                    auxiliary = local_aux(model, fine, pooled, ids, a, cropped, aux_denominator)
                    if model.task != 'UATD':
                        auxiliary = auxiliary*len(ids)/len(macro)
                    loss = main+.25*auxiliary
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite real local training loss')
                    loss.backward(); value += float(loss.detach())
                trainable = [p for group in opt.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                opt.step(); aux_opt.step(); updates += 1; total_loss += value*len(macro)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {'original_DINO_trainable': 0,
                        'global_teacher_trainable': sum(p.numel() for p in model.teacher.parameters() if p.requires_grad),
                        'global_teacher_has_grad': any(p.grad is not None for p in model.teacher.parameters()),
                        'local_bypass_grad_norms': [float(sum(p.grad.square().sum() for p in b.parameters() if p.grad is not None).sqrt()) for b in model.bypasses],
                        'fine_field': list(fine.shape), 'pooled_field': list(pooled.shape), 'macro': len(macro), 'micro': spec['micro'], 'loss': value})
                if updates % 100 == 0:
                    resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': total_loss/len(order), 'wall_seconds': time.monotonic()-begun}
            if epoch % 5 == 0:
                pred = predict_local(model, spec, data['validation'], val_actions)
                score = quick_metric(model.task, spec, pred, dense['validation'] if model.task == 'Marine' else targets['validation'])
                entry['scheduled_random_metric'] = score
                if score > best:
                    best, best_epoch = score, epoch
                print(json.dumps({'task': model.task, 'phase': 'local', **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    model.eval().requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    core.old.write(folder/'train_report.json', {'epochs': 30, 'updates': updates, 'scheduled_random_terminal': score,
        'scheduled_random_best': best, 'scheduled_best_epoch': best_epoch, 'best_scope': 'Random-route auxiliary, not complete learned-policy best.',
        'global_path_fixed': True, 'original_DINO_trainable': 0, 'wall_seconds': time.monotonic()-begun, 'resources': resources()})
    return updates


@torch.no_grad()
def local_bank(model, data):
    result = {}
    for split, (g, tokens) in data.items():
        l = torch.empty((len(g), 4, 384, 16, 16), device='cuda')
        for start in range(0, len(g), 8):
            end = min(start+8, len(g)); current = g[start:end]
            for region in range(4):
                a = torch.full((end-start,), region, device='cuda', dtype=torch.long)
                fine = model.fine_field(tokens[start:end, region], current, a)
                l[start:end, region] = model.pooled(fine, current, a)
            if start % 256 == 0:
                resources()
        result[split] = (g, l)
    return result


@torch.no_grad()
def online_cost(model, teacher, spec, selector=None, sr=None):
    samples = [read_intensity(row) for row in spec['rows']['validation'][:32]]
    cells = []
    for policy in (('global_only',) if selector is None else ('global_only', 'learned')):
        load = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
        baseline = torch.cuda.memory_reserved(); torch.cuda.reset_peak_memory_stats(); records = []
        for visit in range(146):
            i = visit % 32 if visit < 50 else (visit-50) % 32
            torch.cuda.synchronize(); start = time.perf_counter()
            native = (samples[i].log1p()-spec['normalization']['mean'])/spec['normalization']['std']
            global_input = F.interpolate(native[None], (448, 448), mode='bilinear', align_corners=False, antialias=True).cuda()
            g = teacher.global_field(global_input); field = g; action = -1
            if policy == 'learned':
                scores = selector(model.context16(g))*sr['gain_scale']+sr['gain_mean']
                if float(scores.max()) > 0:
                    action = int(scores.argmax(1))
                    large = F.interpolate(native[None], (896, 896), mode='bilinear', align_corners=False, antialias=True)
                    y, x = action//2*448, action%2*448
                    fine_input = large[..., y:y+448, x:x+448].cuda()
                    tokens = teacher.encoder.backbone.prepare_tokens_with_masks(fine_input)
                    a = torch.tensor([action], dtype=torch.long, device='cuda')
                    fine = model.fine_field(tokens, g, a)
                    field = model.corrected(g, model.pooled(fine, g, a), a)
            reader.decode(teacher.task, teacher.head, field, spec['rows']['validation'][i:i+1])
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            if visit >= 50:
                records.append({'row_id': spec['row_ids']['validation'][i], 'round': (visit-50)//32,
                                'seconds': elapsed, 'action': action})
        times = [x['seconds'] for x in records]
        cells.append({'policy': policy, 'median_seconds': float(np.median(times)), 'p95_seconds': float(np.percentile(times, 95)),
            'raw': records, 'GPU_load': load, 'baseline_reserved': baseline,
            'peak_reserved': torch.cuda.max_memory_reserved(), 'peak_allocated': torch.cuda.max_memory_allocated()})
    folder = RUN/teacher.task/('global' if selector is None else model.mode)
    core.old.write(folder/'cost.json', {'cells': cells,
        'boundary': 'Decoded published raster through normalization, resize, H2D, actualglobalencoder, optionalactualfineencoder, adapters/pooling/selection/fusion/head/nativepostproc. Decode/setup excluded.',
        'memory_scope': 'Training token/featurebanks resident; not isolated deployment memory.', 'other_GPU_training_may_be_active': True})


def workflow(task):
    global ACTIVE_TASK
    ACTIVE_TASK = task
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    spec = core.reuse.protocol()['tasks'][task]
    targets = {split: reader.make_targets(task, rows) for split, rows in spec['rows'].items()}
    dense = {split: box_targets(value, 'cuda') for split, value in targets.items()} if task == 'UATD' else (
        {split: value.cuda() for split, value in targets.items()} if task == 'Marine' else {})
    teacher = GlobalModel(task).cuda(); raw = extract(teacher, spec)
    global_result = train_global(teacher, spec, raw, targets, dense)
    data = global_bank(teacher, raw); del raw
    # Both retention endpoints use identical cached G0 and head batch sizes.
    actions = np.full(len(spec['rows']['validation']), -1, np.int64)
    pred = core.reuse.predict(teacher, spec, (data['validation'][0], None), actions)
    canonical = core.reuse.score_prediction(task, pred, targets['validation'], spec, RUN/task/'global/terminal_prediction')
    global_result['metrics'] = {'global_only': canonical}
    global_result['primary_source'] = 'Same cachedG0 and head batches used for no-region endpoint afterlocalstage'
    core.old.write(RUN/task/'global/results.json', global_result)
    online_cost(None, teacher, spec)
    cropped = previous.crop_targets(task, dense['train'] if task == 'Marine' else targets['train'])
    model = LocalModel(teacher).cuda()
    updates = train_local(model, spec, data, targets, dense, cropped)
    bank = local_bank(model, data)
    gains = core.reuse.gain_labels(model, spec, bank['train'], targets['train'], dense.get('train'))
    selector, sr = core.reuse.train_selector(task, model.mode, model.context16(bank['train'][0]), gains)
    local_result = core.evaluate(model, spec, bank, targets, selector, sr, updates)
    online_cost(model, teacher, spec, selector, sr)
    global_score = global_result['metrics']['global_only'][spec['metric']]
    retained = local_result['metrics']['global_only'][spec['metric']]
    report = {'status': 'COMPLETE_GLOBAL_FULL_SCHEDULE_PLUS_LOCAL30', 'task': task,
        'modes': [global_result, local_result], 'global_only_metric_before': global_score,
        'global_only_metric_after': retained, 'global_only_delta': retained-global_score,
        'global_path_parameter_updates_during_local': 0, 'original_DINO_updates': 0,
        'strongest_comparison': max(global_score, core.reuse.protocol()['references'][task]['baseline_terminal_three_seed_mean']),
        'new_weight_files': 0, 'resources': resources()}
    core.old.write(RUN/task/'results.json', report)


def pipeline():
    context = BASE/'gsr_frozen_context_resume_v1'
    if json.loads((context/'actual_foreground_wait.json').read_text())['actual_foreground_exit'] != 0:
        raise RuntimeError('Preceding fixed context scope must actually finish before newGPU2 allocation')
    RUN.mkdir(); (RUN/'logs').mkdir(); start = time.time()
    parent_protocol = json.loads((BASE/'gsr_native_evidence_v1/protocol.json').read_text())
    inherited_seconds = start-parent_protocol['start_unix']
    prep = json.loads((BASE/'downstream_vit_finetuning_v1/cpu_preparation.json').read_text())
    refs = json.loads((BASE/'gsr_detection_transfer_v1/strongest_baseline_comparison.json').read_text())['tasks']
    p = {'start_unix': start, 'deadline_unix': parent_protocol['deadline_unix'], 'original_start_unix': parent_protocol['start_unix'], 'inherited_elapsed_seconds': inherited_seconds, 'engineering_stop': str(BASE/'gsr_native_evidence_v1/engineering_stop.json'), 'encoder': prep['official_config'], 'references': refs,
        'source_utf8': Path(__file__).read_text(), 'tasks': {},
        'authorization': 'User permits original-weight-frozen trainable inter-layer modules; onlyGPU2/3, maxone ownedworker/card, no newweights/smoke/SHA/review.',
        'architecture': 'OriginalDINO fixed; all12 zero-up32dim convolutional bypasses, same4-level spatialneck. Direct published-raster448 global, direct896 raster view with four448crops. Fine encoding32 perquadrant versusglobal32 fullframe.',
        'global_stage': 'Original head60FSOD/100Marine/100UATD schedule; newmodules AdamW.001 wd.0001 clip1,200updatewarmup cosine to1% over fullstage. UATD global4levels32pool16 before neck.',
        'local_stage': 'Freeze entire fitted global model. Separate localbypasses and auxiliaryhead initialized from its in-RAM state.30epochs, originalDINO sharedfixed. Localbypasses AdamWpeak.0001/200warmupcosine; pool+fusionAdamW.001; auxiliaryheadLR.0005FSOD/.0001others with original optimizer types.',
        'local_loss': 'Original full-image taskloss through frozen globalhead on correctedfield +.25croppedGT auxiliary through separatelocalhead. Original GT files/evaluation unchanged. Cropped targets are declared train-only geometric transforms, not new physicalGT.',
        'pool': 'Within each2x2 fine token cell, softmax(globalquery dot localkeys/sqrt32) weighted rawlocalvalues; queryzero initialization givesuniformweights. Ordinary conditionalattention pooling, no novelty certification.',
        'residual': 'sigmoidgate*(pooledlocal-basecrop+learnedglobalcontextcorrection)*sinetaper. This directly exposes fine evidence while preserving the no-region global path.',
        'training_routes': 'Localstage randomonequadrant everyimage; no-region route already provided by fitted/frozen global stage. Inference atmostone, skip when predictedbest signedgain<=0.',
        'microbatch_math': 'UATD macro32/micro16acc2. Full-image dense losses weighted by micropositivecount/macropositivecount. Auxiliary focal and regression sums share max(macroauxpositivecount,1) denominator, retain emptycrop negatives. One optimizer update/macro.',
        'selector': 'Existing spatial selector, balanced signed train-only5action gainlabels, inner80/20 seed37; selector100epochs seed7. Same scope as earlier, not independently held out from global/local fitting.',
        'storage': 'GPU-RAM-only frozen patchembedding/PE tokens and trained-global features; no newweight/featurefiles. Globalmodel remains inRAM forlocal stage. Necessary nativepredictions/gainlabels/logs only.',
        'principle_diagram': str(BASE/'gsr_method_diagrams_v1/03_native_preserved_global.svg'),
        'global_retention_comparison': 'Before/after local stage use identical cachedG0 and head batches; separately recorded online cost is full nativeinput computation.',
        'caps': {'global_seconds': 28800, 'workflow_seconds': 14400-inherited_seconds, 'RSS_bytes': 48*1024**3, 'new_disk_bytes': 4*1024**3},
        'GPU_caps_bytes': {'Marine': 48*1024**3, 'FSOD': 64*1024**3, 'UATD': 88*1024**3},
        'required_free_MiB': {'Marine': 52*1024, 'FSOD': 68*1024, 'UATD': 92*1024},
        'CPU_affinity': {'2': AVAILABLE_CPUS[:4], '3': AVAILABLE_CPUS[4:8]},
        'schedule': 'MarineGPU2 now after contextactualwait0; FSODGPU2 afterMarine. UATDGPU3 only after bypassFSODactualwait0+closure. Atmosttwo ownedmodelworkers acrossallunits.',
        'stop': 'Real errors/nonfinite/hardmemory/deadlines stop this unit; no automatic retries, precision/batch changes or score-based stopping.',
        'SWiSE_references': {'FSOD': .902005317706368, 'Marine': .6059200833586624, 'UATD': .2517123460319482},
        'acceptance': 'Every learnedterminal must exceedSWiSE and be within.01 ofmax(strongestcompletedcompatiblelegacymean, itsnewglobalcontrol). Singleheadseed0 developmentpilot, not worldSOTA/robustness/novelty proof.',
        'comparability': 'Nativepublishedraster inputs/newmodules/extra localtraining differ fromhistoricalSWiSE. Preservedglobalcontrol separateslocalstageutility, not every bundled mechanism. Oldnative_tilesnegative remains.',
        'best_scope': 'Scheduledglobal orrandom-route best onlyauxiliary. Fullmethodterminal usesglobalterminal+local30+selector100; no auxiliary rescue.'}
    for task, batch, micro, epochs in [('Marine', 16, 16, 100), ('FSOD', 8, 8, 60), ('UATD', 32, 16, 100)]:
        folder = RUN/task; folder.mkdir()
        p['tasks'][task] = {**prep['inputs'][task], 'row_ids': prep['row_ids'][task], 'batch': batch, 'micro': micro, 'epochs': epochs,
                           'metric': {'Marine': 'miou12', 'FSOD': 'AP50', 'UATD': 'map50'}[task]}
        n = len(prep['inputs'][task]['rows']['train']); split = np.random.default_rng(37).permutation(n); cut = int(.8*n)
        np.savez(folder/'selector_split.npz', fit=split[:cut], check=split[cut:])
        np.save(folder/'local_actions.npy', np.random.default_rng(1).integers(0, 4, (30, n), dtype=np.int64))
        np.save(folder/'validation_actions.npy', np.random.default_rng(2).integers(0, 4, len(prep['inputs'][task]['rows']['validation']), dtype=np.int64))
    p['total_global_task_updates'] = 43000
    p['total_local_task_updates'] = 16440
    p['static_patch_token_bytes'] = {task: sum(len(x) for x in spec['rows'].values())*5*1025*384*4 for task, spec in p['tasks'].items()}
    core.old.write(RUN/'protocol.json', p)
    active = {}; waits = []; uatd_started = False; fsod_started = False
    def launch(task, gpu):
        admission = core.old.admission(gpu)
        if admission['free_MiB'] < p['required_free_MiB'][task]:
            raise RuntimeError(task+' large-fine-bank admission insufficient '+str(admission))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'XFORMERS_DISABLED': '1'}
        log = (RUN/'logs'/f'{task}.log').open('w'); cpus = ','.join(map(str, p['CPU_affinity'][str(gpu)]))
        child = subprocess.Popen(['/usr/bin/taskset', '-c', cpus, core.old.PYTHON, '-B', '-u', '-m',
            'runs.gsr_native_evidence_indexfix', task], cwd=core.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[task] = (child, log, gpu, time.time(), admission)
    try:
        launch('Marine', 2)
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'workers': {k: v[0].pid for k, v in active.items()},
            'staged_UATD_waiting_for_GPU3': True, 'no_new_weights_saved': True})
        while active or not uatd_started or not fsod_started:
            prior_wait = PREVIOUS/'FSOD/actual_worker_wait.json'
            if not uatd_started and prior_wait.exists():
                if json.loads(prior_wait.read_text())['actual_wait_exit'] != 0:
                    raise RuntimeError('Preceding bypassFSOD failed; no stagedGPU3 launch')
                json.loads((PREVIOUS/'wave_closure.json').read_text())
                launch('UATD', 3); uatd_started = True
            if time.time() > p['deadline_unix']:
                raise TimeoutError('Original8h deadline including stagedwaiting')
            for task, (child, log, gpu, begun, admission) in list(active.items()):
                if time.time()-begun > p['caps']['workflow_seconds']:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del active[task]
                waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code,
                              'wall_seconds': time.time()-begun, 'admission': admission})
                if code != 0:
                    raise RuntimeError(task+' actual worker exit '+str(code))
                if task == 'Marine':
                    launch('FSOD', 2); fsod_started = True
                core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in active.items()}, 'actual_waits': waits})
            time.sleep(1)
        results = {task: json.loads((RUN/task/'results.json').read_text()) for task in p['tasks']}
        core.old.write(RUN/'results.json', {'status': 'COMPLETE_FIXED_NATIVE_EVIDENCE', 'tasks': results})
        size = sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file())
        if size > p['caps']['new_disk_bytes']:
            raise RuntimeError('Original4GiB results-only cap exceeded')
        close = {'actual_waits': waits, 'wall_seconds': time.time()-start, 'new_entity_bytes': size,
            'original_DINO_weights_fixed': True, 'no_new_weights_saved': True, 'status': 'COMPLETE_SCOPE_GOAL_REQUIRES_RESULT_DECISION'}
        core.old.write(RUN/'wave_closure.json', close); core.old.write(RUN/'writing_handoff.json', close)
        core.old.write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        failure = traceback.format_exc()
        for task, (child, log, gpu, begun, admission) in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            code = child.wait(); log.close(); waits.append({'task': task, 'PID': child.pid, 'actual_wait_exit': code, 'cancelled_after_failure': True})
        core.old.write(RUN/'failure_closure.json', {'traceback': failure, 'actual_waits': waits, 'status': 'STOPPED_NO_AUTOMATIC_RETRY'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['pipeline', 'Marine', 'FSOD', 'UATD'])
    args = parser.parse_args(); core.old.runtime()
    if args.phase == 'pipeline':
        pipeline()
    else:
        workflow(args.phase)
