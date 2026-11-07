"""Original-weight-frozen spatial bypass control and locally supervised GSR."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from runs import gsr_frozen_context as core
from runs import downstream_vit_finetuning_readout as reader
from sonar_encoder.training.model import build_encoder
from sonar_encoder.training.uatd_head import box_targets

RUN = core.old.BASE/'gsr_frozen_bypass_v1'
PREVIOUS = core.old.BASE/'gsr_frozen_context_resume_v1'
core.RUN = RUN
core.reuse.RUN = RUN
MODES = ('global_adapter', 'gsr_evidence')
AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))


class SpatialBypass(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(384, 32, 1)
        self.spatial = nn.Conv2d(32, 32, 3, padding=1)
        self.condition = nn.Linear(386, 64)
        self.up = nn.Conv2d(32, 384, 1)
        nn.init.zeros_(self.condition.weight); nn.init.zeros_(self.condition.bias)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, tokens, context=None, actions=None):
        size = int((tokens.shape[1]-1)**.5)
        patch = tokens[:, 1:].transpose(1, 2).reshape(-1, 384, size, size)
        patch = F.gelu(self.down(patch))
        cls = F.gelu(self.down(tokens[:, :1].transpose(1, 2)[..., None]))
        if context is not None:
            xy = torch.stack((actions%2*2-1, actions//2*2-1), 1).to(context.dtype)
            scale, bias = self.condition(torch.cat((context.mean((2, 3)), xy), 1)).chunk(2, 1)
            patch = patch*(1+scale[:, :, None, None])+bias[:, :, None, None]
            cls = cls*(1+scale[:, :, None, None])+bias[:, :, None, None]
        patch = self.up(F.gelu(self.spatial(patch))).flatten(2).transpose(1, 2)
        cls = self.up(F.gelu(self.spatial(cls))).flatten(2).transpose(1, 2)
        return .1*torch.cat((cls, patch), 1)


class Model(nn.Module):
    context16 = core.reuse.Model.context16
    local_images = core.reuse.Model.local_images
    corrected = core.reuse.Model.corrected

    def __init__(self, task, mode):
        super().__init__(); self.task, self.mode = task, mode
        cfg = core.reuse.protocol()['encoder']
        self.encoder = build_encoder(Path(cfg['code_dir']), Path(cfg['weights_path']), 'R010').eval().requires_grad_(False)
        self.head = reader.new_head(task, 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            self.neck = core.SpatialNeck()
            self.bypasses = nn.ModuleList([SpatialBypass() for _ in range(4)])
            self.fusion = core.old.Fusion()
        if mode == 'global_adapter':
            self.fusion.requires_grad_(False)
            for bypass in self.bypasses:
                bypass.condition.requires_grad_(False)

    @torch.no_grad()
    def prefix(self, images, global_path):
        tokens = self.encoder.backbone.prepare_tokens_with_masks(images); levels = []
        grid = images.shape[-1]//14
        for i in range(8):
            tokens = self.encoder.backbone.blocks[i](tokens)
            if global_path and i in (2, 5):
                levels.append(self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, grid, grid))
        return tokens, torch.stack(levels, 1) if global_path else None

    def global_from_prefix(self, early_levels, tokens):
        levels = [early_levels[:, 0], early_levels[:, 1]]; grid = early_levels.shape[-1]
        for i in range(8, 12):
            block = self.encoder.backbone.blocks[i]
            delta = self.bypasses[i-8](block.norm1(tokens))
            tokens = block(tokens)+delta
            if i in (8, 11):
                levels.append(self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, grid, grid))
        return self.neck(torch.stack(levels, 1))

    def local_from_prefix(self, tokens, g, actions):
        for i in range(8, 12):
            block = self.encoder.backbone.blocks[i]
            delta = self.bypasses[i-8](block.norm1(tokens), g, actions)
            tokens = block(tokens)+delta
        return self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, 16, 16)

    def global_field(self, images):
        tokens, levels = self.prefix(images, True)
        return self.global_from_prefix(levels, tokens)

    def local_fields(self, images, actions, g):
        tokens, _ = self.prefix(self.local_images(images, actions), False)
        return self.local_from_prefix(tokens, g, actions)


@torch.no_grad()
def extract(model, images):
    grid = 16 if model.task == 'UATD' else 32
    raw = {}; begun = time.monotonic(); count = 0
    for split, values in images.items():
        levels = torch.empty((len(values), 2, 384, grid, grid), device='cuda')
        global_tokens = torch.empty((len(values), grid*grid+1, 384), device='cuda')
        local_tokens = torch.empty((len(values), 4, 257, 384), device='cuda')
        for start in range(0, len(values), 8):
            x = values[start:start+8]; n = len(x)
            t, l = model.prefix(x, True)
            global_tokens[start:start+n] = t; levels[start:start+n] = l
            t, _ = model.prefix(model.local_images(x), False)
            local_tokens[start:start+n] = t.reshape(n, 4, 257, 384); count += n
            if start % 256 == 0:
                print(json.dumps({'extract': model.task, 'split': split, 'rows': start+n}), flush=True)
                core.reuse.resources()
        raw[split] = (levels, global_tokens, local_tokens)
    core.old.write(RUN/model.task/'extraction.json', {'images': count, 'global_first8_views': count,
        'local_first8_views': count*4, 'GPU_RAM_only': True, 'wall_seconds': time.monotonic()-begun,
        'original_weights_frozen': True, 'resources': core.reuse.resources()})
    return raw


def fields(model, raw, ids, actions):
    levels, global_tokens, local_tokens = raw
    g = model.global_from_prefix(levels[ids], global_tokens[ids])
    if model.mode == 'global_adapter':
        return g, g, None
    local = model.local_from_prefix(local_tokens[ids, actions.clamp_min(0)], g, actions.clamp_min(0))
    return g, model.corrected(g, local, actions), local


def crop_targets(task, targets):
    if task == 'Marine':
        return [targets[:, y:y+240, x:x+160] for y, x in ((0, 0), (0, 160), (240, 0), (240, 160))]
    scale = 224. if task == 'FSOD' else 1.
    cropped = []
    for region in range(4):
        origin = torch.tensor([region%2, region//2], dtype=torch.float32)*scale/2
        items = []
        for row in targets:
            boxes = row['boxes'].clone()
            boxes[:, :2] = torch.maximum(boxes[:, :2], origin)
            boxes[:, 2:] = torch.minimum(boxes[:, 2:], origin+scale/2)
            keep = (boxes[:, 2:] > boxes[:, :2]).all(1)
            boxes = (boxes[keep]-origin.repeat(2))*2
            items.append({'boxes': boxes, 'labels': row['labels'][keep]})
        cropped.append(items if task == 'FSOD' else box_targets(items, 'cuda'))
    return cropped


def local_loss(model, local, ids, actions, cropped):
    selected = actions >= 0
    if not bool(selected.any()):
        return local.sum()*0
    idx = ids[selected].cpu().tolist(); regions = actions[selected].cpu().tolist()
    local = local[selected]
    if model.task == 'Marine':
        labels = torch.stack([cropped[j][i] for i, j in zip(idx, regions)])
        logits = F.interpolate(model.head(local), (240, 160), mode='bilinear', align_corners=False)
        return F.cross_entropy(logits, labels.long())
    if model.task == 'FSOD':
        target = [{k: v.cuda() for k, v in cropped[j][i].items()} for i, j in zip(idx, regions)]
        return reader.task_loss('FSOD', model.head, F.interpolate(local, (32, 32), mode='bilinear', align_corners=False), target)
    heat, reg, pos = [torch.stack([cropped[j][k][i] for i, j in zip(idx, regions)]) for k in range(3)]
    output = model.head(local); logits = output[:, :10]; prob = logits.sigmoid()
    ptarget = torch.where(pos, prob, 1-prob); count = pos.sum().clamp_min(1)
    focal = (torch.where(pos, .25, .75)*(1-ptarget).square()*F.binary_cross_entropy_with_logits(logits, heat, reduction='none')).sum()/count
    raw = output[:, 10:].reshape(-1, 10, 4, 16, 16)
    pred = torch.cat((raw[:, :, :2].sigmoid(), raw[:, :, 2:]), 2)
    regression = (F.smooth_l1_loss(pred, reg, reduction='none')*pos[:, :, None]).sum()/(4*count)
    return focal+regression


@torch.no_grad()
def banks(model, raw):
    result = {}
    for split, (levels, tokens, local_tokens) in raw.items():
        n = len(levels); g = torch.empty((n, 384, *levels.shape[-2:]), device='cuda')
        l = torch.empty((n, 4, 384, 16, 16), device='cuda') if model.mode == 'gsr_evidence' else None
        for start in range(0, n, 8):
            end = min(start+8, n); current = model.global_from_prefix(levels[start:end], tokens[start:end])
            g[start:end] = current
            if l is not None:
                for region in range(4):
                    a = torch.full((end-start,), region, dtype=torch.long, device='cuda')
                    l[start:end, region] = model.local_from_prefix(local_tokens[start:end, region], current, a)
            if start % 256 == 0:
                core.reuse.resources()
        result[split] = (g, l)
    return result


@torch.no_grad()
def scheduled_metric(model, spec, raw, target, dense, actions):
    model.eval(); pieces = []
    for start in range(0, len(actions), spec['batch']):
        ids = torch.arange(start, min(start+spec['batch'], len(actions)), device='cuda')
        _, f, _ = fields(model, raw, ids, actions[ids])
        pieces.append(reader.decode(model.task, model.head, f, spec['rows']['validation'][start:start+len(ids)]))
    if model.task == 'FSOD':
        from runs.shared_supervised_json_readout import canonical_evaluate
        from runs.shared_supervised_readout import prediction_list
        return canonical_evaluate(json.loads(Path(spec['GT_paths']['validation']).read_text()), prediction_list(np.concatenate(pieces)))['AP50']
    if model.task == 'UATD':
        from sonar_encoder.training.uatd_head import detection_metrics
        return detection_metrics([x for part in pieces for x in part], target)['map50']
    from sonar_encoder.training.readout import confusion_matrix, metrics_from_confusion
    return metrics_from_confusion(confusion_matrix(dense.cpu(), torch.as_tensor(np.concatenate(pieces)), 12), 'marine')['miou12']


def train(model, spec, raw, targets, dense, cropped):
    folder = RUN/model.task/model.mode; folder.mkdir()
    head_opt = reader.head_optimizer(model.task, model.head)
    extra = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('head.')]
    opt = torch.optim.AdamW(extra, lr=.001, weight_decay=.0001)
    gen = torch.Generator().manual_seed(0)
    actions = torch.from_numpy(np.load(RUN/model.task/'training_actions.npy')).cuda()
    val_actions = torch.from_numpy(np.load(RUN/model.task/'validation_actions.npy')).cuda()
    updates = 0; best = -1.; best_epoch = 0; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, spec['epochs']+1):
            model.train(); model.encoder.eval()
            if model.task == 'FSOD':
                for group in head_opt.param_groups:
                    group['lr'] = .005 if epoch <= 40 else (.0005 if epoch <= 55 else .00005)
            order = torch.randperm(len(spec['rows']['train']), generator=gen); total = 0.; aux_total = 0.
            for ids in order.split(spec['batch']):
                ids = ids.cuda(); a = actions[epoch-1, ids]
                head_opt.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
                g, f, local = fields(model, raw['train'], ids, a)
                target = core.reuse.targets_batch(model.task, targets['train'], dense.get('train'), ids.cpu())
                loss = reader.task_loss(model.task, model.head, g, target)
                auxiliary = loss.detach()*0
                if model.mode == 'gsr_evidence':
                    auxiliary = local_loss(model, local, ids, a, cropped)
                    loss = .5*loss+.5*reader.task_loss(model.task, model.head, f, target)+.25*auxiliary
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite normal bypass training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(extra, 1., error_if_nonfinite=True)
                head_opt.step(); opt.step(); updates += 1
                total += float(loss.detach())*len(ids); aux_total += float(auxiliary.detach())*len(ids)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {
                        'original_body_trainable': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                        'original_body_has_grad': any(p.grad is not None for p in model.encoder.parameters()),
                        'optimizer_body_parameters': len({id(p) for p in model.encoder.parameters()} & {id(p) for group in opt.param_groups for p in group['params']}),
                        'adapter_grad_norms': [float(sum(p.grad.square().sum() for p in b.parameters() if p.grad is not None).sqrt()) for b in model.bypasses],
                        'new_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'first_loss': float(loss.detach()), 'first_aux_loss': float(auxiliary.detach())})
                if updates % 100 == 0:
                    core.reuse.resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order), 'aux_loss': aux_total/len(order), 'wall_seconds': time.monotonic()-begun}
            if epoch % 5 == 0:
                score = scheduled_metric(model, spec, raw['validation'], targets['validation'], dense.get('validation'), val_actions)
                entry['scheduled_global_or_random_route_metric'] = score
                if score > best:
                    best, best_epoch = score, epoch
                print(json.dumps({'task': model.task, 'mode': model.mode, **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    model.eval().requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    core.old.write(folder/'train_report.json', {'updates': updates, 'epochs': spec['epochs'],
        'scheduled_terminal': score, 'scheduled_best': best, 'scheduled_best_epoch': best_epoch,
        'best_scope': 'Global baseline or random-route auxiliary; not best learned selector.',
        'original_weights_trainable': 0, 'no_new_weight_files': True, 'wall_seconds': time.monotonic()-begun,
        'resources': core.reuse.resources()})
    return updates


def workflow(task):
    spec, images, targets, dense = core.reuse.data_for(task)
    first = Model(task, MODES[0]).cuda(); raw = extract(first, images); del images
    cropped = crop_targets(task, dense['train'] if task == 'Marine' else targets['train']); results = []
    for mode in MODES:
        model = first if mode == MODES[0] else Model(task, mode).cuda()
        updates = train(model, spec, raw, targets, dense, cropped); bank = banks(model, raw)
        if mode == 'gsr_evidence':
            gains = core.reuse.gain_labels(model, spec, bank['train'], targets['train'], dense.get('train'))
            selector, sr = core.reuse.train_selector(task, mode, model.context16(bank['train'][0]), gains)
            result = core.evaluate(model, spec, bank, targets, selector, sr, updates)
            core.online_cost(model, spec, selector, sr)
            del selector
        else:
            a = np.full(len(spec['rows']['validation']), -1, np.int64)
            pred = core.reuse.predict(model, spec, bank['validation'], a)
            score = core.reuse.score_prediction(task, pred, targets['validation'], spec, RUN/task/mode/'global_only_prediction')
            result = {'task': task, 'mode': mode, 'terminal_updates': updates, 'metrics': {'global_only': score},
                'original_weights_trainable': 0, 'new_weights_saved': False, 'references': core.reuse.protocol()['references'][task]}
            core.old.write(RUN/task/mode/'results.json', result)
            core.reuse.online_cost(model, mode, spec, None, None)
        results.append(result); del model, bank
        if mode == MODES[0]:
            del first
    core.old.write(RUN/task/'results.json', {'status': 'COMPLETE_MATCHED_IMAGE_ORDER_BASELINE_AND_GSR', 'modes': results})


def pipeline():
    previous = json.loads((PREVIOUS/'queue_state.json').read_text())
    completed = {row['task']: row['actual_wait_exit'] for row in previous['actual_waits']}
    if completed['Marine'] != 0 or completed['UATD'] != 0:
        raise RuntimeError('Marine/UATD predecessors must actually complete before independent optimization')
    RUN.mkdir(); (RUN/'logs').mkdir(); start = time.time()
    previous_protocol = json.loads((PREVIOUS/'protocol.json').read_text())
    p = {key: previous_protocol[key] for key in
         ('encoder', 'tasks', 'references', 'SWiSE_references', 'SWiSE_sources', 'caps')}
    p.update(start_unix=start, deadline_unix=start+28800, source_utf8=Path(__file__).read_text(), modes=list(MODES))
    p['previous_completed_independent_tasks_and_active_FSOD'] = previous
    p['staged_release'] = 'Marine/UATD predecessors actualexit0 allow next Marine thenUATD on freedGPU3 while originalFSOD continues unchanged onGPU2. NewFSOD only after original whole-scope actualforegroundwait0 and closure. Old failure aborts new queue.'
    p['user_freeze_clarification'] = 'Original weights frozen; new trainable modules between layers explicitly allowed by user.'
    p['global_input'] = 'Existing FSOD/Marine448, UATD224; UATD restored to original ordinaryFT224 interface, not preceding448pool16. New unit declared before forwards.'
    p['architecture'] = 'Four32dim zero-up spatial Convpass-style bypasses after frozen blocks9..12; scale.1. Same weights sharedglobal/local. Local FiLM conditioned on globalmean384+imagexy. Global neck uses normalized3/6/9/12.'
    p['control_scope'] = 'Ordinary global-only spatial bypass+neck+head versus complete GSR with local FiLM/fusion/auxloss; extra parameters and compute differ, not an isolated module ablation.'
    p['loss'] = 'Global control original taskloss. GSR .5global+.5corrected+.25local auxiliary. Main original GT/loss/eval preserved.'
    p['auxiliary_GT'] = 'Train-only quadrant transform of labels: Marine native240x160 mask crop; detection intersect+clip to quadrant then transform to local coordinates. Original GT files unchanged.'
    p['UATD_auxiliary_empty_rule'] = 'Aux crop can be empty: summed focal / max(totalpositivecells,1), summed positiveSmoothL1 / (4*max(totalpositivecells,1)); retains empty-crop negatives. Original full-image dense_loss remains unchanged.'
    p['learning_rates'] = 'Original head optimizer/schedule; new bypasses+neck+fusion AdamW.001 wd.0001 clip1; no original DINO parameters in optimizer.'
    p['storage'] = 'GPU-RAM-only fixed first8 prefixes; active late4 originalweights fixed with gradient flow to newmodules. No weights/features saved. Nativepredictions/gainlabels/logs only.'
    p['selector'] = 'Same fixed spatial selector and balanced signed train-only gain labels. Fixed inner80/20 split; no devGT used to choose actions.'
    p['best_endpoint_scope'] = 'Complete learned terminal primary; scheduled global/random-route best auxiliary, not a complete-method best.'
    p['prior_art'] = 'Convpass https://arxiv.org/html/2207.07039 motivates frozen-weight convolutional bypasses; late4/global-local conditioning/localGT auxiliary is this implementation, not a literal original reproduction or accepted novelty.'
    p['GPU_schedule'] = 'Marine thenUATD onGPU3. NewFSOD onGPU2 only after previous actualforegroundwait0. Across both supervisors at most one ownedmodelworker/card. NoGPU0/1.'
    p['CPU_affinity'] = {'2': AVAILABLE_CPUS[:4], '3': AVAILABLE_CPUS[4:8]}
    p['stop'] = 'Real error/nonfinite/resource/deadline stops ownedworkers. No quality-driven stopping, newseed/HPO, automatic recovery or precision/batch change.'
    p['no_tests_SHA_reviews'] = True
    p['acceptance'] = 'Exceed SWiSE and each terminal metric at most .01 below strongest completed compatible local reference mean. Single seed development, not certified global SOTA.'
    p['modes'] = list(MODES); p['total_new_task_optimizer_updates'] = 86000
    for task in p['tasks']:
        folder = RUN/task; folder.mkdir()
        for name in ('training_actions.npy', 'validation_actions.npy', 'selector_split.npz'):
            shutil.copyfile(PREVIOUS/task/name, folder/name)
    core.old.write(RUN/'protocol.json', p)
    pending = ['UATD']; active = {}; waits = []; fsod_started = False
    def launch(task, gpu):
        admission = core.old.admission(gpu)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'XFORMERS_DISABLED': '1'}
        log = (RUN/'logs'/f'{task}.log').open('w')
        cpus = ','.join(map(str, p['CPU_affinity'][str(gpu)]))
        child = subprocess.Popen(['/usr/bin/taskset', '-c', cpus, core.old.PYTHON, '-B', '-u', '-m', 'runs.gsr_frozen_bypass', task], cwd=core.ROOT,
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[task] = (child, log, gpu, time.time(), admission)
    try:
        launch('Marine', 3)
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'workers': {k: v[0].pid for k, v in active.items()}, 'no_new_weights_saved': True})
        while active or not fsod_started:
            if (PREVIOUS/'failure_closure.json').exists():
                raise RuntimeError('Previous fixed FSOD failed; no successor continuation')
            previous_wait = PREVIOUS/'actual_foreground_wait.json'
            if not fsod_started and previous_wait.exists():
                actual_wait = json.loads(previous_wait.read_text())
                if actual_wait['actual_foreground_exit'] != 0:
                    raise RuntimeError('Previous fixed FSOD actual foreground exit is nonzero')
                json.loads((PREVIOUS/'wave_closure.json').read_text())
                launch('FSOD', 2); fsod_started = True
            if time.time() > p['deadline_unix']:
                raise TimeoutError('Original new-wave8h deadline including staged waiting')
            for task, (child, log, gpu, begun, admission) in list(active.items()):
                if time.time()-begun > 14400 or time.time() > p['deadline_unix']:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del active[task]
                waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code, 'wall_seconds': time.time()-begun, 'admission': admission})
                if code != 0:
                    raise RuntimeError(task+' actual worker exit '+str(code))
                if gpu == 3 and pending:
                    launch(pending.pop(0), 3)
                core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in active.items()}, 'actual_waits': waits})
            time.sleep(1)
        result = {task: json.loads((RUN/task/'results.json').read_text()) for task in p['tasks']}
        core.old.write(RUN/'results.json', {'status': 'COMPLETE_FIXED_BYPASS_PILOT', 'tasks': result})
        size = sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file())
        if size > p['caps']['new_disk_bytes']:
            raise RuntimeError('Results-only disk cap exceeded')
        close = {'actual_waits': waits, 'wall_seconds': time.time()-start, 'new_entity_bytes': size,
                 'status': 'COMPLETE_FIXED_SCOPE_GOAL_REQUIRES_RESULT_DECISION', 'original_weights_frozen': True, 'no_new_weights_saved': True}
        core.old.write(RUN/'wave_closure.json', close); core.old.write(RUN/'writing_handoff.json', close)
        core.old.write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        failure = traceback.format_exc()
        for task, (child, log, gpu, begun, admission) in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            code = child.wait(); log.close(); waits.append({'task': task, 'PID': child.pid, 'actual_wait_exit': code, 'cancelled_after_error': True})
        core.old.write(RUN/'failure_closure.json', {'traceback': failure, 'actual_waits': waits, 'status': 'STOPPED_NO_AUTOMATIC_RETRY'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['pipeline', 'Marine', 'FSOD', 'UATD'])
    args = parser.parse_args(); core.old.runtime()
    if args.phase == 'pipeline':
        pipeline()
    else:
        workflow(args.phase)
