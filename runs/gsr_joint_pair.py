"""Joint-gain pair routing with one shared source encoder and ordinary task-loss FT."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import pickle
import random
import select
import signal
import socket
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from runs import gsr_multidata_training as core
from runs import gsr_multidata_runtime_v2 as parent
from runs.gsr_online_gain_routing import gpu_admission

RUN = core.BASE/'gsr_joint_pair_v1'
SOCKET = RUN/'source_state.sock'
SOURCE = core.SOURCE
TARGETS = ('FSOD', 'Marine', 'UATD', 'NKSID', 'HighResolution', 'Diver')
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
CAPS = {'FSOD': 9000, 'Marine': 3600, 'UATD': 10800, 'NKSID': 2400, 'HighResolution': 2400, 'Diver': 2400}
POLICIES = ('global', 'random', 'fixed0', 'fixed1', 'fixed2', 'fixed3', 'fixed4', 'fixed5', 'uncertainty')
DEADLINE = 0.
write = core.write


def guard(deadline):
    observed = {'RSS_bytes': core.resource.getrusage(core.resource.RUSAGE_SELF).ru_maxrss*1024,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > min(deadline, DEADLINE):
        raise TimeoutError('Joint-pair fixed phase deadline')
    if observed['RSS_bytes'] > 48*1024**3 or observed['GPU_reserved_bytes'] > 32*1024**3:
        raise RuntimeError('Joint-pair resource cap '+str(observed))
    return observed


class PairSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1156, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer('pairs', torch.tensor(PAIRS, dtype=torch.long))
        self.register_buffer('position', torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]]))

    def forward(self, g):
        patch = F.adaptive_avg_pool2d(g, (2, 2)).flatten(2).transpose(1, 2)
        pair = patch[:, self.pairs].flatten(2)
        whole = g.mean((2, 3))[:, None].expand(-1, 6, -1)
        position = self.position[self.pairs].reshape(6, 4)[None].expand(len(g), -1, -1)
        return self.mlp(torch.cat((pair, whole, position), -1)).squeeze(-1)


class PairGSR(core.GSR):
    def __init__(self, config):
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(102)
            self.selector = PairSelector()

    def correct_pair(self, g, local, regions):
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        crops = torch.stack([base[..., r:r+16, c:c+16] for r, c in core.original.QUADS], 1)
        index = torch.arange(len(g), device=g.device)
        field = base
        for side in range(2):
            region = regions[:, side]
            delta = self.fusion(local[:, side], crops[index, region], g, region)
            for j, (r, c) in enumerate(core.original.QUADS):
                field = field+F.pad(delta*(region == j)[:, None, None, None], (c, 16-c, r, 16-r))
        return field

    def pair_features(self, images, g, actions):
        regions = self.selector.pairs[actions]
        large = F.interpolate(images, (448, 448), mode='bilinear', align_corners=False, antialias=True)
        tiles = torch.stack([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in core.original.QUADS], 1)
        selected = tiles[torch.arange(len(images), device=images.device)[:, None], regions]
        local = self.global_field(selected.flatten(0, 1)).reshape(len(images), 2, 384, 16, 16)
        return self.correct_pair(g, local, regions)

    def field(self, images, head, kind, policy='learned', actions=None):
        g = self.global_field(images)
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        if policy == 'global':
            return base
        scores = self.selector(g.detach())
        if actions is None:
            if policy == 'uncertainty':
                if kind == 'detection':
                    probability = head.rpn.head([base])[0][0].sigmoid().clamp(1e-6, 1-1e-6)
                    entropy = -(probability*probability.log()+(1-probability)*(1-probability).log()).mean(1, keepdim=True)
                elif kind == 'classification':
                    patches = F.adaptive_avg_pool2d(base, (2, 2)).flatten(2).transpose(1, 2)
                    probability = head(patches).softmax(-1).clamp_min(1e-6)
                    regional = -(probability*probability.log()).sum(-1)
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
                if kind != 'classification':
                    regional = F.adaptive_avg_pool2d(entropy, (2, 2)).flatten(1)
                regions = regional.topk(2, dim=1).indices.sort(1).values
                actions = (regions[:, None] == self.selector.pairs[None]).all(2).long().argmax(1)
            else:
                actions = scores.argmax(1)
        self.last_actions = actions.detach()
        active = torch.where(actions >= 0)[0]
        if not len(active):
            return base
        field = base.clone()
        field[active] = self.pair_features(images[active], g[active], actions[active])
        if self.training:
            probability = scores.softmax(1)[active].gather(1, actions[active, None]).flatten()
            # Zero in forward; a biased selected-residual surrogate for the hard router in backward.
            correction = (probability-probability.detach())[:, None, None, None]*(field[active]-base[active]).detach()
            output = field.clone(); output[active] = field[active]+correction
            return output
        return field


@torch.no_grad()
def source_vectors(model, images):
    g = model.global_field(images)
    actions = model.selector(g).argmax(1)
    field = model.pair_features(images, g, actions)
    base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
    return base.mean((2, 3)).cpu().numpy(), field.mean((2, 3)).cpu().numpy()


@torch.no_grad()
def evaluate(task, spec, model, head, images, labels, batch, deadline, policy, split='validation'):
    model.eval(); head.eval(); rows = spec['rows'][split]; pieces = []
    random_actions = np.random.RandomState(0).randint(0, 6, len(rows))
    for start in range(0, len(rows), batch):
        end = min(start+batch, len(rows)); actions = None
        if policy == 'random':
            actions = torch.as_tensor(random_actions[start:end], device='cuda')
        elif policy.startswith('fixed'):
            actions = torch.full((end-start,), int(policy[-1]), device='cuda', dtype=torch.long)
        field = model.field(images[start:end], head, 'uatd' if task == 'UATD' else spec['kind'], policy, actions)
        if spec['kind'] == 'detection':
            pieces.append(core.reader.decode('FSOD', head, field, rows[start:end]))
        elif task == 'UATD':
            pieces.extend(core.dense.decode32(head(field)))
        elif spec['kind'] == 'classification':
            pieces.append(head(field.mean((2, 3))).cpu().numpy())
        elif spec['kind'] == 'point_regression':
            pieces.append(core.point_prediction(head, field).cpu().numpy())
        else:
            logits = head(field)
            for i, target in enumerate(labels[start:end]):
                pieces.append(F.interpolate(logits[i:i+1], tuple(target.shape), mode='bilinear', align_corners=False).argmax(1)[0].byte().cpu().numpy())
        guard(deadline)
    prediction = pieces if task == 'UATD' or spec['kind'] == 'segmentation' else np.concatenate(pieces)
    result, independent = core.scores(task, spec, prediction, labels, split)
    return result, independent, prediction


def source_training(model, protocol, tasks):
    model.encoder.eval().requires_grad_(False); model.selector.requires_grad_(False)
    images = {t: torch.from_numpy(np.load(tasks[t]['views']['train'], mmap_mode='r', allow_pickle=False)).cuda() for t in SOURCE}
    labels = {t: core.targets_for(t, tasks[t], 'train') for t in SOURCE}
    heads = nn.ModuleDict({t: core.new_head(t, tasks[t]).cuda() for t in SOURCE})
    optimizers = {t: core.reader.head_optimizer('FSOD' if tasks[t]['kind'] == 'detection' else 'Marine', heads[t]) for t in SOURCE}
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=.001, weight_decay=.0001)
    plans = np.load(protocol['source_plan'], allow_pickle=False)
    pair_actions = np.load(RUN/'pair_actions.npz', allow_pickle=False)
    folder = RUN/'source'; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for update in range(6000):
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = core.previous.shared.native.module_lr(update, 6000, .001)
            values = {}
            for task in SOURCE:
                head = heads[task]; head.train(); opt = optimizers[task]; opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = (.005 if update < 4000 else .0005 if update < 5500 else .00005) if tasks[task]['kind'] == 'detection' else core.previous.shared.native.module_lr(update, 6000, .001)
                ids = plans[task][update]; actions = pair_actions[task][update]; value_sum = 0.
                for start in range(0, len(ids), 4):
                    indices = ids[start:start+4].tolist()
                    action = torch.as_tensor(actions[start:start+4], device='cuda', dtype=torch.long)
                    field = model.field(images[task][indices], head, tasks[task]['kind'], actions=action)
                    target = core.batch_targets(task, tasks[task], labels[task], indices)
                    value = core.task_loss(task, tasks[task], head, field, target)*len(indices)/len(ids)/len(SOURCE)
                    value.backward(); value_sum += float(value.detach())
                values[task] = value_sum; opt.step()
            norm = torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); guard(DEADLINE)
            if update == 0:
                write(folder/'first_counted_update.json', {'updates': 1, 'source_tasks': list(SOURCE),
                    'encoder_trainable_parameters': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                    'fusion_gradient_norm': float(norm), 'local_budget': 2, 'field': list(field.shape),
                    'pair_ids_last_micro': action.tolist(), 'encoder_images': model.encoder_images,
                    'encoder_calls': model.encoder_calls, 'selector_parameters': sum(p.numel() for p in model.selector.parameters())})
            if (update+1) % 100 == 0:
                row = {'updates': update+1, 'losses': values, 'encoder_images': model.encoder_images,
                    'wall_seconds': time.monotonic()-begun, **guard(DEADLINE)}
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
    write(folder/'source_report.json', {'status': 'COMPLETE_JOINT_PAIR_SOURCE6000',
        'visits_per_task': protocol['source_visits'], 'encoder_images': model.encoder_images,
        'encoder_calls': model.encoder_calls, 'original_DINO_updates': 0,
        'wall_seconds': time.monotonic()-begun, **guard(DEADLINE)})
    model.eval(); model.fusion.requires_grad_(False)
    banks, gain_targets, additive_targets, fit_ids, held_ids = {}, {}, {}, {}, {}
    for task in SOURCE:
        subset = plans[task+'_gain_subset']; fields, gains, singles = [], [], []
        heads[task].train(tasks[task]['kind'] == 'detection')
        with torch.no_grad():
            for number, i in enumerate(subset):
                x = images[task][int(i):int(i)+1]; g = model.global_field(x)
                large = F.interpolate(x, (448, 448), mode='bilinear', align_corners=False, antialias=True)
                crops = torch.cat([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in core.original.QUADS])
                local = model.global_field(crops)
                target = core.batch_targets(task, tasks[task], labels[task], [int(i)])
                losses = []
                for case in range(11):
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(10000+int(i))
                        if case == 0:
                            field = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
                        elif case <= 4:
                            action = torch.tensor([case-1], device='cuda')
                            field = model.corrected(g, local[case-1:case], action)
                        else:
                            regions = model.selector.pairs[case-5]
                            field = model.correct_pair(g, local[regions][None], regions[None])
                        losses.append(float(core.task_loss(task, tasks[task], heads[task], field, target)))
                fields.append(g.cpu())
                singles.append([losses[0]-v for v in losses[1:5]])
                gains.append([losses[0]-v for v in losses[5:]])
                if number % 100 == 0: guard(DEADLINE)
            raw = torch.tensor(gains, device='cuda'); single = torch.tensor(singles, device='cuda')
            np.save(folder/(task+'_joint_gains.npy'), raw.cpu().numpy())
            np.save(folder/(task+'_single_gains.npy'), single.cpu().numpy())
            banks[task] = torch.cat(fields).cuda()
            order = plans[task+'_selector_order']; n = int(.8*len(order))
            fit_ids[task], held_ids[task] = torch.from_numpy(order[:n]).cuda(), torch.from_numpy(order[n:]).cuda()
            centered = raw-raw.mean(1, keepdim=True)
            gain_targets[task] = centered/centered[fit_ids[task]].std().clamp_min(1e-6)
            additive = single[:, model.selector.pairs].sum(2)
            centered = additive-additive.mean(1, keepdim=True)
            additive_targets[task] = centered/centered[fit_ids[task]].std().clamp_min(1e-6)
        del images[task], labels[task]
    del heads, optimizers, optimizer
    model.selector.requires_grad_(True)
    control = copy.deepcopy(model.selector)
    optimizer = torch.optim.AdamW(list(model.selector.parameters())+list(control.parameters()), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(31)
    steps = math.ceil(max(len(ids) for ids in fit_ids.values())/64)
    for epoch in range(100):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for task in SOURCE:
                ids = fit_ids[task][torch.randint(len(fit_ids[task]), (64,), generator=generator).cuda()]
                joint_guess = model.selector(banks[task][ids]); additive_guess = control(banks[task][ids])
                joint_loss = F.mse_loss(joint_guess-joint_guess.mean(1, keepdim=True), gain_targets[task][ids])
                additive_loss = F.mse_loss(additive_guess-additive_guess.mean(1, keepdim=True), additive_targets[task][ids])
                ((joint_loss+additive_loss)/len(SOURCE)).backward()
            optimizer.step()
        guard(DEADLINE)
    held = {}
    with torch.no_grad():
        for task in SOURCE:
            raw = torch.from_numpy(np.load(folder/(task+'_joint_gains.npy'))).cuda()
            single = torch.from_numpy(np.load(folder/(task+'_single_gains.npy'))).cuda()
            ids = held_ids[task]
            choice = model.selector(banks[task][ids]).argmax(1)
            additive_choice = control(banks[task][ids]).argmax(1)
            additive_oracle = single[ids][:, model.selector.pairs].sum(2).argmax(1)
            held[task] = {'fit': len(fit_ids[task]), 'held': len(ids),
                'joint_model_selected_gain': float(raw[ids, choice].mean()),
                'additive_model_selected_joint_gain': float(raw[ids, additive_choice].mean()),
                'additive_oracle_selected_joint_gain': float(raw[ids, additive_oracle].mean()),
                'uniform_pair_gain': float(raw[ids].mean()), 'joint_oracle_gain': float(raw[ids].max(1).values.mean())}
    write(folder/'selector_report.json', {'status': 'COMPLETE_SOURCE_JOINT_VS_ADDITIVE_GAIN',
        'epochs_each': 100, 'optimizer_updates_each': 100*steps, 'same_capacity_same_initialization_control': True,
        'held_source_diagnostics': held, 'counterfactual_head_examples': 11*sum(len(v) for v in banks.values()),
        'control_scope': 'Source-held training-data comparison only. Additive control is not exported, no downstream additive superiority claim. Oracle fields use source held GT only and are not inference policies.',
        'source_encoder_images': model.encoder_images, 'source_wall_seconds': time.monotonic()-begun, **guard(DEADLINE)})
    model.fusion.requires_grad_(True)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(task, protocol, initial):
    spec = json.loads(Path(protocol['data_manifest']).read_text())['tasks'][task]
    schedule = protocol['target_schedules'][task]; folder = RUN/task
    model = PairGSR(protocol['encoder']); model.load_state_dict(initial, strict=True)
    core.reader.configure_encoder(model.encoder, 'e2e')
    model.fusion.requires_grad_(True); model.selector.requires_grad_(True); model.cuda()
    head = core.new_head(task, spec).cuda()
    body_opt = core.reader.body_optimizer(model.encoder)
    head_opt = core.reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    fusion_opt = torch.optim.AdamW(model.fusion.parameters(), lr=.0001, weight_decay=.0001)
    route_opt = torch.optim.AdamW(model.selector.parameters(), lr=.0001, weight_decay=.0001)
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r', allow_pickle=False)).cuda() for s in ('train', 'validation')}
    labels = {s: core.targets_for(task, spec, s) for s in images}
    generator = torch.Generator().manual_seed(0); routing = np.random.RandomState(700)
    key = core.metric_key(task, spec); direction = -1 if spec['kind'] == 'point_regression' else 1
    best = -float('inf'); updates = 0; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs']+1):
            model.train(); head.train(); order = torch.randperm(len(spec['rows']['train']), generator=generator)
            if spec['kind'] == 'detection':
                head_opt.param_groups[0]['lr'] = .005*.1**sum(epoch > m for m in (40, 55))
            total = 0.; counts = np.zeros(6, dtype=np.int64)
            for ids in order.split(schedule['batch']):
                guard(DEADLINE)
                for optimizer in (body_opt, head_opt, fusion_opt, route_opt): optimizer.zero_grad(set_to_none=True)
                for group in body_opt.param_groups:
                    group['lr'] = core.reader.body_learning_rate(updates, schedule['updates'])
                for optimizer in (fusion_opt, route_opt):
                    optimizer.param_groups[0]['lr'] = core.previous.shared.native.module_lr(updates, schedule['updates'], .0001)
                actions = torch.as_tensor(routing.randint(0, 6, len(ids)), device='cuda') if routing.rand() < .5 else None
                field = model.field(images['train'][ids.cuda()], head, 'uatd' if task == 'UATD' else spec['kind'], actions=actions)
                target = core.batch_targets(task, spec, labels['train'], ids.tolist())
                loss = core.task_loss(task, spec, head, field, target); loss.backward()
                body_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
                route_norm = torch.nn.utils.clip_grad_norm_(model.selector.parameters(), 1., error_if_nonfinite=True)
                for optimizer in (body_opt, head_opt, fusion_opt, route_opt): optimizer.step()
                updates += 1; total += float(loss.detach())*len(ids)
                counts += torch.bincount(model.last_actions, minlength=6).cpu().numpy()
                if updates == 1:
                    write(folder/'first_counted_update.json', {'updates': 1, 'first_ids': ids.tolist(),
                        'selected_pair_ids': model.last_actions.tolist(), 'field': list(field.shape),
                        'body_gradient_norm': float(body_norm), 'selector_gradient_norm': float(route_norm),
                        'encoder_images': model.encoder_images, 'source_cohort': protocol['source_cohort'],
                        'ordinary_task_loss_only': True, 'backward_scope': 'Biased straight-through selected-residual surrogate; no policy loss or reward-head pass.'})
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order),
                'selected_pair_counts': counts.tolist(), 'wall_seconds': time.monotonic()-begun}
            every = protocol['evaluation_every_epochs'][task]
            if epoch % every == 0:
                metric, independent, pred = evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], DEADLINE, 'learned')
                entry[key] = metric[key]
                payload = {'prediction': pred, 'task': task, 'epoch': epoch, 'policy': 'learned', 'local_budget': 2}
                if direction*metric[key] > best:
                    best, best_epoch, best_metric = direction*metric[key], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = metric, independent
                    torch.save(payload, folder/'terminal_predictions.pt')
            log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)
    policies = {'learned': terminal}
    for policy in POLICIES:
        metric, independent, pred = evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], DEADLINE, policy)
        policies[policy] = metric
        torch.save({'prediction': pred, 'task': task, 'epoch': schedule['epochs'], 'policy': policy, 'local_budget': 2}, folder/(policy+'_terminal_predictions.pt'))
        write(folder/(policy+'_independent.json'), independent)
    held = None
    held_split = 'official_validation' if task == 'HighResolution' else 'test' if task == 'Diver' else None
    if held_split is not None:
        held_images = []
        for row in spec['rows'][held_split]:
            with Image.open(row['image']) as image:
                intensity = np.log1p(np.array(image.convert('L'), dtype=np.uint8).astype(np.float32)/255.)
            x = F.interpolate(torch.from_numpy(intensity)[None, None], (224, 224), mode='bilinear', align_corners=False, antialias=True)
            held_images.append(((x[0]-spec['normalization']['mean'])/spec['normalization']['std']).cuda())
        held_labels = core.targets_for(task, spec, held_split)
        held, independent, pred = evaluate(task, spec, model, head, torch.stack(held_images), held_labels, schedule['batch'], DEADLINE, 'learned', held_split)
        torch.save({'prediction': pred, 'epoch': schedule['epochs'], 'task': task, 'split': held_split}, folder/'heldout_terminal_predictions.pt')
        write(folder/'heldout_independent.json', independent)
    write(folder/'results.json', {'status': 'COMPLETE_FIXED_JOINT_PAIR_TARGET', 'task': task, 'metric': key,
        'higher_is_better': direction == 1, 'epochs': schedule['epochs'], 'updates': updates,
        'terminal': terminal, 'terminal_independent': terminal_independent, 'best': best_metric, 'best_epoch': best_epoch,
        'terminal_policy_metrics': policies, 'heldout_terminal_only': held, 'heldout_split': held_split,
        'source_cohort': protocol['source_cohort'], 'local_budget': 2, 'pairs': PAIRS,
        'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'new_weight_files': 0, 'wall_seconds': time.monotonic()-begun, **guard(DEADLINE)})


def get_source():
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(120); connection.connect(str(SOCKET)); connection.sendall(b'G')
        n = int.from_bytes(parent.read_bytes(connection, 8), 'big')
        return pickle.loads(parent.read_bytes(connection, n))


def worker(job_id):
    global DEADLINE
    protocol = json.loads((RUN/'protocol.json').read_text()); job = json.loads((RUN/job_id/'job.json').read_text())
    DEADLINE = job['source_deadline_unix'] if job_id == 'source' else job['deadline_unix']
    core.guard = guard; core.original.runtime(); random.seed(0); np.random.seed(0); torch.manual_seed(0)
    try:
        if job_id == 'source':
            tasks = json.loads(Path(protocol['data_manifest']).read_text())['tasks']
            model = PairGSR(protocol['encoder']).cuda(); state = source_training(model, protocol, tasks)
            blob = pickle.dumps(state, protocol=5)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(120); connection.connect(str(SOCKET))
                connection.sendall(b'P'+len(blob).to_bytes(8, 'big')+blob)
                if parent.read_bytes(connection, 1) != b'K': raise RuntimeError('Source RAM acknowledgement missing')
            write(RUN/'source/snapshot_published.json', {'source_cohort': protocol['source_cohort'], 'serialized_bytes': len(blob), 'weight_files': 0})
            DEADLINE = min(protocol['deadline_unix'], time.time()+3600)
            core.RUN = RUN/'source'; core.source_vectors = source_vectors
            core.diagnostics(model, protocol)
        else:
            fit(job_id, protocol, get_source())
    except BaseException:
        write(RUN/job_id/'failure.json', {'traceback': traceback.format_exc(), 'time_unix': time.time(),
            'no_automatic_retry': True, 'new_weight_files': 0})
        raise


def pipeline():
    started = time.time()
    old = json.loads((parent.RUN/'protocol.json').read_text())
    state = json.loads((parent.RUN/'queue_state.json').read_text())
    if state['pending'] or state['failed'] or set(state['active'])-{'original_DINO_CFC22', 'shared_GSR_CFC22'}:
        raise RuntimeError('Only established CFC jobs may coexist with the new source')
    online = json.loads((core.BASE/'gsr_online_gain_routing_v1/completion.json').read_text())
    if online['status'] != 'COMPLETE_FIXED_SCOPE':
        raise RuntimeError('Previous fixed candidate has not closed')
    parent_jobs = state['active']; parent_pids = {v['PID'] for v in parent_jobs.values()}
    admission = gpu_admission(3, parent_pids)
    if not admission['eligible']:
        raise RuntimeError(('Source GPU3 admission unavailable; no allocation', admission))
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if available < 192*1024**3:
        raise RuntimeError('Host memory admission below192GiB')
    deadline = min(started+36000, old['deadline_unix'])
    if deadline-started < 14400:
        raise RuntimeError('Insufficient remaining inherited time')
    references = json.loads((parent.RUN/'comparison.json').read_text())['tasks']
    online_references = json.loads((core.BASE/'gsr_online_gain_routing_v1/comparison.json').read_text())['tasks']
    if any(references[t]['arms'][a]['status'] != 'COMPLETE' for t in TARGETS for a in ('shared_GSR', 'original_DINO')):
        raise RuntimeError('Six matched reference endpoints must be complete')
    RUN.mkdir(); (RUN/'source').mkdir()
    source_plan = parent.RUN/'source_plan.npz'
    plans = np.load(source_plan, allow_pickle=False); generator = np.random.RandomState(1700)
    actions = {}
    for task in SOURCE:
        old_actions = plans[task+'_actions']
        values = generator.randint(0, 6, old_actions.shape).astype(np.int8)
        values[old_actions < 0] = -1
        actions[task] = values
    np.savez(RUN/'pair_actions.npz', **actions)
    protocol = {'status': 'FROZEN_BEFORE_MODEL_FORWARD', 'started_unix': started, 'deadline_unix': deadline,
        'entry': str(Path(__file__).resolve()), 'source_cohort': 'gsr_joint_pair_v1/fresh_pair_source',
        'encoder': old['encoder'], 'data_manifest': old['data_manifest'], 'source_plan': str(source_plan),
        'source_macro_per_task': old['source_macro_per_task'], 'source_visits': old['source_visits'],
        'source_updates': 6000, 'source_tasks': list(SOURCE), 'pairs': PAIRS, 'local_budget': 2,
        'source_image_order': 'Reuse all original source sample IDs, gain subsets, selector fit/held order and exact no-local mask. Uniform pair IDs drawn once with NumPy1700 for nonempty actions.',
        'planned_source_training_encoder_images': int(sum(v.size+2*np.count_nonzero(v >= 0) for v in actions.values())),
        'source_recipe': 'Original DINO frozen/eval. Same DifferenceFusion, six independent source heads, macro/micro weights, optimizers and6000 schedules as parent. Two local regions add compute; no native-pixel information is added.',
        'gain_supervision': 'After source fusion/head freeze, per selected source image encode global+all4tiles once; compute11 head losses: global,4singleton,6actualjoint-pair. Same RNG start for each counterfactual. Preserve signed gains. Fit joint-gain PairSelector and identical-initialization/same-capacity additive-gain control100epochs with matching fit draws; only main selector exported.',
        'gain_normalization': 'Per-image centering and per-source fit-only std clamped1e-6 as parent; outputs are relative gain-ranking scores, not calibrated absolute gains.',
        'pair_selector': {'input': 'two canonical ordered region summaries+global mean+four position coordinates',
            'input_dim': 1156, 'hidden': 64, 'output_per_pair': 1, 'expected_parameters': 74113,
            'source_optimizer': 'AdamW.001/.0001,100epochs; originalCPU31fit sampling'},
        'downstream_gradient': 'Fout=F+(p_selected-stopgrad(p_selected))*stopgrad(F-B). Forward hard pair selection retained; selector receives a biased selected-residual straight-through surrogate from ordinary task loss. Selector input G detached. Not an exact gradient of argmax or an unbiased policy gradient.',
        'downstream_recipe': 'One complete shared initialization; original encoder fullFT with unusedmask excluded, fusion and selector AdamW1e-4/wd1e-4/same module LR schedule, each gradclip1. Original task heads/losses/SGD or AdamW/batch/imageorder/epochs unchanged. Half uniform pair/half learned training routing, NumPy700. No policy loss or reward-head pass.',
        'target_schedules': {t: old['target_schedules'][t] for t in TARGETS},
        'evaluation_every_epochs': {'FSOD': 5, 'UATD': 5, 'Marine': 1, 'NKSID': 1, 'HighResolution': 1, 'Diver': 1},
        'target_tasks': list(TARGETS), 'planned_target_updates': sum(old['target_schedules'][t]['updates'] for t in TARGETS),
        'primary': 'Terminal60 FSOD / terminal100 other tasks; best auxiliary, earliest strictbest; all outcomes retained.',
        'routing_controls': list(POLICIES), 'routing_scope': 'Same terminal model/head, K2 for all nonglobal controls; six fixed pair IDs, uniform random and top2 regional uncertainty. Not independently trained control models.',
        'heldout_scope': 'HighResolution39officialvalidation and Diver374test at terminal only, already evaluated historically; no cold/independent-acquisition claim. No new UATDTest/NKSIDouter/Channel access.',
        'diagnostics': 'CANOE and UXO retain original inputs/definitions/teacher targets; new source-frozen pair descriptors, three encoder image visits each, no extra neural predictor or physical correspondence claim.',
        'comparison_scope': 'K2, source fusion training and the selector gradient path differ from K1. More compute, not a single-factor or equal-compute K1 comparison. Source additive control supports source-held analysis only, not downstream additive-model superiority.',
        'references': {t: references[t] for t in TARGETS},
        'online_references': online_references, 'online_reference_caveat': 'UATD online trial evaluated every1 rather than every5; best opportunities/cost differ. Preserve its appended clarification.',
        'source_RAM': str(SOCKET), 'weight_storage': 'Source state only in controller RAM; no trained weight files. Additive source selector discarded after report.',
        'resource': {'GPUs': [2, 3], 'source_GPU': 3, 'max_joint_workers_per_GPU': 2,
            'job_GPU_GiB': 32, 'parent_CFC_cap_GiB': 36, 'free_margin_GiB': 8, 'job_RSS_GiB': 48,
            'source_seconds': 10800, 'diagnostic_seconds': 3600, 'target_seconds': CAPS,
            'global_seconds_max': 36000, 'new_disk_GiB': 8, 'CPU_threads': 4},
        'initial_admission': admission, 'coexisting_parent_jobs': parent_jobs,
        'diagram': str(core.BASE/'gsr_method_diagrams_v1/20_joint_pair_gsr.png'),
        'execution': 'No extra smoke/SHA/test/review. First normal source/target steps are counted. Failures visible, no automatic retries, no quality-based stop or schedule change.'}
    write(RUN/'protocol.json', protocol)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); listener.bind(str(SOCKET))
    os.chmod(SOCKET, 0o600); listener.listen(16)
    pending = ['source', 'FSOD', 'UATD', 'Marine', 'NKSID', 'HighResolution', 'Diver']
    active, waits, launches, blocked = {}, [], {}, []; blob = None; completion_written = False
    supervision_failed = False
    occupied = set().union(*(os.sched_getaffinity(pid) for pid in parent_pids))
    cpus = sorted(set(os.sched_getaffinity(0))-occupied)
    try:
        while pending or active or blob is not None:
            if time.time() > deadline:
                if pending or active: raise TimeoutError('Joint-pair global deadline')
                break
            if (RUN/'stop_source_owner').exists() and not pending and not active: break
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > 8*1024**3:
                raise RuntimeError('Joint-pair8GiB disk cap')
            ready, _, _ = select.select([listener], [], [], .2)
            if ready:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(120); command = parent.read_bytes(connection, 1)
                    if command == b'P':
                        if blob is not None: raise RuntimeError('Source snapshot published twice')
                        blob = parent.read_bytes(connection, int.from_bytes(parent.read_bytes(connection, 8), 'big'))
                        connection.sendall(b'K')
                        write(RUN/'source_RAM_receipt.json', {'source_cohort': protocol['source_cohort'],
                            'serialized_bytes': len(blob), 'immutable_source_for_all_targets': True, 'weight_files': 0})
                    elif command == b'G':
                        connection.sendall(len(blob).to_bytes(8, 'big')+blob)
                    else:
                        raise ValueError('Unknown source RAM command')
            for job_id, (child, log, gpu, began, job_deadline, slot) in list(active.items()):
                if time.time() > job_deadline and child.poll() is None: os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None: continue
                code = child.wait(); log.close(); del active[job_id]
                waits.append({'job': job_id, 'PID': child.pid, 'physical_GPU': gpu,
                    'actual_wait_exit': code, 'wall_seconds': time.time()-began})
                write(RUN/'actual_waits.json', waits)
                if job_id == 'source' and code != 0 and blob is None:
                    blocked = list(pending); pending.clear()
            for job_id in list(pending):
                if job_id != 'source' and blob is None: continue
                candidates = []
                for gpu in (2, 3):
                    if job_id == 'source' and gpu != 3: continue
                    old_count = sum(v['GPU'] == gpu and Path('/proc/'+str(v['PID'])).exists() for v in parent_jobs.values())
                    new_count = sum(v[2] == gpu for v in active.values())
                    if old_count+new_count >= 2: continue
                    observed = gpu_admission(gpu, parent_pids | {v[0].pid for v in active.values()})
                    if observed['eligible']: candidates.append((old_count*36+new_count*32, gpu, observed))
                if not candidates: continue
                _, gpu, observed = min(candidates, key=lambda v: (v[0], v[1]))
                folder = RUN/job_id
                if job_id != 'source': folder.mkdir()
                job_deadline = min(deadline, time.time()+(14400 if job_id == 'source' else CAPS[job_id]))
                job = {'job': job_id, 'physical_GPU': gpu, 'admission': observed, 'deadline_unix': job_deadline}
                if job_id == 'source': job['source_deadline_unix'] = min(deadline, time.time()+10800)
                write(folder/'job.json', job)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log = (folder/'stdout.log').open('w')
                child = subprocess.Popen([core.original.PYTHON, '-B', '-u', '-m', 'runs.gsr_joint_pair', 'worker', job_id],
                    cwd=core.original.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                occupied_slots = {v[5] for v in active.values()}; slot = next(s for s in range(4) if s not in occupied_slots)
                os.sched_setaffinity(child.pid, cpus[slot*4:slot*4+4])
                active[job_id] = (child, log, gpu, time.time(), job_deadline, slot); pending.remove(job_id)
                launches[job_id] = {'PID': child.pid, 'physical_GPU': gpu, 'launched_unix': time.time()}
                write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'workers': launches, 'weight_files': 0})
            failed = [v['job'] for v in waits if v['actual_wait_exit'] != 0]
            if pending or active:
                status = 'RUNNING'
            elif failed or blocked:
                status = 'FAILED_PARTIAL_SOURCE_RAM_RETAINED' if blob is not None else 'FAILED_PARTIAL'
            else:
                status = 'MODELS_COMPLETE_SOURCE_RAM_RETAINED'
            write(RUN/'queue_state.json', {'status': status, 'active': {k: {'PID': v[0].pid, 'GPU': v[2]} for k, v in active.items()},
                'pending': pending, 'failed': failed, 'blocked_unlaunched': blocked, 'completed': len(waits)-len(failed), 'source_in_RAM': blob is not None})
            if not pending and not active and not completion_written:
                results = {}
                for item in waits:
                    task = item['job']
                    if task == 'source' or item['actual_wait_exit'] != 0: continue
                    result = json.loads((RUN/task/'results.json').read_text()); key = result['metric']
                    reference = protocol['references'][task]
                    row = {'evidence': str(RUN/task/'results.json'), 'metric': key, 'higher_is_better': result['higher_is_better'],
                        'terminal': result['terminal'][key], 'best': result['best'][key], 'best_epoch': result['best_epoch'],
                        'delta_K1_terminal': result['terminal'][key]-reference['arms']['shared_GSR']['terminal'],
                        'delta_DINO_terminal': result['terminal'][key]-reference['arms']['original_DINO']['terminal'],
                        'references': reference, 'same_terminal_policies': {k: v[key] for k, v in result['terminal_policy_metrics'].items()},
                        'historical_terminal_differences': {k: result['terminal'][key]-v if v is not None else None for k, v in reference['historical_references'].items()}}
                    if task in protocol['online_references']:
                        row['online_reference'] = protocol['online_references'][task]
                        row['delta_online_terminal'] = result['terminal'][key]-protocol['online_references'][task]['terminal']
                    results[task] = row
                write(RUN/'comparison.json', {'status': status, 'tasks': results, 'scope': protocol['comparison_scope']})
                write(RUN/'completion.json', {'status': status, 'actual_waits': waits, 'blocked_unlaunched': blocked,
                    'elapsed_from_prepare_seconds': time.time()-started, 'new_weight_files': 0,
                    'source_owner_scope': 'Controller retains source in RAM, no GPU allocation, until explicit release or original bounded deadline. No supervisor wait0 claimed here.'})
                completion_written = True
            time.sleep(1)
        write(RUN/'source_owner_exit.json', {'source_RAM_released': True, 'actual_waits': waits})
        write(RUN/'queue_state.json', {'status': 'FAILED_PARTIAL' if failed or blocked else 'COMPLETE',
            'active': {}, 'pending': [], 'failed': failed, 'blocked_unlaunched': blocked, 'source_in_RAM': False})
    except BaseException:
        supervision_failed = True
        write(RUN/'supervision_failure.json', {'traceback': traceback.format_exc(), 'time_unix': time.time()})
        raise
    finally:
        for job_id, (child, log, gpu, began, job_deadline, slot) in active.items():
            if child.poll() is None: os.killpg(child.pid, signal.SIGKILL)
            code = child.wait(); log.close()
            waits.append({'job': job_id, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code,
                'wall_seconds': time.time()-began, 'supervision_cleanup': True})
        write(RUN/'actual_waits.json', waits)
        listener.close(); SOCKET.unlink()
        if supervision_failed:
            write(RUN/'queue_state.json', {'status': 'SUPERVISION_FAILED', 'active': {}, 'pending': pending,
                'actual_waits': waits, 'source_in_RAM': False})
    return 1 if any(v['actual_wait_exit'] != 0 for v in waits) or blocked else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pipeline', 'worker')); parser.add_argument('job_id', nargs='?')
    args = parser.parse_args()
    if args.phase == 'pipeline': raise SystemExit(pipeline())
    worker(args.job_id)
