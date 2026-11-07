"""Original one-of-four GSR, shared across sources and retained in downstream FT."""
import argparse
import gc
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from runs import selective_reencoding as original
from runs import downstream_vit_finetuning_readout as reader
from runs import gsr_shared_encoder_finetuning as shared
from sonar_encoder.training.model import build_encoder

BASE = original.BASE
RUN = BASE/'gsr_shared_complete_v1'
TASKS = ('FSOD', 'Marine', 'UATD')
METRIC = {'FSOD': 'AP50', 'Marine': 'miou12', 'UATD': 'map50'}
QUADS = original.QUADS
write = shared.write


def protocol():
    return json.loads((RUN/'protocol.json').read_text())


def guard(deadline):
    observed = {'RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > deadline:
        raise TimeoutError('Fixed complete-GSR deadline')
    if observed['RSS_bytes'] > 48*1024**3 or observed['GPU_reserved_bytes'] > 80*1024**3:
        raise RuntimeError('Complete GSR resource cap '+str(observed))
    return observed


class DifferenceFusion(original.Fusion):
    def __init__(self):
        super().__init__()
        self.difference_scale = nn.Parameter(torch.zeros(1, 384, 1, 1))
        nn.init.zeros_(self.gate.bias)

    def forward(self, local, crop, g, action):
        n = len(g)
        difference = local-crop
        q = (self.local(difference)+self.base(crop)).flatten(2).transpose(1, 2)
        q = q+self.position(self.local_xy[action])
        v = self.context(g).flatten(2).transpose(1, 2)
        k = v+self.position(self.global_xy)[None]
        qh, kh, vh = [x.reshape(n, 256, 4, 16).transpose(1, 2) for x in (q, k, v)]
        context = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.)
        u = context.transpose(1, 2).reshape(n, 256, 64)
        u = F.gelu(self.norm(q+u)).transpose(1, 2).reshape(n, 64, 16, 16)
        return (self.output(u)+self.difference_scale*difference)*self.gate(u).sigmoid()*self.taper


class GSR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            self.fusion = DifferenceFusion()
            self.selector = original.Selector()

    def global_field(self, x):
        return self.encoder(x)['patch_tokens'].transpose(1, 2).reshape(-1, 384, 16, 16)

    def corrected(self, g, local, action):
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        crops = torch.stack([base[..., r:r+16, c:c+16] for r, c in QUADS], 1)
        index = torch.arange(len(g), device=g.device)
        delta = self.fusion(local, crops[index, action.clamp_min(0)], g, action.clamp_min(0))
        for j, (r, c) in enumerate(QUADS):
            base = base+F.pad(delta*(action == j)[:, None, None, None], (c, 16-c, r, 16-r))
        return base

    def field(self, x, head, task, policy='learned', actions=None):
        g = self.global_field(x)
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        if policy == 'global':
            return base
        if actions is None:
            if policy == 'uncertainty':
                if task == 'FSOD':
                    logits = head.rpn.head([base])[0][0]
                    probability = logits.sigmoid().clamp(1e-6, 1-1e-6)
                    entropy = -(probability*probability.log()+(1-probability)*(1-probability).log()).mean(1, keepdim=True)
                elif task == 'Marine':
                    probability = head(base).softmax(1).clamp_min(1e-6)
                    entropy = -(probability*probability.log()).sum(1, keepdim=True)
                else:
                    probability = head(F.avg_pool2d(base, 2))[:, :10].sigmoid().clamp(1e-6, 1-1e-6)
                    entropy = -(probability*probability.log()+(1-probability)*(1-probability).log()).mean(1, keepdim=True)
                actions = F.adaptive_avg_pool2d(entropy, (2, 2)).flatten(1).argmax(1)
            else:
                actions = self.selector(g.detach()).argmax(1)
        large = F.interpolate(x, (448, 448), mode='bilinear', align_corners=False, antialias=True)
        tiles = torch.stack([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in QUADS], 1)
        local = self.global_field(tiles[torch.arange(len(x), device=x.device), actions])
        return self.corrected(g, local, actions)


def task_features(task, field):
    return F.avg_pool2d(field, 2) if task == 'UATD' else field


def loss(task, head, field, target):
    return reader.task_loss(task, head, task_features(task, field), target)


def cached_source(p, task):
    paths = p['source_caches'][task]
    g = torch.from_numpy(np.load(paths['global'], allow_pickle=False)).cuda()
    stitched = torch.from_numpy(np.load(paths['tiles'], allow_pickle=False)).cuda()
    local = torch.stack([stitched[..., r:r+16, c:c+16] for r, c in QUADS], 1)
    return g, local


def source_training(model, p):
    deadline = min(p['deadline_unix'], time.time()+3600)
    model.encoder.eval().requires_grad_(False)
    model.selector.requires_grad_(False)
    specs = p['source_input_definition']
    banks = {t: cached_source(p, t) for t in ('FSOD', 'Marine')}
    targets = {t: reader.make_targets(t, specs[t]['rows']['train']) for t in banks}
    targets['Marine'] = targets['Marine'].cuda()
    heads = nn.ModuleDict({t: reader.new_head(t, 0).cuda() for t in banks})
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=.001, weight_decay=.0001)
    head_opts = {t: reader.head_optimizer(t, heads[t]) for t in banks}
    plans = np.load(RUN/'source_plan.npz', allow_pickle=False)
    begun = time.monotonic()
    with (RUN/'source_training.jsonl').open('w') as log:
        for update in range(6000):
            guard(deadline)
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = shared.native.module_lr(update, 6000, .001)
            values = {}
            for task in banks:
                heads[task].train()
                opt = head_opts[task]
                opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = (.005 if update < 4000 else .0005 if update < 5500 else .00005) if task == 'FSOD' else shared.native.module_lr(update, 6000, .001)
                ids = torch.from_numpy(plans[task][update]).cuda()
                action = torch.from_numpy(plans[task+'_actions'][update]).cuda()
                g, local = banks[task]
                chosen = local[ids, action.clamp_min(0)]
                field = model.corrected(g[ids], chosen, action)
                target = reader.batch_targets(task, targets[task], ids.tolist(), 'cuda')
                value = .5*loss(task, heads[task], field, target)
                value.backward()
                values[task] = float(value.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for opt in head_opts.values():
                opt.step()
            if update == 0:
                write(RUN/'first_source_update.json', {'updates': 1, 'original_DINO_trainable': 0,
                    'shared_fusion_gradient_norm': float(norm), 'field': list(field.shape),
                    'source_batch_per_task': 4, 'cached_backbone_forward_calls': 0})
            if (update+1) % 100 == 0:
                row = {'updates': update+1, 'losses': values, 'wall_seconds': time.monotonic()-begun, **guard(deadline)}
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
    write(RUN/'source_report.json', {'status': 'COMPLETE_SHARED_FUSION_6000', 'updates': 6000,
        'feature_visits_per_task': 24000, 'new_original_encoder_forwards': 0,
        'terminal_parameter_state': 'raw fixed update6000; no EMA or epoch selection',
        'source_original_DINO_frozen': True, 'wall_seconds': time.monotonic()-begun, **guard(deadline)})
    model.eval(); heads.eval()
    gains = {}
    for task, (g, local) in banks.items():
        values = np.empty((len(g), 5), np.float32)
        heads[task].train(task == 'FSOD')
        with torch.no_grad():
            for i in range(len(g)):
                target = reader.batch_targets(task, targets[task], [i], 'cuda')
                for j in range(-1, 4):
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(10000+i)
                        action = torch.tensor([j], device='cuda')
                        field = model.corrected(g[i:i+1], local[i:i+1, max(j, 0)], action)
                        values[i, j+1] = float(loss(task, heads[task], field, target))
                if i % 100 == 0:
                    guard(deadline)
        gain = values[:, :1]-values[:, 1:]
        np.save(RUN/(task+'_signed_source_gains.npy'), gain)
        gains[task] = torch.from_numpy(gain).cuda()
    model.selector.requires_grad_(True)
    model.fusion.requires_grad_(False)
    opt = torch.optim.AdamW(model.selector.parameters(), lr=.001, weight_decay=.0001)
    fit, held, scaled, generators = {}, {}, {}, {}
    for i, task in enumerate(banks):
        order = torch.randperm(len(gains[task]), generator=torch.Generator().manual_seed(17+i))
        n = int(.8*len(order)); fit[task], held[task] = order[:n].cuda(), order[n:].cuda()
        contrast = gains[task]-gains[task].mean(1, keepdim=True)
        scaled[task] = contrast/contrast[fit[task]].std().clamp_min(1e-6)
        generators[task] = torch.Generator().manual_seed(31+i)
        np.savez(RUN/(task+'_selector_partition.npz'), fit=order[:n].numpy(), held=order[n:].numpy())
    steps = math.ceil(max(len(x) for x in fit.values())/64)
    for epoch in range(100):
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            for task in banks:
                ids = fit[task][torch.randint(len(fit[task]), (64,), generator=generators[task]).cuda()]
                prediction = model.selector(banks[task][0][ids])
                value = .5*F.mse_loss(prediction-prediction.mean(1, keepdim=True), scaled[task][ids])
                value.backward()
            opt.step()
        guard(deadline)
    report = {}
    with torch.no_grad():
        for task in banks:
            ids = held[task]
            choice = model.selector(banks[task][0][ids]).argmax(1)
            real = gains[task][ids]
            picked = real[torch.arange(len(ids), device='cuda'), choice]
            report[task] = {'fit_count': len(fit[task]), 'held_count': len(ids),
                'held_mean_selected_gain': float(picked.mean()),
                'held_mean_uniform_gain': float(real.mean()),
                'held_mean_oracle_regret': float((real.max(1).values-picked).mean()),
                'all_negative_gain_images': int((gains[task].max(1).values < 0).sum())}
    write(RUN/'selector_report.json', {'status': 'COMPLETE_FIXED_100_EPOCHS', 'source_only': True,
        'task_balanced_centered_gain_regression': True, 'partitions': report, **guard(deadline)})
    model.selector.eval().requires_grad_(False)
    model.fusion.requires_grad_(True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del banks, targets, heads, gains, scaled, opt, optimizer, head_opts
    gc.collect()
    return state


@torch.no_grad()
def evaluate(task, model, head, images, rows, targets, coco, batch, deadline, policy):
    model.eval(); head.eval(); pieces = []
    random_actions = np.random.RandomState(0).randint(0, 4, len(rows))
    for start in range(0, len(rows), batch):
        end = min(start+batch, len(rows)); actions = None
        if policy == 'random':
            actions = torch.as_tensor(random_actions[start:end], device='cuda')
        elif policy.startswith('fixed'):
            actions = torch.full((end-start,), int(policy[-1]), device='cuda', dtype=torch.long)
        field = model.field(images[start:end], head, task, policy, actions)
        pieces.append(reader.decode(task, head, task_features(task, field), rows[start:end]))
        guard(deadline)
    pred = [v for piece in pieces for v in piece] if task == 'UATD' else np.concatenate(pieces)
    canonical, independent = reader.original_scores(task, pred, targets, coco)
    return canonical, independent, pred


def fit(task, arm, initial, p):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    spec = p['downstream_tasks'][task]
    data = json.loads(Path(p['downstream_data']).read_text())['inputs'][task]
    folder = RUN/arm/task; folder.mkdir(parents=True)
    deadline = min(p['deadline_unix'], time.time()+p['budget']['fit_seconds'][task])
    model = GSR(p['encoder'])
    if arm == 'shared_GSR':
        model.load_state_dict(initial, strict=True)
    reader.configure_encoder(model.encoder, 'e2e')
    model.selector.requires_grad_(False)
    model.fusion.requires_grad_(arm == 'shared_GSR')
    model.cuda()
    head = reader.new_head(task, 0).cuda()
    body_opt, head_opt = reader.body_optimizer(model.encoder), reader.head_optimizer(task, head)
    fusion_opt = torch.optim.AdamW(model.fusion.parameters(), lr=.0001, weight_decay=.0001) if arm == 'shared_GSR' else None
    images = {s: torch.from_numpy(np.load(p['views'][task][s], allow_pickle=False)).cuda() for s in ('train', 'validation')}
    targets = {s: reader.make_targets(task, data['rows'][s]) for s in ('train', 'validation')}
    if task == 'UATD':
        from sonar_encoder.training.uatd_head import box_targets
        train_targets = box_targets(targets['train'], 'cpu')
    else:
        train_targets = targets['train']
    coco = json.loads(Path(data['GT_paths']['validation']).read_text()) if task == 'FSOD' else None
    generator = torch.Generator().manual_seed(0)
    routing = np.random.RandomState(700)
    primary_policy = 'learned' if arm == 'shared_GSR' else 'global'
    best, updates, begun = -1., 0, time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, spec['epochs']+1):
            model.train(); model.selector.eval(); head.train()
            if task == 'FSOD':
                head_opt.param_groups[0]['lr'] = .005 if epoch <= 40 else .0005 if epoch <= 55 else .00005
            order = torch.randperm(len(data['rows']['train']), generator=generator)
            total = 0.
            for ids in order.split(spec['batch']):
                guard(deadline)
                body_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
                if fusion_opt is not None:
                    fusion_opt.zero_grad(set_to_none=True)
                    fusion_opt.param_groups[0]['lr'] = shared.native.module_lr(updates, spec['updates_per_fit'], .0001)
                for group in body_opt.param_groups:
                    group['lr'] = reader.body_learning_rate(updates, spec['updates_per_fit'])
                x = images['train'][ids.cuda()]
                actions = None
                if arm == 'shared_GSR' and routing.rand() < .5:
                    actions = torch.as_tensor(routing.randint(0, 4, len(ids)), device='cuda')
                field = model.field(x, head, task, primary_policy, actions)
                target = reader.batch_targets(task, train_targets, ids.tolist(), 'cuda')
                value = loss(task, head, field, target)
                value.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                if fusion_opt is not None:
                    torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
                    fusion_opt.step()
                body_opt.step(); head_opt.step(); updates += 1
                total += float(value.detach())*len(ids)
                if updates == 1:
                    write(folder/'first_counted_update.json', {'updates': 1, 'body_gradient_norm': float(norm),
                        'complete_field': list(field.shape), 'head_field': list(task_features(task, field).shape),
                        'first_ids': ids.tolist(), 'shared_backbone_global_and_local': True,
                        'selector_source_fixed': True, 'encoder_input': list(x.shape), 'new_weight_files': 0})
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order), 'wall_seconds': time.monotonic()-begun}
            if epoch % spec['validation_every_epochs'] == 0:
                metric, independent, pred = evaluate(task, model, head, images['validation'], data['rows']['validation'], targets['validation'], coco, spec['batch'], deadline, primary_policy)
                entry['metric'] = metric[METRIC[task]]
                payload = {'prediction': pred, 'epoch': epoch, 'task': task, 'arm': arm, 'policy': primary_policy}
                if metric[METRIC[task]] > best:
                    best, best_epoch, best_metric = metric[METRIC[task]], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                if epoch == spec['epochs']:
                    torch.save(payload, folder/'terminal_predictions.pt')
                    terminal, terminal_independent = metric, independent
                print(json.dumps({'arm': arm, 'task': task, **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    policies = {primary_policy: terminal}
    if arm == 'shared_GSR':
        for policy in ('global', 'random', 'fixed0', 'fixed1', 'fixed2', 'fixed3', 'uncertainty'):
            metric, independent, pred = evaluate(task, model, head, images['validation'], data['rows']['validation'], targets['validation'], coco, spec['batch'], deadline, policy)
            policies[policy] = metric
            torch.save({'prediction': pred, 'epoch': spec['epochs'], 'task': task, 'policy': policy}, folder/(policy+'_terminal_predictions.pt'))
            write(folder/(policy+'_independent.json'), independent)
    write(folder/'results.json', {'status': 'COMPLETE_FIXED_COMPLETE_GSR_FIT', 'task': task, 'arm': arm,
        'epochs': spec['epochs'], 'updates': updates, 'terminal': terminal, 'best': best_metric, 'best_epoch': best_epoch,
        'terminal_independent': terminal_independent, 'terminal_policy_metrics': policies,
        'primary_policy': primary_policy, 'selector_source_fixed_during_downstream': True,
        'routing_controls_scope': 'Same terminal model and head; no separately trained policy baselines. Source selector may face descriptor drift during fullFT.',
        'new_weight_files': 0, 'wall_seconds': time.monotonic()-begun, **guard(deadline)})


def worker(arm):
    p = protocol()
    original.runtime()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    initial = None
    if arm == 'shared_GSR':
        model = GSR(p['encoder']).cuda()
        initial = source_training(model, p)
        del model
        gc.collect()
    for task in TASKS:
        fit(task, arm, initial, p)
        gc.collect()


def pipeline():
    parent = json.loads((BASE/'gsr_common_source_repeats_v1/protocol.json').read_text())
    if json.loads((BASE/'gsr_common_source_repeats_v1/queue_state.json').read_text())['status'] != 'COMPLETE':
        raise RuntimeError('The already frozen repeat queue must finish before new dispatch')
    RUN.mkdir()
    started = time.time()
    detection = json.loads((BASE/'gsr_detection_transfer_v1/protocol.json').read_text())
    marine = json.loads((BASE/'selective_reencoding_marine_v1/protocol.json').read_text())
    source_caches = {'FSOD': {'global': detection['tasks']['FSOD']['global_caches']['train'], 'tiles': detection['tasks']['FSOD']['tile_caches']['train']},
                     'Marine': {k: marine['caches']['train'][k] for k in ('global', 'tiles')}}
    prep = json.loads(Path(parent['downstream_data']).read_text())
    definition = {**parent, 'status': 'FROZEN_COMPLETE_SHARED_GSR_BEFORE_FORWARD', 'started_unix': started,
        'deadline_unix': started+28800, 'source_entry': str(Path(__file__).resolve()),
        'diagram': str(BASE/'gsr_method_diagrams_v1/17_shared_complete_gsr.png'),
        'budget': {'global_seconds': 28800, 'source_seconds': 3600,
                   'fit_seconds': {'FSOD': 5000, 'Marine': 2400, 'UATD': 6500},
                   'RSS_bytes': 48*1024**3, 'GPU_reserved_bytes': 80*1024**3, 'new_disk_bytes': 4*1024**3},
        'source_caches': source_caches, 'source_updates': 6000, 'source_samples_per_task_update': 4,
        'source_parameters': 'Shared difference fusion and independent task heads, original DINO fixed; cached final global16/local16 features, no QKV or discarded auxiliary modules.',
        'source_loss': '.5 L_FSOD + .5 L_Marine; uniform route over no-region and four single-region choices.',
        'selector': 'One shared source-only selector, raw signed gains retained. Frozen source fusion/heads generate all five losses per training image with matched FSOD head-sampling RNG. Task-balanced centered gain regression; per-source fixed80/20 selector fit/diagnostic partition,100epochs,no score gate.',
        'feature_interface': 'Every arm receives identical original common224 input; global16->bilinear32. GSR enlarges common224 to448 and encodes exactly one224 quadrant with the same DINO. Complete output384x32x32. UATD task head uses explicit avgpool2 to retain original16grid GT/loss/decoder; F/M use32 directly.',
        'fusion': 'Cross-attention from local-minus-base/crop features to global tokens with raster position, zero-initialized channelwise difference scale plus zero-initialized learned residual, sigmoid spatial gate initialized.5, sine boundary taper.',
        'downstream': 'Copy the identical complete source state independently for each task, keep selector/fusion rather than merge/drop. Original DINO+fusion+newhead train, source selector fixed, half minibatches source-learned routing/half uniform random. Matched baseline original DINO global224->32, same heads/seeds/orders/schedules. No head architecture upgrade.',
        'primary': 'Terminal60/100/100 learned routing; best scheduled endpoint auxiliary. Same-terminal random/fixed0..3/uncertainty/global policy diagnostics, no taskwise winning-policy selection.',
        'no_smoke_digests_review_or_weight_files': True, 'seeds': [0], 'physical_GPUs': {'shared_GSR': 2, 'original_DINO': 3},
        'queue': 'GPU2 shared source then independent F/M/U fits; GPU3 matched global224 DINO F/M/U. One owned worker/card. Stop visibly on actual runtime/resource/deadline failure; no automatic batch/precision/parameter/seed changes.',
        'SWiSE_fullFT_references': {t: parent['references'][t]['baseline_pool']['ordinary_target_ViT_finetuning']['mean'] for t in ('FSOD', 'Marine')},
        'SWiSE_fullFT_evidence': str(BASE/'downstream_vit_finetuning_v1/writing_handoff.json'),
        'strongest_prior_completed_pool': json.loads((BASE/'gsr_selector_context_recovery_v1/results.json').read_text())['current_references'],
        'historical_reference_scope': 'Frozen SWiSE, SWiSE whole448 fullFT F/M, and strongest completed references are separate complete paths. Current matched DINO uses original224->regrid32; historical Whole448 differs in input/compute. Prior UATD/NK224 fine-tunes are not called SWiSE448. Previous global-only source repeats are controls, not this original GSR system.'}
    views = {}
    for task in TASKS:
        views[task] = {s: str(BASE/'downstream_vit_finetuning_v1/inputs'/task/(s+'.npy')) if task == 'UATD' else str(RUN/'inputs'/task/(s+'.npy')) for s in ('train', 'validation')}
    definition['views'] = views
    write(RUN/'protocol.json', definition)
    for task in ('FSOD', 'Marine'):
        data = prep['inputs'][task]
        if data['rows']['train'] != parent['source_input_definition'][task]['rows']['train']:
            raise RuntimeError(task+' source/downstream row identity mismatch')
        for split in ('train', 'validation'):
            path = Path(views[task][split]); path.parent.mkdir(parents=True, exist_ok=True)
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(data['rows'][split]), 1, 224, 224))
            for i, row in enumerate(data['rows'][split]):
                array[i] = reader.normalized_input(row, data['normalization'], 'UATD').numpy()
            array.flush()
        for key, path in source_caches[task].items():
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            expected = (len(data['rows']['train']), 384, 16 if key == 'global' else 32, 16 if key == 'global' else 32)
            if array.shape != expected or array.dtype != np.float32:
                raise RuntimeError(task+' retained original-feature cache shape/dtype mismatch')
    plans = {}
    for i, task in enumerate(('FSOD', 'Marine')):
        n = len(prep['inputs'][task]['rows']['train']); rng = np.random.RandomState(13+i)
        plans[task] = np.concatenate([rng.permutation(n) for _ in range(math.ceil(24000/n))])[:24000].reshape(6000, 4)
        plans[task+'_actions'] = np.random.RandomState(23+i).randint(-1, 4, (6000, 4))
    np.savez(RUN/'source_plan.npz', **plans)
    write(RUN/'prepare_receipt.json', {'status': 'COMPLETE_COMMON224_VIEWS_AND_SOURCE_ROW_BINDING',
        'new_common224_decodes': 4393, 'UATD_existing224_view_reused': True, 'new_model_forwards': 0,
        'source_existing_cache_scope': 'Retained original frozen feature provenance, rows and headers; no fresh payload hashes or encoder extraction.',
        'wall_seconds': time.time()-started})
    (RUN/'logs').mkdir()
    active, waits = {}, []
    try:
        for i, (arm, gpu) in enumerate((('shared_GSR', 2), ('original_DINO', 3))):
            free, util = map(int, subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
            owned = [int(pid) for pid in pids if Path('/proc/'+pid).stat().st_uid == os.getuid()]
            if free < 24576 or owned:
                raise RuntimeError('Physical GPU admission '+str((gpu, free, owned)))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            log = (RUN/'logs'/(arm+'.log')).open('w')
            child = subprocess.Popen([original.PYTHON, '-B', '-u', '-m', 'runs.gsr_shared_complete', arm], cwd=original.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(child.pid, cpus[i*4:i*4+4])
            active[arm] = (child, log, gpu)
            write(RUN/('admission_GPU'+str(gpu)+'.json'), {'physical_GPU': gpu, 'free_MiB': free, 'utilization_percent': util, 'same_UID_GPU_PIDs': owned, 'idle_claim': False})
        write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'workers': {k: v[0].pid for k, v in active.items()}, 'physical_GPUs': definition['physical_GPUs'], 'new_weight_files': 0})
        while active:
            if time.time() > definition['deadline_unix']:
                raise TimeoutError('Complete GSR8h deadline')
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > definition['budget']['new_disk_bytes']:
                raise RuntimeError('Complete GSR4GiB new disk cap')
            for arm, (child, log, gpu) in list(active.items()):
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del active[arm]
                waits.append({'arm': arm, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code})
                if code != 0:
                    raise RuntimeError(arm+' worker exit '+str(code))
            write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in active.items()}, 'actual_waits': waits})
            time.sleep(2)
        raw = {arm: {t: json.loads((RUN/arm/t/'results.json').read_text()) for t in TASKS} for arm in ('shared_GSR', 'original_DINO')}
        write(RUN/'results.json', {'status': 'COMPLETE_SHARED_GSR_ORIGINAL_STRUCTURE', 'arms': raw,
            'paired_terminal_differences': {t: raw['shared_GSR'][t]['terminal'][METRIC[t]]-raw['original_DINO'][t]['terminal'][METRIC[t]] for t in TASKS},
            'historical_strong_references': parent['references'],
            'strongest_prior_completed_pool': definition['strongest_prior_completed_pool'],
            'SWiSE_fullFT_references': definition['SWiSE_fullFT_references'],
            'SWiSE_fullFT_evidence': definition['SWiSE_fullFT_evidence'],
            'SWiSE': {'FSOD': .902005317706368, 'Marine': .6059200833586624, 'UATD': .2517123460319482}})
        write(RUN/'final_closure.json', {'status': 'COMPLETE_FIXED_SCOPE', 'actual_waits': waits, 'wall_seconds': time.time()-started, 'new_weight_files': 0})
        write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        for child, log, gpu in active.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(); log.close()
        write(RUN/'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pipeline', 'shared_GSR', 'original_DINO'))
    args = parser.parse_args()
    pipeline() if args.phase == 'pipeline' else worker(args.phase)
