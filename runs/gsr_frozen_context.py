"""Frozen original DINO: matched post-encoding and intra-encoding context pilots."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from runs import selective_reencoding as old
from runs import gsr_strong_adaptation as reuse
from runs import downstream_vit_finetuning_readout as reader
from sonar_encoder.training.model import build_encoder, prepare_images

ROOT = old.ROOT
RUN = old.BASE / 'gsr_frozen_context_v1'
reuse.RUN = RUN
MODES = ('post', 'intra')


class SpatialNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1536, 128, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 384, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, levels):
        return levels[:, -1] + self.net(levels.flatten(1, 2))


class Model(nn.Module):
    context16 = reuse.Model.context16
    local_images = reuse.Model.local_images
    corrected = reuse.Model.corrected

    def __init__(self, task, mode):
        super().__init__()
        self.task, self.mode = task, mode
        cfg = reuse.protocol()['encoder']
        self.encoder = build_encoder(Path(cfg['code_dir']), Path(cfg['weights_path']), 'R010')
        self.encoder.eval().requires_grad_(False)
        self.head = reader.new_head(task, 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            self.neck = SpatialNeck()
            self.bridges = nn.ModuleList([old.Fusion(), old.Fusion()])
            self.fusion = old.Fusion()

    @torch.no_grad()
    def global_levels(self, images):
        large = F.interpolate(images, (448, 448), mode='bilinear', align_corners=False,
                              antialias=True) if self.task == 'UATD' else images
        levels = self.encoder.backbone.get_intermediate_layers(large, [2, 5, 8, 11], reshape=True)
        if self.task == 'UATD':
            levels = [F.avg_pool2d(x, 2) for x in levels]
        return torch.stack(levels, 1)

    def global_field(self, images):
        return self.neck(self.global_levels(images))

    @torch.no_grad()
    def early_tokens(self, images):
        tokens = self.encoder.backbone.prepare_tokens_with_masks(images)
        for block in self.encoder.backbone.blocks[:8]:
            tokens = block(tokens)
        return tokens

    def finish_local(self, tokens, g, actions):
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False,
                             antialias=False) if self.task == 'UATD' else g
        crops = torch.stack([base[..., r:r+16, c:c+16] for r, c in old.QUADS], 1)
        crops = crops[torch.arange(len(g), device=g.device), actions]
        context = self.context16(g)
        for index in range(8, 12):
            if self.mode == 'intra' and index in (8, 10):
                local = self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, 16, 16)
                delta = self.bridges[(index-8)//2](local, crops, context, actions)
                tokens = torch.cat((tokens[:, :1], tokens[:, 1:] + delta.flatten(2).transpose(1, 2)), 1)
            tokens = self.encoder.backbone.blocks[index](tokens)
        local = self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, 16, 16)
        if self.mode == 'post':
            for bridge in self.bridges:
                local = local + bridge(local, crops, context, actions)
        return local

    def local_fields(self, images, actions, g):
        tokens = self.early_tokens(self.local_images(images, actions))
        return self.finish_local(tokens, g, actions)


@torch.no_grad()
def extract_prefixes(model, images):
    raw = {}
    global_images = local_images = 0
    begun = time.monotonic()
    grid = 16 if model.task == 'UATD' else 32
    for split, values in images.items():
        levels = torch.empty((len(values), 4, 384, grid, grid), device='cuda')
        early = torch.empty((len(values), 4, 257, 384), device='cuda')
        for start in range(0, len(values), 8):
            x = values[start:start+8]
            levels[start:start+len(x)] = model.global_levels(x)
            tiles = model.local_images(x)
            early[start:start+len(x)] = model.early_tokens(tiles).reshape(len(x), 4, 257, 384)
            global_images += len(x)
            local_images += len(tiles)
            if start % 256 == 0:
                print(json.dumps({'extract': model.task, 'split': split, 'rows': start+len(x)}), flush=True)
                reuse.resources()
        raw[split] = (levels, early)
    old.write(RUN/model.task/'extraction.json', {
        'global_full_encoder_views': global_images, 'four_local_first8_block_views': local_images,
        'wall_seconds': time.monotonic()-begun, 'GPU_RAM_only_no_feature_files': True,
        'all_original_encoder_parameters_frozen': True, 'resources': reuse.resources()})
    return raw


def batch_field(model, data, ids, actions):
    levels, early = data
    g = model.neck(levels[ids])
    tokens = early[ids, actions.clamp_min(0)]
    local = model.finish_local(tokens, g, actions.clamp_min(0))
    return g, model.corrected(g, local, actions)


@torch.no_grad()
def routing_metric(model, spec, data, targets, dense, actions):
    model.eval()
    pieces = []
    for start in range(0, len(actions), spec['batch']):
        ids = torch.arange(start, min(start+spec['batch'], len(actions)), device='cuda')
        _, field = batch_field(model, data, ids, actions[ids])
        pieces.append(reader.decode(model.task, model.head, field,
                                    spec['rows']['validation'][start:start+len(ids)]))
    predictions = [x for part in pieces for x in part] if model.task == 'UATD' else np.concatenate(pieces)
    if model.task == 'FSOD':
        from runs.shared_supervised_json_readout import canonical_evaluate
        from runs.shared_supervised_readout import prediction_list
        return canonical_evaluate(json.loads(Path(spec['GT_paths']['validation']).read_text()),
                                  prediction_list(predictions))['AP50']
    if model.task == 'UATD':
        from sonar_encoder.training.uatd_head import detection_metrics
        return detection_metrics(predictions, targets)['map50']
    from sonar_encoder.training.readout import confusion_matrix, metrics_from_confusion
    return metrics_from_confusion(confusion_matrix(dense.cpu(), torch.as_tensor(predictions), 12), 'marine')['miou12']


def train(model, spec, raw, targets, dense):
    folder = RUN/model.task/model.mode
    folder.mkdir()
    head_opt = reader.head_optimizer(model.task, model.head)
    extra = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('head.')]
    opt = torch.optim.AdamW(extra, lr=.001, weight_decay=.0001)
    gen = torch.Generator().manual_seed(0)
    actions = torch.from_numpy(np.load(RUN/model.task/'training_actions.npy')).cuda()
    validation_actions = torch.from_numpy(np.load(RUN/model.task/'validation_actions.npy')).cuda()
    best, best_epoch, updates = -1., 0, 0
    begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, spec['epochs']+1):
            model.train(); model.encoder.eval()
            order = torch.randperm(len(spec['rows']['train']), generator=gen)
            if model.task == 'FSOD':
                lr = .005 if epoch <= 40 else (.0005 if epoch <= 55 else .00005)
                for group in head_opt.param_groups:
                    group['lr'] = lr
            loss_sum = 0.
            for ids in order.split(spec['batch']):
                ids = ids.cuda()
                head_opt.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
                g, field = batch_field(model, raw['train'], ids, actions[epoch-1, ids])
                target = reuse.targets_batch(model.task, targets['train'], dense.get('train'), ids.cpu())
                loss = .5*reader.task_loss(model.task, model.head, g, target) + .5*reader.task_loss(model.task, model.head, field, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite real task training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(extra, 1., error_if_nonfinite=True)
                head_opt.step(); opt.step(); updates += 1
                loss_sum += float(loss.detach())*len(ids)
                if updates == 1:
                    old.write(folder/'first_counted_update.json', {
                        'update': 1, 'body_trainable_parameters': 0,
                        'body_has_gradient': any(p.grad is not None for p in model.encoder.parameters()),
                        'head_grad_norm': float(sum(p.grad.square().sum() for p in model.head.parameters() if p.grad is not None).sqrt()),
                        'extra_grad_norm': float(sum(p.grad.square().sum() for p in extra if p.grad is not None).sqrt()),
                        'all_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'global_field': list(g.shape), 'loss': float(loss.detach())})
                if updates % 100 == 0:
                    reuse.resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': loss_sum/len(order),
                     'elapsed_seconds': time.monotonic()-begun}
            if epoch % 5 == 0:
                score = routing_metric(model, spec, raw['validation'], targets['validation'], dense.get('validation'), validation_actions)
                entry['scheduled_random_route_metric'] = score
                if score > best:
                    best, best_epoch = score, epoch
                print(json.dumps({'task': model.task, 'mode': model.mode, **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    old.write(folder/'train_report.json', {
        'epochs': spec['epochs'], 'updates': updates, 'train_visits': spec['epochs']*len(order),
        'body_trainable_parameters': 0, 'original_pretrained_encoder_only': True,
        'scheduled_random_route_terminal': score, 'scheduled_random_route_best': best,
        'scheduled_random_route_best_epoch': best_epoch,
        'best_scope': 'Auxiliary random routing, not a best learned-selector endpoint',
        'wall_seconds': time.monotonic()-begun, 'new_weights_saved': False, 'resources': reuse.resources()})
    return updates


@torch.no_grad()
def final_banks(model, raw):
    banks = {}
    for split, (levels, early) in raw.items():
        n = len(levels)
        g = torch.empty((n, 384, *levels.shape[-2:]), device='cuda')
        local = torch.empty((n, 4, 384, 16, 16), device='cuda')
        for start in range(0, n, 8):
            end = min(start+8, n)
            current = model.neck(levels[start:end]); g[start:end] = current
            for region in range(4):
                a = torch.full((end-start,), region, dtype=torch.long, device='cuda')
                local[start:end, region] = model.finish_local(early[start:end, region], current, a)
            if start % 256 == 0:
                reuse.resources()
        banks[split] = (g, local)
    return banks


def evaluate(model, spec, bank, targets, selector, sr, updates):
    g, _ = bank['validation']; n = len(g)
    with torch.no_grad():
        scores = torch.cat([selector(model.context16(g[i:i+64])) for i in range(0, n, 64)]).cpu().numpy()*sr['gain_scale']+sr['gain_mean']
        entropy = reuse.entropy_actions(model, g)
    top = scores.argmax(1); keep = scores.max(1) > 0
    policies = {'global_only': np.full(n, -1, np.int64), 'learned': np.where(keep, top, -1),
        'learned_forced_top1': top, 'random_budgetmatched': np.where(keep, np.random.default_rng(2).integers(0, 4, n), -1),
        'fixed_budgetmatched': np.where(keep, sr['fixed_region_fit_only'], -1),
        'entropy_budgetmatched': np.where(keep, entropy, -1), 'entropy_forced_top1': entropy}
    folder = RUN/model.task/model.mode; metrics = {}
    for policy, actions in policies.items():
        predictions = reuse.predict(model, spec, bank['validation'], actions)
        metrics[policy] = reuse.score_prediction(model.task, predictions, targets['validation'], spec, folder/(policy+'_prediction'))
        print(json.dumps({'score': model.task+'/'+model.mode+'/'+policy, 'value': metrics[policy][spec['metric']]}), flush=True)
    np.savez(folder/'actions.npz', **policies)
    result = {'task': model.task, 'mode': model.mode, 'terminal_updates': updates, 'metrics': metrics,
        'selection_fraction': float(keep.mean()), 'body_trainable_parameters': 0, 'target_finetuned_encoder_reused': False,
        'reference': reuse.protocol()['references'][model.task], 'SWiSE': reuse.protocol()['SWiSE_references'][model.task],
        'new_weight_files': 0, 'original_GT_scores_from_saved_predictions': True, 'resources': reuse.resources()}
    old.write(folder/'results.json', result)
    return result


@torch.no_grad()
def online_cost(model, spec, selector, sr):
    rows = spec['rows']['validation'][:32]; samples = []
    for row in rows:
        with Image.open(row['image']) as image:
            a = torch.from_numpy(np.array(image.convert('L'), dtype=np.uint8)).float()[None]/255
        samples.append({'intensity': a, 'valid': torch.ones(a.shape[-2:], dtype=torch.bool)})
    cells = []
    for policy in ('global_only', 'learned'):
        load = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
        baseline = torch.cuda.memory_reserved(); torch.cuda.reset_peak_memory_stats(); records = []
        for visit in range(146):
            i = visit % 32 if visit < 50 else (visit-50) % 32
            torch.cuda.synchronize(); start = time.perf_counter()
            common = prepare_images([samples[i]], spec['normalization'], None)
            if model.task != 'UATD':
                common = F.interpolate(common, (448, 448), mode='bilinear', align_corners=False, antialias=True)
            images = common.cuda(); g = model.global_field(images); action = -1
            if policy == 'learned':
                scores = selector(model.context16(g))*sr['gain_scale']+sr['gain_mean']
                if float(scores.max()) > 0:
                    action = int(scores.argmax(1))
            if action >= 0:
                a = torch.tensor([action], device='cuda')
                local = model.local_fields(images, a, g)
                field = model.corrected(g, local, a)
            else:
                field = g
            reader.decode(model.task, model.head, field, rows[i:i+1])
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            if visit >= 50:
                records.append({'row_id': spec['row_ids']['validation'][i], 'round': (visit-50)//32,
                                'seconds': elapsed, 'action': action})
        times = [x['seconds'] for x in records]
        cells.append({'policy': policy, 'median_seconds': float(np.median(times)),
            'p95_seconds': float(np.percentile(times, 95)), 'raw': records, 'load': load,
            'baseline_reserved': baseline, 'peak_reserved': torch.cuda.max_memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'selected_fraction_timed': float(np.mean([x['action'] >= 0 for x in records]))})
    old.write(RUN/model.task/model.mode/'cost.json', {'cells': cells,
        'boundary': 'Native CPUimage preprocessing/H2D/one globalencoder/selector/atmostone full localencoder/context/fusion/head/nativepostproc; decode/setup excluded.',
        'memory_scope': 'Training featurebanks still resident; not isolated deployment memory.',
        'other_GPU_training_may_be_active': True, 'same_GPU_other_own_workers': False})


def workflow(task):
    spec, images, targets, dense = reuse.data_for(task)
    first = Model(task, MODES[0]).cuda()
    raw = extract_prefixes(first, images)
    del images
    results = []
    for mode in MODES:
        model = first if mode == MODES[0] else Model(task, mode).cuda()
        updates = train(model, spec, raw, targets, dense)
        bank = final_banks(model, raw)
        gains = reuse.gain_labels(model, spec, bank['train'], targets['train'], dense.get('train'))
        selector, sr = reuse.train_selector(task, mode, model.context16(bank['train'][0]), gains)
        results.append(evaluate(model, spec, bank, targets, selector, sr, updates))
        online_cost(model, spec, selector, sr)
        del model, bank, selector
        if mode == MODES[0]:
            del first
    old.write(RUN/task/'results.json', {'status': 'COMPLETE_BOTH_FROZEN_CONTEXT_MODES', 'modes': results})


def pipeline():
    RUN.mkdir(); (RUN/'logs').mkdir()
    start = time.time()
    prep = json.loads((old.BASE/'downstream_vit_finetuning_v1/cpu_preparation.json').read_text())
    refs = json.loads((old.BASE/'gsr_detection_transfer_v1/strongest_baseline_comparison.json').read_text())['tasks']
    p = {'start_unix': start, 'deadline_unix': start+28800, 'encoder': prep['official_config'], 'tasks': {},
        'references': refs, 'source_utf8': Path(__file__).read_text(), 'modes': list(MODES),
        'user_direction': 'Original pretrained DINO frozen, train new structure and heads; no task-finetuned encoder initialization.',
        'global': 'Common224-derived448; official normalized layers3/6/9/12. UATD pool32to16 before spatial neck.',
        'spatial_neck': 'Concatenated4levels->128 1x1/GN/GELU, two3x3/GN/GELU, zero384residual on lastlayer.',
        'context_comparison': 'Two identical64d cross-attention bridges: post applies after norm; intra injects before frozen blocks9 and11. Same parameters/heads/plans, different placement.',
        'local_correction': 'One of four224tiles; original gated tapered residual scatter. Original task grids/GT/decoder retained.',
        'training': '0.5global+0.5random-route original taskloss, route none20% and quadrants20% each; neck/3fusionmodules AdamW.001 wd.0001 clip1; original taskhead optimizer and epoch LR.',
        'selector': 'Existing spatial predictor, signed balanced train-only gains; fixed80/20 selectorfit split; no devGT at inference.',
        'storage': 'Frozen prefix and final features in GPU RAM only; no new weights/checkpoints or feature files. Necessary predictions/gainlabels/logs only.',
        'best_endpoint_scope': 'Terminal learned policy primary; scheduled random-route best auxiliary, not a best learned-policy checkpoint.',
        'caps': {'global_seconds': 28800, 'workflow_seconds': 14400, 'RSS_bytes': 48*1024**3, 'GPU_reserved_bytes': 48*1024**3, 'new_disk_bytes': 4*1024**3},
        'GPU_schedule': 'Marine2/UATD3 first, FSOD on first freed allowedcard; one ownedworker/card. NoGPU0/1.',
        'stop': 'Real exception/nonfinite/cap stops queue; no performance-based stops or automatic recovery.',
        'no_tests_SHA_reviews_or_new_weight_files': True,
        'acceptance': 'Exceed SWiSE and terminal no more than .01 below strongest completed compatible local baseline; single seed development is not robust or global SOTA certification.',
        'SWiSE_references': {'FSOD': .902005317706368, 'Marine': .6059200833586624, 'UATD': .2517123460319482},
        'SWiSE_sources': {'FSOD_Marine': str(old.BASE/'whole448_frozen_reference_v1/results.json'), 'UATD': str(old.BASE/'gsr_strong_adaptation_v1/UATD/official_swise_pool16/results.json')},
        'comparability': 'Single seed0 with new spatialdecoder/extra local compute, versus historic originals and three-seed strong-reference means; not an equal-capacity, equal-compute or shared adapted encoder claim.'}
    for task, batch, epochs in [('Marine', 16, 100), ('UATD', 32, 100), ('FSOD', 8, 60)]:
        folder = RUN/task; folder.mkdir(); data = prep['inputs'][task]; n = len(data['rows']['train'])
        p['tasks'][task] = {**data, 'batch': batch, 'epochs': epochs, 'input_views': prep['input_caches'][task],
                           'metric': {'Marine': 'miou12', 'FSOD': 'AP50', 'UATD': 'map50'}[task]}
        actions = np.random.default_rng(1).integers(-1, 4, (epochs, n), dtype=np.int64)
        np.save(folder/'training_actions.npy', actions)
        np.save(folder/'validation_actions.npy', np.random.default_rng(2).integers(0, 4, len(data['rows']['validation']), dtype=np.int64))
        split = np.random.default_rng(37).permutation(n); cut = int(.8*n)
        np.savez(folder/'selector_split.npz', fit=split[:cut], check=split[cut:])
    p['total_task_optimizer_updates'] = 2*(6300+19000+17700)
    old.write(RUN/'protocol.json', p)
    pending = [('Marine', 2), ('UATD', 3), ('FSOD', None)]; children = {}; waits = []
    def launch(task, gpu):
        admission = old.admission(gpu)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'XFORMERS_DISABLED': '1'}
        log = (RUN/'logs'/f'{task}.log').open('w')
        child = subprocess.Popen([old.PYTHON, '-B', '-u', '-m', 'runs.gsr_frozen_context', task], cwd=ROOT,
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[task] = (child, log, gpu, time.time(), admission)
    try:
        launch(*pending.pop(0)); launch(*pending.pop(0))
        old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'workers': {k: v[0].pid for k, v in children.items()},
            'protocol': str(RUN/'protocol.json'), 'no_new_weights_saved': True})
        while children:
            for task, (child, log, gpu, begun, admission) in list(children.items()):
                if time.time()-begun > 14400 or time.time() > p['deadline_unix']:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del children[task]
                waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code,
                              'wall_seconds': time.time()-begun, 'admission': admission})
                old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in children.items()}, 'actual_waits': waits})
                if code != 0:
                    raise RuntimeError(task+' actual worker exit '+str(code))
                if pending:
                    next_task, _ = pending.pop(0); launch(next_task, gpu)
            time.sleep(1)
        results = {task: json.loads((RUN/task/'results.json').read_text()) for task in p['tasks']}
        old.write(RUN/'results.json', {'status': 'COMPLETE_FIXED_FROZEN_STRUCTURE_PILOT', 'tasks': results})
        size = sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file())
        if size > p['caps']['new_disk_bytes']:
            raise RuntimeError('Results-only disk cap exceeded')
        close = {'status': 'COMPLETE_FIXED_WAVE_GOAL_NOT_AUTOMATICALLY_ACCEPTED', 'actual_waits': waits,
                 'wall_seconds': time.time()-start, 'new_entity_bytes': size, 'no_weights_saved': True}
        old.write(RUN/'wave_closure.json', close); old.write(RUN/'writing_handoff.json', close)
        old.write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        failure = traceback.format_exc()
        for task, (child, log, gpu, begun, admission) in children.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            code = child.wait(); log.close()
            waits.append({'task': task, 'PID': child.pid, 'actual_wait_exit': code, 'cancelled_after_error': True})
        old.write(RUN/'failure_closure.json', {'status': 'STOPPED_NO_AUTOMATIC_RETRY', 'traceback': failure, 'actual_waits': waits})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['pipeline', 'Marine', 'FSOD', 'UATD'])
    args = parser.parse_args(); old.runtime()
    if args.phase == 'pipeline':
        pipeline()
    else:
        workflow(args.phase)
