"""Shared regional source training, merged DINO, matched ordinary full fine-tuning."""
import argparse
import gc
from functools import cache
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

from runs import gsr_feature_residual as regional
from runs import downstream_vit_finetuning_readout as reader
from sonar_encoder.training.model import build_encoder

base, core, native = regional.base, regional.core, regional.native
ROOT = Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))
RUN = core.old.BASE/'gsr_shared_encoder_finetuning_v1'
PARENT = core.old.BASE/'gsr_selector_context_recovery_v1'
TASKS = ('FSOD', 'Marine', 'UATD')
METRIC = {'FSOD': 'AP50', 'Marine': 'miou12', 'UATD': 'map50'}
CPUS = sorted(os.sched_getaffinity(0))
core.reuse.RUN = RUN


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


@cache
def protocol():
    return json.loads((RUN/'protocol.json').read_text())


def guard(deadline):
    p = protocol()
    observed = {'RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > deadline:
        raise TimeoutError('Shared encoder fixed processing deadline')
    if observed['RSS_bytes'] > p['budget']['RSS_bytes'] or observed['GPU_reserved_bytes'] > p['budget']['GPU_reserved_bytes']:
        raise RuntimeError('Shared encoder actual resource cap '+str(observed))
    return observed


@torch.no_grad()
def source_tokens(encoder, spec, deadline):
    rows = spec['rows']['train']
    g = torch.empty((len(rows), 1025, 384), device='cuda')
    local = torch.empty((len(rows), 4, 1025, 384), device='cuda')
    for start in range(0, len(rows), 8):
        global_images, region_images = [], []
        for row in rows[start:start+8]:
            x = (native.read_intensity(row).log1p()-spec['normalization']['mean'])/spec['normalization']['std']
            global_images.append(F.interpolate(x[None], (448, 448), mode='bilinear', align_corners=False, antialias=True)[0])
            large = F.interpolate(x[None], (768, 768), mode='bilinear', align_corners=False, antialias=True)[0]
            region_images.append(torch.stack([large[..., y:y+448, x:x+448] for y, x in ((0, 0), (0, 320), (320, 0), (320, 320))]))
        n = len(global_images)
        g[start:start+n] = encoder.backbone.prepare_tokens_with_masks(torch.stack(global_images).cuda())
        local[start:start+n] = encoder.backbone.prepare_tokens_with_masks(torch.stack(region_images).flatten(0, 1).cuda()).reshape(n, 4, 1025, 384)
        guard(deadline)
        if start % 256 == 0:
            print(json.dumps({'source_extract': len(rows), 'rows': start+n}), flush=True)
    return g, local


def source_training():
    p = protocol()
    begun = time.monotonic()
    deadline = min(p['deadline_unix'], time.time()+p['budget']['source_seconds'])
    teacher = base.DetailGlobal('FSOD').cuda()
    model = regional.FeatureResidual(teacher).cuda()
    heads = nn.ModuleDict({'FSOD': teacher.head, 'Marine': base.new_head('Marine').cuda()})
    specs = p['source_input_definition']
    targets = {t: reader.make_targets(t, specs[t]['rows']['train']) for t in specs}
    masks = targets['Marine'].cuda()
    counts = torch.bincount(masks.long().flatten(), minlength=12)
    weights = counts.float().rsqrt()
    weights /= (weights*counts).sum()/counts.sum()
    heads['Marine'].loss_weights.copy_(weights)
    banks = {t: source_tokens(teacher.encoder, specs[t], deadline) for t in specs}
    plans = np.load(RUN/'source_plan.npz', allow_pickle=False)
    opt = torch.optim.AdamW([
        {'params': teacher.bypasses.parameters(), 'lr': .0001},
        {'params': model.bypasses.parameters(), 'lr': .0001},
        {'params': list(teacher.neck.parameters())+list(model.bridge.parameters())+list(model.residual.parameters()), 'lr': .001}],
        weight_decay=.0001)
    head_opts = {t: reader.head_optimizer(t, heads[t]) for t in specs}
    parameters = [q for group in opt.param_groups for q in group['params'] if q.requires_grad]
    parameters += [q for head in heads.values() for q in head.parameters()]
    ema = base.ParameterEMA(parameters)
    with (RUN/'source_training.jsonl').open('w') as log:
        for update in range(6000):
            guard(deadline)
            model.train()
            teacher.encoder.eval()
            heads.train()
            opt.zero_grad(set_to_none=True)
            for optimizer in head_opts.values():
                optimizer.zero_grad(set_to_none=True)
            for i, peak in enumerate((.0001, .0001, .001)):
                opt.param_groups[i]['lr'] = native.module_lr(update, 6000, peak)
            head_opts['FSOD'].param_groups[0]['lr'] = .005 if update < 4000 else (.0005 if update < 5500 else .00005)
            head_opts['Marine'].param_groups[0]['lr'] = native.module_lr(update, 6000, .001)
            losses = {}
            for task in ('FSOD', 'Marine'):
                ids = torch.from_numpy(plans[task][update]).cuda()
                actions = torch.from_numpy(plans[task+'_actions'][update]).cuda()
                tokens, local = banks[task]
                g = teacher.field(tokens[ids], teacher.bypasses)
                fine = model.fine_field(local[ids, actions], g, actions)
                corrected = model.corrected(g, fine, actions)
                target = reader.batch_targets(task, targets[task], ids.tolist(), 'cuda')
                loss = .25*(base.task_loss(task, heads[task], g, target)+base.task_loss(task, heads[task], corrected, target))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite shared source loss')
                loss.backward()
                losses[task] = float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            opt.step()
            for optimizer in head_opts.values():
                optimizer.step()
            ema.update()
            if update == 0:
                write(RUN/'first_source_update.json', {
                    'updates': 1, 'same_encoder_for_both_tasks': True,
                    'original_DINO_trainable': sum(q.numel() for q in teacher.encoder.parameters() if q.requires_grad),
                    'original_DINO_has_grad': any(q.grad is not None for q in teacher.encoder.parameters()),
                    'global_QKV_gradient_norm': float(sum(q.grad.square().sum() for q in teacher.bypasses.parameters() if q.grad is not None).sqrt()),
                    'gradient_norm_before_clip': float(norm), 'losses': losses})
            if (update+1) % 100 == 0:
                entry = {'updates': update+1, 'losses': losses, 'wall_seconds': time.monotonic()-begun, **guard(deadline)}
                log.write(json.dumps(entry)+'\n'); log.flush()
                print(json.dumps(entry), flush=True)
    ema.install()
    state = {name: value.detach().cpu().clone() for name, value in teacher.encoder.state_dict().items()}
    deltas = []
    for i, branch in enumerate(teacher.bypasses):
        delta = branch.up.weight.detach().cpu()@branch.down.weight.detach().cpu()
        state[f'backbone.blocks.{i}.attn.qkv.weight'].add_(delta)
        deltas.append(float(delta.norm()))
    write(RUN/'source_report.json', {'status': 'COMPLETE_SHARED_SOURCE', 'updates': 6000,
          'visits_per_source': 24000, 'encoder_views': 96000, 'original_DINO_updates': 0,
          'global_QKV_delta_norms': deltas, 'export_parameters': sum(v.numel() for v in state.values()),
          'export': 'Merged global QKV updates only; immutable CPU RAM snapshot, no trained weights on disk',
          'source_parameter_state': 'Terminal EMA.999', 'wall_seconds': time.monotonic()-begun, **guard(deadline)})
    return state


@torch.no_grad()
def evaluate(task, encoder, head, arrays, rows, targets, coco, batch, deadline):
    encoder.eval(); head.eval(); pieces = []
    for start in range(0, len(rows), batch):
        guard(deadline)
        features = reader.arm_features(encoder, arrays[start:start+batch], task, 'e2e')
        pieces.append(reader.decode(task, head, features, rows[start:start+batch]))
    pred = [row for part in pieces for row in part] if task == 'UATD' else np.concatenate(pieces)
    canonical, independent = reader.original_scores(task, pred, targets, coco)
    return canonical, independent, pred


def fit(task, arm, initial):
    p = protocol(); spec = p['downstream_tasks'][task]
    preparation = json.loads(Path(p['downstream_data']).read_text())
    data = preparation['inputs'][task]
    folder = RUN/'fits'/arm/task; folder.mkdir(parents=True)
    deadline = min(p['deadline_unix'], time.time()+p['budget']['fit_seconds'][task])
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    encoder = build_encoder(Path(p['encoder']['code_dir']), Path(p['encoder']['weights_path']), 'R010')
    if arm == 'shared_source_encoder':
        encoder.load_state_dict(initial, strict=True)
    reader.configure_encoder(encoder, 'e2e')
    encoder.cuda()
    head = reader.new_head(task, 0).cuda()
    body_opt, head_opt = reader.body_optimizer(encoder), reader.head_optimizer(task, head)
    arrays = {}
    for split in ('train', 'validation'):
        rows = data['rows'][split]
        size = 224 if task == 'UATD' else 448
        array = torch.empty((len(rows), 1, size, size), device='cuda')
        for start in range(0, len(rows), 16):
            array[start:start+16] = torch.stack([reader.normalized_input(row, data['normalization'], task) for row in rows[start:start+16]]).cuda()
            guard(deadline)
        arrays[split] = array
    targets = {s: reader.make_targets(task, rows) for s, rows in data['rows'].items()}
    if task == 'UATD':
        from sonar_encoder.training.uatd_head import box_targets
        training_targets = box_targets(targets['train'], 'cpu')
    else:
        training_targets = targets['train']
    coco = json.loads(Path(data['GT_paths']['validation']).read_text()) if task == 'FSOD' else None
    generator = torch.Generator().manual_seed(0)
    best, updates, begun = -1., 0, time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, spec['epochs']+1):
            reader.configure_encoder(encoder, 'e2e'); head.train()
            if task == 'FSOD':
                head_opt.param_groups[0]['lr'] = .005 if epoch <= 40 else (.0005 if epoch <= 55 else .00005)
            order = torch.randperm(len(data['rows']['train']), generator=generator)
            loss_sum = 0.
            for ids in order.split(spec['batch']):
                guard(deadline)
                body_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
                for group in body_opt.param_groups:
                    group['lr'] = reader.body_learning_rate(updates, spec['updates_per_fit'])
                features = reader.arm_features(encoder, arrays['train'][ids.cuda()], task, 'e2e')
                target = reader.batch_targets(task, training_targets, ids.tolist(), 'cuda')
                loss = reader.task_loss(task, head, features, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite ordinary downstream full-FT loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                body_opt.step(); head_opt.step(); updates += 1
                loss_sum += float(loss.detach())*len(ids)
                if updates == 1:
                    write(folder/'first_counted_update.json', {'updates': 1, 'body_gradient_norm_before_clip': float(norm),
                        'body_trainable_parameters': sum(q.numel() for q in encoder.parameters() if q.requires_grad),
                        'feature_shape': list(features.shape), 'body_learning_rate': body_opt.param_groups[0]['lr'],
                        'fresh_head_seed': 0, 'private_order_seed': 0, 'first_ids': ids.tolist(), 'new_weight_files': 0})
            entry = {'epoch': epoch, 'updates': updates, 'loss': loss_sum/len(order), 'wall_seconds': time.monotonic()-begun}
            if epoch % spec['validation_every_epochs'] == 0:
                metric, independent, prediction = evaluate(task, encoder, head, arrays['validation'], data['rows']['validation'],
                      targets['validation'], coco, spec['batch'], deadline)
                entry['metric'] = metric[METRIC[task]]
                payload = {'prediction': prediction, 'row_ids': preparation['row_ids'][task]['validation'],
                           'epoch': epoch, 'task': task, 'arm': arm}
                if metric[METRIC[task]] > best:
                    best, best_epoch, best_metrics = metric[METRIC[task]], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                if epoch == spec['epochs']:
                    torch.save(payload, folder/'terminal_predictions.pt')
                    terminal, terminal_independent = metric, independent
                print(json.dumps({'task': task, 'arm': arm, **entry}), flush=True)
            log.write(json.dumps(entry)+'\n'); log.flush()
    write(folder/'results.json', {'status': 'COMPLETE_FIXED_FULL_FINETUNING', 'task': task, 'arm': arm, 'seed': 0,
          'epochs': spec['epochs'], 'updates': updates, 'terminal': terminal, 'best': best_metrics, 'best_epoch': best_epoch,
          'terminal_independent': terminal_independent, 'scheduled_endpoints_only': True,
          'new_weight_files': 0, 'wall_seconds': time.monotonic()-begun, **guard(deadline)})


def worker(arm):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    initial = source_training() if arm == 'shared_source_encoder' else None
    gc.collect()
    for task in TASKS:
        fit(task, arm, initial)
        gc.collect()


def pipeline():
    p = json.loads((RUN/'protocol.json').read_text())
    started = p['started_unix']
    p.update(status='RESOURCE_SCHEDULE_REFROZEN_BEFORE_ANY_MODEL',
         resource_schedule_revision_unix=time.time(),
         resource_schedule_change='Start shared source on GPU2 after all three parent GPU2 actual waits0. Original baseline GPU3 still waits parent actual foreground0. Mathematical settings and original deadline unchanged.',
         superseded_CPU_wait={'supervisor_PID': 2834761, 'foreground_session': 47389, 'actual_exit': 143,
                              'reason': 'Intentional CPU-only scheduler replacement to use already released GPU2; zero modelworkers/forwards.'})
    write(RUN/'protocol.json', p)
    plans = {}
    for i, task in enumerate(('FSOD', 'Marine')):
        rng = np.random.RandomState(13+i)
        n = len(p['source_input_definition'][task]['rows']['train'])
        ids = np.concatenate([rng.permutation(n) for _ in range(math.ceil(24000/n))])[:24000]
        plans[task] = ids.reshape(6000, 4)
        plans[task+'_actions'] = np.random.RandomState(23+i).randint(0, 4, (6000, 4))
    np.savez(RUN/'source_plan.npz', **plans)
    active = {}
    (RUN/'logs').mkdir()
    waits = []
    try:
        for i, (arm, gpu) in enumerate((('shared_source_encoder', 2), ('original_DINO', 3))):
            while True:
                if gpu == 2:
                    done = json.loads((PARENT/'queue_state.json').read_text())['actual_waits']
                    ready = all(any(w['case'] == case and w['actual_wait_exit'] == 0 for w in done)
                                for case in ('isolated_UATD', 'isolated_FSOD', 'isolated_Marine'))
                else:
                    release = PARENT/'actual_foreground_wait.json'
                    ready = release.exists()
                    if ready and json.loads(release.read_text())['actual_wait_exit'] != 0:
                        raise RuntimeError('Parent recovery actual foreground exit nonzero')
                if ready:
                    break
                if time.time() > p['deadline_unix']:
                    raise TimeoutError('Shared-unit original deadline during resource wait')
                for name, (child, log, physical) in list(active.items()):
                    if child.poll() is not None:
                        code = child.wait(); log.close(); del active[name]
                        waits.append({'arm': name, 'PID': child.pid, 'physical_GPU': physical, 'actual_wait_exit': code})
                        if code != 0:
                            raise RuntimeError(name+' actual wait '+str(code))
                write(RUN/'queue_state.json', {'status': 'WAITING_GPU3_PARENT_RELEASE',
                    'active': {a: v[0].pid for a, v in active.items()}, 'actual_waits': waits})
                time.sleep(2)
            metadata = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            free, utilization = map(int, metadata.strip().split(','))
            if free < 24*1024:
                raise RuntimeError('GPU admission free<24GiB '+metadata)
            process_rows = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
            owned = [int(pid) for pid in process_rows if Path('/proc/'+pid).stat().st_uid == os.getuid()]
            if owned:
                raise RuntimeError('Existing same-UID GPU workers '+str(owned))
            write(RUN/f'admission_GPU{gpu}.json', {'physical_GPU': gpu, 'free_MiB': free,
                  'utilization_percent': utilization, 'compute_PIDs': process_rows, 'same_UID_GPU_PIDs': owned,
                  'idle_claim': False})
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            log = (RUN/'logs'/f'{arm}.log').open('w')
            child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'runs.gsr_shared_encoder_finetuning', arm],
                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            os.sched_setaffinity(child.pid, CPUS[i*4:i*4+4])
            active[arm] = (child, log, gpu)
            write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'workers': {a: v[0].pid for a, v in active.items()},
                  'physical_GPU_binding': {'shared_source_encoder': 2, 'original_DINO': 3}, 'new_weight_files': 0})
        while active:
            if time.time() > p['deadline_unix']:
                raise TimeoutError('Shared encoder 8h processing deadline')
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > p['budget']['new_disk_bytes']:
                raise RuntimeError('Shared encoder4GiB results cap')
            for arm, (child, log, gpu) in list(active.items()):
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del active[arm]
                waits.append({'arm': arm, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code})
                if code != 0:
                    raise RuntimeError(arm+' actual wait '+str(code))
            write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {a: v[0].pid for a, v in active.items()}, 'actual_waits': waits})
            time.sleep(2)
        results = {}
        for task in TASKS:
            arms = {arm: json.loads((RUN/'fits'/arm/task/'results.json').read_text()) for arm in ('original_DINO', 'shared_source_encoder')}
            results[task] = {'arms': arms, 'source_role': 'non-source' if task == 'UATD' else 'source reuse',
                'terminal_delta': arms['shared_source_encoder']['terminal'][METRIC[task]]-arms['original_DINO']['terminal'][METRIC[task]],
                'best_delta': arms['shared_source_encoder']['best'][METRIC[task]]-arms['original_DINO']['best'][METRIC[task]],
                'historical_strong_reference': p['references'][task]}
        write(RUN/'results.json', {'tasks': results, 'primary': 'Matched fullFT terminal', 'single_seed_pilot': True, 'new_weight_files': 0})
        write(RUN/'wave_closure.json', {'status': 'COMPLETE_FIXED_SCOPE', 'actual_waits': waits, 'wall_seconds': time.time()-started})
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
    parser.add_argument('phase', choices=('pipeline', 'shared_source_encoder', 'original_DINO'))
    args = parser.parse_args()
    if args.phase == 'pipeline':
        pipeline()
    else:
        worker(args.phase)
