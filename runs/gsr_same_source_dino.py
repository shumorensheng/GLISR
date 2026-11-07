"""Source-data-matched DINOv2 control for the Attention6 comparison.

The DINO backbone and temporary source heads learn from the same six source
plans as Attention6. Source heads are discarded; each target starts with a
fresh head and its seed's source-trained backbone.
"""
import argparse
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
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from runs import gsr_multidata_training as core
from sonar_encoder.training.model import build_encoder


RUN = core.BASE / 'gsr_same_source_dino_v1'
SOURCE = core.SOURCE
TASKS = ('FSOD', 'Marine', 'UATD', 'NKSID', 'HighResolution', 'Diver')
CAPS = {'FSOD': 10800, 'Marine': 5400, 'UATD': 14400,
        'NKSID': 3600, 'HighResolution': 3600, 'Diver': 3600}
METRICS = {'FSOD': 'AP50', 'Marine': 'miou', 'UATD': 'map50',
           'NKSID': 'macro_f1', 'HighResolution': 'miou', 'Diver': 'mean_error_pixels'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def guard(deadline):
    observed = {'RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > deadline:
        raise TimeoutError('same-source DINO job deadline')
    if observed['RSS_bytes'] > 64 * 1024**3 or observed['GPU_reserved_bytes'] > 40 * 1024**3:
        raise RuntimeError('same-source DINO resource cap ' + str(observed))
    return observed


class DirectDINO(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
        self.encoder_images = 0
        self.encoder_calls = 0

    def field(self, images, head, kind, policy='global', actions=None):
        self.encoder_images += len(images)
        self.encoder_calls += 1
        patches = self.encoder(images)['patch_tokens']
        grid = patches.transpose(1, 2).reshape(len(images), 384, 16, 16)
        return F.interpolate(grid, (32, 32), mode='bilinear', align_corners=False)


def new_head(task, spec, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if task in ('FSOD', 'Marine', 'UATD', 'NKSID'):
            return core.reader.new_head(task, seed)
        if spec['kind'] == 'detection':
            head = core.reader.new_head('FSOD', seed)
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            head.roi_heads.box_predictor = FastRCNNPredictor(1024, spec['classes'] + 1)
            return head
        return nn.Conv2d(384, 1 if spec['kind'] == 'point_regression' else spec['classes'], 1)


def source(seed, protocol, folder):
    tasks = json.loads(Path(protocol['data_manifest']).read_text())['tasks']
    model = DirectDINO(protocol['encoder']).cuda()
    core.reader.configure_encoder(model.encoder, 'e2e')
    body = core.reader.body_optimizer(model.encoder)
    images = {t: torch.from_numpy(np.load(tasks[t]['views']['train'], mmap_mode='r', allow_pickle=False)).cuda() for t in SOURCE}
    labels = {t: core.targets_for(t, tasks[t], 'train') for t in SOURCE}
    heads = nn.ModuleDict({t: new_head(t, tasks[t], seed).cuda() for t in SOURCE})
    head_opts = {t: core.reader.head_optimizer('FSOD' if tasks[t]['kind'] == 'detection' else 'Marine', heads[t]) for t in SOURCE}
    plans = np.load(folder / 'source_plan.npz', allow_pickle=False)
    deadline = time.time() + 21600
    began = time.monotonic()
    with (folder / 'training.jsonl').open('w') as log:
        for update in range(6000):
            model.train()
            body.zero_grad(set_to_none=True)
            for group in body.param_groups:
                group['lr'] = core.reader.body_learning_rate(update, 6000)
            losses = {}
            for task in SOURCE:
                head = heads[task]
                head.train()
                opt = head_opts[task]
                opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = ((.005 if update < 4000 else .0005 if update < 5500 else .00005)
                                             if tasks[task]['kind'] == 'detection' else
                                             core.previous.shared.native.module_lr(update, 6000, .001))
                ids = plans[task][update]
                task_value = 0.
                for start in range(0, len(ids), 4):
                    indices = ids[start:start + 4].tolist()
                    field = model.field(images[task][indices], head, tasks[task]['kind'])
                    target = core.batch_targets(task, tasks[task], labels[task], indices)
                    loss = core.task_loss(task, tasks[task], head, field, target) * len(indices) / len(ids) / len(SOURCE)
                    loss.backward()
                    task_value += float(loss.detach())
                losses[task] = task_value
                opt.step()
            body_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
            body.step()
            if update == 0:
                write(folder / 'first_counted_update.json', {
                    'updates': 1, 'source_tasks': SOURCE, 'source_macro_per_task': protocol['source_macro_per_task'],
                    'DINO_trainable_parameters': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                    'body_gradient_norm': float(body_norm), 'field': list(field.shape),
                    'encoder_images': model.encoder_images, 'source_heads_discarded_after_training': True})
            if (update + 1) % 100 == 0:
                row = {'updates': update + 1, 'losses': losses, 'wall_seconds': time.monotonic() - began,
                       'encoder_images': model.encoder_images, **guard(deadline)}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
            guard(deadline)
    checkpoint = folder / 'source_encoder.pt'
    torch.save({'encoder': model.encoder.cpu().state_dict(), 'seed': seed,
                'source_updates': 6000, 'source_tasks': SOURCE}, checkpoint)
    write(folder / 'results.json', {'status': 'COMPLETE_SOURCE_BACKBONE6000', 'seed': seed,
          'updates': 6000, 'source_visits': protocol['source_visits'],
          'encoder_images': model.encoder_images, 'source_heads_exported': False,
          'checkpoint': str(checkpoint), 'wall_seconds': time.monotonic() - began})


def target(task, seed, protocol, folder):
    spec = json.loads(Path(protocol['data_manifest']).read_text())['tasks'][task]
    schedule = protocol['target_schedules'][task]
    source_checkpoint = RUN / f's{seed}_source' / 'source_encoder.pt'
    model = DirectDINO(protocol['encoder'])
    model.encoder.load_state_dict(torch.load(source_checkpoint, map_location='cpu', weights_only=True)['encoder'], strict=True)
    core.reader.configure_encoder(model.encoder, 'e2e')
    model.cuda()
    head = new_head(task, spec, seed).cuda()
    body = core.reader.body_optimizer(model.encoder)
    head_opt = core.reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r', allow_pickle=False)).cuda()
              for s in ('train', 'validation')}
    labels = {s: core.targets_for(task, spec, s) for s in images}
    generator = torch.Generator().manual_seed(seed)
    metric_key = METRICS[task]
    direction = -1 if task == 'Diver' else 1
    best = -float('inf')
    updates = 0
    began = time.monotonic()
    deadline = time.time() + CAPS[task] + 1800
    with (folder / 'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs'] + 1):
            model.train()
            head.train()
            order = torch.randperm(len(spec['rows']['train']), generator=generator)
            if spec['kind'] == 'detection':
                head_opt.param_groups[0]['lr'] = .005 * .1 ** sum(epoch > m for m in (40, 55))
            total = 0.
            for ids in order.split(schedule['batch']):
                guard(deadline)
                body.zero_grad(set_to_none=True)
                head_opt.zero_grad(set_to_none=True)
                for group in body.param_groups:
                    group['lr'] = core.reader.body_learning_rate(updates, schedule['updates'])
                field = model.field(images['train'][ids.cuda()], head, spec['kind'])
                batch = core.batch_targets(task, spec, labels['train'], ids.tolist())
                loss = core.task_loss(task, spec, head, field, batch)
                loss.backward()
                body_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                body.step()
                head_opt.step()
                updates += 1
                total += float(loss.detach()) * len(ids)
                if updates == 1:
                    write(folder / 'first_counted_update.json', {'updates': 1, 'first_ids': ids.tolist(),
                          'field': list(field.shape), 'body_gradient_norm': float(body_norm),
                          'source_checkpoint': str(source_checkpoint), 'ordinary_task_loss_only': True})
            row = {'epoch': epoch, 'updates': updates, 'loss': total / len(order),
                   'wall_seconds': time.monotonic() - began}
            if epoch % protocol['evaluation_every_epochs'][task] == 0:
                score, independent, prediction = core.evaluate(task, spec, model, head,
                    images['validation'], labels['validation'], schedule['batch'], deadline, 'global')
                row[metric_key] = score[metric_key]
                payload = {'prediction': prediction, 'task': task, 'epoch': epoch, 'policy': 'global', 'split': 'validation'}
                if direction * score[metric_key] > best:
                    best = direction * score[metric_key]
                    best_epoch, best_score = epoch, score
                    torch.save(payload, folder / 'best_predictions.pt')
                    weight_payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'head': {k: v.detach().cpu() for k, v in head.state_dict().items()}, 'task': task, 'seed': seed, 'epoch': epoch, 'endpoint': 'best', 'selection_split': 'validation', 'protocol_path': str(RUN / 'protocol.json'), 'data_manifest': protocol['data_manifest']}
                    torch.save(weight_payload, folder / 'best_weights.tmp')
                    (folder / 'best_weights.tmp').replace(folder / 'best_weights.pt')
                    del weight_payload
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = score, independent
                    torch.save(payload, folder / 'terminal_predictions.pt')
                    weight_payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'head': {k: v.detach().cpu() for k, v in head.state_dict().items()}, 'task': task, 'seed': seed, 'epoch': epoch, 'endpoint': 'terminal', 'selection_split': 'validation', 'protocol_path': str(RUN / 'protocol.json'), 'data_manifest': protocol['data_manifest']}
                    torch.save(weight_payload, folder / 'terminal_weights.tmp')
                    (folder / 'terminal_weights.tmp').replace(folder / 'terminal_weights.pt')
                    del weight_payload
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
    held_split = 'official_validation' if task == 'HighResolution' else 'test' if task == 'Diver' else None
    held = None
    if held_split:
        held_images = []
        for row in spec['rows'][held_split]:
            with Image.open(row['image']) as image:
                intensity = np.log1p(np.array(image.convert('L'), dtype=np.uint8).astype(np.float32) / 255.)
            x = F.interpolate(torch.from_numpy(intensity)[None, None], (224, 224),
                              mode='bilinear', align_corners=False, antialias=True)
            held_images.append(((x[0] - spec['normalization']['mean']) / spec['normalization']['std']).cuda())
        held_labels = core.targets_for(task, spec, held_split)
        held, held_independent, held_prediction = core.evaluate(task, spec, model, head,
            torch.stack(held_images), held_labels, schedule['batch'], deadline, 'global', held_split)
        torch.save({'prediction': held_prediction, 'task': task, 'split': held_split,
                    'epoch': schedule['epochs']}, folder / 'heldout_terminal_predictions.pt')
        write(folder / 'heldout_independent.json', held_independent)
    write(folder / 'results.json', {'status': 'COMPLETE_SAME_SOURCE_DINO_FULLFT', 'task': task,
          'seed': seed, 'metric': metric_key, 'epochs': schedule['epochs'], 'updates': updates,
          'terminal': terminal, 'terminal_independent': terminal_independent,
          'best': best_score, 'best_epoch': best_epoch, 'best_selection_split': 'validation',
          'heldout_terminal_only': held, 'heldout_split': held_split,
          'source_checkpoint': str(source_checkpoint), 'wall_seconds': time.monotonic() - began})


def worker(job_id):
    protocol = json.loads((RUN / 'protocol.json').read_text())
    job = json.loads((RUN / job_id / 'job.json').read_text())
    seed = job['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    core.original.runtime()
    try:
        if job['task'] is None:
            source(seed, protocol, RUN / job_id)
        else:
            target(job['task'], seed, protocol, RUN / job_id)
    except BaseException:
        write(RUN / job_id / 'failure.json', {'traceback': traceback.format_exc(),
              'time_unix': time.time(), 'no_retry': True})
        raise


def pipeline():
    RUN.mkdir(exist_ok=True)
    old = json.loads((core.BASE / 'gsr_attention6_memory_v1' / 'protocol.json').read_text())
    jobs = [{'id': f's{s}_source', 'seed': s, 'task': None} for s in (0, 1, 2)]
    jobs += [{'id': f's{s}_{t}', 'seed': s, 'task': t} for t in TASKS for s in (0, 1, 2)]
    protocol = {key: old[key] for key in ('encoder', 'data_manifest', 'target_schedules',
                                         'evaluation_every_epochs', 'source_macro_per_task', 'source_visits')}
    protocol.update({'entry': str(Path(__file__).resolve()), 'jobs': jobs,
       'source_updates_per_seed': 6000, 'source_training': 'Six original source tasks, original sample-ID plan, microbatch4; DINO backbone plus temporary task heads train on ordinary task losses. Source heads discarded.',
       'source_optimizer': 'Validated downstream DINO AdamW decay.05/nondecay0, LR2e-5 warmup200 cosine to1pct, clip1; source head FSOD SGD.005 milestones4000/5500, other heads AdamW.001 cosine.',
       'target_training': 'Same six task splits, normalization, 224 input, 32x32 bilinear field, original heads/losses, schedules, validation cadence and target seeds0/1/2 as Attention6. Terminal primary; best selected on development validation only.',
       'comparison_scope': 'Source-data-matched control, not parameter-matched: GLISR source freezes original DINO and trains fusion/memory; this arm source-trains DINO backbone. No target/head transfer or test tuning.',
       'source_checkpoints_on_data_disk': True, 'heldout': 'HighResolution official_validation and Diver test terminal only; all other validation results are development-population results.'})
    write(RUN / 'protocol.json', protocol)
    data = json.loads(Path(protocol['data_manifest']).read_text())['tasks']
    for job in jobs:
        folder = RUN / job['id']
        folder.mkdir(exist_ok=True)
        if job['task'] is None:
            plans = {}
            for i, task in enumerate(SOURCE):
                n = len(data[task]['rows']['train'])
                macro = protocol['source_macro_per_task'][task]
                rng = np.random.RandomState(13 + i + 1000 * job['seed'])
                plans[task] = np.concatenate([rng.permutation(n) for _ in range(math.ceil(6000 * macro / n))])[:6000 * macro].reshape(6000, macro)
            np.savez(folder / 'source_plan.npz', **plans)
    pending = list(jobs)
    active = {}
    done = set()
    waits = []
    launched = 0
    started = time.time()
    ablation_completion = core.BASE / 'gsr_attention6_ablations_indexfix_resume_v1' / 'completion.json'
    ablation_failure = core.BASE / 'gsr_attention6_ablations_indexfix_resume_v1' / 'failure.json'
    try:
        while pending or active:
            if ablation_failure.is_file():
                raise RuntimeError(f'Prerequisite ablation queue failed: {ablation_failure}')
            for name, (child, log, gpu, began, limit, task, seed) in list(active.items()):
                if time.time() > limit and child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait()
                log.close()
                del active[name]
                waits.append({'job': name, 'PID': child.pid, 'GPU': gpu,
                              'actual_wait_exit': code, 'wall_seconds': time.time() - began})
                write(RUN / 'actual_waits.json', waits)
                if code:
                    raise RuntimeError(f'{name} exited {code}; no retry')
                if task is None:
                    done.add(seed)
            for gpu in (2, 3):
                if not ablation_completion.is_file():
                    continue
                if any(row[2] == gpu for row in active.values()):
                    continue
                eligible = next((job for job in pending if job['task'] is None or job['seed'] in done), None)
                if eligible is None:
                    continue
                free, util = map(int, subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                    text=True).strip().split(','))
                if free < 70000 or util > 10:
                    continue
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) * 1024
                if available < 128 * 1024**3:
                    continue
                pending.remove(eligible)
                name = eligible['id']
                task = eligible['task']
                cap = 21600 if task is None else CAPS[task] + 1800
                folder = RUN / name
                write(folder / 'job.json', {**eligible, 'GPU': gpu,
                      'admission_free_MiB': free, 'admission_utilization': util,
                      'deadline_unix': time.time() + cap})
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log = (folder / 'stdout.log').open('w')
                child = subprocess.Popen([core.original.PYTHON, '-B', '-u', '-m',
                    'runs.gsr_same_source_dino', 'worker', name], cwd=core.original.ROOT,
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[name] = (child, log, gpu, time.time(), time.time() + cap, task, eligible['seed'])
                launched += 1
            write(RUN / 'queue_state.json', {'status': 'RUNNING' if active else 'WAITING_RESOURCE',
                  'active': {name: {'PID': row[0].pid, 'GPU': row[2]} for name, row in active.items()},
                  'pending': [job['id'] for job in pending], 'completed': len(waits),
                  'source_done': sorted(done), 'supervisor_PID': os.getpid(),
                  'ablation_dependency': str(ablation_completion)})
            time.sleep(5)
        write(RUN / 'completion.json', {'status': 'COMPLETE', 'actual_waits': waits,
              'wall_seconds': time.time() - started})
        write(RUN / 'queue_state.json', {'status': 'COMPLETE', 'active': {}, 'pending': [],
              'completed': len(waits), 'source_done': sorted(done), 'supervisor_PID': os.getpid()})
    except BaseException:
        write(RUN / 'failure.json', {'traceback': traceback.format_exc(),
              'time_unix': time.time(), 'no_retry': True})
        write(RUN / 'queue_state.json', {'status': 'FAILED', 'active': {},
              'pending': [job['id'] for job in pending], 'completed': len(waits),
              'source_done': sorted(done), 'supervisor_PID': os.getpid()})
        raise
    finally:
        for name, (child, log, gpu, began, limit, task, seed) in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
            log.close()
            waits.append({'job': name, 'PID': child.pid, 'GPU': gpu,
                          'actual_wait_exit': code, 'cleanup': True})
        write(RUN / 'actual_waits.json', waits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pipeline', 'worker'))
    parser.add_argument('job', nargs='?')
    args = parser.parse_args()
    if args.phase == 'pipeline':
        pipeline()
    else:
        worker(args.job)
