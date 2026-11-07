"""Official SigLIP2 Base/16 direct full fine-tuning on six fixed sonar tasks."""
import argparse
import json
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
from runs.siglip2_frozen_components import SiglipDetectorHead, load_frozen_vision, make_processor, process_grayscale


RUN = core.BASE / 'gsr_siglip2_fullft_v1'
ASSETS = core.BASE / 'siglip2_frozen_baseline_v1' / 'model_assets'
TASKS = ('FSOD', 'Marine', 'UATD', 'NKSID', 'HighResolution', 'Diver')
CAPS = {'FSOD': 14400, 'Marine': 7200, 'UATD': 18000,
        'NKSID': 5400, 'HighResolution': 5400, 'Diver': 5400}
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
        raise TimeoutError('SigLIP2 direct fullFT job deadline')
    if observed['RSS_bytes'] > 64 * 1024**3 or observed['GPU_reserved_bytes'] > 48 * 1024**3:
        raise RuntimeError('SigLIP2 fullFT resource cap ' + str(observed))
    return observed


class DirectSigLIP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision, self.loading_info = load_frozen_vision(ASSETS)
        self.vision.requires_grad_(True)
        self.vision.vision_model.head.requires_grad_(False)
        self.encoder_images = 0
        self.encoder_calls = 0

    def field(self, images, head, kind, policy='global', actions=None):
        self.encoder_images += len(images)
        self.encoder_calls += 1
        patches = self.vision(pixel_values=images).last_hidden_state
        native = patches.transpose(1, 2).reshape(len(images), 768, 14, 14)
        if kind == 'detection':
            return native
        return F.interpolate(native, (32, 32), mode='bilinear', align_corners=False)


def new_head(task, spec, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if spec['kind'] == 'detection':
            head = SiglipDetectorHead()
            if spec['classes'] != 10:
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                head.roi_heads.box_predictor = FastRCNNPredictor(1024, spec['classes'] + 1)
            return head
        if spec['kind'] == 'classification':
            return nn.Linear(768, spec['classes'])
        return nn.Conv2d(768, 1 if spec['kind'] == 'point_regression' else
                         50 if task == 'UATD' else spec['classes'], 1)


def images_for(spec, split, processor):
    chunks = []
    rows = spec['rows'][split]
    for start in range(0, len(rows), 16):
        opened = [Image.open(row['image']) for row in rows[start:start + 16]]
        try:
            chunks.append(process_grayscale(processor, opened))
        finally:
            for image in opened:
                image.close()
    return torch.cat(chunks).cuda()


@torch.no_grad()
def evaluate(task, spec, model, head, images, labels, batch, deadline, split='validation'):
    model.eval()
    head.eval()
    rows = spec['rows'][split]
    pieces = []
    for start in range(0, len(rows), batch):
        end = min(start + batch, len(rows))
        field = model.field(images[start:end], head, spec['kind'])
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
                pieces.append(F.interpolate(logits[i:i + 1], tuple(target.shape),
                    mode='bilinear', align_corners=False).argmax(1)[0].byte().cpu().numpy())
        guard(deadline)
    prediction = pieces if task == 'UATD' or spec['kind'] == 'segmentation' else np.concatenate(pieces)
    score, independent = core.scores(task, spec, prediction, labels, split)
    return score, independent, prediction


def fit(task, seed, protocol, folder):
    spec = json.loads(Path(protocol['data_manifest']).read_text())['tasks'][task]
    schedule = protocol['target_schedules'][task]
    processor = make_processor(ASSETS)
    images = {split: images_for(spec, split, processor) for split in ('train', 'validation')}
    labels = {split: core.targets_for(task, spec, split) for split in images}
    model = DirectSigLIP2().cuda()
    head = new_head(task, spec, seed).cuda()
    body = core.reader.body_optimizer(model.vision)
    head_opt = core.reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    generator = torch.Generator().manual_seed(seed)
    direction = -1 if task == 'Diver' else 1
    metric_key = METRICS[task]
    best = -float('inf')
    updates = 0
    began = time.monotonic()
    deadline = time.time() + CAPS[task] + 1800
    write(folder / 'initialization.json', {
        'asset': str(ASSETS / 'model.safetensors'),
        'model_class': 'transformers.SiglipVisionModel',
        'loading_missing_keys': model.loading_info['missing_keys'],
        'loading_mismatched_keys': model.loading_info['mismatched_keys'],
        'loading_error_msgs': model.loading_info['error_msgs'],
        'vision_trainable_parameters': sum(p.numel() for p in model.vision.parameters() if p.requires_grad),
        'pretrained_MAP_pooler_parameters_frozen_unused': sum(p.numel() for p in model.vision.vision_model.head.parameters()),
        'head_parameters': sum(p.numel() for p in head.parameters()),
        'preprocessing': 'Official SiglipImageProcessor, grayscale repeated RGB, fixed224, rescale1/255, normalize .5/.5',
        'native_patch_field': [768, 14, 14], 'non_detection_readout_field': [768, 32, 32]})
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
                target = core.batch_targets(task, spec, labels['train'], ids.tolist())
                loss = core.task_loss(task, spec, head, field, target)
                loss.backward()
                body_norm = torch.nn.utils.clip_grad_norm_(model.vision.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                body.step()
                head_opt.step()
                updates += 1
                total += float(loss.detach()) * len(ids)
                if updates == 1:
                    write(folder / 'first_counted_update.json', {'updates': 1,
                          'first_ids': ids.tolist(), 'field': list(field.shape),
                          'body_gradient_norm': float(body_norm),
                          'vision_trainable': True, 'ordinary_task_loss_only': True})
            row = {'epoch': epoch, 'updates': updates, 'loss': total / len(order),
                   'wall_seconds': time.monotonic() - began}
            if epoch % protocol['evaluation_every_epochs'][task] == 0:
                score, independent, pred = evaluate(task, spec, model, head,
                    images['validation'], labels['validation'], schedule['batch'], deadline)
                row[metric_key] = score[metric_key]
                payload = {'prediction': pred, 'task': task, 'epoch': epoch,
                           'split': 'validation', 'encoder': 'SigLIP2 Base/16'}
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
        held_images = images_for(spec, held_split, processor)
        held_labels = core.targets_for(task, spec, held_split)
        held, held_independent, held_pred = evaluate(task, spec, model, head,
            held_images, held_labels, schedule['batch'], deadline, held_split)
        torch.save({'prediction': held_pred, 'task': task, 'split': held_split,
                    'epoch': schedule['epochs']}, folder / 'heldout_terminal_predictions.pt')
        write(folder / 'heldout_independent.json', held_independent)
    write(folder / 'results.json', {'status': 'COMPLETE_SIGLIP2_DIRECT_FULLFT',
          'task': task, 'seed': seed, 'metric': metric_key,
          'epochs': schedule['epochs'], 'updates': updates,
          'terminal': terminal, 'terminal_independent': terminal_independent,
          'best': best_score, 'best_epoch': best_epoch, 'best_selection_split': 'validation',
          'heldout_terminal_only': held, 'heldout_split': held_split,
          'wall_seconds': time.monotonic() - began})


def worker(name):
    protocol = json.loads((RUN / 'protocol.json').read_text())
    job = json.loads((RUN / name / 'job.json').read_text())
    seed = job['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    core.original.runtime()
    try:
        fit(job['task'], seed, protocol, RUN / name)
    except BaseException:
        write(RUN / name / 'failure.json', {'traceback': traceback.format_exc(),
              'time_unix': time.time(), 'no_retry': True})
        raise


def pipeline():
    RUN.mkdir(exist_ok=True)
    base = json.loads((core.BASE / 'gsr_attention6_memory_v1' / 'protocol.json').read_text())
    jobs = [{'id': f's{s}_{task}', 'seed': s, 'task': task}
            for task in TASKS for s in (0, 1, 2)]
    protocol = {key: base[key] for key in ('data_manifest', 'target_schedules', 'evaluation_every_epochs')}
    protocol.update({'entry': str(Path(__file__).resolve()), 'jobs': jobs,
       'official_initialization': 'google/siglip2-base-patch16-224 revision75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2; existing pinned asset directory',
       'asset_path': str(ASSETS / 'model.safetensors'),
       'preprocessing': 'Official SiglipImageProcessor, grayscale RGB; no DINO log normalization. Native14x14 for FSOD detection; bilinear32x32 for other task readers.',
       'training': 'Full spatial vision backbone and fresh task head. The pretrained MAP pooler has no role in patch-field task outputs and is frozen/unused. DINO validated AdamW2e-5 body schedule warmup200/cosine; original target head optimizer, losses, splits, batches, epochs, validation cadence/seeds0/1/2.',
       'scope': 'Independent pretrained-family full-path comparator, not same-parameter, same-pretraining-data, or same-preprocessing ablation. Best is development validation; heldout terminal only where available.'})
    write(RUN / 'protocol.json', protocol)
    for job in jobs:
        (RUN / job['id']).mkdir(exist_ok=True)
    pending = list(jobs)
    active = {}
    waits = []
    started = time.time()
    dependency = core.BASE / 'gsr_same_source_dino_v1' / 'completion.json'
    dependency_failure = core.BASE / 'gsr_same_source_dino_v1' / 'failure.json'
    try:
        while pending or active:
            if dependency_failure.is_file():
                raise RuntimeError(f'Prerequisite same-source queue failed: {dependency_failure}')
            for name, (child, log, gpu, began, limit) in list(active.items()):
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
            if dependency.is_file() and json.loads(dependency.read_text())['status'] == 'COMPLETE':
                for gpu in (2, 3):
                    if not pending or any(row[2] == gpu for row in active.values()):
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
                    job = pending.pop(0)
                    name, task = job['id'], job['task']
                    cap = CAPS[task] + 1800
                    folder = RUN / name
                    write(folder / 'job.json', {**job, 'GPU': gpu,
                          'admission_free_MiB': free, 'admission_utilization': util,
                          'deadline_unix': time.time() + cap})
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    log = (folder / 'stdout.log').open('w')
                    child = subprocess.Popen([core.original.PYTHON, '-B', '-u', '-m',
                        'runs.gsr_siglip2_fullft', 'worker', name], cwd=core.original.ROOT,
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[name] = (child, log, gpu, time.time(), time.time() + cap)
            write(RUN / 'queue_state.json', {'status': 'RUNNING' if active else 'WAITING_RESOURCE',
                  'dependency': str(dependency),
                  'active': {name: {'PID': row[0].pid, 'GPU': row[2]} for name, row in active.items()},
                  'pending': [job['id'] for job in pending], 'completed': len(waits),
                  'supervisor_PID': os.getpid()})
            time.sleep(5)
        write(RUN / 'completion.json', {'status': 'COMPLETE', 'actual_waits': waits,
              'wall_seconds': time.time() - started})
    except BaseException:
        write(RUN / 'failure.json', {'traceback': traceback.format_exc(),
              'time_unix': time.time(), 'no_retry': True})
        write(RUN / 'queue_state.json', {'status': 'FAILED', 'dependency': str(dependency),
              'active': {}, 'pending': [job['id'] for job in pending],
              'completed': len(waits), 'supervisor_PID': os.getpid()})
        raise
    finally:
        for name, (child, log, gpu, began, limit) in active.items():
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
