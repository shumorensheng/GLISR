"""One bounded downstream routing adaptation from the existing shared GSR in RAM."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from runs import gsr_multidata_training as core
from runs import gsr_multidata_runtime_v2 as parent


RUN = core.BASE/'gsr_online_gain_routing_v1'
TASKS = ('FSOD', 'Marine', 'UATD', 'NKSID')
CAPS = {'FSOD': 9000, 'Marine': 2400, 'UATD': 6500, 'NKSID': 1800}
DEADLINE = 0.
write = core.write


def guard(deadline):
    values = {'RSS_bytes': core.resource.getrusage(core.resource.RUSAGE_SELF).ru_maxrss*1024,
              'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > min(deadline, DEADLINE):
        raise TimeoutError('Online routing fixed deadline')
    if values['RSS_bytes'] > 48*1024**3 or values['GPU_reserved_bytes'] > 32*1024**3:
        raise RuntimeError('Online routing resource cap '+str(values))
    return values


def gpu_admission(gpu, known):
    total, free, utilization = map(int, subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    unknown = []
    for pid in subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).splitlines():
        try:
            if Path('/proc/'+pid).stat().st_uid == os.getuid() and int(pid) not in known:
                unknown.append(int(pid))
        except FileNotFoundError:
            continue
    return {'free_MiB': free, 'utilization_percent': utilization, 'unknown_same_UID': unknown,
            'eligible': total >= 76*1024 and free >= 40*1024 and not unknown, 'idle_claim': False}


class OnlineGSR(core.GSR):
    def training_fields(self, images, generator):
        g = self.global_field(images)
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        probability = .5*self.selector(g.detach()).softmax(1)+.125
        actions = torch.multinomial(probability.detach(), 1, generator=generator).flatten()
        log_probability = probability.gather(1, actions[:, None]).log().sum()
        large = F.interpolate(images, (448, 448), mode='bilinear', align_corners=False, antialias=True)
        tiles = torch.stack([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in core.original.QUADS], 1)
        local = self.global_field(tiles[torch.arange(len(images), device=images.device), actions])
        return base, self.corrected(g, local, actions), log_probability, actions


def fit(task, protocol, initial):
    spec = json.loads(Path(protocol['data_manifest']).read_text())['tasks'][task]
    schedule = protocol['target_schedules'][task]
    folder = RUN/task
    model = OnlineGSR(protocol['encoder'])
    model.load_state_dict(initial, strict=True)
    core.reader.configure_encoder(model.encoder, 'e2e')
    model.fusion.requires_grad_(True); model.selector.requires_grad_(True); model.cuda()
    head = core.new_head(task, spec).cuda()
    body_opt = core.reader.body_optimizer(model.encoder)
    head_opt = core.reader.head_optimizer('FSOD' if task == 'FSOD' else 'Marine', head)
    fusion_opt = torch.optim.AdamW(model.fusion.parameters(), lr=.0001, weight_decay=.0001)
    route_opt = torch.optim.AdamW(model.selector.parameters(), lr=.0001, weight_decay=.0001)
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r', allow_pickle=False)).cuda()
              for s in ('train', 'validation')}
    labels = {s: core.targets_for(task, spec, s) for s in images}
    generator = torch.Generator().manual_seed(0)
    routing = torch.Generator(device='cuda').manual_seed(700)
    key = core.metric_key(task, spec)
    best = -float('inf'); updates = 0; extra_head_examples = 0; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs']+1):
            model.train(); head.train()
            if task == 'FSOD':
                head_opt.param_groups[0]['lr'] = .005*.1**sum(epoch > m for m in (40, 55))
            order = torch.randperm(len(spec['rows']['train']), generator=generator)
            total = total_gain = total_route_loss = 0.
            action_counts = np.zeros(4, dtype=np.int64)
            for ids in order.split(schedule['batch']):
                guard(DEADLINE)
                for optimizer in (body_opt, head_opt, fusion_opt, route_opt):
                    optimizer.zero_grad(set_to_none=True)
                for group in body_opt.param_groups:
                    group['lr'] = core.reader.body_learning_rate(updates, schedule['updates'])
                for optimizer in (fusion_opt, route_opt):
                    optimizer.param_groups[0]['lr'] = core.previous.shared.native.module_lr(updates, schedule['updates'], .0001)
                base, field, log_probability, actions = model.training_fields(images['train'][ids.cuda()], routing)
                target = core.batch_targets(task, spec, labels['train'], ids.tolist())
                # The global baseline is action-independent; head RNG is restored before the actual task loss.
                with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                    global_loss = core.task_loss(task, spec, head, base.detach(), target)
                loss = core.task_loss(task, spec, head, field, target)
                gain = global_loss-loss.detach()
                # A batch loss uses the sum of log probabilities, not their mean.
                route_loss = -gain*log_probability
                (loss+route_loss).backward()
                body_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
                route_norm = torch.nn.utils.clip_grad_norm_(model.selector.parameters(), 1., error_if_nonfinite=True)
                for optimizer in (body_opt, head_opt, fusion_opt, route_opt):
                    optimizer.step()
                updates += 1; extra_head_examples += len(ids)
                total += float(loss.detach())*len(ids); total_gain += float(gain)*len(ids)
                total_route_loss += float(route_loss.detach())*len(ids)
                action_counts += torch.bincount(actions, minlength=4).cpu().numpy()
                if updates == 1:
                    write(folder/'first_counted_update.json', {'updates': 1, 'first_ids': ids.tolist(),
                        'actions': actions.tolist(), 'global_loss': float(global_loss), 'task_loss': float(loss.detach()),
                        'gain': float(gain), 'policy_loss': float(route_loss.detach()),
                        'body_gradient_norm': float(body_norm), 'selector_gradient_norm': float(route_norm),
                        'source_cohort': protocol['source_cohort'], 'encoder_images': model.encoder_images,
                        'selector_input_detached': True, 'global_head_rng_scope': 'torch.random.fork_rng(devices=[0])'})
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order),
                'mean_gain': total_gain/len(order), 'mean_policy_loss': total_route_loss/len(order),
                'sampled_region_counts': action_counts.tolist(), 'wall_seconds': time.monotonic()-begun}
            every = 5 if task == 'FSOD' else 1
            if epoch % every == 0:
                metric, independent, pred = core.evaluate(task, spec, model, head, images['validation'],
                    labels['validation'], schedule['batch'], DEADLINE, 'learned')
                entry[key] = metric[key]
                payload = {'prediction': pred, 'epoch': epoch, 'task': task, 'arm': 'online_gain_GSR', 'policy': 'learned'}
                if metric[key] > best:
                    best, best_epoch, best_metric = metric[key], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = metric, independent
                    torch.save(payload, folder/'terminal_predictions.pt')
            log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)
    policies = {'learned': terminal}
    for policy in ('global', 'random', 'fixed0', 'fixed1', 'fixed2', 'fixed3', 'uncertainty'):
        metric, independent, pred = core.evaluate(task, spec, model, head, images['validation'],
            labels['validation'], schedule['batch'], DEADLINE, policy)
        policies[policy] = metric
        torch.save({'prediction': pred, 'task': task, 'epoch': schedule['epochs'], 'policy': policy},
                   folder/(policy+'_terminal_predictions.pt'))
        write(folder/(policy+'_independent.json'), independent)
    write(folder/'results.json', {'status': 'COMPLETE_FIXED_ONLINE_GAIN_ROUTING', 'task': task,
        'metric': key, 'higher_is_better': True, 'epochs': schedule['epochs'], 'updates': updates,
        'terminal': terminal, 'terminal_independent': terminal_independent,
        'best': best_metric, 'best_epoch': best_epoch, 'terminal_policy_metrics': policies,
        'source_cohort': protocol['source_cohort'], 'extra_global_head_examples': extra_head_examples,
        'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'extra_encoder_forwards_for_reward': 0, 'new_weight_files': 0,
        'wall_seconds': time.monotonic()-begun, **guard(DEADLINE)})


def worker(task):
    global DEADLINE
    protocol = json.loads((RUN/'protocol.json').read_text())
    job = json.loads((RUN/task/'job.json').read_text()); DEADLINE = job['deadline_unix']
    core.guard = guard
    core.original.runtime(); random.seed(0); np.random.seed(0); torch.manual_seed(0)
    try:
        fit(task, protocol, parent.get_source())
    except BaseException:
        write(RUN/task/'failure.json', {'traceback': traceback.format_exc(), 'time_unix': time.time(),
            'no_automatic_retry': True, 'new_weight_files': 0})
        raise


def pipeline():
    state = json.loads((parent.RUN/'queue_state.json').read_text())
    if (state['pending'] or state['failed'] or not state['source_in_RAM'] or
            set(state['active'])-{'original_DINO_CFC22', 'shared_GSR_CFC22'}):
        raise RuntimeError('Only the two established CFC jobs may coexist with this candidate')
    old = json.loads((parent.RUN/'protocol.json').read_text())
    source = json.loads((parent.RUN/'source_RAM_receipt.json').read_text())
    comparison = json.loads((parent.RUN/'comparison.json').read_text())
    references = {task: comparison['tasks'][task] for task in TASKS}
    if any(r['arms'][arm]['status'] != 'COMPLETE' for r in references.values()
           for arm in ('shared_GSR', 'original_DINO')):
        raise RuntimeError('All four core reference endpoints must already be complete')
    parent_jobs = state['active']
    parent_pids = {v['PID'] for v in parent_jobs.values()}
    admissions = {gpu: gpu_admission(gpu, parent_pids) for gpu in (2, 3)}
    if not all(a['eligible'] for a in admissions.values()):
        raise RuntimeError(('GPU admission unavailable; no allocation', admissions))
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if available < 192*1024**3:
        raise RuntimeError('Host RAM admission below192GiB; no allocation')
    started = time.time(); deadline = min(started+21600, old['deadline_unix'])
    if deadline-started < max(CAPS.values()):
        raise RuntimeError('Insufficient remaining inherited time for fixed candidate')
    RUN.mkdir()
    protocol = {'status': 'FROZEN_BEFORE_MODEL_FORWARD', 'started_unix': started, 'deadline_unix': deadline,
        'entry': str(Path(__file__).resolve()),
        'parent_protocol': str(parent.RUN/'protocol.json'), 'source_receipt': source,
        'source_cohort': old['source_cohort'], 'encoder': old['encoder'], 'data_manifest': old['data_manifest'],
        'target_schedules': {task: old['target_schedules'][task] for task in TASKS}, 'seed': 0,
        'tasks': list(TASKS), 'updates': 45700, 'source_retraining': False,
        'baseline_reuse': 'Four new GSR target fits only; all fixed-router GSR, DINO and stronger references inherited without retraining/rescoring.',
        'source_weight_storage': 'Existing immutable RAM snapshot; no weight files.',
        'change': 'During downstream FT, train the existing selector with a batch REINFORCE loss using global-only loss as an action-independent baseline. Source initialization/architecture/heads/task losses and schedules unchanged.',
        'policy': 'Training pi=.5softmax(selector(G.detach()))+.125, sample with private CUDAseed700. Inference argmax, exactlyone local region. This also changes the old half-argmax/half-uniform training sampling rule; not an isolated gradient-only ablation.',
        'route_loss': '(task_loss.detach()-global_loss_no_grad)*sum(log pi(selected)); coefficient1, signed reward retained, no reward clipping/normalization. Selector AdamW1e-4/wd1e-4/cosine as fusion; gradclip1.',
        'baseline': 'Same current task head, global32 field, no_grad and fork_rng before actual task loss; no extra encoder pass, no head running-stat buffers in these four interfaces. Batch reward can have high variance; not calibrated gain prediction or novelty acceptance.',
        'selector_input_stop_gradient': True, 'primary': 'terminal60 FSOD / terminal100 others; best auxiliary; no direction-based stopping',
        'resource': {'physical_GPUs': [2, 3], 'maximum_joint_workers_per_GPU': 2,
            'job_GPU_GiB': 32, 'coexisting_parent_CFC_cap_GiB': 36, 'physical_free_margin_GiB': 8, 'job_RSS_GiB': 48,
            'job_seconds': CAPS, 'global_seconds_max': 21600, 'new_disk_GiB': 2, 'CPU_threads_per_job': 4},
        'dispatch': 'One new worker per GPU while its existing CFC worker is live; up to two new workers when CFC finishes. Existing CFC settings and deadlines unchanged. FSOD/UATD first, then Marine/NKSID, with actual free-memory admission. Shared-load timing, not an isolated speed benchmark.',
        'coexisting_parent_jobs_at_freeze': parent_jobs,
        'admission': admissions, 'reference_comparison': str(parent.RUN/'comparison.json'),
        'reference_tasks': references,
        'diagram': str(core.BASE/'gsr_method_diagrams_v1/19_online_gain_router_candidate.png'),
        'validation_scope': 'No extra smoke, SHA, synthetic test or model probe. Record actual first training step and original independent saved-prediction metrics. No saved trained weights.'}
    write(RUN/'protocol.json', protocol)
    active, waits, launches = {}, [], {}
    occupied = set().union(*(os.sched_getaffinity(pid) for pid in parent_pids))
    cpus = sorted(set(os.sched_getaffinity(0))-occupied)
    pending = ['FSOD', 'UATD', 'Marine', 'NKSID']
    try:
        while pending or active:
            if time.time() > deadline:
                raise TimeoutError('Candidate global deadline')
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > 2*1024**3:
                raise RuntimeError('Online routing2GiB disk cap')
            for task, (child, log, gpu, began) in list(active.items()):
                if time.time() > min(deadline, began+CAPS[task]) and child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None: continue
                code = child.wait(); log.close(); del active[task]
                waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu,
                    'actual_wait_exit': code, 'wall_seconds': time.time()-began})
                write(RUN/'actual_waits.json', waits)
            for task in list(pending):
                gpu = 2 if task in ('FSOD', 'NKSID') else 3
                old_live = sum(v['GPU'] == gpu and Path('/proc/'+str(v['PID'])).exists() for v in parent_jobs.values())
                new_live = sum(v[2] == gpu for v in active.values())
                if old_live+new_live >= 2: continue
                admission = gpu_admission(gpu, parent_pids | {v[0].pid for v in active.values()})
                if not admission['eligible']: continue
                folder = RUN/task; folder.mkdir()
                write(folder/'job.json', {'task': task, 'physical_GPU': gpu, 'admission': admission,
                    'deadline_unix': min(deadline, time.time()+CAPS[task])})
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log = (folder/'stdout.log').open('w')
                child = subprocess.Popen([core.original.PYTHON, '-B', '-u', '-m', 'runs.gsr_online_gain_routing',
                    'worker', task], cwd=core.original.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                index = TASKS.index(task); os.sched_setaffinity(child.pid, cpus[index*4:index*4+4])
                active[task] = (child, log, gpu, time.time()); pending.remove(task)
                launches[task] = {'PID': child.pid, 'physical_GPU': gpu, 'launched_unix': time.time()}
                write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'workers': launches, 'new_weight_files': 0})
            write(RUN/'queue_state.json', {'status': 'RUNNING',
                'active': {task: {'PID': v[0].pid, 'GPU': v[2]} for task, v in active.items()},
                'pending': pending, 'actual_waits': waits})
            time.sleep(2)
        results = {}
        for row in waits:
            if row['actual_wait_exit'] != 0: continue
            task = row['task']; result = json.loads((RUN/task/'results.json').read_text()); key = result['metric']
            reference = protocol['reference_tasks'][task]
            results[task] = {'result': str(RUN/task/'results.json'), 'terminal': result['terminal'][key],
                'best': result['best'][key], 'best_epoch': result['best_epoch'], 'references': reference,
                'terminal_minus_fixed_router_GSR': result['terminal'][key]-reference['arms']['shared_GSR']['terminal'],
                'terminal_minus_matched_DINO': result['terminal'][key]-reference['arms']['original_DINO']['terminal'],
                'best_minus_fixed_router_GSR': result['best'][key]-reference['arms']['shared_GSR']['best'],
                'best_minus_matched_DINO': result['best'][key]-reference['arms']['original_DINO']['best'],
                'same_terminal_routing': {k: v[key] for k, v in result['terminal_policy_metrics'].items()},
                'historical_terminal_differences': {k: result['terminal'][key]-v if v is not None else None
                    for k, v in reference['historical_references'].items()}}
        status = 'COMPLETE_FIXED_SCOPE' if all(r['actual_wait_exit'] == 0 for r in waits) else 'FAILED_PARTIAL'
        write(RUN/'comparison.json', {'status': status, 'tasks': results, 'primary': protocol['primary']})
        write(RUN/'completion.json', {'status': status, 'actual_waits': waits,
            'wall_seconds_from_protocol_freeze': time.time()-started, 'new_weight_files': 0, 'no_automatic_retry': True})
        write(RUN/'queue_state.json', {'status': status, 'active': {}, 'actual_waits': waits})
    except BaseException:
        write(RUN/'supervision_failure.json', {'traceback': traceback.format_exc(), 'time_unix': time.time()})
        raise
    finally:
        for task, (child, log, gpu, began) in active.items():
            if child.poll() is None: os.killpg(child.pid, signal.SIGKILL)
            code = child.wait(); log.close()
            waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code,
                'wall_seconds': time.time()-began, 'supervision_cleanup': True})
        write(RUN/'actual_waits.json', waits)
        if active:
            write(RUN/'queue_state.json', {'status': 'SUPERVISION_FAILED', 'active': {}, 'actual_waits': waits})
    return 0 if status == 'COMPLETE_FIXED_SCOPE' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pipeline', 'worker')); parser.add_argument('task', nargs='?')
    args = parser.parse_args()
    if args.phase == 'pipeline':
        raise SystemExit(pipeline())
    worker(args.task)
