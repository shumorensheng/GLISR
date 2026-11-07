"""Train full-image residual feature corrections with fixed original DINO weights."""
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

from runs import gsr_mirror_ema as base

core, native, halo, reader = base.core, base.native, base.halo, base.reader
RUN = core.old.BASE/'gsr_feature_residual_v1'
PARENT = core.old.BASE/'gsr_mirror_ema_v1'
for module in (base, native, halo, base.attention, core, core.reuse):
    module.RUN = RUN
CPUS = sorted(os.sched_getaffinity(0))


class FeatureResidual(nn.Module):
    context16 = core.reuse.Model.context16

    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher
        self.head = teacher.head
        self.task = teacher.task
        self.mode = 'feature_residual'
        self.bypasses = copy.deepcopy(teacher.bypasses).requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(102)
            self.bridge = halo.FineContext()
            self.residual = nn.Sequential(nn.Conv2d(1152, 64, 1), nn.GELU(),
                                          nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.GELU(),
                                          nn.Conv2d(64, 384, 1))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        y = torch.sin(math.pi*(torch.arange(32)+.5)/32)
        self.register_buffer('taper', (y[:, None]*y[None])[None, None])
        y, x = torch.meshgrid((torch.arange(32)+.5)/32, (torch.arange(32)+.5)/32, indexing='ij')
        self.register_buffer('xy', torch.stack((x, y), -1)[None])

    def fine_field(self, tokens, g, actions):
        return self.bridge(self.teacher.field(tokens, self.bypasses, True, g, actions), g, actions)

    def corrected(self, g, fine, actions):
        crop = F.grid_sample(g, halo.crop_grid(actions, 32, 32), mode='bilinear',
                             padding_mode='border', align_corners=False)
        delta = self.residual(torch.cat((fine, crop, fine-crop), 1))*self.taper
        origin = torch.stack((actions%2, actions//2), 1).float()*halo.ROI_STEP
        grid = 2*(self.xy-origin[:, None, None])/halo.ROI_SIZE-1
        valid = ((grid >= -1) & (grid < 1)).all(-1)[:, None]
        scattered = F.grid_sample(delta, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return g+scattered*valid


def train_local(model, spec, data, targets, dense):
    folder = RUN/model.task/model.mode
    folder.mkdir()
    optimizer = torch.optim.AdamW([
        {'params': model.bypasses.parameters(), 'lr': .0001},
        {'params': list(model.bridge.parameters())+list(model.residual.parameters()), 'lr': .001},
    ], weight_decay=.0001)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    ema = base.ParameterEMA(parameters)
    actions = torch.from_numpy(np.load(RUN/model.task/'local_actions.npy')).cuda()
    generator = torch.Generator().manual_seed(0)
    total = 30*math.ceil(len(spec['rows']['train'])/spec['batch'])
    updates = 0
    begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, 31):
            model.train()
            model.teacher.eval()
            if model.task == 'FSOD':
                model.head.train()
            order = torch.randperm(len(spec['rows']['train']), generator=generator)
            loss_sum = 0.
            for macro in order.split(spec['batch']):
                optimizer.zero_grad(set_to_none=True)
                optimizer.param_groups[0]['lr'] = native.module_lr(updates, total, .0001)
                denominator = dense['train'][2][macro].sum() if model.task == 'UATD' else len(macro)
                value = 0.
                for ids in macro.split(spec['micro']):
                    a = actions[epoch-1, ids]
                    g = data['train'][0][ids]
                    fine = model.fine_field(data['train'][1][ids, a], g, a)
                    field = model.corrected(g, fine, a)
                    target = (tuple(t[ids] for t in dense['train']) if model.task == 'UATD' else
                              dense['train'][ids] if model.task == 'Marine' else
                              reader.batch_targets('FSOD', targets['train'], ids.tolist(), 'cuda'))
                    weight = target[2].sum()/denominator if model.task == 'UATD' else len(ids)/denominator
                    loss = base.task_loss(model.task, model.head, field, target)*weight
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite full-image residual training loss')
                    loss.backward()
                    value += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                ema.update()
                updates += 1
                loss_sum += value*len(macro)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {
                        'original_DINO_trainable': sum(p.numel() for p in model.teacher.encoder.parameters() if p.requires_grad),
                        'global_trainable': sum(p.numel() for p in model.teacher.parameters() if p.requires_grad),
                        'global_has_grad': any(p.grad is not None for p in model.teacher.parameters()),
                        'full_image_target': True, 'one_shared_frozen_task_head': True,
                        'new_parameters': sum(p.numel() for p in parameters),
                        'residual_output_grad_norm': float(model.residual[-1].weight.grad.norm()),
                        'fine_field': list(fine.shape), 'corrected_field': list(field.shape),
                        'macro': len(macro), 'micro': spec['micro'], 'loss': value,
                        'zero_up_initialization': 'First gradient reaches residual output; earlier local modules may first move on later updates.'})
                if updates % 100 == 0:
                    native.resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': loss_sum/len(order), 'wall_seconds': time.monotonic()-begun}
            log.write(json.dumps(entry)+'\n')
            log.flush()
            if epoch % 5 == 0:
                print(json.dumps({'task': model.task, 'phase': model.mode, **entry}), flush=True)
    ema.install()
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    core.old.write(folder/'train_report.json', {'epochs': 30, 'updates': updates,
        'wall_seconds': time.monotonic()-begun, 'original_DINO_updates': 0, 'global_head_updates': 0,
        'terminal_parameter_state': 'EMA.999', 'new_weights_saved': False, 'resources': native.resources()})
    return updates


@torch.no_grad()
def gain_labels(model, spec, bank, targets, dense):
    g, local = bank['train']
    losses = np.empty((len(g), 5), np.float32)
    if model.task == 'FSOD':
        model.head.train()
    for i in range(len(g)):
        target = (tuple(t[i:i+1] for t in dense['train']) if model.task == 'UATD' else
                  dense['train'][i:i+1] if model.task == 'Marine' else
                  reader.batch_targets('FSOD', targets['train'], [i], 'cuda'))
        for j in range(-1, 4):
            # Identical seeded RPN/RoI sampling context for all five counterfactuals.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(10000+i)
                field = g[i:i+1] if j < 0 else model.corrected(
                    g[i:i+1], local[i:i+1, j], torch.tensor([j], device='cuda'))
                losses[i, j+1] = float(base.task_loss(model.task, model.head, field, target))
        if (i+1) % 200 == 0:
            print(json.dumps({'gain_labels': model.task, 'images': i+1}), flush=True)
            native.resources()
    model.head.eval()
    np.save(RUN/model.task/model.mode/'counterfactual_training_losses.npy', losses)
    return losses[:, :1]-losses[:, 1:]


@torch.no_grad()
def predict(model, spec, bank, actions):
    g, local = bank
    pieces = []
    for start in range(0, len(g), spec['micro']):
        end = min(start+spec['micro'], len(g))
        field = g[start:end].clone()
        a = torch.as_tensor(actions[start:end], device='cuda')
        ids = torch.where(a >= 0)[0]
        if len(ids):
            field[ids] = model.corrected(field[ids], local[start:end][ids, a[ids]], a[ids])
        pieces.append(base.decode(model.task, model.head, field, spec['rows']['validation'][start:end]))
    return [p for piece in pieces for p in piece] if model.task == 'UATD' else np.concatenate(pieces)


@torch.no_grad()
def evaluate(model, spec, bank, targets, selector, sr, updates):
    g = bank['validation'][0]
    scores = torch.cat([selector(model.context16(g[i:i+64])) for i in range(0, len(g), 64)]).cpu().numpy()*sr['gain_scale']+sr['gain_mean']
    top = scores.argmax(1)
    keep = scores.max(1) > 0
    entropy = base.entropy_actions(model, g)
    policies = {'global_only': np.full(len(g), -1, np.int64), 'learned': np.where(keep, top, -1),
        'learned_forced_top1': top, 'random_budgetmatched': np.where(keep, np.random.default_rng(2).integers(0, 4, len(g)), -1),
        'fixed_budgetmatched': np.where(keep, sr['fixed_region_fit_only'], -1),
        'entropy_budgetmatched': np.where(keep, entropy, -1), 'entropy_forced_top1': entropy}
    metrics = {}
    folder = RUN/model.task/model.mode
    for policy, actions in policies.items():
        prediction = predict(model, spec, bank['validation'], actions)
        metrics[policy] = core.reuse.score_prediction(model.task, prediction, targets['validation'], spec, folder/(policy+'_prediction'))
        print(json.dumps({'score': model.task+'/'+policy, 'value': metrics[policy][spec['metric']]}), flush=True)
    np.savez(folder/'actions.npz', **policies)
    report = {'task': model.task, 'mode': model.mode, 'metrics': metrics, 'local_updates': updates,
        'selection_fraction': float(keep.mean()), 'original_DINO_updates': 0, 'global_head_updates': 0,
        'terminal_only': True, 'new_weights_saved': False, 'resources': native.resources()}
    core.old.write(folder/'results.json', report)
    return report


@torch.no_grad()
def online_cost(model, spec, selector, sr):
    samples = [native.read_intensity(row) for row in spec['rows']['validation'][:32]]
    cells = []
    for policy in ('global_only', 'learned'):
        load = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
        baseline = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        records = []
        for visit in range(146):
            i = visit % 32 if visit < 50 else (visit-50) % 32
            torch.cuda.synchronize()
            begun = time.perf_counter()
            image = (samples[i].log1p()-spec['normalization']['mean'])/spec['normalization']['std']
            g = model.teacher.global_field(F.interpolate(image[None], (448, 448), mode='bilinear', align_corners=False, antialias=True).cuda())
            action = -1
            if policy == 'learned':
                scores = selector(model.context16(g))*sr['gain_scale']+sr['gain_mean']
                if float(scores.max()) > 0:
                    action = int(scores.argmax(1))
                    a = torch.tensor([action], device='cuda')
                    large = F.interpolate(image[None], (768, 768), mode='bilinear', align_corners=False, antialias=True)
                    y, x = action//2*320, action%2*320
                    tokens = model.teacher.encoder.backbone.prepare_tokens_with_masks(large[..., y:y+448, x:x+448].cuda())
                    fine = model.fine_field(tokens, g, a)
                    g = model.corrected(g, fine, a)
            base.decode(model.task, model.head, g, spec['rows']['validation'][i:i+1])
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begun
            if visit >= 50:
                records.append({'row_id': spec['row_ids']['validation'][i], 'round': (visit-50)//32, 'action': action, 'seconds': elapsed})
        values = [r['seconds'] for r in records]
        cells.append({'policy': policy, 'median_seconds': float(np.median(values)), 'p95_seconds': float(np.percentile(values, 95)),
            'raw': records, 'GPU_load': load, 'baseline_reserved': baseline, 'baseline_allocated': allocated,
            'peak_reserved': torch.cuda.max_memory_reserved(), 'peak_allocated': torch.cuda.max_memory_allocated(),
            'incremental_peak_allocated': torch.cuda.max_memory_allocated()-allocated})
    core.old.write(RUN/model.task/model.mode/'cost.json', {'cells': cells,
        'boundary': 'Decodednative normalization/resize/H2D/globalencoder/selector/optionallocalencoder/feature residual/sharedhead/NMS/nativepostproc; decode excluded.',
        'memory_scope': 'Trainingbanks resident; not isolateddeployment memory. External load recorded.'})


def workflow(task):
    native.ACTIVE_TASK = task
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    spec = core.reuse.protocol()['tasks'][task]
    targets = {s: reader.make_targets(task, rows) for s, rows in spec['rows'].items()}
    dense = ({s: base.box_targets32(t, 'cuda') for s, t in targets.items()} if task == 'UATD' else
             {s: t.cuda() for s, t in targets.items()} if task == 'Marine' else {})
    teacher = base.DetailGlobal(task).cuda()
    if task == 'Marine':
        counts = torch.bincount(dense['train'].long().flatten(), minlength=12)
        weights = counts.float().rsqrt()
        weights /= (weights*counts).sum()/counts.sum()
        teacher.head.loss_weights.copy_(weights)
        core.old.write(RUN/task/'training_class_weights.json', {'exact_train_pixel_counts': counts.tolist(), 'FP32_loss_weights': weights.tolist()})
    raw, mirror = base.extract_mirror(teacher, spec)
    global_report = base.train_global(teacher, spec, raw, mirror, targets, dense)
    del mirror
    data = base.global_bank(teacher, raw)
    del raw
    prediction = base.predict_global(teacher, data['validation'][0], spec)
    global_report['metrics'] = core.reuse.score_prediction(task, prediction, targets['validation'], spec, RUN/task/'global/terminal_prediction')
    core.old.write(RUN/task/'global/results.json', global_report)
    model = FeatureResidual(teacher).cuda()
    updates = train_local(model, spec, data, targets, dense)
    bank = halo.packed_bank(model, data)
    gains = gain_labels(model, spec, bank, targets, dense)
    selector, sr = core.reuse.train_selector(task, model.mode, model.context16(bank['train'][0]), gains)
    report = evaluate(model, spec, bank, targets, selector, sr, updates)
    online_cost(model, spec, selector, sr)
    previous = json.loads((PARENT/task/'results.json').read_text())
    detail = json.loads((core.old.BASE/'gsr_detail_readout_v1'/task/'results.json').read_text())
    reference = max(core.reuse.protocol()['references'][task]['baseline_terminal_three_seed_mean'],
        previous['local_stage']['metrics']['learned'][spec['metric']], previous['global_stage']['metrics'][spec['metric']],
        detail['local_stage']['metrics']['learned'][spec['metric']], detail['global_stage']['metrics'][spec['metric']],
        global_report['metrics'][spec['metric']])
    core.old.write(RUN/task/'results.json', {'status': 'COMPLETE_FULL_IMAGE_FEATURE_RESIDUAL', 'task': task,
        'global_stage': global_report, 'local_stage': report, 'new_weights_saved': False,
        'strongest_completed_comparison_value': reference,
        'learned_minus_strongest': report['metrics']['learned'][spec['metric']]-reference, 'resources': native.resources()})


def pipeline():
    begun = time.time()
    parent = json.loads((PARENT/'protocol.json').read_text())
    RUN.mkdir()
    (RUN/'logs').mkdir()
    protocol = copy.deepcopy(parent)
    protocol.update(status='FROZEN_BEFORE_NEW_MODEL', source=str(Path(__file__).resolve()), parent=str(PARENT),
        started_unix=begun, deadline_unix=begun+28800,
        principle_diagram=str(core.old.BASE/'gsr_method_diagrams_v1/09_feature_residual.svg'),
        global_stage='Exactly reused mirror/EMA global training and dense readouts; same seed0, GT/input normalization, heads/optimizers/full epochs. Fresh fit, not weight recovery.',
        local_stage='Separate conditional rank8 QKV bank + FineContext + 1152->64->depthwise3x3->384 zero-up feature residual. Inputs local, global ROI, local-global. Smooth sine taper and spatial scatter add residual to full32 global field; one fixed global task head. Originalglobal model/head fixed. Full-image originalGT loss, not croppedGT or separate local-head prediction merge. 30epochs randomoneROI; local EMA.999.',
        local_optimizer='AdamWwd.0001; conditionalattentionpeak.0001/200warmupcosine, FineContext/residual .001. Frozen global/head excluded; no copiedlocalhead/calibrator.',
        Marine_loss='Same trainingclassbalancedCE+.5presentclassDice as global, evaluated against fullnative480x320 masks.',
        detector_loss='Same original full-image FSOD fourRPN/RoI losses or UATD32focal+regression and wholemacropositive denominator; no cropGT or geometry changes.',
        gain_labels='Fixed terminal fullmodel; pertrainimage actual full-image taskloss l0-lj for four ROIs. Signed values retained. FSOD fivecounterfactuals share fork_rng seed10000+imageordinal, feature-dependent proposal/sampling outcomes neednotmatch. No AP/F1 surrogate or devGT inputs.',
        detector_merge='Removed: single shared full-image head after residual fusion. No localbox remapping/scorecalibrator/NMSmerge; originalheadpostprocessing only.',
        training_regularization='Reused global mirrorp.5/EMA.999. Localoriginalviews, EMA.999 only fineattention/FineContext/residual; globalheadfixed. No bestepochchoice.',
        endpoints='Fixed global60/100 + local30 + selector100; no calibrator. Terminal fullmethod main, global scheduledbest auxiliary only. Original saved-prediction scorers.',
        cost='50warmup + fixedfirst32devimages x3/B1; fullglobalencoder/selector/optionallocalencoder/featurefusion/singlehead/nativepostprocessing. Decode/setup excluded; residentbankmemory scope.',
        calibrator='None', policies='global_only,learned,forcedtop1,random/fixed/entropybudgetmatched,entropyforced;7fixedpolicies; no posthocprimaryswitch.',
        schedule='FSODGPU2 after actualparentMarineexit0, thenMarineGPU2; UATDGPU3 after retained parent foreground93827 actualexit0. Oneownedworker/card and2total acrossunits.',
        release_receipts='ROOT publishes immutable parentGPU2release.json from actualparentMarineworkerwait0 and parentactualforegroundwait.json after retained93827 actualexit0; no score-based decisions.',
        comparison_limit='Structural hypothesis: full-image task-supervised feature correction versus independently trained localpredictionmerge. Localhead/calibrator removed and objective/fusion changed together; not isolatedcause/novelty/SOTA claim.',
        total_local_updates=16440, total_global_updates=43000)
    for task in protocol['tasks']:
        folder = RUN/task
        folder.mkdir()
        for name in ('local_actions.npy', 'selector_split.npz', 'global_flip_plan.npy'):
            (folder/name).write_bytes((PARENT/task/name).read_bytes())
    core.old.write(RUN/'protocol.json', protocol)
    active, waits = {}, []
    started_f = started_u = False

    def launch(task, gpu):
        admission = core.old.admission(gpu)
        if admission['free_MiB'] < protocol['required_free_MiB'][task]:
            raise RuntimeError(task+' memory admission insufficient '+str(admission))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'XFORMERS_DISABLED': '1'}
        cpus = ','.join(map(str, CPUS[:4] if gpu == 2 else CPUS[4:8]))
        log = (RUN/'logs'/f'{task}.log').open('w')
        child = subprocess.Popen(['/usr/bin/taskset', '-c', cpus, core.old.PYTHON, '-B', '-u', '-m',
            'runs.gsr_feature_residual', task], cwd=core.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[task] = (child, log, gpu, time.time(), admission)
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'observed_unix': time.time(),
            'workers': {t: x[0].pid for t, x in active.items()}, 'new_weight_files': 0})

    try:
        core.old.write(RUN/'queue_state.json', {'status': 'CPU_STAGED', 'active': {}, 'actual_waits': []})
        while active or not started_f or not started_u:
            if (PARENT/'failure_closure.json').exists():
                raise RuntimeError('Parent stopped; no automatic continuation')
            release = RUN/'parentGPU2release.json'
            if not started_f and release.exists():
                receipt = json.loads(release.read_text())
                if receipt['task'] != 'Marine' or receipt['actual_wait_exit'] != 0:
                    raise RuntimeError('ParentMarine release nonzero')
                launch('FSOD', 2)
                started_f = True
            release = PARENT/'actual_foreground_wait.json'
            if not started_u and release.exists():
                receipt = json.loads(release.read_text())
                if receipt['actual_foreground_exit'] != 0:
                    raise RuntimeError('Parent foreground nonzero')
                launch('UATD', 3)
                started_u = True
            if time.time() > protocol['deadline_unix']:
                raise TimeoutError('Fixed8h residual-feature deadline')
            for task, (child, log, gpu, started, admission) in list(active.items()):
                if time.time()-started > protocol['caps']['workflow_seconds']:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait()
                log.close()
                del active[task]
                waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code,
                              'wall_seconds': time.time()-started, 'admission': admission})
                if code != 0:
                    raise RuntimeError(task+' actual worker exit '+str(code))
                if task == 'FSOD':
                    launch('Marine', 2)
            core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {t: x[0].pid for t, x in active.items()}, 'actual_waits': waits})
            time.sleep(1)
        tasks = {task: json.loads((RUN/task/'results.json').read_text()) for task in protocol['tasks']}
        core.old.write(RUN/'results.json', {'status': 'COMPLETE_FIXED_FEATURE_RESIDUAL', 'tasks': tasks})
        size = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
        if size > protocol['caps']['new_disk_bytes']:
            raise RuntimeError('Results-only disk cap exceeded')
        closure = {'status': 'COMPLETE_SCOPE_GOAL_REQUIRES_RESULT_DECISION', 'actual_waits': waits,
                   'wall_seconds': time.time()-begun, 'new_entity_bytes': size, 'new_weights_saved': False}
        core.old.write(RUN/'wave_closure.json', closure)
        core.old.write(RUN/'writing_handoff.json', closure)
        core.old.write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        failure = traceback.format_exc()
        for task, (child, log, gpu, started, admission) in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            code = child.wait()
            log.close()
            waits.append({'task': task, 'PID': child.pid, 'actual_wait_exit': code, 'cancelled_after_failure': True})
        core.old.write(RUN/'failure_closure.json', {'status': 'STOPPED_NO_AUTOMATIC_RETRY', 'traceback': failure, 'actual_waits': waits})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['pipeline', 'FSOD', 'Marine', 'UATD'])
    args = parser.parse_args()
    core.old.runtime()
    if args.phase == 'pipeline':
        pipeline()
    else:
        workflow(args.phase)
