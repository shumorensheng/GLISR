"""Original DINO weights fixed; conditional low-rank attention and fine prediction."""
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

from runs import gsr_halo_predictions as halo
from sonar_encoder.training.uatd_head import box_targets

native = halo.native
core = halo.core
reader = halo.reader
RUN = core.old.BASE/'gsr_attention_fixed_v1'
PARENT = core.old.BASE/'gsr_halo_predictions_v1'
native.RUN = RUN
halo.RUN = RUN
core.RUN = RUN
core.reuse.RUN = RUN
AVAILABLE_CPUS = sorted(os.sched_getaffinity(0))


class AttentionDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(384, 8, bias=False)
        self.up = nn.Linear(8, 1152, bias=False)
        self.condition = nn.Linear(386, 16)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.condition.weight); nn.init.zeros_(self.condition.bias)

    def forward(self, x, context, actions):
        z = self.down(x)
        if context is not None:
            center = (torch.stack((actions%2, actions//2), 1).float()*halo.ROI_STEP+halo.ROI_SIZE/2)*2-1
            scale, bias = self.condition(torch.cat((context.mean((2, 3)), center), 1)).chunk(2, 1)
            z = z*(1+scale[:, None])+bias[:, None]
        return self.up(z)


class AttentionGlobal(nn.Module):
    global_field = native.GlobalModel.global_field

    def __init__(self, task):
        super().__init__(); self.task = task
        cfg = core.reuse.protocol()['encoder']
        self.encoder = native.build_encoder(Path(cfg['code_dir']), Path(cfg['weights_path']), 'R010').eval().requires_grad_(False)
        self.head = reader.new_head(task, 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            self.neck = core.SpatialNeck()
            self.bypasses = nn.ModuleList([AttentionDelta() for _ in range(12)])
        for module in self.bypasses:
            module.condition.requires_grad_(False)
        # The dictionary selects one external adapter bank; it does not register new parameters inside original DINO.
        self.attention_context = {'layers': self.bypasses, 'context': None, 'actions': None}
        self.hook_calls = [0]*12; self.first_hook_shapes = [{} for _ in range(12)]
        for i, block in enumerate(self.encoder.backbone.blocks):
            def add_delta(module, inputs, output, index=i):
                self.hook_calls[index] += 1
                if self.hook_calls[index] == 1:
                    self.first_hook_shapes[index] = {'input': list(inputs[0].shape), 'output': list(output.shape),
                                                     'original_QKV_weight_trainable': module.weight.requires_grad}
                return output+self.attention_context['layers'][index](inputs[0], self.attention_context['context'], self.attention_context['actions'])
            block.attn.qkv.register_forward_hook(add_delta)

    def field(self, tokens, bypasses, local=False, context=None, actions=None):
        self.attention_context.update(layers=bypasses, context=context, actions=actions)
        values = []
        for i, block in enumerate(self.encoder.backbone.blocks):
            tokens = block(tokens)
            if i in (2, 5, 8, 11):
                field = self.encoder.backbone.norm(tokens[:, 1:]).transpose(1, 2).reshape(-1, 384, 32, 32)
                values.append(F.avg_pool2d(field, 2) if self.task == 'UATD' and not local else field)
        return self.neck(torch.stack(values, 1))


class AttentionLocal(nn.Module):
    context16 = core.reuse.Model.context16

    def __init__(self, teacher):
        super().__init__(); self.task = teacher.task; self.mode = 'attention_fixed'
        self.teacher = teacher; self.head = teacher.head
        self.local_head = copy.deepcopy(teacher.head).requires_grad_(True)
        self.bypasses = copy.deepcopy(teacher.bypasses).requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(102)
            self.bridge = halo.FineContext()
            if self.task != 'Marine':
                self.calibrator = nn.Sequential(nn.Linear(19, 32), nn.GELU(), nn.Linear(32, 1))
                nn.init.zeros_(self.calibrator[-1].weight); nn.init.zeros_(self.calibrator[-1].bias)
        if self.task == 'Marine':
            y = torch.sin(math.pi*(torch.arange(32)+.5)/32)
            self.register_buffer('taper', (y[:, None]*y[None])[None, None])
            y, x = torch.meshgrid((torch.arange(480)+.5)/480, (torch.arange(320)+.5)/320, indexing='ij')
            self.register_buffer('native_xy', torch.stack((x, y), -1)[None])

    def fine_field(self, tokens, g, actions):
        return self.bridge(self.teacher.field(tokens, self.bypasses, True, g, actions), g, actions)

    def marine_log_probability(self, g, fine, actions, fixed_merge=False):
        global_log = F.log_softmax(F.interpolate(self.head(g), (480, 320), mode='bilinear', align_corners=False), 1)
        origin = torch.stack((actions%2, actions//2), 1).float()*halo.ROI_STEP
        grid = 2*(self.native_xy-origin[:, None, None])/halo.ROI_SIZE-1
        valid = ((grid >= -1) & (grid < 1)).all(-1)[:, None]
        window = F.grid_sample(self.taper.expand(len(g), -1, -1, -1), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        # Main fusion is fixed .5*taper; the predeclared alternate uses full local weight*taper.
        log_alpha = window.log()+(0. if fixed_merge else math.log(.5))
        log_alpha = torch.where(valid, log_alpha, -torch.inf)
        local = F.grid_sample(self.local_head(fine), grid, mode='bilinear', padding_mode='border', align_corners=False)
        return torch.logaddexp(global_log+torch.log1p(-log_alpha.exp()), F.log_softmax(local, 1)+log_alpha)


def train_local(model, spec, data, targets, dense, cropped):
    folder = RUN/model.task/model.mode; folder.mkdir()
    head_opt = reader.head_optimizer(model.task, model.local_head)
    for group in head_opt.param_groups:
        group['lr'] = .0005 if model.task == 'FSOD' else .0001
    optimizer = torch.optim.AdamW([{'params': model.bypasses.parameters(), 'lr': .0001},
                                  {'params': model.bridge.parameters(), 'lr': .001}], weight_decay=.0001)
    actions = torch.from_numpy(np.load(RUN/model.task/'local_actions.npy')).cuda()
    counts = torch.stack([value[2].flatten(1).sum(1) for value in cropped], 1) if model.task == 'UATD' else None
    if model.task == 'Marine':
        class_count = torch.bincount(dense['train'].long().flatten(), minlength=12).float()
        weights = class_count.rsqrt(); weights /= (weights*class_count).sum()/class_count.sum()
        core.old.write(folder/'local_class_weights.json', {'train_pixel_counts': class_count.tolist(), 'inverse_sqrt_weights': weights.tolist(),
            'scope': 'Traininglabels only, all12classes. No inputnormalization refit or evaluationweighting.'})
    generator = torch.Generator().manual_seed(0); updates = 0; begun = time.monotonic()
    total = 30*math.ceil(len(spec['rows']['train'])/spec['batch'])
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, 31):
            model.train(); model.teacher.encoder.eval()
            order = torch.randperm(len(spec['rows']['train']), generator=generator); loss_sum = 0.
            for macro in order.split(spec['batch']):
                a = actions[epoch-1, macro]
                denominator = counts[macro, a].sum().clamp_min(1) if model.task == 'UATD' else len(macro)
                optimizer.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
                optimizer.param_groups[0]['lr'] = native.module_lr(updates, total, .0001)
                value = 0.
                for ids in macro.split(spec['micro']):
                    a = actions[epoch-1, ids]; g = data['train'][0][ids]
                    fine = model.fine_field(data['train'][1][ids, a], g, a)
                    if model.task == 'Marine':
                        labels = torch.stack([cropped[int(j)][int(i)] for i, j in zip(ids, a)]).long()
                        logits = F.interpolate(model.local_head(fine), (280, 187), mode='bilinear', align_corners=False)
                        probability = logits.softmax(1); onehot = F.one_hot(labels, 12).permute(0, 3, 1, 2)
                        support = onehot.sum((0, 2, 3))
                        dice = 1-(2*(probability*onehot).sum((0, 2, 3))+1)/(probability.sum((0, 2, 3))+support+1)
                        loss = F.cross_entropy(logits, labels, weight=weights)+.5*dice[support > 0].mean()
                    else:
                        loss = native.local_aux(model, fine, F.avg_pool2d(fine, 2), ids, a, cropped, denominator)
                    if model.task != 'UATD':
                        loss = loss*len(ids)/len(macro)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite attention-local training loss')
                    loss.backward(); value += float(loss.detach())
                parameters = [p for group in optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); head_opt.step(); updates += 1; loss_sum += value*len(macro)
                if updates == 1:
                    core.old.write(folder/'first_counted_update.json', {'original_DINO_trainable': sum(p.numel() for p in model.teacher.encoder.parameters() if p.requires_grad),
                        'global_trainable': sum(p.numel() for p in model.teacher.parameters() if p.requires_grad),
                        'global_has_grad': any(p.grad is not None for p in model.teacher.parameters()),
                        'fine_shape': list(fine.shape), 'new_adapter_grad_norms': [float(sum(p.grad.square().sum() for p in layer.parameters() if p.grad is not None).sqrt()) for layer in model.bypasses],
                        'local_adapter_parameters': sum(p.numel() for p in model.bypasses.parameters()),
                        'local_head_parameters': sum(p.numel() for p in model.local_head.parameters()), 'pixel_gate_parameters': 0,
                        'loss': value, 'macro': len(macro), 'micro': spec['micro']})
                if updates % 100 == 0:
                    native.resources()
            entry = {'epoch': epoch, 'updates': updates, 'loss': loss_sum/len(order), 'wall_seconds': time.monotonic()-begun}
            log.write(json.dumps(entry)+'\n'); log.flush()
            if epoch % 5 == 0:
                print(json.dumps({'task': model.task, 'phase': 'attention_local', **entry}), flush=True)
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    core.old.write(folder/'train_report.json', {'epochs': 30, 'updates': updates, 'wall_seconds': time.monotonic()-begun,
        'original_DINO_updates': 0, 'global_parameter_updates': 0, 'new_weights_saved': False, 'resources': native.resources()})
    return updates


@torch.no_grad()
def evaluate(model, spec, bank, targets, predictions, selector, sr, updates):
    g = bank['validation'][0]; n = len(g)
    scores = torch.cat([selector(model.context16(g[i:i+64])) for i in range(0, n, 64)]).cpu().numpy()*sr['gain_scale']+sr['gain_mean']
    top = scores.argmax(1); keep = scores.max(1) > 0; entropy = core.reuse.entropy_actions(model, g)
    policies = {'global_only': np.full(n, -1, np.int64), 'learned': np.where(keep, top, -1), 'learned_forced_top1': top,
        'random_budgetmatched': np.where(keep, np.random.default_rng(2).integers(0, 4, n), -1),
        'fixed_budgetmatched': np.where(keep, sr['fixed_region_fit_only'], -1), 'entropy_budgetmatched': np.where(keep, entropy, -1),
        'entropy_forced_top1': entropy, 'merge_control_budgetmatched': np.where(keep, top, -1)}
    metrics = {}; folder = RUN/model.task/model.mode
    for policy, actions in policies.items():
        pred = halo.predict(model, spec, bank['validation'], actions,
                            predictions['validation'] if model.task != 'Marine' else None,
                            fixed_merge=policy == 'merge_control_budgetmatched')
        metrics[policy] = core.reuse.score_prediction(model.task, pred, targets['validation'], spec, folder/(policy+'_prediction'))
        print(json.dumps({'score': model.task+'/'+policy, 'value': metrics[policy][spec['metric']]}), flush=True)
    np.savez(folder/'actions.npz', **policies)
    report = {'task': model.task, 'mode': model.mode, 'metrics': metrics, 'local_updates': updates,
        'selection_fraction': float(keep.mean()), 'original_DINO_updates': 0, 'global_parameter_updates': 0,
        'terminal_only': True, 'new_weights_saved': False, 'resources': native.resources()}
    core.old.write(folder/'results.json', report)
    return report


def workflow(task):
    native.ACTIVE_TASK = task
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    spec = core.reuse.protocol()['tasks'][task]
    targets = {split: reader.make_targets(task, rows) for split, rows in spec['rows'].items()}
    dense = {split: box_targets(value, 'cuda') for split, value in targets.items()} if task == 'UATD' else (
        {split: value.cuda() for split, value in targets.items()} if task == 'Marine' else {})
    teacher = AttentionGlobal(task).cuda(); raw = halo.extract(teacher, spec)
    global_report = native.train_global(teacher, spec, raw, targets, dense)
    core.old.write(RUN/task/'global/attention_adapter_report.json', {'normal_forward_hook_calls': teacher.hook_calls,
        'normal_first_hook_shapes': teacher.first_hook_shapes, 'rank': 8, 'alpha': 8, 'scale': 1,
        'original_encoder_trainable_parameters': sum(p.numel() for p in teacher.encoder.parameters() if p.requires_grad),
        'external_global_adapter_parameters': sum(p.numel() for p in teacher.bypasses.parameters()),
        'source': 'Actualnormalglobaltraining/evaluation calls; no extra forward/test/SHA.'})
    data = native.global_bank(teacher, raw); del raw
    pred = core.reuse.predict(teacher, spec, (data['validation'][0], None), np.full(len(spec['rows']['validation']), -1, np.int64))
    global_report['metrics'] = core.reuse.score_prediction(task, pred, targets['validation'], spec, RUN/task/'global/terminal_prediction')
    core.old.write(RUN/task/'global/results.json', global_report)
    cropped = halo.crop_targets(task, dense['train'] if task == 'Marine' else targets['train'])
    model = AttentionLocal(teacher).cuda(); updates = train_local(model, spec, data, targets, dense, cropped)
    bank = halo.packed_bank(model, data)
    predictions = {} if task == 'Marine' else halo.detection_banks(model, spec, bank)
    if task != 'Marine':
        halo.train_calibrator(model, predictions['train'], halo.normalized_targets(task, targets['train']))
    gains = halo.gain_labels(model, spec, bank, dense, targets, predictions)
    selector, sr = core.reuse.train_selector(task, model.mode, model.context16(bank['train'][0]), gains)
    report = evaluate(model, spec, bank, targets, predictions, selector, sr, updates)
    halo.online_cost(model, spec, selector, sr)
    parent_result = json.loads((PARENT/task/'results.json').read_text())
    previous = {'learned': parent_result['local_stage']['metrics']['learned'][spec['metric']],
                'global_only': parent_result['global_stage']['metrics'][spec['metric']], 'scope': 'Completedsingleheadseed0; not3seedmean.'}
    reference = max(core.reuse.protocol()['references'][task]['baseline_terminal_three_seed_mean'],
                    previous['learned'], previous['global_only'], global_report['metrics'][spec['metric']])
    core.old.write(RUN/task/'results.json', {'status': 'COMPLETE_ATTENTION_FIXED', 'task': task,
        'global_stage': global_report, 'local_stage': report, 'new_weights_saved': False,
        'prior_completed_terminal_comparisons': previous, 'strongest_completed_comparison_value': reference,
        'learned_minus_strongest': report['metrics']['learned'][spec['metric']]-reference, 'resources': native.resources()})

def pipeline():
    parent = json.loads((PARENT/'protocol.json').read_text())
    prior = {task: {'result_path': str(PARENT/task/'results.json'),
                    'scope': 'Samepredeclaredparentcondition/task/metric, completed singleheadseed0 value whenavailable. Not a3seedmean; no output-dependent training/stopping.'}
             for task in parent['tasks']}
    RUN.mkdir(); (RUN/'logs').mkdir(); begun = time.time()
    protocol = {'start_unix': begun, 'deadline_unix': begun+28800, 'budget_scope': 'Separatefixednewscientificmethod8h coordinator/workerwall; previousunitcosts preserved, not a continuationreset. PrelaunchLLMdrafting/serviceCPU isnotmeasured inthisclock.',
        'encoder': parent['encoder'], 'tasks': parent['tasks'], 'references': parent['references'],
        'prior_completed_terminal_comparisons': prior,
        'SWiSE_references': parent['SWiSE_references'],
        'source_utf8': Path(__file__).read_text(), 'reused_native_source_utf8': Path(native.__file__).read_text(), 'reused_halo_source_utf8': Path(halo.__file__).read_text(),
        'parent_fixed_run': str(PARENT),
        'principle_diagram': str(core.old.BASE/'gsr_method_diagrams_v1/06_attention_fixed.svg'),
        'authorization': 'User authorizes rapid frozen-original-weight structural optimization on all3tasks, onlyphysicalGPU2/3, maxoneownedworker/card. No newsmoke/SHA/review/trainedweightfiles.',
        'global_stage': '12externalrank8 alpha8 unmergedQKVLoRAdeltas (scale1), globalFiLMexcluded, officialDINOqkvweightsfixed andofficialblockforwardretainedviaoutputhooks. Fourlevelneck, directpublished448, originalhead60FSOD/100Marine/UATD. OriginalDINO fixed. Same headoptimizers/order/schedules; modulesAdamW.001/wd.0001/clip1/200warmupcosine.',
        'local_stage': 'Separatecopiedlocalattentiondeltas/head, fixedglobalmodel,30epochs randomonecornerROI/normalmacro. Directpublished768 four448crops atorigins0/320, overlap128pixels. NormalizedROIspan7/12, origins0/5/12. Local32field retainsdensity via localhead; UATDpool16 coversspan7/12 oftheframe, not scattereddown8. Localencodingdensity is12/7 ofglobal448; not the2xnonoverlappingcase.',
        'local_context': 'Localrank8featuresFiLMconditionedbyglobalmean384+actualROIcenter2;12attentionmodules. 64dim4head crossattention,1024localquerytokens/256globaltokens plus fullframe positions, zero-up residual. Global-conditioned localattentiondeltas; legacyattribute namebypasses nowholdsQKVdeltas, notConvpass.',
        'local_optimizer': 'BYPASSESAdamWpeak.0001/200warmupcosine; bridgeAdamW.001; noMarinepixelgate; copiedlocalheadSGD.0005FSOD orAdamW.0001others. Allwd as originalnative localstage.',
        'Marine_loss': 'Local280x187cropCE weightedbyalltrainnativeclasspixelcounts inverse-sqrt/weightedmean normalization, plus.5softDice overpresentcropGTclasses (smooth1). No mixedglobalNLL/pixelgate optimization. Maininference .5*taper probabilityfusion; alternate1*taper. Originalnative480x320 all12evaluationGT unchanged.',
        'detector_loss': 'OriginalcroppedGT losses only, fullframe evaluationGT unchanged. FSOD sumoriginal4RPN/ROI losses, UATD focal+regression with macropositiveauxdenominator max(P,1), emptycropsretainnegativefocal. No nondifferentiableNMS loss claim.',
        'detector_merge': 'Globalboxes/scores fixed; maplocal224 ornormalizedROI to fullframe using origin(0or5/12) andspan7/12. Calibrate localscores, concatenate, classwiseIoU.5NMS/top100/score>=.05. FP32coordinate roundtrip is declared, not a bitwiseactive-path identity claim.',
        'calibrator': '10epochsAdamW.001/wd.0001/B512/private0.19inputs: scorelogit(eps1e-5),10classonehot,boxcenter/size,maxsameclassglobalIoU/globalconfidence,cornerIDxy.32GELUhidden,zero-output init. Train-only localbranch one-to-one scoreorderedIoU.5/fullframeGT binarylabels. Globalconfidenceunchanged before NMS.',
        'gain_labels': 'Signedtrain-only mergedprediction utility: Marineexistingclass/boundaryweightedNLL reduction; detection classweights inverse-sqrt alltrainGTcounts, weighted2TP/(GT+predictedcount), matchingIoU.5/score.05. Correctemptyimage utility1, falsepositiveempty0. Proxy isnot AP orphysicalGT.',
        'selector': 'Existingselector100epochs/private7, parentinner80/20seed37 unchanged. Global/localheads fittedalltrain; innercheck is selector-only, not independentof taskfitting. Retainnegativegains, skipifmaxpredictedgain<=0.',
        'policies': 'global_only,learned,forcedtop1,random/fixed/entropybudgetmatched,entropyforced,mergecontrolsamelearnedactions. Marinecontrol1*taper versusmain.5*taper; detectorcontroluncalibratedlocalscores. No extraheadfit or posthocprimaryswitch.',
        'endpoints': 'Terminal fullmethod only after globalfullschedule/local30/calibration10whereapplicable/selector100. Global scheduledbest isauxiliary, not fullmethodbest. OriginalnativeGT/canonicalandindependentNumPy scoring fromsavedpredictions.',
        'cost': '50warmup+first32fixeddevimages3rounds/B1, actualglobalencoder+optionalonefineencoder+localhead/merge/selector. Decodedraster through nativepostproc; decode excluded, trainingbanksresident, external/cardloadrecorded.',
        'prior_work': 'LoRA borrowedmatrixfactorization https://arxiv.org/abs/2106.09685 (abstractverified; language-modelpaper, not sonarvalidation). Conditionalposition/globalFiLM andhalo taskreadout are thiscandidate, no noveltycertification.',
        'storage': 'No newweights/featurefiles. GPU-RAMbanks only; destructivelyreuse localpatchembeddingstorage for finalfine32features afteralllocaltraining. Requiredpred/gainlabels/logs/meta only.',
        'caps': {'global_seconds': 28800, 'workflow_seconds': 14400, 'RSS_bytes': 48*1024**3, 'new_disk_bytes': 4*1024**3},
        'GPU_caps_bytes': parent['GPU_caps_bytes'], 'required_free_MiB': parent['required_free_MiB'],
        'CPU_affinity': {'2': AVAILABLE_CPUS[:4], '3': AVAILABLE_CPUS[4:8]},
        'schedule': 'CPUstagedqueue now. MarineGPU2 onlyafterparentFSODactualworkerwait0 andnormaladmission; no furtherparentGPU2phases. NewUATDGPU3 onlyafteractualparentforegroundwait0. NewFSODGPU2 afternewMarine. Acrossbothunits maxoneownedworker/card,2total.',
        'release_receipts': 'ROOT atomicallypublishes release_Marine_GPU2.json from actualparentFSODwait0; avoids concurrentlyreading mutableparentqueueJSON. Parentactualforegroundwait.json is likewisepublished atomically byROOT after retainedsession99147 actuallyreturns0. Neither receipt grants protocol/math changes.',
        'stop': 'Realerror/nonfinite/hardresources/deadlines stopentireunit. No score-basedstop/automaticretry/batchprecisioncapchanges.',
        'total_global_updates': 43000, 'total_local_updates': 16440,
        'acceptance': 'All3learnedterminals exceedqualifiedSWiSE and arewithin.01 of max(strongestcompatiblecompletedlegacymean,completedparentlearned/globalterminals,ownnewglobalbaseline). Prior singleheadseed0 observations remain separately labeled; no significance/worldSOTA/noveltyclaim.',
        'train_geometry_source': parent['train_geometry_source'],
        'comparison_limit': 'Versuscompletedconvpasshalo this changesattentionadaptation andMarinefusion/localobjective together, so differences frompriorrun do not isolate a cause. Completepaths/pretraining/budgets differ fromstrongspecialists.'}
    for task in protocol['tasks']:
        folder = RUN/task; folder.mkdir()
        for name in ('local_actions.npy', 'selector_split.npz'):
            (folder/name).write_bytes((PARENT/task/name).read_bytes())
    core.old.write(RUN/'protocol.json', protocol)
    active = {}; waits = []; marine_started = False; uatd_started = False
    def launch(task, gpu):
        admission = core.old.admission(gpu)
        if admission['free_MiB'] < protocol['required_free_MiB'][task]:
            raise RuntimeError(task+' fine-bank memory admission insufficient '+str(admission))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'XFORMERS_DISABLED': '1'}
        cpus = ','.join(map(str, protocol['CPU_affinity'][str(gpu)]))
        log = (RUN/'logs'/f'{task}.log').open('w')
        child = subprocess.Popen(['/usr/bin/taskset', '-c', cpus, core.old.PYTHON, '-B', '-u', '-m',
            'runs.gsr_attention_fixed', task], cwd=core.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[task] = (child, log, gpu, time.time(), admission)
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'observed_unix': time.time(),
            'workers': {name: value[0].pid for name, value in active.items()}, 'new_weight_files': 0,
            'resource_staging': protocol['schedule']})
        core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {name: value[0].pid for name, value in active.items()}, 'actual_waits': waits})
    try:
        core.old.write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'observed_unix': time.time(),
            'workers': {}, 'status': 'CPU_STAGED_NO_NEW_MODEL', 'resource_staging': protocol['schedule']})
        core.old.write(RUN/'queue_state.json', {'status': 'CPU_STAGED_WAIT_PARENT_FSOD_ACTUAL_EXIT0', 'active': {}, 'actual_waits': []})
        while active or not marine_started or not uatd_started:
            if (PARENT/'failure_closure.json').exists():
                raise RuntimeError('Parentfixedunit stopped; no automaticparentrepair or successorcontinuation')
            release = RUN/'release_Marine_GPU2.json'
            if not marine_started and release.exists():
                finished = json.loads(release.read_text())
                if finished['task'] != 'FSOD' or finished['actual_wait_exit'] != 0:
                    raise RuntimeError('ParentFSOD actualexitnonzero; stagedGPU2 launchcancelled')
                launch('Marine', 2); marine_started = True
            if not uatd_started and (PARENT/'actual_foreground_wait.json').exists():
                wait = json.loads((PARENT/'actual_foreground_wait.json').read_text())
                if wait['actual_foreground_exit'] != 0:
                    raise RuntimeError('Parentactualforegroundnonzero; stagedGPU3 launchcancelled')
                core.old.write(RUN/'parent_actual_foreground_wait.json', wait)
                launch('UATD', 3); uatd_started = True
            if time.time() > protocol['deadline_unix']:
                raise TimeoutError('Fixed8h dense-prediction deadline')
            for task, (child, log, gpu, started, admission) in list(active.items()):
                if time.time()-started > protocol['caps']['workflow_seconds']:
                    os.killpg(child.pid, signal.SIGKILL)
                if child.poll() is None:
                    continue
                code = child.wait(); log.close(); del active[task]
                waits.append({'task': task, 'PID': child.pid, 'physical_GPU': gpu, 'actual_wait_exit': code,
                              'wall_seconds': time.time()-started, 'admission': admission})
                if code != 0:
                    raise RuntimeError(task+' actual worker exit '+str(code))
                if task == 'Marine':
                    launch('FSOD', 2)
                core.old.write(RUN/'queue_state.json', {'status': 'RUNNING', 'active': {k: v[0].pid for k, v in active.items()}, 'actual_waits': waits})
            time.sleep(1)
        tasks = {task: json.loads((RUN/task/'results.json').read_text()) for task in protocol['tasks']}
        core.old.write(RUN/'results.json', {'status': 'COMPLETE_FIXED_ATTENTION_PREDICTIONS', 'tasks': tasks})
        size = sum(path.stat().st_size for path in RUN.rglob('*') if path.is_file())
        if size > protocol['caps']['new_disk_bytes']:
            raise RuntimeError('Results-only4GiB cap exceeded')
        closure = {'status': 'COMPLETE_SCOPE_GOAL_REQUIRES_RESULT_DECISION', 'actual_waits': waits,
                   'wall_seconds': time.time()-begun, 'new_entity_bytes': size, 'new_weights_saved': False}
        core.old.write(RUN/'wave_closure.json', closure); core.old.write(RUN/'writing_handoff.json', closure)
        core.old.write(RUN/'queue_state.json', {'status': 'COMPLETE', 'actual_waits': waits})
    except BaseException:
        failure = traceback.format_exc()
        for task, (child, log, gpu, started, admission) in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            code = child.wait(); log.close()
            waits.append({'task': task, 'PID': child.pid, 'actual_wait_exit': code, 'cancelled_after_failure': True})
        core.old.write(RUN/'failure_closure.json', {'status': 'STOPPED_NO_AUTOMATIC_RETRY', 'traceback': failure, 'actual_waits': waits})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['pipeline', 'Marine', 'FSOD', 'UATD'])
    args = parser.parse_args(); core.old.runtime()
    if args.phase == 'pipeline':
        pipeline()
    else:
        workflow(args.phase)
