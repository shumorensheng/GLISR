"""Complete shared K1 ordinary-task-loss full fine-tuning, three downstream seeds."""
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import select
import signal
import socket
import statistics
import subprocess
import time
import traceback
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from runs import gsr_joint_pair as pair
core=pair.core
parent=pair.parent
RUN=core.BASE/'gsr_k1_complete_fullft_v1'
SOCKET=RUN/'source_state.sock'
DEADLINE=0.
write=core.write
guard=pair.guard
evaluate=core.evaluate
POLICIES=('global','random','fixed0','fixed1','fixed2','fixed3','uncertainty')

class SingleGSR(core.GSR):
    def field(self, images, head, kind, policy='learned', actions=None):
        g = self.global_field(images)
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        if policy == 'global':
            return base
        scores = self.selector(g.detach())
        background = base
        if actions is None:
            if policy == 'uncertainty':
                if kind == 'detection':
                    probability = head.rpn.head([base])[0][0].sigmoid().clamp(1e-6, 1-1e-6)
                    entropy = -(probability*probability.log()+(1-probability)*(1-probability).log()).mean(1, keepdim=True)
                elif kind == 'classification':
                    patches = F.adaptive_avg_pool2d(base, (2, 2)).flatten(2).transpose(1, 2)
                    probability = head(patches).softmax(-1).clamp_min(1e-6)
                    actions = (-(probability*probability.log()).sum(-1)).argmax(1)
                    entropy = None
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
                if entropy is not None:
                    actions = F.adaptive_avg_pool2d(entropy, (2, 2)).flatten(1).argmax(1)
            else:
                actions = scores.argmax(1)
        self.last_actions = actions.detach()
        active = torch.where(actions >= 0)[0]
        if len(active):
            large = F.interpolate(images[active], (448, 448), mode='bilinear', align_corners=False, antialias=True)
            tiles = torch.stack([large[..., r*14:r*14+224, c*14:c*14+224] for r, c in core.original.QUADS], 1)
            local = self.global_field(tiles[torch.arange(len(active), device=images.device), actions[active]])
            base = base.clone()
            base[active] = self.corrected(g[active], local, actions[active])
        if self.training and len(active):
            probability = scores.softmax(1)[active].gather(1, actions[active, None]).flatten()
            correction = (probability-probability.detach())[:, None, None, None]*(base[active]-background[active]).detach()
            output = base.clone(); output[active] = base[active]+correction
            return output
        return base


def fit(task, seed, job_id, protocol, initial):
    spec = json.loads(Path(protocol['data_manifest']).read_text())['tasks'][task]
    schedule = protocol['target_schedules'][task]; folder = RUN/job_id
    model = SingleGSR(protocol['encoder'])
    model.load_state_dict(initial, strict=True)
    policy = 'learned'
    core.reader.configure_encoder(model.encoder, 'e2e')
    model.fusion.requires_grad_(True); model.selector.requires_grad_(True); model.cuda()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if task in ('FSOD', 'Marine', 'UATD', 'NKSID'):
            head = core.reader.new_head(task, seed)
        else:
            head = nn.Conv2d(384, 1 if spec['kind'] == 'point_regression' else spec['classes'], 1)
    head = head.cuda()
    body_opt = core.reader.body_optimizer(model.encoder)
    head_opt = core.reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    fusion_opt = torch.optim.AdamW(model.fusion.parameters(), lr=.0001, weight_decay=.0001)
    route_opt = torch.optim.AdamW(model.selector.parameters(), lr=.0001, weight_decay=.0001)
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r', allow_pickle=False)).cuda() for s in ('train', 'validation')}
    labels = {s: core.targets_for(task, spec, s) for s in images}
    generator = torch.Generator().manual_seed(seed); routing = np.random.RandomState(700+seed)
    key = core.metric_key(task, spec); direction = -1 if spec['kind'] == 'point_regression' else 1
    best = -float('inf'); updates = 0; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs']+1):
            model.train(); head.train(); order = torch.randperm(len(spec['rows']['train']), generator=generator)
            if spec['kind'] == 'detection':
                head_opt.param_groups[0]['lr'] = .005*.1**sum(epoch > m for m in (40, 55))
            total = 0.; counts = np.zeros(4, dtype=np.int64)
            for ids in order.split(schedule['batch']):
                guard(DEADLINE)
                for optimizer in (body_opt, head_opt, fusion_opt, route_opt): optimizer.zero_grad(set_to_none=True)
                for group in body_opt.param_groups:
                    group['lr'] = core.reader.body_learning_rate(updates, schedule['updates'])
                for optimizer in (fusion_opt, route_opt):
                    optimizer.param_groups[0]['lr'] = core.previous.shared.native.module_lr(updates, schedule['updates'], .0001)
                actions = torch.as_tensor(routing.randint(0, 4, len(ids)), device='cuda') if routing.rand() < .5 else None
                field = model.field(images['train'][ids.cuda()], head, 'uatd' if task == 'UATD' else spec['kind'], policy, actions=actions)
                target = core.batch_targets(task, spec, labels['train'], ids.tolist())
                loss = core.task_loss(task, spec, head, field, target); loss.backward()
                body_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
                route_norm = torch.nn.utils.clip_grad_norm_(model.selector.parameters(), 1., error_if_nonfinite=True)
                for optimizer in (body_opt, head_opt, fusion_opt, route_opt): optimizer.step()
                updates += 1; total += float(loss.detach())*len(ids)
                counts += torch.bincount(model.last_actions, minlength=4).cpu().numpy()
                if updates == 1:
                    write(folder/'first_counted_update.json', {'updates': 1, 'first_ids': ids.tolist(),
                        'selected_region_ids': model.last_actions.tolist(), 'field': list(field.shape),
                        'body_gradient_norm': float(body_norm), 'selector_gradient_norm': float(route_norm),
                        'encoder_images': model.encoder_images, 'source_cohort': protocol['source_cohort'],
                        'ordinary_task_loss_only': True, 'backward_scope': 'Biased straight-through selected-residual surrogate'})
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order),
                'selected_region_counts': counts.tolist(), 'wall_seconds': time.monotonic()-begun}
            every = protocol['evaluation_every_epochs'][task]
            if epoch % every == 0:
                metric, independent, pred = evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], DEADLINE, policy)
                entry[key] = metric[key]
                payload = {'prediction': pred, 'task': task, 'epoch': epoch, 'policy': policy, 'local_budget': 1}
                if direction*metric[key] > best:
                    best, best_epoch, best_metric = direction*metric[key], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = metric, independent
                    torch.save(payload, folder/'terminal_predictions.pt')
            log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)
    policies = {policy: terminal}
    for policy in POLICIES:
        metric, independent, pred = evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], DEADLINE, policy)
        policies[policy] = metric
        torch.save({'prediction': pred, 'task': task, 'epoch': schedule['epochs'], 'policy': policy, 'local_budget': 1}, folder/(policy+'_terminal_predictions.pt'))
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
    write(folder/'results.json', {'status': 'COMPLETE_FIXED_PAIRED_TARGET_REPEAT', 'task': task, 'metric': key,
        'higher_is_better': direction == 1, 'epochs': schedule['epochs'], 'updates': updates,
        'terminal': terminal, 'terminal_independent': terminal_independent, 'best': best_metric, 'best_epoch': best_epoch,
        'terminal_policy_metrics': policies, 'heldout_terminal_only': held, 'heldout_split': held_split,
        'source_cohort': protocol['source_cohort'], 'arm': 'K1', 'seed': seed, 'local_budget': 1, 'regions': core.original.QUADS,
        'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'new_weight_files': 0, 'wall_seconds': time.monotonic()-begun, **guard(DEADLINE)})


def worker(job_id):
    global DEADLINE
    p=json.loads((RUN/'protocol.json').read_text());job=json.loads((RUN/job_id/'job.json').read_text())
    DEADLINE=pair.DEADLINE=job['deadline_unix'];core.guard=guard
    core.original.runtime();seed=job['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as c:
            c.settimeout(120);c.connect(str(SOCKET));c.sendall(b'G')
            size=int.from_bytes(parent.read_bytes(c,8),'big');initial=pickle.loads(parent.read_bytes(c,size))
        fit(job['task'],seed,job_id,p,initial)
    except BaseException:
        write(RUN/job_id/'failure.json',{'traceback':traceback.format_exc(),'time_unix':time.time(),'no_retry':True})
        raise


def summarize(p,waits):
    reference=json.loads((core.BASE/'gsr_joint_pair_finetune_repeats_v1/results.json').read_text())
    tasks={}
    for task in pair.TARGETS:
        rows=[]
        for seed in (0,1,2):
            path=RUN/f's{seed}_{task}'/'results.json';d=json.loads(path.read_text());key=d['metric']
            baseline=reference['tasks'][task]['per_seed'][seed]['DINO']
            rows.append({'seed':seed,'K1_terminal':d['terminal'][key],'K1_best':d['best'][key],'best_epoch':d['best_epoch'],
                'DINO':baseline,'terminal_difference':d['terminal'][key]-baseline['terminal'],'evidence':str(path)})
        delta=[v['terminal_difference'] for v in rows]
        tasks[task]={'per_seed':rows,'paired_terminal_mean':statistics.mean(delta),'paired_terminal_sample_SD':statistics.stdev(delta),
            'higher_is_better':d['higher_is_better'],'reference_family':reference['tasks'][task]['historical_references']}
    write(RUN/'results.json',{'status':'COMPLETE_18_K1_FULL_PARAMETER_FITS','tasks':tasks,'actual_waits':waits,
        'scope':'Same shared source K1, downstream seeds only; no source repeats. All original DINO, selector, fusion and fresh head parameters optimized subject to existing unused-mask exclusion. Terminal primary, no seed selection.'})


def pipeline():
    started=time.time();deadline=started+28800
    source=json.loads((parent.RUN/'protocol.json').read_text());old=json.loads((pair.RUN/'protocol.json').read_text())
    jobs=[{'id':f's{s}_{t}','seed':s,'task':t} for t in ('NKSID','HighResolution','Diver','Marine','UATD','FSOD') for s in (0,1,2)]
    dependencies=[core.BASE/'gsr_k_attention_depth_v1'/j/'ready_for_cost.json' for j in ('m3_FSOD','m6_FSOD')]
    dependencies.append(core.BASE/'gsr_k_attention_training_burst_v1/m9_FSOD/ready_for_cost.json')
    RUN.mkdir()
    p={k:old[k] for k in ('encoder','data_manifest','target_schedules','evaluation_every_epochs','references')}
    p.update({'status':'FROZEN_BEFORE_FIRST_FORWARD','started_unix':started,'deadline_unix':deadline,'source_cohort':source['source_cohort'],
        'entry':str(Path(__file__).resolve()),'jobs':jobs,'new_fits':18,'new_updates':146400,
        'source':'Existing complete immutable K1 source RAM; no retraining, same source for every downstream fit.',
        'gradient':'Ordinary task loss only. Hard one-of-four forward, biased selected-residual straight-through router gradient as existing K2; no claim of exact derivative of argmax. All backbone/fusion/selector/head optimizers enabled; existing unused mask exclusion.',
        'schedule':'Same six target schedules and evaluation cadence as K2/DINO repeats; terminal primary, earliest strict best auxiliary; no direction-based stop.',
        'source_roles':'FSOD/Marine/Xing detection+segmentation/LSOD/CFC22 source; UATD/NKSID/HighResolution/Diver excluded from source training.',
        'waiting_for':[str(x) for x in dependencies],'wait_reason':'Avoid new contention while three existing FSOD fits have tight frozen deadlines.',
        'resource':{'GPUs':[2,3],'workers_per_GPU':1,'global_seconds':28800,'job_seconds':pair.CAPS,'GPU_GiB_per_job':32,'RSS_GiB_per_job':48,'free_GPU_MiB_required':40960,'host_free_GiB_required':128,'new_disk_GiB':12,'CPU_threads':4},
        'timing_authorized':False,'new_weight_files':0,'diagram':'Same K1 forward graph; ordinary task-loss updates original encoder, selector ST surrogate, fusion and task head.'})
    write(RUN/'protocol.json',p)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as c:
        c.settimeout(120);c.connect(str(parent.SOCKET));c.sendall(b'G')
        size=int.from_bytes(parent.read_bytes(c,8),'big');blob=parent.read_bytes(c,size)
    write(RUN/'source_RAM_receipt.json',{'source_cohort':p['source_cohort'],'bytes':size,'weight_files':0,'same_blob_for_all_workers':True})
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(SOCKET));os.chmod(SOCKET,0o600);listener.listen(4)
    pending=list(jobs);active={};waits=[];launches={};cpus=sorted(os.sched_getaffinity(0))
    write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':launches,'status':'CPU_SOURCE_RETAINED_WAITING_EXISTING_FITS'})
    try:
        while pending or active:
            if time.time()>deadline:raise TimeoutError('K1 fullFT eight-hour deadline including queue wait')
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>12*1024**3:raise RuntimeError('K1 fullFT12GiB disk cap')
            ready,_,_=select.select([listener],[],[],.2)
            if ready:
                c,_=listener.accept()
                with c:
                    c.settimeout(120)
                    if parent.read_bytes(c,1)!=b'G':raise ValueError('Unknown source request')
                    c.sendall(len(blob).to_bytes(8,'big')+blob)
            for jid,(child,log,gpu,began,limit) in list(active.items()):
                if time.time()>limit and child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                if child.poll() is None:continue
                code=child.wait();log.close();del active[jid];waits.append({'job':jid,'PID':child.pid,'GPU':gpu,'actual_wait_exit':code,'wall_seconds':time.time()-began});write(RUN/'actual_waits.json',waits)
                if code:raise RuntimeError(f'{jid} exited {code}; no automatic retry')
            if any((x.parent/'failure.json').is_file() for x in dependencies):raise RuntimeError('Existing FSOD dependency failed; no automatic new allocation')
            deps_ready=all(x.is_file() for x in dependencies)
            if deps_ready:
                for gpu in (2,3):
                    if not pending or any(v[2]==gpu for v in active.values()):continue
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                    if free<40960 or available<128*1024**3:continue
                    job=pending.pop(0);jid=job['id'];folder=RUN/jid;folder.mkdir();limit=min(deadline,time.time()+pair.CAPS[job['task']])
                    write(folder/'job.json',{**job,'GPU':gpu,'deadline_unix':limit,'admission_free_MiB':free,'shared_GPU':True})
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    log=(folder/'stdout.log').open('w');child=subprocess.Popen([core.original.PYTHON,'-B','-u','-m','runs.gsr_k1_complete_fullft','worker',jid],cwd=core.original.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    offset=120+4*(gpu-2);os.sched_setaffinity(child.pid,cpus[offset:offset+4])
                    active[jid]=(child,log,gpu,time.time(),limit);launches[jid]={'PID':child.pid,'GPU':gpu}
                    write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':launches,'status':'RUNNING'})
            write(RUN/'queue_state.json',{'status':'RUNNING' if deps_ready else 'WAITING_EXISTING_FSOD_FITS','active':{j:{'PID':v[0].pid,'GPU':v[2]} for j,v in active.items()},'pending':[j['id'] for j in pending],'completed':len(waits),'timing_grants':0})
            time.sleep(2)
        summarize(p,waits);write(RUN/'completion.json',{'status':'COMPLETE','actual_waits':waits,'wall_seconds':time.time()-started,'new_weight_files':0})
        write(RUN/'queue_state.json',{'status':'COMPLETE','active':{},'pending':[],'completed':len(waits)})
    except BaseException:
        write(RUN/'failure.json',{'traceback':traceback.format_exc(),'time_unix':time.time(),'no_retry':True});raise
    finally:
        for jid,(child,log,gpu,began,limit) in active.items():
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            code=child.wait();log.close();waits.append({'job':jid,'PID':child.pid,'actual_wait_exit':code,'cleanup':True})
        write(RUN/'actual_waits.json',waits);listener.close();SOCKET.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('pipeline','worker'));parser.add_argument('job',nargs='?');args=parser.parse_args()
    if args.phase=='pipeline':pipeline()
    else:worker(args.job)
