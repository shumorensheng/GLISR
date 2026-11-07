"""Retain the shared source in controller RAM and isolate fixed target jobs."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import random
import select
import signal
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from runs import gsr_multidata_training as core

BASE = core.BASE
OLD = BASE/'gsr_multidata_expansion_v1'
RUN = BASE/'gsr_multidata_runtime_v2'
SOCKET = RUN/'source_state.sock'
write = core.write
SOURCE_PHASE_DEADLINE = 0.
JOB_GPU_CAP = 0
GLOBAL_DEADLINE = 0.
legacy_targets, legacy_scores = core.targets_for, core.scores


def guard(deadline):
    values = {'RSS_bytes': core.resource.getrusage(core.resource.RUSAGE_SELF).ru_maxrss*1024,
              'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > min(deadline, SOURCE_PHASE_DEADLINE, GLOBAL_DEADLINE):
        raise TimeoutError('Inherited fixed job/global deadline')
    if values['RSS_bytes'] > 48*1024**3 or values['GPU_reserved_bytes'] > JOB_GPU_CAP:
        raise RuntimeError('Declared job resource cap '+str(values))
    return values


def read_bytes(connection, n):
    pieces = []
    while n:
        part = connection.recv(min(n, 1024**2))
        if not part:
            raise EOFError('Source RAM connection closed during transfer')
        pieces.append(part);n -= len(part)
    return b''.join(pieces)


def get_source():
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(120);connection.connect(str(SOCKET));connection.sendall(b'G')
        n = int.from_bytes(read_bytes(connection, 8),'big')
        return pickle.loads(read_bytes(connection,n))


def targets_for(task, spec, split):
    if task.startswith('MarineClass_'):
        return torch.from_numpy(np.load(spec['labels'][split],allow_pickle=False)).long()
    return legacy_targets(task,spec,split)


def scores(task, spec, prediction, labels, split):
    if not task.startswith('MarineClass_'):
        return legacy_scores(task,spec,prediction,labels,split)
    k = spec['classes'];truth = labels.numpy();guess = prediction.argmax(1)
    matrix = np.bincount(truth*k+guess,minlength=k*k).reshape(k,k)
    tp = matrix.diagonal().astype(np.float64);support = matrix.sum(1);predicted = matrix.sum(0)
    f1 = 2*tp/np.maximum(support+predicted,1)
    result = {'macro_f1':float(f1.mean()),'accuracy':float(tp.sum()/matrix.sum()),'per_class_f1':f1.tolist(),
        'support':support.tolist(),'confusion_matrix':matrix.tolist(),'class_names':spec['class_names']}
    return result,{'confusion_matrix':matrix.tolist(),'support':support.tolist()}


def fit_classification(task,arm,initial,p,spec):
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    schedule = p['target_schedules'][task];folder = core.RUN/arm/task;folder.mkdir(parents=True)
    model = core.GSR(p['encoder'])
    if arm == 'shared_GSR':model.load_state_dict(initial,strict=True)
    core.reader.configure_encoder(model.encoder,'e2e');model.selector.requires_grad_(False)
    model.fusion.requires_grad_(arm == 'shared_GSR');model.cuda()
    head = core.new_head(task,spec).cuda()
    body_opt = core.reader.body_optimizer(model.encoder);head_opt = core.reader.head_optimizer('Marine',head)
    extra = torch.optim.AdamW(model.fusion.parameters(),lr=.0001,weight_decay=.0001) if arm == 'shared_GSR' else None
    images = {s:torch.from_numpy(np.load(spec['views'][s],mmap_mode='r',allow_pickle=False)).cuda() for s in ('train','validation','test')}
    labels = {s:targets_for(task,spec,s) for s in images}
    generator = torch.Generator().manual_seed(0);routing = np.random.RandomState(700)
    policy = 'learned' if arm == 'shared_GSR' else 'global';best = -1.;updates = 0;begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1,101):
            model.train();model.selector.eval();head.train();order = torch.randperm(len(images['train']),generator=generator);total = 0.
            for ids in order.split(64):
                guard(SOURCE_PHASE_DEADLINE);body_opt.zero_grad(set_to_none=True);head_opt.zero_grad(set_to_none=True)
                if extra is not None:
                    extra.zero_grad(set_to_none=True);extra.param_groups[0]['lr'] = core.previous.shared.native.module_lr(updates,schedule['updates'],.0001)
                for group in body_opt.param_groups:group['lr'] = core.reader.body_learning_rate(updates,schedule['updates'])
                actions = torch.as_tensor(routing.randint(0,4,len(ids)),device='cuda') if arm == 'shared_GSR' and routing.rand()<.5 else None
                field = model.field(images['train'][ids.cuda()],head,'classification',policy,actions)
                loss = torch.nn.functional.cross_entropy(head(field.mean((2,3))),labels['train'][ids].cuda());loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),1.,error_if_nonfinite=True)
                if extra is not None:
                    torch.nn.utils.clip_grad_norm_(model.fusion.parameters(),1.,error_if_nonfinite=True);extra.step()
                body_opt.step();head_opt.step();updates += 1;total += float(loss.detach())*len(ids)
                if updates == 1:write(folder/'first_counted_update.json',{'updates':1,'batch':len(ids),'head_classes':spec['classes'],'body_gradient_norm':float(norm),'source_cohort':p['source_cohort'] if arm == 'shared_GSR' else None})
            metric,independent,pred = core.evaluate(task,spec,model,head,images['validation'],labels['validation'],64,SOURCE_PHASE_DEADLINE,policy)
            payload = {'prediction':pred,'epoch':epoch,'task':task,'arm':arm,'policy':policy}
            if metric['macro_f1']>best:
                best,best_epoch,best_metric = metric['macro_f1'],epoch,metric;torch.save(payload,folder/'best_predictions.pt')
            if epoch == 100:
                terminal = metric;torch.save(payload,folder/'terminal_predictions.pt')
            entry = {'epoch':epoch,'updates':updates,'loss':total/len(order),'macro_f1':metric['macro_f1'],'wall_seconds':time.monotonic()-begun}
            log.write(json.dumps(entry)+'\n');log.flush()
            if epoch%10 == 0:print(json.dumps(entry),flush=True)
    variants = {policy:terminal}
    if arm == 'shared_GSR':
        for other in ('global','random','fixed0','fixed1','fixed2','fixed3','uncertainty'):
            metric,_,pred = core.evaluate(task,spec,model,head,images['validation'],labels['validation'],64,SOURCE_PHASE_DEADLINE,other)
            variants[other] = metric;torch.save({'prediction':pred,'task':task,'epoch':100,'policy':other},folder/(other+'_terminal_predictions.pt'))
    test,independent,pred = core.evaluate(task,spec,model,head,images['test'],labels['test'],64,SOURCE_PHASE_DEADLINE,policy,'test')
    torch.save({'prediction':pred,'task':task,'epoch':100,'policy':policy,'split':'test'},folder/'heldout_terminal_predictions.pt')
    write(folder/'results.json',{'status':'COMPLETE_PUBLISHED_CLASSIFICATION_CONDITION','task':task,'arm':arm,'role':spec['role'],
        'metric':'macro_f1','higher_is_better':True,'epochs':100,'updates':updates,'terminal':terminal,'best':best_metric,'best_epoch':best_epoch,
        'terminal_policy_metrics':variants,'heldout_terminal_only':test,'heldout_split':'test','source_cohort':p['source_cohort'] if arm == 'shared_GSR' else None,
        'encoder_images':model.encoder_images,'new_weight_files':0,'wall_seconds':time.monotonic()-begun,**guard(SOURCE_PHASE_DEADLINE)})


def worker(job_id):
    global SOURCE_PHASE_DEADLINE,JOB_GPU_CAP,GLOBAL_DEADLINE
    p = json.loads((RUN/'protocol.json').read_text());job = json.loads((RUN/'jobs'/job_id/'job.json').read_text())
    SOURCE_PHASE_DEADLINE = job['source_deadline_unix'] if job['kind']=='source' else job['deadline_unix']
    GLOBAL_DEADLINE = p['deadline_unix'];JOB_GPU_CAP = job['GPU_cap_GiB']*1024**3
    core.RUN = RUN/'jobs'/job_id;core.guard = guard;core.targets_for = targets_for;core.scores = scores
    core.original.runtime();random.seed(0);np.random.seed(0);torch.manual_seed(0)
    tasks = json.loads(Path(job['data_manifest']).read_text())['tasks']
    if job['kind'] == 'source':
        model = core.GSR(p['encoder']).cuda();state = core.source_training(model,p,tasks)
        blob = pickle.dumps(state,protocol=5)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.settimeout(120);connection.connect(str(SOCKET));connection.sendall(b'P'+len(blob).to_bytes(8,'big')+blob)
            if read_bytes(connection,1)!=b'K':raise RuntimeError('Source snapshot not acknowledged')
        write(core.RUN/'source_snapshot_published.json',{'source_cohort':p['source_cohort'],'serialized_bytes':len(blob),'weight_files':0})
        SOURCE_PHASE_DEADLINE = min(GLOBAL_DEADLINE,time.time()+p['diagnostic_seconds_remaining'])
        core.diagnostics(model,p)
    else:
        initial = get_source() if job['arm'] == 'shared_GSR' else None
        if job['task'].startswith('MarineClass_'):fit_classification(job['task'],job['arm'],initial,p,tasks[job['task']])
        else:core.fit(job['task'],job['arm'],initial,p,tasks[job['task']])


def pipeline():
    oldp = json.loads((OLD/'protocol.json').read_text());data = json.loads((RUN/'model_inputs.json').read_text())
    hdf = json.loads((OLD/'classification_inputs/manifest.json').read_text())
    for name,case in hdf['cases'].items():
        task = 'MarineClass_'+name;rows = {}
        for split,part in case['parts'].items():
            indices = np.load(part['indices'],allow_pickle=False);labels = np.load(part['labels'],allow_pickle=False)
            rows[split] = [{'row_id':Path(case['source']).name+'/'+part['published_array']+'/'+str(int(i)),'classID':int(y)} for i,y in zip(indices,labels)]
        data['tasks'][task] = {'kind':'classification','role':'source_related_downstream_diagnostic','classes':len(case['classes']),
            'class_names':case['classes'],'rows':rows,'counts':{s:len(v) for s,v in rows.items()},'normalization':case['normalization'],
            'views':{s:p['view'] for s,p in case['parts'].items()},'labels':{s:p['labels'] for s,p in case['parts'].items()},
            'identity_scope':'Published HDF condition separately; cross-version/crop/segmentation identity unknown, not an independent source or domain.'}
    for task,spec in data['tasks'].items():
        if spec['kind']!='detection':continue
        labels={a['category_id'] if 'annotations'in row else a['class_id'] for rows in spec['rows'].values()
                for row in rows for a in (row['annotations'] if 'annotations'in row else row['boxes_xyxy'])}
        if not labels.issubset(set(range(1,spec['classes']+1))):
            raise ValueError(('Detection annotation/head vocabulary mismatch',task,labels,spec['classes']))
    write(RUN/'model_inputs.json',data)
    closure = json.loads((OLD/'executor_failure_closure.json').read_text())
    olddiag = json.loads((OLD/'diagnostics/UXO_results.json').read_text())['wall_seconds']
    schedule = dict(oldp['target_schedules'])
    for task,spec in data['tasks'].items():
        if task.startswith('MarineClass_'):schedule[task] = {'epochs':100,'batch':64,'updates':100*((spec['counts']['train']+63)//64),'seed':0}
    p = {**oldp,'status':'FROZEN_REPAIRED_COHORT_AND_CONCURRENT_JOBS_BEFORE_FORWARD','source_entry':str(Path(__file__).resolve()),
        'source_cohort':'gsr_multidata_runtime_v2/fresh_source_1','data_manifest':str(RUN/'model_inputs.json'),
        'source_seconds_remaining':10800-closure['source_elapsed_launch_to_selector_report_seconds'],
        'diagnostic_seconds_remaining':3600-olddiag,'target_schedules':schedule,
        'source_lineage':'New source fit from same source protocol/plan, because old RAM state was lost. No old checkpoint recovery/bitwise claim. Every new GSR target receives identical immutable controller-RAM bytes. Old GSR F/M retained as historical, not mixed with this cohort.',
        'source_socket':str(SOCKET),'source_weight_storage':'Controller RAM only, local Unix socket transfer; no neural weight files.',
        'physical_GPUs':[2,3],'source_model_GPU':2,
        'dispatch':'Fresh source producer onGPU2 publishes immutableCPUstate tocontroller RAM; independent target jobs onGPU2/3, max6/card subject to memory admission. No coupled cancellation on a target failure; no automatic retry. Source bytes remain available in controller RAM after model completion until explicit release or originaldeadline.',
        'category_fix':str(RUN/'mini_category_fix.json'),'resource_authorization':'Latest user objective explicitly removes one-worker/card and requests maximum useful GPU memory; only physical2/3. Frozen old model math/results/budgets untouched.',
        'resource_policy':'Max6 workers/card subject to declared live reservations, >=8GiB physical free margin, actual NV free memory and host-memory admission. One CFC full-fit/card, no interference with unknown own/foreign jobs. Batch/epochs/precision fixed. Target failures isolated; no automatic retries. Controller retains source RAM for explicit repair requests until original shared36h deadline.',
        'deadline_unix':oldp['deadline_unix'],'new_disk_cap_bytes':48*1024**3,
        'inherited_elapsed_seconds':closure['elapsed_from_CPU_plan_seconds'] if 'elapsed_from_CPU_plan_seconds'in closure else json.loads((OLD/'failure.json').read_text())['elapsed_from_CPU_plan_seconds'],
        'HDF_classification':'100epochs/B64, Linear384toK of mean32, originalDINO bodyAdamW2e-5/warmup200/cosine, headAdamW.001/.0001, GSRfusion.0001; source selector fixed. Published test only at terminal, earliest devF1 best auxiliary, separate12/11class vocabularies and versions. No source use of classification labels.',
        'repair_scope':'Clip name-based Mini class mapping; RAM source retention and per-target workers/concurrent resource binding; three already-authorized classification diagnostics. All other source/head/loss/schedules/GT/norm/prediction thresholds unchanged.'}
    p.pop('source_release_file')
    p['source_gain_supervision']=p['source_gain_supervision'].replace('No non-source label access.','Common metadata contains non-source annotations; none are used in source/gain/selector fitting.')
    (RUN/'source_plan.npz').symlink_to(OLD/'source_plan.npz')
    (RUN/'jobs').mkdir();(RUN/'requests').mkdir()
    old_bytes = sum(f.stat().st_size for f in OLD.rglob('*') if f.is_file() and not f.is_symlink())
    write(RUN/'parent_entity_bytes.json',{'old_root_bytes_once':old_bytes,'symlinks_not_new_allocations':True})
    tasks = list(core.TARGETS)+['MarineClass_'+n for n in hdf['cases']]
    jobs = [{'id':'source','kind':'source','GPU_cap_GiB':32,'seconds':p['source_seconds_remaining']+p['diagnostic_seconds_remaining'],'source_seconds':p['source_seconds_remaining']}]
    inherited = {}
    for task in core.TARGETS:
        f = BASE/'gsr_shared_complete_v1/original_DINO'/task/'results.json' if task in ('FSOD','Marine') else OLD/'original_DINO'/task/'results.json'
        if f.exists():inherited[task] = str(f)
    for task in tasks:
        for arm in ('shared_GSR','original_DINO'):
            if arm == 'original_DINO' and task in inherited:continue
            k = data['tasks'][task]['kind'];g = arm == 'shared_GSR'
            memory = 36 if task == 'CFC22' else 22 if g and (k == 'classification' or task == 'UATD') else 12 if k == 'classification' else 20 if task == 'UATD' else 16 if g else 12
            seconds = 1800 if task.startswith('MarineClass_') else oldp['fit_caps'][task]
            jobs.append({'id':arm+'_'+task,'kind':'target','task':task,'arm':arm,'GPU_cap_GiB':memory,'seconds':seconds})
    # Start the expensive new CFC control early; source and smaller controls can overlap.
    jobs.sort(key=lambda j:0 if j['kind']=='source' else 1 if j['arm']=='original_DINO' and j['task']=='CFC22' else 2 if j['arm']=='original_DINO' else 3 if j['task']=='CFC22' else 4)
    for job in jobs:job['data_manifest']=p['data_manifest']
    p['new_GSR_heads']=sum(j.get('arm')=='shared_GSR' for j in jobs)
    p['new_baseline_heads']=sum(j.get('arm')=='original_DINO' for j in jobs)
    p['inherited_baseline_heads']=len(inherited)
    p['planned_new_target_updates']=sum(schedule[j['task']]['updates'] for j in jobs if j['kind']=='target')
    p['model_freeze_unix']=time.time()
    write(RUN/'protocol.json',p)
    write(RUN/'job_manifest.json',{'jobs':jobs,'inherited_DINO':inherited,'all_GSR_new_coherent_source':True})
    listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(SOCKET));os.chmod(SOCKET,0o600);listener.listen(16)
    gpu_totals={gpu:int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip()) for gpu in (2,3)}
    pending,active,completed,failed = list(jobs),{},[],[];blob = None;processed_requests = set();failure_history=[]
    completion_written=False
    cpus = sorted(os.sched_getaffinity(0));next_cpu_slot = 0
    try:
        while pending or active or failed or blob is not None:
            if time.time()>p['deadline_unix']:
                if not pending and not active:break
                raise TimeoutError('Original shared36h deadline')
            if (RUN/'stop_source_owner').exists() and not pending and not active:break
            live_bytes=sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
            if old_bytes+live_bytes>48*1024**3:raise RuntimeError('Inherited+new entity48GiB cap')
            ready,_,_=select.select([listener],[],[],.2)
            if ready:
                connection,_=listener.accept()
                with connection:
                    connection.settimeout(120);command=read_bytes(connection,1)
                    if command==b'P':
                        if blob is not None:raise RuntimeError('Source snapshot published twice')
                        blob=read_bytes(connection,int.from_bytes(read_bytes(connection,8),'big'));connection.sendall(b'K')
                        write(RUN/'source_RAM_receipt.json',{'source_cohort':p['source_cohort'],'serialized_bytes':len(blob),'immutable_source_bytes_for_all_jobs':True,'weight_files':0,'published_unix':time.time()})
                    elif command==b'G':connection.sendall(len(blob).to_bytes(8,'big')+blob)
                    else:raise ValueError('Unknown source RAM command')
            for job_id,(child,log,gpu,job,started) in list(active.items()):
                if time.time()>job['deadline_unix'] and child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                if child.poll() is None:continue
                code=child.wait();log.close();del active[job_id]
                item={'job':job,'PID':child.pid,'physical_GPU':gpu,'actual_wait_exit':code,'actual_wall_seconds':time.time()-started}
                (completed if code==0 else failed).append(item)
                if code!=0:failure_history.append(item)
                write(RUN/'actual_job_waits.json',{'completed':completed,'failed_history':failure_history,'unresolved':[f['job']['id'] for f in failed]})
            for request in sorted((RUN/'requests').glob('*.json')):
                if request.name in processed_requests:continue
                job=json.loads(request.read_text())
                if job['task'] not in tasks or job['source_cohort']!=p['source_cohort']:raise ValueError('Repair request source/task mismatch')
                pending.append(job);processed_requests.add(request.name)
                failed=[f for f in failed if f['job']['id']!=job['repairs_job']]
            for job in list(pending):
                if job['kind']=='target' and job['arm']=='shared_GSR' and blob is None:continue
                if job['kind']=='source' and blob is not None:continue
                candidates=[]
                for gpu in (2,3):
                    own=[v for v in active.values() if v[2]==gpu]
                    if len(own)>=6:continue
                    if job['kind']=='source' and gpu!=2:continue
                    if job.get('task')=='CFC22' and any(v[3].get('task')=='CFC22' for v in own):continue
                    reserved=sum(v[3]['GPU_cap_GiB'] for v in own)
                    if reserved+job['GPU_cap_GiB']>gpu_totals[gpu]/1024-8:continue
                    total,free=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
                    known={v[0].pid for v in active.values()}
                    listed=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
                    unknown=[]
                    for pid in listed:
                        try:
                            if Path('/proc/'+pid).stat().st_uid==os.getuid() and int(pid) not in known:unknown.append(int(pid))
                        except FileNotFoundError:continue
                    if unknown:continue
                    if reserved+job['GPU_cap_GiB']>total/1024-8 or free<job['GPU_cap_GiB']*1024+8192:continue
                    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
                    if available<32*1024**3:continue
                    candidates.append((reserved,gpu,free))
                if not candidates:continue
                reserved,gpu,free=min(candidates);job=dict(job);job['deadline_unix']=min(p['deadline_unix'],time.time()+job['seconds'])
                if job['kind']=='source':job['source_deadline_unix']=min(p['deadline_unix'],time.time()+job['source_seconds'])
                folder=RUN/'jobs'/job['id'];folder.mkdir();write(folder/'job.json',job)
                write(folder/'admission.json',{'physical_GPU':gpu,'free_MiB_observed':free,'reserved_GiB_before_launch':reserved,
                    'new_job_GPU_cap_GiB':job['GPU_cap_GiB'],'unknown_same_UID_processes':[],'foreign_processes_not_interfered':True,'idle_claim':False})
                if job['kind']=='source':(folder/'source_plan.npz').symlink_to(OLD/'source_plan.npz')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log=(folder/'stdout.log').open('w')
                child=subprocess.Popen([core.original.PYTHON,'-B','-u','-m','runs.gsr_multidata_runtime_v2','worker',job['id']],cwd=core.original.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                offset=(next_cpu_slot*4)%len(cpus);os.sched_setaffinity(child.pid,cpus[offset:offset+4]);next_cpu_slot+=1
                active[job['id']]=(child,log,gpu,job,time.time());pending.remove(next(j for j in pending if j['id']==job['id']))
            status='RUNNING' if pending or active else 'FAILED_PARTIAL_SOURCE_RAM_RETAINED' if failed else 'MODELS_COMPLETE_SOURCE_RAM_RETAINED'
            write(RUN/'queue_state.json',{'status':status,
                'supervisor_PID':os.getpid(),'active':{k:{'PID':v[0].pid,'GPU':v[2]} for k,v in active.items()},'pending':[j['id'] for j in pending],
                'completed':len(completed),'failed':[f['job']['id'] for f in failed],'source_in_RAM':blob is not None})
            if status=='MODELS_COMPLETE_SOURCE_RAM_RETAINED' and not completion_written:
                write(RUN/'completion.json',{'status':status,'actual_job_waits':completed,'failed_history':failure_history,'inherited_DINO':inherited,
                    'shared_source_cohort':p['source_cohort'],'new_weight_files':0,'elapsed_from_original_CPU_plan_seconds':time.time()-p['started_unix'],
                    'source_owner_scope':'No GPU allocation in this controller. Source RAM remains available until explicit stop_source_owner or original deadline; no weights on disk.'})
                completion_written=True
            time.sleep(1)
        write(RUN/'source_owner_exit.json',{'actual_children_completed':len(completed),'unresolved_failures':[f['job']['id'] for f in failed],
            'failed_history':failure_history,'source_RAM_released':True})
        write(RUN/'queue_state.json',{'status':'COMPLETE' if not failed else 'FAILED_PARTIAL','completed':len(completed),'failed':[f['job']['id'] for f in failed]})
    finally:
        for child,log,gpu,job,started in active.values():
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            child.wait();log.close()
        listener.close();SOCKET.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('pipeline','worker'));parser.add_argument('job_id',nargs='?')
    args=parser.parse_args();pipeline() if args.phase=='pipeline' else worker(args.job_id)
