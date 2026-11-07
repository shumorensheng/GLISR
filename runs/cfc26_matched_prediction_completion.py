"""Complete missing CFC26 inference/evaluation without repeating trained models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms

from runs import cfc26_matched_native_training as core

OLD = core.RUN
RUN = core.BASE/'cfc26_matched_prediction_completion_v1'
write = core.write


def prepare():
    RUN.mkdir(exist_ok=False)
    p = json.loads((OLD/'protocol.json').read_text())
    control = json.loads((OLD/'dispatch_control.json').read_text())
    complete, partial, unstarted = [], [], []
    for job in p['jobs']:
        folder = OLD/control['retry_outputs'].get(job['id'], job['id'])
        record = {**job, 'old_folder': str(folder)}
        if (folder/'results.json').exists():
            record.update(status='REUSE_COMPLETE', evidence=str(folder/'results.json'))
            complete.append(record)
        elif (folder/'native128/results.json').exists():
            state = json.loads((folder/'native128/results.json').read_text())['state']
            if state['epoch'] != 10 or state['updates'] != 8750:
                raise RuntimeError(f'Incomplete persisted training: {job["id"]}')
            for name in ['latest.pt', 'best.pt']:
                if not (folder/'native128'/name).is_file():
                    raise FileNotFoundError(folder/'native128'/name)
            partial.append({**record, 'status': 'PREDICTION_EVALUATION_ONLY',
                            'terminal_checkpoint': str(folder/'native128/latest.pt'),
                            'best_checkpoint': str(folder/'native128/best.pt'), 'best_epoch': state['best_epoch']})
        elif job['status'] != 'NEW':
            partial.append({**record, 'status': 'REUSE_TRAINED_RTAU0',
                'terminal_checkpoint': str(core.reference.RUN/core.reference.ARM/'latest.pt'),
                'best_checkpoint': str(core.reference.RUN/core.reference.ARM/'best.pt'),
                'best_epoch': json.loads((core.reference.RUN/core.reference.ARM/'results.json').read_text())['best_validation_epoch']})
        else:
            unstarted.append({**record, 'status': 'NOT_PREVIOUSLY_STARTED'})
    deadline = min(json.loads((OLD/'launch_receipt.json').read_text())['deadline_unix'], time.time()+86400)
    protocol = {'status': 'FIXED_BEFORE_FIRST_COMPLETION_FORWARD', 'created_unix': time.time(),
        'parent_protocol': str(OLD/'protocol.json'), 'parent_queue': str(OLD/'queue_state.json'),
        'reused_complete': complete, 'jobs': partial+unstarted,
        'priority': 'All16 trained incomplete conditions plus trainedRtau0 auxiliary first; three never-started jobs remain authorized after completion slots release. No existing trained condition is retrained.',
        'inference': 'Exact original native128 windows/stride96+last/B16/FP32/field/head/NMS. Commit one64-frame native-detection shard at a time; exact rows and checkpoint endpoint bound. Existing complete prediction files and quality results reused. Old interrupted phase held predictions only in RAM: absent persisted native detections cannot be reconstructed from coordinates, so rerun ONLY that missing phase, never training.',
        'quality': 'Same own-train-only38features/31leaves/100iterations, fixed.30; original GT/TrackEval/officialLOI and terminalprimary/bestaux unchanged. Existing completed endpoint evaluations reused.',
        'budget': {'deadline_unix': deadline, 'maximum_new_seconds': 86400,
                   'worker_seconds': 43200, 'GPU_workers': 16, 'GPUs': [0,1,2,3],
                   'minimum_GPU_free_MiB': 24576, 'minimum_host_available_GiB': 32,
                   'worker_RSS_GiB': 64, 'worker_GPU_reserved_GiB': 40, 'new_disk_GiB': 512},
        'accounting': 'Original failures/time limits/cost retained. This separately authorized bounded prediction completion does not retroactively satisfy old8h limits. No timer reset or result rewrite in original run.',
        'stop': 'Actual deadline/resource/numeric/code failure visible; stop releasing pending after failure, existing jobs retain guards. Persist completed chunks. No automatic retraining, new variants, seed/LR/epoch/precision/threshold changes.',
        'new_training_jobs': len(unstarted), 'existing_training_replayed': 0,
    }
    write(RUN/'protocol.json', protocol)
    write(RUN/'run_manifest.json', {'reused_complete': complete, 'jobs': partial+unstarted})
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    shutil.copyfile(core.__file__, RUN/'training_engine_snapshot.py')


@torch.no_grad()
def infer_chunks(model, head, method, split, folder, checkpoint, deadline):
    folder.mkdir(parents=True, exist_ok=True)
    rows = core.tracking.load_rows(split)
    binding = {'checkpoint': str(checkpoint), 'split': split, 'method': method,
               'row_ids': [r['row_id'] for r in rows], 'tile':128, 'batch':16}
    bound = folder/'binding.json'
    if bound.exists():
        if json.loads(bound.read_text()) != binding:
            raise RuntimeError('Existing inference chunks bind a different endpoint/population')
    else:
        write(bound, binding)
    model.eval().requires_grad_(False); head.eval().requires_grad_(False)
    begun = time.monotonic(); new_frames = new_calls = new_views = 0
    teacher_before = head.teacher_samples
    predictions = []
    for start in range(0, len(rows), 64):
        end = min(start+64, len(rows)); part = folder/f'frames_{start:05}_{end:05}.pt'
        if part.exists():
            saved = torch.load(part, map_location='cpu', weights_only=False)
            if saved['row_ids'] != [r['row_id'] for r in rows[start:end]]:
                raise RuntimeError('Persisted prediction chunk row mismatch')
            predictions.extend(saved['native_detections']); continue
        batch_predictions, transforms = [], []
        for row in rows[start:end]:
            with Image.open(row['path']) as image:
                pixels = np.array(image.convert('RGB'), copy=True)
            h,w = row['actual_shape_HWC'][:2]
            positions = [(x,y) for y in core.reference.rgb.origins(h,128) for x in core.reference.rgb.origins(w,128)]
            boxes, scores, counts = [], [], []
            for offset in range(0,len(positions),16):
                selected=positions[offset:offset+16]
                g,local=core.views([pixels[y:y+128,x:x+128] for x,y in selected],method)
                detected=head(core.field(model,method,g,local))
                for det,(x,y) in zip(detected,selected):
                    boxes.append(det['boxes']*(128/224)+det['boxes'].new_tensor([x,y,x,y]))
                    scores.append(det['scores'])
                if method in core.SPARSE:
                    counts.extend(model.last_counts.tolist())
                new_calls+=1;new_views+=len(selected);core.guard(deadline)
            box,score=torch.cat(boxes),torch.cat(scores);keep=nms(box,score,.25)[:300]
            batch_predictions.append({'boxes':box[keep].cpu().numpy(),'scores':score[keep].cpu().numpy(),
                                      'labels':np.ones(len(keep),dtype=np.int64)})
            transforms.append({'row_id':row['row_id'],'native_HW':[h,w],'tile':128,
                               'origins_xy':positions,'K':counts if method in core.SPARSE else None})
            new_frames+=1
        value={'row_ids':[r['row_id'] for r in rows[start:end]],'native_detections':batch_predictions,
               'checkpoint':str(checkpoint),'transforms':transforms}
        temp=part.with_suffix('.tmp');torch.save(value,temp);temp.replace(part)
        predictions.extend(batch_predictions)
        write(folder/'progress.json',{'frames_persisted':end,'total_frames':len(rows),'new_frames':new_frames,
              'seconds':time.monotonic()-begun,'checkpoint':str(checkpoint),**core.guard(deadline)})
        print('persisted',folder.parent.name,folder.name,end,len(rows),flush=True)
    if head.teacher_samples!=teacher_before:
        raise RuntimeError('Teacher ran during inference')
    path=folder/'predictions.pt';temp=folder/'predictions.tmp'
    torch.save({'row_ids':[r['row_id'] for r in rows],'native_detections':predictions,'split':split},temp);temp.replace(path)
    write(folder/'inference_receipt.json',{'frames':len(rows),'new_frames':new_frames,'calls':new_calls,
          'views':new_views,'seconds':time.monotonic()-begun,'teacher_inference_forwards':0,
          'checkpoint':str(checkpoint),**core.guard(deadline)})
    return path


def prediction(model,head,job,name,split,old,folder,checkpoint,deadline):
    original=old/name/'predictions.pt'
    if original.exists():
        write(folder/f'reuse_{name}.json',{'source':str(original),'checkpoint':str(checkpoint),
              'operation':'Reuse persisted complete original prediction file; no inference replay'})
        return original
    target=folder/name/'predictions.pt'
    if target.exists():
        return target
    return infer_chunks(model,head,job['method'],split,folder/name,checkpoint,deadline)


def worker(job_id,deadline):
    p=json.loads((RUN/'protocol.json').read_text());parent=json.loads(Path(p['parent_protocol']).read_text())
    job=next(j for j in p['jobs'] if j['id']==job_id);folder=RUN/job_id;folder.mkdir(exist_ok=True)
    begun=time.monotonic();old=Path(job['old_folder']);new_updates=0
    try:
        core.reference.rgb.native.setup(job['seed'])
        core.quality.guard=lambda end:core.guard(end)['RSS_bytes']
        write(folder/'launch.json',{'PID':os.getpid(),'GPU':os.environ['CUDA_VISIBLE_DEVICES'],
              'deadline_unix':deadline,'job':job})
        model,head=core.construct(job,parent)
        if job['status']=='NOT_PREVIOUSLY_STARTED':
            cache=core.SharedPixels(core.tracking.load_rows('train'))
            for spec in parent['stages']:
                core.train_stage(job,parent,model,head,cache,spec,folder/f"native{spec['tile']}",deadline)
                new_updates+=spec['updates']
            del cache
            old=folder
        if job['status']=='REUSE_TRAINED_RTAU0':
            last=core.reference.RUN/core.reference.ARM
            stage=json.loads((last/'results.json').read_text());best_epoch=stage['best_validation_epoch']
            profile=json.loads(Path(parent['reuse_Rtau0']['terminal_profile']).read_text())
            terminal={'status':'REUSE_COMPLETE','results':{'test':{'tracking':profile['expected_test']['tracking'],
                       'official_LOI_counts':profile['expected_test']['official_LOI_counts']}},
                       'profile':parent['reuse_Rtau0']['terminal_profile']}
        else:
            last=old/'native128';stage=json.loads((last/'results.json').read_text());best_epoch=stage['state']['best_epoch']
            terminal_file=old/'terminal_quality/results.json'
            if terminal_file.exists():
                terminal=json.loads(terminal_file.read_text())
                write(folder/'reuse_terminal_quality.json',{'source':str(terminal_file)})
            else:
                checkpoint=last/'latest.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if saved['progress']['epoch']!=10 or saved['progress']['updates']!=8750:
                    raise RuntimeError('Terminal checkpoint is not actual epoch10')
                model.load_state_dict(saved['model'],strict=True);head.load_state_dict(saved['head'],strict=True);del saved
                test=prediction(model,head,job,'terminal_test','test',old,folder,checkpoint,deadline)
                train=prediction(model,head,job,'terminal_train','train',old,folder,checkpoint,deadline)
                terminal=core.endpoint(job,parent,model,head,'terminal10',train,test,last/'val_epoch10/predictions.pt',
                                       folder/'terminal_quality',deadline)
        if best_epoch==10:
            best=terminal
        elif (old/'best_quality/results.json').exists():
            best=json.loads((old/'best_quality/results.json').read_text())
        else:
            checkpoint=last/'best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['progress']['epoch']!=best_epoch:
                raise RuntimeError('Best checkpoint epoch differs from recorded validation selection')
            model.load_state_dict(saved['model'],strict=True);head.load_state_dict(saved['head'],strict=True);del saved
            train=prediction(model,head,job,'best_train','train',old,folder,checkpoint,deadline)
            test=prediction(model,head,job,'best_test','test',last if job['status']=='REUSE_TRAINED_RTAU0' else old,
                            folder,checkpoint,deadline)
            best=core.endpoint(job,parent,model,head,f'best_epoch{best_epoch}',train,test,
                               last/f'val_epoch{best_epoch:02}/predictions.pt',folder/'best_quality',deadline)
        result={'status':'COMPLETE','job':job,'terminal':terminal,'best':best,'best_epoch':best_epoch,
                'new_target_updates':new_updates,'previous_training_replayed':0,
                'seconds':time.monotonic()-begun,**core.guard(deadline)}
        write(folder/'results.json',result)
        print(json.dumps({'COMPLETE':job_id,'seconds':result['seconds']}),flush=True)
    except Exception:
        write(folder/'failure.json',{'PID':os.getpid(),'job':job_id,'traceback':traceback.format_exc(),
                                    'seconds':time.monotonic()-begun})
        raise


def pipeline():
    p=json.loads((RUN/'protocol.json').read_text());pending=list(p['jobs']);active={};waits=[];failed=False
    deadline=p['budget']['deadline_unix'];begun=time.time();last_disk_check=begun
    write(RUN/'launch_receipt.json',{'PID':os.getpid(),'started_unix':begun,'deadline_unix':deadline,
          'reused_complete':len(p['reused_complete']),'completion_jobs':len(pending),
          'new_training_jobs':p['new_training_jobs'],'GPU_workers':16,'new_source_training':0})
    while pending or active:
        if time.time()-last_disk_check>60:
            size=sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
            last_disk_check=time.time()
            write(RUN/'disk_observation.json',{'unix':last_disk_check,'bytes':size})
            if size>p['budget']['new_disk_GiB']*1024**3:
                failed=True
                write(RUN/'disk_failure.json',{'bytes':size,'cap_GiB':p['budget']['new_disk_GiB']})
                for child,_,_,_,_,_ in active.values():
                    if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
        for slot,(child,log,job,gpu,end,start) in list(active.items()):
            code=child.poll()
            if code is None and time.time()>end:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            if code is not None:
                log.close();record={'job':job['id'],'PID':child.pid,'GPU':gpu,'actual_wait_exit':code,
                                   'seconds':time.monotonic()-start}
                waits.append(record);write(RUN/job['id']/'actual_wait.json',record)
                failed=failed or code!=0;del active[slot]
        if not failed and time.time()<deadline:
            for slot in range(16):
                if slot in active or not pending:continue
                available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                if available<32*1024**3:break
                gpu=slot%4
                info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
                if int(info)<24576:continue
                job=pending.pop(0);end=min(deadline,time.time()+p['budget']['worker_seconds'])
                dep=core.BASE/'featsharp_marine_matched_v1/build_dependencies'
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                     'OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','YOLOv5_AUTOINSTALL':'False',
                     'WANDB_MODE':'disabled','MPLBACKEND':'Agg','MAX_JOBS':'4','XFORMERS_DISABLED':'1',
                     'PATH':str(dep/'bin')+':'+os.environ['PATH'],'PYTHONPATH':str(dep)+':'+str(core.ROOT),
                     'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                log=(RUN/f"{job['id']}.log").open('w')
                child=subprocess.Popen([sys.executable,'-u','-m','runs.cfc26_matched_prediction_completion','worker',
                    '--job',job['id'],'--deadline',str(end)],cwd=core.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[slot]=(child,log,job,gpu,end,time.monotonic())
        write(RUN/'queue_state.json',{'supervisor_PID':os.getpid(),'updated_unix':time.time(),
              'active':[{'job':v[2]['id'],'PID':v[0].pid,'GPU':v[3]} for v in active.values()],
              'pending':[j['id'] for j in pending],'actual_waits':waits,'failed':failed})
        if (failed or time.time()>deadline) and not active:break
        if active or pending:time.sleep(10)
    results=[json.loads(Path(j['evidence']).read_text()) for j in p['reused_complete']]
    results += [json.loads((RUN/j['id']/'results.json').read_text()) for j in p['jobs'] if (RUN/j['id']/'results.json').exists()]
    complete=len(results)==36 and not failed
    write(RUN/'results.json',{'status':'COMPLETE' if complete else 'PARTIAL_OR_FAILED','jobs':results,
          'actual_waits':waits,'pending':[j['id'] for j in pending],'protocol':str(RUN/'protocol.json')})
    write(RUN/'completion.json',{'status':'COMPLETE' if complete else 'PARTIAL_OR_FAILED','total_results':len(results),
          'reused_results':len(p['reused_complete']),'actual_waits':waits,'seconds':time.time()-begun})
    lines=['# CFC26完整配方统一结果','','|方法|seed|末轮HOTA%|IDF1%|MOTA%|计数nMAE|最佳轮|最佳HOTA%|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(results,key=lambda r:r['job']['id']):
        m=r['terminal']['results']['test']['tracking'];nmae=next(c['nMAE'] for c in r['terminal']['results']['test']['official_LOI_counts'] if c['location']=='ALL')
        lines.append(f"|{r['job']['method']}|{r['job']['seed']}|{100*m['HOTA']:.4f}|{100*m['IDF1']:.4f}|{100*m['MOTA']:.4f}|{nmae:.6f}|{r['best_epoch']}|{100*r['best']['results']['test']['tracking']['HOTA']:.4f}|")
    (RUN/'summary.md').write_text('\n'.join(lines)+'\n')
    if not complete:sys.exit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','worker','pipeline'])
    parser.add_argument('--job');parser.add_argument('--deadline',type=float);args=parser.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='worker':worker(args.job,args.deadline)
    else:pipeline()
