"""Official ByteTrack/OC-SORT on four existing CFC26 detector outputs, CPU only."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

RUN=Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))/'cfc26_mature_trackers_v1'
ROOT=Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path[:0]=[p for p in [str(RUN/'dependencies'), str((Path(__file__).resolve().parents[1] / 'third_party')/'ByteTrack'), str((Path(__file__).resolve().parents[1] / 'third_party')/'OC_SORT')] if Path(p).exists()]
import numpy as np
import torch
from runs import cfc26_tracking_interface as tracking


def write(path, value):
    temporary=Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def candidate(key, tracker_name, threshold, split):
    torch.set_num_threads(4)
    protocol=json.loads((RUN/'protocol.json').read_text())
    source=Path(protocol['detectors'][key][split])
    payload=torch.load(source,map_location='cpu',weights_only=False)
    rows=tracking.load_rows(split)
    if payload['row_ids']!=[r['row_id'] for r in rows]:
        raise RuntimeError('Original detector row identities differ from frozen split')
    detections=payload['native_detections']
    folder=RUN/split/f'{key}_{tracker_name}_{threshold:.2f}'
    folder.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['sequence_id']].append(i)
    tracks=[None]*len(rows)
    begun=time.monotonic()
    for indices in groups.values():
        indices.sort(key=lambda i:rows[i]['frame_index'])
        if tracker_name=='ByteTrack':
            from yolox.tracker.byte_tracker import BYTETracker
            from yolox.tracker.basetrack import BaseTrack
            BaseTrack._count=0
            engine=BYTETracker(SimpleNamespace(track_thresh=threshold,track_buffer=30,match_thresh=.8,mot20=False),frame_rate=30)
        else:
            from trackers.ocsort_tracker.ocsort import OCSort
            engine=OCSort(det_thresh=threshold,max_age=30,min_hits=3,iou_threshold=.3,delta_t=3,asso_func='iou',inertia=.2,use_byte=False)
        for i in indices:
            det=detections[i]
            keep=np.asarray(det['labels'])==1
            data=np.column_stack((np.asarray(det['boxes'])[keep],np.asarray(det['scores'])[keep])).astype(np.float64,copy=True)
            hw=tuple(rows[i]['actual_shape_HWC'][:2])
            emitted=engine.update(data,hw,hw)
            if tracker_name=='ByteTrack':
                tracks[i]={'boxes':np.asarray([t.tlbr for t in emitted],dtype=np.float64).reshape(-1,4),
                           'ids':np.array([t.track_id for t in emitted],dtype=np.int64),
                           'scores':np.array([t.score for t in emitted])}
            else:
                tracks[i]={'boxes':emitted[:,:4].copy(),'ids':emitted[:,4].astype(np.int64)}
    metrics=tracking.score(rows,tracks)
    result={'status':'COMPLETE','detector':key,'tracker':tracker_name,'threshold':threshold,'split':split,
            'input_predictions':str(source),'metrics':metrics,'seconds':time.monotonic()-begun,
            'model_forwards':0,'sequence_resets':len(groups),'input_frames':len(rows)}
    if split=='test':
        torch.save({'row_ids':payload['row_ids'],'tracks':tracks,'input_predictions':str(source),
                    'tracker':tracker_name,'threshold':threshold,'OC_SORT_score_note':'Official OC-SORT outputs boxes and IDs only; input confidence preserved in input_predictions.'},folder/'tracks.pt')
    write(folder/'result.json',result)
    print(json.dumps({'status':'COMPLETE','detector':key,'tracker':tracker_name,'threshold':threshold,'split':split,
                      **{k:metrics[k] for k in ['HOTA','IDF1','MOTA']}}),flush=True)


def batch(jobs, deadline, phase):
    pending=list(jobs);active={};waits=[]
    while pending or active:
        while pending and len(active)<4 and time.time()<deadline:
            key,tracker,threshold,split=pending.pop(0)
            name=f'{key}_{tracker}_{threshold:.2f}'
            folder=RUN/split/name;folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'stdout.log').open('w')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'}
            child=subprocess.Popen([sys.executable,'-m','runs.cfc26_mature_trackers','candidate','--detector',key,'--tracker',tracker,'--threshold',str(threshold),'--split',split],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=(child,log,min(time.time()+600,deadline),folder)
        for name,(child,log,end,folder) in list(active.items()):
            status=child.poll()
            if status is None and time.time()>end:
                os.killpg(child.pid,signal.SIGKILL);status=child.wait()
            if status is not None:
                log.close();row={'job':name,'PID':child.pid,'actual_wait_exit':status};write(folder/'actual_wait.json',row);waits.append(row);del active[name]
                print(json.dumps({'phase':phase,**row}),flush=True)
        if time.time()>=deadline and pending:
            write(RUN/(phase+'_not_started.json'),pending);pending=[]
        write(RUN/'queue_state.json',{'phase':phase,'active':{k:v[0].pid for k,v in active.items()},'pending':len(pending),'actual_waits':waits})
        if active:time.sleep(1)
    return waits


def run_all():
    protocol=json.loads((RUN/'protocol.json').read_text())
    deadline=time.time()+7200
    write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'started_unix':time.time(),'deadline_unix':deadline,'GPU_allocations':0})
    jobs=[(key,tr,t,'val') for key in protocol['detectors'] for tr in ['ByteTrack','OC_SORT'] for t in protocol['thresholds']]
    valwait=batch(jobs,deadline,'validation')
    if len(valwait)!=32 or any(r['actual_wait_exit']!=0 for r in valwait):
        write(RUN/'completion.json',{'status':'VALIDATION_FAILED_TEST_NOT_STARTED','actual_waits':valwait});return
    results=[json.loads(p.read_text()) for p in (RUN/'val').glob('*/result.json')]
    write(RUN/'all_validation_results.json',results)
    selected=[]
    for key in protocol['detectors']:
        for tracker in ['ByteTrack','OC_SORT']:
            choices=sorted([r for r in results if r['detector']==key and r['tracker']==tracker],key=lambda r:protocol['thresholds'].index(r['threshold']))
            best=max(choices,key=lambda r:r['metrics']['HOTA'])
            selected.append({'detector':key,'tracker':tracker,'threshold':best['threshold'],'validation_metrics':best['metrics']})
    write(RUN/'selection.json',{'locked_before_test_association':True,'created_unix':time.time(),'selection':'highest full validation HOTA; first threshold wins ties','selected':selected})
    testwait=batch([(r['detector'],r['tracker'],r['threshold'],'test') for r in selected],deadline,'test')
    testresults=[json.loads(p.read_text()) for p in (RUN/'test').glob('*/result.json')]
    status='COMPLETE' if len(testwait)==8 and all(r['actual_wait_exit']==0 for r in testwait) else 'TEST_FAILED_OR_INCOMPLETE'
    write(RUN/'results.json',{'status':status,'selection':selected,'test_results':testresults,'model_forwards':0,'note':'All four detector paths reported; no detector/tracker choice made on test scores.'})
    write(RUN/'completion.json',{'status':status,'validation_actual_waits':valwait,'test_actual_waits':testwait,'GPU_allocations':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','candidate']);p.add_argument('--detector');p.add_argument('--tracker',choices=['ByteTrack','OC_SORT']);p.add_argument('--threshold',type=float);p.add_argument('--split',choices=['val','test'])
    a=p.parse_args()
    if a.phase=='run':run_all()
    else:candidate(a.detector,a.tracker,a.threshold,a.split)
