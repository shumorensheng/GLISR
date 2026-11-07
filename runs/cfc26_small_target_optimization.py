"""Two bounded CFC26 native-window refinements; existing results stay read-only."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
from scipy.optimize import linear_sum_assignment

from runs import glisr_cfc26_train as native
from runs import cfc26_tracking_interface as tracking
from runs.cfc_independent_ap50 import canonical_evaluate, independent_ap50

ROOT = Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))
RUN = tracking.BASE / 'cfc26_small_target_optimization_v1'
OLD = tracking.BASE / 'glisr_supplement_20261104_v1'
TOTAL = 17500
TILE = 512
write = native.write


def origins(length):
    return sorted(set(list(range(0, length-TILE+1, 384)) + [length-TILE]))


def crop_boxes(row, origin):
    x, y = origin
    boxes = np.asarray([a['bbox'] for a in row['annotations']], dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes -= np.array([x, y, x, y], dtype=np.float32)
    boxes = np.clip(boxes, 0, TILE)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    return torch.from_numpy(boxes) * (224/TILE)


def crop_input(gray, origin, method, normalization):
    x, y = map(int, origin)
    if method == 'Specialist':
        resized = Image.fromarray(gray[y:y+TILE, x:x+TILE]).resize((224,224), Image.Resampling.BILINEAR)
        return (torch.from_numpy(np.array(resized,copy=True)).float()[None]/255).expand(3,-1,-1).contiguous()
    tile = torch.from_numpy(np.array(gray[y:y+TILE, x:x+TILE], copy=True)).float()[None, None]/255
    tile = (tile.log1p()-normalization['mean'])/normalization['std']
    tile = F.interpolate(tile, (224,224), mode='bilinear', align_corners=False, antialias=True)[0]
    return tile


class Crops(Dataset):
    def __init__(self, method):
        self.rows = tracking.load_rows('train')
        plan = np.load(RUN/'training_plan.npz')
        self.order, self.xy = plan['order'], plan['xy']
        self.method = method
        self.norm = json.loads((tracking.RUN/'normalization.json').read_text())

    def __len__(self):
        return 20*7000

    def __getitem__(self, index):
        epoch, position = divmod(index, 7000)
        rid = int(self.order[epoch, position])
        row, xy = self.rows[rid], self.xy[epoch, rid]
        with Image.open(row['path']) as image:
            gray = np.asarray(image.convert('L'))
        boxes = crop_boxes(row, xy)
        return crop_input(gray, xy, self.method, self.norm), {'boxes':boxes, 'labels':torch.ones(len(boxes), dtype=torch.int64)}, rid, xy.tolist()


def collate(items):
    images, targets, ids, xy = zip(*items)
    return torch.stack(images), list(targets), list(ids), list(xy)


def velocity_sequence(detections):
    tracks, next_id, output = {}, 1, []
    for frame, detection in enumerate(detections):
        keep = (detection['scores'] >= .25) & (detection['labels'] == 1)
        boxes, scores = detection['boxes'][keep], detection['scores'][keep]
        tracks = {k:v for k,v in tracks.items() if frame-v['frame'] <= 3}
        ids = list(tracks)
        assigned = np.zeros(len(boxes),dtype=np.int64)
        if ids and len(boxes):
            predicted = [tracks[k]['box'] + np.tile(tracks[k]['velocity']*(frame-tracks[k]['frame']),2) for k in ids]
            similarity = tracking.box_iou(predicted, boxes)
            ii,jj = linear_sum_assignment(np.where(similarity>=.3,1-similarity,1e6))
            for i,j in zip(ii,jj):
                if similarity[i,j] >= .3:
                    assigned[j] = ids[i]
        for j,box in enumerate(boxes):
            identity = int(assigned[j])
            velocity = np.zeros(2)
            if identity:
                last = tracks[identity]
                velocity = ((box[:2]+box[2:])-(last['box'][:2]+last['box'][2:])) / (2*(frame-last['frame']))
            else:
                identity, next_id = next_id, next_id+1
                assigned[j] = identity
            tracks[identity] = {'box':box.copy(),'frame':frame,'velocity':velocity}
        output.append({'boxes':boxes,'ids':assigned,'scores':scores})
    return output


def velocity_associate(rows, detections):
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        groups[row['sequence_id']].append(i)
    output = [None]*len(rows)
    for indices in groups.values():
        indices.sort(key=lambda i:rows[i]['frame_index'])
        for i,pred in zip(indices,velocity_sequence([detections[i] for i in indices])):
            output[i] = pred
    return output


def detection_scores(rows, detections):
    coco = {'info':{},'images':[],'annotations':[],'categories':[{'id':1,'name':'fish'}]}
    predictions = []
    for i,(row,det) in enumerate(zip(rows,detections)):
        h,w = row['actual_shape_HWC'][:2]
        coco['images'].append({'id':i,'width':w,'height':h})
        for annotation in row['annotations']:
            box = annotation['bbox']
            coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':i,'category_id':1,'bbox':box,'area':box[2]*box[3],'iscrowd':0})
        for box,score in zip(det['boxes'],det['scores']):
            predictions.append({'image_id':i,'category_id':1,'bbox':[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])],'score':float(score)})
    canonical = canonical_evaluate(coco,predictions)
    independent = independent_ap50(coco,predictions)
    difference = abs(canonical['AP50']-independent['AP50'])
    if difference > 1e-9:
        raise RuntimeError(f'AP50 scorer discrepancy {difference}')
    return {'AP50':canonical['AP50'],'AP50_95':canonical['stats'][0],
            'AR100':canonical['stats'][8],'gt_count':canonical['gt_count'],
            'independent_AP50':independent['AP50'],'maximum_AP50_difference':difference}


def score_detections(rows, detections):
    return {'detection':detection_scores(rows,detections),
            'IoU':tracking.score(rows,tracking.associate(rows,detections)),
            'velocity':tracking.score(rows,velocity_associate(rows,detections))}


def guard(deadline):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    gpu = torch.cuda.max_memory_reserved()
    if time.time()>deadline:
        raise TimeoutError('12h actual worker deadline')
    if rss>64*1024**3 or gpu>48*1024**3:
        raise RuntimeError(f'Actual resource cap exceeded: RSS={rss}, GPU={gpu}')
    return {'RSS_bytes':rss,'GPU_reserved_bytes':gpu}


@torch.no_grad()
def evaluate(model, head, method, split, folder, tag, deadline):
    model.eval()
    if head is not None:
        head.eval()
    rows = tracking.load_rows(split)
    norm = json.loads((tracking.RUN/'normalization.json').read_text())
    detections, window_counts = [], []
    before_K = dict(model.K_histogram) if method=='Rtau' else {}
    started = time.monotonic()
    with (folder/(tag+'_windows.jsonl')).open('w') as metadata:
        for i,row in enumerate(rows):
            h,w = row['actual_shape_HWC'][:2]
            windows = [(x,y) for y in origins(h) for x in origins(w)]
            with Image.open(row['path']) as image:
                gray = np.asarray(image.convert('L'))
            all_boxes, all_scores = [], []
            for start in range(0,len(windows),16):
                chunk = windows[start:start+16]
                images = torch.stack([crop_input(gray,xy,method,norm) for xy in chunk]).cuda()
                results = model(list(images)) if method=='Specialist' else head(model.field(images,head,'detection'))
                for det,(x,y) in zip(results,chunk):
                    box = det['boxes']*(TILE/224)+det['boxes'].new_tensor([x,y,x,y])
                    all_boxes.append(box)
                    all_scores.append(det['scores'])
                guard(deadline)
            boxes, scores = torch.cat(all_boxes),torch.cat(all_scores)
            keep = nms(boxes,scores,.5)[:300]
            detections.append({'boxes':boxes[keep].float().cpu().numpy(),'scores':scores[keep].float().cpu().numpy(),'labels':np.ones(len(keep),dtype=np.int64)})
            window_counts.append(len(windows))
            metadata.write(json.dumps({'row_id':row['row_id'],'native_HW':[h,w],'origins_xy':windows})+'\n')
            if i==0 or (i+1)%100==0:
                print(json.dumps({'phase':tag,'frames':i+1,'total':len(rows),'seconds':time.monotonic()-started}),flush=True)
    metrics = score_detections(rows,detections)
    metrics.update({'split':split,'frames':len(rows),'tile_views':sum(window_counts),'window_count_min':min(window_counts),
                    'window_count_max':max(window_counts),'seconds':time.monotonic()-started,'resources':guard(deadline)})
    if method=='Rtau':
        metrics['K_histogram'] = {k:v-before_K.get(k,0) for k,v in model.K_histogram.items() if v>before_K.get(k,0)}
    payload = {'row_ids':[r['row_id'] for r in rows],'native_detections':detections,'split':split,'tag':tag}
    torch.save(payload,folder/(tag+'_predictions.pt'))
    write(folder/(tag+'_metrics.json'),metrics)
    return metrics


def save(folder, name, method, model, head, state, optimizers):
    payload = {'method':method,'model':model.state_dict(),'head':None if head is None else head.state_dict(),
               'progress':state,'protocol':str(RUN/'protocol.json')}
    if name=='last':
        payload.update({'optimizers':[o.state_dict() for o in optimizers], 'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':__import__('random').getstate()})
    temporary = folder/(name+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(folder/(name+'.pt'))


def worker(method, deadline):
    folder = RUN/method
    folder.mkdir(exist_ok=True)
    protocol = json.loads((RUN/'protocol.json').read_text())
    old_protocol = json.loads((OLD/'protocol.json').read_text())
    native.setup(0)
    init = protocol['initializations'][method]
    model,head = native.construct(method,init['old_seed'],old_protocol)
    saved = torch.load(init['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True)
    if head is not None:
        head.load_state_dict(saved['head'],strict=True)
        head = head.cuda()
    model = model.cuda()
    optimizers = native.optimizers_for(method,model,head)
    peaks = [.001] if method=='Specialist' else [2e-6,.001,1e-5]
    state = {'epoch':0,'updates':0,'best_HOTA':-1.,'best_epoch':None,'inherited_updates':saved['progress']['updates']}
    del saved
    write(folder/'initialization.json',{'method':method,'checkpoint':init['checkpoint'],'inherited_updates':state['inherited_updates'],
          'old_seed':init['old_seed'],'new_sampling_seed':0,'optimizer_reset':True,'actual_parameters':sum(p.numel() for p in model.parameters())+ (0 if head is None else sum(p.numel() for p in head.parameters())),
          'actual_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)+(0 if head is None else sum(p.numel() for p in head.parameters() if p.requires_grad))})
    old_val = torch.load(Path(init['checkpoint']).parent/'val_epoch60_predictions.pt',map_location='cpu',weights_only=False)
    val_rows = tracking.load_rows('val')
    if old_val['row_ids'] != [r['row_id'] for r in val_rows]:
        raise RuntimeError('Existing validation prediction identities differ')
    write(folder/'original_whole224_validation.json',score_detections(val_rows,old_val['native_detections']))
    baseline = evaluate(model,head,method,'val',folder,'tiled_before_adaptation',deadline)
    state['best_HOTA'] = -1.
    data = Crops(method)
    loader = DataLoader(data,batch_size=8,num_workers=4,collate_fn=collate,pin_memory=True,multiprocessing_context='spawn')
    begun = time.monotonic()
    training_K = defaultdict(int)
    with (folder/'training.jsonl').open('w') as log:
        for batch,(images,targets,ids,xy) in enumerate(loader):
            epoch = batch//875+1
            if batch%875==0:
                model.train()
                if head is not None:
                    head.train()
            for opt,peak in zip(optimizers,peaks):
                step = state['updates']+1
                mult = step/200 if step<=200 else .01+.99*(1+math.cos(math.pi*(step-200)/(TOTAL-200)))/2
                for group in opt.param_groups:
                    group['lr']=peak*mult
                opt.zero_grad(set_to_none=True)
            x = images.cuda(non_blocking=True)
            targets = [{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
            loss = native.loss_for(method,model,head,x,targets)
            if method=='Rtau':
                for k in model.last_counts.tolist():
                    training_K[str(k)] += 1
            loss.backward()
            norms = native.clip_gradients(method,optimizers)
            for opt in optimizers:
                opt.step()
            state['updates']+=1
            state['epoch']=epoch
            resources = guard(deadline)
            if state['updates']==1:
                write(folder/'first_counted_update.json',{'updates':1,'native_origins_xy':xy,'train_row_indices':ids,
                      'input_shape':list(x.shape),'target_box_counts':[len(t['boxes']) for t in targets],
                      'loss':float(loss.detach()),'gradient_norms':norms,'resources':resources,
                      'GLISR_sparse_witness':model.first_sparse_witness if method=='Rtau' else None})
            if state['updates']==1 or state['updates']%100==0:
                record={'epoch':epoch,'updates':state['updates'],'loss':float(loss.detach()),'gradient_norms':norms,'elapsed_seconds':time.monotonic()-begun}
                log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            if state['updates']%875==0:
                if epoch%5==0:
                    metrics=evaluate(model,head,method,'val',folder,f'val_epoch{epoch:02}',deadline)
                    if metrics['velocity']['HOTA']>state['best_HOTA']:
                        state['best_HOTA']=metrics['velocity']['HOTA'];state['best_epoch']=epoch
                        save(folder,'best',method,model,head,state,optimizers)
                save(folder,'last',method,model,head,state,optimizers)
    if state['updates']!=TOTAL:
        raise RuntimeError('Incomplete fixed training schedule')
    save(folder,'terminal',method,model,head,state,optimizers)
    terminal=evaluate(model,head,method,'test',folder,'test_terminal',deadline)
    write(folder/'results.json',{'status':'COMPLETE','method':method,'new_epochs':20,'new_updates':TOTAL,
           'inherited_updates':state['inherited_updates'],'initialization':init,'best_epoch':state['best_epoch'],
           'best_validation_HOTA':state['best_HOTA'],'before_adaptation_validation':baseline,
           'terminal_test':terminal,'training_K_histogram':dict(training_K),
           'protocol':str(RUN/'protocol.json'),'resources':guard(deadline)})


def score_saved(method):
    folder=RUN/method
    checks=[]
    for path in sorted(folder.glob('*_predictions.pt')):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        rows=tracking.load_rows(payload['split'])
        if payload['row_ids'] != [r['row_id'] for r in rows]:
            raise RuntimeError(f'Saved prediction identity mismatch {path}')
        actual=score_detections(rows,payload['native_detections'])
        expected=json.loads(path.with_name(path.name.replace('_predictions.pt','_metrics.json')).read_text())
        for key in ['AP50','AP50_95']:
            if abs(actual['detection'][key]-expected['detection'][key])>1e-9:
                raise RuntimeError(f'Detection scoring mismatch {path}')
        for tracker in ['IoU','velocity']:
            for key in ['HOTA','IDF1','MOTA','IDs','FP','FN']:
                if abs(actual[tracker][key]-expected[tracker][key])>1e-12:
                    raise RuntimeError(f'Tracking scoring mismatch {path}')
        checks.append({'path':str(path),'status':'PASS_ORIGINAL_GT_RELOADED_PREDICTIONS'})
    write(folder/'verification.json',{'status':'PASS','endpoints':checks,'model_forwards':0})


def prepare():
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Existing run already prepared')
    rows=tracking.load_rows('train')
    rng=np.random.RandomState(731)
    g=torch.Generator().manual_seed(0)
    order=np.stack([torch.randperm(7000,generator=g).numpy() for _ in range(20)])
    xy=np.empty((20,7000,2),dtype=np.int32)
    for e in range(20):
        for i,r in enumerate(rows):
            h,w=r['actual_shape_HWC'][:2]
            xy[e,i]=[rng.randint(w-TILE+1),rng.randint(h-TILE+1)]
    np.savez(RUN/'training_plan.npz',order=order,xy=xy)
    # One geometry regression covers an actual retained annotation and the image edge.
    row=next(r for r in rows if r['annotations'])
    box=np.array(row['annotations'][0]['bbox'],dtype=float);box[2:]+=box[:2]
    h,w=row['actual_shape_HWC'][:2]
    origin=np.maximum(0,np.minimum([w-TILE,h-TILE],np.floor((box[:2]+box[2:])/2-TILE/2))).astype(int)
    cropped=crop_boxes(row,origin).numpy()
    restored=cropped*(TILE/224)+np.tile(origin,2)
    original=np.array([a['bbox'] for a in row['annotations']],dtype=np.float32);original[:,2:]+=original[:,:2]
    clipped=np.minimum(np.maximum(original,np.tile(origin,2)),np.tile(origin+TILE,2))
    clipped=clipped[(clipped[:,2]>clipped[:,0])&(clipped[:,3]>clipped[:,1])]
    if not np.allclose(restored,clipped,atol=1e-4,rtol=0):
        raise RuntimeError('Crop coordinate roundtrip failed')
    def det(x):return {'boxes':np.array([[x,0,x+10,10]],dtype=float),'scores':np.ones(1),'labels':np.ones(1,dtype=int)}
    empty={'boxes':np.empty((0,4)),'scores':np.empty(0),'labels':np.empty(0,dtype=int)}
    seq=[det(0),det(3),det(6),empty,empty,det(15)]
    cv=velocity_sequence(seq);iou=tracking.associate_sequence(seq)
    if int(cv[-1]['ids'][0])!=1 or int(iou[-1]['ids'][0])==1:
        raise RuntimeError('Constant velocity gap regression failed')
    write(RUN/'geometry_check.json',{'status':'PASS','actual_row_id':row['row_id'],'native_origin_xy':origin.tolist(),
          'maximum_coordinate_error':float(np.abs(restored-clipped).max()),'velocity_kept_ID':1,'old_IoU_new_ID':int(iou[-1]['ids'][0])})
    write(RUN/'protocol.json',{'status':'FROZEN_BEFORE_GPU_FORWARD','created_unix':time.time(),'plan':str(RUN/'PLAN.md'),
          'methods':['Rtau','Specialist'],'old_protocol':str(OLD/'protocol.json'),'new_seed':0,'new_epochs':20,'updates_per_model':TOTAL,
          'tile512_stride384':True,'native_crop_then_original224':True,'normalization':str(tracking.RUN/'normalization.json'),
          'initializations':{'Rtau':{'checkpoint':str(OLD/'CFC26_Rtau_s0/terminal.pt'),'old_seed':0},
                             'Specialist':{'checkpoint':str(OLD/'CFC26_Specialist_s2/terminal.pt'),'old_seed':2}},
          'GPU_assignments':{'Rtau':2,'Specialist':3},'worker_seconds':43200,'global_seconds':46800,
          'best_selection':'earliest strict validation constant-velocity HOTA every5epochs; terminal20 test primary',
          'tracking':{'score':.25,'IoU':.3,'max_missing':2,'velocity':'last two observations, no learned parameters'},
          'tile_merge':{'NMS':.5,'max_detections':300},'test_exposure':'Previously examined development test; no blind confirmation claim'})
    shutil.copyfile(__file__,RUN/'entry_snapshot.py')


def supervisor():
    protocol=json.loads((RUN/'protocol.json').read_text())
    deadline=time.time()+46800
    active={};finished={}
    for method,gpu in protocol['GPU_assignments'].items():
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<24576:
            raise RuntimeError(f'GPU{gpu} free {free}MiB <24GiB; no allocation')
        folder=RUN/method;folder.mkdir(exist_ok=True)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONUNBUFFERED':'1'}
        log=(folder/'stdout.log').open('w')
        end=min(time.time()+43200,deadline)
        child=subprocess.Popen([sys.executable,'-m','runs.cfc26_small_target_optimization','worker','--method',method,'--deadline',str(end)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[method]=(child,log,end,'train')
        write(folder/'launch_receipt.json',{'PID':child.pid,'physical_GPU':gpu,'deadline_unix':end,'protocol':str(RUN/'protocol.json')})
    write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':{m:c.pid for m,(c,_,_,_) in active.items()},'started_unix':time.time(),'global_deadline_unix':deadline})
    while active:
        for method,(child,log,end,phase) in list(active.items()):
            status=child.poll()
            if status is None and time.time()>end:
                os.killpg(child.pid,signal.SIGKILL);status=child.wait()
            if status is not None:
                log.close();write(RUN/method/(phase+'_actual_wait.json'),{'PID':child.pid,'actual_wait_exit':status})
                finished.setdefault(method,{})[phase+'_exit']=status
                if phase=='train' and status==0:
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
                    f=(RUN/method/'score_stdout.log').open('w')
                    scored=subprocess.Popen([sys.executable,'-m','runs.cfc26_small_target_optimization','score','--method',method],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    active[method]=(scored,f,min(time.time()+900,deadline),'score')
                else:
                    del active[method]
        write(RUN/'queue_state.json',{'status':'RUNNING' if active else 'CLOSED','active':{m:{'PID':c.pid,'phase':phase} for m,(c,_,_,phase) in active.items()},'finished':finished})
        if active:time.sleep(5)
    write(RUN/'completion.json',{'status':'COMPLETE' if all(v=={'train_exit':0,'score_exit':0} for v in finished.values()) else 'CLOSED_WITH_FAILURES','actual_waits':finished})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','supervisor','worker','score']);p.add_argument('--method',choices=['Rtau','Specialist']);p.add_argument('--deadline',type=float)
    args=p.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='supervisor':supervisor()
    elif args.phase=='score':torch.set_num_threads(4);score_saved(args.method)
    else:worker(args.method,args.deadline)
