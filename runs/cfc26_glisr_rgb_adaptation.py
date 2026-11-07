"""Two bounded RGB, native-detail GLISR downstream adaptations."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from runs import cfc26_tracking_interface as tracking
from runs import glisr_cfc26_train as native

BASE = tracking.BASE
RUN = BASE/'cfc26_glisr_rgb_adaptation_v1'
ROOT = Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))
INIT = BASE/'cfc26_small_target_optimization_v1/Rtau/terminal.pt'
PARENT = BASE/'glisr_supplement_20261104_v1/protocol.json'
EPOCHS = 20
UPDATES = 17500


def guard(deadline):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    reserved = torch.cuda.max_memory_reserved()
    if time.time()>deadline:
        raise TimeoutError('Fixed RGB adaptation actual deadline')
    if rss>64*1024**3 or reserved>24*1024**3:
        raise RuntimeError(f'Actual RGB resource cap: RSS={rss}, GPU={reserved}')
    return {'RSS_bytes':rss,'GPU_reserved_bytes':reserved}


def views(crops):
    pixels = torch.from_numpy(np.stack(crops)).permute(0,3,1,2).contiguous().cuda().float()/255
    mean = pixels.new_tensor([.485,.456,.406])[None,:,None,None]
    std = pixels.new_tensor([.229,.224,.225])[None,:,None,None]
    global_view = (F.interpolate(pixels,(224,224),mode='bilinear',align_corners=False,antialias=True)-mean)/std
    local_view = (F.interpolate(pixels,(448,448),mode='bilinear',align_corners=False,antialias=True)-mean)/std
    return global_view,local_view


def make_plan(rows,tile,folder):
    order_generator = torch.Generator().manual_seed(0)
    rng = np.random.RandomState(731)
    order = np.stack([torch.randperm(len(rows),generator=order_generator).numpy() for _ in range(EPOCHS)])
    xy = np.empty((EPOCHS,len(rows),2),dtype=np.int32)
    flips = np.empty((EPOCHS,len(rows)),dtype=bool)
    selected = np.full((EPOCHS,len(rows)),-1,dtype=np.int16)
    for epoch in range(EPOCHS):
        for i,row in enumerate(rows):
            h,w = row['actual_shape_HWC'][:2]
            if w<tile or h<tile:
                raise RuntimeError('Native training image smaller than fixed crop')
            if row['annotations'] and rng.rand()<.75:
                index = rng.randint(len(row['annotations']))
                bx,by,bw,bh = row['annotations'][index]['bbox']
                cx = (max(0,bx)+min(w,bx+bw))/2
                cy = (max(0,by)+min(h,by+bh))/2
                x = rng.randint(max(0,math.ceil(cx-tile)),min(w-tile,math.floor(cx))+1)
                y = rng.randint(max(0,math.ceil(cy-tile)),min(h-tile,math.floor(cy))+1)
                selected[epoch,i] = index
            else:
                x,y = rng.randint(w-tile+1),rng.randint(h-tile+1)
            xy[epoch,i] = x,y
            flips[epoch,i] = rng.rand()<.5
    np.savez(folder/'augmentation_plan.npz',order=order.astype(np.int32),xy=xy,flip=flips,selected_GT=selected)
    native.write(folder/'sampling_receipt.json',{'epochs':EPOCHS,'frames_each':len(rows),'tile':tile,
                 'GT_guided_draws':int((selected>=0).sum()),'same_private_order_seed':0,
                 'augmentation_seed':731,'first_epoch_first_IDs':order[0,:16].tolist()})
    return order,xy,flips


def save(folder,name,model,head,state,optimizers):
    data={'model':model.state_dict(),'head':head.state_dict(),'progress':dict(state),
          'protocol':str(RUN/'protocol.json'),'input_channels':3,'independent_native448':True}
    if name=='latest':
        data.update({'optimizers':[o.state_dict() for o in optimizers],
                     'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                     'numpy_rng':np.random.get_state(),'python_rng':__import__('random').getstate()})
    temporary=folder/(name+'.tmp')
    torch.save(data,temporary)
    temporary.replace(folder/(name+'.pt'))


def origins(length,tile):
    return sorted(set(list(range(0,length-tile+1,tile*3//4))+[length-tile]))


@torch.no_grad()
def evaluate(model,head,tile,split,folder,deadline,score_module):
    folder.mkdir(parents=True,exist_ok=True)
    model.eval();head.eval()
    rows=tracking.load_rows(split)
    predictions=[]
    begun=time.monotonic()
    tile_count=0
    with (folder/'input_transforms.jsonl').open('w') as metadata:
        for i,row in enumerate(rows):
            h,w=row['actual_shape_HWC'][:2]
            with Image.open(row['path']) as image:
                rgb=np.array(image.convert('RGB'),copy=True)
            positions=[(x,y) for y in origins(h,tile) for x in origins(w,tile)]
            all_boxes=[];all_scores=[]
            for start in range(0,len(positions),16):
                current=positions[start:start+16]
                crops=[rgb[y:y+tile,x:x+tile] for x,y in current]
                global_view,local_view=views(crops)
                detections=head(model.field(global_view,head,'detection',high_resolution=local_view))
                for det,(x,y),k in zip(detections,current,model.last_counts.tolist()):
                    all_boxes.append(det['boxes']*(tile/224)+det['boxes'].new_tensor([x,y,x,y]))
                    all_scores.append(det['scores'])
                    metadata.write(json.dumps({'row_id':row['row_id'],'native_HW':[h,w],
                        'origin_xy':[x,y],'tile':tile,'global224_local448_independent':True,'K':k})+'\n')
                guard(deadline)
            boxes,scores=torch.cat(all_boxes),torch.cat(all_scores)
            keep=nms(boxes,scores,.25)[:300]
            predictions.append({'boxes':boxes[keep].float().cpu().numpy(),
                'scores':scores[keep].float().cpu().numpy(),'labels':np.ones(len(keep),dtype=np.int64)})
            tile_count+=len(positions)
            if (i+1)%200==0:
                print('evaluation',folder.name,split,i+1,len(rows),flush=True)
    path=folder/'predictions.pt'
    torch.save({'row_ids':[r['row_id'] for r in rows],'native_detections':predictions,'split':split},path)
    native.write(folder/'protocol.json',{'splits':[split],'detectors':{folder.parent.name:{split:str(path)}},
        'parent_training_protocol':str(RUN/'protocol.json'),'checkpoint_endpoint':folder.name,
        'tracking_score_module':score_module,
        'official_counter_reference':str(BASE/'cfc26_author_weights_eval_v1/protocol.json')})
    native.write(folder/'inference_receipt.json',{'frames':len(rows),'native_window_views':tile_count,
        'seconds':time.monotonic()-begun,'resources':guard(deadline)})
    # Independent CPU process reloads predictions for the prescribed tracker/counter.
    env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    with (folder/'score_stdout.log').open('w') as log:
        subprocess.run([sys.executable,'-m',score_module,'score',
                       '--folder',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                       timeout=min(600,max(1,deadline-time.time())),check=True)
    return json.loads((folder/'results.json').read_text())['results'][0]


def worker(arm,deadline,construct,initialization,head_lr,selection_metric,restore_RGB_projection,teacher_training,score_module):
    protocol=json.loads((RUN/'protocol.json').read_text())
    tile=protocol['candidates'][arm]['tile']
    folder=RUN/arm;folder.mkdir(exist_ok=True)
    rows=tracking.load_rows('train')
    order,xy,flips=make_plan(rows,tile,folder)
    native.setup(0)
    parent=json.loads(PARENT.read_text())
    model,head=construct(parent)
    saved=torch.load(initialization,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);head.load_state_dict(saved['head'],strict=True)
    inherited_updates=saved['progress']['updates'];del saved
    if restore_RGB_projection:
        original=torch.load(parent['encoder']['weights_path'],map_location='cpu',weights_only=True)
        projection=nn.Conv2d(3,384,kernel_size=14,stride=14)
        projection.load_state_dict({'weight':original['patch_embed.proj.weight'],
                                   'bias':original['patch_embed.proj.bias']},strict=True)
        model.encoder.backbone.patch_embed.proj=projection;del original
    native.reader.configure_encoder(model.encoder,'e2e')
    model=model.cuda();head=head.cuda()
    optimizers=native.optimizers_for('Rtau',model,head)
    for group in optimizers[1].param_groups:
        group['params']=[p for p in group['params'] if p.requires_grad]
    native.write(folder/'initialization.json',{'checkpoint':str(initialization),'inherited_last_stage_updates':inherited_updates,
        'RGB_projection':'Restore original official DINO pretrained3channel patch projection' if restore_RGB_projection else 'Preserve actual3channel projection from initialization checkpoint',
        'actual_projection_shape':list(model.encoder.backbone.patch_embed.proj.weight.shape),'total_parameters':sum(p.numel() for p in model.parameters())+sum(p.numel() for p in head.parameters()),
        'mask_in_optimizer':any(p is model.encoder.backbone.mask_token for opt in optimizers for g in opt.param_groups for p in g['params']),
        'optimizer_reset':True,'new_stage_updates':UPDATES,'input':protocol['preprocessing']})
    cache=[];cache_bytes=0
    for i,row in enumerate(rows):
        with Image.open(row['path']) as im:
            rgb=np.array(im.convert('RGB'),copy=True)
        if list(rgb.shape)!=row['actual_shape_HWC']:
            raise RuntimeError('Original image and annotation native shape differ')
        cache.append(rgb);cache_bytes+=rgb.nbytes
        if (i+1)%500==0:
            native.write(folder/'cache_progress.json',{'frames':i+1,'RAM_bytes':cache_bytes,'resources':guard(deadline)})
            print('RGB RAM cache',arm,i+1,len(rows),flush=True)
    native.write(folder/'cache_receipt.json',{'frames':len(cache),'RAM_bytes':cache_bytes,'persisted_image_copies':0})
    state={'epoch':0,'updates':0,'best_epoch':None,'best_HOTA':-1.,'best_AP50':None,
           'selection_metric':selection_metric,'best_selection_score':-1.}
    begun=time.monotonic();training_K=Counter();validation=[]
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(EPOCHS):
            model.train();head.train()
            for start in range(0,len(rows),8):
                ids=order[epoch,start:start+8].tolist()
                crops=[];targets=[];positive=0
                for i in ids:
                    x,y=xy[epoch,i];crop=cache[i][y:y+tile,x:x+tile]
                    box=np.asarray([a['bbox'] for a in rows[i]['annotations']],dtype=np.float32).reshape(-1,4)
                    box[:,2:]+=box[:,:2]
                    box-=np.array([x,y,x,y],dtype=np.float32)
                    box=np.clip(box,0,tile)
                    box=box[(box[:,2]>box[:,0]) & (box[:,3]>box[:,1])]
                    if flips[epoch,i]:
                        crop=crop[:,::-1]
                        box[:,[0,2]]=tile-box[:,[2,0]]
                    crops.append(crop)
                    targets.append({'boxes':torch.from_numpy(box*(224/tile)).cuda(),
                                    'labels':torch.ones(len(box),dtype=torch.int64,device='cuda')})
                    positive+=int(len(box)>0)
                step=state['updates']+1
                multiplier=step/200 if step<=200 else .01+.99*(1+math.cos(math.pi*(step-200)/(UPDATES-200)))/2
                for opt,peak in zip(optimizers,[1e-5,head_lr,.0001]):
                    for group in opt.param_groups:group['lr']=peak*multiplier
                    opt.zero_grad(set_to_none=True)
                global_view,local_view=views(crops)
                if teacher_training:
                    head.teacher_images=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).contiguous().cuda().float()/255
                loss=sum(head(model.field(global_view,head,'detection',high_resolution=local_view),targets).values())
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite RGB adaptation loss')
                loss.backward()
                norms=native.clip_gradients('Rtau',optimizers)
                for opt in optimizers:opt.step()
                state['updates']=step;state['epoch']=epoch+1
                training_K.update(map(int,model.last_counts.tolist()))
                if step==1:
                    if teacher_training:
                        report=head.teacher_report()
                        report['neck_all_parameter_grads_None']=all(p.grad is None for p in head.core.model.parameters())
                        report['head_optimized_parameters']=sum(p.numel() for g in optimizers[1].param_groups for p in g['params'])
                        native.write(folder/'first_teacher_update.json',report)
                    native.write(folder/'first_counted_update.json',{'update':step,'input_rows':[rows[i]['row_id'] for i in ids],
                        'global_shape':list(global_view.shape),'local_shape':list(local_view.shape),'tile':tile,
                        'real_target_counts':[len(t['boxes']) for t in targets],'loss':float(loss.detach()),
                        'gradient_norms':norms,'sparse_RGB_witness':model.first_sparse_witness,
                        'local_not_upsampled224_maxabs':float((local_view-F.interpolate(global_view,(448,448),mode='bilinear',align_corners=False,antialias=True)).abs().max()),
                        'resources':guard(deadline)})
                if step==1 or step%100==0:
                    record={**state,'loss':float(loss.detach()),'positive_crops_in_batch':positive,
                            'gradient_norms':norms,'elapsed_seconds':time.monotonic()-begun,
                            'resources':guard(deadline)}
                    log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            if (epoch+1)%5==0:
                measured=evaluate(model,head,tile,'val',folder/f'val_epoch{epoch+1:02}',deadline,score_module)
                validation.append({'epoch':epoch+1,'result':measured})
                hota=measured['tracking']['HOTA']
                selection_score=hota if selection_metric=='HOTA' else measured['detection']['AP50']
                if selection_score>state['best_selection_score']:
                    state['best_selection_score']=selection_score
                    state['best_HOTA']=hota;state['best_epoch']=epoch+1;state['best_AP50']=measured['detection']['AP50']
                    save(folder,'best',model,head,state,optimizers)
            save(folder,'latest',model,head,state,optimizers)
        if state['updates']!=UPDATES:raise RuntimeError('Incomplete fixed update schedule')
        terminal=evaluate(model,head,tile,'test',folder/'test_terminal',deadline,score_module)
        if state['best_epoch']==EPOCHS:
            best=terminal
        else:
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True);head.load_state_dict(saved['head'],strict=True);del saved
            best=evaluate(model,head,tile,'test',folder/'test_best',deadline,score_module)
    if teacher_training:
        native.write(folder/'teacher_receipt.json',head.teacher_report())
    native.write(folder/'results.json',{'status':'COMPLETE','arm':arm,'tile':tile,'new_updates':state['updates'],
                 'best_validation_epoch':state['best_epoch'],'checkpoint_selection_metric':selection_metric,
                 'terminal_test':terminal,'best_test':best,
                 'validation':validation,'training_K_histogram':dict(training_K),'resources':guard(deadline),
                 'new_stage_wall_seconds':time.monotonic()-begun,'protocol':str(RUN/'protocol.json')})


def pipeline(worker_module):
    protocol=json.loads((RUN/'protocol.json').read_text())
    started=time.time();deadline=started+protocol['budgets']['global_seconds']
    admissions={}
    for arm,spec in protocol['candidates'].items():
        gpu=spec['GPU']
        info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True).strip()
        if int(info.split(',')[0])<24576:
            raise RuntimeError(f'Actual GPU{gpu} admission failed:{info}')
        admissions[arm]=info
    native.write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'started_unix':started,
                 'deadline_unix':deadline,'candidates':protocol['candidates'],'admissions':admissions,'new_training_updates':35000})
    active={};waits=[]
    for arm,spec in protocol['candidates'].items():
        gpu=spec['GPU']
        info=admissions[arm]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
             'OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg','YOLOv5_AUTOINSTALL':'False',
             'WANDB_MODE':'disabled','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        child_deadline=min(deadline,time.time()+14400)
        log=(RUN/f'{arm}_stdout.log').open('w')
        child=subprocess.Popen([sys.executable,'-u','-m',worker_module,'worker',
                '--arm',arm,'--deadline',str(child_deadline)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[arm]=(child,log,child_deadline,time.monotonic())
        native.write(RUN/f'{arm}_launch.json',{'PID':child.pid,'physical_GPU':gpu,'admission':info,'deadline_unix':child_deadline})
    while active:
        for arm,(child,log,end,begun) in list(active.items()):
            code=child.poll()
            if code is None and time.time()>end:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            if code is not None:
                log.close();record={'arm':arm,'PID':child.pid,'actual_wait_exit':code,'wall_seconds':time.monotonic()-begun}
                waits.append(record);native.write(RUN/f'{arm}_actual_wait.json',record);del active[arm]
                print(json.dumps(record),flush=True)
        native.write(RUN/'queue_state.json',{'active':{k:v[0].pid for k,v in active.items()},'waits':waits})
        if active:time.sleep(5)
    results={arm:json.loads((RUN/arm/'results.json').read_text()) for arm in protocol['candidates'] if (RUN/arm/'results.json').exists()}
    complete=len(results)==2 and all(r['actual_wait_exit']==0 for r in waits)
    native.write(RUN/'results.json',{'status':'COMPLETE' if complete else 'PARTIAL_OR_FAILED','candidates':results,
                 'author_reference':protocol['working_target_reference'],'all_directions_retained':True})
    native.write(RUN/'completion.json',{'status':'COMPLETE' if complete else 'PARTIAL_OR_FAILED','actual_waits':waits,
                 'wall_seconds':time.time()-started,'new_source_training':0,'full_dataset_downloads':0})
    if not complete:sys.exit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['pipeline','worker','score'])
    parser.add_argument('--arm',choices=['RGB512','RGB384'])
    parser.add_argument('--deadline',type=float)
    parser.add_argument('--folder',type=Path)
    args=parser.parse_args()
    if args.phase=='pipeline':pipeline('runs.cfc26_glisr_rgb_adaptation')
    elif args.phase=='worker':worker(args.arm,args.deadline,lambda parent:native.construct('Rtau',0,parent),INIT,.001,'HOTA',True,False,'runs.cfc26_glisr_rgb_adaptation')
    else:
        from runs.cfc26_author_weights_eval import score
        score(args.folder)
