"""Fixed frozen-GSR pilots on FSOD and UATD, using existing task readers."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from runs import selective_reencoding as base
from runs import downstream_vit_finetuning_readout as reader
from sonar_encoder.training.uatd_head import box_targets, dense_loss, decode_detections, detection_metrics

ROOT = base.ROOT
BASE = base.BASE
RUN = BASE/'gsr_detection_transfer_v1'


def protocol():
    return json.loads((RUN/'protocol.json').read_text())


def observe():
    x={'RSS_HWM_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(0)}
    if x['RSS_HWM_bytes']>32*1024**3 or x['GPU_reserved_peak_bytes']>32*1024**3:
        raise RuntimeError('Detection pilot resource cap: '+str(x))
    if time.time()>protocol()['deadline_unix']:raise TimeoutError('Original six-hour deadline')
    return x


class Detector(nn.Module):
    def __init__(self,task):
        super().__init__();self.task=task
        self.head=reader.new_head(task,0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101);self.fusion=base.Fusion()

    def field(self,g,tiles,action):
        b=F.interpolate(g,(32,32),mode='bilinear',align_corners=False,antialias=False)
        crops=torch.stack([b[...,r:r+16,c:c+16] for r,c in base.QUADS],1)
        ids=torch.arange(len(g),device=g.device)
        delta=self.fusion(tiles[ids,action.clamp_min(0)],crops[ids,action.clamp_min(0)],g,action)
        residual=torch.zeros_like(b)
        for j,(r,c) in enumerate(base.QUADS):residual=residual+F.pad(delta*(action==j)[:,None,None,None],(c,16-c,r,16-r))
        # Preserve the exact original UATD16 task-head baseline when no region is added.
        return g+F.avg_pool2d(residual,2) if self.task=='UATD' else b+residual


def load_inputs(task):
    p=protocol()['tasks'][task];data={}
    for split,rows in p['rows'].items():
        g=torch.from_numpy(np.load(p['global_caches'][split],allow_pickle=False)).cuda()
        src=np.load(p['tile_caches'][split],mmap_mode='r')
        tiles=torch.empty((len(rows),4,384,16,16),device='cuda')
        for start in range(0,len(rows),64):
            field=torch.from_numpy(np.array(src[start:start+64],copy=True))
            batch=torch.stack([field[...,r:r+16,c:c+16] for r,c in base.QUADS],1)
            tiles[start:start+len(field)]=batch.cuda()
        targets=reader.make_targets(task,rows)
        if task=='UATD':
            dense=box_targets(targets,'cuda')
            if not bool(dense[2].flatten(1).sum(1).gt(0).all()):
                raise RuntimeError('Original UATD per-image positive-target assumption violated')
        else:dense=None
        data[split]={'g':g,'tiles':tiles,'targets':targets,'dense':dense}
    return p,data


def batch_targets(task,data,ids):
    if task=='FSOD':return reader.batch_targets(task,data['targets'],ids,'cuda')
    return tuple(x[ids] for x in data['dense'])


def save(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


@torch.no_grad()
def extract_uatd():
    from sonar_encoder.training.model import build_encoder
    from runs.marine_native_information import tile_frames,stitch_tokens
    p=protocol();spec=p['tasks']['UATD']
    encoder=build_encoder(Path(p['encoder']['code_dir']),Path(p['encoder']['weights_path']),'R010').cuda().eval().requires_grad_(False)
    samples=calls=0
    for split in ('train','validation'):
        src=np.load(spec['input_views'][split],mmap_mode='r');n=len(src)
        out=np.lib.format.open_memmap(spec['tile_caches'][split],mode='w+',dtype=np.float32,shape=(n,384,32,32))
        for start in range(0,n,16):
            common=torch.from_numpy(np.array(src[start:start+16],copy=True))
            large=F.interpolate(common,(448,448),mode='bilinear',align_corners=False,antialias=True)
            images=tile_frames(large).cuda()
            field=stitch_tokens(encoder(images)['patch_tokens'])
            out[start:start+len(common)]=field.cpu().numpy();samples+=len(images);calls+=1
            if calls%50==0:print(json.dumps({'extract':'UATD','samples':samples,'calls':calls}),flush=True);observe()
        out.flush()
    base.write(RUN/'UATD/extraction.json',{'status':'COMPLETE_COUNTED_FOUR_CROPS','images':7556,'encoder_views':samples,'calls':calls,
        'global_encoder_forwards':0,'image_decodes':0,'encoder_frozen':True,'cache_source':'Original gain1 common224 input views enlarged onCPU, no new native pixels','resources':observe()})


@torch.no_grad()
def evaluate(task,model,data,rows,actions):
    model.eval();pieces=[];batch=8 if task=='FSOD' else 32
    for start in range(0,len(rows),batch):
        sl=slice(start,start+batch);a=torch.tensor(actions[sl],device='cuda')
        field=model.field(data['g'][sl],data['tiles'][sl],a)
        pieces.append(reader.decode(task,model.head,field,rows[sl]))
    return np.concatenate(pieces) if task=='FSOD' else [v for piece in pieces for v in piece]


def quick_metric(task,p,predictions,targets):
    if task=='FSOD':
        from runs.shared_supervised_json_readout import canonical_evaluate
        from runs.shared_supervised_readout import prediction_list
        return canonical_evaluate(json.loads(Path(p['GT_paths']['validation']).read_text()),prediction_list(predictions))
    return detection_metrics(predictions,targets)


def fit(task,p,data):
    folder=RUN/task;model=Detector(task).cuda()
    head_opt=reader.head_optimizer(task,model.head)
    fusion_opt=torch.optim.AdamW(model.fusion.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(0)
    plan=np.load(folder/'training_actions.npy');random_actions=np.load(folder/'validation_random_actions.npy')
    best=-1.;best_epoch=updates=0;metric='AP50' if task=='FSOD' else 'map50';begun=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1,p['epochs']+1):
            model.train();order=torch.randperm(len(p['rows']['train']),generator=generator);total=0.
            if task=='FSOD':
                lr=.005 if epoch<=40 else (.0005 if epoch<=55 else .00005)
                for group in head_opt.param_groups:group['lr']=lr
            for ids in order.split(p['batch']):
                head_opt.zero_grad(set_to_none=True);fusion_opt.zero_grad(set_to_none=True)
                actions=torch.tensor(plan[epoch-1,ids],device='cuda')
                field=model.field(data['train']['g'][ids],data['train']['tiles'][ids],actions)
                loss=reader.task_loss(task,model.head,field,batch_targets(task,data['train'],ids))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite original detection task loss')
                loss.backward();head_opt.step();fusion_opt.step();updates+=1;total+=float(loss.detach())*len(ids)
                if updates%100==0:observe()
            entry={'epoch':epoch,'updates':updates,'train_loss':total/len(order),'elapsed_seconds':time.monotonic()-begun}
            if epoch%5==0:
                predictions=evaluate(task,model,data['validation'],p['rows']['validation'],random_actions)
                val=quick_metric(task,p,predictions,data['validation']['targets']);entry['validation_random_routing']=val
                if val[metric]>best:
                    best,best_epoch=val[metric],epoch
                    save(folder/'best_random_routing.pt',{'model':model.state_dict(),'epoch':epoch,'task':task})
                print(json.dumps({'fit':task,'epoch':epoch,'updates':updates,'random_metric':val[metric],'resources':observe()}),flush=True)
            if epoch%10==0 or epoch==1 or epoch==p['epochs']:
                save(folder/'latest.pt',{'model':model.state_dict(),'epoch':epoch,'updates':updates,'task':task,
                    'head_optimizer':head_opt.state_dict(),'fusion_optimizer':fusion_opt.state_dict(),
                    'private_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
            log.write(json.dumps(entry)+'\n');log.flush()
    base.write(folder/'fit_report.json',{'status':'COMPLETE_FIXED_FULL_SCHEDULE','epochs':p['epochs'],'updates':updates,
        'task':task,'head_parameters':sum(x.numel() for x in model.head.parameters()),'fusion_parameters':sum(x.numel() for x in model.fusion.parameters()),
        'random_routing_terminal':val,'best_random_metric':best,'best_random_epoch':best_epoch,'wall_seconds':time.monotonic()-begun,
        'best_random_checkpoint_not_complete_method_best':True,'encoder_loaded_for_fit':False,'resources':observe()})
    return model.eval().requires_grad_(False)


def uatd_components(output,targets):
    heat,reg,pos=targets;logits=output[:,:10];prob=logits.sigmoid()
    ptarget=torch.where(pos,prob,1-prob)
    focal=(torch.where(pos,.25,.75)*(1-ptarget).square()*F.binary_cross_entropy_with_logits(logits,heat,reduction='none')).flatten(1).sum(1)
    raw=output[:,10:].reshape(-1,10,4,16,16)
    pred=torch.cat((raw[:,:,:2].sigmoid(),raw[:,:,2:]),2)
    boxes=(F.smooth_l1_loss(pred,reg,reduction='none')*pos[:,:,None]).flatten(1).sum(1)
    count=pos.flatten(1).sum(1)
    return torch.stack((focal/count,boxes/(4*count)),1)


@torch.no_grad()
def utility_labels(task,p,model,data):
    n=len(p['rows']['train']);k=4 if task=='FSOD' else 2
    components=np.empty((n,5,k),np.float32)
    if task=='FSOD':
        model.train()
        names=['loss_objectness','loss_rpn_box_reg','loss_classifier','loss_box_reg']
        for i in range(n):
            # Same RNG start reduces sampling noise; proposals/samples can still differ.
            for action in range(-1,4):
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(10000+i)
                    a=torch.tensor([action],device='cuda');ids=torch.tensor([i])
                    field=model.field(data['g'][i:i+1],data['tiles'][i:i+1],a)
                    losses=model.head(field,batch_targets(task,data,ids))
                    components[i,action+1]=torch.stack([losses[name] for name in names]).cpu().numpy()
            if (i+1)%100==0:print(json.dumps({'gains':task,'training_images':i+1,'total':n}),flush=True);observe()
    else:
        model.eval();names=['focal','box_regression']
        for start in range(0,n,32):
            sl=slice(start,start+32);targets=tuple(x[sl] for x in data['dense'])
            for action in range(-1,4):
                a=torch.full((len(data['g'][sl]),),action,device='cuda',dtype=torch.long)
                field=model.field(data['g'][sl],data['tiles'][sl],a)
                components[sl,action+1]=uatd_components(model.head(field),targets).cpu().numpy()
            observe()
    model.eval()
    split=np.load(RUN/task/'selector_split.npz');fit_ids=split['fit']
    scale=np.maximum(components[fit_ids,0].mean(0),1e-6)
    gains={'ordinary':components[:,0:1].sum(2)-components[:,1:].sum(2),
           'balanced':((components[:,0:1]-components[:,1:])/scale).mean(2)}
    np.savez(RUN/task/'utility_labels.npz',components=components,ordinary_gains=gains['ordinary'],balanced_gains=gains['balanced'],component_scales=scale)
    base.write(RUN/task/'utility_definition.json',{'component_names':names,'baseline_mean_scales_fit_only':scale.tolist(),
        'ordinary':'Sum of original per-image task-loss components before minus after local correction',
        'balanced':'Mean of component-wise signed loss decreases divided by fit-only no-region mean component loss; floor1e-6',
        'no_mask_boundary_label_in_detection':True,'FSOD_counterfactual_sampling':'Same per-image RNG seed10000+i for five actions, in RNG forks; feature-dependent proposals/samples need not match',
        'encoder_updates':0,'head_updates_during_labels':0,'source_images':n,'counterfactual_feature_head_examples':5*n})
    return gains


def train_selectors(task,p,data,gains):
    folder=RUN/task;split=np.load(folder/'selector_split.npz');fit_ids,check_ids=split['fit'],split['check'];g=data['g']
    for criterion in ('ordinary','balanced'):
        y=gains[criterion];mean=float(y[fit_ids].mean());scale=max(float(y[fit_ids].std()),1e-8)
        target=torch.from_numpy((y-mean)/scale).cuda();torch.manual_seed(7);selector=base.Selector().cuda()
        optimizer=torch.optim.AdamW(selector.parameters(),lr=.001,weight_decay=.0001);generator=torch.Generator().manual_seed(7)
        updates=0
        for epoch in range(1,101):
            order=torch.tensor(fit_ids)[torch.randperm(len(fit_ids),generator=generator)];selector.train()
            for ids in order.split(64):
                optimizer.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(selector(g[ids]),target[ids])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite selector loss')
                loss.backward();optimizer.step();updates+=1
            if epoch%10==0:observe()
        selector.eval()
        with torch.no_grad():scores=torch.cat([selector(g[i:i+64]) for i in range(0,len(g),64)]).cpu().numpy()*scale+mean
        j=scores[check_ids].argmax(1);truth=y[check_ids]
        save(folder/f'selector_{criterion}.pt',{'selector':selector.state_dict(),'criterion':criterion,'task':task,'epochs':100,'updates':updates,'gain_mean':mean,'gain_scale':scale})
        base.write(folder/f'selector_{criterion}_report.json',{'criterion':criterion,'epochs':100,'updates':updates,
            'fit_count':len(fit_ids),'internal_check_count':len(check_ids),'fixed_region':int(y[fit_ids].mean(0).argmax()),
            'gain_min':float(y.min()),'gain_max':float(y.max()),'negative_fraction':float((y<0).mean()),
            'internal_check_argmax':float((j==truth.argmax(1)).mean()),
            'internal_check_regret':float((truth.max(1)-truth[np.arange(len(check_ids)),j]).mean()),
            'check_is_not_heldout_from_fusion_training':True,'resources':observe()})
        print(json.dumps({'selector':task+'/'+criterion,'updates':updates,'check_agreement':float((j==truth.argmax(1)).mean())}),flush=True)


@torch.no_grad()
def uncertainty(task,model,g):
    if task=='UATD':logits=model.head(g);size=8
    else:
        field=F.interpolate(g,(32,32),mode='bilinear',align_corners=False,antialias=False)
        logits=model.head.rpn.head([field])[0][0];size=16
    if task=='UATD':logits=logits[:,:10]
    entropy=(logits.sigmoid()*F.logsigmoid(logits)+(1-logits.sigmoid())*F.logsigmoid(-logits)).neg().mean(1)
    return torch.stack([entropy[:,r:r+size,c:c+size].mean((1,2)) for r,c in ((0,0),(0,size),(size,0),(size,size))],1).argmax(1).cpu().numpy()


def finish_task(task,p,model,data):
    folder=RUN/task;g=data['validation']['g'];n=len(g)
    policies={'base':np.full(n,-1,np.int64),'random':np.load(folder/'validation_random_actions.npy'),
              'uncertainty':uncertainty(task,model,g)}
    for criterion in ('ordinary','balanced'):
        selector=base.Selector().cuda();selector.load_state_dict(torch.load(folder/f'selector_{criterion}.pt',map_location='cpu',weights_only=True)['selector']);selector.eval()
        with torch.no_grad():scores=torch.cat([selector(g[i:i+64]) for i in range(0,n,64)]).cpu().numpy()
        report=json.loads((folder/f'selector_{criterion}_report.json').read_text())
        policies['learned_'+criterion]=scores.argmax(1)
        policies['fixed_'+criterion]=np.full(n,report['fixed_region'],np.int64)
    np.savez(folder/'validation_actions.npz',**policies)
    result={};checks=[];metric='AP50' if task=='FSOD' else 'map50'
    for name,actions in policies.items():
        pred=evaluate(task,model,data['validation'],p['rows']['validation'],actions)
        if task=='FSOD':path=folder/(name+'_prediction.npy');np.save(path,pred);saved=np.load(path);coco=json.loads(Path(p['GT_paths']['validation']).read_text())
        else:path=folder/(name+'_prediction.pt');save(path,pred);saved=torch.load(path,map_location='cpu',weights_only=True);coco=None
        canonical,independent=reader.original_scores(task,saved,data['validation']['targets'],coco)
        result[name]=canonical;checks.append({'policy':name,'prediction':str(path),'original_GT_independent_score_pass':True,'value':independent[metric]})
        print(json.dumps({'task':task,'policy':name,'metric':metric,'value':canonical[metric]}),flush=True)
    contrasts={}
    for name in ('ordinary','balanced'):
        for control in ('base','random','uncertainty','fixed_'+name):contrasts['learned_'+name+'_minus_'+control]=result['learned_'+name][metric]-result[control][metric]
    contrasts['balanced_minus_ordinary']=result['learned_balanced'][metric]-result['learned_ordinary'][metric]
    base.write(folder/'results.json',{'task':task,'states':result,'contrasts':contrasts,'primary':'Fixed full-schedule terminal model and100epoch selectors; single seed0',
        'GT_metric_scope':'Original FSOD COCO101-point AP50 / UATD continuous-PR AP50; not interchangeable',
        'resources':observe()})
    base.write(folder/'verification.json',{'status':'PASS_SEVEN_ORIGINAL_GT_SAVED_POLICY_SCORES','checks':checks})


def workflow(task):
    if task=='UATD':extract_uatd()
    p,data=load_inputs(task);model=fit(task,p,data)
    gains=utility_labels(task,p,model,data['train']);train_selectors(task,p,data['train'],gains)
    finish_task(task,p,model,data)


def cost():
    from sonar_encoder.training.model import build_encoder,prepare_images
    p=protocol();encoder=build_encoder(Path(p['encoder']['code_dir']),Path(p['encoder']['weights_path']),'R010').cuda().eval().requires_grad_(False)
    out=[]
    for task in ('FSOD','UATD'):
        spec=p['tasks'][task];model=Detector(task).cuda();model.load_state_dict(torch.load(RUN/task/'latest.pt',map_location='cpu',weights_only=True)['model']);model.eval().requires_grad_(False)
        selectors={}
        for name in ('ordinary','balanced'):
            s=base.Selector().cuda();s.load_state_dict(torch.load(RUN/task/f'selector_{name}.pt',map_location='cpu',weights_only=True)['selector']);s.eval().requires_grad_(False);selectors[name]=s
        rows=spec['rows']['validation'][:32];natives=[]
        for row in rows:
            with Image.open(row['image']) as image:a=torch.from_numpy(np.array(image.convert('L'),dtype=np.uint8)).float()[None]/255
            natives.append({'intensity':a,'valid':torch.ones(a.shape[-2:],dtype=torch.bool)})
        for policy in ('ordinary','balanced','uncertainty','base'):
            records=[];load=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
            baseline=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats()
            for visit in range(146):
                i=visit%32 if visit<50 else (visit-50)%32
                torch.cuda.synchronize();begin=time.perf_counter()
                common=prepare_images([natives[i]],spec['normalization'],None)
                with torch.no_grad():
                    g=encoder(common.cuda())['patch_tokens'].transpose(1,2).reshape(1,384,16,16)
                    if policy=='base':action=-1
                    elif policy=='uncertainty':action=int(uncertainty(task,model,g)[0])
                    else:action=int(selectors[policy](g).argmax(1))
                    tiles=torch.zeros((1,4,384,16,16),device='cuda')
                    if action>=0:
                        large=F.interpolate(common,(448,448),mode='bilinear',align_corners=False,antialias=True);r,c=base.QUADS[action]
                        tiles[:,action]=encoder(large[...,r*14:r*14+224,c*14:c*14+224].cuda())['patch_tokens'].transpose(1,2).reshape(1,384,16,16)
                    field=model.field(g,tiles,torch.tensor([action],device='cuda'))
                    pred=reader.decode(task,model.head,field,rows[i:i+1])
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                if visit>=50:records.append({'row_id':spec['row_ids']['validation'][i],'round':(visit-50)//32,'seconds':seconds})
            values=[x['seconds'] for x in records]
            out.append({'task':task,'policy':policy,'median_seconds':float(np.median(values)),'p95_seconds':float(np.percentile(values,95)),
                'raw':records,'baseline_reserved':baseline,'peak_reserved':torch.cuda.max_memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'load':load})
            print(json.dumps({'cost':task+'/'+policy,'median_ms':float(np.median(values))*1000}),flush=True);observe()
    base.write(RUN/'cost_results.json',{'status':'COMPLETE_EIGHT_REAL_ONLINE_CELLS','cells':out,
        'boundary':'CPU native image RAM through preproc/H2D/frozenencoder(s)/selector/fusion/head/postproc/native predictions; decode/setup excluded',
        'memory_scope':'Shared resident models and allocator history, not independent deployment processes','feature_caches_read_in_timed_calls':0,'own_training_concurrent':False})


def pipeline():
    RUN.mkdir();start=time.time();prep=json.loads((BASE/'downstream_vit_finetuning_v1/cpu_preparation.json').read_text())
    oldprotocol=json.loads((BASE/'selective_reencoding_marine_v1/protocol.json').read_text())
    ext=json.loads((BASE/'shared_supervised_reference_json_resume_v1/readouts/T0/fsod/extraction.json').read_text())
    tiled=json.loads((BASE/'interp_tiles_fsod_transfer_v1/extraction.json').read_text())
    p={'unit':RUN.name,'start_unix':start,'deadline_unix':start+21600,'encoder':oldprotocol['encoder'],
        'authorization':'User requested FSOD and UATD detection extensions after Marine-only fast pilots; GPU2/3, no smoke/SHA/review',
        'seed':0,'encoder_frozen':True,'source_utf8':Path(__file__).read_text(),'tasks':{},
        'fusion':'Same64d cross-attention/gate/sine-edge residual as Marine; 99265 trainable fusion parameters',
        'gain':'Ordinary original per-image loss decrease versus component-normalized decrease; trainfit80% mean no-region components, floor1e-6',
        'no_segmentation_masks_or_boundary_labels_in_detection':True,
        'source_scope':'FSOD D-bound local snapshotC / UATD original gain1 train-development; no Test, new split or native geometry',
        'selector':'Same49409-param mean-feature selector; seed7, AdamW.001 wd.0001 B64,100epochs,80/20 innertrain split37',
        'same_inference_budget':'One global224 plus one local224, four fixed candidates; same224-derived observation',
        'caps':{'global_wall_seconds':21600,'FSOD_workflow':14400,'UATD_workflow':7200,'cost':1200,
                'RSS':32*1024**3,'GPU_reserved':32*1024**3,'new_disk':16*1024**3},
        'stop':'Real error/resource/nonfinite stops owned jobs; no automatic fallback/retry/HPO/extra seed',
        'preparation_nonmodel_error':'One metadata query expected obsolete data key; actual CPU preparation uses inputs. No model was involved.'}
    for task,n,epochs,batch in [('FSOD',2357,60,8),('UATD',6051,100,32)]:
        folder=RUN/task;folder.mkdir();source=prep['inputs'][task]
        globals_={s:ext['caches'][s]['path'] for s in ('train','validation')} if task=='FSOD' else {
            s:str(BASE/f'ssl_global_keep_v1/t0/t0_{s}.npy') for s in ('train','validation')}
        locals_={s:tiled['caches'][s]['path'] for s in ('train','validation')} if task=='FSOD' else {
            s:str(folder/f'tiles_{s}.npy') for s in ('train','validation')}
        spec={'rows':source['rows'],'row_ids':prep['row_ids'][task],'normalization':source['normalization'],
              'global_caches':globals_,'tile_caches':locals_,'epochs':epochs,'batch':batch,
              'input_views':{s:str(BASE/f'downstream_vit_finetuning_v1/inputs/{task}/{s}.npy') for s in ('train','validation')},
              'head_and_task_math':'Original11class RPN/ROI canvas224, field32 stride7 ROI1/7' if task=='FSOD' else 'OriginalConv384->50 dense16 focal/SmoothL1; field=G16+averagepool(scattered residual32)',
              'fusion_optimizer':'AdamW.001 wd.0001, fixed; encoder excluded',
              'head_optimizer':'OriginalSGD.005/.9/.0005, drops40/55' if task=='FSOD' else 'OriginalAdamW.001/.0001',
              'scheduled_validation':'Every5epochs random routing; main full-method terminal only, best random auxiliary'}
        if task=='FSOD':spec['GT_paths']=source['GT_paths']
        for s in ('train','validation'):
            a=np.load(globals_[s],mmap_mode='r')
            if a.shape!=(len(spec['rows'][s]),384,16,16) or a.dtype!=np.float32:raise RuntimeError('Global feature header incompatible')
            if task=='FSOD':
                a=np.load(locals_[s],mmap_mode='r')
                if a.shape!=(len(spec['rows'][s]),384,32,32) or ext['caches'][s]['row_ids']!=spec['row_ids'][s] or tiled['caches'][s]['row_ids']!=spec['row_ids'][s]:raise RuntimeError('FSOD local/global population mismatch')
        np.save(folder/'training_actions.npy',np.random.default_rng(1).integers(-1,4,(epochs,n),dtype=np.int64))
        np.save(folder/'validation_random_actions.npy',np.random.default_rng(2).integers(0,4,len(spec['rows']['validation']),dtype=np.int64))
        ids=np.random.default_rng(37).permutation(n);cut=int(.8*n)
        np.savez(folder/'selector_split.npz',fit=ids[:cut],check=ids[cut:])
        p['tasks'][task]=spec
    base.write(RUN/'protocol.json',p);(RUN/'logs').mkdir();children={};records=[]
    def launch(phase,gpu,cap):
        entry=base.admission(gpu);env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
            'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','XFORMERS_DISABLED':'1'}
        log=(RUN/'logs'/f'{phase}.log').open('w')
        child=subprocess.Popen([base.PYTHON,'-B','-u','-m','runs.gsr_detection_transfer',phase],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[phase]=(child,log,time.time(),min(cap,p['deadline_unix']-time.time()),entry)
        base.write(RUN/'queue_state.json',{'status':'RUNNING','active':{k:v[0].pid for k,v in children.items()},'completed':records})
    def finish(phase):
        child,log,begun,cap,entry=children[phase]
        try:code=child.wait(timeout=max(0.,cap-(time.time()-begun)))
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        log.close();del children[phase];row={'phase':phase,'PID':child.pid,'physical_GPU':entry['physical_GPU'],'actual_wait_exit':code,'wall_seconds':time.time()-begun,'admission':entry}
        records.append(row)
        with (RUN/'compute_ledger.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if code!=0:raise RuntimeError(phase+' actual exit '+str(code))
    try:
        launch('FSOD',2,14400);launch('UATD',3,7200)
        base.write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':{k:v[0].pid for k,v in children.items()},'protocol':str(RUN/'protocol.json'),'started_unix':time.time()})
        while children:
            for phase in list(children):
                child,log,begun,cap,entry=children[phase]
                if child.poll() is not None or time.time()-begun>=cap:finish(phase)
            time.sleep(1)
        launch('cost',2,1200);finish('cost')
        results={task:json.loads((RUN/task/'results.json').read_text()) for task in ('FSOD','UATD')}
        base.write(RUN/'results.json',{'status':'COMPLETE_BOTH_FIXED_DETECTION_TASKS','tasks':results,'Marine_reference_inherited':str(BASE/'gsr_sonar_adaptation_v1/results.json')})
        disk=sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file() and not x.is_symlink())
        if disk>16*1024**3:raise RuntimeError('New disk cap exceeded')
        close={'status':'COMPLETE_GSR_FSOD_UATD_DETECTION_PILOTS','supervisor_PID':os.getpid(),'actual_waits':records,
            'pipeline_wall_seconds':time.time()-start,'summed_child_wall_seconds':sum(x['wall_seconds'] for x in records),
            'new_entity_bytes':disk,'results':str(RUN/'results.json'),'cost':str(RUN/'cost_results.json'),'no_next_queue':True,
            'smoke_tests_SHA_reviews':0,'encoder_training_updates':0,'new_encoder_sample_forwards_for_UATD_cache':30224}
        base.write(RUN/'final_closure.json',close);base.write(RUN/'writing_handoff.json',close)
        base.write(RUN/'queue_state.json',{'status':'COMPLETE','actual_waits':records})
    except BaseException:
        for phase,(child,log,begun,cap,entry) in children.items():
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            code=child.wait();log.close();records.append({'phase':phase,'PID':child.pid,'actual_wait_exit':code,'cancelled_after_failure':True})
        base.write(RUN/'failure_closure.json',{'status':'STOPPED_REAL_ERROR_NO_RETRY','traceback':traceback.format_exc(),'actual_waits':records})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pipeline','FSOD','UATD','cost']);args=parser.parse_args();base.runtime()
    if args.phase=='pipeline':pipeline()
    elif args.phase=='cost':cost()
    else:workflow(args.phase)
