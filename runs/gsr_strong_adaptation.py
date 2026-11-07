"""Performance-oriented GSR continuation; all new parameters stay in RAM."""
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

from runs import selective_reencoding as old
from runs import gsr_detection_transfer as detection
from runs import gsr_sonar_adaptation as marine
from runs import downstream_vit_finetuning_readout as reader
from sonar_encoder.training.model import build_encoder,prepare_images
from sonar_encoder.training.readout import load_targets
from sonar_encoder.training.uatd_head import box_targets,detection_metrics

ROOT=old.ROOT
BASE=old.BASE
RUN=BASE/'gsr_strong_adaptation_v1'
MODES=('continue','frozen','partial','full')


def protocol():
    return json.loads((RUN/'protocol.json').read_text())


def resources():
    p=protocol();v={'PID':os.getpid(),'RSS_HWM_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(0)}
    if v['RSS_HWM_bytes']>48*1024**3 or v['GPU_reserved_peak_bytes']>48*1024**3:raise RuntimeError('Performance-wave resource cap '+str(v))
    if time.time()>p['deadline_unix']:raise TimeoutError('Original8h performance-wave deadline')
    return v


class SpatialSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(386,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1,groups=64),nn.GELU(),nn.Conv2d(64,1,1))
        y,x=torch.meshgrid((torch.arange(16)+.5)/8-1,(torch.arange(16)+.5)/8-1,indexing='ij')
        self.register_buffer('xy',torch.stack((x,y))[None])

    def forward(self,g):
        scores=self.net(torch.cat((g,self.xy.expand(len(g),-1,-1,-1)),1))
        return F.adaptive_avg_pool2d(scores,(2,2)).flatten(1)


class Model(nn.Module):
    def __init__(self,task,initial):
        super().__init__();self.task=task
        cfg=protocol()['encoder']
        self.encoder=build_encoder(Path(cfg['code_dir']),Path(cfg['weights_path']),'R010')
        self.encoder.load_state_dict(initial['encoder'],strict=True)
        self.head=reader.new_head(task,0);self.head.load_state_dict(initial['head'],strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101);self.fusion=old.Fusion()

    def global_field(self,images):
        grid=16 if self.task=='UATD' else 32
        return self.encoder(images)['patch_tokens'].transpose(1,2).reshape(-1,384,grid,grid)

    def context16(self,g):
        return g if self.task=='UATD' else F.avg_pool2d(g,2)

    def local_images(self,images,actions=None):
        large=F.interpolate(images,(448,448),mode='bilinear',align_corners=False,antialias=True) if self.task=='UATD' else images
        tiles=torch.stack([large[...,r*14:r*14+224,c*14:c*14+224] for r,c in old.QUADS],1)
        return tiles.flatten(0,1) if actions is None else tiles[torch.arange(len(images),device=images.device),actions]

    def local_fields(self,images,actions=None):
        value=self.encoder(self.local_images(images,actions))['patch_tokens'].transpose(1,2).reshape(-1,384,16,16)
        return value.reshape(len(images),4,384,16,16) if actions is None else value

    def corrected(self,g,local,actions):
        if not bool((actions>=0).any()):return g
        base=F.interpolate(g,(32,32),mode='bilinear',align_corners=False,antialias=False) if self.task=='UATD' else g
        crops=torch.stack([base[...,r:r+16,c:c+16] for r,c in old.QUADS],1)
        ids=torch.arange(len(g),device=g.device)
        if local.ndim==5:local=local[ids,actions.clamp_min(0)]
        delta=self.fusion(local,crops[ids,actions.clamp_min(0)],self.context16(g),actions)
        residual=torch.zeros_like(base)
        for j,(r,c) in enumerate(old.QUADS):residual=residual+F.pad(delta*(actions==j)[:,None,None,None],(c,16-c,r,16-r))
        return g+F.avg_pool2d(residual,2) if self.task=='UATD' else g+residual


def data_for(task):
    spec=protocol()['tasks'][task];images={};targets={};dense={}
    for split,rows in spec['rows'].items():
        images[split]=torch.from_numpy(np.load(spec['input_views'][split],allow_pickle=False)).cuda()
        targets[split]=reader.make_targets(task,rows)
        if task=='UATD':dense[split]=box_targets(targets[split],'cuda')
        elif task=='Marine':dense[split]=targets[split].cuda()
    return spec,images,targets,dense


def targets_batch(task,targets,dense,ids):
    if task=='FSOD':return reader.batch_targets(task,targets,ids,'cuda')
    if task=='UATD':return tuple(x[ids] for x in dense)
    return dense[ids]


def configure(model,mode):
    model.train()
    model.encoder.requires_grad_(mode in ('partial','full','continue'))
    if mode=='partial':
        for name,param in model.encoder.named_parameters():
            param.requires_grad_(name.startswith(tuple(f'backbone.blocks.{i}.' for i in range(8,12))) or name.startswith('backbone.norm.'))
    for name,param in model.encoder.named_parameters():
        if name.endswith('mask_token'):param.requires_grad_(False)
    if mode=='frozen':model.encoder.eval();model.head.requires_grad_(False)
    model.fusion.requires_grad_(mode!='continue')
    body=reader.body_optimizer(model.encoder) if mode!='frozen' else None
    if body:
        for group in body.param_groups:group['lr']=2e-6
    head=reader.head_optimizer(model.task,model.head) if mode!='frozen' else None
    if head:
        for group in head.param_groups:group['lr']=1e-4
    fusion=torch.optim.AdamW(model.fusion.parameters(),lr=.001,weight_decay=.0001) if mode!='continue' else None
    return body,head,fusion


def train(model,mode,spec,images,targets,dense):
    plan=np.load(RUN/model.task/'plan.npy');regions=np.load(RUN/model.task/'regions.npy')
    optimizers=configure(model,mode);begun=time.monotonic();folder=RUN/model.task/mode;folder.mkdir()
    counts={'body_trainable':sum(x.numel() for x in model.encoder.parameters() if x.requires_grad),
            'head_trainable':sum(x.numel() for x in model.head.parameters() if x.requires_grad),
            'fusion_trainable':sum(x.numel() for x in model.fusion.parameters() if x.requires_grad)}
    with (folder/'training.jsonl').open('w') as log:
        for step in range(2000):
            ids=torch.from_numpy(plan[step]);actions=torch.from_numpy(regions[step]).cuda()
            for opt in optimizers:
                if opt:opt.zero_grad(set_to_none=True)
            x=images[ids];target=targets_batch(model.task,targets,dense,ids)
            if mode=='frozen':
                with torch.no_grad():g=model.global_field(x);local=model.local_fields(x,actions)
            else:
                g=model.global_field(x);local=model.local_fields(x,actions) if mode!='continue' else None
            global_loss=reader.task_loss(model.task,model.head,g,target)
            if mode=='continue':loss=global_loss
            else:
                corrected=model.corrected(g,local,actions)
                loss=.5*global_loss+.5*reader.task_loss(model.task,model.head,corrected,target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real training loss')
            loss.backward()
            if optimizers[0]:torch.nn.utils.clip_grad_norm_((p for p in model.encoder.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
            if optimizers[1]:torch.nn.utils.clip_grad_norm_(model.head.parameters(),float('inf'),error_if_nonfinite=True)
            if optimizers[2]:torch.nn.utils.clip_grad_norm_(model.fusion.parameters(),1.,error_if_nonfinite=True)
            for opt in optimizers:
                if opt:opt.step()
            if (step+1)%100==0:
                entry={'step':step+1,'loss':float(loss.detach()),'global_loss':float(global_loss.detach()),'elapsed_seconds':time.monotonic()-begun,'resources':resources()}
                log.write(json.dumps(entry)+'\n');log.flush()
                print(json.dumps({'task':model.task,'mode':mode,**entry}),flush=True)
    model.eval().requires_grad_(False)
    for p in model.parameters():p.grad=None
    report={'task':model.task,'mode':mode,'updates':2000,'train_visits':2000*spec['batch'],
        'local_encoder_visits':0 if mode=='continue' else 2000*spec['batch'],
        'global_and_local_task_losses_share_GT':True,'trainable_parameter_counts':counts,
        'wall_seconds':time.monotonic()-begun,'new_weight_files':0,'resources':resources()}
    old.write(folder/'train_report.json',report)


@torch.no_grad()
def feature_bank(model,images,include_local):
    bank={}
    grid=16 if model.task=='UATD' else 32
    for split,x in images.items():
        g=torch.empty((len(x),384,grid,grid),device='cuda')
        l=torch.empty((len(x),4,384,16,16),device='cuda') if include_local else None
        for start in range(0,len(x),8):
            sl=slice(start,start+8);g[sl]=model.global_field(x[sl])
            if include_local:l[sl]=model.local_fields(x[sl])
            if start%256==0:resources()
        bank[split]=(g,l)
    return bank


@torch.no_grad()
def gain_labels(model,spec,bank,targets,dense):
    g,l=bank;n=len(g);k=4 if model.task=='FSOD' else 2
    split=np.load(RUN/model.task/'selector_split.npz');fit_ids=split['fit']
    if model.task=='Marine':
        count=torch.bincount(dense[fit_ids].long().flatten(),minlength=12).float()
        weights=count.clamp_min(1).rsqrt();weights=weights/((weights*count).sum()/count.sum())
        losses=np.empty((n,5),np.float32)
        for start in range(0,n,16):
            sl=slice(start,start+16);w=marine.boundary_weights(dense[sl],weights)
            for j in range(-1,4):
                a=torch.full((len(g[sl]),),j,device='cuda',dtype=torch.long)
                field=g[sl] if j<0 else model.corrected(g[sl],l[sl],a)
                ce=F.cross_entropy(old.native_logits(model.head(field)),dense[sl].long(),reduction='none')
                losses[sl,j+1]=((ce*w).sum((1,2))/w.sum((1,2))).cpu().numpy()
            resources()
        return losses[:,:1]-losses[:,1:]
    comp=np.empty((n,5,k),np.float32)
    if model.task=='UATD':
        for start in range(0,n,32):
            sl=slice(start,start+32);t=tuple(x[sl] for x in dense)
            for j in range(-1,4):
                a=torch.full((len(g[sl]),),j,device='cuda',dtype=torch.long)
                f=g[sl] if j<0 else model.corrected(g[sl],l[sl],a)
                comp[sl,j+1]=detection.uatd_components(model.head(f),t).cpu().numpy()
            resources()
    else:
        model.head.train();names=['loss_objectness','loss_rpn_box_reg','loss_classifier','loss_box_reg']
        for i in range(n):
            for j in range(-1,4):
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(10000+i);a=torch.tensor([j],device='cuda')
                    f=g[i:i+1] if j<0 else model.corrected(g[i:i+1],l[i:i+1],a)
                    loss=model.head(f,reader.batch_targets('FSOD',targets,[i],'cuda'))
                    comp[i,j+1]=torch.stack([loss[name] for name in names]).cpu().numpy()
            if (i+1)%200==0:print(json.dumps({'gain_labels':model.task,'images':i+1}),flush=True);resources()
        model.head.eval()
    scale=np.maximum(comp[fit_ids,0].mean(0),1e-6)
    return ((comp[:,:1]-comp[:,1:])/scale).mean(2)


def train_selector(task,mode,g,gains):
    folder=RUN/task/mode;split=np.load(RUN/task/'selector_split.npz');fit,check=split['fit'],split['check']
    mean=float(gains[fit].mean());scale=max(float(gains[fit].std()),1e-8)
    y=torch.from_numpy((gains-mean)/scale).cuda();torch.manual_seed(7);selector=SpatialSelector().cuda()
    optimizer=torch.optim.AdamW(selector.parameters(),lr=.001,weight_decay=.0001);gen=torch.Generator().manual_seed(7);updates=0
    for epoch in range(100):
        order=torch.tensor(fit)[torch.randperm(len(fit),generator=gen)]
        for ids in order.split(64):
            optimizer.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(selector(g[ids]),y[ids])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite spatial-selector loss')
            loss.backward();optimizer.step();updates+=1
        if epoch%10==0:resources()
    selector.eval().requires_grad_(False)
    with torch.no_grad():scores=torch.cat([selector(g[i:i+64]) for i in range(0,len(g),64)]).cpu().numpy()*scale+mean
    selected=scores[check].argmax(1);truth=gains[check]
    report={'updates':updates,'epochs':100,'parameters':sum(p.numel() for p in selector.parameters()),
        'fit_count':len(fit),'check_count':len(check),'check_argmax_agreement':float((selected==truth.argmax(1)).mean()),
        'check_regret':float((truth.max(1)-truth[np.arange(len(check)),selected]).mean()),
        'gain_mean':mean,'gain_scale':scale,'negative_fraction':float((gains<0).mean()),
        'fixed_region_fit_only':int(gains[fit].mean(0).argmax()),'no_selector_weight_file':True}
    np.save(folder/'signed_training_gains.npy',gains);old.write(folder/'selector_report.json',report)
    return selector,report


@torch.no_grad()
def entropy_actions(model,g):
    if model.task=='Marine':
        logits=model.head(g);h=-(logits.softmax(1)*logits.log_softmax(1)).sum(1);size=16
    elif model.task=='UATD':
        logits=model.head(g)[:,:10];h=-(logits.sigmoid()*F.logsigmoid(logits)+(1-logits.sigmoid())*F.logsigmoid(-logits)).mean(1);size=8
    else:
        logits=model.head.rpn.head([g])[0][0];h=-(logits.sigmoid()*F.logsigmoid(logits)+(1-logits.sigmoid())*F.logsigmoid(-logits)).mean(1);size=16
    return torch.stack([h[:,r:r+size,c:c+size].mean((1,2)) for r,c in ((0,0),(0,size),(size,0),(size,size))],1).argmax(1).cpu().numpy()


@torch.no_grad()
def predict(model,spec,bank,actions):
    g,l=bank;pieces=[];batch=spec['batch']
    for start in range(0,len(g),batch):
        sl=slice(start,start+batch);a=torch.tensor(actions[sl],device='cuda')
        f=g[sl] if bool((a<0).all()) else model.corrected(g[sl],l[sl],a)
        pieces.append(reader.decode(model.task,model.head,f,spec['rows']['validation'][sl]))
    if model.task=='UATD':return [x for part in pieces for x in part]
    return np.concatenate(pieces)


def score_prediction(task,pred,targets,spec,path):
    if task=='UATD':
        old.write(path.with_suffix('.json'),[{k:v.tolist() for k,v in row.items()} for row in pred])
        rows=json.loads(path.with_suffix('.json').read_text())
        pred=[{'boxes':torch.tensor(x['boxes'],dtype=torch.float32).reshape(-1,4),'scores':torch.tensor(x['scores'],dtype=torch.float32),'labels':torch.tensor(x['labels'],dtype=torch.long)} for x in rows]
    else:np.save(path.with_suffix('.npy'),pred);pred=np.load(path.with_suffix('.npy'))
    coco=json.loads(Path(spec['GT_paths']['validation']).read_text()) if task=='FSOD' else None
    canonical,independent=reader.original_scores(task,pred,targets,coco)
    return canonical


def evaluate_mode(model,mode,spec,bank,targets,selector=None,selector_report=None):
    g,l=bank['validation'];n=len(g);policies={'global_only':np.full(n,-1,np.int64)}
    if selector is not None:
        with torch.no_grad():scores=torch.cat([selector(model.context16(g[i:i+64])) for i in range(0,n,64)]).cpu().numpy()*selector_report['gain_scale']+selector_report['gain_mean']
        top=scores.argmax(1);keep=scores.max(1)>0;learned=np.where(keep,top,-1)
        random=np.random.default_rng(2).integers(0,4,n,dtype=np.int64);fixed=selector_report['fixed_region_fit_only']
        entropy=entropy_actions(model,g)
        policies.update(learned=learned,learned_forced_top1=top,random_budgetmatched=np.where(keep,random,-1),
            fixed_budgetmatched=np.where(keep,fixed,-1),entropy_budgetmatched=np.where(keep,entropy,-1),entropy_forced_top1=entropy)
    folder=RUN/model.task/mode;metrics={}
    for policy,actions in policies.items():
        predictions=predict(model,spec,bank['validation'],actions)
        metrics[policy]=score_prediction(model.task,predictions,targets['validation'],spec,folder/(policy+'_prediction'))
        print(json.dumps({'score':model.task+'/'+mode+'/'+policy,'value':metrics[policy][spec['metric']]}),flush=True)
    np.savez(folder/'actions.npz',**policies)
    report={'task':model.task,'mode':mode,'terminal_updates':2000,'metrics':metrics,
        'selection_fraction':float((policies['learned']>=0).mean()) if 'learned' in policies else 0.,
        'original_GT_scores_from_saved_predictions':True,'new_model_weight_files':0,
        'references':protocol()['references'][model.task],'resources':resources()}
    old.write(folder/'results.json',report)
    return report,policies


def online_cost(model,mode,spec,selector,selector_report):
    rows=spec['rows']['validation'][:32];samples=[]
    for row in rows:
        with Image.open(row['image']) as im:a=torch.from_numpy(np.array(im.convert('L'),dtype=np.uint8)).float()[None]/255
        samples.append({'intensity':a,'valid':torch.ones(a.shape[-2:],dtype=torch.bool)})
    cells=[]
    for policy in (('global_only','learned') if selector is not None else ('global_only',)):
        load=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        baseline=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats();records=[];selected_count=0
        for visit in range(146):
            i=visit%32 if visit<50 else (visit-50)%32
            torch.cuda.synchronize();start=time.perf_counter()
            common=prepare_images([samples[i]],spec['normalization'],None)
            if model.task!='UATD':common=F.interpolate(common,(448,448),mode='bilinear',align_corners=False,antialias=True)
            with torch.no_grad():
                images=common.cuda();g=model.global_field(images);action=-1
                if policy=='learned':
                    scores=selector(model.context16(g))*selector_report['gain_scale']+selector_report['gain_mean']
                    if float(scores.max())>0:action=int(scores.argmax(1))
                if action>=0:
                    a=torch.tensor([action],device='cuda');local=model.local_fields(images,a);field=model.corrected(g,local,a)
                else:field=g
                prediction=reader.decode(model.task,model.head,field,rows[i:i+1])
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if visit>=50:records.append({'row_id':spec['row_ids']['validation'][i],'round':(visit-50)//32,'seconds':seconds,'action':action});selected_count+=int(action>=0)
        times=[x['seconds'] for x in records]
        cells.append({'policy':policy,'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),
            'selected_fraction_timed':selected_count/96,'raw':records,'baseline_reserved':baseline,
            'peak_reserved':torch.cuda.max_memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'load':load})
    old.write(RUN/model.task/mode/'cost.json',{'cells':cells,'boundary':'CPU native image through preprocessing/H2D/actualencoder(s)/selector/fusion/head/nativepostproc; decode/setup excluded',
        'other_GPU_training_may_be_active':True,'same_GPU_other_own_model_workers':False,
        'memory_scope':'Feature banks/images remain resident in this process; this is not isolated deployment memory'})


def swise_uatd_control(spec,images,targets,dense):
    folder=RUN/'UATD/official_swise_pool16';folder.mkdir()
    cfg=protocol()['encoder'];encoder=build_encoder(Path(cfg['code_dir']),Path(cfg['weights_path']),'R010').cuda().eval().requires_grad_(False)
    fields={}
    with torch.no_grad():
        for split,x in images.items():
            fields[split]=torch.empty((len(x),384,16,16),device='cuda')
            for start in range(0,len(x),8):
                large=F.interpolate(x[start:start+8],(448,448),mode='bilinear',align_corners=False,antialias=True)
                f=encoder(large)['patch_tokens'].transpose(1,2).reshape(-1,384,32,32)
                fields[split][start:start+len(f)]=F.avg_pool2d(f,2)
                if start%256==0:resources()
    head=reader.new_head('UATD',0).cuda();opt=reader.head_optimizer('UATD',head);gen=torch.Generator().manual_seed(0);updates=0
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1,101):
            for ids in torch.randperm(6051,generator=gen).split(32):
                opt.zero_grad(set_to_none=True);loss=reader.task_loss('UATD',head,fields['train'][ids],tuple(x[ids] for x in dense['train']))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SWiSE-UATD control loss')
                loss.backward();opt.step();updates+=1
            if epoch%10==0:log.write(json.dumps({'epoch':epoch,'updates':updates})+'\n');log.flush();print(json.dumps({'control':'UATD Whole448 pool16','epoch':epoch}),flush=True);resources()
    head.eval();pieces=[]
    with torch.no_grad():
        for start in range(0,1505,32):pieces.extend(reader.decode('UATD',head,fields['validation'][start:start+32],spec['rows']['validation'][start:start+32]))
    metrics=score_prediction('UATD',pieces,targets['validation'],spec,folder/'terminal_prediction')
    old.write(folder/'results.json',{'terminal':metrics,'epochs':100,'updates':updates,'no_weight_files':True,
        'definition':'New original official frozenDINO whole448 then averagepool32to16, original dense16 head/GT; this is explicit task-interface adaptation, not an old UATD SWiSE result',
        'resources':resources()})


def task_workflow(task):
    spec,images,targets,dense=data_for(task)
    if task=='UATD':swise_uatd_control(spec,images,targets,dense)
    initial=torch.load(spec['initial_checkpoint'],map_location='cpu',weights_only=False)
    summaries=[]
    for mode in MODES:
        model=Model(task,initial).cuda();train(model,mode,spec,images['train'],targets['train'],dense.get('train'))
        bank=feature_bank(model,images,mode!='continue');selector=sr=None
        if mode!='continue':
            gains=gain_labels(model,spec,bank['train'],targets['train'],dense.get('train'))
            selector,sr=train_selector(task,mode,model.context16(bank['train'][0]),gains)
        result,actions=evaluate_mode(model,mode,spec,bank,targets,selector,sr)
        online_cost(model,mode,spec,selector,sr);summaries.append(result)
        del model,bank,selector,sr
    old.write(RUN/task/'results.json',{'status':'COMPLETE_FOUR_PRESPECIFIED_MODES','modes':summaries,'weights_saved':False})


def pipeline():
    RUN.mkdir();start=time.time();prep=json.loads((BASE/'downstream_vit_finetuning_v1/cpu_preparation.json').read_text())
    references=json.loads((BASE/'gsr_detection_transfer_v1/strongest_baseline_comparison.json').read_text())['tasks']
    p={'unit':RUN.name,'start_unix':start,'deadline_unix':start+28800,'encoder':prep['official_config'],'tasks':{},'references':references,
        'source_utf8':Path(__file__).read_text(),'authorization':'User permits frozen, partial/full DINO adaptation; onlyphysicalGPU2/3; newweights not saved, oldfiles retained',
        'initialization':'Existing seed0 target-finetuned terminal encoder+head for each task. Full ancestral training costs remain historical; not a from-scratch equal-budget comparison.',
        'modes':list(MODES),'new_updates_per_mode':2000,'task_mode_fits':12,
        'learning_rates':{'body':2e-6,'task_head':1e-4,'fusion':.001},'optimizers':'Original bodyAdamW/matrixwd.05, original head optimizer, fusionAdamWwd.0001; new optimizer moments, no exact old-run resume',
        'training_objective':'continue: global taskloss; GSR:0.5global+0.5random-local-corrected taskloss, sameGT. Extra encoder/head compute is reported.',
        'partial_trainability':'Last4 transformerblocks8..11 and finalnorm, taskhead andfusion; unusedmask protected',
        'strong_global_path':'FSOD/Marine use448 global32, UATD224 global16; performance version, not old low-budget224-global recipe',
        'selector':'Spatial386->64/Depthwise3x3/score-map -> quadrantpool; only trainG16+coords, noGT at inference; balanced signed gains, seed7 AdamW.001 B64 100epochs fit80/check20',
        'inference_budget':'At most one additional local224; select none when predicted best signed gain<=0. Alsoforcedtop1 and matchingselection-mask random/fixed/entropy controls.',
        'selector_change_is_part_of_whole_candidate':'Not an isolated attribution to gain target, spatial module, trainability or global input size.',
        'new_official_UATD_SWiSE_control':'Frozen official whole448->averagepool16->originaldense head100epochs19000updates; required because no matched old UATDWhole448 result existed.',
        'storage':'No new model/optimizer/encoder/selector weight files or persistent feature caches; log/results/gain labels/necessary predictions only',
        'caps':{'workflow_seconds':10800,'global_seconds':28800,'GPU_reserved':48*1024**3,'RSS':48*1024**3,'new_disk':4*1024**3},
        'stop':'Genuine software/nonfinite/resource error stops owned workers; no automatic fallback or precision/batch/threshold/seed change',
        'SOTA_terminology':'Listed models are strongest local completed compatible references, not a certified worldwideSOTA. Recent CTFS/RSOD/UATDDEIMv2 sources differ in dataset/split/labels/metric and are kept separate.',
        'SWiSE_references':{'FSOD':{'terminal':.902005317706368,'source':str(BASE/'whole448_frozen_reference_v1/results.json')},'Marine':{'terminal':.6059200833586624,'source':str(BASE/'whole448_frozen_reference_v1/results.json')},'UATD':{'source':'New official_swise_pool16 control in this wave','old_comparable_endpoint_available':False}},
        'GPU_schedule':'StartFSOD on2/UATD on3; Marine starts on whichever permitted card actually finishes first, one owned worker per card.',
        'external_primary_sources':{'CTFS':'https://github.com/pingggg516/CTFS','RSOD':'https://arxiv.org/abs/2601.12715','UATD_DEIMv2':'https://www.preprints.org/manuscript/202609.0449'},
        'no_smoke_SHA_review':True}
    for task,batch in [('FSOD',8),('Marine',16),('UATD',32)]:
        folder=RUN/task;folder.mkdir();data=prep['inputs'][task]
        spec={**data,'batch':batch,'row_ids':prep['row_ids'][task],'metric':{'FSOD':'AP50','Marine':'miou12','UATD':'map50'}[task],
            'input_views':prep['input_caches'][task],'initial_checkpoint':str(BASE/f'downstream_vit_finetuning_v1/fits/{task}/e2e/seed0/latest.pt')}
        p['tasks'][task]=spec;n=len(data['rows']['train']);generator=torch.Generator().manual_seed(0);ids=[]
        while len(ids)<2000*batch:ids.extend(torch.randperm(n,generator=generator).tolist())
        np.save(folder/'plan.npy',np.array(ids[:2000*batch],np.int64).reshape(2000,batch))
        np.save(folder/'regions.npy',np.random.default_rng(1).integers(0,4,(2000,batch),dtype=np.int64))
        split=np.random.default_rng(37).permutation(n);cut=int(.8*n)
        np.savez(folder/'selector_split.npz',fit=split[:cut],check=split[cut:])
    old.write(RUN/'protocol.json',p);(RUN/'logs').mkdir();pending=[('FSOD',2),('UATD',3),('Marine',3)];children={};rows=[]
    def launch(task,gpu):
        admission=old.admission(gpu);env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','XFORMERS_DISABLED':'1'}
        log=(RUN/'logs'/f'{task}.log').open('w');child=subprocess.Popen([old.PYTHON,'-B','-u','-m','runs.gsr_strong_adaptation',task],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[task]=(child,log,time.time(),gpu,admission)
    try:
        launch(*pending.pop(0));launch(*pending.pop(0))
        old.write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':{k:v[0].pid for k,v in children.items()},'protocol':str(RUN/'protocol.json'),'new_weights_saved':False})
        while children:
            for task,(child,log,begun,gpu,admission) in list(children.items()):
                if time.time()-begun>10800 or time.time()>p['deadline_unix']:os.killpg(child.pid,signal.SIGKILL)
                if child.poll() is None:continue
                code=child.wait();log.close();del children[task];row={'task':task,'PID':child.pid,'physical_GPU':gpu,'actual_wait_exit':code,'wall_seconds':time.time()-begun,'admission':admission};rows.append(row)
                old.write(RUN/'queue_state.json',{'status':'RUNNING','active':{k:v[0].pid for k,v in children.items()},'completed':rows})
                if code!=0:raise RuntimeError(task+' actual exit '+str(code))
                if pending:
                    next_task,_=pending.pop(0);launch(next_task,gpu)
            time.sleep(1)
        summary={task:json.loads((RUN/task/'results.json').read_text()) for task in ('FSOD','Marine','UATD')}
        old.write(RUN/'results.json',{'status':'COMPLETE_FIXED_PERFORMANCE_WAVE_NOT_GOAL_ACCEPTANCE','tasks':summary,'references':references})
        disk=sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file())
        if disk>4*1024**3:raise RuntimeError('Results-only disk cap exceeded')
        close={'status':'COMPLETE_FIXED_WAVE_GOAL_REQUIRES_RESULT_DECISION','actual_waits':rows,'supervisor_PID':os.getpid(),'pipeline_wall_seconds':time.time()-start,
            'summed_child_wall_seconds':sum(x['wall_seconds'] for x in rows),'new_entity_bytes':disk,'new_weight_files':0,'no_new_tests_SHA_reviews':True,'results':str(RUN/'results.json')}
        old.write(RUN/'wave_closure.json',close);old.write(RUN/'writing_handoff.json',close);old.write(RUN/'queue_state.json',{'status':'COMPLETE_WAVE','actual_waits':rows})
    except BaseException:
        for task,(child,log,begun,gpu,admission) in children.items():
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            code=child.wait();log.close();rows.append({'task':task,'PID':child.pid,'actual_wait_exit':code,'cancelled_after_failure':True})
        old.write(RUN/'failure_closure.json',{'status':'STOPPED_NO_AUTOMATIC_RETRY','traceback':traceback.format_exc(),'actual_waits':rows,'new_weight_files':0})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pipeline','FSOD','Marine','UATD']);args=parser.parse_args();old.runtime()
    if args.phase=='pipeline':pipeline()
    else:task_workflow(args.phase)
