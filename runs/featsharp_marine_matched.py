"""Paper-default FeatSharp architecture under the matched GLISR source/Marine recipe."""
import argparse, importlib.util, json, os, random, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1]))); BASE=Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts'))); RUN=BASE/'featsharp_marine_matched_v1'
PARENT=Path(__file__).resolve().parent
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / 'third_party') / 'featsharp_adapted'))
os.environ['XFORMERS_DISABLED']='1'
from featsharp.upsampler_modules.featsharp import FeatSharp
spec=importlib.util.spec_from_file_location('matched_reference',PARENT/'entry_snapshot.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
core=ref.core; reader=ref.reader
DEADLINE=0.

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def guard(deadline):
    obs={'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_reserved_bytes':torch.cuda.max_memory_reserved()}
    if time.time()>deadline:raise TimeoutError('Frozen stage deadline')
    if obs['RSS_bytes']>96*1024**3 or obs['GPU_reserved_bytes']>64*1024**3:raise RuntimeError(str(obs))
    return obs

class Model(nn.Module):
    def __init__(self,config,seed):
        super().__init__();self.encoder=ref.build_encoder(Path(config['code_dir']),Path(config['weights_path']),'R010')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101+seed)
            self.fusion=FeatSharp(224,14,384,2,inputs=['jbu','tiles'],no_ln=False,window_size=7)
        self.encoder_images=0;self.encoder_calls=0
    def tile_generator(self,x,return_summary=False):
        self.encoder_images+=len(x);self.encoder_calls+=1
        return self.encoder(x)['patch_tokens'].transpose(1,2).reshape(len(x),384,16,16)
    def field(self,x,head=None,kind=None,policy='featsharp',actions=None):
        # The only observation is the original standardized224 tensor; no native high-res pixels.
        large=F.interpolate(x,(448,448),mode='bilinear',align_corners=False)
        # Official create_tiles downsamples this guidance to224 for its global view.
        # Preserve the exact original224 global observation instead via the callback below.
        self.global_input=x
        result=self.fusion(large.repeat(1,3,1,1),self.matched_tiles)[1]
        del self.global_input
        return result
    def matched_tiles(self,packed,return_summary=False):
        n=len(self.global_input)
        # Official tile layout is global B first then (B,row,col) four local views.
        packed=torch.cat((self.global_input,packed[n:,:1]),0)
        return self.tile_generator(packed,return_summary)

def setup(seed):
    ref.original.runtime();ref.REPEAT=seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    core.guard=guard

def checkpoint(folder,name,model,head,epoch,optimizers,generator):
    payload={'model':model.state_dict(),'head':head.state_dict(),'epoch':epoch,'seed':int(folder.name[1:]),'protocol':str(RUN/'protocol.json'),'optimizers':[o.state_dict() for o in optimizers],'private_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
    tmp=folder/(name+'.tmp');torch.save(payload,tmp);tmp.replace(folder/(name+'.pt'))

def check(p):
    setup(0);model=Model(p['encoder'],0).cuda().train();reader.configure_encoder(model.encoder,'e2e');head=reader.new_head('Marine',0).cuda()
    tasks=json.loads(Path(p['data_manifest']).read_text())['tasks'];marine=tasks['Marine'];x=torch.from_numpy(np.load(marine['views']['train'],mmap_mode='r')[:16].copy()).cuda();labels=core.targets_for('Marine',marine,'train')[:16].cuda()
    captured=[]
    handle=model.encoder.backbone.patch_embed.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    before=model.encoder.backbone.blocks[0].attn.qkv.weight.detach().clone();opt=torch.optim.SGD(model.parameters(),lr=.001)
    out=model.field(x);loss=core.task_loss('Marine',marine,head,out,labels);loss.backward()
    groups={'backbone':model.encoder.backbone.blocks[0].attn.qkv.weight,'JBU':next(model.fusion.ups[0].jbu.parameters()),'attention':model.fusion.ups[0].block.attn.qkv.weight,'MLP':model.fusion.ups[0].block.mlp.w12.weight,'head':head.weight}
    grads={k:float(v.grad.norm()) for k,v in groups.items()}
    if not all(np.isfinite(v) and v>0 for v in grads.values()):raise RuntimeError(str(grads))
    opt.step();handle.remove()
    expected=F.interpolate(x,(448,448),mode='bilinear',align_corners=False).reshape(16,1,2,224,2,224).permute(0,2,4,1,3,5).reshape(64,1,224,224)
    assert torch.equal(captured[0][:16],x) and torch.equal(captured[0][16:],expected)
    assert out.shape==(16,384,32,32) and out.dtype==torch.float32
    assert not torch.equal(before,model.encoder.backbone.blocks[0].attn.qkv.weight)
    prediction=F.interpolate(head(out.detach()),(480,320),mode='bilinear',align_corners=False).argmax(1).cpu().numpy()
    # Same native-label confusion formula as full evaluation, verified against direct bincount.
    matrix=np.bincount((labels.cpu().numpy()*12+prediction).ravel(),minlength=144).reshape(12,12)
    assert int(matrix.sum())==16*480*320
    model.eval()
    with torch.no_grad(): model.field(x[:11])
    model.train();opt.zero_grad(set_to_none=True)
    core.task_loss('Marine',marine,head,model.field(x),labels).backward()
    assert torch.isfinite(model.encoder.backbone.blocks[0].attn.qkv.weight.grad).all()
    write(RUN/'integration_check.json',{'status':'PASS','real_training_rows':list(range(16)),'field':list(out.shape),'dtype':str(out.dtype),'actual_encoder_input':list(captured[0].shape),'global224_exact':True,'four_tiles_layout_exact':True,'finite_nonzero_gradients':grads,'actual_backbone_step_moved':True,'native_confusion_pixels':int(matrix.sum()),'batch16_train_eval11_train_backward':True,'scope':'Discarded engineering check; no validation selection; no training state reused','resources':guard(DEADLINE)})

def source(seed,p,folder):
    setup(seed);model=Model(p['encoder'],seed).cuda();model.encoder.eval().requires_grad_(False);versions=[v._version for v in model.encoder.parameters()]
    tasks=json.loads(Path(p['data_manifest']).read_text())['tasks'];names=p['source_tasks'];images={t:torch.from_numpy(np.load(tasks[t]['views']['train'],mmap_mode='r').copy()).cuda() for t in names};labels={t:core.targets_for(t,tasks[t],'train') for t in names}
    heads=nn.ModuleDict({t:ref.new_head(t,tasks[t]).cuda() for t in names});opts={t:reader.head_optimizer('FSOD' if tasks[t]['kind']=='detection' else 'Marine',heads[t]) for t in names}
    opt=torch.optim.AdamW(model.fusion.parameters(),lr=.001,weight_decay=.0001);plans=np.load(folder/'source_plan.npz');begun=time.monotonic()
    with (folder/'source.jsonl').open('w') as log:
        for u in range(6000):
            opt.zero_grad(set_to_none=True);opt.param_groups[0]['lr']=ref.previous.shared.native.module_lr(u,6000,.001);losses={}
            for task in names:
                h=heads[task];h.train();o=opts[task];o.zero_grad(set_to_none=True);o.param_groups[0]['lr']=(.005 if u<4000 else .0005 if u<5500 else .00005) if tasks[task]['kind']=='detection' else ref.previous.shared.native.module_lr(u,6000,.001)
                ids=plans[task][u];losses[task]=0.
                for start in range(0,len(ids),4):
                    ix=ids[start:start+4].tolist();field=model.field(images[task][ix]);target=core.batch_targets(task,tasks[task],labels[task],ix)
                    loss=core.task_loss(task,tasks[task],h,field,target)*len(ix)/len(ids)/len(names);loss.backward();losses[task]+=float(loss.detach())
                o.step()
            norm=torch.nn.utils.clip_grad_norm_(model.fusion.parameters(),1.,error_if_nonfinite=True);opt.step();obs=guard(DEADLINE)
            if u==0:write(folder/'first_source_update.json',{'counted_updates':1,'fusion_grad_norm':float(norm),'backbone_requires_grad':any(v.requires_grad for v in model.encoder.parameters()),'field':list(field.shape),'encoder_views':model.encoder_images})
            if (u+1)%50==0:
                row={'update':u+1,'losses':losses,'wall_seconds':time.monotonic()-begun,**obs};log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if (u+1)%1000==0:
                torch.save({'model':model.state_dict(),'heads':heads.state_dict(),'optimizer':opt.state_dict(),'head_optimizers':{t:o.state_dict() for t,o in opts.items()},'update':u+1,'seed':seed,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},folder/'source_latest.tmp');(folder/'source_latest.tmp').replace(folder/'source_latest.pt')
    assert versions==[v._version for v in model.encoder.parameters()]
    write(folder/'source_results.json',{'status':'COMPLETE','updates':6000,'student_visits':p['source_visits'],'encoder_views':model.encoder_images,'encoder_calls':model.encoder_calls,'backbone_unchanged':True,'wall_seconds':time.monotonic()-begun,**guard(DEADLINE)})

def fit(seed,p,folder):
    setup(seed);model=Model(p['encoder'],seed);model.load_state_dict(torch.load(folder/'source_latest.pt',map_location='cpu',weights_only=False)['model'],strict=True);reader.configure_encoder(model.encoder,'e2e');model.cuda();head=reader.new_head('Marine',seed).cuda()
    bo=reader.body_optimizer(model.encoder);ho=reader.head_optimizer('Marine',head);fo=torch.optim.AdamW(model.fusion.parameters(),lr=.0001,weight_decay=.0001);opts=(bo,ho,fo)
    spec=json.loads(Path(p['data_manifest']).read_text())['tasks']['Marine'];images={s:torch.from_numpy(np.load(spec['views'][s],mmap_mode='r').copy()).cuda() for s in ('train','validation')};labels={s:core.targets_for('Marine',spec,s) for s in images}
    gen=torch.Generator().manual_seed(seed);updates=0;best=-float('inf');begun=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1,101):
            model.train();head.train();order=torch.randperm(len(images['train']),generator=gen);total=0.
            for ids in order.split(16):
                for o in opts:o.zero_grad(set_to_none=True)
                for g in bo.param_groups:g['lr']=reader.body_learning_rate(updates,6300)
                fo.param_groups[0]['lr']=ref.previous.shared.native.module_lr(updates,6300,.0001)
                field=model.field(images['train'][ids.cuda()]);target=core.batch_targets('Marine',spec,labels['train'],ids.tolist());loss=core.task_loss('Marine',spec,head,field,target);loss.backward()
                bn=torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),1.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(head.parameters(),float('inf'),error_if_nonfinite=True);fn=torch.nn.utils.clip_grad_norm_(model.fusion.parameters(),1.,error_if_nonfinite=True)
                for o in opts:o.step()
                updates+=1;total+=float(loss.detach())*len(ids);guard(DEADLINE)
                if updates==1:write(folder/'first_target_update.json',{'first_ids':ids.tolist(),'body_grad_norm':float(bn),'fusion_grad_norm':float(fn),'field':list(field.shape),'dtype':str(field.dtype)})
            metric,ind,pred=core.evaluate('Marine',spec,model,head,images['validation'],labels['validation'],16,DEADLINE,'featsharp')
            row={'epoch':epoch,'updates':updates,'loss':total/len(order),'miou12':metric['miou12'],'wall_seconds':time.monotonic()-begun};log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if metric['miou12']>best:
                best=metric['miou12'];best_metric=metric;best_epoch=epoch;checkpoint(folder,'best',model,head,epoch,opts,gen);torch.save({'prediction':pred,'epoch':epoch,'task':'Marine'},folder/'best_predictions.pt')
            checkpoint(folder,'latest',model,head,epoch,opts,gen)
            if epoch==100:torch.save({'prediction':pred,'epoch':epoch,'task':'Marine'},folder/'terminal_predictions.pt')
    assert updates==6300
    write(folder/'results.json',{'status':'COMPLETE','seed':seed,'terminal':metric,'terminal_independent':ind,'best':best_metric,'best_epoch':best_epoch,'terminal_checkpoint':str(folder/'latest.pt'),'terminal_epoch':100,'updates':updates,'encoder_views':model.encoder_images,'wall_seconds':time.monotonic()-begun,**guard(DEADLINE)})

def score(seed,p,folder):
    setup(seed);spec=json.loads(Path(p['data_manifest']).read_text())['tasks']['Marine'];labels=core.targets_for('Marine',spec,'validation');out={}
    r=json.loads((folder/'results.json').read_text())
    for endpoint in ('terminal','best'):
        pred=torch.load(folder/f'{endpoint}_predictions.pt',map_location='cpu',weights_only=False)['prediction'];matrix=np.zeros((12,12),dtype=np.int64)
        for a,y in zip(pred,labels.numpy()):
            for c in range(12):matrix[c]+=np.bincount(np.asarray(a)[y==c],minlength=12)
        inter=np.diag(matrix);union=matrix.sum(0)+matrix.sum(1)-inter;iou=np.divide(inter,union,out=np.zeros(12),where=union>0);miou=float(iou.mean())
        assert abs(miou-r[endpoint]['miou12'])<1e-12
        out[endpoint]={'miou12':miou,'per_class_IoU':iou.tolist(),'confusion_matrix':matrix.tolist(),'epoch':100 if endpoint=='terminal' else r['best_epoch']}
    write(folder/'verification.json',{'status':'PASS_INDEPENDENT_SAVED_PREDICTIONS','endpoints':out})

def prepare():
    p=json.loads((PARENT/'protocol.json').read_text());p={k:p[k] for k in ('encoder','data_manifest','source_tasks','source_macro_per_task','source_visits','target_schedules','evaluation_every_epochs')}
    p.update({'name':'FeatSharp JBU+Tiles architecture, matched supervised source and downstream recipe','seeds':[0,1,2],'source_updates':6000,'official_commit':'6ce9b18e501dd136a63bf852ec02a15cbe82491d','architecture':{'upsampler':'JBU+Tiles','LayerNorm':True,'attention_heads':16,'window':7,'tile_position':True,'feature_dim':384,'output':[384,32,32]},'adaptations':['ViT-S/14 without register, same project R010 input instead of official L/G RGB wrapper','original standardized224 used exactly for global view; bilinear448 generated only from same224','JBU guidance repeats standardized grayscale intoRGB; no additional ImageNet normalization','No PHI-S or learned debias; matched postLN feature interface','6000 supervised six-interface source replaces original multiview consistency; original backbone frozen only during source','full backbone/fusion/head fine-tuning onMarine; official no_grad/autocast featurizer wrapper not used','Remove custom CUDA architecture flag suppression; compile for installedGPU usingPyTorch defaults'], 'resources':{'global_seconds':259200,'source_seconds':86400,'target_seconds':21600,'score_seconds':300,'check_seconds':1200,'GPUs':[2,3],'GPU_reserved_GiB':64,'RSS_GiB':96,'new_disk_GiB':24,'free_MiB':70000},'primary':'Marine terminal epoch100; best auxiliary; report all seeds regardless direction','seed_order':'seed0 full source+target+score first; then seeds1 and2 independently onGPU2/3','no_extra_tasks':'UATD not started in this unit; no HPO or extra variants','baseline_paths':[str(PARENT/f'full_s{s}_Marine/results.json') for s in (0,1,2)],'created_unix':time.time()})
    write(RUN/'protocol.json',p)
    tasks=json.loads(Path(p['data_manifest']).read_text())['tasks']
    for seed in range(3):
        folder=RUN/f's{seed}';folder.mkdir();plans={}
        for i,t in enumerate(p['source_tasks']):
            n=len(tasks[t]['rows']['train']);k=p['source_macro_per_task'][t];rng=np.random.RandomState(13+i+1000*seed);plans[t]=np.concatenate([rng.permutation(n) for _ in range(int(np.ceil(6000*k/n)))])[:6000*k].reshape(6000,k)
        np.savez(folder/'source_plan.npz',**plans)
    write(RUN/'source_scope.json',{'HSMLA':'Paper changes backbone attention, not an upsampler. Paper HTML and one targeted repository search did not bind an official implementation/checkpoint; no model experiment allocated. No claim that none exists.','sources':['https://arxiv.org/html/2502.16025v2','https://github.com/NVlabs/FeatSharp','https://arxiv.org/html/2608.07616v2'],'official_weights':'README says pretrained upsampler weights not available; this is trained from scratch.','planning':'3x6000 source updates +3x6300 Marine updates; perseed source216000 observations/1080000 backbone views plusMarine5views per sample.72hglobal hardbudget, measured prefix timings will inform estimate, no quality stopping.'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','check','source','fit','score']);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--deadline',type=float,default=0);a=parser.parse_args();DEADLINE=a.deadline
    if a.phase=='prepare':prepare()
    else:
        p=json.loads((RUN/'protocol.json').read_text());folder=RUN/f's{a.seed}'
        try:
            if a.phase=='check':check(p)
            else:globals()[a.phase](a.seed,p,folder)
        except BaseException:
            write((RUN if a.phase=='check' else folder)/(a.phase+'_failure.json'),{'phase':a.phase,'traceback':traceback.format_exc(),'time_unix':time.time()});raise
