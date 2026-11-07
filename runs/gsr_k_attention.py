"""Frozen CLS-attention-guided sparse448 encoding; three fixed budget-matched rules."""
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from runs import gsr_multidata_training as core
from runs.gsr_online_gain_routing import gpu_admission
from sonar_encoder.training.model import build_encoder, prepare_images

RUN = core.BASE/'gsr_k_attention_v1'
RULES = ('attention', 'fixed', 'random')
TASKS = ('FSOD', 'UATD', 'Marine')
write = core.write
DEADLINE = 0.


def guard():
    if time.time() > DEADLINE: raise TimeoutError('K=attention fixed deadline')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    peak = torch.cuda.max_memory_reserved()
    if rss > 48*1024**3 or peak > 32*1024**3: raise RuntimeError(f'Resource cap RSS={rss},GPU={peak}')
    return {'RSS_bytes': rss, 'GPU_reserved_bytes': peak}


class SparseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010').eval().requires_grad_(False).cuda()
        self.bb = self.encoder.backbone
        if self.bb.num_register_tokens != 0 or self.bb.patch_size != 14: raise RuntimeError('Unexpected official backbone layout')
        self.pe = self.bb.interpolate_pos_encoding(torch.empty(1,1025,384,device='cuda'),448,448)
        self.branch = 'none'; self.histograms = {}; self.first_witness = None
        self.bb.blocks[-1].attn.qkv.register_forward_hook(self.capture)
        for i, block in enumerate(self.bb.blocks): block.register_forward_pre_hook(self.block_hook(i))
        ids = np.arange(1024); row,col = ids//32,ids%32
        distance = np.maximum(abs(row-15.5),abs(col-15.5))
        self.fixed = torch.tensor(np.lexsort((ids,distance)),device='cuda')
        self.versions = [v._version for v in self.encoder.parameters()]

    def capture(self, module, args, output):
        if self.branch == 'global': self.qkv = output.detach()

    def block_hook(self, index):
        def hook(module,args):
            shape=list(args[0].shape)
            key=f'{self.branch}/block{index}/tokens{shape[1]}'
            self.histograms[key]=self.histograms.get(key,0)+shape[0]
            if self.branch in RULES and not 65 <= shape[1] <= 257: raise RuntimeError('Sparse Transformer received non-sparse sequence')
        return hook

    @torch.no_grad()
    def global_features(self,x):
        self.branch='global'
        g=self.encoder(x)['patch_tokens'].transpose(1,2).reshape(len(x),384,16,16)
        a=self.bb.blocks[-1].attn; qkv=self.qkv.reshape(len(x),257,3,a.num_heads,384//a.num_heads)
        q=qkv[:,0,0].unsqueeze(2);k=qkv[:,:,1].transpose(1,2)
        probabilities=((q*a.scale)@k.transpose(-1,-2)).softmax(-1)
        heat=probabilities[:,:,:,1:].mean(1).reshape(len(x),1,16,16)
        mass=F.interpolate(heat,(32,32),mode='bilinear',align_corners=False).flatten(1)
        mass=mass/mass.sum(1,keepdim=True)
        order=mass.argsort(dim=1,descending=True,stable=True)
        cumulative=mass.gather(1,order).cumsum(1)
        k80=(cumulative < .8).sum(1)+1
        counts=k80.clamp(64,256)
        return g,mass,order,counts,k80

    def select(self, order, counts, rule, row_indices):
        result=torch.full((len(counts),256),-1,device='cuda',dtype=torch.long)
        for i,k in enumerate(counts.tolist()):
            if rule=='attention': chosen=order[i,:k]
            elif rule=='fixed': chosen=self.fixed[:k]
            else:
                chosen=torch.tensor(np.random.RandomState(17000+int(row_indices[i])).permutation(1024)[:k],device='cuda')
            result[i,:k]=chosen.sort().values
        return result

    @torch.no_grad()
    def sparse_features(self,x,ids,counts,rule):
        self.branch=rule
        large=F.interpolate(x,(448,448),mode='bilinear',align_corners=False,antialias=True)
        patch_view=large.unfold(2,14,14).unfold(3,14,14).permute(0,2,3,1,4,5)
        output=torch.zeros(len(x),256,384,device='cuda')
        for k in counts.unique().tolist():
            group=torch.where(counts==k)[0]; selected=ids[group,:k]
            patches=patch_view[group[:,None],selected//32,selected%32].reshape(-1,1,14,14)
            projected=self.bb.patch_embed.proj(patches).reshape(len(group),k,384)
            projected=self.bb.patch_embed.norm(projected)
            selected_pe=self.pe[:,1:][0,selected]
            tokens=torch.cat(((self.bb.cls_token+self.pe[:,:1]).expand(len(group),-1,-1),projected+selected_pe),1)
            if self.first_witness is None:
                expected=torch.stack([large[int(group[0]),:,int(j)//32*14:int(j)//32*14+14,int(j)%32*14:int(j)%32*14+14] for j in selected[0]])
                if not torch.equal(patches[:k],expected): raise RuntimeError('Patch raster extraction disagreement')
                pe_expected=torch.stack([self.pe[0,int(j)+1] for j in selected[0]])
                if not torch.equal(selected_pe[0],pe_expected): raise RuntimeError('Original448 positional encoding disagreement')
                lengths=[]
                handles=[block.register_forward_pre_hook(lambda m,a: lengths.append(int(a[0].shape[1]))) for block in self.bb.blocks]
                for block in self.bb.blocks: tokens=block(tokens)
                for handle in handles: handle.remove()
                self.first_witness={'selected_K':k,'selected_raster_ids':selected[0].tolist(),'actual_block_sequence_lengths':lengths,
                    'patch_pixels_exact':True,'original448_position_exact':True,'projection_patch_count':len(group)*k,
                    'projection_input_shape':list(patches.shape),'global_token_count':1,'dense448_encoding_before_selection':False,
                    'no_zero_mask_or_padding_inside_transformer':True}
                if lengths != [k+1]*12: raise RuntimeError('Actual sparse block length mismatch')
            else:
                for block in self.bb.blocks: tokens=block(tokens)
            output[group,:k]=self.bb.norm(tokens)[:,1:]
        return output

    def frozen_witness(self):
        return {'requires_grad_any':any(v.requires_grad for v in self.encoder.parameters()),
            'gradient_any':any(v.grad is not None for v in self.encoder.parameters()),
            'parameter_versions_unchanged':self.versions==[v._version for v in self.encoder.parameters()]}


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(1154,64);self.output=nn.Linear(64,384);self.gate=nn.Linear(64,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias);nn.init.zeros_(self.gate.bias)

    def forward(self,g,local,ids,counts):
        base=F.interpolate(g,(32,32),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
        index=ids.clamp_min(0);selected=base.gather(1,index[:,:,None].expand(-1,-1,384))
        whole=g.mean((2,3))[:,None].expand(-1,256,-1)
        xy=torch.stack(((index%32+.5)/16-1,(index//32+.5)/16-1),-1)
        u=F.gelu(self.input(torch.cat((local-selected,selected,whole,xy),-1)))
        delta=self.output(u)*self.gate(u).sigmoid()
        delta=delta*(torch.arange(256,device=g.device)[None]<counts[:,None])[:,:,None]
        return base.scatter_add(1,index[:,:,None].expand(-1,-1,384),delta).transpose(1,2).reshape(-1,384,32,32)


def extract(task,spec,encoder,deadline):
    folder=RUN/task/'features';folder.mkdir()
    reports={};begun=time.monotonic()
    for split in ('train','validation'):
        x=np.load(spec['views'][split],mmap_mode='r',allow_pickle=False);n=len(x)
        arrays={'global':np.lib.format.open_memmap(folder/f'{split}_global.npy',mode='w+',dtype='float32',shape=(n,384,16,16))}
        for rule in RULES:
            arrays[rule]=np.lib.format.open_memmap(folder/f'{split}_{rule}.npy',mode='w+',dtype='float32',shape=(n,256,384))
            arrays[rule+'_ids']=np.lib.format.open_memmap(folder/f'{split}_{rule}_ids.npy',mode='w+',dtype='int16',shape=(n,256))
        traces=[]
        for start in range(0,n,16):
            if time.time()>deadline:raise TimeoutError('Extraction phase deadline')
            end=min(n,start+16);images=torch.from_numpy(np.array(x[start:end])).cuda()
            g,mass,order,counts,k80=encoder.global_features(images);arrays['global'][start:end]=g.cpu().numpy()
            selected_mass={}
            for rule in RULES:
                ids=encoder.select(order,counts,rule,np.arange(start,end)+(1000000 if split=='validation' else 0))
                local=encoder.sparse_features(images,ids,counts,rule)
                arrays[rule][start:end]=local.cpu().numpy();arrays[rule+'_ids'][start:end]=ids.cpu().numpy()
                selected_mass[rule]=[float(mass[i,ids[i,:int(k)]].sum()) for i,k in enumerate(counts)]
            for i in range(end-start):traces.append({'row_index':start+i,'K':int(counts[i]),'K80_unclamped':int(k80[i]),
                'actual_sparse_sequence_length':int(counts[i])+1,'selected_mass':{rule:selected_mass[rule][i] for rule in RULES}})
            if start==0:write(RUN/task/'first_sparse_witness.json',encoder.first_witness)
            if start%256==0:print('extract',task,split,end,n,flush=True)
            guard()
        for a in arrays.values():a.flush()
        del arrays
        counts=np.array([v['K'] for v in traces],np.int64);np.save(folder/f'{split}_counts.npy',counts)
        write(folder/f'{split}_selection.json',traces)
        reports[split]={'images':n,'K_histogram':dict(Counter(map(int,counts))),'K_mean':float(counts.mean()),'K_min':int(counts.min()),'K_max':int(counts.max()),
            'capped_below80_count':sum(v['selected_mass']['attention']<.8-1e-6 for v in traces),
            'sequence_length_histogram':dict(Counter(int(k)+1 for k in counts)),
            'actual_global_tokens_per_image':257,'high_resolution_tokens_total_per_rule':int((counts+1).sum()),
            'dense448_tokens_equivalent_per_rule':1025*n,'attention_pair_entries_ratio_global_plus_sparse_vs_whole448':float((257**2+(counts+1)**2).sum()/(1025**2*n))}
    write(RUN/task/'extraction.json',{'status':'COMPLETE','splits':reports,'actual_block_input_histograms':encoder.histograms,
        'frozen_encoder':encoder.frozen_witness(),'wall_seconds':time.monotonic()-begun,**guard()})


def decode(task,spec,head,field,rows):
    if task=='FSOD':return core.reader.decode('FSOD',head,field,rows)
    if task=='UATD':return core.dense.decode32(head(field))
    logits=head(field)
    return F.interpolate(logits,(480,320),mode='bilinear',align_corners=False).argmax(1).byte().cpu().numpy()


@torch.no_grad()
def evaluate(task,spec,fusion,head,bank,labels,batch):
    fusion.eval();head.eval();pieces=[]
    for start in range(0,len(bank['global']),batch):
        end=min(start+batch,len(bank['global']));field=fusion(*(bank[k][start:end] for k in ('global','local','ids','counts')))
        value=decode(task,spec,head,field,spec['rows']['validation'][start:end])
        if task=='UATD':pieces.extend(value)
        else:pieces.append(value)
        guard()
    predictions=pieces if task=='UATD' else np.concatenate(pieces)
    metrics,independent=core.scores(task,spec,predictions,labels,'validation')
    return metrics,independent,predictions


def load_bank(task,rule,split):
    folder=RUN/task/'features'
    paths={'global':f'{split}_global.npy','local':f'{split}_{rule}.npy','ids':f'{split}_{rule}_ids.npy','counts':f'{split}_counts.npy'}
    return {k:torch.from_numpy(np.load(folder/v,allow_pickle=False)).cuda().to(torch.long if k in ('ids','counts') else torch.float32) for k,v in paths.items()}


def fit(task,rule,spec,protocol):
    schedule=protocol['schedules'][task];folder=RUN/task/rule;folder.mkdir()
    bank={s:load_bank(task,rule,s) for s in ('train','validation')};labels={s:core.targets_for(task,spec,s) for s in bank}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(101);fusion=Fusion().cuda()
    head=core.new_head(task,spec).cuda();hopt=core.reader.head_optimizer('FSOD' if task=='FSOD' else 'Marine',head)
    fopt=torch.optim.AdamW(fusion.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(0);torch.manual_seed(0)
    key=core.metric_key(task,spec);best=-float('inf');updates=0;begun=time.monotonic();limit=min(DEADLINE,time.time()+protocol['resource']['fit_seconds'][task])
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1,schedule['epochs']+1):
            fusion.train();head.train();order=torch.randperm(len(bank['train']['global']),generator=generator);total=0.
            if task=='FSOD':hopt.param_groups[0]['lr']=.005*.1**sum(epoch>m for m in (40,55))
            for indices in order.split(schedule['batch']):
                if time.time()>limit:raise TimeoutError('Fixed fusion/head fit deadline')
                ids=indices.cuda();hopt.zero_grad(set_to_none=True);fopt.zero_grad(set_to_none=True)
                fopt.param_groups[0]['lr']=core.previous.shared.native.module_lr(updates,schedule['updates'],.001)
                field=fusion(*(bank['train'][k][ids] for k in ('global','local','ids','counts')))
                target=core.batch_targets(task,spec,labels['train'],indices.tolist())
                loss=core.task_loss(task,spec,head,field,target);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(fusion.parameters(),1.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(),float('inf'),error_if_nonfinite=True)
                hopt.step();fopt.step();updates+=1;total+=float(loss.detach())*len(ids)
                if updates==1:write(folder/'first_counted_update.json',{'first_indices':indices.tolist(),'field_shape':list(field.shape),'fusion_gradient_norm':float(norm),'encoder_optimizer_parameters':0})
                guard()
            row={'epoch':epoch,'updates':updates,'loss':total/len(order),'wall_seconds':time.monotonic()-begun}
            if epoch%protocol['eval_every'][task]==0:
                metric,independent,pred=evaluate(task,spec,fusion,head,bank['validation'],labels['validation'],schedule['batch']);row[key]=metric[key]
                if metric[key]>best:
                    best=metric[key];best_epoch=epoch;best_metric=metric
                    torch.save({'prediction':pred,'epoch':epoch,'task':task,'rule':rule},folder/'best_predictions.pt')
                if epoch==schedule['epochs']:
                    terminal=metric;terminal_independent=independent
                    torch.save({'prediction':pred,'epoch':epoch,'task':task,'rule':rule},folder/'terminal_predictions.pt')
            log.write(json.dumps(row)+'\n');log.flush();print(task,rule,json.dumps(row),flush=True)
    result={'status':'COMPLETE_FROZEN_BACKBONE_FIT','task':task,'selection_rule':rule,'seed':0,'metric':key,'terminal':terminal,'best':best_metric,
        'best_epoch':best_epoch,'terminal_independent':terminal_independent,'updates':updates,'epochs':schedule['epochs'],'fusion_parameters':sum(v.numel() for v in fusion.parameters()),
        'head_parameters':sum(v.numel() for v in head.parameters()),'wall_seconds':time.monotonic()-begun,'trained_weight_files':0,**guard()}
    write(folder/'results.json',result)
    del bank,labels,hopt,fopt;gc.collect();torch.cuda.empty_cache()
    return fusion.eval().cpu(),head.eval().cpu()


def preprocess(row,norm):
    with Image.open(row['image']) as im:intensity=torch.from_numpy(np.array(im.convert('L'),dtype=np.uint8).astype(np.float32)/255.)[None]
    return prepare_images([{'intensity':intensity,'valid':torch.ones(intensity.shape[-2:],dtype=torch.bool)}],norm,None).cuda()


@torch.no_grad()
def benchmark(task,spec,encoder,models,protocol):
    folder=RUN/task;records=[];gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    load=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    before=encoder.frozen_witness()
    for rule in (*RULES,'whole448_feature_only'):
        if rule in RULES:
            fusion,head=models[rule];fusion.cuda();head.cuda()
        torch.cuda.reset_peak_memory_stats()
        indices=list(range(32));timings=[];features=[]
        for number in range(50+96):
            index=indices[(number-50)%32 if number>=50 else number%32];row=spec['rows']['validation'][index];torch.cuda.synchronize();begun=time.perf_counter()
            x=preprocess(row,spec['normalization'])
            if rule=='whole448_feature_only':
                encoder.branch='whole448_cost_only';large=F.interpolate(x,(448,448),mode='bilinear',align_corners=False,antialias=True)
                field=encoder.encoder(large)['patch_tokens']
                torch.cuda.synchronize();feature_elapsed=time.perf_counter()-begun
            else:
                g,mass,order,counts,k80=encoder.global_features(x);ids=encoder.select(order,counts,rule,[1000000+index]);local=encoder.sparse_features(x,ids,counts,rule)
                torch.cuda.synchronize();feature_elapsed=time.perf_counter()-begun
                fusion,head=models[rule];field=fusion(g,local,ids,counts);prediction=decode(task,spec,head,field,[row])
                torch.cuda.synchronize()
            elapsed=time.perf_counter()-begun
            if number>=50:
                timings.append(elapsed);features.append(feature_elapsed)
                records.append({'rule':rule,'round':(number-50)//32,'validation_index':index,'feature_including_preprocess_seconds':feature_elapsed,
                    'E2E_seconds':elapsed if rule!='whole448_feature_only' else None,'K':int(counts[0]) if rule!='whole448_feature_only' else 1024})
            guard()
        summary={'batch':1,'warmup':50,'measured':96,'feature_including_preprocess_median':float(np.median(features)),
            'feature_including_preprocess_p95':float(np.percentile(features,95)),
            'E2E_median':float(np.median(timings)) if rule!='whole448_feature_only' else None,
            'E2E_p95':float(np.percentile(timings,95)) if rule!='whole448_feature_only' else None,
            'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved(),
            'resident_model_scope':'Only the measured fusion/head plus shared encoder are onGPU; whole448feature-only has encoder only.',
            'scope':'Raw image decode/log1p/tasknorm/AA224/transfer/globalattention/select/sparse/fusion/head/nativepostprocessing. Feature-boundary sync included; no training on this physicalGPU. Whole448 is separately measured feature-only, not a new accuracy reference.',
            'GPU_load_before':load,'idle_claim':False,'frozen_before':before,'frozen_after':encoder.frozen_witness()}
        write(folder/(rule+'_cost.json'),summary)
        if rule in RULES:fusion.cpu();head.cpu()
        gc.collect();torch.cuda.empty_cache()
    write(folder/'cost_records.json',records)


def worker(task):
    global DEADLINE
    p=json.loads((RUN/'protocol.json').read_text());job=json.loads((RUN/task/'job.json').read_text());DEADLINE=job['deadline_unix']
    core.original.runtime();random.seed(0);np.random.seed(0);torch.manual_seed(0)
    try:
        spec=json.loads(Path(p['data_manifest']).read_text())['tasks'][task];encoder=SparseEncoder(p['encoder'])
        extract(task,spec,encoder,min(DEADLINE,time.time()+1800))
        models={}
        for rule in RULES:models[rule]=fit(task,rule,spec,p)
        benchmark(task,spec,encoder,models,p)
        witness=encoder.frozen_witness()
        if witness['requires_grad_any'] or witness['gradient_any'] or not witness['parameter_versions_unchanged']:raise RuntimeError('Encoder freeze violated')
        write(RUN/task/'completion.json',{'status':'COMPLETE_THREE_RULES_AND_COST','task':task,'frozen_encoder':witness,'trained_weight_files':0})
    except BaseException:
        write(RUN/task/'failure.json',{'traceback':traceback.format_exc(),'time_unix':time.time()});raise


def pipeline():
    started=time.time();parent=core.BASE/'gsr_joint_pair_v1';old=json.loads((parent/'protocol.json').read_text())
    admissions={str(g):gpu_admission(g,set()) for g in (2,3)}
    if not all(v['eligible'] for v in admissions.values()):raise RuntimeError(('GPU admission unavailable',admissions))
    RUN.mkdir();deadline=started+8*3600
    p={'status':'FROZEN_BEFORE_ANY_MODEL_FORWARD','entry':str(Path(__file__).resolve()),'started_unix':started,'deadline_unix':deadline,
        'encoder':old['encoder'],'data_manifest':old['data_manifest'],'tasks':TASKS,'rules':RULES,'seed':0,'threshold_mass':.8,'K_min':64,'K_max':256,
        'selection':'Last block actual same-global-pass CLS Q and patchK, softmax across CLS+patches then headmean patch map. Bilinear16to32 alignFalse, patchmass renormalized. Stable descending/rasterties; smallest>=.8 then clamp64..256. Record capshortfall.',
        'controls':'Same image exactK for allrules. Fixedcenter Chebyshev distance+rastertie; random permutation seed17000+rowindex(+1000000validation), fixed perimage. No GT or devscore selection.',
        'sparse_path':'Only selected raw14x14 patches projected; original official448PE gathered by32grid rasterID+CLSPE, originalCLS then12blocks. EqualK grouped, no tokenpadding/masking or full448encoding beforeselection. Padding exists only in cache/fusionstorage.',
        'fusion':'Shared pertoken1154to64GELUto384 gatedresidual, inputs[L-B,B,globalmean,xy],zerooutputinit;scatter-add selectedonly. Independenttask/rule fits, not one sharednewpretrainedencoder.',
        'schedules':{t:old['target_schedules'][t] for t in TASKS},'eval_every':{'FSOD':5,'Marine':1,'UATD':5},
        'training':'FrozenDINO throughout; no encoderoptimizer. Originaltaskhead/loss/FP32/noAMP/TF32off/detFalse. Headseed0/privateorder0; fusionseed101/AdamW.001 wd.0001 originalwarmcosine,clip1. FSODSGD.005/.9/.0005 milestones40/55;otherheadAdamW.001/.0001.',
        'new_fits':9,'planned_updates':129000,'primary':'terminal; earlieststrictbest auxiliary, allrules retained regardlessdirection',
        'targeted_checks':'Firstnormalbatch actual12block lengths, exactselectedrawpatch geometry and official448PEgather; originalGT canonical/independentscorers reused. No extra smoke/SHA/review.',
        'cost':'After all3fits pertask: nootherownworker on sameGPU. Batch1 first32originaldevrows fixed beforeforward,50warmup+3rounds32, synchronized raw timings,median/p95,peakallocated/reserved. Whole448featureonly separately. No GTscore duringcost; originalpublisheddata notcold.',
        'resource':{'GPUs':[2,3],'max_workers_per_GPU':1,'task_seconds':{'FSOD':14400,'UATD':14400,'Marine':9000},'fit_seconds':{'FSOD':3600,'UATD':3600,'Marine':1800},'extract_seconds':1800,'global_seconds':28800,'GPU_GiB':32,'RSS_GiB':48,'CPU_threads':4,'new_disk_GiB':24},
        'references':{t:old['references'][t] for t in TASKS},'GSR_K2_repeats':str(core.BASE/'gsr_joint_pair_finetune_repeats_v1/results.json'),
        'whole448_reference':str(core.BASE/'whole448_frozen_reference_v1/results.json'),'strong_reference_pool':str(core.BASE/'swise_supplement_v1/C_specialists/results.json'),
        'reference_scope':'Inheritaccuracy only; fullFT GSRK2/strongspecialists differ in weights/head/training/input/pretraining/search. Report terminal andbestobserved separately, not worldSOTA. UATD whole448 priorpool16 differs from current32head.',
        'diagram':str(core.BASE/'gsr_method_diagrams_v1/21_k_attention_sparse.png'),'initial_admission':admissions,
        'weight_storage':'No trainedweightfiles; keep heads/fusion only RAM until onlinecost ends. Features+predictions retained ondata disk.',
        'failure_rule':'Visibleerror orresource/deadline stop entireunit, preservefiles, no automaticretry/batch/precision/threshold/capchanges.'}
    pool=json.loads((core.BASE/'swise_supplement_v1/C_specialists/results.json').read_text())['models']
    strong={}
    for task,name,key in (('FSOD','rtdetr_r18vd','AP50'),('Marine','deeplabv3plus_resnet50','miou12')):
        model=pool[name]; candidates=model['selected_three_seeds']+model['seed0_both_configs']
        top=max(candidates,key=lambda v:v['best'][key])
        strong[task]={'name':name,'terminal_selected_seed_mean':model['terminal_mean'],'best_selected_seed_mean':model['best_auxiliary_mean'],
            'best_observed_score':top['best'][key],'best_observed_seed':top['seed'],'best_epoch':top['best_epoch'],
            'scope':'Best observed development endpoint among completed declared configurations/seeds, not matched protocol or worldSOTA; different pretrained backbone/input/head/fullbodytraining/search.'}
    up=core.BASE/'gsr_selector_context_recovery_v1/dual/UATD/results.json';u=json.loads(up.read_text())
    strong['UATD']={'name':'completed dual-context frozen-DINO global stage','terminal':u['global_stage']['scheduled_terminal_metric'],
        'best_observed_score':u['global_stage']['scheduled_best'],'best_epoch':u['global_stage']['best_epoch'],
        'same_terminal_random_policy_reference':u['local_stage']['metrics']['random_budgetmatched']['map50'],
        'evidence':str(up),'scope':'Different added-module/head/EMA/trainingrecipe; developmentbest auxiliary, not worldSOTA.'}
    p['strongest_completed_reference_best_scores']=strong
    p['benchmark_validation_row_indices']=list(range(32))
    write(RUN/'protocol.json',p)
    pending=list(TASKS);active={};waits=[];launches={};cpus=sorted(os.sched_getaffinity(0))
    try:
        while pending or active:
            if time.time()>deadline:raise TimeoutError('Global deadline')
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>24*1024**3:raise RuntimeError('24GiB disk cap')
            for task,(child,log,gpu,begin,limit) in list(active.items()):
                if time.time()>limit and child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                if child.poll() is None:continue
                code=child.wait();log.close();del active[task];waits.append({'task':task,'PID':child.pid,'GPU':gpu,'actual_wait_exit':code,'wall_seconds':time.time()-begin});write(RUN/'actual_waits.json',waits)
                if code!=0:raise RuntimeError(f'{task} failed with{code}')
            for gpu in (2,3):
                if not pending or any(v[2]==gpu for v in active.values()):continue
                observed=gpu_admission(gpu,{v[0].pid for v in active.values()})
                if not observed['eligible']:continue
                task=pending.pop(0);folder=RUN/task;folder.mkdir();limit=min(deadline,time.time()+p['resource']['task_seconds'][task]);write(folder/'job.json',{'GPU':gpu,'deadline_unix':limit,'admission':observed})
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log=(folder/'stdout.log').open('w');child=subprocess.Popen([core.original.PYTHON,'-B','-u','-m','runs.gsr_k_attention','worker',task],cwd=core.original.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                os.sched_setaffinity(child.pid,cpus[(gpu-2)*4:(gpu-1)*4]);active[task]=(child,log,gpu,time.time(),limit);launches[task]={'PID':child.pid,'GPU':gpu};write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':launches})
            write(RUN/'queue_state.json',{'status':'RUNNING','active':{t:{'PID':v[0].pid,'GPU':v[2]} for t,v in active.items()},'pending':pending,'completed':len(waits)})
            time.sleep(2)
        results={t:{rule:json.loads((RUN/t/rule/'results.json').read_text()) for rule in RULES} for t in TASKS}
        write(RUN/'results.json',{'status':'COMPLETE_FIXED_NINE_FITS','tasks':results,'scope':p['reference_scope']})
        write(RUN/'completion.json',{'status':'COMPLETE','actual_waits':waits,'wall_seconds':time.time()-started,'trained_weight_files':0,'updates':sum(v['updates'] for task in results.values() for v in task.values())})
        write(RUN/'queue_state.json',{'status':'COMPLETE','active':{},'pending':[],'completed':3})
    except BaseException:
        write(RUN/'failure.json',{'traceback':traceback.format_exc(),'time_unix':time.time()});write(RUN/'queue_state.json',{'status':'STOPPED_FAILURE','pending':pending});raise
    finally:
        for task,(child,log,gpu,begin,limit) in active.items():
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            code=child.wait();log.close();waits.append({'task':task,'PID':child.pid,'GPU':gpu,'actual_wait_exit':code,'wall_seconds':time.time()-begin,'cleanup':True})
        write(RUN/'actual_waits.json',waits)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('pipeline','worker'));parser.add_argument('task',nargs='?');args=parser.parse_args()
    if args.phase=='pipeline':pipeline()
    else:worker(args.task)
