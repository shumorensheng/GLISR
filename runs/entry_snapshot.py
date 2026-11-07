"""Seven finite independently trained shared Attention6 component ablations."""
import argparse
import json
import math
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
from runs import gsr_k1_complete_fullft as full
core=full.core
pair=full.pair
parent=full.parent
original=core.original
previous=core.previous
reader=core.reader
SOURCE=core.SOURCE
def new_head(task,spec):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(REPEAT)
        if spec['kind']=='detection':
            head=reader.new_head('FSOD',REPEAT)
            if spec['classes']!=10:
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                head.roi_heads.box_predictor=FastRCNNPredictor(1024,spec['classes']+1)
            return head
        return nn.Conv2d(384,spec['classes'],1)
batch_targets=core.batch_targets
targets_for=core.targets_for
task_loss=core.task_loss
RUN=core.BASE/'gsr_attention6_ablations_indexfix_resume_v1'
SOCKET=RUN/'source_state.sock'
ARM=''
REPEAT=0
VARIANTS=('attention256','fixed256','random256','no_gate','memory3','memory6','regrid_capacity')
from runs.gsr_k_attention import Fusion as SparseFusion
from sonar_encoder.training.model import build_encoder
DEADLINE=0.
CAPS={'FSOD':10800,'Marine':5400,'UATD':14400,'NKSID':3600,'HighResolution':3600,'Diver':3600}
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def evaluate(*args, **kwargs):
    model=args[2];saved=model.selection_generator.get_state()
    model.selection_generator.manual_seed(17000)
    result=core.evaluate(*args, **kwargs)
    model.selection_generator.set_state(saved)
    return result
POLICIES=('global','fixed0','random')


def guard(deadline):
    observed={'RSS_bytes':core.resource.getrusage(core.resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_reserved_bytes':torch.cuda.max_memory_reserved()}
    if time.time()>min(deadline,DEADLINE):raise TimeoutError('Attention6 fixed deadline')
    if observed['RSS_bytes']>64*1024**3 or observed['GPU_reserved_bytes']>40*1024**3:raise RuntimeError('Attention6 resource cap '+str(observed))
    return observed


class AblationFusion(SparseFusion):
    def forward(self,g,local,ids,counts):
        if ARM!='no_gate':return super().forward(g,local,ids,counts)
        base=F.interpolate(g,(32,32),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
        index=ids.clamp_min(0);selected=base.gather(1,index[:,:,None].expand(-1,-1,384))
        whole=g.mean((2,3))[:,None].expand(-1,256,-1)
        xy=torch.stack(((index%32+.5)/16-1,(index//32+.5)/16-1),-1)
        u=F.gelu(self.input(torch.cat((local-selected,selected,whole,xy),-1)))
        delta=self.output(u)*(torch.arange(256,device=g.device)[None]<counts[:,None])[:,:,None]
        return base.scatter_add(1,index[:,:,None].expand(-1,-1,384),delta).transpose(1,2).reshape(-1,384,32,32)


class GlobalMemoryResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(384)
        self.memory_norm = nn.LayerNorm(384)
        self.query = nn.Linear(384, 64)
        self.key = nn.Linear(384, 64)
        self.value = nn.Linear(384, 64)
        self.position = nn.Linear(2, 64, bias=False)
        self.output = nn.Linear(64, 384)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens, global_field, local_xy, global_xy):
        n = len(tokens)
        patches = tokens[:, 1:]
        memory = F.adaptive_avg_pool2d(global_field, (4, 4)).flatten(2).transpose(1, 2)
        memory = self.memory_norm(memory)
        xy = F.adaptive_avg_pool2d(global_xy.T.reshape(1, 2, 16, 16), (4, 4)).flatten(2).transpose(1, 2)
        q = self.query(self.query_norm(patches)) + self.position(local_xy)
        k = self.key(memory) + self.position(xy)
        v = self.value(memory)
        q = q.reshape(n, patches.shape[1], 4, 16).transpose(1, 2)
        k = k.reshape(n, 16, 4, 16).transpose(1, 2)
        v = v.reshape(n, 16, 4, 16).transpose(1, 2)
        update = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        update = self.output(update.transpose(1, 2).reshape(n, patches.shape[1], 64))
        return torch.cat((tokens[:, :1], patches + update), dim=1)


class GlobalMemoryGSR(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder=build_encoder(Path(config['code_dir']),Path(config['weights_path']),'R010')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101+REPEAT);self.fusion=AblationFusion()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(106+REPEAT)
            self.local_memory=nn.ModuleList([GlobalMemoryResidual() for _ in range(2)])
        self.interaction_blocks=[3] if ARM=='memory3' else [6] if ARM=='memory6' else [3,6]
        if ARM=='memory3':self.local_memory=nn.ModuleList([self.local_memory[0]])
        if ARM=='memory6':self.local_memory=nn.ModuleList([self.local_memory[1]])
        if ARM=='no_gate':self.fusion.gate.requires_grad_(False)
        y,x=torch.meshgrid((torch.arange(16)+.5)/8-1,(torch.arange(16)+.5)/8-1,indexing='ij')
        self.register_buffer('global_xy',torch.stack((x,y),-1).reshape(256,2))
        ids=np.arange(1024);row,col=ids//32,ids%32
        self.register_buffer('fixed_ids',torch.tensor(np.lexsort((ids,np.maximum(abs(row-15.5),abs(col-15.5))))))
        self.encoder.backbone.blocks[-1].attn.qkv.register_forward_hook(self.capture)
        self.encoder_images=0;self.encoder_calls=0;self.K_histogram={};self.sequence_histogram={}
        self.memory_hook_calls=[0,0];self.global_hook_bypasses=[0,0]
        self.selection_generator=torch.Generator().manual_seed(17000)
        self.first_sparse_witness=None;self.mass_shortfalls=0

    def capture(self,module,args,output):
        self.global_qkv=output.detach()

    def adapter_parameters(self):
        return [p for p in self.local_memory.parameters() if p.requires_grad]

    def global_field(self,x):
        self.encoder_images+=len(x);self.encoder_calls+=1
        return self.encoder(x)['patch_tokens'].transpose(1,2).reshape(len(x),384,16,16)

    def field(self,images,head,kind,policy='attention',actions=None):
        g=self.global_field(images);bb=self.encoder.backbone
        if policy=='global':return F.interpolate(g,(32,32),mode='bilinear',align_corners=False)
        a=bb.blocks[-1].attn;qkv=self.global_qkv.reshape(len(images),257,3,a.num_heads,384//a.num_heads)
        q=qkv[:,0,0].unsqueeze(2);k=qkv[:,:,1].transpose(1,2)
        heat=((q*a.scale)@k.transpose(-1,-2)).softmax(-1)[:,:,:,1:].mean(1).reshape(len(images),1,16,16)
        mass=F.interpolate(heat,(32,32),mode='bilinear',align_corners=False).flatten(1)
        mass=mass/mass.sum(1,keepdim=True);order=mass.argsort(dim=1,descending=True,stable=True)
        k80=(mass.gather(1,order).cumsum(1)<.8).sum(1)+1;counts=k80.clamp(64,256)
        if ARM in ('attention256','fixed256','random256'):counts=torch.full_like(counts,256)
        if ARM=='fixed256':policy='fixed0'
        if ARM=='random256':policy='random'
        self.last_counts=counts.detach()
        chosen=torch.full((len(images),256),-1,device=images.device,dtype=torch.long)
        for i,n in enumerate(counts.tolist()):
            if policy=='random':selected=torch.randperm(1024,generator=self.selection_generator)[:n].to(images.device)
            elif policy=='fixed0':selected=self.fixed_ids[:n]
            else:selected=order[i,:n]
            chosen[i,:n]=selected.sort().values
            self.K_histogram[str(n)]=self.K_histogram.get(str(n),0)+1
            self.mass_shortfalls+=int(k80[i]>256)
        self.last_selected_ids=chosen.detach();self.last_attention_mass=mass.detach()
        if ARM!='regrid_capacity':
            large=F.interpolate(images,(448,448),mode='bilinear',align_corners=False,antialias=True)
            patches_view=large.unfold(2,14,14).unfold(3,14,14).permute(0,2,3,1,4,5)
            pe=bb.interpolate_pos_encoding(torch.empty(1,1025,384,device=images.device),448,448)
        else:
            base=F.interpolate(g,(32,32),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
        local=torch.zeros(len(images),256,384,device=images.device)
        for n in counts.unique().tolist():
            group=torch.where(counts==n)[0];ids=chosen[group,:n]
            xy=torch.stack(((ids%32+.5)/16-1,(ids//32+.5)/16-1),-1)
            lengths=[]
            if ARM=='regrid_capacity':
                patches=base[group[:,None],ids]
                tokens=torch.cat((bb.cls_token.expand(len(group),-1,-1),patches),1)
                for mi,module in enumerate(self.local_memory):
                    tokens=module(tokens,g[group],xy,self.global_xy);self.memory_hook_calls[mi]+=1
            else:
                pixels=patches_view[group[:,None],ids//32,ids%32].reshape(-1,1,14,14)
                patches=bb.patch_embed.norm(bb.patch_embed.proj(pixels).reshape(len(group),n,384))
                position=pe[0,1:][ids]
                tokens=torch.cat(((bb.cls_token+pe[:,:1]).expand(len(group),-1,-1),patches+position),1)
                for index,block in enumerate(bb.blocks[:6]):
                    lengths.append(int(tokens.shape[1]));tokens=block(tokens)
                    if index+1 in self.interaction_blocks:
                        mi=self.interaction_blocks.index(index+1)
                        tokens=self.local_memory[mi](tokens,g[group],xy,self.global_xy);self.memory_hook_calls[mi]+=1
                self.encoder_images+=len(group);self.encoder_calls+=1
                self.sequence_histogram[str(n+1)]=self.sequence_histogram.get(str(n+1),0)+len(group)
            local[group,:n]=bb.norm(tokens)[:,1:]
            if self.first_sparse_witness is None:
                if ARM!='regrid_capacity':
                    expected=torch.stack([large[int(group[0]),:,int(j)//32*14:int(j)//32*14+14,int(j)%32*14:int(j)%32*14+14] for j in ids[0]])
                    if not torch.equal(expected,pixels[:n]) or lengths!=[n+1]*6:raise RuntimeError('Sparse pixel/sequence mismatch')
                    if not torch.equal(position[0],pe[0,ids[0]+1]):raise RuntimeError('Original448 PE mismatch')
                self.first_sparse_witness={'variant':ARM,'K':n,'global_blocks':12,'local_blocks':0 if ARM=='regrid_capacity' else 6,'actual_local_sequence_lengths':lengths,'selected_ids':ids[0].tolist(),'original448_PE_exact':None if ARM=='regrid_capacity' else True,'interaction_blocks':self.interaction_blocks,'local_input':'global regrid only' if ARM=='regrid_capacity' else 'selected448 pixels','fusion_trainable_parameters':sum(p.numel() for p in self.fusion.parameters() if p.requires_grad),'interaction_trainable_parameters':sum(p.numel() for p in self.local_memory.parameters() if p.requires_grad)}
        return self.fusion(g,local,chosen,counts)


def source_training(model, p, tasks):
    deadline = min(p['deadline_unix'], time.time()+21600)
    model.encoder.eval().requires_grad_(False)
    images = {t: torch.from_numpy(np.load(tasks[t]['views']['train'], mmap_mode='r', allow_pickle=False)).cuda() for t in SOURCE}
    labels = {t: targets_for(t, tasks[t], 'train') for t in SOURCE}
    heads = nn.ModuleDict({t: new_head(t, tasks[t]).cuda() for t in SOURCE})
    optimizers = {t: reader.head_optimizer('FSOD' if tasks[t]['kind'] == 'detection' else 'Marine', heads[t]) for t in SOURCE}
    optimizer = torch.optim.AdamW([{'params':list(model.fusion.parameters()),'lr':.001},{'params':model.adapter_parameters(),'lr':.0001}],weight_decay=.0001)
    plans = np.load(RUN/'source_plan.npz', allow_pickle=False)
    begun = time.monotonic()
    with (RUN/'source_training.jsonl').open('w') as log:
        for update in range(6000):
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = previous.shared.native.module_lr(update, 6000, .001)
            optimizer.param_groups[1]['lr'] = previous.shared.native.module_lr(update, 6000, .0001)
            values = {}
            for task in SOURCE:
                head = heads[task]; head.train(); opt = optimizers[task]; opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = (.005 if update < 4000 else .0005 if update < 5500 else .00005) if tasks[task]['kind'] == 'detection' else previous.shared.native.module_lr(update, 6000, .001)
                ids = plans[task][update]; value_sum = 0.
                for start in range(0, len(ids), 4):
                    indices = ids[start:start+4].tolist()
                    field = model.field(images[task][indices], head, tasks[task]['kind'])
                    target = batch_targets(task, tasks[task], labels[task], indices)
                    value = task_loss(task, tasks[task], head, field, target)*len(indices)/len(ids)/len(SOURCE)
                    value.backward(); value_sum += float(value.detach())
                values[task] = value_sum
                opt.step()
            adapter_norm = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), 1., error_if_nonfinite=True)
            norm = torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); guard(deadline)
            if update == 0:
                write(RUN/'first_source_update.json', {'updates': 1, 'source_tasks': list(SOURCE),
                    'original_DINO_trainable': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad), 'adapter_gradient_norm':float(adapter_norm), 'adapter_parameters':sum(p.numel() for p in model.adapter_parameters()), 'shared_fusion_gradient_norm': float(norm),
                    'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
                    'source_macro': p['source_macro_per_task'], 'field': list(field.shape), 'local_memory_calls_ordered_modules':model.memory_hook_calls, 'sparse_witness':model.first_sparse_witness})
            if (update+1) % 100 == 0:
                row = {'updates': update+1, 'losses': values, 'wall_seconds': time.monotonic()-begun,
                       'encoder_images': model.encoder_images, **guard(deadline)}
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
    write(RUN/'source_report.json', {'status': 'COMPLETE_EXPANDED_FROZEN_BACKBONE_SOURCE6000',
        'visits_per_task': p['source_visits'], 'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'wall_seconds': time.monotonic()-begun, 'original_DINO_updates': 0, **guard(deadline)})
    write(RUN/'source_sparse_statistics.json',{'K_histogram':model.K_histogram,'sequence_histogram':model.sequence_histogram,'capped_below80_count':model.mass_shortfalls,'first_witness':model.first_sparse_witness})
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fit(task, seed, job_id, protocol, initial):
    spec = json.loads(Path(protocol['data_manifest']).read_text())['tasks'][task]
    schedule = protocol['target_schedules'][task]; folder = RUN/job_id
    model = GlobalMemoryGSR(protocol['encoder'])
    model.load_state_dict(initial, strict=True)
    policy = 'random' if ARM=='random256' else 'fixed0' if ARM=='fixed256' else 'attention'
    core.reader.configure_encoder(model.encoder, 'e2e')
    model.fusion.requires_grad_(True)
    if ARM=='no_gate':model.fusion.gate.requires_grad_(False)
    model.cuda()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if task in ('FSOD', 'Marine', 'UATD', 'NKSID'):
            head = core.reader.new_head(task, seed)
        else:
            head = nn.Conv2d(384, 1 if spec['kind'] == 'point_regression' else spec['classes'], 1)
    head = head.cuda()
    body_opt = core.reader.body_optimizer(model.encoder)
    head_opt = core.reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    fusion_opt = torch.optim.AdamW(list(model.fusion.parameters())+model.adapter_parameters(), lr=.0001, weight_decay=.0001)
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r', allow_pickle=False)).cuda() for s in ('train', 'validation')}
    labels = {s: core.targets_for(task, spec, s) for s in images}
    generator = torch.Generator().manual_seed(seed)
    key = core.metric_key(task, spec); direction = -1 if spec['kind'] == 'point_regression' else 1
    best = -float('inf'); updates = 0; begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs']+1):
            model.train(); head.train(); order = torch.randperm(len(spec['rows']['train']), generator=generator)
            if spec['kind'] == 'detection':
                head_opt.param_groups[0]['lr'] = .005*.1**sum(epoch > m for m in (40, 55))
            total = 0.; counts = np.zeros(257, dtype=np.int64)
            for ids in order.split(schedule['batch']):
                guard(DEADLINE)
                for optimizer in (body_opt, head_opt, fusion_opt): optimizer.zero_grad(set_to_none=True)
                for group in body_opt.param_groups:
                    group['lr'] = core.reader.body_learning_rate(updates,schedule['updates'])
                for optimizer in (fusion_opt,):
                    optimizer.param_groups[0]['lr'] = core.previous.shared.native.module_lr(updates,schedule['updates'],.0001)
                field = model.field(images['train'][ids.cuda()], head, 'uatd' if task == 'UATD' else spec['kind'], policy)
                target = core.batch_targets(task, spec, labels['train'], ids.tolist())
                loss = core.task_loss(task, spec, head, field, target); loss.backward()
                body_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(list(model.fusion.parameters())+model.adapter_parameters(), 1., error_if_nonfinite=True)
                for optimizer in (body_opt, head_opt, fusion_opt): optimizer.step()
                updates += 1; total += float(loss.detach())*len(ids)
                counts += torch.bincount(model.last_counts, minlength=257).cpu().numpy()
                if updates == 1:
                    write(folder/'first_counted_update.json', {'updates': 1, 'first_ids': ids.tolist(),
                        'actual_K': model.last_counts.tolist(), 'field': list(field.shape),
                        'body_gradient_norm': float(body_norm),
                        'encoder_images': model.encoder_images, 'source_cohort': protocol['source_cohorts'][ARM],
                        'local_memory_calls_ordered_modules':model.memory_hook_calls, 'sparse_witness':model.first_sparse_witness, 'ordinary_task_loss_only': True, 'backward_scope': 'Full task gradients through global/local encoder and trained modules; hard attention selection detached, no learned selector or ST'})
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order),
                'K_histogram': counts.tolist(), 'wall_seconds': time.monotonic()-begun}
            every = protocol['evaluation_every_epochs'][task]
            if epoch % every == 0:
                metric, independent, pred = evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], DEADLINE, policy)
                entry[key] = metric[key]
                if 'stats' in metric:entry['COCO_AP_AP50_AP75']=metric['stats'][:3]
                payload = {'prediction': pred, 'task': task, 'epoch': epoch, 'policy': policy, 'local_budget': 'K256' if ARM in ('attention256','fixed256','random256') else 'adaptive K64-256' }
                if direction*metric[key] > best:
                    best, best_epoch, best_metric = direction*metric[key], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                    weight_payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'head': {k: v.detach().cpu() for k, v in head.state_dict().items()}, 'task': task, 'seed': seed, 'epoch': epoch, 'endpoint': 'best', 'selection_split': 'validation', 'arm': ARM, 'source_cohort': protocol['source_cohorts'][ARM], 'model_entry_snapshot': str(__file__), 'protocol_path': str(RUN / 'protocol.json'), 'data_manifest': protocol['data_manifest']}
                    torch.save(weight_payload, folder / 'best_weights.tmp')
                    (folder / 'best_weights.tmp').replace(folder / 'best_weights.pt')
                    del weight_payload
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = metric, independent
                    torch.save(payload, folder/'terminal_predictions.pt')
                    weight_payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'head': {k: v.detach().cpu() for k, v in head.state_dict().items()}, 'task': task, 'seed': seed, 'epoch': epoch, 'endpoint': 'terminal', 'selection_split': 'validation', 'arm': ARM, 'source_cohort': protocol['source_cohorts'][ARM], 'model_entry_snapshot': str(__file__), 'protocol_path': str(RUN / 'protocol.json'), 'data_manifest': protocol['data_manifest']}
                    torch.save(weight_payload, folder / 'terminal_weights.tmp')
                    (folder / 'terminal_weights.tmp').replace(folder / 'terminal_weights.pt')
                    del weight_payload
            log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)
    policies = {policy: terminal}
    held=None;held_split=None
    write(folder/'results.json', {'status': 'COMPLETE_FIXED_INDEPENDENT_ABLATION_TARGET', 'task': task, 'metric': key,
        'higher_is_better': direction == 1, 'epochs': schedule['epochs'], 'updates': updates,
        'terminal': terminal, 'terminal_independent': terminal_independent, 'best': best_metric, 'best_epoch': best_epoch,
        'terminal_policy_metrics': policies, 'heldout_terminal_only': held, 'heldout_split': held_split,
        'source_cohort': protocol['source_cohorts'][ARM], 'arm': ARM, 'seed': seed, 'local_budget': 'K256' if ARM in ('attention256','fixed256','random256') else 'adaptive K64-256' , 'sparse_depth':0 if ARM=='regrid_capacity' else 6,'interaction_blocks':model.interaction_blocks,'K_histogram':model.K_histogram,'sequence_histogram':model.sequence_histogram,'capped_below80_count':model.mass_shortfalls,
        'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'new_weight_files': 2, 'wall_seconds': time.monotonic()-begun, **guard(DEADLINE)})


def worker(job_id):
    global RUN,DEADLINE,ARM,REPEAT
    root=RUN
    reservation=root/'parallel_dispatch'/(job_id+'_reservation.json')
    if reservation.is_file():
        actual=Path(json.loads(reservation.read_text())['actual_wait_path'])
        deadline=json.loads((root/'protocol.json').read_text())['deadline_unix']
        while not actual.is_file():
            if time.time()>deadline:raise TimeoutError('Parallel join globaldeadline')
            time.sleep(2)
        result=json.loads(actual.read_text())
        if result['actual_wait_exit']!=0:raise RuntimeError('Actual parallel model failed: '+str(result))
        write(root/job_id/'controller_join_receipt.json',{'controller_join_PID':os.getpid(),'actual_worker':result,'new_model_updates_in_join':0,'CUDA_initialized_in_join':torch.cuda.is_initialized()})
        return
    root=RUN;p=json.loads((root/'protocol.json').read_text());job=json.loads((root/job_id/'job.json').read_text())
    ARM=job['arm'];REPEAT=job['seed'];DEADLINE=job['deadline_unix'];core.guard=guard;original.runtime()
    seed=job['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    key=str(VARIANTS.index(ARM)).encode()
    try:
        if job['task'] is None:
            RUN=root/job_id;core.RUN=RUN
            model=GlobalMemoryGSR(p['encoder']).cuda();versions=[x._version for x in model.encoder.parameters()]
            tasks=json.loads(Path(p['data_manifest']).read_text())['tasks'];state=source_training(model,p,tasks)
            witness={'original_DINO_trainable':sum(x.numel() for x in model.encoder.parameters() if x.requires_grad),'original_DINO_has_grad':any(x.grad is not None for x in model.encoder.parameters()),'original_parameter_versions_unchanged':versions==[x._version for x in model.encoder.parameters()],'retained_modules':['DINO',ARM,'token_fusion'],'variant':ARM,'source_heads_exported':False,'memory_output_norms':[float(m.output.weight.norm()) for m in model.local_memory]}
            if witness['original_DINO_trainable'] or witness['original_DINO_has_grad'] or not witness['original_parameter_versions_unchanged']:raise RuntimeError('Source backbone freeze violated')
            write(RUN/'frozen_encoder_witness.json',witness);blob=pickle.dumps(state,protocol=5)
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as c:
                c.settimeout(120);c.connect(str(SOCKET));c.sendall(b'P'+key+len(blob).to_bytes(8,'big')+blob)
                if parent.read_bytes(c,1)!=b'1':raise RuntimeError('RAM publication not acknowledged')
            write(RUN/'completion.json',{'status':'COMPLETE_SHARED_SOURCE6000','arm':ARM,'source_updates':6000,'serialized_bytes':len(blob),'weights_saved':0,**guard(DEADLINE)})
        else:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as c:
                c.settimeout(120);c.connect(str(SOCKET));c.sendall(b'G'+key)
                size=int.from_bytes(parent.read_bytes(c,8),'big');initial=pickle.loads(parent.read_bytes(c,size))
            fit(job['task'],seed,job_id,p,initial)
    except BaseException:
        write(root/job_id/'failure.json',{'traceback':traceback.format_exc(),'time_unix':time.time(),'no_retry':True});raise


def pipeline():
    started=time.time();failed=json.loads((core.BASE/'gsr_attention6_ablations_v1/protocol.json').read_text());deadline=failed['deadline_unix'];RUN.mkdir()
    old=json.loads((core.BASE/'gsr_attention6_memory_v1/protocol.json').read_text())
    tasks=('FSOD','Marine','UATD','NKSID','HighResolution','Diver');arms=VARIANTS
    jobs=[{'id':a+'_source','arm':a,'task':None,'seed':0} for i,a in enumerate(arms,0)]
    jobs += [{'id':a+'_'+t,'arm':a,'task':t,'seed':0} for t in tasks for i,a in enumerate(arms,0)]
    data=json.loads(Path(old['data_manifest']).read_text())['tasks']
    p=dict(old)
    p['target_schedules']={t:dict(old['target_schedules'][t]) for t in tasks}
    p.update({'inherited_failed_run':str(core.BASE/'gsr_attention6_ablations_indexfix_v1'),'inherited_original_deadline':deadline,'repair_scope':'Resume after parallel dispatcher CPU affinity EINVAL and cleanup SIGKILL. Preserve original indexfix_v1 artifacts; regenerate RAM-only source cohorts in isolated run. Same architecture/splits/schedules and original deadline. Affinity chooses indices within actual allowed CPU set.','diagram':str(core.BASE/'gsr_method_diagrams_v1/40_attention6_ablation_matrix.png'),'interaction':'Variant-specific active memory blocks in variant_definitions; all other main structure retained unless explicit capacity control.','variant_definitions':{'attention256':'Top256 CLS attention positions','fixed256':'Center16x16 block256 positions','random256':'Uniform256 unique positions','no_gate':'Adaptive main, gate removed factor1; gate weights frozen unused and excluded from trainablecount','memory3':'Adaptive main interaction only afterblock3','memory6':'Adaptive main interaction only afterblock6','regrid_capacity':'Adaptive attention IDs on global regrid; same memory/fusion trainablecount; no second DINOforward'},'entry':str(Path(__file__).resolve()),'started_unix':started,'deadline_unix':deadline,'jobs':jobs,'arms':arms,'target_seeds':[0],'source_cohorts':{a:RUN.name+'/'+a for a in arms},'weights_saved':0,'latency_authorized':False,'source_repeat':'Same seed0 source plan/head initialization, retained module initializations matched to full main; one complete initializer pervariant serves six targets.','source_updates_per_arm':6000,'target_updates_total':341600,'new_scope':'Seven seed0 source6000 variants, each same six independent fullFT tasks original60/100epochs. K256 selection trio is exact token-count comparison; remaining adaptive variants compare original adaptive main with actualK reported. No new Test/policy-only evaluations/latency/threshold sweep.','resources':{**old['resources'],'workers_per_GPU':2,'target_seconds':CAPS,'required_free_MiB':40960,'new_disk_GiB':48,'global_seconds':172800,'cost_window':'No new timing in this unit'},'comparison':'Seven variant definitions fixed before forward. Reuse existing main/no-interaction seed0 references, no exact source identity claim. Report every sign/seed/endpoint/class; no winner-driven extra configurations. Random uniform withoutreplacement each training access with privategenerator; evaluation reset17000 and restored. Regrid capacity keeps same learned memory+fusion count but no localimage encoding; not compute-matched or one-factor causal proof.'})
    write(RUN/'protocol.json',p)
    for j in jobs:
        folder=RUN/j['id'];folder.mkdir()
        if j['task'] is None:
            plans={}
            for i,t in enumerate(SOURCE):
                n=len(data[t]['rows']['train']);macro=p['source_macro_per_task'][t];rng=np.random.RandomState(13+i+1000*j['seed'])
                plans[t]=np.concatenate([rng.permutation(n) for _ in range(math.ceil(6000*macro/n))])[:6000*macro].reshape(6000,macro)
            np.savez(folder/'source_plan.npz',**plans)
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(SOCKET));os.chmod(SOCKET,0o600);listener.listen(8)
    blobs={};done=set();pending=list(jobs);active={};waits=[];launches={};cpus=sorted(os.sched_getaffinity(0))
    write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':{},'status':'WAITING_RESOURCE'})
    try:
        while pending or active:
            if time.time()>deadline:raise TimeoutError('Fixed48h globalcap')
            total=0
            import stat
            for f in RUN.rglob('*'):
                try:st=f.lstat()
                except FileNotFoundError:continue
                if stat.S_ISREG(st.st_mode):total+=st.st_size
            if total>48*1024**3:raise RuntimeError('Fixed48GiBnewdiskcap')
            ready,_,_=select.select([listener],[],[],.2)
            if ready:
                c,_=listener.accept()
                with c:
                    c.settimeout(120);command=parent.read_bytes(c,1);key=parent.read_bytes(c,1)
                    if command==b'P':
                        size=int.from_bytes(parent.read_bytes(c,8),'big');blobs[key]=parent.read_bytes(c,size);c.sendall(b'1')
                        write(RUN/(VARIANTS[int(key.decode())]+'_source_RAM_receipt.json'),{'serialized_bytes':size,'same_blob_for_all_targets':True,'weight_files':0})
                    elif command==b'G':c.sendall(len(blobs[key]).to_bytes(8,'big')+blobs[key])
                    else:raise ValueError('Unknown RAM command')
            for jid,(child,log,gpu,began,limit,arm,task) in list(active.items()):
                if time.time()>limit and child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                if child.poll() is None:continue
                code=child.wait();log.close();del active[jid];waits.append({'job':jid,'PID':child.pid,'GPU':gpu,'actual_wait_exit':code,'wall_seconds':time.time()-began});write(RUN/'actual_waits.json',waits)
                if code:raise RuntimeError(f'{jid} exited{code}; no retry')
                if task is None:done.add(arm)
            for gpu in (2,3):
                own={j:v for j,v in active.items() if v[2]==gpu}
                if len(own)>=2:continue
                eligible=[j for j in pending if j['task'] is None or j['arm'] in done]
                if not eligible:continue
                free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
                available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                if free<40960 or available<128*1024**3:continue
                job=eligible[0];pending.remove(job);jid=job['id'];folder=RUN/jid;cap=(21480 if job['arm'] in VARIANTS[:4] else 21600) if job['task'] is None else CAPS[job['task']];limit=min(deadline,time.time()+cap)
                write(folder/'job.json',{**job,'GPU':gpu,'deadline_unix':limit,'admission_free_MiB':free,'utilization':util})
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log=(folder/'stdout.log').open('w');child=subprocess.Popen([original.PYTHON,'-B','-u','-m','runs.gsr_attention6_ablations_resume','worker',jid],cwd=original.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                os.sched_setaffinity(child.pid,{cpus[(4*len(launches)+i)%len(cpus)] for i in range(min(4,len(cpus)))});active[jid]=(child,log,gpu,time.time(),limit,job['arm'],job['task']);launches[jid]={'PID':child.pid,'GPU':gpu}
                write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':launches,'status':'RUNNING'})
            write(RUN/'queue_state.json',{'status':'RUNNING' if active else 'WAITING_RESOURCE','active':{j:{'PID':v[0].pid,'GPU':v[2]} for j,v in active.items()},'pending':[j['id'] for j in pending],'completed':len(waits),'source_done':sorted(done)})
            time.sleep(2)
        results={t:{a:json.loads((RUN/(a+'_'+t)/'results.json').read_text()) for a in arms} for t in tasks}
        write(RUN/'results.json',{'status':'COMPLETE_SEVEN_SHARED_ABLATIONS_42_TARGETS','tasks':results,'actual_waits':waits})
        write(RUN/'completion.json',{'status':'COMPLETE','actual_waits':waits,'wall_seconds':time.time()-started,'weights_saved':0})
        write(RUN/'queue_state.json',{'status':'COMPLETE','active':{},'pending':[],'completed':len(waits)})
    except BaseException:
        write(RUN/'failure.json',{'traceback':traceback.format_exc(),'time_unix':time.time(),'no_retry':True});raise
    finally:
        for jid,(child,log,gpu,began,limit,arm,task) in active.items():
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            code=child.wait();log.close();waits.append({'job':jid,'PID':child.pid,'actual_wait_exit':code,'cleanup':True})
        write(RUN/'actual_waits.json',waits);listener.close();SOCKET.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('pipeline','worker'));parser.add_argument('job',nargs='?');args=parser.parse_args()
    if args.phase=='pipeline':pipeline()
    else:worker(args.job)

