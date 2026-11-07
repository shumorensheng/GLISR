"""HSMLA equation-block adaptation, not a released HSMLA backbone reproduction."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from runs import featsharp_marine_matched as fm
from runs.hsmla_formula import HSMLABlock
BASE=fm.BASE;RUN=BASE/'hsmla_marine_matched_v1'

class Model(nn.Module):
    def __init__(self,config,seed):
        super().__init__();self.encoder=fm.ref.build_encoder(Path(config['code_dir']),Path(config['weights_path']),'R010')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101+seed);self.fusion=HSMLABlock()
        self.encoder_images=0;self.encoder_calls=0;self.gate_trace=[]
    def field(self,x,head=None,kind=None,policy='hsmla',actions=None):
        self.encoder_images+=len(x);self.encoder_calls+=1
        large=F.interpolate(x,(448,448),mode='bilinear',align_corners=False)
        field=self.encoder(large)['patch_tokens'].transpose(1,2).reshape(len(x),384,32,32)
        out=self.fusion(field)
        if not self.training:
            self.gate_trace.append({'batch':len(x),'selected_fraction':float(self.fusion.last_gate_fraction),'local_queries':self.fusion.last_local_query_count})
        return out

def prepare():
    import numpy as np
    RUN.mkdir(exist_ok=True);parent=json.loads((BASE/'featsharp_marine_matched_v1/protocol.json').read_text());p={k:parent[k] for k in ['encoder','data_manifest','source_tasks','source_macro_per_task','source_visits','target_schedules','evaluation_every_epochs']}
    p.update({'name':'HSMLA paper-equation block adaptation on DINOv2 dense features','seeds':[0,1,2],'new_source_updates':18000,'new_Marine_updates':18900,'architecture':{'placement':'one fullEq28block after originalDINOwhole448 postLN32x32 field','dim':384,'heads':6,'depth':1,'window':7,'routing_tile':4,'threshold':.15,'multiscale_kernels':[3,5,7],'FFN_ratio':4,'FFN_activation':'GELU','sandwich_DWConv':3,'epsilon':1e-6,'rho_initial':.3,'rho_parameterization':'learned sigmoid logit','budget_lambda':.01,'spatial_lambda':.005,'gate':'Algorithm1 Conv3x3 normalizedX -> avgpool4 -> sigmoid, not alternativehead-QKVEq12','local_softmax':'rawQKV; same-window local-linear uses hattedQKV','training':'softgate allqueries','inference':'hardthreshold with actual selected-query localcomputation','out_of_grid':'excluded from both localdenominators'},'scope':'User explicitly accepted paper-formula adaptation. Missing architecture parameters fixed here, not claimed author settings. OriginalDINO remains original backbone; this block is downstream of encoder, not a HSMLAB2 replacement. No official checkpoint or kernel-speedup claim.','paper':'https://arxiv.org/html/2608.07616v2','input':'original standardized224 observation bilinear448; no native highresolution or extraImageNetnormalization','training':'Same sixinterfaces/source6000 with originalDINO frozen; sourceblockAdamW.001; Marine100/6300 body2e-5/head.001/block.0001 original schedules pluspaper regularizers. All three seeds retained.','resources':{'global_seconds':172800,'source_seconds':43200,'target_seconds':14400,'score_seconds':600,'GPUs':[2,3],'GPU_reserved_GiB':40,'RSS_GiB':96,'new_disk_GiB':24,'free_MiB':50000},'created_unix':time.time()})
    fm.write(RUN/'protocol.json',p);tasks=json.loads(Path(p['data_manifest']).read_text())['tasks']
    for seed in range(3):
        folder=RUN/f's{seed}';folder.mkdir();plans={}
        for i,t in enumerate(p['source_tasks']):
            n=len(tasks[t]['rows']['train']);k=p['source_macro_per_task'][t];rng=np.random.RandomState(13+i+1000*seed);plans[t]=np.concatenate([rng.permutation(n) for _ in range(int(np.ceil(6000*k/n)))])[:6000*k].reshape(6000,k)
        np.savez(folder/'source_plan.npz',**plans)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','source','fit','score']);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--deadline',type=float,default=0);args=parser.parse_args()
    if args.phase=='prepare':prepare()
    else:
        p=json.loads((RUN/'protocol.json').read_text());folder=RUN/f's{args.seed}';fm.RUN=RUN;fm.DEADLINE=args.deadline
        oldloss=fm.core.task_loss;realmodel=[]
        original_guard=fm.guard
        def guard(deadline):
            observed=original_guard(deadline)
            if observed['GPU_reserved_bytes']>40*1024**3:raise RuntimeError('HSMLA40GiBreservedcap')
            return observed
        fm.guard=guard
        class TracedModel(Model):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);realmodel.append(self)
        fm.Model=TracedModel
        def loss(task,spec,head,field,target):
            return oldloss(task,spec,head,field,target)+realmodel[-1].fusion.regularization_loss
        fm.core.task_loss=loss
        try:
            getattr(fm,args.phase)(args.seed,p,folder)
            if args.phase!='score':
                model=realmodel[-1];fm.write(folder/(args.phase+'_hsmla_stats.json'),{'regularization_final':float(model.fusion.regularization_loss.detach()),'last_gate_fraction':float(model.fusion.last_gate_fraction),'rho':float(model.fusion.rho_logit.sigmoid().detach()),'evaluation_gate_batches':model.gate_trace,'scope':'Last forward only; not a full-population selection rate','encoder_views':model.encoder_images})
        except BaseException:
            fm.write(folder/(args.phase+'_failure.json'),{'traceback':traceback.format_exc(),'time_unix':time.time()});raise
