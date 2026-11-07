"""One fixed 2x2 Marine pilot: task-balanced utility and contextual crops."""
import argparse
import json
import os
from pathlib import Path
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
from sonar_encoder.training.readout import load_targets, metrics_from_confusion

BASE = old.BASE
ROOT = old.ROOT
RUN = BASE / 'gsr_sonar_adaptation_v1'
PARENT = BASE / 'selective_reencoding_marine_v1'


def configure(variant):
    old.RUN = RUN / variant
    return old.inputs()


def load_model(variant):
    path = PARENT/'full/latest.pt' if variant == 'plain' else RUN/'halo/full/latest.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['epoch'] != 100:
        raise RuntimeError('Utility labels require terminal100 fusion/head')
    model = old.Predictor('full').cuda()
    model.load_state_dict(state['model'])
    return model.eval().requires_grad_(False)


def halo_images(common448):
    boxes = json.loads((RUN/'protocol.json').read_text())['halo_boxes_xyxy']
    return torch.stack([F.interpolate(common448[...,y0:y1,x0:x1], (224,224),
                        mode='bilinear', align_corners=False, antialias=True)
                        for x0,y0,x1,y1 in boxes], 1).flatten(0,1)


def align_core(fields, region_ids):
    # Token centers are mapped through the actual clipped source crop.
    boxes = json.loads((RUN/'protocol.json').read_text())['halo_boxes_xyxy']
    grids = []
    for j,(r,c) in enumerate(old.QUADS):
        x0,y0,x1,y1 = boxes[j]
        x = 2*(c*14+(torch.arange(16,device=fields.device)+.5)*14-x0)/(x1-x0)-1
        y = 2*(r*14+(torch.arange(16,device=fields.device)+.5)*14-y0)/(y1-y0)-1
        yy,xx = torch.meshgrid(y,x,indexing='ij')
        grids.append(torch.stack((xx,yy),-1))
    grid = torch.stack(grids)[region_ids]
    return F.grid_sample(fields,grid,mode='bilinear',padding_mode='border',align_corners=False)


@torch.no_grad()
def extract_halo():
    from sonar_encoder.training.model import build_encoder
    p = json.loads((RUN/'protocol.json').read_text())
    old.RUN = RUN/'halo'
    encoder = build_encoder(Path(p['encoder']['code_dir']),Path(p['encoder']['weights_path']),'R010')
    encoder.cuda().eval().requires_grad_(False)
    sample_count = calls = 0
    for split,n in [('train',1000),('validation',251)]:
        source = np.load(p['input_views'][split],mmap_mode='r')
        target = np.lib.format.open_memmap(RUN/'halo'/f'{split}.npy',mode='w+',dtype=np.float32,shape=(n,384,32,32))
        for start in range(0,n,4):
            common448 = torch.from_numpy(np.array(source[start:start+4],copy=True))
            tiles = halo_images(common448).cuda()
            tokens = encoder(tiles)['patch_tokens'].transpose(1,2).reshape(-1,384,16,16)
            ids = torch.arange(4,device='cuda').repeat(len(common448))
            core = align_core(tokens,ids).reshape(len(common448),4,384,16,16)
            field = torch.empty((len(common448),384,32,32),device='cuda')
            for j,(r,c) in enumerate(old.QUADS):field[...,r:r+16,c:c+16] = core[:,j]
            target[start:start+len(common448)] = field.cpu().numpy()
            calls += 1; sample_count += len(tiles)
            old.observe()
        target.flush()
        print(json.dumps({'phase':'halo_extract','split':split,'images':n,'encoder_views':sample_count,'calls':calls}),flush=True)
    old.write(RUN/'halo/extraction.json',{'status':'COMPLETE_FIXED_CONTEXT_CROPS','images':1251,
        'encoder_views':sample_count,'calls':calls,'new_global_encoder_views':0,'image_decodes':0,
        'encoder_eval':not encoder.training,'encoder_any_requires_grad':any(v.requires_grad for v in encoder.parameters()),
        'source':'Existing normalized common224->448 input views; no new native information',
        'halo_boxes_xyxy':p['halo_boxes_xyxy'],'resources':old.observe()})


def boundary_weights(gt,class_weights):
    boundary = torch.zeros_like(gt,dtype=torch.bool)
    diff = gt[:,1:] != gt[:,:-1]
    boundary[:,1:] |= diff; boundary[:,:-1] |= diff
    diff = gt[:,:,1:] != gt[:,:,:-1]
    boundary[:,:,1:] |= diff; boundary[:,:,:-1] |= diff
    band = F.max_pool2d(boundary.float()[:,None],5,stride=1,padding=2)[:,0]
    return class_weights[gt.long()] * (1 + band)


def class_weights_from_fit(gt):
    split = np.load(PARENT/'selector_split.npz')
    count = torch.bincount(gt[split['fit']].long().flatten(),minlength=12).float()
    weights = count.clamp_min(1).rsqrt()
    weights = weights / ((weights*count).sum()/count.sum())
    return weights,count


@torch.no_grad()
def losses_for_actions(model,arrays,class_weights,pred_path=None):
    g,tiles,gt = arrays
    loss = np.empty((len(g),5,2),np.float32)
    predictions = None
    if pred_path is not None:
        predictions = np.lib.format.open_memmap(pred_path,mode='w+',dtype=np.uint8,shape=(5,len(g),480,320))
    for start in range(0,len(g),16):
        sl = slice(start,start+16)
        weights = boundary_weights(gt[sl],class_weights)
        for action in range(-1,4):
            selected = torch.full((len(g[sl]),),action,device='cuda',dtype=torch.long)
            logits = old.native_logits(model(g[sl],tiles[sl],selected))
            ce = F.cross_entropy(logits,gt[sl].long(),reduction='none')
            loss[sl,action+1,0] = ce.mean((1,2)).cpu().numpy()
            loss[sl,action+1,1] = ((ce*weights).sum((1,2))/weights.sum((1,2))).cpu().numpy()
            if predictions is not None:predictions[action+1,sl] = logits.argmax(1).to(torch.uint8).cpu().numpy()
        old.observe()
    if predictions is not None:predictions.flush()
    return loss


def fit_selector(variant,criterion,g,gains):
    folder = RUN/variant/criterion;folder.mkdir()
    split = np.load(PARENT/'selector_split.npz');fit,check = split['fit'],split['check']
    mean = float(gains[fit].mean());scale = max(float(gains[fit].std()),1e-8)
    target = torch.from_numpy((gains-mean)/scale).cuda()
    torch.manual_seed(7);selector = old.Selector().cuda()
    optimizer = torch.optim.AdamW(selector.parameters(),lr=.001,weight_decay=.0001)
    generator = torch.Generator().manual_seed(7)
    begun = time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1,101):
            selector.train();total = 0.
            order = torch.tensor(fit)[torch.randperm(800,generator=generator)]
            for ids in order.split(64):
                optimizer.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(selector(g[ids]),target[ids])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite selector loss')
                loss.backward();optimizer.step();total += float(loss.detach())*len(ids)
            if epoch%10==0:
                log.write(json.dumps({'epoch':epoch,'fit_loss':total/800})+'\n');log.flush();old.observe()
    selector.eval()
    with torch.no_grad():scores = torch.cat([selector(g[i:i+64]) for i in range(0,1000,64)]).cpu().numpy()*scale+mean
    truth = gains[check];selected = scores[check].argmax(1)
    torch.save({'selector':selector.cpu().state_dict(),'epoch':100,'gain_mean':mean,'gain_scale':scale,
                'criterion':criterion,'variant':variant},folder/'terminal.pt')
    fixed = int(gains[fit].mean(0).argmax())
    report = {'variant':variant,'criterion':criterion,'epochs':100,'updates':1300,'parameters':49409,
        'fit800':fit.tolist(),'check200':check.tolist(),'fixed_region_from_fit800':fixed,
        'gain_min':float(gains.min()),'gain_max':float(gains.max()),'negative_fraction':float((gains<0).mean()),
        'internal200_MSE':float(np.square(scores[check]-truth).mean()),
        'internal200_argmax_agreement':float((selected==truth.argmax(1)).mean()),
        'internal200_regret':float((truth.max(1)-truth[np.arange(200),selected]).mean()),
        'wall_seconds':time.monotonic()-begun,'resources':old.observe(),
        'selector_check_not_heldout_from_fusion_training':True,'dev_GT_for_fit_or_actions':False}
    old.write(folder/'report.json',report)
    print(json.dumps({'selector':variant+'/'+criterion,'epoch':100,'check_argmax':report['internal200_argmax_agreement']}),flush=True)


def workflow(variant):
    if variant=='halo':
        extract_halo()
        old.RUN = RUN/'halo';old.fit('full')
    p,manifest,arrays = configure(variant)
    model = load_model(variant)
    weights,count = class_weights_from_fit(arrays['train'][2])
    old.write(RUN/variant/'loss_definition.json',{'class_pixel_counts_fit800':count.cpu().tolist(),
        'class_weights':weights.cpu().tolist(),'boundary':'4-neighbor GT transitions, 5x5 dilation, x2 weight',
        'background0_and_wall11_retained':True,'stat_population':'Selector fit800 only; dev and internal200 excluded'})
    loss = losses_for_actions(model,arrays['train'],weights)
    gains = loss[:,:1,:] - loss[:,1:,:]
    np.savez(RUN/variant/'training_gains.npz',losses=loss,signed_gains=gains,row_ids=np.array(p['row_ids']['train']))
    # The original no-halo ordinary selector is reused, not fitted again.
    for name,axis in ([('ordinary',0),('balanced',1)] if variant=='halo' else [('balanced',1)]):
        fit_selector(variant,name,arrays['train'][0],gains[:,:,axis])
    old.write(RUN/variant/'workflow_complete.json',{'status':'COMPLETE','variant':variant,'resources':old.observe()})


def load_selector(variant,criterion):
    path = PARENT/'selector_terminal.pt' if (variant,criterion)==('plain','ordinary') else RUN/variant/criterion/'terminal.pt'
    state = torch.load(path,map_location='cpu',weights_only=True)
    selector = old.Selector().cuda();selector.load_state_dict(state['selector'])
    return selector.eval().requires_grad_(False)


def selected_metrics(gt,predictions,slots):
    matrix = np.zeros((12,12),np.int64)
    for i,j in enumerate(slots):
        matrix += np.bincount(gt[i].astype(np.int64).ravel()*12+predictions[j,i].ravel(),minlength=144).reshape(12,12)
    return metrics_from_confusion(torch.from_numpy(matrix),'marine')


def score():
    states,actions,diagnostics = {},{},{}
    for variant in ('plain','halo'):
        p,manifest,arrays = configure(variant);model = load_model(variant)
        g,tiles,y = arrays['validation'];gt = y.cpu().numpy()
        if variant=='halo':
            weights,count = class_weights_from_fit(arrays['train'][2])
            pred_path = RUN/'halo/validation_counterfactual_predictions.npy'
            loss = losses_for_actions(model,arrays['validation'],weights,pred_path)
            np.save(RUN/'halo/validation_counterfactual_losses.npy',loss)
        else:pred_path = PARENT/'full/validation_counterfactual_predictions.npy'
        predictions = np.load(pred_path,mmap_mode='r')
        gains = np.load(RUN/variant/'training_gains.npz')['signed_gains']
        fit = np.load(PARENT/'selector_split.npz')['fit']
        policies = {'base':np.zeros(251,np.int64),
                    'random':np.load(PARENT/'validation_random_actions.npy')+1,
                    'uncertainty':old.uncertainty_actions(model,g)+1}
        for name,axis in [('ordinary',0),('balanced',1)]:
            selector = load_selector(variant,name)
            with torch.no_grad():j = torch.cat([selector(g[i:i+64]) for i in range(0,251,64)]).argmax(1).cpu().numpy()
            policies['learned_'+name] = j+1
            if (variant,name)==('plain','ordinary'):
                # Keep the original frozen baseline selection exactly as evaluated.
                policies['learned_ordinary'] = np.load(PARENT/'validation_actions.npz')['learned']+1
                fixed = json.loads((PARENT/'selector_report.json').read_text())['fixed_region_selected_from_fit800']
            else:fixed = int(gains[fit,:,axis].mean(0).argmax())
            policies['fixed_'+name] = np.full(251,fixed+1,np.int64)
        if variant=='halo':
            for name,axis in [('ordinary',0),('balanced',1)]:
                diagnostics[name] = selected_metrics(gt,predictions,loss[:,1:,axis].argmin(1)+1)
        states[variant] = {name:selected_metrics(gt,predictions,slots) for name,slots in policies.items()}
        actions[variant] = {name:slots.tolist() for name,slots in policies.items()}
        old.write(RUN/variant/'evaluation.json',{'metrics':states[variant],'slots':actions[variant],
            'source_predictions':str(pred_path),'model_checkpoint_epoch':100,'validation_images':251})
        del model,selector,arrays,g,tiles,y,predictions
    contrast = {
        'balanced_gain_without_halo':states['plain']['learned_balanced']['miou12']-states['plain']['learned_ordinary']['miou12'],
        'halo_with_ordinary_gain':states['halo']['learned_ordinary']['miou12']-states['plain']['learned_ordinary']['miou12'],
        'halo_with_balanced_gain':states['halo']['learned_balanced']['miou12']-states['plain']['learned_balanced']['miou12'],
        'balanced_gain_with_halo':states['halo']['learned_balanced']['miou12']-states['halo']['learned_ordinary']['miou12'],
        'both_minus_original':states['halo']['learned_balanced']['miou12']-states['plain']['learned_ordinary']['miou12']}
    for v in ('plain','halo'):
        for control in ('random','fixed_balanced','uncertainty','base'):
            contrast[v+'_balanced_minus_'+control]=states[v]['learned_balanced']['miou12']-states[v][control]['miou12']
    old.write(RUN/'results.json',{'status':'COMPLETE_FIXED_2X2_MARINE_ADAPTATION','states':states,'contrasts':contrast,
        'halo_loss_oracles_diagnostic_only':diagnostics,'actions':actions,
        'primary':'Fixed terminal100 fusion/head and selector, native12-class mIoU',
        'ordinary_plain_reference_inherited_no_refit_no_new_prediction_forward':True,
        'scope':'Single seed, development split; formula/halo innovation candidates, no novelty or significance acceptance',
        'halo_confounds':'Context is added inside same224 local input; core sampling density changes. Not pure isolated context causality.',
        'new_fusion_updates':6300,'new_selector_updates':3900,'resources':old.observe()})
    print(json.dumps({'contrasts':contrast,'matrix':{v:{k:x['miou12'] for k,x in z.items()} for v,z in states.items()}}),flush=True)


def cost():
    from sonar_encoder.training.model import build_encoder,prepare_images
    p=json.loads((RUN/'protocol.json').read_text());old.RUN=RUN
    models={v:load_model(v) for v in ('plain','halo')}
    selectors={(v,c):load_selector(v,c) for v in ('plain','halo') for c in ('ordinary','balanced')}
    encoder=build_encoder(Path(p['encoder']['code_dir']),Path(p['encoder']['weights_path']),'R010').cuda().eval().requires_grad_(False)
    rows=p['cost_rows'];natives=[]
    for row in rows:
        with Image.open(row['image']) as im:a=torch.from_numpy(np.array(im.convert('L'),dtype=np.uint8)).float()[None]/255
        natives.append({'intensity':a,'valid':torch.ones(a.shape[-2:],dtype=torch.bool)})
    heads={}
    for name,spec in p['cost_reference_heads'].items():
        h=nn.Conv2d(384,12,1).cuda();h.load_state_dict(torch.load(spec['path'],map_location='cpu',weights_only=True)['head']);h.eval().requires_grad_(False);heads[name]=h
    cases=[(v,c) for v in ('plain','halo') for c in ('ordinary','balanced','uncertainty')]+[(x,None) for x in ('reference_regrid','reference_tiles','reference_whole')]
    results=[]
    for variant,criterion in cases:
        raw=[];load=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        baseline=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats()
        for visit in range(146):
            i=visit%32 if visit<50 else (visit-50)%32
            torch.cuda.synchronize();start=time.perf_counter()
            common=prepare_images([natives[i]],p['normalization'],None)
            with torch.no_grad():
                if variant in ('reference_tiles','reference_whole'):
                    large=F.interpolate(common,(448,448),mode='bilinear',align_corners=False,antialias=True)
                    if variant=='reference_whole':field=encoder(large.cuda())['patch_tokens'].transpose(1,2).reshape(1,384,32,32)
                    else:
                        images=torch.cat([large[...,r*14:r*14+224,c*14:c*14+224] for r,c in old.QUADS]).cuda()
                        local=encoder(images)['patch_tokens'].transpose(1,2).reshape(4,384,16,16)
                        field=torch.cat((torch.cat((local[0],local[1]),2),torch.cat((local[2],local[3]),2)),1)[None]
                    logits=heads[variant](field)
                else:
                    g=encoder(common.cuda())['patch_tokens'].transpose(1,2).reshape(1,384,16,16)
                    if variant=='reference_regrid':logits=heads[variant](F.interpolate(g,(32,32),mode='bilinear',align_corners=False,antialias=False))
                    else:
                        model=models[variant]
                        if criterion=='uncertainty':j=int(old.uncertainty_actions(model,g)[0])
                        else:j=int(selectors[(variant,criterion)](g).argmax(1))
                        large=F.interpolate(common,(448,448),mode='bilinear',align_corners=False,antialias=True)
                        r,c=old.QUADS[j]
                        if variant=='halo':
                            x0,y0,x1,y1=p['halo_boxes_xyxy'][j]
                            image=F.interpolate(large[...,y0:y1,x0:x1],(224,224),mode='bilinear',align_corners=False,antialias=True)
                        else:image=large[...,r*14:r*14+224,c*14:c*14+224]
                        local=encoder(image.cuda())['patch_tokens'].transpose(1,2).reshape(1,384,16,16)
                        if variant=='halo':local=align_core(local,torch.tensor([j],device='cuda'))
                        tiles=torch.zeros((1,4,384,16,16),device='cuda');tiles[:,j]=local
                        logits=model(g,tiles,torch.tensor([j],device='cuda'))
                prediction=old.native_logits(logits).argmax(1).to(torch.uint8).cpu()
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if visit>=50:raw.append({'row_id':rows[i]['row_id'],'round':(visit-50)//32,'seconds':seconds})
        values=[x['seconds'] for x in raw]
        results.append({'variant':variant,'criterion':criterion,'median_seconds':float(np.median(values)),
            'p95_seconds':float(np.percentile(values,95)),'raw':raw,'warmups':50,'timed':96,
            'allocator_baseline_reserved':baseline,'absolute_peak_reserved':torch.cuda.max_memory_reserved(),
            'peak_allocated':torch.cuda.max_memory_allocated(),'GPU_load':load})
        print(json.dumps({'cost':variant+'/'+str(criterion),'median_ms':1000*np.median(values)}),flush=True);old.observe()
    old.write(RUN/'cost_results.json',{'status':'COMPLETE_NINE_REAL_ONLINE_CELLS','cells':results,
        'boundary':'CPU native image RAM, preprocessing/H2D/real encoder(s)/selector/fusion/head/native prediction CPU; decode/setup excluded',
        'feature_cache_reads_in_timed_calls':0,'own_training_concurrent':False,
        'memory_scope':'Shared resident encoder/head/selector pool and allocator history; not isolated deployment-process peaks',
        'reference_head_role':'Existing development-selected best; new method uses terminal100',
        'resources':old.observe()})


def pipeline():
    RUN.mkdir();begun=time.time();parent=json.loads((PARENT/'protocol.json').read_text())
    receipt=json.loads((BASE/'downstream_vit_finetuning_v1/inputs/Marine/receipt.json').read_text())
    views={}
    for split in ('train','validation'):
        record=receipt['records'][split]
        if record['row_ids']!=parent['row_ids'][split] or record['shape']!=[len(parent['row_ids'][split]),1,448,448]:
            raise RuntimeError('Existing normalized view population/header incompatible')
        views[split]=record['path']
    boxes=[]
    for r,c in old.QUADS:
        x,y=c*14,r*14;boxes.append([max(0,x-32),max(0,y-32),min(448,x+256),min(448,y+256)])
    protocol={**parent,'unit':RUN.name,'created_unix':begun,'deadline_unix':begun+10800,
        'user_authorized_scope':'Two specific sonar-adaptation candidates and fixed2x2 comparison; no tests/hashes/review or new task/seed sweep',
        'input_views':views,'input_view_receipt':str(BASE/'downstream_vit_finetuning_v1/inputs/Marine/receipt.json'),
        'input_view_operation':'Native log1p/norm -> CPU AA224 -> CPU AA448, matching the bound existing common observation',
        'halo_boxes_xyxy':boxes,'halo_margin_source448_pixels':32,'halo_input_size':224,
        'halo_alignment':'Output core16 grid via bilinear grid_sample, border padding, alignFalse, using clipped source-crop coordinates; unchanged global positions',
        'balanced_utility':'Signed decrease of inverse-sqrt fit800 class-count weighted CE, floor count1, normalized class weights; all12 classes includingbg/wall. GT4-neighbor boundary dilated5x5 receives x2 weight; normalize by image weight sum.',
        'fusion_training_loss_unchanged':'Unweighted native480x320 all12 CE; AdamW .001 wd.0001 B16 100epochs6300updates; same old action plan/order/init',
        'matrix':['plain+ordinary inherited','plain+balanced new selector','halo+ordinary new selector','halo+balanced new selector'],
        'new_fits':{'fusion_heads':1,'selectors':3,'fusion_updates':6300,'selector_updates':3900},
        'gain_label_populations':'Training1000 only; weights/scaling/fixed region derived fromfit800; check200 not held out from fusion/head fitting',
        'primary':'All terminal100; no candidate/best/seed selection after development outcomes',
        'same_inference_budget':'One global224 and one local224 encoder call; no extra native pixels, ring inside existing common448 only',
        'controls':'Each variant random, training-selected fixed and base-entropy selection; all signed2x2 differences',
        'online_cost':'Nine cells, batch1 50warmups+same32dev IDs x3, after own model fitting ended',
        'source_utf8':Path(__file__).read_text(),'reused_source':str(ROOT/'runs/selective_reencoding.py'),
        'stop':'Real errors/nonfinite/resources stop all owned workers; no automatic retry, threshold or scope expansion'}
    old.write(RUN/'protocol.json',protocol)
    for variant in ('plain','halo'):
        folder=RUN/variant;folder.mkdir()
        sub={**protocol,'caches':json.loads(json.dumps(parent['caches']))}
        if variant=='halo':
            for split in ('train','validation'):sub['caches'][split]['tiles']=str(folder/f'{split}.npy')
        old.write(folder/'protocol.json',sub)
        for name in ['training_actions.npy','validation_random_actions.npy']:(folder/name).symlink_to(PARENT/name)
    (RUN/'logs').mkdir();children={};rows=[]
    def launch(phase,gpu,cap):
        entry=old.admission(gpu)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','XFORMERS_DISABLED':'1'}
        log=(RUN/'logs'/f'{phase}.log').open('w')
        command=[old.PYTHON,'-B','-u','-m','runs.gsr_sonar_adaptation',phase]
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[phase]=(child,log,time.time(),min(cap,protocol['deadline_unix']-time.time()),entry)
        old.write(RUN/'queue_state.json',{'status':'RUNNING','active':{k:v[0].pid for k,v in children.items()},'completed':rows})
    def finish(phase):
        child,log,start,cap,entry=children[phase]
        try:code=child.wait(timeout=max(0.,cap-(time.time()-start)))
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        log.close();del children[phase]
        row={'phase':phase,'PID':child.pid,'physical_GPU':entry['physical_GPU'],'actual_wait_exit':code,'wall_seconds':time.time()-start,'admission':entry}
        rows.append(row)
        with (RUN/'compute_ledger.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if code!=0:raise RuntimeError(phase+' actual exit '+str(code))
    try:
        launch('plain',2,1800);launch('halo',3,3600)
        old.write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':{k:v[0].pid for k,v in children.items()},'protocol':str(RUN/'protocol.json'),'actual_start_unix':time.time(),'GPU_allowed':[2,3]})
        while children:
            for phase in list(children):
                child,log,start,cap,entry=children[phase]
                if child.poll() is not None or time.time()-start>=cap:finish(phase)
            time.sleep(1)
        for phase,cap in [('score',1800),('cost',1200)]:launch(phase,2,cap);finish(phase)
        disk=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
        if disk>8*1024**3:raise RuntimeError('New entity disk cap exceeded')
        close={'status':'COMPLETE_FIXED_GSR_SONAR_2X2_PILOT','actual_waits':rows,'supervisor_PID':os.getpid(),
            'pipeline_wall_seconds':time.time()-begun,'summed_child_wall_seconds':sum(r['wall_seconds'] for r in rows),
            'new_entity_bytes':disk,'results':str(RUN/'results.json'),'cost':str(RUN/'cost_results.json'),
            'new_updates':10200,'new_frozen_encoder_views':5004,'new_frozen_encoder_calls':313,
            'no_next_queue':True,'no_new_smoke_SHA_review':True}
        old.write(RUN/'final_closure.json',close);old.write(RUN/'writing_handoff.json',close)
        old.write(RUN/'queue_state.json',{'status':'COMPLETE','actual_waits':rows})
    except BaseException:
        for phase,(child,log,start,cap,entry) in children.items():
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            code=child.wait();log.close();rows.append({'phase':phase,'PID':child.pid,'actual_wait_exit':code,'cancelled_after_failure':True})
        old.write(RUN/'failure_closure.json',{'status':'STOPPED_NO_RETRY','traceback':traceback.format_exc(),'actual_waits':rows})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pipeline','plain','halo','score','cost']);args=parser.parse_args()
    old.runtime()
    if args.phase=='pipeline':pipeline()
    elif args.phase in ('plain','halo'):workflow(args.phase)
    elif args.phase=='score':score()
    else:cost()
