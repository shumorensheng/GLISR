"""Fixed one-of-four frozen-feature Marine pilot; no smoke or digest work."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from sonar_encoder.training.readout import load_targets, confusion_matrix, metrics_from_confusion

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'selective_reencoding_marine_v1'
ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
QUADS = ((0, 0), (0, 16), (16, 0), (16, 16))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def runtime():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('highest')
    torch.serialization.set_crc32_options(False)


def observe():
    v = {'worker_PID': os.getpid(), 'RSS_HWM_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
         'GPU_reserved_peak_bytes': torch.cuda.max_memory_reserved(0) if torch.cuda.is_initialized() else 0}
    if v['RSS_HWM_bytes'] > 32 * 1024**3 or v['GPU_reserved_peak_bytes'] > 24 * 1024**3:
        raise RuntimeError('Pilot resource cap exceeded: ' + str(v))
    if time.time() >= json.loads((RUN / 'protocol.json').read_text())['deadline_unix']:
        raise TimeoutError('Original three-hour pilot deadline reached')
    return v


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(384, 64, 1)
        self.base = nn.Conv2d(384, 64, 1)
        self.context = nn.Conv2d(384, 64, 1)
        self.position = nn.Linear(2, 64)
        self.norm = nn.LayerNorm(64)
        self.output = nn.Conv2d(64, 384, 1)
        self.gate = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)
        y, x = torch.meshgrid((torch.arange(16) + .5) / 8 - 1,
                              (torch.arange(16) + .5) / 8 - 1, indexing='ij')
        self.register_buffer('global_xy', torch.stack((x, y), -1).reshape(256, 2))
        y, x = torch.meshgrid((torch.arange(32) + .5) / 16 - 1,
                              (torch.arange(32) + .5) / 16 - 1, indexing='ij')
        xy = torch.stack((x, y), -1)
        self.register_buffer('local_xy', torch.stack([xy[r:r+16, c:c+16].reshape(256, 2) for r, c in QUADS]))
        taper = torch.sin(torch.pi * (torch.arange(16) + .5) / 16)
        self.register_buffer('taper', (taper[:, None] * taper[None, :])[None, None])

    def forward(self, local, base, global_field, action):
        n = len(local)
        q = (self.local(local) + self.base(base)).flatten(2).transpose(1, 2)
        q = q + self.position(self.local_xy[action.clamp_min(0)])
        v = self.context(global_field).flatten(2).transpose(1, 2)
        k = v + self.position(self.global_xy)[None]
        heads = lambda a: a.reshape(n, 256, 4, 16).transpose(1, 2)
        attention = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0.)
        u = attention.transpose(1, 2).reshape(n, 256, 64)
        u = F.gelu(self.norm(q + u)).transpose(1, 2).reshape(n, 64, 16, 16)
        return self.output(u) * self.gate(u).sigmoid() * self.taper


class Predictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        torch.manual_seed(0)
        self.head = nn.Conv2d(384, 12, 1)
        if kind != 'replace':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(101)
                self.fusion = Fusion()

    def forward(self, g, tiles, action):
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False, antialias=False)
        if not bool((action >= 0).any()):
            return self.head(base)
        crops = torch.stack([base[..., r:r+16, c:c+16] for r, c in QUADS], 1)
        ids = torch.arange(len(g), device=g.device)
        crop = crops[ids, action.clamp_min(0)]
        local = tiles[ids, action.clamp_min(0)] if self.kind != 'capacity' else crop
        delta = local - crop if self.kind == 'replace' else self.fusion(local, crop, g, action)
        field = base
        for j, (r, c) in enumerate(QUADS):
            selected = (action == j)[:, None, None, None]
            field = field + F.pad(delta * selected, (c, 16-c, r, 16-r))
        return self.head(field)


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(770, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer('position', torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]]))

    def forward(self, g):
        patch = F.adaptive_avg_pool2d(g, (2, 2)).flatten(2).transpose(1, 2)
        whole = g.mean((2, 3))[:, None].expand(-1, 4, -1)
        position = self.position[None].expand(len(g), -1, -1)
        return self.mlp(torch.cat((patch, whole, position), -1)).squeeze(-1)


def inputs():
    p = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(p['manifest']).read_text())
    arrays = {}
    for split in ('train', 'validation'):
        g = torch.from_numpy(np.load(p['caches'][split]['global'], allow_pickle=False)).cuda()
        stitched = torch.from_numpy(np.load(p['caches'][split]['tiles'], allow_pickle=False)).cuda()
        tiles = torch.stack([stitched[..., r:r+16, c:c+16] for r, c in QUADS], 1)
        y = load_targets(manifest[split], 'marine').cuda()
        arrays[split] = (g, tiles, y)
        del stitched
    return p, manifest, arrays


def native_logits(logits):
    return F.interpolate(logits, (480, 320), mode='bilinear', align_corners=False)


def save_model(path, model, epoch, optimizer=None, generator=None):
    obj = {'kind': model.kind, 'epoch': epoch, 'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if optimizer is not None:
        obj.update(optimizer=optimizer.state_dict(), private_rng=generator.get_state(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluation(model, arrays, actions):
    model.eval()
    g, tiles, gt = arrays
    matrix = torch.zeros((12, 12), dtype=torch.long, device='cuda')
    for start in range(0, len(g), 16):
        sl = slice(start, start+16)
        pred = native_logits(model(g[sl], tiles[sl], actions[sl])).argmax(1)
        matrix += confusion_matrix(gt[sl], pred, 12)
    return metrics_from_confusion(matrix.cpu(), 'marine')


def fit(kind):
    p, manifest, arrays = inputs()
    folder = RUN / kind
    folder.mkdir()
    model = Predictor(kind).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(0)
    actions = torch.from_numpy(np.load(RUN / 'training_actions.npy')).cuda()
    validation_actions = torch.from_numpy(np.load(RUN / 'validation_random_actions.npy')).cuda()
    g, tiles, y = arrays['train']
    best, best_epoch = -1., 0
    updates = 0
    begun = time.monotonic()
    with (folder / 'metrics.jsonl').open('w') as log:
        for epoch in range(1, 101):
            model.train()
            order = torch.randperm(1000, generator=generator)
            loss_sum = 0.
            for ids in order.split(16):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(native_logits(model(g[ids], tiles[ids], actions[epoch-1, ids])), y[ids].long())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite normal training loss')
                loss.backward()
                optimizer.step()
                updates += 1
                loss_sum += float(loss.detach()) * len(ids)
                if updates % 100 == 0:
                    observe()
            metrics = evaluation(model, arrays['validation'], validation_actions)
            entry = {'epoch': epoch, 'updates': updates, 'loss': loss_sum / 1000,
                     'validation_random_routing': metrics, 'elapsed_seconds': time.monotonic()-begun}
            log.write(json.dumps(entry) + '\n'); log.flush()
            if metrics['miou12'] > best:
                best, best_epoch = metrics['miou12'], epoch
                save_model(folder / 'best_random_routing.pt', model, epoch)
            if epoch % 10 == 0 or epoch == 1:
                save_model(folder / 'latest.pt', model, epoch, optimizer, generator)
                print(json.dumps({'phase': kind, 'epoch': epoch, 'updates': updates, 'miou12_random': metrics['miou12'], 'resources': observe()}), flush=True)
    write(folder / 'fit_report.json', {'status': 'COMPLETE_100_EPOCHS', 'kind': kind, 'epochs': 100,
          'updates': updates, 'parameters': sum(v.numel() for v in model.parameters()),
          'terminal_random_routing': metrics, 'best_random_routing': best, 'best_epoch': best_epoch,
          'elapsed_seconds': time.monotonic()-begun, 'resources': observe(),
          'encoder_constructs': 0, 'encoder_forwards': 0,
          'full_method_evaluation_uses_terminal100_not_best_random_checkpoint': True})


@torch.no_grad()
def counterfactuals(model, arrays, destination, save_predictions):
    model.eval()
    g, tiles, gt = arrays
    losses = np.empty((len(g), 5), np.float32)
    predictions = (np.lib.format.open_memmap(destination, mode='w+', dtype=np.uint8,
                    shape=(5, len(g), 480, 320)) if save_predictions else None)
    for start in range(0, len(g), 16):
        sl = slice(start, start+16)
        for slot in range(5):
            action = torch.full((len(g[sl]),), slot-1, dtype=torch.long, device='cuda')
            logits = native_logits(model(g[sl], tiles[sl], action))
            values = F.cross_entropy(logits, gt[sl].long(), reduction='none').mean((1, 2))
            losses[sl, slot] = values.cpu().numpy()
            if save_predictions:
                predictions[slot, sl] = logits.argmax(1).to(torch.uint8).cpu().numpy()
        observe()
    if save_predictions:
        predictions.flush()
    return losses


@torch.no_grad()
def uncertainty_actions(model, g):
    fields = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False, antialias=False)
    logits = model.head(fields)
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    scores = torch.stack([entropy[:, r:r+16, c:c+16].mean((1, 2)) for r, c in QUADS], 1)
    return scores.argmax(1).cpu().numpy()


def selector_stage():
    p, manifest, arrays = inputs()
    saved = torch.load(RUN / 'full/latest.pt', map_location='cpu', weights_only=True)
    model = Predictor('full').cuda()
    model.load_state_dict(saved['model'])
    model.eval().requires_grad_(False)
    if saved['epoch'] != 100:
        raise RuntimeError('Gain labels require the fixed terminal100 fusion/head')
    losses = counterfactuals(model, arrays['train'], None, False)
    gains = losses[:, :1] - losses[:, 1:]
    np.savez(RUN / 'training_gain_labels.npz', losses=losses, signed_gains=gains,
             row_ids=np.array(p['row_ids']['train']))
    split = np.load(RUN / 'selector_split.npz')
    fit_ids, check_ids = split['fit'], split['check']
    mean, scale = float(gains[fit_ids].mean()), max(float(gains[fit_ids].std()), 1e-8)
    targets = torch.from_numpy((gains-mean)/scale).cuda()
    torch.manual_seed(7)
    selector = Selector().cuda()
    optimizer = torch.optim.AdamW(selector.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(7)
    g = arrays['train'][0]
    with (RUN / 'selector_training.jsonl').open('w') as log:
        for epoch in range(1, 101):
            selector.train()
            order = torch.tensor(fit_ids)[torch.randperm(800, generator=generator)]
            loss_sum = 0.
            for ids in order.split(64):
                optimizer.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(selector(g[ids]), targets[ids])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite selector loss')
                loss.backward(); optimizer.step()
                loss_sum += float(loss.detach()) * len(ids)
            if epoch % 10 == 0:
                log.write(json.dumps({'epoch':epoch,'fit_loss':loss_sum/800})+'\n');log.flush()
                observe()
    selector.eval()
    with torch.no_grad():
        scores = torch.cat([selector(g[i:i+64]) for i in range(0,1000,64)]).cpu().numpy()*scale+mean
        dev_g = arrays['validation'][0]
        dev_scores = torch.cat([selector(dev_g[i:i+64]) for i in range(0,251,64)]).cpu().numpy()*scale+mean
    torch.save({'selector':selector.cpu().state_dict(),'epoch':100,'fit_ids':fit_ids.tolist(),
                'check_ids':check_ids.tolist(),'gain_mean':mean,'gain_scale':scale}, RUN/'selector_terminal.pt')
    actions = dev_scores.argmax(1).astype(np.int64)
    fixed = int(gains[fit_ids].mean(0).argmax())
    np.savez(RUN/'validation_actions.npz',learned=actions,
             random=np.load(RUN/'validation_random_actions.npy'),fixed=np.full(251,fixed,np.int64),
             uncertainty=uncertainty_actions(model,dev_g),predicted_gain=dev_scores)
    checks = {}
    for name, ids in [('fit800',fit_ids),('selector_check200',check_ids)]:
        selected=scores[ids].argmax(1);truth=gains[ids]
        checks[name]={'MSE':float(np.square(scores[ids]-truth).mean()),
          'argmax_agreement':float((selected==truth.argmax(1)).mean()),
          'mean_selected_gain':float(truth[np.arange(len(ids)),selected].mean()),
          'mean_oracle_regret':float((truth.max(1)-truth[np.arange(len(ids)),selected]).mean())}
    write(RUN/'selector_report.json',{'status':'COMPLETE_FIXED_TRAIN_ONLY_SELECTOR', 'training_examples':800,
         'internal_check_examples':200,'epochs':100,'updates':1300,'fixed_region_selected_from_fit800':fixed,
         'gain_label_min':float(gains.min()),'gain_label_max':float(gains.max()),
         'negative_gain_fraction':float((gains<0).mean()),'checks':checks,'resources':observe(),
         'internal_check_role':'Held out from selector fitting, not from fusion/head training',
         'validation_GT_used_for_selector_fit_or_actions':False})


def score():
    from runs.marine_native_information import independent_metrics
    p, manifest, arrays = inputs()
    gt = arrays['validation'][2].cpu().numpy()
    actions = dict(np.load(RUN/'validation_actions.npz'))
    results, checks = {}, []
    primary_ids = actions['learned']
    for kind in ('full','capacity','replace'):
        saved=torch.load(RUN/kind/'latest.pt',map_location='cpu',weights_only=True)
        model=Predictor(kind).cuda();model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
        if saved['epoch']!=100:raise RuntimeError('Nonterminal model in final scoring')
        pred_path=RUN/kind/'validation_counterfactual_predictions.npy'
        losses=counterfactuals(model,arrays['validation'],pred_path,True)
        np.save(RUN/kind/'validation_counterfactual_losses.npy',losses)
        predictions=np.load(pred_path,mmap_mode='r')
        policies={'base':np.zeros(251,np.int64),'learned_shared_selection':primary_ids+1,
                  'random':actions['random']+1,'fixed':actions['fixed']+1,'uncertainty':actions['uncertainty']+1}
        if kind=='full':
            policies.update(oracle_loss_diagnostic=losses[:,1:].argmin(1)+1)
        values={}
        for policy,slots in policies.items():
            matrix=np.zeros((12,12),np.int64)
            for i,slot in enumerate(slots):
                matrix+=np.bincount(gt[i].astype(np.int64).ravel()*12+predictions[slot,i].ravel(),minlength=144).reshape(12,12)
            metrics=metrics_from_confusion(torch.from_numpy(matrix),'marine')
            # Separate NumPy formula over the actual saved predictions and original GT.
            selected=np.asarray(predictions[slots,np.arange(251)])
            independent=independent_metrics(gt,selected)
            if independent['confusion_matrix']!=metrics['confusion_matrix'] or abs(independent['miou12']-metrics['miou12'])>1e-12:
                raise RuntimeError('Saved-prediction score mismatch')
            values[policy]=independent
            checks.append({'kind':kind,'policy':policy,'status':'PASS_SAVED_PREDICTIONS_ORIGINAL_GT',
                           'predictions':str(pred_path),'slots':slots.tolist()})
        results[kind]=values
        del model, predictions
    full=results['full']; contrasts={}
    for control in ('base','random','fixed','uncertainty'):
        contrasts['learned_minus_'+control]=full['learned_shared_selection']['miou12']-full[control]['miou12']
    for control in ('capacity','replace'):
        contrasts['full_minus_'+control+'_same_selection']=full['learned_shared_selection']['miou12']-results[control]['learned_shared_selection']['miou12']
    write(RUN/'verification.json',{'status':'PASS_ALL_16_POLICY_SCORES','checks':checks,'new_encoder_calls':0,
          'policy_predictions_selected_from_saved_five_counterfactual_actions':True})
    write(RUN/'results.json',{'status':'COMPLETE_MARINE_ONE_REGION_SEED0_PILOT','primary':'terminal100 mIoU12',
          'states':results,'contrasts':contrasts,'references_inherited_not_rescored':p['references'],
          'selector_report':str(RUN/'selector_report.json'),'resources':observe(),
          'scope':'Single-seed supervised Marine development pilot; no FSOD, test or repeatability claim. Oracle uses validation GT only as labeled diagnostic, never inference selection.',
          'best_random_routing_is_not_best_complete_selector_pipeline':True})
    print(json.dumps({'results':str(RUN/'results.json'),'contrasts':contrasts}),flush=True)


def cost():
    from sonar_encoder.training.model import build_encoder, prepare_images
    p=json.loads((RUN/'protocol.json').read_text());norm=p['normalization'];rows=p['cost_rows']
    model=Predictor('full').cuda();model.load_state_dict(torch.load(RUN/'full/latest.pt',map_location='cpu',weights_only=True)['model']);model.eval().requires_grad_(False)
    selector=Selector().cuda();ss=torch.load(RUN/'selector_terminal.pt',map_location='cpu',weights_only=True);selector.load_state_dict(ss['selector']);selector.eval().requires_grad_(False)
    encoder=build_encoder(Path(p['encoder']['code_dir']),Path(p['encoder']['weights_path']),'R010').cuda().eval().requires_grad_(False)
    old={}
    for name,spec in p['cost_reference_heads'].items():
        head=nn.Conv2d(384,12,1).cuda();saved=torch.load(spec['path'],map_location='cpu',weights_only=True)
        head.load_state_dict(saved['head']);head.eval().requires_grad_(False);old[name]=head
    natives=[]
    for row in rows:
        with Image.open(row['image']) as im:gray=np.array(im.convert('L'),dtype=np.uint8)
        a=torch.from_numpy(gray).float()[None]/255
        natives.append({'intensity':a,'valid':torch.ones(a.shape[-2:],dtype=torch.bool)})
    fixed=json.loads((RUN/'selector_report.json').read_text())['fixed_region_selected_from_fit800']
    random_actions=np.load(RUN/'validation_random_actions.npy')
    random_by_id=dict(zip(p['row_ids']['validation'],random_actions.tolist()))
    measured=[]
    for policy in ('learned','random','fixed','uncertainty','base','reference_regrid','reference_tiles','reference_whole'):
        load=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
        baseline=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats()
        records=[]
        for visit in range(146):
            i=(visit%32) if visit<50 else ((visit-50)%32)
            torch.cuda.synchronize();start=time.perf_counter()
            common=prepare_images([natives[i]],norm,None)
            with torch.no_grad():
                if policy in ('reference_tiles','reference_whole'):
                    image=F.interpolate(common,(448,448),mode='bilinear',align_corners=False,antialias=True)
                    if policy=='reference_tiles':
                        batch=torch.cat([image[...,r*14:r*14+224,c*14:c*14+224] for r,c in QUADS]).cuda()
                        t=encoder(batch)['patch_tokens'].transpose(1,2).reshape(4,384,16,16)
                        field=torch.cat((torch.cat((t[0],t[1]),2),torch.cat((t[2],t[3]),2)),1)[None]
                    else:field=encoder(image.cuda())['patch_tokens'].transpose(1,2).reshape(1,384,32,32)
                    logits=old[policy](field)
                else:
                    g=encoder(common.cuda())['patch_tokens'].transpose(1,2).reshape(1,384,16,16)
                    if policy=='reference_regrid':logits=old[policy](F.interpolate(g,(32,32),mode='bilinear',align_corners=False,antialias=False))
                    else:
                        if policy=='learned':j=int(selector(g).argmax(1))
                        elif policy=='random':j=random_by_id[rows[i]['row_id']]
                        elif policy=='fixed':j=fixed
                        elif policy=='uncertainty':j=int(uncertainty_actions(model,g)[0])
                        else:j=-1
                        tiles=torch.zeros((1,4,384,16,16),device='cuda')
                        if j>=0:
                            image=F.interpolate(common,(448,448),mode='bilinear',align_corners=False,antialias=True)
                            r,c=QUADS[j];local=image[...,r*14:r*14+224,c*14:c*14+224]
                            tiles[:,j]=encoder(local.cuda())['patch_tokens'].transpose(1,2).reshape(1,384,16,16)
                        logits=model(g,tiles,torch.tensor([j],device='cuda'))
                prediction=native_logits(logits).argmax(1).to(torch.uint8).cpu()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if visit>=50:records.append({'row_id':rows[i]['row_id'],'round':(visit-50)//32,'seconds':elapsed})
        v=[r['seconds'] for r in records]
        measured.append({'policy':policy,'median_seconds':float(np.median(v)),'p95_seconds':float(np.percentile(v,95)),
             'raw':records,'warmup':50,'timed':96,'baseline_reserved_bytes':baseline,
             'absolute_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_load_metadata':load,
             'load_scope':'Observed shared-machine load; no idle guarantee',
             'timing_boundary':'Native image in CPU RAM through preprocessing, H2D, encoder(s), selector/fusion/head, native interpolation/argmax and CPU prediction; disk decode and setup excluded'})
        print(json.dumps({'cost_policy':policy,'median':float(np.median(v)),'p95':float(np.percentile(v,95))}),flush=True)
        observe()
    write(RUN/'cost_results.json',{'status':'COMPLETE_8_ONLINE_CELLS','rows':measured,
          'no_training_concurrent_on_own_GPUs':True,'reference_heads':p['cost_reference_heads'],
          'learned_method_checkpoint_role':'Fixed terminal100 fusion/head plus terminal100 selector',
          'old_reference_cost_checkpoint_role':'Development-selected best, explicitly different from terminal accuracy references',
          'feature_cache_access_in_online_calls':False,'resources':observe()})


def admission(gpu):
    line=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    free,util=[int(x.strip()) for x in line.split(',')]
    uuids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    app=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes=[]
    for line in app.splitlines():
        uuid,pid,memory=[v.strip() for v in line.split(',')]
        if uuid!=uuids:continue
        proc=Path('/proc')/pid
        if proc.exists():processes.append({'pid':int(pid),'UID':proc.stat().st_uid,'memory_MiB':memory})
    if free<24576 or any(x['UID']==os.getuid() for x in processes):
        raise RuntimeError('GPU admission unavailable: '+str((gpu,free,processes)))
    return {'physical_GPU':gpu,'free_MiB':free,'utilization_percent':util,'processes':processes}


def pipeline():
    RUN.mkdir()
    begun=time.time()
    parent=json.loads((BASE/'marine_native_information_v1/protocol.json').read_text())
    manifest=json.loads(Path(parent['manifest']).read_text())
    previous=json.loads((BASE/'marine_feature_regrid_repeats_v1/protocol.json').read_text())
    if parent['row_ids']!=previous['row_ids']:
        raise RuntimeError('Existing cache row identities differ')
    caches={}
    for split,n in [('train',1000),('validation',251)]:
        caches[split]={'global':str(BASE/f'marine_frozen_transfer_v1/t0/{split}.npy'),
                       'tiles':str(BASE/f'marine_native_information_v1/interp_tiles/{split}.npy')}
        for name,shape in [('global',(n,384,16,16)),('tiles',(n,384,32,32))]:
            a=np.load(caches[split][name],mmap_mode='r')
            if a.shape!=shape or a.dtype!=np.float32:raise RuntimeError('Incompatible frozen cache header')
    cost_parent=json.loads((BASE/'swise_supplement_v1/E_cost/protocol.json').read_text())
    cost_ids=[r['original_image_id'] for r in cost_parent['inputs']['marine']]
    by_id={r['row_id']:r for r in manifest['validation']}
    norm=json.loads(Path(parent['normalization']).read_text())
    p={'unit':RUN.name,'created_unix':begun,'deadline_unix':begun+10800,
       'user_authorized_scope':'Fast first one-of-four selective-reencoding Marine pilot; skip smoke/SHA/review; GPU2/3 only',
       'manifest':parent['manifest'],'row_ids':parent['row_ids'],'normalization':norm,'caches':caches,
       'cache_identity_scope':'Reuse existing bound parents, exact declared row order/header. No fresh payload digest validation.',
       'encoder':parent['official_config'],'encoder_frozen':True,'seed':0,
       'structure':'384x16x16 global -> bilinear32 base; one of four 384x16x16 local fields; 64d four-head cross attention into all global tokens, gated zero-initialized residual, fixed sine edge taper; original1x1 12-class head',
       'controls':['replace: raw selected local block, no new fusion parameters','capacity: identical Fusion parameters fed global base crop instead of local features'],
       'fits':{'each_epochs':100,'each_updates':6300,'batch_size':16,'AdamW_lr':.001,'weight_decay':.0001,
               'route_plan':'Same prewritten per-image/per-epoch choices: uniform five actions none/TL/TR/BL/BR; private image-order seed0',
               'primary':'Terminal100 models; per-epoch random-route validation best is auxiliary only'},
       'selector':{'fit':800,'internal_check':200,'split_seed':37,'seed':7,'epochs':100,'batch':64,'updates':1300,
                   'architecture':'quadrant mean384 + whole mean384 + position2 -> Linear64 GELU Linear1 shared across four candidates',
                   'optimizer':'AdamW .001 wd.0001; normalized signed gain SmoothL1',
                   'gain':'Whole-native-image unweighted12-class CE(no additional encoding) - CE(one region); frozen terminal100 fusion/head',
                   'fixed_control':'Single region with largest average gain over fit800; never selected on development GT',
                   'random_control':'One fixed private-seed random region per development image',
                   'uncertainty':'Highest mean softmax entropy of base-head32 logits within quadrant',
                   'all_negative':'Still select one region under fixed exactly-one budget; no new rejection threshold',
                   'evaluation_GT':'Only final metric, loss-oracle diagnostic and selector diagnostic; never learned/fixed/random/uncertainty actions'},
       'cost_rows':[by_id[i] for i in cost_ids],
       'cost_reference_heads':{name:cost_parent['heads']['marine'][key] for name,key in [('reference_regrid','regrid224_32'),('reference_tiles','tiles448_32'),('reference_whole','swise448_32')]},
       'cost':'Batch1 50 warmups + same32 original dev images x3; actual encoder paths; after all own training ended',
       'references':{'original_regrid_seed0':str(BASE/'marine_feature_regrid_repeats_v1/seed0/report.json'),
                     'all_tiles_seed0':str(BASE/'marine_native_information_v1/head_interp_tiles/report.json'),
                     'whole448_seed0':str(BASE/'whole448_frozen_reference_v1/head_marine/report.json')},
       'caps':{'wall_deadline_seconds':10800,'fit_each_seconds':3600,'selector':1800,'score':1800,'online_cost':1200,
               'RSS_bytes':32*1024**3,'GPU_reserved_bytes':24*1024**3,'new_disk_bytes':8*1024**3},
       'stop':'Real software/nonfinite/resource errors preserve files and stop; no automatic retry/config/seed expansion',
       'source_utf8':Path(__file__).read_text(),'no_SHA_digests_or_smoke_tests':True,
       'novelty':'Candidate implementation only, no new-mechanism or publication acceptance claim',
       'inherited_preparation_lookup_errors':['One rg lookup used an absent per-arm cache_manifest; no model process involved']}
    write(RUN/'protocol.json',p)
    generator=np.random.default_rng(1)
    np.save(RUN/'training_actions.npy',generator.integers(-1,4,(100,1000),dtype=np.int64))
    np.save(RUN/'validation_random_actions.npy',np.random.default_rng(2).integers(0,4,251,dtype=np.int64))
    order=np.random.default_rng(37).permutation(1000)
    np.savez(RUN/'selector_split.npz',fit=order[:800],check=order[800:])
    (RUN/'logs').mkdir()
    jobs=[]
    children={}
    def launch(phase,gpu,cap,kind=None):
        entry=admission(gpu)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','XFORMERS_DISABLED':'1'}
        cmd=[PYTHON,'-B','-u','-m','runs.selective_reencoding',phase]
        if kind:cmd+=['--kind',kind]
        key=kind or phase;log=(RUN/'logs'/f'{key}.log').open('w')
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[key]=(child,log,time.time(),min(cap,p['deadline_unix']-time.time()),entry)
        write(RUN/'queue_state.json',{'status':'RUNNING','supervisor_PID':os.getpid(),'active':{k:v[0].pid for k,v in children.items()},'completed':jobs})
    def finish(key):
        child,log,start,cap,entry=children[key]
        try:code=child.wait(timeout=max(0.,cap-(time.time()-start)))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        log.close();del children[key]
        row={'phase':key,'worker_PID':child.pid,'physical_GPU':entry['physical_GPU'],'actual_wait_exit':code,
             'wall_seconds':time.time()-start,'admission':entry}
        jobs.append(row)
        with (RUN/'compute_ledger.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if code!=0:raise RuntimeError(f'{key} actual child exit {code}; see logs/{key}.log')
    try:
        launch('fit',2,3600,'full');launch('fit',3,3600,'capacity')
        write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'workers':{k:v[0].pid for k,v in children.items()},'protocol':str(RUN/'protocol.json'),'started_unix':time.time(),'no_new_smoke_SHA_review':True})
        # Finish the two independent fits in actual completion order.
        while children:
            for key in list(children):
                child,log,start,cap,entry=children[key]
                if child.poll() is not None or time.time()-start>=cap:
                    finish(key)
                    if key=='capacity':launch('fit',3,3600,'replace')
                    if key=='full':launch('selector',2,1800)
            time.sleep(1)
        for phase,cap in [('score',1800),('cost',1200)]:
            launch(phase,2,cap);finish(phase)
        disk=sum(q.stat().st_size for q in RUN.rglob('*') if q.is_file())
        if disk>8*1024**3:raise RuntimeError('New disk cap exceeded')
        closure={'status':'COMPLETE_FIXED_MARINE_ONE_REGION_PILOT','supervisor_PID':os.getpid(),
                 'actual_child_waits':jobs,'child_wall_seconds':sum(j['wall_seconds'] for j in jobs),
                 'end_to_end_pipeline_wall_seconds':time.time()-begun,'new_disk_bytes':disk,
                 'results':str(RUN/'results.json'),'verification':str(RUN/'verification.json'),
                 'selector':str(RUN/'selector_report.json'),'online_cost':str(RUN/'cost_results.json'),
                 'no_next_experiment_queue':True,'scope':'Single seed, Marine development population; no novelty/generalization acceptance'}
        write(RUN/'final_closure.json',closure);write(RUN/'queue_state.json',{'status':'COMPLETE','completed':jobs})
        write(RUN/'writing_handoff.json',closure)
    except BaseException:
        # Stop the other specifically owned worker when one genuine stage fails.
        for key,(child,log,start,cap,entry) in list(children.items()):
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            code=child.wait();log.close()
            jobs.append({'phase':key,'worker_PID':child.pid,'actual_wait_exit':code,'wall_seconds':time.time()-start,'cancelled_after_failure':True})
        write(RUN/'failure_closure.json',{'status':'STOPPED_REAL_ERROR_NO_AUTOMATIC_RETRY','traceback':traceback.format_exc(),'actual_child_waits':jobs})
        write(RUN/'queue_state.json',{'status':'STOPPED','completed':jobs})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pipeline','fit','selector','score','cost']);parser.add_argument('--kind',choices=['full','capacity','replace'])
    args=parser.parse_args();runtime()
    if args.phase=='pipeline':pipeline()
    else:
        try:
            if args.phase=='fit':fit(args.kind)
            elif args.phase=='selector':selector_stage()
            elif args.phase=='score':score()
            else:cost()
        except BaseException:
            write(RUN/(f'failure_{args.kind or args.phase}.json'),{'PID':os.getpid(),'traceback':traceback.format_exc()})
            raise
