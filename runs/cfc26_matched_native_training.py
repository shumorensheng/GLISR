"""Matched native-window/YOLO/teacher adaptations of the retained CFC26 models."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import pickle
import random
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from sklearn.ensemble import HistGradientBoostingClassifier

from runs import cfc26_glisr_native128 as reference
from runs import cfc26_observation_quality as quality
from runs.glisr_supplement_variants import GlobalMemoryGSR
from runs.cfc26_glisr_teacher_head import TeacherPyramidHead
from runs.cfc26_glisr_yolo_head import Conv

BASE = reference.BASE
ROOT = Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))
RUN = BASE/'cfc26_matched_native_training_v1'
METHODS = ['Rtau', 'R256', 'A1', 'A2', 'A3', 'A4', 'B2',
           'same_source_DINO', 'SigLIP2', 'FeatSharp', 'HSMLA', 'Adapter']
SPARSE = ['Rtau', 'R256', 'A1', 'A2', 'A3', 'A4', 'B2']
write = reference.rgb.native.write
tracking = reference.rgb.tracking


class MatchedTeacherHead(TeacherPyramidHead):
    def __init__(self, channels, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            super().__init__(False)
            if channels == 768:
                self.core.p3 = Conv(768, 192, 1)
                self.core.p4 = Conv(768, 384, 1)
                self.core.p5 = Conv(768, 768, 1)

    def forward(self, field, targets=None):
        if self.training and self.teacher_images.shape[-1] == 128:
            self.teacher_images = F.interpolate(self.teacher_images, (256, 256),
                mode='bilinear', align_corners=False, antialias=True)
        if field.shape[-2:] != (32, 32):
            field = F.interpolate(field, (32, 32), mode='bilinear', align_corners=False)
        return super().forward(field, targets)


def prepare():
    RUN.mkdir(exist_ok=False)
    parent = json.loads((BASE/'glisr_supplement_20261104_v1/protocol.json').read_text())
    bindings = json.loads((BASE/'cfc26_structural_ablations_v1/source_bindings.json').read_text())['records']
    jobs = []
    # Release structural controls and a direct baseline before the remaining repeats.
    priority = [('R256', 0), ('A1', 0), ('A2', 0), ('A3', 0),
                ('A4', 0), ('same_source_DINO', 0), ('SigLIP2', 0), ('Rtau', 1)]
    order = priority+[(m, s) for s in range(3) for m in METHODS
                      if (m, s) not in priority and (m, s) != ('Rtau', 0)]+[('Rtau', 0)]
    for method, seed in order:
        if method in ('R256', 'A1', 'A2', 'A3', 'A4'):
            binding = next(b for b in bindings if b['method'] == method and b['seed'] == seed)
            source = binding['path']; state_key = binding['state_key']
        elif method == 'SigLIP2':
            source = str(BASE/'siglip2_frozen_baseline_v1/model_assets/model.safetensors')
            state_key = 'official_vision_loader'
        else:
            source = parent['reuse_source_paths'][method][str(seed)]
            state_key = 'plain_state' if method == 'Rtau' else 'encoder' if method == 'same_source_DINO' else 'model'
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        jobs.append({'id': f'{method}_s{seed}', 'method': method, 'seed': seed,
                     'source': source, 'source_state_key': state_key,
                     'status': 'REUSE_TRAINING_WITH_MATCHED_AUXILIARY' if (method, seed) == ('Rtau', 0) else 'NEW',
                     'new_target_updates': 0 if (method, seed) == ('Rtau', 0) else 26250})
    shared = RUN/'plans'; shared.mkdir()
    plans = {}
    for tile, epochs, path in [(256, 20, BASE/'cfc26_glisr_teacher_pyramid_v1/shared_plan/augmentation_plan.npz'),
                              (128, 10, reference.RUN/reference.ARM/'augmentation_plan.npz')]:
        with np.load(path) as old:
            for seed in range(3):
                generator = torch.Generator().manual_seed(seed)
                order = np.stack([torch.randperm(7000, generator=generator).numpy() for _ in range(epochs)])
                if seed == 0 and not np.array_equal(order, old['order']):
                    raise RuntimeError('Seed0 reference order differs')
                target = shared/f'native{tile}_s{seed}.npz'
                np.savez(target, order=order.astype(np.int32), xy=old['xy'], flip=old['flip'], selected_GT=old['selected_GT'])
                plans[f'{tile}_{seed}'] = str(target)
    old = json.loads((BASE/'cfc26_observation_calibration_v1/protocol.json').read_text())
    protocol = {
        'status': 'FIXED_BEFORE_FIRST_MODEL_FORWARD', 'created_unix': time.time(),
        'jobs': jobs, 'encoder': parent['encoder'], 'methods': METHODS, 'seeds': [0, 1, 2],
        'new_training_jobs': 35, 'new_target_updates': 918750, 'new_source_training': 0,
        'data': str(BASE/'cfc26_tracking_10000_v1'), 'frames': {'train': 7000, 'val': 1000, 'test': 2000},
        'stages': [{'tile': 256, 'epochs': 20, 'updates': 17500}, {'tile': 128, 'epochs': 10, 'updates': 8750}],
        'plans': plans,
        'sampling': 'Each original train frame once/epoch; same existing reference GT-centered/GT-empty crop and flip per frame/epoch for all methods and repeats. Private order seeded 0/1/2, identical across methods of a seed. Crop/flip plans deliberately fixed across repeats; source initialization, head initialization, order and runtime RNG vary. No source or target GT changes.',
        'initialization': 'Existing corresponding method/source seed, no historical CFC26 target weights. Restore official RGB DINO patch projection for every DINO-derived method, matching reference stage256. Head neck/Detect from same author public weights; same seeded P3/P4/P5 initialization where shapes agree. Seed0 384-channel head reuses exact retained original head initializer. Stage128 starts stage256 terminal, resets optimizers as reference.',
        'input': 'Stored RGB native windows, stride floor(3*tile/4)+last origin at inference. Independent bilinear-AA/ImageNet224 global and448 local views for DINO methods. SigLIP2 uses RGB PIL bilinear224 and official .5/.5 normalization, native768x14x14 field bilinear32 before its 768-input YOLO projections. Same teacher raw RGB/255,128 teacher crop resized256. SigLIP remains a different pretrained backbone.',
        'field_interfaces': 'GLISR variants use original memory/gate definitions and actual selected-token computation. DirectDINO/Adapter keep original global-only field. FeatSharp keeps its existing official-adapted fusion but RGB callbacks retain all3 channels and receive actual448 guidance. HSMLA keeps its paper-equation block and regularization, with actual448 DINO field. Original model-specific architecture retained.',
        'teacher': 'Same frozen author CNN0..9, taps4/6/9; train-only feature loss .25 mean(normalized pyramid MSE), same pretrained student YOLO neck/Detect, teacher absent from inference/checkpoints/optimizers.',
        'training': 'Batch8 FP32/noAMP/TF32off; whole encoder+task head+existing modules fullFT; mask excluded. BodyAdamW1e-5, headSGD.001/.9/.0005, moduleAdamW1e-4/WD1e-4. Each stage200warmup/cosine1percentfloor, separate FP64-norm clipping cap1. Preserve HSMLA budget/smooth regularizer. No HPO or direction-based stops.',
        'validation': 'Every5epochs with fixed raw OC-SORT, same original stage256 and stage128 tracker settings. Stage128 earliest strict maximum validation HOTA chooses auxiliary best. Stage256 test omitted because its outputs never select/init stage128. Final terminal10 primary and val-selected best auxiliary only; no test-based checkpoint selection.',
        'postprocess': {'association': old['association']['GLISR_native128_terminal'],
                        'confirmation': old['confirmations']['GLISR_native128_terminal'],
                        'location_priors': old['location_priors'], 'quality_threshold': .30,
                        'estimator': {**old['estimator'], 'max_leaf_nodes': 31}},
        'quality_training': 'For EACH method/seed/selected endpoint, infer own7000train frames, build same38features and originalGT one-to-one Hungarian IoU>=.5 observation labels; fit own31leaf100iteration classifier. No validation/test labels in fit; common fixed.30, no threshold sweep. If best=terminal reuse same actual endpoint. Rtau seed0 terminal reuses exact already completed neural predictions/quality model; best auxiliary receives its own matched train-only scorer.',
        'outputs': 'Original TrackEval HOTA(primary)/IDF1/MOTA/FP/FN/IDs; official LOI count on motion-qualified complete tracks before quality filtering. Store original native detections and both actual outputs. One unified report, all seed directions retained, historical strong author original result retained.',
        'reuse_Rtau0': {'run': str(reference.RUN/reference.ARM), 'terminal_profile': str(BASE/'cfc26_tracking_best_available_v1/best_observation_profile.json')},
        'budgets': {'physical_GPUs': [0, 1, 2, 3], 'workers_per_GPU': 2, 'minimum_free_MiB': 24576,
                    'global_seconds': 172800, 'worker_seconds': 28800, 'RSS_each_GiB': 64,
                    'GPU_reserved_each_GiB': 40, 'new_disk_GiB': 512, 'CPU_threads_each': 4},
        'planning': 'Historical single GLISR stage256+128 cost~9376s before additional full-train quality inference; 35new jobs on four cards with up to two workers/card. Rough conditional24-48h, shared-load/model costs unmeasured, not a completion guarantee. Normal counted prefix reports actual speed; no forecast-only stop.',
        'failure': 'Actual wall/RSS/GPU/numeric/code failure visible with original latest/best/logs retained. No LR/epoch/threshold/seed fallback or automatic retry. First failed worker stops releasing pending jobs; already running authorized jobs finish or meet their own guards. Never touch unrelated processes.',
        'scope': 'Same local repeatedly developed dataset. Public fish-task pretraining/teacher supplied to every method. Existing source-model/source-training differences and HSMLA formula-adaptation identity retained; no world-SOTA/encoder-only causal or fresh blind-generalization claim.',
    }
    write(RUN/'protocol.json', protocol)
    write(RUN/'run_manifest.json', {'jobs': jobs, 'new_target_updates': 918750})
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')


def prepare_shared_pixels():
    folder = RUN/'shared_pixels'; folder.mkdir(exist_ok=False)
    rows = tracking.load_rows('train')
    shapes = np.asarray([r['actual_shape_HWC'] for r in rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(shapes.prod(1))]
    pixels = np.lib.format.open_memmap(folder/'train_rgb.npy', mode='w+', dtype=np.uint8,
                                      shape=(int(offsets[-1]),))
    for i, row in enumerate(rows):
        with Image.open(row['path']) as image:
            current = np.array(image.convert('RGB'), copy=True)
        if tuple(current.shape) != tuple(shapes[i]):
            raise RuntimeError(f'Native image geometry mismatch: {row["row_id"]}')
        pixels[offsets[i]:offsets[i+1]] = current.reshape(-1)
        if (i+1) % 500 == 0:
            print('shared RGB cache', i+1, len(rows), flush=True)
    pixels.flush(); del pixels
    np.savez(folder/'index.npz', offsets=offsets, shapes=shapes,
             row_ids=np.asarray([r['row_id'] for r in rows]))
    write(folder/'completion.json', {'status': 'COMPLETE', 'frames': len(rows),
          'payload_bytes': int(offsets[-1]), 'definition': 'Lossless original uint8 RGB bytes in shared read-only mmap; no resizing, normalization, GT or sampling change'})


def guard(deadline):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    reserved = torch.cuda.max_memory_reserved()
    if time.time() > deadline:
        raise TimeoutError('Actual matched native job deadline')
    if rss > 64*1024**3 or reserved > 40*1024**3:
        raise RuntimeError(f'Actual matched native resource limit RSS={rss}, GPU={reserved}')
    return {'RSS_bytes': rss, 'GPU_reserved_bytes': reserved}


class SharedPixels:
    """Canonical RGB frames, filled once as needed while the bulk writer runs."""
    def __init__(self, rows):
        self.rows = rows
        with np.load(RUN/'shared_pixels/index_live.npz') as index:
            if index['row_ids'].tolist() != [r['row_id'] for r in rows]:
                raise RuntimeError('Shared native RGB row identities differ')
            self.offsets, self.shapes = index['offsets'].copy(), index['shapes'].copy()
        self.bank = np.load(RUN/'shared_pixels/train_rgb.npy', mmap_mode='r+')
        self.ready = np.load(RUN/'shared_pixels/ready.npy', mmap_mode='r+')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        view = self.bank[self.offsets[index]:self.offsets[index+1]].reshape(tuple(self.shapes[index]))
        if not self.ready[index]:
            with Image.open(self.rows[index]['path']) as image:
                view[:] = np.asarray(image.convert('RGB'))
            # Concurrent fills only write identical immutable source bytes.
            self.ready[index] = 1
        return view


def construct(job, p):
    method, seed = job['method'], job['seed']
    if method in SPARSE:
        model = GlobalMemoryGSR(p['encoder'], 'Rtau' if method == 'B2' else method, seed)
    elif method == 'same_source_DINO':
        from runs.gsr_same_source_dino import DirectDINO
        model = DirectDINO(p['encoder'])
    elif method == 'SigLIP2':
        from runs.gsr_siglip2_fullft import DirectSigLIP2
        model = DirectSigLIP2()
    elif method == 'FeatSharp':
        from runs.featsharp_marine_matched import Model
        model = Model(p['encoder'], seed)
        model._matched_attention_batch = None
    elif method == 'HSMLA':
        from runs.hsmla_marine_matched import Model
        model = Model(p['encoder'], seed)
    else:
        model = reference.rgb.native.AdapterModel(p['encoder'], seed)
    if method != 'SigLIP2':
        saved = torch.load(job['source'], map_location='cpu', weights_only=False)
        if method == 'same_source_DINO':
            model.encoder.load_state_dict(saved['encoder'], strict=True)
        else:
            model.load_state_dict(saved if job['source_state_key'] == 'plain_state' else saved[job['source_state_key']], strict=True)
        del saved
        original = torch.load(p['encoder']['weights_path'], map_location='cpu', weights_only=True)
        projection = nn.Conv2d(3, 384, 14, stride=14)
        projection.load_state_dict({'weight': original['patch_embed.proj.weight'], 'bias': original['patch_embed.proj.bias']}, strict=True)
        model.encoder.backbone.patch_embed.proj = projection
        reference.rgb.native.reader.configure_encoder(model.encoder, 'e2e')
        del original
    head = MatchedTeacherHead(768 if method == 'SigLIP2' else 384, seed)
    if seed == 0 and method != 'SigLIP2':
        saved = torch.load(BASE/'cfc26_glisr_yolo_adaptation_v1/initialization.pt', map_location='cpu', weights_only=False)
        head.load_state_dict(saved['head'], strict=True)
        del saved
    return model.cuda(), head.cuda()


def views(crops, method):
    if method == 'SigLIP2':
        images = np.stack([np.asarray(Image.fromarray(c).resize((224, 224), Image.Resampling.BILINEAR)) for c in crops])
        return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().cuda().float()/127.5-1, None
    return reference.rgb.views(crops)


def field(model, method, global_view, local_view):
    if method in SPARSE:
        return model.field(global_view, high_resolution=local_view)
    if method == 'FeatSharp':
        # Upstream cached the first attention batch; validation can be larger.
        from featsharp.upsampler_modules.featsharp import AlibiAttention
        if model._matched_attention_batch != len(global_view):
            for module in model.fusion.modules():
                if isinstance(module, AlibiAttention):
                    module._initial_bs = None
                    module.kernel_block_mask = None
            model._matched_attention_batch = len(global_view)
        def tiles(packed, return_summary=False):
            return model.tile_generator(torch.cat((global_view, packed[len(global_view):]), 0), return_summary)
        return model.fusion(local_view, tiles)[1]
    if method == 'HSMLA':
        tokens = model.encoder(local_view)['patch_tokens']
        return model.fusion(tokens.transpose(1, 2).reshape(len(global_view), 384, 32, 32))
    return model.field(global_view, None, 'detection')


def optimizers(model, head, method):
    body = model.vision if method == 'SigLIP2' else model.encoder
    result = [reference.rgb.native.reader.body_optimizer(body),
              torch.optim.SGD(head.parameters(), lr=.001, momentum=.9, weight_decay=.0005)]
    if method not in ('same_source_DINO', 'SigLIP2'):
        extra = [v for v in model.fusion.parameters() if v.requires_grad]
        if method in SPARSE:
            extra += model.adapter_parameters()
        result.append(torch.optim.AdamW(extra, lr=.0001, weight_decay=.0001))
    return result


def save(folder, name, model, head, state, opts):
    value = {'model': model.state_dict(), 'head': head.state_dict(), 'progress': dict(state),
             'protocol': str(RUN/'protocol.json'), 'input_channels': 3}
    if name == 'latest':
        value.update(optimizers=[o.state_dict() for o in opts], torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
    temp = folder/f'{name}.tmp'; torch.save(value, temp); temp.replace(folder/f'{name}.pt')


@torch.no_grad()
def infer(model, head, method, tile, split, folder, deadline):
    folder.mkdir(parents=True)
    model.eval(); head.eval()
    rows = tracking.load_rows(split)
    predictions, calls, windows = [], 0, 0
    begun = time.monotonic(); teacher_before = head.teacher_samples
    with (folder/'input_transforms.jsonl').open('w') as log:
        for i, row in enumerate(rows):
            with Image.open(row['path']) as image:
                pixels = np.array(image.convert('RGB'), copy=True)
            h, w = row['actual_shape_HWC'][:2]
            positions = [(x, y) for y in reference.rgb.origins(h, tile) for x in reference.rgb.origins(w, tile)]
            all_boxes, all_scores, all_counts = [], [], []
            for start in range(0, len(positions), 16):
                selected = positions[start:start+16]
                g, local = views([pixels[y:y+tile, x:x+tile] for x, y in selected], method)
                detected = head(field(model, method, g, local))
                for det, (x, y) in zip(detected, selected):
                    all_boxes.append(det['boxes']*(tile/224)+det['boxes'].new_tensor([x, y, x, y]))
                    all_scores.append(det['scores'])
                if method in SPARSE:
                    all_counts.extend(model.last_counts.tolist())
                calls += 1; windows += len(selected); guard(deadline)
            boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
            keep = nms(boxes, scores, .25)[:300]
            predictions.append({'boxes': boxes[keep].cpu().numpy(), 'scores': scores[keep].cpu().numpy(),
                                'labels': np.ones(len(keep), dtype=np.int64)})
            log.write(json.dumps({'row_id': row['row_id'], 'native_HW': [h, w], 'tile': tile,
                                  'origins_xy': positions, 'K': all_counts if method in SPARSE else None})+'\n')
            if (i+1) % 200 == 0:
                print('infer', folder.name, split, i+1, len(rows), flush=True)
    if head.teacher_samples != teacher_before:
        raise RuntimeError('Teacher executed during student inference')
    target = folder/'predictions.pt'
    torch.save({'row_ids': [r['row_id'] for r in rows], 'native_detections': predictions, 'split': split}, target)
    write(folder/'inference_receipt.json', {'frames': len(rows), 'calls': calls, 'views': windows,
          'seconds': time.monotonic()-begun, 'teacher_inference_forwards': 0, **guard(deadline)})
    return target


def association(path, split, p):
    flexible = quality.quality.location.motion.flexible
    rows, detections, _ = flexible.load_inputs('current', split, {'detectors': {'current': {split: str(path)}}})
    tracks = flexible.associate(rows, detections, p['association'])
    tracks, _ = quality.quality.location.motion.confirmation.confirmed(rows, tracks, p['confirmation'])
    tracks = quality.quality.location.filtered(rows, tracks,
        quality.quality.location.motion.descriptors(rows, tracks), 'location', p)
    return rows, detections, tracks


def train_stage(job, p, model, head, cache, spec, folder, deadline):
    folder.mkdir(); method, seed = job['method'], job['seed']
    tile, epochs, total = spec['tile'], spec['epochs'], spec['updates']
    with np.load(p['plans'][f'{tile}_{seed}']) as plan:
        order, xy, flip = plan['order'].copy(), plan['xy'].copy(), plan['flip'].copy()
    rows = tracking.load_rows('train'); opts = optimizers(model, head, method)
    state = {'epoch': 0, 'updates': 0, 'best_epoch': None, 'best_HOTA': -1., 'tile': tile}
    start_epoch = 0
    if tile == 256 and 'resume_checkpoint' in job:
        saved = torch.load(job['resume_checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        head.load_state_dict(saved['head'], strict=True)
        for optimizer, value in zip(opts, saved['optimizers']):
            optimizer.load_state_dict(value)
        state = saved['progress']; start_epoch = state['epoch']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        write(folder/'resume_receipt.json', {'checkpoint': job['resume_checkpoint'],
              'restored_epoch': start_epoch, 'restored_updates': state['updates'],
              'next_update': state['updates']+1, 'optimizer_and_all_saved_RNG_restored': True,
              'scope': 'Resume persisted epoch4; failed unsaved epoch5 is extra physical cost, not seamless update4375 continuation'})
        del saved
    began = time.monotonic(); histogram = Counter(); validation = []
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(start_epoch, epochs):
            model.train(); head.train()
            for start in range(0, len(rows), 8):
                ids = order[epoch, start:start+8].tolist(); crops, targets = [], []
                for index in ids:
                    x, y = xy[epoch, index]; crop = cache[index][y:y+tile, x:x+tile]
                    boxes = np.asarray([a['bbox'] for a in rows[index]['annotations']], dtype=np.float32).reshape(-1, 4)
                    boxes[:, 2:] += boxes[:, :2]; boxes -= np.asarray([x, y, x, y], dtype=np.float32)
                    boxes = np.clip(boxes, 0, tile)
                    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
                    if flip[epoch, index]:
                        crop = crop[:, ::-1].copy(); boxes[:, [0, 2]] = tile-boxes[:, [2, 0]]
                    crops.append(crop)
                    targets.append({'boxes': torch.from_numpy(boxes*(224/tile)).cuda(),
                                    'labels': torch.ones(len(boxes), dtype=torch.int64, device='cuda')})
                step = state['updates']+1
                multi = step/200 if step <= 200 else .01+.99*(1+math.cos(math.pi*(step-200)/(total-200)))/2
                for opt, peak in zip(opts, [1e-5, .001, .0001]):
                    for group in opt.param_groups:
                        group['lr'] = peak*multi
                    opt.zero_grad(set_to_none=True)
                g, local = views(crops, method)
                head.teacher_images = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous().cuda().float()/255
                feature = field(model, method, g, local)
                loss = sum(head(feature, targets).values())
                if method == 'HSMLA':
                    loss = loss+model.fusion.regularization_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite matched native training loss')
                loss.backward()
                norms = reference.rgb.native.clip_gradients(method, opts)
                for opt in opts:
                    opt.step()
                state.update(epoch=epoch+1, updates=step)
                if method in SPARSE:
                    histogram.update(model.last_counts.tolist())
                if step == 1:
                    witness = {'counted_update': 1, 'method': method, 'seed': seed,
                        'source': job['source'], 'row_ids': [rows[i]['row_id'] for i in ids],
                        'input_shape': list(g.shape), 'field_shape': list(feature.shape), 'native_tile': tile,
                        'gradient_norms': norms, 'head_parameters': sum(v.numel() for v in head.parameters()),
                        'model_parameters': sum(v.numel() for v in model.parameters()),
                        'teacher': head.teacher_report(), **guard(deadline)}
                    if method in SPARSE:
                        witness.update(K=model.last_counts.tolist(), sparse=model.first_sparse_witness,
                                       memory_blocks=model.interaction_blocks, gate=model.fusion.use_gate)
                    write(folder/'first_counted_update.json', witness)
                if step == 1 or step % 100 == 0:
                    record = {**state, 'loss': float(loss.detach()), 'gradient_norms': norms,
                              'seconds': time.monotonic()-began, **guard(deadline)}
                    log.write(json.dumps(record)+'\n'); log.flush(); print(json.dumps(record), flush=True)
            if (epoch+1) % 5 == 0:
                path = infer(model, head, method, tile, 'val', folder/f'val_epoch{epoch+1:02}', deadline)
                pp = {'association': reference.teacher.TRACKER if tile == 256 else reference.TRACKER,
                      'confirmation': p['postprocess']['confirmation']}
                flexible = quality.quality.location.motion.flexible
                val_rows, det, _ = flexible.load_inputs('v', 'val', {'detectors': {'v': {'val': str(path)}}})
                val_tracks = flexible.associate(val_rows, det, pp['association'])
                metrics = tracking.score(val_rows, val_tracks)
                validation.append({'epoch': epoch+1, 'tracking': metrics, 'predictions': str(path)})
                write(folder/f'val_epoch{epoch+1:02}/metrics.json', metrics)
                if metrics['HOTA'] > state['best_HOTA']:
                    state.update(best_HOTA=metrics['HOTA'], best_epoch=epoch+1)
                    save(folder, 'best', model, head, state, opts)
            save(folder, 'latest', model, head, state, opts)
        if state['updates'] != total:
            raise RuntimeError('Fixed stage update count incomplete')
    write(folder/'results.json', {'status': 'COMPLETE', 'state': state, 'validation': validation,
          'training_K': dict(histogram), 'seconds': time.monotonic()-began,
          'teacher': head.teacher_report(), **guard(deadline)})
    return state


def endpoint(job, p, model, head, name, train_path, test_path, val_path, folder, deadline):
    folder.mkdir()
    pp = p['postprocess']; quality.RUN = folder
    rows, detections, tracks = association(train_path, 'train', pp)
    x, _ = quality.observation_features(rows, detections, tracks, False, 'train', deadline)
    labels = []
    for row, track in zip(rows, tracks):
        matched = np.zeros(len(track['ids']), dtype=bool)
        if len(track['ids']) and row['annotations']:
            gt = np.asarray([a['bbox'] for a in row['annotations']], dtype=np.float64)
            gt[:, 2:] += gt[:, :2]
            overlap = tracking.box_iou(gt, track['boxes'])
            i, j = linear_sum_assignment(np.where(overlap >= .5, 1-overlap, 1e6))
            matched[j[overlap[i, j] >= .5]] = True
        labels.extend(matched.tolist())
    y = np.asarray(labels, dtype=bool)
    np.save(folder/'train_labels.npy', y)
    estimator = HistGradientBoostingClassifier(**{**pp['estimator'], 'random_state': job['seed']}).fit(x, y)
    with (folder/'quality_estimator.pkl').open('wb') as f:
        pickle.dump(estimator, f)
    write(folder/'fit_receipt.json', {'train_observations': len(y), 'positive_observations': int(y.sum()),
          'features': x.shape[1], 'iterations': estimator.n_iter_, 'validation_test_labels_used': False})
    results = {}
    for split, path in [('val', val_path), ('test', test_path)]:
        rows, det, tracks = association(path, split, pp)
        x, _ = quality.observation_features(rows, det, tracks, False, split, deadline)
        scores = estimator.predict_proba(x)[:, 1]
        filtered = quality.retained(rows, tracks, scores, .30)
        counter = folder/f'{split}_count'; counter.mkdir()
        counts = quality.quality.location.motion.flexible.count_tracks(rows, tracks, counter, split)
        metrics = tracking.score(rows, filtered)
        torch.save({'row_ids': [r['row_id'] for r in rows], 'tracking_output': filtered,
                    'count_input_tracks': tracks, 'quality_scores': scores, 'predictions': str(path)}, folder/f'{split}_outputs.pt')
        results[split] = {'tracking': metrics, 'official_LOI_counts': counts, 'predictions': str(path)}
        guard(deadline)
    result = {'status': 'COMPLETE', 'job': job['id'], 'endpoint': name, 'results': results,
              'threshold': .30, 'own_train_quality_model': str(folder/'quality_estimator.pkl')}
    write(folder/'results.json', result)
    return result


def worker(job_id, deadline, output_id=None):
    p = json.loads((RUN/'protocol.json').read_text()); job = next(j for j in p['jobs'] if j['id'] == job_id)
    control_path = RUN/'dispatch_control.json'
    if control_path.exists():
        control = json.loads(control_path.read_text())
        if job_id in control['resume_checkpoints'] and output_id == control['retry_outputs'][job_id]:
            job['resume_checkpoint'] = control['resume_checkpoints'][job_id]
    folder = RUN/(output_id if output_id is not None else job_id); folder.mkdir()
    begun = time.monotonic()
    try:
        reference.rgb.native.setup(job['seed'])
        # The reused feature builder's old 3GiB process cap belonged to a CPU-only
        # unit; this process has the declared 64GiB training/cache budget.
        quality.guard = lambda end: guard(end)['RSS_bytes']
        write(folder/'launch.json', {'PID': os.getpid(), 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                                   'logical_GPU': 0, 'deadline_unix': deadline, 'job': job})
        model, head = construct(job, p)
        if job['status'] == 'NEW':
            rows = tracking.load_rows('train')
            cache = SharedPixels(rows)
            write(folder/'cache_receipt.json', {'frames': len(cache), 'cache_kind': 'shared_readonly_mmap',
                  'shared_backing_bytes': int(cache.offsets[-1]), 'private_image_payload_bytes': 0,
                  'construction': 'Canonical uint8 RGB filled on demand; already-filled frames read without copying', **guard(deadline)})
            for spec in p['stages']:
                state = train_stage(job, p, model, head, cache, spec, folder/f"native{spec['tile']}", deadline)
            del cache
            last = folder/'native128'
            best_epoch = state['best_epoch']
            terminal_test = infer(model, head, job['method'], 128, 'test', folder/'terminal_test', deadline)
            terminal_train = infer(model, head, job['method'], 128, 'train', folder/'terminal_train', deadline)
            terminal_val = last/'val_epoch10/predictions.pt'
            terminal = endpoint(job, p, model, head, 'terminal10', terminal_train, terminal_test, terminal_val, folder/'terminal_quality', deadline)
        else:
            last = reference.RUN/reference.ARM
            state = json.loads((last/'results.json').read_text())
            best_epoch = state['best_validation_epoch']
            profile = json.loads(Path(p['reuse_Rtau0']['terminal_profile']).read_text())
            terminal = {'status': 'REUSE_COMPLETE', 'endpoint': 'terminal10',
                        'results': {'test': {'tracking': profile['expected_test']['tracking'],
                                    'official_LOI_counts': profile['expected_test']['official_LOI_counts']}},
                        'profile': p['reuse_Rtau0']['terminal_profile']}
        if best_epoch == 10:
            best = terminal
        else:
            saved = torch.load(last/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'], strict=True); head.load_state_dict(saved['head'], strict=True)
            del saved
            train_path = infer(model, head, job['method'], 128, 'train', folder/'best_train', deadline)
            if job['status'] == 'NEW':
                test_path = infer(model, head, job['method'], 128, 'test', folder/'best_test', deadline)
            else:
                test_path = last/'test_best/predictions.pt'
            best = endpoint(job, p, model, head, f'best_epoch{best_epoch}', train_path, test_path,
                            last/f'val_epoch{best_epoch:02}/predictions.pt', folder/'best_quality', deadline)
        result = {'status': 'COMPLETE', 'job': job, 'terminal': terminal, 'best': best,
                  'best_epoch': best_epoch, 'best_terminal_shared': best_epoch == 10,
                  'new_target_updates': job['new_target_updates'],
                  'seconds': time.monotonic()-begun, **guard(deadline)}
        write(folder/'results.json', result)
        print(json.dumps({'COMPLETE': job_id, 'seconds': result['seconds']}), flush=True)
    except Exception:
        write(folder/'failure.json', {'job': job, 'PID': os.getpid(), 'traceback': traceback.format_exc(),
                                     'seconds': time.monotonic()-begun})
        raise


def pipeline(reattach=False):
    p = json.loads((RUN/'protocol.json').read_text())
    active, attached = {}, {}
    if reattach:
        original = json.loads((RUN/'launch_receipt.json').read_text())
        previous = json.loads((RUN/'queue_state.json').read_text())
        if Path('/proc', str(previous['supervisor_PID'])).exists():
            raise RuntimeError('Existing supervisor is still present; refuse a duplicate queue')
        begun, deadline = original['started_unix'], original['deadline_unix']
        pending = [next(j for j in p['jobs'] if j['id'] == name) for name in previous['pending']]
        waits = previous['actual_waits']; failed = previous['failed']
        control_path = RUN/'dispatch_control.json'
        if control_path.exists():
            control = json.loads(control_path.read_text())
            acknowledged = control['acknowledged_environment_failures']
            failed = any(w['job'] not in acknowledged and w['actual_wait_exit'] not in (0, None) for w in waits)
            for job_id in control['retry_outputs']:
                retry = dict(next(j for j in p['jobs'] if j['id'] == job_id))
                retry['output_id'] = control['retry_outputs'][job_id]
                if job_id in control['resume_checkpoints']:
                    retry['resume_checkpoint'] = control['resume_checkpoints'][job_id]
                pending = [j for j in pending if j['id'] != job_id]
                if not any(j['job'] == job_id for j in previous['active']) and not (RUN/retry['output_id']/'results.json').exists():
                    pending.insert(0, retry)
        recovery = RUN/'supervision_recovery_v7'; recovery.mkdir(exist_ok=True)
        write(recovery/'prior_queue_state.json', previous)
        write(recovery/'prior_launch_receipt.json', original)
        for slot, record in enumerate(previous['active']):
            job = dict(next(j for j in p['jobs'] if j['id'] == record['job']))
            if 'output_id' in record:
                job['output_id'] = record['output_id']
            if control_path.exists() and job['id'] in control['resume_checkpoints']:
                job['resume_checkpoint'] = control['resume_checkpoints'][job['id']]
            folder = RUN/(job['output_id'] if 'output_id' in job else job['id'])
            launch = json.loads((folder/'launch.json').read_text())
            attached[slot] = (record['PID'], job, record['physical_GPU'], launch['deadline_unix'])
        write(recovery/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'started_unix': time.time(),
              'original_supervisor_PID': previous['supervisor_PID'], 'original_exit_code': None,
              'original_foreground_wait': 'Unavailable: exec session66149 no longer exists; no exit-code inference',
              'unchanged_global_deadline_unix': deadline, 'attached_existing_workers': previous['active'],
              'no_training_replay': True, 'maximum_total_workers': 36,
              'new_worker_image_cache': 'Shared read-only mmap; no private30GB image cache',
              'minimum_available_host_RAM_GiB': 32})
    else:
        begun = time.time(); deadline = begun+p['budgets']['global_seconds']
        pending, waits, failed = list(p['jobs']), [], False
        write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'started_unix': begun,
              'deadline_unix': deadline, 'GPUs': [0, 1, 2, 3], 'slots_each_GPU': 2,
              'jobs': len(pending), 'new_training_jobs': 35, 'new_target_updates': 918750})
    slots = json.loads((RUN/'dispatch_capacity.json').read_text())['maximum_workers'] if reattach else 8
    disk_checked = time.time()
    while pending or active or attached:
        if time.time()-disk_checked > 60:
            size = sum(f.stat().st_size for f in RUN.rglob('*')
                       if f.is_file() and not f.is_symlink() and f.suffix != '.tmp')
            disk_checked = time.time()
            write(RUN/'disk_observation.json', {'unix': disk_checked, 'bytes_excluding_inflight_tmp': size})
            if size > p['budgets']['new_disk_GiB']*1024**3:
                failed = True
                write(RUN/'disk_failure.json', {'actual_bytes': size, 'cap_GiB': p['budgets']['new_disk_GiB']})
                for child, _, _, _, _, _ in active.values():
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                for pid, _, _, _ in attached.values():
                    if Path('/proc', str(pid)).exists():
                        os.killpg(pid, signal.SIGKILL)
        for slot, (pid, job, gpu, end) in list(attached.items()):
            proc = Path('/proc', str(pid), 'stat')
            running = proc.exists() and proc.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
            if running and time.time() > end:
                os.killpg(pid, signal.SIGKILL)
            if not running:
                folder = RUN/(job['output_id'] if 'output_id' in job else job['id'])
                result = folder/'results.json'
                complete = result.exists() and json.loads(result.read_text())['status'] == 'COMPLETE'
                record = {'job': job['id'], 'PID': pid, 'physical_GPU': gpu,
                          'actual_wait_exit': None, 'wait_scope': 'Non-child after original supervisor loss; wait status unavailable',
                          'completion_evidence': str(result) if complete else None,
                          'status': 'COMPLETE_RESULT_PROCESS_ENDED_WAIT_UNAVAILABLE' if complete else 'FAILED_OR_INCOMPLETE_PROCESS_ENDED'}
                waits.append(record); write(folder/'reattached_completion.json', record)
                failed = failed or not complete
                del attached[slot]
        for slot, (child, log, job, gpu, end, start) in list(active.items()):
            code = child.poll()
            if code is None and time.time() > end:
                os.killpg(child.pid, signal.SIGKILL); code = child.wait()
            if code is not None:
                log.close(); record = {'job': job['id'], 'PID': child.pid, 'physical_GPU': gpu,
                                      'actual_wait_exit': code, 'seconds': time.monotonic()-start}
                folder = RUN/(job['output_id'] if 'output_id' in job else job['id'])
                record['output_id'] = folder.name
                waits.append(record); write(folder/'actual_wait.json', record)
                failed = failed or code != 0
                del active[slot]
        if not failed and time.time() < deadline and (RUN/'shared_pixels/index_live.npz').exists():
            for slot in range(slots):
                if not pending or slot in active or slot in attached:
                    continue
                available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                # A newly spawned interpreter has not yet allocated its model.
                # Reserve real headroom until its counted update exposes usage.
                launching = sum(not (RUN/(v[2]['output_id'] if 'output_id' in v[2] else v[2]['id'])/'native256/first_counted_update.json').exists()
                                and time.monotonic()-v[5] < 180 for v in active.values())
                if available-(launching+1)*6*1024**3 < 32*1024**3:
                    continue
                occupied = [v[3] for v in active.values()]+[v[2] for v in attached.values()]
                gpu = min(range(4), key=lambda g: occupied.count(g))
                info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
                if int(info.split(',')[0]) < p['budgets']['minimum_free_MiB']:
                    continue
                job = pending.pop(0); end = min(deadline, time.time()+p['budgets']['worker_seconds'])
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
                       'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1',
                       'YOLOv5_AUTOINSTALL': 'False', 'WANDB_MODE': 'disabled', 'MPLBACKEND': 'Agg',
                       'XFORMERS_DISABLED': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
                dependencies = BASE/'featsharp_marine_matched_v1/build_dependencies'
                env.update(PATH=str(dependencies/'bin')+':'+os.environ['PATH'],
                           PYTHONPATH=str(dependencies)+':'+str(ROOT), MAX_JOBS='4')
                output_id = job['output_id'] if 'output_id' in job else job['id']
                log = (RUN/f'{output_id}.log').open('w')
                child = subprocess.Popen([sys.executable, '-u', '-m', 'runs.cfc26_matched_native_training',
                    'worker', '--job', job['id'], '--deadline', str(end), '--output-id', output_id], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[slot] = (child, log, job, gpu, end, time.monotonic())
                break
        write(RUN/'queue_state.json', {'updated_unix': time.time(), 'supervisor_PID': os.getpid(),
              'active': [{'job': v[2]['id'], 'PID': v[0].pid, 'physical_GPU': v[3],
                          'output_id': v[2]['output_id'] if 'output_id' in v[2] else v[2]['id']} for v in active.values()]
                       +[{'job': v[1]['id'], 'PID': v[0], 'physical_GPU': v[2], 'reattached': True,
                          'output_id': v[1]['output_id'] if 'output_id' in v[1] else v[1]['id']}
                         for v in attached.values()],
              'pending': [j['id'] for j in pending], 'actual_waits': waits, 'failed': failed})
        if (failed or time.time() > deadline) and not active and not attached:
            break
        if pending or active or attached:
            time.sleep(10)
    results = []
    for j in p['jobs']:
        folder = RUN/j['id']
        if (RUN/'dispatch_control.json').exists():
            outputs = json.loads((RUN/'dispatch_control.json').read_text())['retry_outputs']
            if j['id'] in outputs:
                folder = RUN/outputs[j['id']]
        if (folder/'results.json').exists():
            results.append(json.loads((folder/'results.json').read_text()))
    status = 'COMPLETE' if len(results) == len(p['jobs']) and not failed else 'STOPPED_WITH_PARTIAL_RESULTS'
    write(RUN/'results.json', {'status': status, 'jobs': results, 'pending': [j['id'] for j in pending],
                              'actual_waits': waits, 'protocol': str(RUN/'protocol.json')})
    write(RUN/'completion.json', {'status': status, 'seconds': time.time()-begun, 'complete_jobs': len(results),
                                 'actual_waits': waits, 'pending': [j['id'] for j in pending]})
    lines = ['# CFC26 完整改进配方：统一结果', '',
             '|方法|seed|末轮HOTA%|末轮IDF1%|末轮MOTA%|计数nMAE|最佳轮|最佳HOTA%|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        test = row['terminal']['results']['test']; best = row['best']['results']['test']
        m = test['tracking']; nmae = next(c['nMAE'] for c in test['official_LOI_counts'] if c['location'] == 'ALL')
        lines.append(f"|{row['job']['method']}|{row['job']['seed']}|{100*m['HOTA']:.4f}|{100*m['IDF1']:.4f}|{100*m['MOTA']:.4f}|{nmae:.6f}|{row['best_epoch']}|{100*best['tracking']['HOTA']:.4f}|")
    (RUN/'summary.md').write_text('\n'.join(lines)+'\n')
    if status == 'COMPLETE':
        statistics = {}
        for method in METHODS:
            selected = [r for r in results if r['job']['method'] == method]
            statistics[method] = {k: {'mean': float(np.mean(v)), 'sample_SD': float(np.std(v, ddof=1)),
                                      'per_seed': [float(x) for x in v]}
                for k in ['HOTA', 'IDF1', 'MOTA']
                for v in [[r['terminal']['results']['test']['tracking'][k]
                           for r in sorted(selected, key=lambda r: r['job']['seed'])]]}
        interactions = {}
        for key in ['HOTA', 'IDF1', 'MOTA']:
            values = [sum(sign*next(r for r in results if r['job']['method'] == method and r['job']['seed'] == seed)['terminal']['results']['test']['tracking'][key]
                          for method, sign in [('R256', 1), ('A1', -1), ('A2', -1), ('A3', 1)]) for seed in range(3)]
            interactions[key] = {'per_seed': values, 'mean': float(np.mean(values)), 'sample_SD': float(np.std(values, ddof=1))}
        write(RUN/'statistics.json', {'terminal': statistics, 'fixed256_memory_gate_interactions': interactions,
                                     'definition': 'Three declared training repeats; sample SD, not CI/significance.'})
    if status != 'COMPLETE':
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'worker', 'pipeline', 'cache'])
    parser.add_argument('--job')
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--reattach', action='store_true')
    parser.add_argument('--output-id')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'worker':
        worker(args.job, args.deadline, args.output_id)
    elif args.phase == 'cache':
        prepare_shared_pixels()
    else:
        pipeline(args.reattach)
