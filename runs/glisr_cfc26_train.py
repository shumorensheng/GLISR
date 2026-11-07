"""Central-protocol CFC26 detector/tracker full fine-tuning and technical check."""
import argparse
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runs import cfc26_tracking_interface as tracking
from runs import downstream_vit_finetuning_readout as reader
from sonar_encoder.training.model import build_encoder

BASE = tracking.BASE
RUN = BASE / 'glisr_supplement_20261104_v1'
PROTOCOL = RUN / 'protocol.json'
METHODS = ('Rtau', 'same_source_DINO', 'SigLIP2', 'B2', 'FeatSharp', 'HSMLA', 'Adapter', 'Specialist')
TOTAL = 52500
ASSETS = BASE / 'siglip2_frozen_baseline_v1/model_assets'


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_images():
    """Official SigLIP PIL preprocessing; store identical RGB channels once."""
    from runs.siglip2_frozen_components import make_processor, process_grayscale
    processor = make_processor(ASSETS)
    begun = time.monotonic()
    for split in ('train', 'val', 'test'):
        rows = tracking.load_rows(split)
        cache = np.lib.format.open_memmap(tracking.RUN / f'{split}_siglip224.npy', mode='w+',
                                          dtype=np.float32, shape=(len(rows), 1, 224, 224))
        for start in range(0, len(rows), 16):
            opened = [Image.open(row['path']) for row in rows[start:start + 16]]
            pixels = process_grayscale(processor, opened)
            for image in opened:
                image.close()
            cache[start:start + len(pixels)] = pixels[:, :1].numpy()
            if time.monotonic() - begun > 1800:
                raise TimeoutError('30-minute CPU image preparation cap')
        cache.flush()
        del cache
    disk_bytes = sum(path.stat().st_size for path in tracking.RUN.rglob('*') if path.is_file())
    if disk_bytes > 4 * 1024**3:
        raise RuntimeError('4 GiB prepared-view disk cap')
    write(tracking.RUN / 'siglip_prepare_receipt.json', {'seconds': time.monotonic() - begun, 'disk_bytes': disk_bytes,
          'definition': 'Official SiglipImageProcessor PIL L→RGB bilinear224 rescale1/255 mean/std .5; identical RGB channels stored once FP32 and expanded on GPU',
          'cache_bytes': 10000 * 224 * 224 * 4, 'fitted_statistics': False})


class PatchAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(384, 390)
        self.output = nn.Linear(390, 384)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return x + self.output(F.gelu(self.input(x)))


class AdapterModel(nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101 + seed)
            self.fusion = PatchAdapter()

    def field(self, images, head, kind):
        tokens = self.fusion(self.encoder(images)['patch_tokens'])
        grid = tokens.transpose(1, 2).reshape(len(images), 384, 16, 16)
        return F.interpolate(grid, (32, 32), mode='bilinear', align_corners=False)


def setup(seed):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()


def construct(method, seed, protocol):
    config = protocol['encoder']
    if method in ('Rtau', 'B2'):
        from runs.glisr_supplement_variants import GlobalMemoryGSR
        model = GlobalMemoryGSR(config, 'Rtau', seed)
    elif method == 'same_source_DINO':
        from runs.gsr_same_source_dino import DirectDINO
        model = DirectDINO(config)
    elif method == 'SigLIP2':
        from runs.gsr_siglip2_fullft import DirectSigLIP2
        model = DirectSigLIP2()
    elif method == 'FeatSharp':
        from runs.featsharp_marine_matched import Model
        model = Model(config, seed)
    elif method == 'HSMLA':
        from runs.hsmla_marine_matched import Model
        model = Model(config, seed)
    elif method == 'Adapter':
        model = AdapterModel(config, seed)
    else:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91,
                                         trainable_backbone_layers=5, min_size=224, max_size=224)
        model.load_state_dict(torch.load(BASE / 'task_specialist_reference_v1/model_assets/fasterrcnn.pth',
                                        map_location='cpu', weights_only=True), strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
        return model, None
    if method != 'SigLIP2':
        path = protocol['reuse_source_paths'][method][str(seed)]
        state = torch.load(path, map_location='cpu', weights_only=False)
        if method == 'same_source_DINO':
            model.encoder.load_state_dict(state['encoder'], strict=True)
        else:
            model.load_state_dict(state if method == 'Rtau' else state['model'], strict=True)
        del state
        reader.configure_encoder(model.encoder, 'e2e')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = tracking.make_head(siglip=method == 'SigLIP2')
    return model, head


def optimizers_for(method, model, head):
    if method == 'Specialist':
        return [torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=.0005)]
    body = model.vision if method == 'SigLIP2' else model.encoder
    body_opt = reader.body_optimizer(body)
    head_opt = torch.optim.SGD(head.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
    optimizers = [body_opt, head_opt]
    if method in ('Rtau', 'B2', 'FeatSharp', 'HSMLA', 'Adapter'):
        extra = list(model.fusion.parameters())
        if method in ('Rtau', 'B2'):
            extra += model.adapter_parameters()
        optimizers.append(torch.optim.AdamW(extra, lr=.0001, weight_decay=.0001))
    return optimizers


def schedule(optimizers, method, epoch, completed):
    head_lr = .005 * .1 ** sum(epoch > milestone for milestone in (40, 55))
    if method == 'Specialist':
        optimizers[0].param_groups[0]['lr'] = head_lr
        return
    step = completed + 1
    multiplier = step / 200 if step <= 200 else .01 + .99 * (1 + math.cos(math.pi * (step - 200) / (TOTAL - 200))) / 2
    for group in optimizers[0].param_groups:
        group['lr'] = 2e-5 * multiplier
    optimizers[1].param_groups[0]['lr'] = head_lr
    if len(optimizers) == 3:
        optimizers[2].param_groups[0]['lr'] = .0001 * multiplier


def images_for(method, split):
    if method in ('SigLIP2', 'Specialist'):
        # The receipt ensures caches are complete before GPU execution.
        json.loads((tracking.RUN / 'siglip_prepare_receipt.json').read_text())
        return np.load(tracking.RUN / f'{split}_siglip224.npy', mmap_mode='r')
    return np.load(tracking.RUN / f'{split}_224.npy', mmap_mode='r')


def batch_images(method, images, indices):
    x = torch.from_numpy(np.array(images[indices], copy=True)).cuda()
    if method in ('SigLIP2', 'Specialist'):
        x = x.expand(-1, 3, -1, -1).contiguous()
        if method == 'Specialist':
            x = (x + 1) / 2
    return x


def cuda_targets(rows, indices):
    return [{key: value.cuda() for key, value in tracking.target(rows[i]).items()} for i in indices]


def loss_for(method, model, head, x, targets):
    if method == 'Specialist':
        return sum(model(list(x), targets).values())
    field = model.field(x, head, 'detection')
    value = sum(head(field, targets).values())
    if method == 'HSMLA':
        value += model.fusion.regularization_loss
    return value


def clip_gradients(method, optimizers):
    from runs.featsharp_fsod_stabilized import clip_stable
    names = ['detector'] if method == 'Specialist' else ['body', 'head', 'module']
    return {name: float(clip_stable(parameter for group in optimizer.param_groups for parameter in group['params']))
            for name, optimizer in zip(names, optimizers)}


def resources(deadline):
    observed = {'RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > deadline:
        raise TimeoutError('CFC26 fit deadline')
    if observed['RSS_bytes'] > 96 * 1024**3 or observed['GPU_reserved_bytes'] > 48 * 1024**3:
        raise RuntimeError(str(observed))
    return observed


@torch.no_grad()
def evaluate(method, model, head, split, folder, tag, deadline):
    rows = tracking.load_rows(split)
    images = images_for(method, split)
    model.eval()
    if head is not None:
        head.eval()
    detections = []
    begun = time.monotonic()
    for start in range(0, len(rows), 8):
        indices = list(range(start, min(start + 8, len(rows))))
        x = batch_images(method, images, indices)
        results = model(list(x)) if method == 'Specialist' else head(model.field(x, head, 'detection'))
        detections.extend(tracking.native_detections(result, rows[i]) for result, i in zip(results, indices))
        resources(deadline)
    tracks = tracking.associate(rows, detections)
    result = tracking.score(rows, tracks)
    result['seconds'] = time.monotonic() - begun
    result['split'] = split
    torch.save({'row_ids': [row['row_id'] for row in rows], 'sequence_ids': [row['sequence_id'] for row in rows],
                'frame_indices': [row['frame_index'] for row in rows], 'native_detections': detections, 'native_tracks': tracks},
               folder / f'{tag}_predictions.pt')
    write(folder / f'{tag}_metrics.json', result)
    return result


def checkpoint(folder, name, method, model, head, state, optimizers, generator, resumable):
    payload = {'method': method, 'model': model.state_dict(), 'head': None if head is None else head.state_dict(),
               'progress': state, 'protocol': str(PROTOCOL)}
    if resumable:
        payload.update({'optimizers': [optimizer.state_dict() for optimizer in optimizers],
                        'shuffle_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
                        'python_rng': random.getstate()})
    temporary = folder / f'{name}.tmp'
    torch.save(payload, temporary)
    temporary.replace(folder / f'{name}.pt')


def check(method, seed, protocol, folder, deadline):
    if method != 'Rtau' or seed != 0:
        raise ValueError('The authorized technical check is Rtau seed0 only')
    begun = time.monotonic()
    model, head = construct(method, seed, protocol)
    model, head = model.cuda().train(), head.cuda().train()
    optimizers = optimizers_for(method, model, head)
    rows = tracking.load_rows('train')
    indices = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))[:8].tolist()
    x = batch_images(method, images_for(method, 'train'), indices)
    targets = cuda_targets(rows, indices)
    before = model.encoder.backbone.blocks[0].attn.qkv.weight.detach().clone()
    schedule(optimizers, method, 1, 0)
    value = loss_for(method, model, head, x, targets)
    value.backward()
    gradient_norms = clip_gradients(method, optimizers)
    for optimizer in optimizers:
        optimizer.step()
    torch.cuda.synchronize()
    body_changed = not torch.equal(before, model.encoder.backbone.blocks[0].attn.qkv.weight)
    if not body_changed:
        raise RuntimeError('Technical check did not update original backbone')
    model.eval()
    head.eval()
    with torch.no_grad():
        field = model.field(x, head, 'detection')
        predictions = head(field)
    if field.shape != (8, 384, 32, 32):
        raise RuntimeError(str(field.shape))
    write(folder / 'check_receipt.json', {'status': 'TECHNICAL_CHECK_PASSED_DIAGNOSTIC_STATE_DISCARDED',
          'method': method, 'seed': seed, 'extra_training_visits': 8, 'extra_eval_visits': 8,
          'loss': float(value.detach()), 'FP64_gradient_norms': gradient_norms, 'body_changed': body_changed,
          'field_shape': list(field.shape), 'detection_counts': [len(p['boxes']) for p in predictions],
          'model_parameters': sum(p.numel() for p in model.parameters()),
          'head_parameters': sum(p.numel() for p in head.parameters()), 'seconds': time.monotonic() - begun,
          'resources': resources(deadline)})


def train(method, seed, protocol, folder, deadline, resume=False, stop_after=None):
    model, head = construct(method, seed, protocol)
    model = model.cuda()
    if head is not None:
        head = head.cuda()
    optimizers = optimizers_for(method, model, head)
    generator = torch.Generator().manual_seed(seed)
    state = {'epoch': 1, 'cursor': 0, 'order': None, 'updates': 0, 'best_HOTA': -1., 'best_epoch': None,
             'training_seconds': 0., 'evaluation_seconds': 0.}
    if resume:
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        if head is not None:
            head.load_state_dict(saved['head'], strict=True)
        for optimizer, saved_optimizer in zip(optimizers, saved['optimizers']):
            optimizer.load_state_dict(saved_optimizer)
        state = saved['progress']
        generator.set_state(saved['shuffle_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    rows = tracking.load_rows('train')
    images = images_for(method, 'train')
    write(folder / 'config.json', {'method': method, 'seed': seed, 'protocol': str(PROTOCOL),
          'updates': TOTAL, 'epochs': 60, 'batch': 8, 'source_training': method not in ('B2', 'SigLIP2', 'Specialist'),
          'head': 'COCO FasterRCNNResNet50FPNV2 fullFT, new2classpredictor' if method == 'Specialist' else 'Fresh FasterRCNN RPN+ROI fish/background, 224canvas; grid32/stride7 except nativeSigLIP14/stride16',
          'specialist_context': 'COCO supervised full detector and ImageNet normalization; separate task specialist, different source context' if method == 'Specialist' else None,
          'selection': 'Validation HOTA every5epochs including60; test only terminal state; no test selection',
          'head_stability': 'Separate FP64 norm max1 clipping body/head/module; specialist whole detector. Same new CFC recipe for all encoders',
          'metric_config': {'association_score': .25, 'association_IoU': .3, 'max_missing_frames': 2, 'min_hits': 1,
                            'evaluation': 'Official TrackEval HOTA19alphas, Identity/CLEAR threshold.5'},
          'test_exposure': 'Preparation GT audits and previous CFC26 Mini diagnostic history; not globally unseen'})
    with (folder / 'training.jsonl').open('a' if resume else 'w') as log:
        while state['epoch'] <= 60:
            epoch = state['epoch']
            if state['order'] is None:
                state['order'] = torch.randperm(len(rows), generator=generator).tolist()
            model.train()
            if head is not None:
                head.train()
            for cursor in range(state['cursor'], len(rows), 8):
                resources(deadline)
                indices = state['order'][cursor:cursor + 8]
                begun = time.monotonic()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                schedule(optimizers, method, epoch, state['updates'])
                x = batch_images(method, images, indices)
                loss = loss_for(method, model, head, x, cuda_targets(rows, indices))
                loss.backward()
                gradient_norms = clip_gradients(method, optimizers)
                for optimizer in optimizers:
                    optimizer.step()
                torch.cuda.synchronize()
                state['training_seconds'] += time.monotonic() - begun
                state['updates'] += 1
                state['cursor'] = cursor + len(indices)
                if state['updates'] == 1 or state['updates'] % 100 == 0:
                    log.write(json.dumps({'epoch': epoch, 'updates': state['updates'], 'loss': float(loss.detach()),
                                         'FP64_gradient_norms': gradient_norms, 'training_seconds': state['training_seconds']}) + '\n')
                    log.flush()
                if state['updates'] == 50:
                    write(folder / 'counted_prefix50.json', {'updates': 50, 'training_seconds': state['training_seconds'],
                          'estimated_full_training_seconds': state['training_seconds'] / 50 * TOTAL,
                          'extra_check_visits_excluded': True, 'resources': resources(deadline)})
                if stop_after is not None and state['updates'] == stop_after:
                    checkpoint(folder, 'last', method, model, head, state, optimizers, generator, True)
                    write(folder / 'prefix_paused.json', {'status': 'COUNTED_PREFIX_SAVED', 'updates': state['updates']})
                    return
            if epoch % 5 == 0:
                metrics = evaluate(method, model, head, 'val', folder, f'val_epoch{epoch:02d}', deadline)
                state['evaluation_seconds'] += metrics['seconds']
                if metrics['HOTA'] > state['best_HOTA']:
                    state['best_HOTA'], state['best_epoch'] = metrics['HOTA'], epoch
                    checkpoint(folder, 'best', method, model, head, state, optimizers, generator, False)
            state['epoch'] += 1
            state['cursor'], state['order'] = 0, None
            checkpoint(folder, 'last', method, model, head, state, optimizers, generator, True)
        checkpoint(folder, 'terminal', method, model, head, state, optimizers, generator, False)
        write(folder / 'train_completion.json', {'status': 'COMPLETE_CFC26_TRAIN_AND_VALIDATION',
              'updates': state['updates'], 'best_HOTA': state['best_HOTA'], 'best_epoch': state['best_epoch'],
              'training_seconds': state['training_seconds'], 'evaluation_seconds': state['evaluation_seconds'],
              'resources': resources(deadline)})


def score_terminal(method, seed, protocol, folder, deadline):
    # Central main must freeze all condition protocols before permitting test evaluation.
    if not protocol['CFC26']['all_condition_protocols_locked']:
        raise RuntimeError('CFC26 condition protocols are not locked')
    model, head = construct(method, seed, protocol)
    saved = torch.load(folder / 'terminal.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    model = model.cuda()
    if head is not None:
        head.load_state_dict(saved['head'], strict=True)
        head = head.cuda()
    metrics = evaluate(method, model, head, 'test', folder, 'test_terminal', deadline)
    write(folder / 'completion.json', {'status': 'COMPLETE_CFC26_TERMINAL_TEST', 'method': method,
          'seed': seed, 'updates': saved['progress']['updates'], 'terminal_test': metrics,
          'best_validation_HOTA': saved['progress']['best_HOTA'], 'best_validation_epoch': saved['progress']['best_epoch']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare-images', 'check', 'train', 'score'])
    parser.add_argument('--method', choices=METHODS, default='Rtau')
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.phase == 'prepare-images':
        prepare_images()
    else:
        if args.deadline is None:
            parser.error('--deadline is required for GPU phases')
        protocol = json.loads(PROTOCOL.read_text())
        folder = RUN / f'CFC26_{args.method}_s{args.seed}'
        folder.mkdir(parents=True, exist_ok=True)
        setup(args.seed)
        if args.phase == 'check':
            check(args.method, args.seed, protocol, folder, args.deadline)
        elif args.phase == 'train':
            train(args.method, args.seed, protocol, folder, args.deadline, args.resume, args.stop_after)
        else:
            score_terminal(args.method, args.seed, protocol, folder, args.deadline)
