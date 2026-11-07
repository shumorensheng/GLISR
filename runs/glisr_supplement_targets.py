"""Matched full fine-tuning for the finite GLISR mechanism supplement."""
import argparse
import json
import random
import resource
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from runs import glisr_supplement_variants as variants

RUN = variants.RUN
ref = variants.parent
core, reader = ref.core, ref.reader
write = ref.write
TASKS = ('FSOD', 'UATD', 'Marine')
VARIANTS = ('A1', 'A2', 'A3', 'A4', 'B1')


def guard(deadline):
    observed = {'RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > deadline:
        raise TimeoutError('Supplement target deadline')
    if observed['RSS_bytes'] > 64*1024**3 or observed['GPU_reserved_bytes'] > 40*1024**3:
        raise RuntimeError(str(observed))
    return observed


class Model(variants.GlobalMemoryGSR):
    capture_selection = False

    def field(self, *args, **kwargs):
        result = super().field(*args, **kwargs)
        if self.capture_selection:
            self.selection_ids.append(self.last_selected_ids.cpu().numpy().astype(np.uint16))
            self.selection_counts.append(self.last_counts.cpu().numpy().astype(np.uint16))
        return result


def save_checkpoint(folder, name, model, head, epoch, updates, optimizers, generator, config):
    payload = {'model': model.state_dict(), 'head': head.state_dict(), 'epoch': epoch,
               'updates': updates, 'config': config, 'optimizers': [o.state_dict() for o in optimizers],
               'private_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
               'python_rng': random.getstate(), 'selection_rng': model.selection_generator.get_state()}
    temporary = folder/(name+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(folder/(name+'.pt'))


def save_selection(folder, task, spec, model):
    ids = np.concatenate(model.selection_ids)
    counts = np.concatenate(model.selection_counts)
    rows = spec['rows']['validation']
    assert len(ids) == len(rows)
    np.savez_compressed(folder/'terminal_selection.npz', selected_ids=ids, actual_K=counts)
    metadata = []
    for index, row in enumerate(rows):
        with Image.open(row['image']) as image:
            width, height = image.size
        if task == 'FSOD':
            transform = {'boxes': 'native annotation xywh to xyxy; scale x by224/native_width, y by224/native_height',
                         'scale_xy': [224/width, 224/height], 'head_image_wh': [224, 224]}
        elif task == 'UATD':
            transform = {'boxes': 'normalized native xyxy; original dense.box_targets32 assignment and decode32',
                         'output_grid_wh': [32, 32], 'channels': 50}
        else:
            transform = {'mask': 'unchanged native integer mask; bilinear logits resize to native mask shape before argmax',
                         'classes': 12, 'mask_path': row['mask']}
        metadata.append({'validation_index': index, 'row': row, 'native_wh': [width, height],
                         'input_wh': [224, 224], 'field_wh': [32, 32], 'GT_transform': transform,
                         'actual_K': int(counts[index]), 'selection_array_row': index})
    write(folder/'terminal_selection.json', {'status': 'ACTUAL_SCHEDULED_TERMINAL_TRACE',
        'arrays': str(folder/'terminal_selection.npz'), 'selected_ids_dtype': 'uint16',
        'selected_ids_coordinate': 'raster32 original448 patch IDs; patch14; original interpolated32grid PE',
        'unused_padding': 65535, 'new_extra_forwards': 0, 'rows': metadata})


def target(variant, task, seed, deadline, p, folder):
    ref.original.runtime()
    ref.REPEAT = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    core.guard = guard
    spec = json.loads(Path(p['data_manifest']).read_text())['tasks'][task]
    schedule = p['target_schedules'][task]
    source = RUN/f'{variant}_s{seed}_source/source_latest.pt'
    model = Model(p['encoder'], variant, seed)
    state = torch.load(source, map_location='cpu', weights_only=False)
    assert state['updates'] == 6000
    model.load_state_dict(state['model'], strict=True)
    del state
    reader.configure_encoder(model.encoder, 'e2e')
    model.cuda()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = reader.new_head(task, seed)
    head = head.cuda()
    extra = [v for v in model.fusion.parameters() if v.requires_grad] + model.adapter_parameters()
    body_opt = reader.body_optimizer(model.encoder)
    head_opt = reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    fusion_opt = torch.optim.AdamW(extra, lr=.0001, weight_decay=.0001)
    optimizers = (body_opt, head_opt, fusion_opt)
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r', allow_pickle=False)).cuda() for s in ('train', 'validation')}
    labels = {s: core.targets_for(task, spec, s) for s in images}
    config = {'variant': variant, 'task': task, 'seed': seed, 'source': str(source),
              'protocol': str(RUN/'protocol.json'), 'schedule': schedule, 'policy': 'attention',
              'data_manifest': p['data_manifest'], 'encoder_batching': 'original; no activation checkpoint',
              'head_gradient_clip': 'unbounded original recipe', 'full_parameters': True}
    write(folder/'config.json', config)
    generator = torch.Generator().manual_seed(seed)
    key = core.metric_key(task, spec)
    best, updates, begun = -float('inf'), 0, time.monotonic()
    step_seconds = 0.
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs']+1):
            model.train()
            head.train()
            order = torch.randperm(len(spec['rows']['train']), generator=generator)
            if spec['kind'] == 'detection':
                head_opt.param_groups[0]['lr'] = .005*.1**sum(epoch > m for m in (40, 55))
            total = 0.
            histogram = np.zeros(1025, dtype=np.int64)
            for ids in order.split(schedule['batch']):
                guard(deadline)
                step_begun = time.monotonic()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                for group in body_opt.param_groups:
                    group['lr'] = reader.body_learning_rate(updates, schedule['updates'])
                fusion_opt.param_groups[0]['lr'] = ref.previous.shared.native.module_lr(updates, schedule['updates'], .0001)
                if updates == 0:
                    witness = {'backbone': model.encoder.backbone.blocks[0].attn.qkv.weight,
                               'fusion_input': model.fusion.input.weight, 'fusion_output': model.fusion.output.weight,
                               'head': next(head.parameters())}
                    witness.update({f'memory{i}': module.output.weight for i, module in enumerate(model.local_memory)})
                    if model.fusion.use_gate:
                        witness['gate'] = model.fusion.gate.weight
                    before = {name: value.detach().clone() for name, value in witness.items()}
                field = model.field(images['train'][ids.cuda()], head, 'uatd' if task == 'UATD' else spec['kind'])
                truth = core.batch_targets(task, spec, labels['train'], ids.tolist())
                loss = core.task_loss(task, spec, head, field, truth)
                loss.backward()
                body_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
                head_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                fusion_norm = torch.nn.utils.clip_grad_norm_(extra, 1., error_if_nonfinite=True)
                if updates == 0:
                    gradients = {name: float(value.grad.norm()) for name, value in witness.items()}
                    assert all(np.isfinite(v) and v > 0 for v in gradients.values()), gradients
                    assert field.shape == (len(ids), 384, 32, 32) and field.dtype == torch.float32
                for optimizer in optimizers:
                    optimizer.step()
                updates += 1
                total += float(loss.detach())*len(ids)
                histogram += torch.bincount(model.last_counts, minlength=1025).cpu().numpy()
                step_seconds += time.monotonic()-step_begun
                if updates == 1:
                    moved = {name: not torch.equal(value.detach(), before[name]) for name, value in witness.items()}
                    assert all(moved.values()), moved
                    parameter_ids = {id(v) for opt in optimizers for group in opt.param_groups for v in group['params']}
                    write(folder/'first_counted_update.json', {'updates': 1, 'first_ids': ids.tolist(),
                        'field': list(field.shape), 'dtype': str(field.dtype), 'gradient_norms': gradients,
                        'parameter_movement': moved, 'body_gradient_norm': float(body_norm),
                        'head_gradient_norm': float(head_norm), 'fusion_gradient_norm': float(fusion_norm),
                        'actual_K': model.last_counts.tolist(), 'sparse_witness': model.first_sparse_witness,
                        'optimizer_parameter_names': [name for name, v in model.named_parameters() if id(v) in parameter_ids],
                        'gate_in_optimizer': any(id(v) in parameter_ids for v in model.fusion.gate.parameters()),
                        'mask_token_in_optimizer': id(model.encoder.backbone.mask_token) in parameter_ids,
                        'encoder_images': model.encoder_images, 'config': config, **guard(deadline)})
                    del before, witness
                if updates == 50:
                    elapsed = time.monotonic()-begun
                    write(folder/'counted_target_prefix50.json', {'status': 'COUNTED_PREFIX_CONTINUING',
                        'updates': updates, 'wall_seconds': elapsed, 'model_step_seconds': step_seconds,
                        'seconds_per_update': step_seconds/updates,
                        'estimated_remaining_model_step_seconds': step_seconds/updates*(schedule['updates']-updates),
                        'remaining_deadline_seconds': deadline-time.time(), 'config': config, **guard(deadline)})
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order),
                     'K_histogram': histogram.tolist(), 'wall_seconds': time.monotonic()-begun}
            if epoch % p['evaluation_every_epochs'][task] == 0:
                model.capture_selection = epoch == schedule['epochs']
                model.selection_ids, model.selection_counts = [], []
                metric, independent, prediction = core.evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], deadline, 'attention')
                model.capture_selection = False
                entry[key] = metric[key]
                payload = {'prediction': prediction, 'epoch': epoch, 'task': task, 'config': config}
                if metric[key] > best:
                    best, best_epoch, best_metric = metric[key], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                    save_checkpoint(folder, 'best', model, head, epoch, updates, optimizers, generator, config)
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = metric, independent
                    torch.save(payload, folder/'terminal_predictions.pt')
                    save_selection(folder, task, spec, model)
            save_checkpoint(folder, 'latest', model, head, epoch, updates, optimizers, generator, config)
            log.write(json.dumps(entry)+'\n')
            log.flush()
            print(json.dumps(entry), flush=True)
    assert updates == schedule['updates']
    write(folder/'results.json', {'status': 'COMPLETE', **config, 'metric': key, 'higher_is_better': True,
        'terminal': terminal, 'terminal_independent': terminal_independent, 'best': best_metric,
        'best_epoch': best_epoch, 'terminal_epoch': schedule['epochs'], 'updates': updates,
        'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'K_histogram': model.K_histogram, 'sequence_histogram': model.sequence_histogram,
        'wall_seconds': time.monotonic()-begun, **guard(deadline)})


def score(variant, task, seed, deadline, p, folder):
    spec = json.loads(Path(p['data_manifest']).read_text())['tasks'][task]
    labels = core.targets_for(task, spec, 'validation')
    result = json.loads((folder/'results.json').read_text())
    endpoints = {}
    for endpoint in ('terminal', 'best'):
        if time.time() > deadline:
            raise TimeoutError('CPU scoring deadline')
        saved = torch.load(folder/f'{endpoint}_predictions.pt', map_location='cpu', weights_only=False)
        metric, independent = core.scores(task, spec, saved['prediction'], labels, 'validation')
        assert json.loads(json.dumps(metric)) == result[endpoint], endpoint
        endpoints[endpoint] = {'metric': metric, 'independent': independent, 'epoch': saved['epoch']}
    write(folder/'verification.json', {'status': 'PASS_NATIVE_SAVED_PREDICTIONS', 'variant': variant,
        'task': task, 'seed': seed, 'endpoints': endpoints, 'GPU_forwards': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('target', 'score'))
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    p = json.loads((RUN/'protocol.json').read_text())
    folder = RUN/f'{args.variant}_s{args.seed}_{args.task}'
    folder.mkdir(parents=True, exist_ok=True)
    try:
        globals()[args.phase](args.variant, args.task, args.seed, args.deadline, p, folder)
    except BaseException:
        write(folder/(args.phase+'_failure.json'), {'traceback': traceback.format_exc(), 'time_unix': time.time()})
        raise
