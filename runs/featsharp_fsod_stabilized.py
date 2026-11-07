"""Paired full fine-tuning from the existing FeatSharp and GLISR source states."""
import argparse
import os
import json
import resource
import time
import traceback
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from runs import featsharp_marine_matched as fm

BASE = fm.BASE
RUN = BASE/'featsharp_fsod_stabilized_v1'
TASKS = ('FSOD', 'UATD', 'NKSID', 'HighResolution', 'Diver')
METHODS = ('FeatSharp', 'GLISR')
core, reader, ref = fm.core, fm.reader, fm.ref
write = fm.write


def source_path(method, seed):
    if method == 'FeatSharp':
        return BASE/f'featsharp_marine_matched_v1/s{seed}/source_latest.pt'
    return BASE/f'gsr_frozen_matched_readouts_v1/s{seed}_GLISR_source/source_glisr.pt'


def guard(deadline):
    observed = {'RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
    if time.time() > deadline:
        raise TimeoutError('Fixed target deadline')
    if observed['RSS_bytes'] > 96*1024**3 or observed['GPU_reserved_bytes'] > 48*1024**3:
        raise RuntimeError(str(observed))
    return observed


@torch.no_grad()
def clip_stable(parameters):
    grads=[p.grad for p in parameters if p.grad is not None]
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g,dtype=torch.float64) for g in grads]))
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient elements or FP64 norm')
    scale=(1./(norm+1e-6)).clamp(max=1.)
    for g in grads:g.mul_(scale.to(dtype=g.dtype))
    return norm


def chunked_tiles(self, images, return_summary=False):
    def encode(chunk):
        self.checkpoint_encoder_executed_images += len(chunk)
        return self.encoder(chunk)

    pieces = []
    for chunk in images.split(80):
        self.encoder_images += len(chunk)
        self.encoder_calls += 1
        if self.training:
            tokens = activation_checkpoint(encode, chunk, use_reentrant=False)['patch_tokens']
        else:
            tokens = encode(chunk)['patch_tokens']
        pieces.append(tokens.transpose(1, 2).reshape(len(chunk), 384, 16, 16))
    return torch.cat(pieces)


def save_checkpoint(folder, name, model, head, epoch, optimizers, generator, config):
    payload = {'model': model.state_dict(), 'head': head.state_dict(), 'epoch': epoch,
               'config': config, 'optimizers': [o.state_dict() for o in optimizers],
               'private_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
               'python_rng': fm.random.getstate()}
    temporary = folder/(name+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(folder/(name+'.pt'))


def worker(method, task, seed, deadline, p, folder):
    fm.setup(seed)
    core.guard = guard
    ref.ARM = 'full'
    ref.REPEAT = seed
    spec = json.loads(Path(p['data_manifest']).read_text())['tasks'][task]
    schedule = p['target_schedules'][task]
    model = fm.Model(p['encoder'], seed) if method == 'FeatSharp' else ref.GlobalMemoryGSR(p['encoder'])
    state = torch.load(source_path(method, seed), map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'] if method == 'FeatSharp' else state, strict=True)
    del state
    reader.configure_encoder(model.encoder, 'e2e')
    if method == 'FeatSharp' and task == 'NKSID':
        model.tile_generator = MethodType(chunked_tiles, model)
        model.checkpoint_encoder_executed_images = 0
    model.cuda()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = reader.new_head(task, seed) if task in ('FSOD', 'UATD', 'NKSID') else nn.Conv2d(384, 1 if spec['kind'] == 'point_regression' else spec['classes'], 1)
    head = head.cuda()
    extra = list(model.fusion.parameters())
    if method == 'GLISR':
        extra += model.adapter_parameters()
    body_opt = reader.body_optimizer(model.encoder)
    head_opt = reader.head_optimizer('FSOD' if spec['kind'] == 'detection' else 'Marine', head)
    fusion_opt = torch.optim.AdamW(extra, lr=.0001, weight_decay=.0001)
    optimizers = (body_opt, head_opt, fusion_opt)
    images = {s: torch.from_numpy(np.load(spec['views'][s], mmap_mode='r').copy()).cuda() for s in ('train', 'validation')}
    labels = {s: core.targets_for(task, spec, s) for s in images}
    policy = 'featsharp' if method == 'FeatSharp' else 'attention'
    config = {'method': method, 'task': task, 'seed': seed, 'source': str(source_path(method, seed)),
              'protocol': str(RUN/'protocol.json'), 'schedule': schedule, 'policy': policy,
              'encoder_batching': 'chunks<=80 with non-reentrant activation checkpoint during training' if method == 'FeatSharp' and task == 'NKSID' else 'original',
              'data_manifest': p['data_manifest'], 'full_parameters': True, 'head_clip_norm':1., 'clip_reduction_dtype':'float64', 'model_dtype':'float32'}
    write(folder/'config.json', config)
    generator = torch.Generator().manual_seed(seed)
    key = core.metric_key(task, spec)
    direction = -1 if spec['kind'] == 'point_regression' else 1
    best, updates, begun = -float('inf'), 0, time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, schedule['epochs']+1):
            model.train(); head.train()
            order = torch.randperm(len(spec['rows']['train']), generator=generator)
            if spec['kind'] == 'detection':
                head_opt.param_groups[0]['lr'] = .005*.1**sum(epoch > m for m in (40, 55))
            total = 0.
            for ids in order.split(schedule['batch']):
                guard(deadline)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                for group in body_opt.param_groups:
                    group['lr'] = reader.body_learning_rate(updates, schedule['updates'])
                fusion_opt.param_groups[0]['lr'] = ref.previous.shared.native.module_lr(updates, schedule['updates'], .0001)
                if updates == 0:
                    witness = {'backbone': model.encoder.backbone.blocks[0].attn.qkv.weight,
                               'fusion': next(model.fusion.parameters()), 'head': next(head.parameters())}
                    if method == 'FeatSharp':
                        witness.update(JBU=next(model.fusion.ups[0].jbu.parameters()),
                                       attention=model.fusion.ups[0].block.attn.qkv.weight,
                                       MLP=model.fusion.ups[0].block.mlp.w12.weight)
                    if method == 'GLISR':
                        witness['memory'] = model.local_memory[0].output.weight
                    before = {name: value.detach().clone() for name, value in witness.items()}
                field = model.field(images['train'][ids.cuda()], head, 'uatd' if task == 'UATD' else spec['kind'], policy)
                target = core.batch_targets(task, spec, labels['train'], ids.tolist())
                loss = core.task_loss(task, spec, head, field, target)
                loss.backward()
                body_norm = clip_stable(model.encoder.parameters())
                head_norm = clip_stable(head.parameters())
                fusion_norm = clip_stable(extra)
                if updates == 0:
                    gradients = {name: float(value.grad.norm()) for name, value in witness.items()}
                    assert all(np.isfinite(v) and v > 0 for v in gradients.values()), gradients
                    assert field.shape == (len(ids), 384, 32, 32) and field.dtype == torch.float32
                for optimizer in optimizers:
                    optimizer.step()
                updates += 1
                total += float(loss.detach())*len(ids)
                if updates == 1:
                    moved = {name: not torch.equal(value.detach(), before[name]) for name, value in witness.items()}
                    assert all(moved.values()), moved
                    report = {'updates': 1, 'first_ids': ids.tolist(), 'field': list(field.shape),
                              'dtype': str(field.dtype), 'gradient_norms': gradients, 'parameter_movement': moved,
                              'body_gradient_norm': float(body_norm), 'head_gradient_norm': float(head_norm),
                              'fusion_gradient_norm': float(fusion_norm), 'config': config,
                              'encoder_images': model.encoder_images, **guard(deadline)}
                    if method == 'GLISR':
                        report.update(actual_K=model.last_counts.tolist(), sparse_witness=model.first_sparse_witness)
                    if method == 'FeatSharp' and task == 'NKSID':
                        report['checkpoint_encoder_executed_images'] = model.checkpoint_encoder_executed_images
                        report['checkpoint_recomputed_images'] = model.checkpoint_encoder_executed_images-model.encoder_images
                    write(folder/'first_counted_update.json', report)
                    del before, witness
            entry = {'epoch': epoch, 'updates': updates, 'loss': total/len(order), 'wall_seconds': time.monotonic()-begun}
            if epoch % p['evaluation_every_epochs'][task] == 0:
                metric, independent, prediction = core.evaluate(task, spec, model, head, images['validation'], labels['validation'], schedule['batch'], deadline, policy)
                entry[key] = metric[key]
                payload = {'prediction': prediction, 'epoch': epoch, 'task': task, 'config': config}
                if direction*metric[key] > best:
                    best, best_epoch, best_metric = direction*metric[key], epoch, metric
                    torch.save(payload, folder/'best_predictions.pt')
                    save_checkpoint(folder, 'best', model, head, epoch, optimizers, generator, config)
                if epoch == schedule['epochs']:
                    terminal, terminal_independent = metric, independent
                    torch.save(payload, folder/'terminal_predictions.pt')
            save_checkpoint(folder, 'latest', model, head, epoch, optimizers, generator, config)
            log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)
    assert updates == schedule['updates']
    result = {'status': 'COMPLETE', **config, 'metric': key, 'higher_is_better': direction == 1,
              'terminal': terminal, 'terminal_independent': terminal_independent, 'best': best_metric,
              'best_epoch': best_epoch, 'terminal_epoch': schedule['epochs'], 'updates': updates,
              'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
              'wall_seconds': time.monotonic()-begun, **guard(deadline)}
    if method == 'GLISR':
        result.update(K_histogram=model.K_histogram, sequence_histogram=model.sequence_histogram)
    if method == 'FeatSharp' and task == 'NKSID':
        result['checkpoint_encoder_executed_images'] = model.checkpoint_encoder_executed_images
        result['checkpoint_recomputed_images'] = model.checkpoint_encoder_executed_images-model.encoder_images
    write(folder/'results.json', result)


def score(method, task, seed, deadline, p, folder):
    spec = json.loads(Path(p['data_manifest']).read_text())['tasks'][task]
    labels = core.targets_for(task, spec, 'validation')
    result = json.loads((folder/'results.json').read_text())
    endpoints = {}
    for endpoint in ('terminal', 'best'):
        if time.time() > deadline:
            raise TimeoutError('CPU scoring deadline')
        saved = torch.load(folder/f'{endpoint}_predictions.pt', map_location='cpu', weights_only=False)
        metric, independent = core.scores(task, spec, saved['prediction'], labels, 'validation')
        # Compare the complete canonical dictionary, including per-class outputs.
        assert json.loads(json.dumps(metric)) == result[endpoint], endpoint
        endpoints[endpoint] = {'metric': metric, 'independent': independent, 'epoch': saved['epoch']}
    write(folder/'verification.json', {'status': 'PASS_NATIVE_SAVED_PREDICTIONS', 'method': method,
                                     'task': task, 'seed': seed, 'endpoints': endpoints, 'GPU_forwards': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('worker', 'score'))
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--deadline', type=float, default=0.)
    args = parser.parse_args()
    p = json.loads((RUN/'protocol.json').read_text())
    folder = RUN/f's{args.seed}_{args.method}_{args.task}'
    folder.mkdir(parents=True, exist_ok=True)
    try:
        globals()[args.phase](args.method, args.task, args.seed, args.deadline, p, folder)
    except BaseException:
        write(folder/(args.phase+'_failure.json'), {'traceback': traceback.format_exc(), 'time_unix': time.time()})
        raise
