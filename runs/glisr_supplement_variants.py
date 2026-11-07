"""Finite GLISR mechanism variants, using the retained R3 implementation."""
import os
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'glisr_supplement_20261104_v1'
PARENT = Path(__file__).with_name('entry_snapshot.py')
spec = importlib.util.spec_from_file_location('glisr_supplement_parent', PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
SOURCE = ('FSOD', 'Marine', 'Xing_detection', 'Xing_segmentation', 'LSOD', 'CFC22')
VARIANTS = ('Rtau', 'R256', 'A1', 'A2', 'A3', 'A4', 'B1')
write = parent.write


class Fusion(parent.SparseFusion):
    def forward(self, g, local, ids, counts):
        width = ids.shape[1]
        base = F.interpolate(g, (32, 32), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        index = ids.clamp_min(0)
        selected = base.gather(1, index[:, :, None].expand(-1, -1, 384))
        whole = g.mean((2, 3))[:, None].expand(-1, width, -1)
        xy = torch.stack(((index % 32 + .5) / 16 - 1, (index // 32 + .5) / 16 - 1), -1)
        u = F.gelu(self.input(torch.cat((local-selected, selected, whole, xy), -1)))
        residual = self.output(u)
        gate = self.gate(u).sigmoid() if self.use_gate else torch.ones_like(residual[:, :, :1])
        delta = residual * gate
        delta = delta * (torch.arange(width, device=g.device)[None] < counts[:, None])[:, :, None]
        if self.trace:
            self.last_trace = {k: v.detach() for k, v in {
                'global_field': g, 'selected_global': selected, 'local': local,
                'g': gate, 'R': residual, 'delta': delta}.items()}
        return base.scatter_add(1, index[:, :, None].expand(-1, -1, 384), delta).transpose(1, 2).reshape(-1, 384, 32, 32)


class GlobalMemoryGSR(parent.GlobalMemoryGSR):
    def __init__(self, config, variant='Rtau', seed=0, trace=False):
        if variant not in VARIANTS:
            raise ValueError(variant)
        parent.REPEAT = seed
        parent.ARM = 'attention256'
        super().__init__(config)
        # The inherited initializer fixes fusion101+s and the two memories106+s.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101+seed)
            self.fusion = Fusion()
        self.variant = variant
        self.seed = seed
        self.trace = trace
        self.fusion.trace = trace
        self.fusion.use_gate = variant not in ('A1', 'A3')
        if not self.fusion.use_gate:
            self.fusion.gate.requires_grad_(False)
        if variant in ('A2', 'A3'):
            self.local_memory = nn.ModuleList()
            self.interaction_blocks = []
        elif variant == 'A4':
            self.interaction_blocks = [6, 6]
        self.local_capacity = 1024 if variant == 'B1' else 256

    def field(self, images, head=None, kind=None, policy='attention', actions=None, high_resolution=None):
        g = self.global_field(images)
        bb = self.encoder.backbone
        if policy == 'global':
            return F.interpolate(g, (32, 32), mode='bilinear', align_corners=False)
        a = bb.blocks[-1].attn
        qkv = self.global_qkv.reshape(len(images), 257, 3, a.num_heads, 384 // a.num_heads)
        q = qkv[:, 0, 0].unsqueeze(2)
        k = qkv[:, :, 1].transpose(1, 2)
        heat = ((q*a.scale) @ k.transpose(-1, -2)).softmax(-1)[:, :, :, 1:].mean(1).reshape(len(images), 1, 16, 16)
        mass = F.interpolate(heat, (32, 32), mode='bilinear', align_corners=False).flatten(1)
        mass = mass / mass.sum(1, keepdim=True)
        order = mass.argsort(dim=1, descending=True, stable=True)
        k80 = (mass.gather(1, order).cumsum(1) < .8).sum(1) + 1
        counts = k80.clamp(64, self.local_capacity) if self.variant == 'Rtau' else torch.full_like(k80, self.local_capacity)
        chosen = torch.full((len(images), self.local_capacity), -1, device=images.device, dtype=torch.long)
        for i, n in enumerate(counts.tolist()):
            chosen[i, :n] = order[i, :n].sort().values
            self.K_histogram[str(n)] = self.K_histogram.get(str(n), 0) + 1
            self.mass_shortfalls += int(k80[i] > self.local_capacity)
        self.last_counts = counts.detach()
        self.last_selected_ids = chosen.detach()
        self.last_attention_mass = mass.detach()
        large = F.interpolate(images, (448, 448), mode='bilinear', align_corners=False, antialias=True) if high_resolution is None else high_resolution
        patches_view = large.unfold(2, 14, 14).unfold(3, 14, 14).permute(0, 2, 3, 1, 4, 5)
        pe = bb.interpolate_pos_encoding(torch.empty(1, 1025, 384, device=images.device), 448, 448)
        local = torch.zeros(len(images), self.local_capacity, 384, device=images.device)
        for n in counts.unique().tolist():
            group = torch.where(counts == n)[0]
            ids = chosen[group, :n]
            xy = torch.stack(((ids % 32 + .5) / 16 - 1, (ids // 32 + .5) / 16 - 1), -1)
            pixels = patches_view[group[:, None], ids // 32, ids % 32].reshape(-1, images.shape[1], 14, 14)
            patches = bb.patch_embed.norm(bb.patch_embed.proj(pixels).reshape(len(group), n, 384))
            position = pe[0, 1:][ids]
            tokens = torch.cat(((bb.cls_token+pe[:, :1]).expand(len(group), -1, -1), patches+position), 1)
            lengths = []
            for index, block in enumerate(bb.blocks[:6]):
                lengths.append(int(tokens.shape[1]))
                tokens = block(tokens)
                for mi, after_block in enumerate(self.interaction_blocks):
                    if index + 1 == after_block:
                        tokens = self.local_memory[mi](tokens, g[group], xy, self.global_xy)
                        self.memory_hook_calls[mi] += 1
            self.encoder_images += len(group)
            self.encoder_calls += 1
            self.sequence_histogram[str(n+1)] = self.sequence_histogram.get(str(n+1), 0) + len(group)
            local[group, :n] = bb.norm(tokens)[:, 1:]
            if self.first_sparse_witness is None:
                expected = torch.stack([large[int(group[0]), :, int(j)//32*14:int(j)//32*14+14, int(j)%32*14:int(j)%32*14+14] for j in ids[0]])
                if not torch.equal(expected, pixels[:n]) or lengths != [n+1]*6:
                    raise RuntimeError('Local pixel/sequence mismatch')
                if not torch.equal(position[0], pe[0, ids[0]+1]):
                    raise RuntimeError('Original448 PE mismatch')
                self.first_sparse_witness = {'variant': self.variant, 'K': n, 'global_blocks': 12,
                    'local_blocks': 6, 'actual_local_sequence_lengths': lengths,
                    'selected_ids': ids[0].tolist(), 'original448_PE_exact': True,
                    'interaction_blocks': self.interaction_blocks,
                    'fusion_trainable_parameters': sum(p.numel() for p in self.fusion.parameters() if p.requires_grad),
                    'interaction_trainable_parameters': sum(p.numel() for p in self.local_memory.parameters() if p.requires_grad)}
        result = self.fusion(g, local, chosen, counts)
        if self.trace:
            self.last_trace = {**self.fusion.last_trace, 'selected_ids': chosen.detach(), 'counts': counts.detach(), 'attention_mass': mass.detach()}
        return result


Model = GlobalMemoryGSR


def source_training(model, p, tasks, folder, deadline, micro_batch=4, on_update=None):
    """Original six equal task losses and source optimizers; checkpoint every1000."""
    parent.REPEAT = model.seed
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.encoder.eval().requires_grad_(False)
    versions = [v._version for v in model.encoder.parameters()]
    images = {t: torch.from_numpy(np.load(tasks[t]['views']['train'], mmap_mode='r', allow_pickle=False)).cuda() for t in SOURCE}
    labels = {t: parent.targets_for(t, tasks[t], 'train') for t in SOURCE}
    heads = nn.ModuleDict({t: parent.new_head(t, tasks[t]).cuda() for t in SOURCE})
    optimizers = {t: parent.reader.head_optimizer('FSOD' if tasks[t]['kind'] == 'detection' else 'Marine', heads[t]) for t in SOURCE}
    fusion_parameters = [v for v in model.fusion.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW([{'params': fusion_parameters, 'lr': .001}, {'params': model.adapter_parameters(), 'lr': .0001}], weight_decay=.0001)
    plan_path = folder/'source_plan.npz'
    if not plan_path.exists():
        plans = {}
        for i, task in enumerate(SOURCE):
            n = len(tasks[task]['rows']['train'])
            macro = p['source_macro_per_task'][task]
            rng = np.random.RandomState(13+i+1000*parent.REPEAT)
            plans[task] = np.concatenate([rng.permutation(n) for _ in range(math.ceil(6000*macro/n))])[:6000*macro].reshape(6000, macro)
        np.savez(plan_path, **plans)
    plans = np.load(plan_path, allow_pickle=False)
    begun = time.monotonic()
    completed = 0
    stopped = False
    with (folder/'source_training.jsonl').open('w') as log:
        for update in range(6000):
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = parent.previous.shared.native.module_lr(update, 6000, .001)
            optimizer.param_groups[1]['lr'] = parent.previous.shared.native.module_lr(update, 6000, .0001)
            values = {}
            for task in SOURCE:
                head = heads[task]
                head.train()
                opt = optimizers[task]
                opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = (.005 if update < 4000 else .0005 if update < 5500 else .00005) if tasks[task]['kind'] == 'detection' else parent.previous.shared.native.module_lr(update, 6000, .001)
                ids = plans[task][update]
                value_sum = 0.
                for start in range(0, len(ids), micro_batch):
                    indices = ids[start:start+micro_batch].tolist()
                    field = model.field(images[task][indices], head, tasks[task]['kind'])
                    target = parent.batch_targets(task, tasks[task], labels[task], indices)
                    value = parent.task_loss(task, tasks[task], head, field, target)*len(indices)/len(ids)/6
                    value.backward()
                    value_sum += float(value.detach())
                values[task] = value_sum
                opt.step()
            adapter_norm = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), 1., error_if_nonfinite=True)
            norm = torch.nn.utils.clip_grad_norm_(fusion_parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            completed = update+1
            if time.time() > deadline:
                raise TimeoutError('Supplement source deadline')
            observed = {'RSS_bytes': parent.core.resource.getrusage(parent.core.resource.RUSAGE_SELF).ru_maxrss*1024,
                        'GPU_reserved_bytes': torch.cuda.max_memory_reserved()}
            if observed['RSS_bytes'] > 64*1024**3 or observed['GPU_reserved_bytes'] > 40*1024**3:
                raise RuntimeError('Supplement source resource cap '+str(observed))
            if completed == 1:
                write(folder/'first_source_update.json', {'updates': completed, 'source_tasks': SOURCE,
                    'source_macro': p['source_macro_per_task'], 'micro_batch': micro_batch, 'task_weight': 1/6,
                    'original_DINO_trainable': sum(v.numel() for v in model.encoder.parameters() if v.requires_grad),
                    'adapter_gradient_norm': float(adapter_norm), 'shared_fusion_gradient_norm': float(norm),
                    'local_memory_calls_ordered_modules': model.memory_hook_calls, 'sparse_witness': model.first_sparse_witness,
                    'actual_K': model.last_counts.tolist(),
                    'fusion_trainable_parameters': sum(v.numel() for v in fusion_parameters),
                    'adapter_trainable_parameters': sum(v.numel() for v in model.adapter_parameters()),
                    'optimizer_parameter_names': [name for name, v in model.named_parameters()
                        if any(v is q for group in optimizer.param_groups for q in group['params'])],
                    'gate_in_optimizer': any(v is q for v in model.fusion.gate.parameters()
                        for group in optimizer.param_groups for q in group['params'])})
            if completed == 50:
                elapsed = time.monotonic()-begun
                write(folder/'counted_source_prefix50.json', {
                    'status': 'COUNTED_PREFIX_CONTINUING', 'source_updates': completed,
                    'variant': model.variant, 'seed': model.seed, 'micro_batch': micro_batch,
                    'wall_seconds': elapsed, 'seconds_per_update': elapsed/completed,
                    'estimated_remaining_seconds': elapsed/completed*(6000-completed),
                    'deadline_unix': deadline, 'remaining_deadline_seconds': deadline-time.time(),
                    'original_DINO_parameter_versions_unchanged': versions == [v._version for v in model.encoder.parameters()],
                    'fusion_trainable_parameters': sum(v.numel() for v in fusion_parameters),
                    'adapter_trainable_parameters': sum(v.numel() for v in model.adapter_parameters()),
                    'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
                    'K_histogram': model.K_histogram, 'sequence_histogram': model.sequence_histogram,
                    'local_memory_calls_ordered_modules': model.memory_hook_calls, **observed})
            if completed % 100 == 0:
                row = {'updates': completed, 'losses': values, 'wall_seconds': time.monotonic()-begun,
                       'encoder_images': model.encoder_images, **observed}
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
            stopped = on_update is not None and on_update(completed, model) is False
            if completed % 1000 == 0 or completed == 6000 or stopped:
                payload = {'model': model.state_dict(), 'heads': heads.state_dict(),
                    'optimizer': optimizer.state_dict(), 'head_optimizers': {t: o.state_dict() for t, o in optimizers.items()},
                    'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state_all(), 'selection': model.selection_generator.get_state()},
                    'source_plan': str(plan_path), 'plans_cursor': completed, 'updates': completed,
                    'variant': model.variant, 'seed': parent.REPEAT, 'micro_batch': micro_batch}
                temporary = folder/'source_latest.tmp'
                torch.save(payload, temporary)
                temporary.replace(folder/'source_latest.pt')
            if stopped:
                break
    frozen = versions == [v._version for v in model.encoder.parameters()]
    if not frozen or any(v.grad is not None for v in model.encoder.parameters()):
        raise RuntimeError('Source original DINO changed')
    model.source_updates = completed
    model.source_complete = completed == 6000
    write(folder/'source_report.json', {'status': 'COMPLETE_SOURCE6000' if model.source_complete else 'STOPPED_COUNTED_PREFIX',
        'source_updates': completed, 'variant': model.variant, 'seed': parent.REPEAT,
        'task_weight': 1/6, 'micro_batch': micro_batch, 'original_DINO_parameter_versions_unchanged': frozen,
        'visits_per_task': {t: completed*p['source_macro_per_task'][t] for t in SOURCE},
        'encoder_images': model.encoder_images, 'encoder_calls': model.encoder_calls,
        'K_histogram': model.K_histogram, 'sequence_histogram': model.sequence_histogram,
        'local_memory_calls_ordered_modules': model.memory_hook_calls, 'first_sparse_witness': model.first_sparse_witness,
        'wall_seconds': time.monotonic()-begun})
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def worker(variant, seed, deadline, micro_batch=4):
    p = json.loads((RUN/'protocol.json').read_text())
    parent.REPEAT = seed
    parent.original.runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    folder = RUN/f'{variant}_s{seed}_source'
    model = GlobalMemoryGSR(p['encoder'], variant, seed).cuda()
    tasks = json.loads(Path(p['data_manifest']).read_text())['tasks']
    source_training(model, p, tasks, folder, deadline, micro_batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['source'])
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--micro-batch', type=int, choices=[1, 4], default=4)
    args = parser.parse_args()
    worker(args.variant, args.seed, args.deadline, args.micro_batch)
