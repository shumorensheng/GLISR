"""One bounded UATD residual adapter and supervised capacity-control pilot."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

PROCESS_STARTED = time.monotonic()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs.recompute_uatd_area_results import compare_metrics, dense_metrics, digest, global_metrics
from sonar_encoder.training.readout import state_digest, write_json
from sonar_encoder.training.uatd_head import (box_targets, decode_detections, dense_loss,
                                             detection_metrics, global_targets, multilabel_metrics)

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'readout_preserving_adapter_v1'
T0 = BASE / 'ssl_global_keep_v1/t0'
REAL = BASE / 'ssl_real_only_v1/real'
MANIFEST = ROOT / 'runs/data/uatd_readout_v1/manifest.json'
NORMALIZATION = MANIFEST.with_name('normalization.json')
STATES = {'t0': 'b9fc4072daeecd3250c87c4f4b6994248f5a84526d3e62effa5bb050dbbf973e',
          'real': '4744af7045878201960f67be2fd475b43680f3ad43394502efbb086821473849'}
LIMITS = {'prepare': 180, 'adapter_a': 300, 'adapter_b': 300,
          'heads_a': 900, 'heads_b': 900, 'heads_c': 900, 'close': 300}


def make_adapter():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        adapter = nn.Sequential(nn.Linear(384, 96, bias=True), nn.GELU(approximate='none'),
                                nn.Linear(96, 384, bias=False))
        nn.init.zeros_(adapter[2].weight)
    return adapter


def token_delta(adapter, field, projected):
    delta = adapter(field.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(field)
    if projected:
        delta = delta - delta.mean((-2, -1), keepdim=True)
    return delta


def paired_target(base, real):
    source = base.astype(np.float64)
    teacher = real.astype(np.float64)
    source -= source.mean((-2, -1), keepdims=True)
    teacher -= teacher.mean((-2, -1), keepdims=True)
    teacher -= source
    return teacher.astype(np.float32)


def draw_indices(generator):
    return torch.randint(0, 6051, (32,), generator=generator)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def resources():
    return {'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            'disk_bytes': sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())}


def check_resources():
    measured = resources()
    assert measured['max_rss_bytes'] <= 8 * 1024**3, measured
    assert measured['max_gpu_reserved_bytes'] <= 8 * 1024**3, measured
    assert measured['disk_bytes'] <= 128 * 1024**2, measured
    return measured


def prepare():
    manifest = json.loads(MANIFEST.read_text())
    ids = {s: [r['row_id'] for r in manifest[s]] for s in ('train', 'validation')}
    assert [len(ids[s]) for s in ids] == [6051, 1505]
    inputs = [Path(__file__), ROOT / 'tests/test_readout_preserving_adapter.py', MANIFEST, NORMALIZATION,
              ROOT / 'runs/train_uatd_area_heads.py', ROOT / 'runs/recompute_uatd_area_results.py',
              ROOT / 'sonar_encoder/training/uatd_head.py', ROOT / 'sonar_encoder/training/readout.py',
              RUN / 'cpu_red.log', RUN / 'cpu_green.log', RUN / 'cpu_test_receipt.json', RUN / 'source_review_receipt.json']
    references = {}
    for parent, closed_name in ((T0.parent, 'post_completion_check.json'), (REAL.parent, 'completion.json'),
                                (BASE / 'cls_global_readout_v1', 'completion.json')):
        result = json.loads((parent / 'results.json').read_text())
        verified = json.loads((parent / 'verification.json').read_text())
        completed = json.loads((parent / closed_name).read_text())
        assert digest(parent / 'results.json') == verified['results_sha256'] == completed['results_sha256']
        assert digest(parent / 'verification.json') == completed['verification_sha256']
        references[parent.name] = {'results': str(parent / 'results.json'), 'results_sha256': digest(parent / 'results.json')}
        inputs += [parent / n for n in ('results.json', 'verification.json', closed_name)]
    origins = {}
    settings = json.loads((T0 / 'protocol.json').read_text())['heads']
    for state, origin in (('t0', T0), ('real', REAL)):
        child = json.loads((origin / 'protocol.json').read_text())
        cache = json.loads((origin / 'cache_manifest.json').read_text())
        folder = origin / ('heads_' + state)
        report = json.loads((folder / 'report.json').read_text())
        assert child['prepared_manifest'] == str(MANIFEST) and child['heads'] == settings
        assert digest(MANIFEST) == child['inputs_sha256'][str(MANIFEST)]
        assert cache['row_ids'] == ids
        assert cache['model_state_sha256_before'] == cache['model_state_sha256_after'] == STATES[state]
        assert cache['protocol_sha256'] == digest(origin / 'protocol.json')
        assert cache['parent_protocol_sha256'] == digest(origin.parent / 'protocol.json')
        assert cache['normalization_sha256'] == digest(NORMALIZATION)
        assert report['provenance']['cache_manifest_sha256'] == digest(origin / 'cache_manifest.json')
        assert report['initial_head_sha256'] == json.loads((T0 / 'heads_t0/report.json').read_text())['initial_head_sha256']
        origins[state] = {'encoder_state_sha256': STATES[state], 'features': {}, 'headers': {},
                          'heads_report': str(folder / 'report.json'), 'initial_head_sha256': report['initial_head_sha256']}
        inputs += [origin / 'protocol.json', origin.parent / 'protocol.json', origin / 'cache_manifest.json',
                   folder / 'report.json', folder / 'metrics.jsonl']
        for split in (('train', 'validation') if state == 't0' else ('train',)):
            path = origin / f'{state}_{split}.npy'
            assert digest(path) == cache['cache_files_sha256'][path.name]
            array = np.load(path, mmap_mode='r')
            assert array.shape == (len(ids[split]), 384, 16, 16) and array.dtype == np.float32 and array.flags.c_contiguous
            origins[state]['features'][split] = str(path)
            origins[state]['headers'][split] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                                'sha256': cache['cache_files_sha256'][path.name]}
            del array
    head_path = T0 / 'heads_t0/best_global.pt'
    assert digest(head_path) == json.loads((T0 / 'heads_t0/report.json').read_text())['best_heads_sha256']['global']
    inputs.append(head_path)
    generator = torch.Generator().manual_seed(1)
    draws = torch.stack([draw_indices(generator) for _ in range(1000)]).numpy()
    order = hashlib.sha256()
    generator = torch.Generator().manual_seed(0)
    for _ in range(100):
        order.update(torch.randperm(6051, generator=generator).numpy().tobytes())
    initial = make_adapter()
    assert sum(p.numel() for p in initial.parameters()) == 73824
    torch.manual_seed(0)
    global_head, dense_head = nn.Linear(384, 10), nn.Conv2d(384, 50, 1)
    with torch.no_grad():
        dense_head.bias[:10].fill_(math.log(.01 / .99))
    initial_heads = {'global': state_digest(global_head.state_dict()), 'dense': state_digest(dense_head.state_dict())}
    assert initial_heads == origins['t0']['initial_head_sha256']
    for path in (Path(__file__), ROOT / 'tests/test_readout_preserving_adapter.py'):
        shutil.copyfile(path, RUN / path.name)
    cache_bytes = 6051 * 384 * 256 * 4
    static = {'largest_parent_mapped_bytes_unlabeled': 2 * cache_bytes,
              'unlabeled_CPU_torch_and_batch_allowance_bytes': 1536 * 1024**2,
              'downstream_base_GPU_bytes': 7556 * 384 * 256 * 4,
              'downstream_GPU_batch_targets_workspace_allowance_bytes': 1536 * 1024**2,
              'new_disk_upper_bound_bytes': 64 * 1024**2,
              'disk_bound_basis': '3 dense best predictions <=100/image (12.64MB numeric plus <=8MB zip overhead), 3 mean predictions <0.2MB; A/B last+optimizers, 4 fitted best states incl C adapter and atomic temp <8MB; all traces/snapshots/receipts <16MB. No feature/target/body file.'}
    assert 2 * cache_bytes + static['unlabeled_CPU_torch_and_batch_allowance_bytes'] <= 8 * 1024**3
    assert static['downstream_base_GPU_bytes'] + static['downstream_GPU_batch_targets_workspace_allowance_bytes'] <= 8 * 1024**3
    protocol = {'scope': 'Single source-domain feature-distillation usefulness and capacity pilot; no novelty, generic encoder, external-transfer or causal claim.',
                'prepared_manifest': str(MANIFEST), 'row_ids': ids, 'class_names': manifest['class_names'],
                'origins': origins, 'references': references, 'heads': settings, 'frozen_T0_mean_head': str(head_path),
                'adapter': {'structure': 'CPU seed0 384->96 biasTrue; GELU exact; 96->384 biasFalse zero weight; shared per token after final LN. No LN/dropout/pos/scale.',
                            'parameters': 73824, 'initial_sha256': state_digest(initial.state_dict()),
                            'A': 'Z0+Ph(Z0)', 'B': 'Z0+h(Z0)', 'C': 'Z0+PhC(Z0), supervised hC+dense jointly fit',
                            'P': 'Per image per channel subtract FP32 CHW spatial mean, same field/token/channel ordering.',
                            'CLS': 'Base unchanged, only historical result reference; not fed into h or newly evaluated.'},
                'unlabeled': {'train_count': 6051, 'validation_used': False, 'steps': 1000, 'batch_size': 32,
                              'target': 'FP32 parents -> CPU FP64; separately subtract own CHW spatial means; real residual minus base residual; once cast FP32. No persisted target.',
                              'loss': 'SmoothL1 beta1 all B/channel/token mean: A=Ph; B=h; target stopgrad.',
                              'sampling': 'Private CPU Generator seed1 randint(0,6051,(32,)) replacement; actual indices recorded, A/B identical.',
                              'order_sha256': hashlib.sha256(draws.tobytes()).hexdigest(),
                              'successful_exposures_each': 32000, 'unique_expected': int(np.unique(draws).size),
                              'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'betas': [.9, .999], 'eps': 1e-8},
                              'precision': 'FP32 TF32off no AMP', 'schedule': 'constant LR, no clip; fixed step1000 last no validation selection'},
                'initial_head_sha256': initial_heads, 'expected_head_order_sha256': order.hexdigest(),
                'downstream': 'A dense frozen h; B mean+dense frozen h, heads independent; C projected h+dense supervised same AdamW. CPU Linear then Conv init, h separate fork. 100epoch/19000updates/eval5 earliest strictbest each task.',
                'compatibility': {'arms': ['a', 'c'], 'C_state': 'selected best joint hC+dense', 'population': 'all6051train+1505validation',
                                  'actual_readout': 'FP32 adapted final field CHW mean vs FP32 parent mean; no bypass.',
                                  'absolute_sanity_tolerance': 1e-4, 'distribution_quantiles': [0, .5, .9, .99, 1],
                                  'validation_head': 'Original frozen T0 best mean head applied to actual adapted means; no fitting; save/recompute predictions'},
                'metrics': 'A/B/C dense and Bmean best primary and epoch100 log auxiliary; A/C frozenhead mean compatibility; full T0/REAL reference; no composite or interface selection.',
                'limits': ['6051 training feature distillation is additional unlabeled exposure, not original1024-only SSL or equal compute/data.',
                           'Projection also adds nonlocal mean computation; A/B difference not purely semantic.',
                           'A/C mean invariance is real-arithmetic identity, not FP32 bit equality or all semantic preservation.',
                           'A<=C cannot support teacher usefulness beyond this capacity control. Negative A does not license more width/steps.',
                           'Single seed source UATD development only; no external dense task or publication/novelty proof; old NO_GO/C1/C2 unchanged.'],
                'budgets': {'phase_seconds': LIMITS, 'total_seconds': 3780, 'cpu_threads': 4, 'gpu_index': 0,
                            'rss_bytes': 8 * 1024**3, 'gpu_reserved_bytes': 8 * 1024**3, 'new_bytes': 128 * 1024**2},
                'static_resources': static, 'runtime': 'cudnn benchmarkFalse, deterministic algorithmsTrue, CUBLAS_WORKSPACE_CONFIG=:4096:8; TF32off',
                'stop': 'Six saved-pred NumPy checks, full-population A/C compatibility, source/resource closure. No external data/new SSL/width/seed/step search.',
                'input_sha256': {str(p): digest(p) for p in dict.fromkeys(inputs)},
                'cache_sha256': {v['features'][s]: v['headers'][s]['sha256'] for v in origins.values() for s in v['features']},
                'snapshot_sha256': {str(RUN / p.name): digest(RUN / p.name) for p in (Path(__file__), ROOT / 'tests/test_readout_preserving_adapter.py')},
                'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
    write_json(RUN / 'protocol.json', protocol)
    print(json.dumps({'status': 'FROZEN_BEFORE_ADAPTER_AND_HEAD_UPDATES', 'protocol_sha256': digest(RUN / 'protocol.json'),
                      'initial_adapter_sha256': protocol['adapter']['initial_sha256'], 'static_resources': static}), flush=True)


def fit_adapter(arm):
    protocol = json.loads((RUN / 'protocol.json').read_text())
    sources = {name: np.load(origin['features']['train'], mmap_mode='r') for name, origin in protocol['origins'].items()}
    adapter = make_adapter()
    assert state_digest(adapter.state_dict()) == protocol['adapter']['initial_sha256']
    adapter = adapter.to('cuda:0').train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8)
    generator = torch.Generator().manual_seed(1)
    order_hash = hashlib.sha256()
    counts = np.zeros(6051, dtype=np.int64)
    losses = []
    output = RUN / ('adapter_' + arm)
    output.mkdir()
    with (output / 'trace.jsonl').open('w') as log:
        for step in range(1, 1001):
            indices = draw_indices(generator).numpy()
            order_hash.update(indices.tobytes())
            np.add.at(counts, indices, 1)
            base, real = sources['t0'][indices], sources['real'][indices]
            target = torch.from_numpy(paired_target(base, real)).to('cuda:0')
            field = torch.from_numpy(base).to('cuda:0')
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(token_delta(adapter, field, arm == 'a'), target, beta=1., reduction='mean')
            assert torch.isfinite(loss), f'Nonfinite distillation loss at {arm} step{step}'
            loss.backward()
            optimizer.step()
            measured = float(loss.detach())
            losses.append(measured)
            log.write(json.dumps({'step': step, 'actual_train_indices': indices.tolist(), 'loss': measured}) + '\n')
            if step % 100 == 0:
                log.flush()
                print(json.dumps({'arm': arm, 'success': step, 'loss': measured, 'resources': check_resources()}), flush=True)
    assert int(counts.sum()) == 32000 and order_hash.hexdigest() == protocol['unlabeled']['order_sha256']
    assert int(np.count_nonzero(counts)) == protocol['unlabeled']['unique_expected']
    assert all(torch.isfinite(value).all() for value in adapter.state_dict().values())
    final_sha = state_digest(adapter.state_dict())
    saved = {'adapter': adapter.state_dict(), 'optimizer': optimizer.state_dict(), 'step': 1000,
             'sampler_rng': generator.get_state(), 'order_sha256': order_hash.hexdigest(),
             'protocol_sha256': digest(RUN / 'protocol.json'), 'arm': arm, 'adapter_state_sha256': final_sha}
    torch.save(saved, output / 'last.tmp')
    check_resources()
    (output / 'last.tmp').replace(output / 'last.pt')
    write_json(output / 'report.json', {'status': 'COMPLETE_FIXED1000_UNLABELED_ADAPTER', 'arm': arm, 'steps': 1000,
               'exposures': int(counts.sum()), 'unique': int(np.count_nonzero(counts)),
               'repeat_exposures': int(counts.sum() - np.count_nonzero(counts)), 'exposure_counts_in_manifest_order': counts.tolist(),
               'actual_order_sha256': order_hash.hexdigest(), 'trace_sha256': digest(output / 'trace.jsonl'),
               'initial_adapter_sha256': protocol['adapter']['initial_sha256'], 'final_adapter_sha256': final_sha,
               'last_sha256': digest(output / 'last.pt'), 'initial_loss': losses[0], 'last_loss': losses[-1],
               'loss_interpretation': 'Fitting description only, not downstream benefit', 'validation_features_or_labels_used': False})


@torch.no_grad()
def evaluate(adapter, dense, global_head, features, targets, projected):
    adapter.eval()
    dense.eval()
    if global_head is not None:
        global_head.eval()
    detections, scores = [], []
    for start in range(0, len(features), 32):
        base = features[start:start + 32]
        field = base + token_delta(adapter, base, projected)
        detections.extend(decode_detections(dense(field)))
        if global_head is not None:
            scores.append(global_head(field.mean((-2, -1))).sigmoid().cpu())
    metrics = {'dense': detection_metrics(detections, targets)}
    predictions = {'dense': detections}
    if global_head is not None:
        predictions['global'] = torch.cat(scores)
        metrics['global'] = multilabel_metrics(predictions['global'], global_targets(targets, 'cpu'))
    return metrics, predictions


@torch.no_grad()
def check_mean_compatibility(adapter, features, targets, protocol, output, provenance):
    adapter.eval()
    old = torch.load(protocol['frozen_T0_mean_head'], map_location='cpu', weights_only=True)
    frozen = nn.Linear(384, 10)
    frozen.load_state_dict(old['head'])
    frozen = frozen.to('cuda:0').eval().requires_grad_(False)
    before = state_digest(frozen.state_dict())
    distribution, scores = {}, []
    for split in ('train', 'validation'):
        errors = []
        for start in range(0, len(features[split]), 32):
            base = features[split][start:start + 32]
            field = base + token_delta(adapter, base, True)
            actual_mean = field.mean((-2, -1))
            errors.append((actual_mean - base.mean((-2, -1))).abs().cpu())
            if split == 'validation':
                scores.append(frozen(actual_mean).sigmoid().cpu())
        error = torch.cat(errors).numpy()
        assert np.isfinite(error).all()
        summary = {'shape': list(error.shape), 'raw_FP32_abs_max': float(error.max()),
                   'raw_FP32_abs_mean': float(error.mean(dtype=np.float64)),
                   'absolute_quantiles': np.quantile(error, protocol['compatibility']['distribution_quantiles']).tolist(),
                   'per_channel_abs_max': error.max(axis=0).tolist()}
        distribution[split] = summary
        assert summary['raw_FP32_abs_max'] <= protocol['compatibility']['absolute_sanity_tolerance'], summary
    scores = torch.cat(scores)
    metrics = multilabel_metrics(scores, global_targets(targets['validation'], 'cpu'))
    assert state_digest(frozen.state_dict()) == before
    torch.save({'row_ids': protocol['row_ids']['validation'], 'predictions': scores,
                'metrics': metrics, 'provenance': provenance}, output / 'frozen_t0_mean_validation_predictions.pt')
    write_json(output / 'mean_compatibility.json', {'status': 'PASS_ACTUAL_FIELD_MEAN_SANITY',
               'adapter_state_sha256': state_digest(adapter.state_dict()), 'population': distribution,
               'sanity_tolerance': protocol['compatibility']['absolute_sanity_tolerance'], 'metrics': metrics,
               'frozen_head_state_sha256_before_after': before, 'head_origin': protocol['frozen_T0_mean_head'],
               'prediction_sha256': digest(output / 'frozen_t0_mean_validation_predictions.pt'),
               'interpretation': 'Frozen-head numerical compatibility, not newly fitted global or semantics proof.'})


def fit_heads(arm):
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    features, targets = {}, {}
    for split in ('train', 'validation'):
        features[split] = torch.from_numpy(np.load(protocol['origins']['t0']['features'][split], mmap_mode='c')).to('cuda:0')
        targets[split] = [{'boxes': torch.tensor(row['boxes'], dtype=torch.float32),
                           'labels': torch.tensor(row['labels'], dtype=torch.int64)} for row in manifest[split]]
    adapter = make_adapter()
    initial_adapter = state_digest(adapter.state_dict())
    assert initial_adapter == protocol['adapter']['initial_sha256']
    if arm in ('a', 'b'):
        saved = torch.load(RUN / ('adapter_' + arm) / 'last.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == 1000 and saved['arm'] == arm and saved['protocol_sha256'] == digest(RUN / 'protocol.json')
        adapter.load_state_dict(saved['adapter'])
        assert state_digest(adapter.state_dict()) == saved['adapter_state_sha256']
        adapter.requires_grad_(False)
    adapter = adapter.to('cuda:0')
    fixed_adapter_sha = state_digest(adapter.state_dict())
    torch.manual_seed(0)
    global_head = nn.Linear(384, 10).to('cuda:0')
    dense = nn.Conv2d(384, 50, 1).to('cuda:0')
    with torch.no_grad():
        dense.bias[:10].fill_(math.log(.01 / .99))
    initial_heads = {'global': state_digest(global_head.state_dict()), 'dense': state_digest(dense.state_dict())}
    assert initial_heads == protocol['initial_head_sha256']
    if arm != 'b':
        del global_head
        global_head = None
    train_boxes = tuple(v.to('cuda:0') for v in box_targets(targets['train'], 'cpu'))
    train_presence = global_targets(targets['train'], 'cpu').to('cuda:0') if arm == 'b' else None
    parameters = list(dense.parameters())
    if arm == 'b':
        parameters = list(global_head.parameters()) + parameters
    elif arm == 'c':
        parameters = list(adapter.parameters()) + parameters
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8)
    generator = torch.Generator().manual_seed(0)
    order_hash = hashlib.sha256()
    output = RUN / ('heads_' + arm)
    output.mkdir()
    best = {name: {'score': -1., 'epoch': 0} for name in (('global', 'dense') if arm == 'b' else ('dense',))}
    provenance = {'arm': arm, 'protocol_sha256': digest(RUN / 'protocol.json'), 'base_encoder_sha256': STATES['t0'],
                  'adapter_start_sha256': fixed_adapter_sha, 'adapter_trainable': arm == 'c', 'base_encoder_in_optimizer': False}
    updates = 0
    with (output / 'metrics.jsonl').open('w') as log:
        for epoch in range(1, 101):
            adapter.train(arm == 'c')
            dense.train()
            if arm == 'b':
                global_head.train()
            order = torch.randperm(6051, generator=generator)
            order_hash.update(order.numpy().tobytes())
            losses = 0.
            for indices in order.split(32):
                ix = indices.to('cuda:0')
                base = features['train'][ix]
                optimizer.zero_grad(set_to_none=True)
                field = base + token_delta(adapter, base, arm != 'b')
                loss = dense_loss(dense(field), tuple(v[ix] for v in train_boxes))
                if arm == 'b':
                    loss = F.binary_cross_entropy_with_logits(global_head(field.mean((-2, -1))), train_presence[ix]) + loss
                assert torch.isfinite(loss), f'Nonfinite task loss at {arm} epoch{epoch} update{updates + 1}'
                loss.backward()
                optimizer.step()
                updates += 1
                losses += float(loss.detach()) * len(indices)
            row = {'epoch': epoch, 'updates': updates, 'train_loss': losses / 6051}
            if epoch % 5 == 0:
                metrics, predictions = evaluate(adapter, dense, global_head, features['validation'], targets['validation'], arm != 'b')
                row['validation'] = metrics
                for name, head, key in (('dense', dense, 'map50'), ('global', global_head, 'map')):
                    if name not in best:
                        continue
                    if metrics[name][key] > best[name]['score']:
                        best[name] = {'score': metrics[name][key], 'epoch': epoch}
                        state = {'head': head.state_dict(), 'epoch': epoch, 'metrics': metrics[name], 'provenance': provenance}
                        if arm == 'c':
                            state['adapter'] = adapter.state_dict()
                        torch.save(state, output / ('best_' + name + '.pt'))
                        torch.save({'row_ids': protocol['row_ids']['validation'], 'predictions': predictions[name]},
                                   output / ('best_' + name + '_validation_predictions.pt'))
                print(json.dumps({'arm': arm, 'epoch': epoch, 'updates': updates, 'metrics': {k: v['map50' if k == 'dense' else 'map'] for k, v in metrics.items()},
                                  'resources': check_resources()}), flush=True)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
    assert updates == 19000 and order_hash.hexdigest() == protocol['expected_head_order_sha256']
    if arm != 'c':
        assert state_digest(adapter.state_dict()) == fixed_adapter_sha
    final_adapter_sha = state_digest(adapter.state_dict())
    report = {'status': 'COMPLETE_FIXED100EPOCH_UATD_HEADS', 'arm': arm, 'updates': updates, 'epochs': 100,
              'selected': best, 'epoch100_metrics': row['validation'], 'initial_head_sha256': initial_heads,
              'initial_zero_adapter_sha256': initial_adapter, 'adapter_start_sha256': fixed_adapter_sha,
              'final_adapter_sha256': final_adapter_sha, 'actual_private_order_sha256': order_hash.hexdigest(),
              'provenance': provenance, 'class_names': protocol['class_names'],
              'best_heads_sha256': {k: digest(output / ('best_' + k + '.pt')) for k in best},
              'best_predictions_sha256': {k: digest(output / ('best_' + k + '_validation_predictions.pt')) for k in best}}
    if arm in ('a', 'c'):
        if arm == 'c':
            selected = torch.load(output / 'best_dense.pt', map_location='cpu', weights_only=True)
            adapter.load_state_dict(selected['adapter'])
            report['selected_joint_adapter_sha256'] = state_digest(adapter.state_dict())
        check_mean_compatibility(adapter, features, targets, protocol, output, provenance)
    write_json(output / 'report.json', report)


def close():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    boxes = [np.asarray(r['boxes'], dtype=np.float32) for r in manifest['validation']]
    labels = [np.asarray(r['labels'], dtype=np.int64) for r in manifest['validation']]
    previous = json.loads((REAL.parent / 'results.json').read_text())
    cls_reference = json.loads((BASE / 'cls_global_readout_v1/results.json').read_text())
    states = {}
    for name in ('t0', 'real'):
        old = previous['states'][name]
        states[name] = {key: {'best': old['best'][task], 'epoch100': old['epoch100_log_only'][task],
                            'selected': old['selected'][task], 'newly_evaluated': False}
                        for key, task in (('mean', 'global'), ('dense', 'dense'))}
        states[name]['CLS_reference_only'] = {'source': protocol['references']['cls_global_readout_v1'],
                                             'values': cls_reference['states'][name]['cls'], 'newly_evaluated': False}
    checks, outputs, reports = {}, {}, {}
    actual_orders = []
    for arm in ('a', 'b'):
        folder = RUN / ('adapter_' + arm)
        report = json.loads((folder / 'report.json').read_text())
        records = jsonl(folder / 'trace.jsonl')
        indices = np.asarray([row['actual_train_indices'] for row in records], dtype=np.int64)
        assert [r['step'] for r in records] == list(range(1, 1001)) and indices.shape == (1000, 32)
        actual_orders.append(indices)
        assert hashlib.sha256(indices.tobytes()).hexdigest() == report['actual_order_sha256'] == protocol['unlabeled']['order_sha256']
        counts = np.bincount(indices.flatten(), minlength=6051)
        assert counts.tolist() == report['exposure_counts_in_manifest_order'] and counts.sum() == report['exposures'] == 32000
        assert np.count_nonzero(counts) == report['unique']
        assert digest(folder / 'last.pt') == report['last_sha256'] and digest(folder / 'trace.jsonl') == report['trace_sha256']
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == 1000 and saved['protocol_sha256'] == digest(RUN / 'protocol.json')
        assert state_digest(saved['adapter']) == saved['adapter_state_sha256'] == report['final_adapter_sha256']
        reports['adapter_' + arm] = report
    assert np.array_equal(*actual_orders)
    for arm in ('a', 'b', 'c'):
        folder = RUN / ('heads_' + arm)
        report = json.loads((folder / 'report.json').read_text())
        log = jsonl(folder / 'metrics.jsonl')
        assert len(log) == 100 and [r['updates'] for r in log] == [190 * e for e in range(1, 101)]
        assert report['updates'] == 19000 and report['actual_private_order_sha256'] == protocol['expected_head_order_sha256']
        assert report['initial_head_sha256'] == protocol['initial_head_sha256']
        assert report['initial_zero_adapter_sha256'] == protocol['adapter']['initial_sha256']
        assert report['provenance']['protocol_sha256'] == digest(RUN / 'protocol.json')
        assert report['epoch100_metrics'] == log[-1]['validation']
        if arm != 'c':
            assert report['adapter_start_sha256'] == report['final_adapter_sha256'] == reports['adapter_' + arm]['final_adapter_sha256']
        reports['heads_' + arm] = report
        states[arm] = {}
        for task in report['selected']:
            key = 'map50' if task == 'dense' else 'map'
            selected_row = max((r for r in log if 'validation' in r), key=lambda r: r['validation'][task][key])
            saved_path = folder / ('best_' + task + '.pt')
            pred_path = folder / ('best_' + task + '_validation_predictions.pt')
            saved = torch.load(saved_path, map_location='cpu', weights_only=True)
            predicted = torch.load(pred_path, map_location='cpu', weights_only=True)
            assert digest(saved_path) == report['best_heads_sha256'][task]
            assert digest(pred_path) == report['best_predictions_sha256'][task]
            assert saved['epoch'] == selected_row['epoch'] == report['selected'][task]['epoch']
            assert saved['metrics'] == selected_row['validation'][task] and saved['provenance'] == report['provenance']
            assert saved['metrics'][key] == report['selected'][task]['score']
            assert predicted['row_ids'] == protocol['row_ids']['validation']
            if task == 'dense':
                measured = dense_metrics([{k: v.numpy() for k, v in p.items()} for p in predicted['predictions']], boxes, labels)
            else:
                measured = global_metrics(predicted['predictions'].numpy(), labels)
            name = arm + '_' + task
            checks[name] = compare_metrics(measured, saved['metrics'])
            assert all(v['pass'] for v in checks[name].values()), checks[name]
            states[arm]['dense' if task == 'dense' else 'mean'] = {'best': measured, 'selected': report['selected'][task],
                  'epoch100': report['epoch100_metrics'][task],
                  'epoch100_evidence': 'same_saved_best_independently_rescored' if saved['epoch'] == 100 else 'training_log_and_report_only',
                  'saved_head_sha256': digest(saved_path), 'saved_predictions_sha256': digest(pred_path)}
            if arm == 'c':
                assert state_digest(saved['adapter']) == report['selected_joint_adapter_sha256']
        if arm in ('a', 'c'):
            compat = json.loads((folder / 'mean_compatibility.json').read_text())
            expected_sha = report['final_adapter_sha256'] if arm == 'a' else report['selected_joint_adapter_sha256']
            assert compat['adapter_state_sha256'] == expected_sha
            assert [compat['population'][s]['shape'] for s in ('train', 'validation')] == [[6051, 384], [1505, 384]]
            assert all(compat['population'][s]['raw_FP32_abs_max'] <= 1e-4 for s in ('train', 'validation'))
            pred_path = folder / 'frozen_t0_mean_validation_predictions.pt'
            assert digest(pred_path) == compat['prediction_sha256']
            predicted = torch.load(pred_path, map_location='cpu', weights_only=True)
            assert predicted['row_ids'] == protocol['row_ids']['validation'] and predicted['provenance'] == report['provenance']
            measured = global_metrics(predicted['predictions'].numpy(), labels)
            checks[arm + '_frozen_t0_mean'] = compare_metrics(measured, compat['metrics'])
            assert all(v['pass'] for v in checks[arm + '_frozen_t0_mean'].values())
            states[arm]['frozen_t0_mean_compatibility'] = {'metrics': measured, 'population': compat['population'],
                      'delta_AP_vs_T0': measured['map'] - states['t0']['mean']['best']['map'], 'new_head_fit': False,
                      'FP32_not_claimed_bitwise': True}
        states[arm]['CLS'] = 'Unchanged base, original T0 source only; not newly evaluated'
    assert len(checks) == 6
    bindings = {**protocol['input_sha256'], **protocol['cache_sha256'], **protocol['snapshot_sha256']}
    for path, expected in bindings.items():
        assert digest(Path(path)) == expected, path
    for path in RUN.rglob('*'):
        if path.is_file() and path.suffix in ('.pt', '.json', '.jsonl') and path.name != 'compute_ledger.jsonl':
            outputs[str(path)] = digest(path)
    differences = {'dense': {f'{arm.upper()}_minus_{parent.upper()}':
                   {phase: states[arm]['dense'][phase]['map50'] - states[parent]['dense'][phase]['map50'] for phase in ('best', 'epoch100')}
                   for arm in ('a', 'b', 'c') for parent in ('t0', 'real')},
                   'Bmean': {f'B_minus_{p.upper()}': {phase: states['b']['mean'][phase]['map'] - states[p]['mean'][phase]['map'] for phase in ('best', 'epoch100')} for p in ('t0', 'real')},
                   'A_dense_minus_C': {phase: states['a']['dense'][phase]['map50'] - states['c']['dense'][phase]['map50'] for phase in ('best', 'epoch100')},
                   'A_dense_minus_B': {phase: states['a']['dense'][phase]['map50'] - states['b']['dense'][phase]['map50'] for phase in ('best', 'epoch100')}}
    write_json(RUN / 'results.json', {'status': 'COMPLETE_BOUNDED_ADAPTER_CAPACITY_PILOT', 'scope': protocol['scope'],
               'states': states, 'differences_within_interface': differences, 'unlabeled_actual': {arm: reports['adapter_' + arm] for arm in ('a', 'b')},
               'actual_A_B_indices_exactly_equal': True, 'new_encoder_forwards': 0, 'new_feature_or_target_cache_files': 0,
               'new_fitted_task_heads': 4, 'unlabeled_adapters': 2, 'supervised_adapters': 1, 'limits': protocol['limits'],
               'next_experiment_started': False, 'old_NO_GO_unchanged': True})
    write_json(RUN / 'verification.json', {'status': 'PASS_SIX_SAVED_PRED_NUMPY_MEAN_POPULATION_AND_BINDINGS',
               'checks': checks, 'closed_input_sha256': bindings, 'closed_output_sha256': outputs,
               'results_sha256': digest(RUN / 'results.json'), 'protocol_sha256': digest(RUN / 'protocol.json')})
    print(json.dumps({'status': 'CLOSED_SIX_PREDICTIONS', 'differences': differences, 'resources': check_resources()}), flush=True)


def timeout_phase(signum, frame):
    raise TimeoutError('Frozen adapter pilot compute budget exhausted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=tuple(LIMITS))
    args = parser.parse_args()
    ledger_path = RUN / 'compute_ledger.jsonl'
    ledger = jsonl(ledger_path)
    prior = sum(r['seconds'] for r in ledger)
    phase_prior = sum(r['seconds'] for r in ledger if r['phase'] == args.phase)
    available = min(LIMITS[args.phase] - phase_prior, 3780 - prior)
    signal.signal(signal.SIGALRM, timeout_phase)
    signal.setitimer(signal.ITIMER_REAL, available - (time.monotonic() - PROCESS_STARTED))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if args.phase != 'prepare':
        protocol = json.loads((RUN / 'protocol.json').read_text())
        assert all(digest(Path(p)) == sha for p, sha in protocol['input_sha256'].items())
    print(json.dumps({'phase': args.phase, 'pid': os.getpid(), 'prior_compute_seconds': prior, 'available_seconds': available}), flush=True)
    status = 'FAILED'
    try:
        if args.phase == 'prepare':
            prepare()
        elif args.phase.startswith('adapter_'):
            fit_adapter(args.phase[-1])
        elif args.phase.startswith('heads_'):
            fit_heads(args.phase[-1])
        else:
            close()
        check_resources()
        status = 'PASS'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - PROCESS_STARTED
        measured = resources()
        record = {'phase': args.phase, 'status': status, 'pid': os.getpid(), 'seconds': elapsed,
                  'total_compute_seconds': prior + elapsed, 'resources': measured}
        with ledger_path.open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        write_json(RUN / (args.phase + '_receipt.json'), record)
        assert elapsed + phase_prior <= LIMITS[args.phase] and elapsed + prior <= 3780
        check_resources()
    if args.phase == 'close':
        write_json(RUN / 'completion.json', {'status': 'COMPLETE_WITHIN_CUMULATIVE_BUDGETS',
                   'protocol_sha256': digest(RUN / 'protocol.json'), 'results_sha256': digest(RUN / 'results.json'),
                   'verification_sha256': digest(RUN / 'verification.json'), 'ledger_sha256': digest(ledger_path),
                   'total_compute_seconds': prior + elapsed, 'resources': measured, 'next_stage_started': False})


if __name__ == '__main__':
    main()
