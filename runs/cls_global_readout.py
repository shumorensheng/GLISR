"""Fixed final-CLS readout coverage for frozen T0/S/REAL, without encoder training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs.recompute_uatd_area_results import compare_metrics, digest, global_metrics
from runs.ssl_global_keep import BASE, UATD, disk_bytes
from runs.ssl_real_domain_mix import INITIAL
from sonar_encoder.training.model import build_encoder, prepare_images
from sonar_encoder.training.readout import state_digest
from sonar_encoder.training.uatd_readout import native_sample
from sonar_encoder.training.uatd_head import global_targets, multilabel_metrics

REAL = BASE / 'ssl_real_only_v1'
SIM = BASE / 'ssl_real_domain_mix_v1'
T0 = BASE / 'ssl_global_keep_v1/t0'
STATES = ('t0', 'sim', 'real')
GLOBAL_INITIAL = 'd0d9eed1abe3acbb93be421ded5347c1ccb2a227de4d7b7c6e3a33324595d45a'


def read_cls_and_patches(encoder, samples, normalization):
    device = encoder.backbone.patch_embed.proj.weight.device
    complete = prepare_images(samples, normalization, None).to(device)
    output = encoder.backbone.forward_features(complete, masks=None)
    return output['x_norm_clstoken'], output['x_norm_patchtokens']


def prepare(run):
    started = time.monotonic()
    parent = json.loads((REAL / 'protocol.json').read_text())
    verification = json.loads((REAL / 'verification.json').read_text())
    completion = json.loads((REAL / 'completion.json').read_text())
    assert digest(REAL / 'results.json') == verification['results_sha256'] == completion['results_sha256']
    assert digest(REAL / 'verification.json') == completion['verification_sha256']
    assert verification['status'] == 'PASS_NUMPY_SAVED_BEST_AND_CLOSED_INPUT_BINDINGS'
    sim_verification = json.loads((SIM / 'verification.json').read_text())
    prior = json.loads((REAL / 'results.json').read_text())
    config = parent['config']
    origins = {'t0': T0, 'sim': SIM / 'sim', 'real': REAL / 'real'}
    sources = [Path(__file__), ROOT / 'tests/test_cls_global_readout.py',
               UATD / 'manifest.json', UATD / 'normalization.json', REAL / 'protocol.json',
               REAL / 'results.json', REAL / 'verification.json', REAL / 'completion.json',
               SIM / 'protocol.json', SIM / 'verification.json',
               ROOT / 'runs/recompute_uatd_area_results.py', ROOT / 'runs/train_uatd_area_heads.py',
               ROOT / 'runs/ssl_global_keep.py', ROOT / 'runs/ssl_real_domain_mix.py', Path(config['weights_path']),
               *sorted(Path(config['code_dir']).rglob('*.py')),
               *[ROOT / 'sonar_encoder/training' / name for name in
                 ('model.py', 'frontends.py', 'readout.py', 'uatd_readout.py', 'uatd_head.py', 'artifacts.py')],
               run / 'cpu_tests.log', run / 'cpu_red.log', run / 'preflight_receipt.json', run / 'source_review_receipt.json']
    manifest = json.loads((UATD / 'manifest.json').read_text())
    assert (len(manifest['train']), len(manifest['validation'])) == (6051, 1505)
    normal = json.loads((UATD / 'normalization.json').read_text())
    assert normal['source_manifest_sha256'] == digest(UATD / 'manifest.json')
    states = {}
    for state, origin in origins.items():
        cache = json.loads((origin / 'cache_manifest.json').read_text())
        head_report = json.loads((origin / f'heads_{state}/report.json').read_text())
        child = json.loads((origin / 'protocol.json').read_text())
        parent_path = BASE / 'ssl_global_keep_v1/protocol.json' if state == 't0' else SIM / 'protocol.json' if state == 'sim' else REAL / 'protocol.json'
        original_parent = json.loads(parent_path.read_text())
        sources.append(parent_path)
        assert cache['protocol_sha256'] == digest(origin / 'protocol.json')
        assert cache['parent_protocol_sha256'] == digest(parent_path)
        assert original_parent['feature_batch'] == 16 and child['heads'] == parent['heads']
        assert cache['normalization_sha256'] == digest(UATD / 'normalization.json')
        assert cache['model_state_sha256_before'] == cache['model_state_sha256_after'] == prior['states'][state]['encoder_state_sha256']
        assert head_report['initial_head_sha256']['global'] == GLOBAL_INITIAL
        sources.extend(origin / name for name in ('protocol.json', 'cache_manifest.json', f'heads_{state}/report.json',
                       f'heads_{state}/metrics.jsonl', f'heads_{state}/best_global.pt', f'heads_{state}/best_global_validation_predictions.pt'))
        patch_caches = {}
        for split in ('train', 'validation'):
            path = origin / f'{state}_{split}.npy'
            array = np.load(path, mmap_mode='r')
            assert array.shape == (len(manifest[split]), 384, 16, 16) and array.dtype == np.float32
            assert cache['row_ids'][split] == [row['row_id'] for row in manifest[split]]
            patch_caches[split] = {'path': str(path), 'previous_closed_sha256': cache['cache_files_sha256'][path.name]}
        checkpoint = None
        if state != 't0':
            report = json.loads((origin / 'report1000.json').read_text())
            checkpoint = origin / 'checkpoints/last.pt'
            assert digest(checkpoint) == report['last_sha256'] == cache['checkpoint_sha256']
            assert report['step'] == 1000 and report['initial_states_sha256']['student'] == INITIAL
            sources.extend((origin / 'report1000.json', checkpoint))
        else:
            assert cache['checkpoint_sha256'] is None and cache['initial_state_sha256'] == INITIAL
        states[state] = {'origin': str(origin), 'source_protocol_path': str(parent_path), 'checkpoint': str(checkpoint) if checkpoint else None,
                         'encoder_state_sha256': cache['model_state_sha256_after'], 'patch_caches': patch_caches,
                         'mean': prior['states'][state]}
    bindings = {}
    known = {**sim_verification['closed_inputs_sha256'], **verification['closed_inputs_sha256']}
    for path in sources:
        value = digest(path)
        if str(path) in known:
            assert value == known[str(path)], str(path)
        bindings[str(path)] = value
    order_rng = torch.Generator().manual_seed(parent['heads']['seed'])
    order_hash = hashlib.sha256()
    for _ in range(100):
        order_hash.update(torch.randperm(6051, generator=order_rng).numpy().tobytes())
    feature_bytes = 3 * 7556 * 384 * 4 + 6 * 128
    reserved = 16 * 1024**2
    assert feature_bytes + reserved < 128 * 1024**2
    protocol = {'scope': 'Does mean-global decline also occur in a fixed final-LN CLS384 global readout of T0/S/REAL?',
                'config': config, 'states': states, 'heads': parent['heads'], 'inputs_sha256': bindings,
                'prepared_manifest': str(UATD / 'manifest.json'), 'task_normalization': str(UATD / 'normalization.json'),
                'feature_batch': 16, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'new_sample_forwards_max': 22668, 'new_batch_forwards': 1422,
                'readout': 'Official final LayerNorm x_norm_clstoken384 only; no intermediate layers/concat/extra normalization. Complete native prepare_images AA224 gain1 FP32/TF32off/officialPE; encoders strictly frozen.',
                'patch_interface': {'indices': {'train': [0, 3025, 6050], 'validation': [0, 752, 1504]},
                    'absolute_FP32_tolerance': 1e-5, 'sampling': 'Same original16batch boundaries; same forward patch outputs, 18rows total. No score-driven selection or extra forward.'},
                'global_initial_sha256': GLOBAL_INITIAL, 'expected_private_order_sha256': order_hash.hexdigest(),
                'head_identity': 'CPU Linear384->10 first head, manual_seed0, no dense constructed; independent CPU generatorseed0 shuffle6051,split32,100epoch19000updates; original BCE/AdamW/eval5/earlieststrictmax.',
                'budgets': {'cpu_threads': 4, 'gpu_index': 0, 'prepare_seconds': 180, 'feature_seconds_total': 1200,
                            'head_seconds_per_state': 300, 'score_closing_seconds': 180, 'total_compute_seconds': 2700,
                            'rss_bytes': 8 * 1024**3, 'gpu_bytes': 8 * 1024**3, 'new_peak_bytes': 128 * 1024**2},
                'static_peak': {'six_CLS_npy_bytes': feature_bytes, 'metadata_heads_predictions_reserve': reserved,
                                'conservative_total_bytes': feature_bytes + reserved},
                'metrics': 'Six absolute mean/CLS×T0/S/REAL globalAP cells, S-T0 andREAL-T0 within each readout; best main/epoch100 auxiliary. No choosing representation per cell.',
                'limits': ['Same image-split/task-training-domain exploratory seed0 development; no Test or independent scene claim.',
                           'CLS preserved with mean decline weakens inference of overall global forgetting; two declining readouts still do not prove irrecoverable information or a loss rootcause.',
                           'Old dense/source byte closure cited only, not newly evaluated; old NO_GO/C1/C2/retentionFAIL unchanged.'],
                'stop': 'Three frozen CLS extractions/threeglobal-only heads/three NumPy best scores; no SSL/dense/new data/model search/ensemble.'}
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for state in STATES:
        (run / state).mkdir()
    for path in (Path(__file__), ROOT / 'tests/test_cls_global_readout.py'):
        shutil.copyfile(path, run / path.name)
    receipt = {'status': 'FROZEN_BEFORE_NEW_MODEL_COMPUTATION', 'input_bindings': len(bindings),
               'old_patch_payloads_not_rehashed': True, 'no_new_weights_saved': True,
               'wall_seconds': time.monotonic() - started, 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    assert receipt['wall_seconds'] < 180 and receipt['max_rss_bytes'] < 8 * 1024**3
    (run / 'prepare_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


@torch.no_grad()
def extract(run, state):
    started = time.monotonic()
    protocol = json.loads((run / 'protocol.json').read_text())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = protocol['config']
    origin = protocol['states'][state]
    torch.manual_seed(config['seed'])
    encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
    assert state_digest(encoder.state_dict()) == INITIAL
    if state != 't0':
        checkpoint_path = Path(origin['checkpoint'])
        assert digest(checkpoint_path) == protocol['inputs_sha256'][str(checkpoint_path)]
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['stream']['step'] == 1000 and checkpoint['initial_sha256'] == INITIAL
        assert checkpoint['protocol_sha256'] == digest(Path(origin['source_protocol_path']))
        encoder.load_state_dict(checkpoint['student'])
        del checkpoint
    encoder = encoder.to('cuda:0').eval().requires_grad_(False)
    before = state_digest(encoder.state_dict())
    assert before == origin['encoder_state_sha256']
    torch.cuda.reset_peak_memory_stats()
    normalization = json.loads(Path(protocol['task_normalization']).read_text())
    manifest = json.loads(Path(protocol['prepared_manifest']).read_text())
    prior_seconds = sum(json.loads((run / previous / 'cache_manifest.json').read_text())['resources']['wall_seconds']
                        for previous in STATES[:STATES.index(state)])
    cache_hashes, row_ids, checks, forwards, calls = {}, {}, [], 0, 0
    folder = run / state
    for split in ('train', 'validation'):
        rows = manifest[split]
        old = np.load(origin['patch_caches'][split]['path'], mmap_mode='r')
        path = folder / f'cls_{split}.npy'
        features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 384))
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            cls, patches = read_cls_and_patches(encoder, [native_sample(row) for row in batch], normalization)
            assert cls.shape == (len(batch), 384) and patches.shape == (len(batch), 256, 384)
            assert cls.dtype == patches.dtype == torch.float32 and torch.isfinite(cls).all()
            features[start:start + len(batch)] = cls.cpu().numpy()
            forwards += len(batch)
            calls += 1
            for index in protocol['patch_interface']['indices'][split]:
                if start <= index < start + len(batch):
                    actual = patches[index - start].transpose(0, 1).reshape(384, 16, 16).cpu().numpy()
                    error = float(np.abs(actual - old[index]).max())
                    check = {'split': split, 'index': index, 'row_id': rows[index]['row_id'],
                             'batch_start': start, 'batch_stop': start + len(batch), 'max_abs_error': error,
                             'tolerance': protocol['patch_interface']['absolute_FP32_tolerance']}
                    checks.append(check)
                    print(json.dumps({'state': state, 'patch_interface_check': check}), flush=True)
                    assert error <= check['tolerance'], f'Patch interface mismatch {state}/{split}/{index}; stop, preserve evidence, consult Astra.'
            if start % 800 == 0:
                print(json.dumps({'state': state, 'split': split, 'rows_done': start + len(batch)}), flush=True)
            assert prior_seconds + time.monotonic() - started < protocol['budgets']['feature_seconds_total']
        features.flush()
        del features, old
        cache_hashes[path.name] = digest(path)
        row_ids[split] = [row['row_id'] for row in rows]
    after = state_digest(encoder.state_dict())
    assert after == before and forwards == 7556 and calls == 474 and len(checks) == 6
    resources = {'wall_seconds': time.monotonic() - started,
                 'cumulative_feature_seconds': prior_seconds + time.monotonic() - started,
                 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved(), 'disk_bytes_after': disk_bytes(run)}
    assert resources['max_rss_bytes'] < 8 * 1024**3 and resources['max_gpu_reserved_bytes'] < 8 * 1024**3
    assert resources['cumulative_feature_seconds'] < 1200 and resources['disk_bytes_after'] < 128 * 1024**2
    record = {'status': 'COMPLETE_FROZEN_FINAL_CLS', 'state': state, 'protocol_sha256': digest(run / 'protocol.json'),
              'checkpoint_sha256': protocol['inputs_sha256'][origin['checkpoint']] if state != 't0' else None,
              'initial_state_sha256': INITIAL, 'model_state_sha256_before': before, 'model_state_sha256_after': after,
              'new_sample_forwards': forwards, 'new_batch_forwards': calls, 'patch_interface_checks': checks,
              'row_ids': row_ids, 'cache_files_sha256': cache_hashes, 'resources': resources}
    (folder / 'cache_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'state': state, 'status': record['status'], 'patch_errors': [row['max_abs_error'] for row in checks], 'resources': resources}), flush=True)


def train_global(run, state):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((run / 'protocol.json').read_text())
    folder = run / state
    cache_record = json.loads((folder / 'cache_manifest.json').read_text())
    assert cache_record['protocol_sha256'] == digest(run / 'protocol.json')
    manifest = json.loads(Path(protocol['prepared_manifest']).read_text())
    cache, targets = {}, {}
    for split in ('train', 'validation'):
        path = folder / f'cls_{split}.npy'
        assert digest(path) == cache_record['cache_files_sha256'][path.name]
        assert cache_record['row_ids'][split] == [row['row_id'] for row in manifest[split]]
        cache[split] = torch.from_numpy(np.load(path, mmap_mode='c')).to('cuda:0')
        targets[split] = global_targets([{'labels': torch.tensor(row['labels'])} for row in manifest[split]], 'cpu')
    train_presence = targets['train'].to('cuda:0')
    output = folder / 'heads_cls'
    output.mkdir()
    settings = protocol['heads']
    torch.manual_seed(settings['seed'])
    head = nn.Linear(384, 10).to('cuda:0')
    initial = state_digest(head.state_dict())
    assert initial == protocol['global_initial_sha256']
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    generator = torch.Generator().manual_seed(settings['seed'])
    order_trace = hashlib.sha256()
    best = {'score': -1., 'epoch': 0}
    updates = 0
    provenance = {'state': state, 'readout': 'final_LN_CLS384', 'protocol_sha256': digest(run / 'protocol.json'),
                  'cache_manifest_sha256': digest(folder / 'cache_manifest.json'),
                  'encoder_state_sha256': cache_record['model_state_sha256_after'], 'encoder_parameters_in_optimizer': False}
    with (output / 'metrics.jsonl').open('w', buffering=1) as log:
        for epoch in range(1, settings['epochs'] + 1):
            head.train()
            order = torch.randperm(len(cache['train']), generator=generator)
            order_trace.update(order.numpy().tobytes())
            loss_sum = 0.
            for indices in order.split(settings['batch_size']):
                device_indices = indices.to('cuda:0')
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(head(cache['train'][device_indices]), train_presence[device_indices])
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
                updates += 1
            row = {'epoch': epoch, 'updates': updates, 'train_loss': loss_sum / len(order)}
            if epoch % settings['evaluate_every_epochs'] == 0:
                head.eval()
                with torch.no_grad():
                    scores = torch.cat([head(cache['validation'][start:start + 32]).sigmoid().cpu()
                                        for start in range(0, len(cache['validation']), 32)])
                metrics = multilabel_metrics(scores, targets['validation'])
                row['validation'] = metrics
                if metrics['map'] > best['score']:
                    best = {'score': metrics['map'], 'epoch': epoch}
                    torch.save({'head': head.state_dict(), 'epoch': epoch, 'metrics': metrics, 'provenance': provenance}, output / 'best_global.pt')
                    torch.save({'row_ids': cache_record['row_ids']['validation'], 'predictions': scores}, output / 'best_global_validation_predictions.pt')
                print(json.dumps({'state': state, 'epoch': epoch, 'CLS_global_map': metrics['map']}), flush=True)
            log.write(json.dumps(row) + '\n')
            assert time.monotonic() - started < protocol['budgets']['head_seconds_per_state']
    assert order_trace.hexdigest() == protocol['expected_private_order_sha256']
    resources = {'wall_seconds': time.monotonic() - started,
                 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved()}
    assert resources['max_rss_bytes'] < 8 * 1024**3 and resources['max_gpu_reserved_bytes'] < 8 * 1024**3
    assert resources['wall_seconds'] < 300
    report = {'status': 'COMPLETE_FIXED_CLS_GLOBAL_HEAD', 'state': state, 'epochs': 100, 'updates': updates,
              'selected': best, 'epoch100_metrics_log_only': row['validation'], 'initial_head_sha256': initial,
              'actual_order_sha256': order_trace.hexdigest(), 'original_private_order_sha256': protocol['expected_private_order_sha256'],
              'provenance': provenance, 'resources': resources, 'best_head_sha256': digest(output / 'best_global.pt'),
              'best_predictions_sha256': digest(output / 'best_global_validation_predictions.pt'), 'new_dense_fit': False, 'test_evaluated': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def score(run):
    started = time.monotonic()
    torch.set_num_threads(4)
    protocol = json.loads((run / 'protocol.json').read_text())
    bindings = {}
    for name, expected in protocol['inputs_sha256'].items():
        actual = digest(Path(name))
        assert actual == expected, name
        bindings[name] = actual
    manifest = json.loads(Path(protocol['prepared_manifest']).read_text())
    validation_ids = [row['row_id'] for row in manifest['validation']]
    labels = [np.asarray(row['labels'], dtype=np.int64) for row in manifest['validation']]
    states, checks, heads, orders = {}, {}, {}, {}
    forwards = 0
    for state in STATES:
        folder = run / state
        cache = json.loads((folder / 'cache_manifest.json').read_text())
        output = folder / 'heads_cls'
        report = json.loads((output / 'report.json').read_text())
        log = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        assert cache['status'] == 'COMPLETE_FROZEN_FINAL_CLS' and cache['protocol_sha256'] == digest(run / 'protocol.json')
        assert cache['model_state_sha256_before'] == cache['model_state_sha256_after'] == protocol['states'][state]['encoder_state_sha256']
        assert report['status'] == 'COMPLETE_FIXED_CLS_GLOBAL_HEAD' and report['epochs'] == len(log) == 100 and report['updates'] == 19000
        assert report['new_dense_fit'] is False and report['test_evaluated'] is False
        for epoch, row in enumerate(log, 1):
            assert row['epoch'] == epoch and row['updates'] == epoch * 190
            assert ('validation' in row) == (epoch % 5 == 0)
        assert report['epoch100_metrics_log_only'] == log[-1]['validation']
        best = {'score': -1., 'epoch': 0}
        for row in log:
            if 'validation' in row and row['validation']['map'] > best['score']:
                best = {'score': row['validation']['map'], 'epoch': row['epoch']}
        assert best == report['selected']
        provenance = {'state': state, 'readout': 'final_LN_CLS384', 'protocol_sha256': digest(run / 'protocol.json'),
                      'cache_manifest_sha256': digest(folder / 'cache_manifest.json'),
                      'encoder_state_sha256': cache['model_state_sha256_after'], 'encoder_parameters_in_optimizer': False}
        assert report['provenance'] == provenance
        head_path, prediction_path = output / 'best_global.pt', output / 'best_global_validation_predictions.pt'
        assert digest(head_path) == report['best_head_sha256'] and digest(prediction_path) == report['best_predictions_sha256']
        saved = torch.load(head_path, map_location='cpu', weights_only=True)
        predictions = torch.load(prediction_path, map_location='cpu', weights_only=True)
        expected = log[best['epoch'] - 1]['validation']
        assert saved['epoch'] == best['epoch'] and saved['metrics'] == expected and saved['provenance'] == provenance
        assert predictions['row_ids'] == validation_ids and predictions['predictions'].shape == (1505, 10)
        metrics = global_metrics(predictions['predictions'].numpy(), labels)
        checks[state] = compare_metrics(metrics, expected)
        assert all(row['pass'] for row in checks[state].values())
        for split in ('train', 'validation'):
            path = folder / f'cls_{split}.npy'
            assert digest(path) == cache['cache_files_sha256'][path.name]
            array = np.load(path, mmap_mode='r')
            assert array.shape == (len(manifest[split]), 384) and array.dtype == np.float32
            assert cache['row_ids'][split] == [row['row_id'] for row in manifest[split]]
            bindings[str(path)] = digest(path)
        assert cache['new_sample_forwards'] == 7556 and cache['new_batch_forwards'] == 474
        assert len(cache['patch_interface_checks']) == 6 and all(row['max_abs_error'] <= 1e-5 for row in cache['patch_interface_checks'])
        forwards += cache['new_sample_forwards']
        heads[state] = report['initial_head_sha256']
        orders[state] = report['actual_order_sha256']
        assert orders[state] == report['original_private_order_sha256']
        old = protocol['states'][state]['mean']
        states[state] = {'mean': {'best': old['best']['global'], 'selected': old['selected']['global'],
                                 'epoch100_log': old['epoch100_log_only']['global'], 'source': str(REAL / 'results.json')},
                         'cls': {'best': metrics, 'selected': best, 'epoch100_log': report['epoch100_metrics_log_only']},
                         'old_dense_reference_only': old['best']['dense'], 'encoder_state_sha256': cache['model_state_sha256_after']}
        for path in (folder / 'cache_manifest.json', output / 'report.json', output / 'metrics.jsonl', head_path, prediction_path):
            bindings[str(path)] = digest(path)
    assert forwards == 22668 and len(set(heads.values())) == len(set(orders.values())) == 1
    assert heads['t0'] == GLOBAL_INITIAL
    deltas = {readout: {f'{a}-T0': {source: states[a][readout][source]['map'] - states['t0'][readout][source]['map']
                                    for source in ('best', 'epoch100_log')} for a in ('sim', 'real')} for readout in ('mean', 'cls')}
    for name in ('protocol.json', 'prepare_receipt.json', 'cls_global_readout.py', 'test_cls_global_readout.py',
                 'cpu_tests.log', 'cpu_red.log', 'preflight_receipt.json', 'source_review_receipt.json'):
        bindings[str(run / name)] = digest(run / name)
    assert digest(run / 'cls_global_readout.py') == protocol['inputs_sha256'][str(Path(__file__))]
    assert digest(run / 'test_cls_global_readout.py') == protocol['inputs_sha256'][str(ROOT / 'tests/test_cls_global_readout.py')]
    resources = {'wall_seconds': time.monotonic() - started, 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'new_disk_bytes_before_score_record': disk_bytes(run)}
    assert resources['wall_seconds'] < 180 and resources['max_rss_bytes'] < 8 * 1024**3
    results = {'status': 'COMPLETE_FIXED_FINAL_CLS_COVERAGE', 'states': states, 'deltas_within_each_readout': deltas,
               'initial_global_heads_sha256': heads, 'actual_private_order_sha256': orders, 'new_sample_forwards': forwards,
               'resources': resources, 'limits': protocol['limits'],
               'epoch100_evidence': 'CLS epoch100 from original logs; independently scored if also savedbest. Mean reference has its original best/log distinction; no extra weights.',
               'old_dense_newly_evaluated': False, 'new_SSL_or_Test': False, 'old_NO_GO_unchanged': True, 'next_experiment_started': False}
    (run / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    verified = {'status': 'PASS_THREE_CLS_NUMPY_BEST_AND_CLOSED_BINDINGS', 'metric_checks': checks,
                'closed_inputs_sha256': bindings, 'results_sha256': digest(run / 'results.json'),
                'patch_interface_checks': {state: json.loads((run / state / 'cache_manifest.json').read_text())['patch_interface_checks'] for state in STATES}}
    (run / 'verification.json').write_text(json.dumps(verified, indent=2) + '\n')
    print(json.dumps({'status': results['status'], 'deltas': deltas, 'resources': resources, 'verification': verified['status']}), flush=True)


def pipeline(run):
    assert not (run / 'compute_ledger.jsonl').exists(), 'Preserve existing run/budget; do not restart pipeline into same folder.'
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    groups = {'prepare': 180, 'features': 1200, 'heads_t0': 300, 'heads_sim': 300, 'heads_real': 300, 'score_closing': 180}
    prior = json.loads((run / 'precompute_receipts.json').read_text())['records']
    (run / 'compute_ledger.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in prior))
    steps = [('prepare', 'prepare', [])]
    steps += [('extract', 'features', ['--state', state]) for state in STATES]
    steps += [('heads', f'heads_{state}', ['--state', state]) for state in STATES]
    steps += [('score', 'score_closing', [])]
    for number, (phase, group, extra) in enumerate(steps):
        records = [json.loads(line) for line in (run / 'compute_ledger.jsonl').read_text().splitlines()]
        used = sum(row['seconds'] for row in records)
        remaining = min(2700 - used, groups[group] - sum(row['seconds'] for row in records if row['group'] == group))
        assert remaining > 0
        command = [sys.executable, str(Path(__file__)), phase, '--run', str(run), *extra]
        log_path = run / f'{number:02d}_{phase}_{group}.log'
        started = time.monotonic()
        peak_disk, peak_rss = disk_bytes(run), 0
        with log_path.open('w') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'phase': phase, 'group': group, 'PID': process.pid, 'command': command,
                              'hard_remaining_seconds': remaining, 'log': str(log_path)}), flush=True)
            violation = None
            try:
                while process.poll() is None:
                    if time.monotonic() - started >= remaining:
                        violation = 'CUMULATIVE_TIME_BUDGET'
                    else:
                        try:
                            status = Path(f'/proc/{process.pid}/status').read_text()
                        except FileNotFoundError:
                            continue  # Child can exit between poll and /proc read.
                        rss = next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmHWM:')), 0)
                        peak_rss = max(peak_rss, rss)
                        if rss > 8 * 1024**3:
                            violation = 'RSS_BUDGET'
                        live_bytes = disk_bytes(run)
                        peak_disk = max(peak_disk, live_bytes)
                        if live_bytes > 128 * 1024**2:
                            violation = 'NEW_STORAGE_PEAK_BUDGET'
                    if violation is not None:
                        break
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    violation = violation or 'MONITOR_INTERRUPTED'
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                record = {'phase': phase, 'group': group, 'PID': process.pid, 'command': command,
                          'seconds': time.monotonic() - started, 'returncode': process.returncode,
                          'budget_violation': violation, 'log_sha256': digest(log_path),
                          'monitored_peak_disk_bytes': peak_disk, 'monitored_peak_rss_bytes': peak_rss,
                          'new_disk_bytes_after': disk_bytes(run)}
                with (run / 'compute_ledger.jsonl').open('a') as ledger:
                    ledger.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        assert process.returncode == 0 and violation is None, f'Stopped at {phase}: see {log_path}'
    records = [json.loads(line) for line in (run / 'compute_ledger.jsonl').read_text().splitlines()]
    total = sum(row['seconds'] for row in records)
    assert total < 2700 and disk_bytes(run) < 128 * 1024**2
    completion = {'status': 'COMPLETE_WITH_ALL_CUMULATIVE_BUDGETS', 'compute_seconds_including_failed_CPU': total,
                  'stage_seconds': {group: sum(row['seconds'] for row in records if row['group'] == group) for group in groups},
                  'new_disk_bytes': disk_bytes(run), 'ledger_sha256': digest(run / 'compute_ledger.jsonl'),
                  'results_sha256': digest(run / 'results.json'), 'verification_sha256': digest(run / 'verification.json'),
                  'next_stage_started': False}
    (run / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pipeline', 'prepare', 'extract', 'heads', 'score'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--state', choices=STATES)
    args = parser.parse_args()
    if args.phase == 'pipeline':
        pipeline(args.run)
    elif args.phase == 'prepare':
        prepare(args.run)
    elif args.phase == 'extract':
        extract(args.run, args.state)
    elif args.phase == 'heads':
        train_global(args.run, args.state)
    else:
        score(args.run)
