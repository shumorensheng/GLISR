"""Fixed NKSID development readouts of T0/REAL1000, mean primary and CLS auxiliary."""

import argparse
import hashlib
import json
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
from runs.cls_global_readout import read_cls_and_patches
from runs.recompute_uatd_area_results import digest
from runs.ssl_global_keep import BASE, disk_bytes
from runs.ssl_real_domain_mix import INITIAL
from sonar_encoder.training.model import build_encoder
from sonar_encoder.training.readout import (confusion_matrix, grayscale, load_targets,
                                          metrics_from_confusion, state_digest, write_json)

RUN = BASE / 'nksid_real_global_readout_v1'
DATA = ROOT / 'runs/data/readout_v1/nksid'
REAL = BASE / 'ssl_real_only_v1'
REAL_STATE = '4744af7045878201960f67be2fd475b43680f3ad43394502efbb086821473849'
STATES = ('t0', 'real')
INTERFACES = ('mean', 'cls')
CELLS = ('t0_mean', 'real_mean', 't0_cls', 'real_cls')
LIMITS = {'prepare': 180, 'extract': 600, **{cell: 180 for cell in CELLS}, 'close': 120}


def read_vectors(encoder, rows, normalization):
    samples = []
    for row in rows:
        values = torch.from_numpy(grayscale(row['image'])).float().unsqueeze(0) / 255
        samples.append({'intensity': values, 'valid': torch.ones(values.shape[-2:], dtype=torch.bool)})
    cls, patches = read_cls_and_patches(encoder, samples, normalization)
    return {'cls': cls, 'mean': patches.mean(dim=1)}


def resources():
    gpu = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
    result = {'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'max_gpu_reserved_bytes': gpu, 'disk_bytes_observed': disk_bytes(RUN)}
    return result


def prepare():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    normal = json.loads((DATA / 'normalization.json').read_text())
    assert manifest['task'] == 'nksid'
    assert [len(manifest[split]) for split in ('train', 'validation')] == [1680, 417]
    assert manifest['class_names'] == ['big_propeller', 'cylinder', 'fishing_net', 'floats',
                                      'iron_pipeline', 'small_propeller', 'soft_pipeline', 'tire']
    train_ids = [row['row_id'] for row in manifest['train']]
    val_ids = [row['row_id'] for row in manifest['validation']]
    assert train_ids == sorted(set(train_ids)) and val_ids == sorted(set(val_ids))
    assert not set(train_ids) & set(val_ids)
    assert normal['source_manifest_sha256'] == digest(DATA / 'manifest.json')
    assert digest(DATA / 'manifest.json') == '8be7dd3ac6d3620bb31ee76ed638b0b9bf5ad951034c700938692252f20005bf'
    assert digest(DATA / 'normalization.json') == '11e70ed8b24368d758aae5cacbfca4de4fbfe47ef1c2a16526362cb4da3ec4d9'
    assert (normal['mean'], normal['std']) == (.18591995626843225, .13096970400869368)
    assert all(0 <= row['classID'] < 8 for split in ('train', 'validation') for row in manifest[split])
    parent = json.loads((REAL / 'protocol.json').read_text())
    verified = json.loads((REAL / 'verification.json').read_text())
    completed = json.loads((REAL / 'completion.json').read_text())
    prior = json.loads((REAL / 'results.json').read_text())
    report = json.loads((REAL / 'real/report1000.json').read_text())
    assert digest(REAL / 'results.json') == verified['results_sha256'] == completed['results_sha256']
    assert digest(REAL / 'verification.json') == completed['verification_sha256']
    assert verified['status'] == 'PASS_NUMPY_SAVED_BEST_AND_CLOSED_INPUT_BINDINGS'
    assert report['step'] == report['attempts'] == 1000
    assert report['initial_states_sha256']['student'] == INITIAL
    assert prior['states']['real']['encoder_state_sha256'] == REAL_STATE
    checkpoint = REAL / 'real/checkpoints/last.pt'
    assert digest(checkpoint) == report['last_sha256'] == verified['closed_inputs_sha256'][str(checkpoint)]
    config = parent['config']
    sources = [Path(__file__), ROOT / 'tests/test_nksid_real_global_readout.py',
               DATA / 'manifest.json', DATA / 'normalization.json', checkpoint,
               *[REAL / name for name in ('protocol.json', 'results.json', 'verification.json', 'completion.json',
                                         'final_closure.json', 'real/protocol.json', 'real/report1000.json', 'real/cache_manifest.json')],
               ROOT / 'runs/cls_global_readout.py', ROOT / 'runs/recompute_uatd_area_results.py',
               ROOT / 'runs/ssl_global_keep.py', ROOT / 'runs/ssl_real_domain_mix.py',
               *[ROOT / 'sonar_encoder/training' / name for name in
                 ('model.py', 'frontends.py', 'readout.py', 'train.py', 'data.py', 'artifacts.py', 'uatd_readout.py', 'uatd_head.py')],
               Path(config['weights_path']), *sorted(Path(config['code_dir']).rglob('*.py')),
               RUN / 'preflight_receipt.json', RUN / 'cpu_tests.log', RUN / 'source_review_receipt.json']
    bindings = {str(path): digest(path) for path in sources}
    for path, value in bindings.items():
        if path in verified['closed_inputs_sha256']:
            assert value == verified['closed_inputs_sha256'][path], path
    images = {row['image']: digest(Path(row['image'])) for split in ('train', 'validation') for row in manifest[split]}
    for split in ('train', 'validation'):
        for row in manifest[split]:
            assert images[row['image']] == row['sha256'], row['image']
    write_json(RUN / 'image_identity.json', {'scope': 'Only 1680 train and 417 validation prepared images; no outer manifest/image/label reads.',
               'image_sha256': images, 'row_ids': {'train': train_ids, 'validation': val_ids}})
    bindings[str(RUN / 'image_identity.json')] = digest(RUN / 'image_identity.json')
    torch.manual_seed(0)
    initial_head = state_digest(nn.Linear(384, 8).state_dict())
    rng = torch.Generator().manual_seed(0)
    order_hash = hashlib.sha256()
    for _ in range(100):
        order_hash.update(torch.randperm(1680, generator=rng).numpy().tobytes())
    cache_bytes = 2 * 2 * 2097 * 384 * 4 + 8 * 128
    reserve = 16 * 1024**2
    assert cache_bytes + reserve < 64 * 1024**2
    protocol = {
        'question': 'Does REAL1000 versus T0 classification decline also occur in the fixed NKSID target-crop recognition definition?',
        'scope': 'Exploratory development only; two frozen encoders, mean primary and CLS fixed auxiliary, complete four cells.',
        'config': {'code_dir': config['code_dir'], 'weights_path': config['weights_path'], 'seed': 0, 'variant': 'R010'},
        'models': {'t0': {'state_sha256': INITIAL, 'checkpoint': None},
                   'real': {'state_sha256': REAL_STATE, 'checkpoint': str(checkpoint), 'checkpoint_sha256': report['last_sha256'],
                            'stream_step': 1000, 'parent_protocol_sha256': digest(REAL / 'protocol.json')}},
        'inputs_sha256': bindings, 'manifest': str(DATA / 'manifest.json'), 'normalization': str(DATA / 'normalization.json'),
        'preprocessing': 'PIL L uint8/255 -> once log1p -> fixed1680-only mean/std -> bilinearAA224 align_cornersFalse; gain1, unmasked, no augmentation/extra normalization/physical coordinates.',
        'features': 'One backbone.forward_features per batch, finalLN x_norm_clstoken384 and GPU FP32 x_norm_patchtokens.mean(dim=1); no patch cache/concat/intermediate layers.',
        'feature_batch': 16, 'new_sample_forwards': 4194, 'new_batch_forwards': 264,
        'heads': {'structure': 'CPU-seed0 Linear384->8 biasTrue, then GPU', 'seed': 0, 'batch_size': 64,
                  'epochs': 100, 'updates': 2700, 'loss': 'unweighted mean cross_entropy', 'optimizer': 'AdamW',
                  'lr': .001, 'weight_decay': .0001, 'betas': [.9, .999], 'eps': 1e-8,
                  'evaluation': 'All417 validation every epoch, fixed8 classes; earliest strict-max macroF1',
                  'initial_sha256': initial_head, 'private_order_sha256': order_hash.hexdigest()},
        'metrics': {'primary': 'mean macroF1', 'auxiliary': 'CLS macroF1; accuracy/perclassF1/P/R/support/confusion8x8',
                    'comparisons': 'T0/REAL absolute and REAL-T0 separately for each interface, no selecting an interface winner.',
                    'best': 'main saved-best independently NumPy scored from original prepared classID',
                    'epoch100': 'Fixed-budget auxiliary from true epoch100 log; no additional weights unless best already100.',
                    'confusion_tolerance': 0, 'float64_metric_tolerance': 1e-12},
        'runtime': {'cpu_threads': 4, 'gpu_index': 0, 'precision': 'FP32, no autocast, TF32off',
                    'deterministic_algorithms': True, 'cudnn_benchmark': False,
                    'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']},
        'budgets': {'phase_seconds': LIMITS, 'total_compute_seconds': 1620, 'rss_bytes': 8 * 1024**3,
                    'gpu_reserved_bytes': 8 * 1024**3, 'new_disk_peak_bytes': 64 * 1024**2},
        'static_disk': {'eight_vector_caches_bytes': cache_bytes, 'best_predictions_records_reserve_bytes': reserve,
                        'conservative_total_bytes': cache_bytes + reserve},
        'limits': ['No session/physical-object identity; UATD acquisition overlap unknown; image-split development, not acquisition-independent generalization.',
                   'Validation fishing_net has3 samples; fixed class set/denominators retained. Author val.jsonl is outer role, not this validation.',
                   'Old R01010k NKSID final result is historical context only; no new outer-evaluation image/label reads or norm fitting.',
                   'REAL remains a working candidate with UATD tradeoff; no evidence from this task changes old NO_GO or frozen results.'],
        'stop': 'After four heads and four independent saved-best scores close; no SSL/other datasets/seeds/HPO/ensemble.'}
    write_json(RUN / 'protocol.json', protocol)
    (RUN / 'code_snapshot').mkdir()
    for path in (Path(__file__), ROOT / 'tests/test_nksid_real_global_readout.py'):
        shutil.copyfile(path, RUN / 'code_snapshot' / path.name)
    for state in STATES:
        (RUN / state).mkdir()
    print(json.dumps({'status': 'FROZEN_BEFORE_FORMAL_ENCODER_FORWARD', 'protocol_sha256': digest(RUN / 'protocol.json'),
                      'input_bindings': len(bindings), 'image_bindings': len(images), 'static_disk': protocol['static_disk']}), flush=True)


@torch.no_grad()
def extract():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    normalization = json.loads(Path(protocol['normalization']).read_text())
    config = protocol['config']
    sample_forwards, batch_forwards = 0, 0
    for state in STATES:
        torch.manual_seed(0)
        encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
        assert state_digest(encoder.state_dict()) == INITIAL
        model = protocol['models'][state]
        if state == 'real':
            checkpoint = Path(model['checkpoint'])
            assert digest(checkpoint) == model['checkpoint_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['stream']['step'] == 1000 and saved['initial_sha256'] == INITIAL
            assert saved['protocol_sha256'] == model['parent_protocol_sha256']
            encoder.load_state_dict(saved['student'])
            del saved
        before = state_digest(encoder.state_dict())
        assert before == model['state_sha256']
        encoder.to('cuda:0').eval().requires_grad_(False)
        hashes, identities = {}, {}
        for split in ('train', 'validation'):
            rows = manifest[split]
            paths = {name: RUN / state / f'{name}_{split}.npy' for name in INTERFACES}
            vectors = {name: np.lib.format.open_memmap(paths[name], mode='w+', dtype=np.float32, shape=(len(rows), 384)) for name in INTERFACES}
            for start in range(0, len(rows), 16):
                batch = rows[start:start + 16]
                output = read_vectors(encoder, batch, normalization)
                for name in INTERFACES:
                    assert output[name].shape == (len(batch), 384) and output[name].dtype == torch.float32
                    assert torch.isfinite(output[name]).all()
                    vectors[name][start:start + len(batch)] = output[name].cpu().numpy()
                sample_forwards += len(batch)
                batch_forwards += 1
                measured = resources()
                assert measured['max_rss_bytes'] <= 8 * 1024**3 and measured['max_gpu_reserved_bytes'] <= 8 * 1024**3
                assert measured['disk_bytes_observed'] <= 64 * 1024**2
                if start % 400 == 0:
                    print(json.dumps({'state': state, 'split': split, 'rows_done': start + len(batch), 'total_forwarded': sample_forwards}), flush=True)
            for name in INTERFACES:
                vectors[name].flush()
                hashes[paths[name].name] = digest(paths[name])
            del vectors, output
            identities[split] = [row['row_id'] for row in rows]
        after = state_digest(encoder.state_dict())
        assert before == after
        write_json(RUN / state / 'cache_manifest.json', {'protocol_sha256': digest(RUN / 'protocol.json'),
                   'state_sha256_before': before, 'state_sha256_after': after, 'cache_sha256': hashes,
                   'row_ids': identities, 'sample_forwards': 2097, 'batch_forwards': 132,
                   'mean_reduction': 'GPU FP32 before CPU copy', 'resources': resources()})
        del encoder
        torch.cuda.empty_cache()
    assert sample_forwards == 4194 and batch_forwards == 264


@torch.no_grad()
def evaluate(head, features, targets):
    logits = torch.cat([head(batch.to('cuda:0')).cpu() for batch in features.split(64)])
    predictions = logits.argmax(dim=1)
    metrics = metrics_from_confusion(confusion_matrix(targets, predictions, 8), 'nksid')
    return metrics, logits, predictions


def train_head(cell):
    protocol = json.loads((RUN / 'protocol.json').read_text())
    state, interface = cell.split('_')
    manifest = json.loads(Path(protocol['manifest']).read_text())
    cache = json.loads((RUN / state / 'cache_manifest.json').read_text())
    features = {}
    for split in ('train', 'validation'):
        path = RUN / state / f'{interface}_{split}.npy'
        assert digest(path) == cache['cache_sha256'][path.name]
        assert cache['row_ids'][split] == [row['row_id'] for row in manifest[split]]
        features[split] = torch.from_numpy(np.load(path))
    targets = {split: load_targets(manifest[split], 'nksid') for split in ('train', 'validation')}
    torch.manual_seed(0)
    head = nn.Linear(384, 8)
    initial = state_digest(head.state_dict())
    assert initial == protocol['heads']['initial_sha256']
    head.to('cuda:0')
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
    rng = torch.Generator().manual_seed(0)
    order_hash = hashlib.sha256()
    folder = RUN / cell
    folder.mkdir()
    best_score, best_epoch, updates = -1., 0, 0
    with (folder / 'metrics.jsonl').open('w') as log:
        for epoch in range(1, 101):
            head.train()
            order = torch.randperm(1680, generator=rng)
            order_hash.update(order.numpy().tobytes())
            loss_sum = 0.
            for indices in order.split(64):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(features['train'][indices].to('cuda:0')), targets['train'][indices].to('cuda:0'))
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
                updates += 1
            head.eval()
            metrics, logits, predictions = evaluate(head, features['validation'], targets['validation'])
            entry = {'epoch': epoch, 'updates': updates, 'train_loss': loss_sum / 1680, 'validation': metrics}
            log.write(json.dumps(entry, allow_nan=False) + '\n')
            log.flush()
            if metrics['macro_f1'] > best_score:
                best_score, best_epoch = metrics['macro_f1'], epoch
                torch.save({'head': {name: value.detach().cpu() for name, value in head.state_dict().items()},
                            'epoch': epoch, 'validation': metrics, 'protocol_sha256': digest(RUN / 'protocol.json')}, folder / 'best.pt')
                torch.save({'logits': logits, 'predictions': predictions, 'row_ids': cache['row_ids']['validation'],
                            'epoch': epoch, 'protocol_sha256': digest(RUN / 'protocol.json')}, folder / 'best_validation_predictions.pt')
            measured = resources()
            assert measured['max_rss_bytes'] <= 8 * 1024**3 and measured['max_gpu_reserved_bytes'] <= 8 * 1024**3
            assert measured['disk_bytes_observed'] <= 64 * 1024**2
            if epoch % 10 == 0:
                print(json.dumps({'cell': cell, 'epoch': epoch, 'updates': updates, 'macro_f1': metrics['macro_f1'], 'best_epoch': best_epoch}), flush=True)
    assert updates == 2700 and order_hash.hexdigest() == protocol['heads']['private_order_sha256']
    write_json(folder / 'report.json', {'cell': cell, 'epochs': 100, 'updates': updates, 'selected_epoch': best_epoch,
               'best_macro_f1': best_score, 'epoch100': entry['validation'], 'initial_head_sha256': initial,
               'actual_private_order_sha256': order_hash.hexdigest(), 'best_head_sha256': digest(folder / 'best.pt'),
               'best_predictions_sha256': digest(folder / 'best_validation_predictions.pt'),
               'encoder_updated': False, 'outer_evaluation_read': False, 'resources': resources()})


def close():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    target = np.array([row['classID'] for row in manifest['validation']], dtype=np.int64)
    rows = [row['row_id'] for row in manifest['validation']]
    checks, results, outputs = [], {}, {}
    for cell in CELLS:
        folder = RUN / cell
        report = json.loads((folder / 'report.json').read_text())
        selected = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        predicted = torch.load(folder / 'best_validation_predictions.pt', map_location='cpu', weights_only=True)
        assert selected['epoch'] == predicted['epoch'] == report['selected_epoch']
        assert predicted['row_ids'] == rows and selected['protocol_sha256'] == predicted['protocol_sha256'] == digest(RUN / 'protocol.json')
        assert report['initial_head_sha256'] == protocol['heads']['initial_sha256']
        assert report['actual_private_order_sha256'] == protocol['heads']['private_order_sha256']
        logits = predicted['logits'].numpy()
        prediction = logits.argmax(axis=1)
        assert logits.shape == (417, 8) and logits.dtype == np.float32 and np.isfinite(logits).all()
        assert np.array_equal(prediction, predicted['predictions'].numpy())
        matrix = np.bincount(target * 8 + prediction, minlength=64).reshape(8, 8)
        assert np.array_equal(matrix, np.array(selected['validation']['confusion_matrix']))
        tp = matrix.diagonal().astype(np.float64)
        actual, guessed = matrix.sum(1), matrix.sum(0)
        independent = {'confusion_matrix': matrix.tolist(), 'support': actual.tolist(),
                       'macro_f1': float((2 * tp / np.maximum(actual + guessed, 1)).mean()),
                       'accuracy': float(tp.sum() / actual.sum()),
                       'per_class_f1': (2 * tp / np.maximum(actual + guessed, 1)).tolist(),
                       'per_class_precision': (tp / np.maximum(guessed, 1)).tolist(),
                       'per_class_recall': (tp / np.maximum(actual, 1)).tolist()}
        errors = {key: float(np.max(np.abs(np.asarray(independent[key]) - np.asarray(selected['validation'][key])))) for key in independent}
        assert errors['confusion_matrix'] == errors['support'] == 0 and max(errors.values()) <= 1e-12
        log = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert len(log) == 100 and log[-1]['updates'] == 2700
        earliest = max(log, key=lambda entry: entry['validation']['macro_f1'])
        assert earliest['epoch'] == selected['epoch'] and earliest['validation'] == selected['validation']
        assert log[-1]['validation'] == report['epoch100']
        results[cell] = {'best_epoch': selected['epoch'], 'best': independent, 'epoch100': report['epoch100'],
                         'epoch100_source': 'saved-best + true log' if selected['epoch'] == 100 else 'true epoch100 log only; no epoch100 weight',
                         'initial_head_sha256': report['initial_head_sha256'], 'order_sha256': report['actual_private_order_sha256']}
        checks.append({'cell': cell, 'status': 'PASS_INDEPENDENT_NUMPY_COUNTS_AND_FLOAT64_METRICS', 'max_errors': errors})
        for path in folder.iterdir():
            outputs[str(path)] = digest(path)
    for state in STATES:
        cache = json.loads((RUN / state / 'cache_manifest.json').read_text())
        assert cache['state_sha256_before'] == cache['state_sha256_after'] == protocol['models'][state]['state_sha256']
        assert cache['sample_forwards'] == 2097 and cache['batch_forwards'] == 132
        for name, value in cache['cache_sha256'].items():
            assert digest(RUN / state / name) == value
        for path in (RUN / state).iterdir():
            outputs[str(path)] = digest(path)
    for path, value in protocol['inputs_sha256'].items():
        assert digest(Path(path)) == value, path
    image_identity = json.loads((RUN / 'image_identity.json').read_text())
    for path, value in image_identity['image_sha256'].items():
        assert digest(Path(path)) == value, path
    differences = {interface: {when: results[f'real_{interface}'][when]['macro_f1'] - results[f't0_{interface}'][when]['macro_f1']
                              for when in ('best', 'epoch100')} for interface in INTERFACES}
    write_json(RUN / 'results.json', {'scope': protocol['scope'], 'protocol_sha256': digest(RUN / 'protocol.json'),
               'class_names': manifest['class_names'], 'cells': results, 'REAL_minus_T0_macro_f1': differences,
               'limits': protocol['limits'], 'resources_at_score_close': resources()})
    write_json(RUN / 'verification.json', {'status': 'PASS_FOUR_NUMPY_SAVED_BEST_AND_CLOSED_BINDINGS',
               'results_sha256': digest(RUN / 'results.json'), 'checks': checks,
               'closed_inputs_sha256': protocol['inputs_sha256'], 'image_identity_sha256': digest(RUN / 'image_identity.json'),
               'images_checked': len(image_identity['image_sha256']), 'closed_outputs_sha256': outputs})
    print(json.dumps({'status': 'FOUR_SAVED_BEST_CLOSED', 'differences': differences, 'resources': resources()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'extract', *CELLS, 'close'))
    args = parser.parse_args()
    started = PROCESS_STARTED
    ledger_path = RUN / 'compute_ledger.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    prior = sum(entry['seconds'] for entry in ledger)
    same_phase = sum(entry['seconds'] for entry in ledger if entry['phase'] == args.phase)
    available = min(LIMITS[args.phase] - same_phase, 1620 - prior)
    assert available > 0
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen compute budget exhausted')))
    signal.setitimer(signal.ITIMER_REAL, available - (time.monotonic() - started))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(json.dumps({'phase': args.phase, 'pid': os.getpid(), 'prior_compute_seconds': prior, 'seconds_available': available}), flush=True)
    status = 'FAILED'
    try:
        if args.phase == 'prepare':
            prepare()
        elif args.phase == 'extract':
            extract()
        elif args.phase in CELLS:
            train_head(args.phase)
        else:
            close()
        status = 'PASS'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - started
        measured = resources()
        receipt = {'phase': args.phase, 'status': status, 'pid': os.getpid(), 'seconds': elapsed,
                   'total_compute_seconds': prior + elapsed, 'resources': measured}
        with ledger_path.open('a') as handle:
            handle.write(json.dumps(receipt, allow_nan=False) + '\n')
        write_json(RUN / f'{args.phase}_receipt.json', receipt)
        assert same_phase + elapsed <= LIMITS[args.phase] and prior + elapsed <= 1620
        assert measured['max_rss_bytes'] <= 8 * 1024**3 and measured['max_gpu_reserved_bytes'] <= 8 * 1024**3
        assert measured['disk_bytes_observed'] <= 64 * 1024**2
    if args.phase == 'close':
        write_json(RUN / 'completion.json', {'status': 'COMPLETE_FIXED_FOUR_CELLS',
                   'results_sha256': digest(RUN / 'results.json'), 'verification_sha256': digest(RUN / 'verification.json'),
                   'protocol_sha256': digest(RUN / 'protocol.json'), 'compute_ledger_sha256': digest(ledger_path),
                   'total_compute_seconds': prior + elapsed, 'resources': measured})


if __name__ == '__main__':
    main()
