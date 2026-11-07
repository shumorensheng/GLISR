"""One fixed CPU-FP32 T0/REAL encoder weight midpoint, with existing task readouts."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs import nksid_real_global_readout as nksid
from runs.cls_global_readout import GLOBAL_INITIAL, read_cls_and_patches, train_global
from runs.recompute_uatd_area_results import compare_metrics, dense_metrics, digest, global_metrics
from runs.ssl_global_keep import BASE, UATD, disk_bytes
from runs.ssl_real_domain_mix import INITIAL
from runs.train_uatd_area_heads import main as train_dual_heads
from sonar_encoder.training.model import build_encoder
from sonar_encoder.training.readout import grayscale, state_digest, write_json
from sonar_encoder.training.uatd_readout import native_sample

RUN = BASE / 'encoder_midpoint_readout_v1'
REAL = BASE / 'ssl_real_only_v1'
OLD_CLS = BASE / 'cls_global_readout_v1'
OLD_NKSID = BASE / 'nksid_real_global_readout_v1'
CHECKPOINT = REAL / 'real/checkpoints/last.pt'
REAL_STATE = '4744af7045878201960f67be2fd475b43680f3ad43394502efbb086821473849'
LIMITS = {'prepare': 180, 'extract': 900, 'uatd-dual': 900, 'uatd-cls': 300,
          'nksid-mean': 180, 'nksid-cls': 180, 'close': 180}


def midpoint_state(t0, real):
    assert t0.keys() == real.keys()
    result = {}
    for key, value in t0.items():
        other = real[key]
        assert value.device.type == other.device.type == 'cpu'
        assert value.dtype == other.dtype == torch.float32, f'Non-FP32 state requires advisor definition: {key}'
        assert value.shape == other.shape
        result[key] = (0.5 * value) + (0.5 * other)
    return result


def read_outputs(encoder, samples, normalization):
    cls, patches = read_cls_and_patches(encoder, samples, normalization)
    return {'cls': cls, 'patches': patches.transpose(1, 2).reshape(-1, 384, 16, 16),
            'mean': patches.mean(dim=1)}


def build_midpoint(config):
    torch.manual_seed(0)
    encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
    initial = encoder.state_dict()
    assert state_digest(initial) == INITIAL
    report = json.loads((REAL / 'real/report1000.json').read_text())
    assert digest(CHECKPOINT) == report['last_sha256'] == '0765ded329d6dbbf9e89204ffc1c3cd61ac81d1024b99f3de125734f41e2257c'
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert saved['stream']['step'] == 1000 and saved['initial_sha256'] == INITIAL
    assert saved['protocol_sha256'] == digest(REAL / 'protocol.json')
    assert state_digest(saved['student']) == REAL_STATE
    midpoint = midpoint_state(initial, saved['student'])
    midpoint_hash = state_digest(midpoint)
    count = len(midpoint)
    encoder.load_state_dict(midpoint)
    del initial, midpoint, saved
    assert state_digest(encoder.state_dict()) == midpoint_hash
    return encoder, midpoint_hash, count


def resources():
    return {'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            'disk_bytes_observed': disk_bytes(RUN)}


def check_resources():
    measured = resources()
    assert measured['max_rss_bytes'] <= 8 * 1024**3 and measured['max_gpu_reserved_bytes'] <= 8 * 1024**3
    assert measured['disk_bytes_observed'] <= 4 * 1024**3
    return measured


def prepare():
    real_protocol = json.loads((REAL / 'protocol.json').read_text())
    cls_protocol = json.loads((OLD_CLS / 'protocol.json').read_text())
    nksid_protocol = json.loads((OLD_NKSID / 'protocol.json').read_text())
    inputs = [Path(__file__), ROOT / 'tests/test_encoder_midpoint_readout.py',
              ROOT / 'runs/nksid_real_global_readout.py', ROOT / 'runs/cls_global_readout.py',
              ROOT / 'runs/train_uatd_area_heads.py', ROOT / 'runs/recompute_uatd_area_results.py',
              ROOT / 'runs/ssl_global_keep.py', ROOT / 'runs/ssl_real_domain_mix.py', CHECKPOINT,
              UATD / 'manifest.json', UATD / 'normalization.json', nksid.DATA / 'manifest.json', nksid.DATA / 'normalization.json',
              RUN / 'preflight_receipt.json', RUN / 'cpu_tests.log', RUN / 'source_review_receipt.json',
              *[ROOT / 'sonar_encoder/training' / name for name in
                ('model.py', 'frontends.py', 'readout.py', 'train.py', 'data.py', 'artifacts.py', 'uatd_readout.py', 'uatd_head.py')]]
    for parent in (REAL, OLD_CLS, OLD_NKSID):
        verified = json.loads((parent / 'verification.json').read_text())
        completed = json.loads((parent / 'completion.json').read_text())
        assert digest(parent / 'results.json') == verified['results_sha256'] == completed['results_sha256']
        assert digest(parent / 'verification.json') == completed['verification_sha256']
        inputs.extend(parent / name for name in ('protocol.json', 'results.json', 'verification.json', 'completion.json', 'final_closure.json'))
    inputs.append(REAL / 'real/report1000.json')
    config = real_protocol['config']
    inputs.extend([Path(config['weights_path']), *sorted(Path(config['code_dir']).rglob('*.py'))])
    bindings = {str(path): digest(path) for path in inputs}
    datasets = {}
    for task, folder, counts in (('uatd', UATD, [6051, 1505]), ('nksid', nksid.DATA, [1680, 417])):
        manifest = json.loads((folder / 'manifest.json').read_text())
        normal = json.loads((folder / 'normalization.json').read_text())
        assert [len(manifest[s]) for s in ('train', 'validation')] == counts
        assert normal['source_manifest_sha256'] == digest(folder / 'manifest.json')
        datasets[task] = {'manifest': str(folder / 'manifest.json'), 'normalization': str(folder / 'normalization.json'),
                          'counts': counts, 'class_names': manifest['class_names']}
    assert (json.loads((nksid.DATA / 'normalization.json').read_text())['mean'],
            json.loads((nksid.DATA / 'normalization.json').read_text())['std']) == (.18591995626843225, .13096970400869368)
    encoder, midpoint_hash, tensors = build_midpoint(config)
    del encoder
    cache_bytes = 7556 * 384 * (256 + 1) * 4 + 2 * 2097 * 384 * 4 + 8 * 128
    assert cache_bytes == 2989189120
    # Bound all validation grid positions plus serialized best/prediction records.
    temporary_prediction_bound = 1505 * 10 * 256 * (5 * 4 + 8)
    reserve = 64 * 1024**2
    assert cache_bytes + temporary_prediction_bound + reserve < 4 * 1024**3
    protocol = {'question': 'Does reducing the learned encoder parameter change by half soften classification/detection tradeoffs in two existing development tasks?',
                'scope': 'One ordinary fixed checkpoint-weight midpoint control; not a new method,500stepSSL,feature/prediction ensemble,or fullWiSE-FT replication.',
                'precedents': ['https://arxiv.org/abs/2109.01903', 'https://github.com/mlfoundations/wise-ft'],
                'alpha': .5, 'arithmetic': 'CPU FP32 for every complete single-channel R010 encoder state tensor: (0.5*T0)+(0.5*REAL), two multiplies then add; no extra normalization.',
                'encoder_state_sha256': midpoint_hash, 'encoder_state_tensor_count': tensors,
                'parents': {'t0': {'student_state_sha256': INITIAL, 'RGB_conversion': 'Official foundation -> existing single-channel SUM projection -> interpolate full wrapper state.'},
                            'real': {'student_state_sha256': REAL_STATE, 'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': digest(CHECKPOINT), 'stream_step': 1000}},
                'config': {'seed': 0, 'code_dir': config['code_dir'], 'weights_path': config['weights_path'], 'variant': 'R010'},
                'datasets': datasets, 'inputs_sha256': bindings,
                'features': 'One in-memory MID encoder continuously processes UATD then NKSID in one process; full eval FP32/TF32off AA224/gain1/officialPE and each original task norm. No mask/aug/new norm/coordinates.',
                'reductions': {'uatd': 'Cache finalLN patch[B,384,16,16] +CLS384; original dual trainer spatial mean on patch cache unchanged.',
                               'nksid': 'Same forward finalLN CLS384 and GPU FP32 patch_tokens.mean(dim=1), before CPU copy.'},
                'new_sample_forwards': 9653, 'new_batch_forwards': 606, 'feature_batch': 16,
                'task_settings': {'uatd': real_protocol['heads'], 'nksid': nksid_protocol['heads']},
                'expected_initial_heads': {'uatd_global': GLOBAL_INITIAL, 'uatd_dense': '3d4b303a5aebfc34ea04542a645f3420e174d48525b4242656e0b621c766b7c8',
                                           'nksid': nksid_protocol['heads']['initial_sha256']},
                'expected_orders': {'uatd': cls_protocol['expected_private_order_sha256'], 'nksid': nksid_protocol['heads']['private_order_sha256']},
                'runtime': {'cpu_threads': 4, 'GPU_index': 0, 'FP32': True, 'TF32': False, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']},
                'budgets': {'phase_seconds': LIMITS, 'total_seconds': 2820, 'rss_bytes': 8 * 1024**3, 'gpu_reserved_bytes': 8 * 1024**3, 'new_disk_peak_bytes': 4 * 1024**3},
                'static_disk': {'eight_caches_bytes': cache_bytes, 'grid_prediction_temporary_upper_bound_bytes': temporary_prediction_bound,
                                'best_predictions_metadata_reserve_bytes': reserve, 'conservative_peak_upper_bound_bytes': cache_bytes + temporary_prediction_bound + reserve},
                'endpoints': {'uatd_primary': ['cls', 'dense'], 'uatd_auxiliary': ['mean'], 'nksid_primary': ['mean'], 'nksid_auxiliary': ['cls'],
                              'comparison': 'Full15cells T0/REAL/MID x fivefixedtaskinterfaces, best main/actual100 auxiliary; MID-T0 andMID-REAL within interface only.'},
                'references': {'uatd_cls_mean': str(OLD_CLS / 'results.json'), 'uatd_dense': str(REAL / 'results.json'), 'nksid': str(OLD_NKSID / 'results.json')},
                'limits': ['Existing image-split development/seed0 only; no confirmatory or acquisition-independent generalization.',
                           'NKSID lacks session/object IDs, UATD overlap unknown, fishing_net val3; no outer520/Test or new data.',
                           'Only this alpha.5; negative result does not exclude all merging ratios. No alpha/seed/SSL/loss/HPO or teacher selection.',
                           'All encoder state P/PE/CLS/mask/LN/LayerScale uses samealpha; predictor/teacher/optimizer/taskhead excluded. No new encoder checkpoint.',
                           'Historical NO_GO/C1/C2 and all frozen numerical results remain unchanged.'],
                'stop': 'After9653sampleforward,5newheads,5savedbestNumPyscores and binding closure; no follow-on experiment.'}
    write_json(RUN / 'protocol.json', protocol)
    (RUN / 'code_snapshot').mkdir()
    for path in (Path(__file__), ROOT / 'tests/test_encoder_midpoint_readout.py'):
        shutil.copyfile(path, RUN / 'code_snapshot' / path.name)
    for task in ('uatd', 'nksid'):
        folder = RUN / task
        (folder / 'mid').mkdir(parents=True)
        common = {'parent_protocol_sha256': digest(RUN / 'protocol.json'), 'encoder_state_sha256': midpoint_hash}
        if task == 'uatd':
            child = {**common, 'prepared_manifest': datasets[task]['manifest'], 'heads': real_protocol['heads'],
                     'inputs_sha256': bindings, 'global_initial_sha256': GLOBAL_INITIAL,
                     'expected_private_order_sha256': protocol['expected_orders']['uatd'],
                     'budgets': {'head_seconds_per_arm': 900, 'head_seconds_per_state': 300, 'rss_bytes': 8 * 1024**3, 'gpu_bytes': 8 * 1024**3}}
            write_json(folder / 'protocol.json', child)
            (folder / 'mid/protocol.json').symlink_to('../protocol.json')
        else:
            write_json(folder / 'protocol.json', {**common, 'manifest': datasets[task]['manifest'], 'heads': nksid_protocol['heads']})
    print(json.dumps({'status': 'FROZEN_MID_STATE_AND_SOURCE_BEFORE_FORWARD', 'MID_state_sha256': midpoint_hash,
                      'tensor_count': tensors, 'protocol_sha256': digest(RUN / 'protocol.json'), 'static_disk': protocol['static_disk']}), flush=True)


@torch.no_grad()
def extract():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    encoder, before, _ = build_midpoint(protocol['config'])
    assert before == protocol['encoder_state_sha256']
    encoder.to('cuda:0').eval().requires_grad_(False)
    sample_forwards, batch_forwards = 0, 0
    for task in ('uatd', 'nksid'):
        settings = protocol['datasets'][task]
        manifest = json.loads(Path(settings['manifest']).read_text())
        normal = json.loads(Path(settings['normalization']).read_text())
        folder = RUN / task / 'mid'
        hashes, row_ids = {}, {}
        for split in ('train', 'validation'):
            rows = manifest[split]
            names = ('mid', 'cls') if task == 'uatd' else ('mean', 'cls')
            arrays = {name: np.lib.format.open_memmap(folder / f'{name}_{split}.npy', mode='w+', dtype=np.float32,
                      shape=(len(rows), 384, 16, 16) if name == 'mid' else (len(rows), 384)) for name in names}
            for start in range(0, len(rows), 16):
                batch = rows[start:start + 16]
                samples = []
                for row in batch:
                    expected_image = row['image_sha256'] if task == 'uatd' else row['sha256']
                    assert digest(Path(row['image'])) == expected_image, row['image']
                    if task == 'uatd':
                        samples.append(native_sample(row))
                    else:
                        values = torch.from_numpy(grayscale(row['image'])).float().unsqueeze(0) / 255
                        samples.append({'intensity': values, 'valid': torch.ones(values.shape[-2:], dtype=torch.bool)})
                output = read_outputs(encoder, samples, normal)
                for name in names:
                    tensor = output['patches' if name == 'mid' else name]
                    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
                    arrays[name][start:start + len(batch)] = tensor.cpu().numpy()
                sample_forwards += len(batch)
                batch_forwards += 1
                check_resources()
                if start % 800 == 0:
                    print(json.dumps({'task': task, 'split': split, 'rows_done': start + len(batch), 'sample_forwards': sample_forwards}), flush=True)
            for name in names:
                arrays[name].flush()
                path = folder / f'{name}_{split}.npy'
                hashes[path.name] = digest(path)
            del arrays, output
            row_ids[split] = [row['row_id'] for row in rows]
        after = state_digest(encoder.state_dict())
        assert before == after
        record = {'protocol_sha256': digest(RUN / task / 'protocol.json'), 'parent_protocol_sha256': digest(RUN / 'protocol.json'),
                  'model_state_sha256_before': before, 'model_state_sha256_after': after,
                  'row_ids': row_ids, 'sample_forwards': sum(settings['counts']),
                  'batch_forwards': 474 if task == 'uatd' else 132,
                  'image_bytes_checked': sum(settings['counts']), 'resources': check_resources()}
        record['cache_files_sha256' if task == 'uatd' else 'cache_sha256'] = hashes
        write_json(folder / 'cache_manifest.json', record)
    assert sample_forwards == 9653 and batch_forwards == 606


def dual_heads():
    original = torch.randperm
    order_hash = hashlib.sha256()
    calls = 0
    def record_order(*args, **kwargs):
        nonlocal calls
        order = original(*args, **kwargs)
        order_hash.update(order.numpy().tobytes())
        calls += 1
        return order
    torch.randperm = record_order
    try:
        train_dual_heads(RUN / 'uatd/mid', 'mid', 'cuda:0')
    finally:
        torch.randperm = original
    protocol = json.loads((RUN / 'protocol.json').read_text())
    assert calls == 100 and order_hash.hexdigest() == protocol['expected_orders']['uatd']
    write_json(RUN / 'uatd/dual_order_receipt.json', {'actual_order_sha256': order_hash.hexdigest(), 'calls': calls,
               'method': 'Read and hash original private torch.randperm returned orders without adding/reordering RNG operations; restored original function.'})


def score_nksid(folder, manifest, protocol_sha):
    report = json.loads((folder / 'report.json').read_text())
    selected = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    pred = torch.load(folder / 'best_validation_predictions.pt', map_location='cpu', weights_only=True)
    assert selected['epoch'] == pred['epoch'] == report['selected_epoch']
    assert selected['protocol_sha256'] == pred['protocol_sha256'] == protocol_sha
    assert pred['row_ids'] == [row['row_id'] for row in manifest['validation']]
    logits = pred['logits'].numpy()
    prediction = logits.argmax(1)
    target = np.array([row['classID'] for row in manifest['validation']], dtype=np.int64)
    assert np.array_equal(prediction, pred['predictions'].numpy())
    matrix = np.bincount(target * 8 + prediction, minlength=64).reshape(8, 8)
    tp = matrix.diagonal().astype(np.float64)
    actual, guessed = matrix.sum(1), matrix.sum(0)
    metrics = {'confusion_matrix': matrix.tolist(), 'support': actual.tolist(),
               'macro_f1': float((2 * tp / np.maximum(actual + guessed, 1)).mean()), 'accuracy': float(tp.sum() / actual.sum()),
               'per_class_f1': (2 * tp / np.maximum(actual + guessed, 1)).tolist(),
               'per_class_precision': (tp / np.maximum(guessed, 1)).tolist(), 'per_class_recall': (tp / np.maximum(actual, 1)).tolist()}
    errors = {key: float(np.abs(np.asarray(value) - np.asarray(selected['validation'][key])).max()) for key, value in metrics.items()}
    assert errors['confusion_matrix'] == errors['support'] == 0 and max(errors.values()) <= 1e-12
    log = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    assert len(log) == report['epochs'] == 100 and report['updates'] == log[-1]['updates'] == 2700
    assert max(log, key=lambda entry: entry['validation']['macro_f1'])['epoch'] == selected['epoch']
    assert selected['validation'] == log[selected['epoch'] - 1]['validation'] and report['epoch100'] == log[-1]['validation']
    return {'best': metrics, 'best_epoch': selected['epoch'], 'epoch100': report['epoch100'], 'source': str(folder / 'report.json'),
            'epoch100_independently_scored': selected['epoch'] == 100}, errors


def close():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    datasets = {task: json.loads(Path(value['manifest']).read_text()) for task, value in protocol['datasets'].items()}
    uatd = datasets['uatd']
    labels = [np.array(row['labels'], dtype=np.int64) for row in uatd['validation']]
    boxes = [np.array(row['boxes'], dtype=np.float32) for row in uatd['validation']]
    new, checks, outputs = {'uatd': {}, 'nksid': {}}, {}, {}
    for interface, directory, kind, key in (('mean', 'heads_mid', 'global', 'map'), ('dense', 'heads_mid', 'dense', 'map50'), ('cls', 'heads_cls', 'global', 'map')):
        folder = RUN / 'uatd/mid' / directory
        report = json.loads((folder / 'report.json').read_text())
        log = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert report['epochs'] == len(log) == 100 and report['updates'] == 19000
        for epoch, entry in enumerate(log, 1):
            assert entry['epoch'] == epoch and entry['updates'] == epoch * 190
            assert ('validation' in entry) == (epoch % 5 == 0)
        if interface == 'cls':
            selection = report['selected']
            expected = log[selection['epoch'] - 1]['validation']
            final = report['epoch100_metrics_log_only']
            assert report['initial_head_sha256'] == GLOBAL_INITIAL
            assert report['actual_order_sha256'] == protocol['expected_orders']['uatd']
        else:
            selection = report['selected'][kind]
            expected = log[selection['epoch'] - 1]['validation'][kind]
            final = report['epoch100_metrics'][kind]
            assert report['initial_head_sha256'] == {'global': GLOBAL_INITIAL, 'dense': protocol['expected_initial_heads']['uatd_dense']}
        eligible = [entry for entry in log if 'validation' in entry]
        earliest = max(eligible, key=lambda entry: entry['validation'][key] if interface == 'cls' else entry['validation'][kind][key])
        assert earliest['epoch'] == selection['epoch']
        assert final == (log[-1]['validation'] if interface == 'cls' else log[-1]['validation'][kind])
        head = torch.load(folder / f'best_{kind}.pt', map_location='cpu', weights_only=True)
        pred = torch.load(folder / f'best_{kind}_validation_predictions.pt', map_location='cpu', weights_only=True)
        assert head['epoch'] == selection['epoch'] and head['metrics'] == expected
        assert head['provenance'] == report['provenance']
        assert head['provenance']['encoder_state_sha256'] == protocol['encoder_state_sha256']
        assert head['provenance']['encoder_parameters_in_optimizer'] is False
        assert pred['row_ids'] == [row['row_id'] for row in uatd['validation']]
        metrics = (dense_metrics([{k: value.numpy() for k, value in row.items()} for row in pred['predictions']], boxes, labels)
                   if interface == 'dense' else global_metrics(pred['predictions'].numpy(), labels))
        check = compare_metrics(metrics, expected)
        assert all(value['pass'] for value in check.values())
        checks['uatd_' + interface] = check
        new['uatd'][interface] = {'best': metrics, 'best_epoch': selection['epoch'], 'epoch100': final,
                                 'epoch100_independently_scored': selection['epoch'] == 100, 'source': str(folder / 'report.json')}
        for path in folder.iterdir():
            outputs[str(path)] = digest(path)
    order = json.loads((RUN / 'uatd/dual_order_receipt.json').read_text())
    assert order['calls'] == 100 and order['actual_order_sha256'] == protocol['expected_orders']['uatd']
    for interface in ('mean', 'cls'):
        folder = RUN / 'nksid' / ('mid_' + interface)
        new['nksid'][interface], checks['nksid_' + interface] = score_nksid(folder, datasets['nksid'], digest(RUN / 'nksid/protocol.json'))
        report = json.loads((folder / 'report.json').read_text())
        assert report['initial_head_sha256'] == protocol['expected_initial_heads']['nksid']
        assert report['actual_private_order_sha256'] == protocol['expected_orders']['nksid']
        for path in folder.iterdir():
            outputs[str(path)] = digest(path)
    for task in ('uatd', 'nksid'):
        folder = RUN / task / 'mid'
        cache = json.loads((folder / 'cache_manifest.json').read_text())
        assert cache['parent_protocol_sha256'] == digest(RUN / 'protocol.json')
        assert cache['protocol_sha256'] == digest(RUN / task / 'protocol.json')
        assert cache['model_state_sha256_before'] == cache['model_state_sha256_after'] == protocol['encoder_state_sha256']
        assert cache['sample_forwards'] == sum(protocol['datasets'][task]['counts'])
        assert cache['batch_forwards'] == (474 if task == 'uatd' else 132)
        for split in ('train', 'validation'):
            assert cache['row_ids'][split] == [row['row_id'] for row in datasets[task][split]]
        for name, value in cache['cache_files_sha256' if task == 'uatd' else 'cache_sha256'].items():
            path = folder / name
            assert digest(path) == value
            outputs[str(path)] = value
        outputs[str(folder / 'cache_manifest.json')] = digest(folder / 'cache_manifest.json')
        outputs[str(RUN / task / 'protocol.json')] = digest(RUN / task / 'protocol.json')
    old_cls = json.loads(Path(protocol['references']['uatd_cls_mean']).read_text())
    old_real = json.loads(Path(protocol['references']['uatd_dense']).read_text())
    old_nksid = json.loads(Path(protocol['references']['nksid']).read_text())
    states = {'mid': new}
    for state, expected_state in (('t0', INITIAL), ('real', REAL_STATE)):
        assert old_cls['states'][state]['encoder_state_sha256'] == old_real['states'][state]['encoder_state_sha256'] == expected_state
        states[state] = {'uatd': {}, 'nksid': {}}
        for interface in ('cls', 'mean'):
            old = old_cls['states'][state][interface]
            states[state]['uatd'][interface] = {'best': old['best'], 'best_epoch': old['selected']['epoch'],
                'epoch100': old['epoch100_log'], 'source': protocol['references']['uatd_cls_mean'], 'source_record': old}
        old = old_real['states'][state]
        states[state]['uatd']['dense'] = {'best': old['best']['dense'], 'best_epoch': old['selected']['dense']['epoch'],
            'epoch100': old['epoch100_log_only']['dense'], 'source': protocol['references']['uatd_dense']}
        for interface in ('mean', 'cls'):
            old = old_nksid['cells'][state + '_' + interface]
            states[state]['nksid'][interface] = {'best': old['best'], 'best_epoch': old['best_epoch'], 'epoch100': old['epoch100'],
                'source': protocol['references']['nksid'], 'epoch100_source': old['epoch100_source']}
    differences = {baseline: {task: {interface: {when: new[task][interface][when][key] - states[baseline][task][interface][when][key]
        for when in ('best', 'epoch100')} for interface, key in (('cls', 'map'), ('mean', 'map'), ('dense', 'map50'))} if task == 'uatd' else
        {interface: {when: new[task][interface][when]['macro_f1'] - states[baseline][task][interface][when]['macro_f1'] for when in ('best', 'epoch100')}
         for interface in ('mean', 'cls')} for task in ('uatd', 'nksid')} for baseline in ('t0', 'real')}
    for path, value in protocol['inputs_sha256'].items():
        assert digest(Path(path)) == value, path
    write_json(RUN / 'results.json', {'scope': protocol['scope'], 'encoder_state_sha256': protocol['encoder_state_sha256'],
               'states': states, 'MID_minus_reference_within_interface': differences, 'limits': protocol['limits'],
               'endpoints': protocol['endpoints'], 'new_sample_forwards': 9653, 'new_heads': 5,
               'old_models_or_heads_rerun': False, 'no_outer_or_Test': True})
    write_json(RUN / 'verification.json', {'status': 'PASS_FIVE_SAVED_BEST_NUMPY_SCORES_AND_BINDINGS', 'checks': checks,
               'results_sha256': digest(RUN / 'results.json'), 'closed_inputs_sha256': protocol['inputs_sha256'],
               'closed_outputs_sha256': outputs, 'protocol_sha256': digest(RUN / 'protocol.json')})
    print(json.dumps({'status': 'FIVE_SAVED_BEST_CLOSED', 'differences': differences, 'resources': check_resources()}), flush=True)


def timeout_phase(signum, frame):
    raise TimeoutError('Frozen midpoint readout compute budget exhausted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=tuple(LIMITS))
    args = parser.parse_args()
    ledger_path = RUN / 'compute_ledger.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    prior = sum(row['seconds'] for row in ledger)
    same_phase = sum(row['seconds'] for row in ledger if row['phase'] == args.phase)
    available = min(LIMITS[args.phase] - same_phase, 2820 - prior)
    signal.signal(signal.SIGALRM, timeout_phase)
    signal.setitimer(signal.ITIMER_REAL, available - (time.monotonic() - PROCESS_STARTED))
    if args.phase not in ('uatd-dual', 'uatd-cls'):
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
        elif args.phase == 'uatd-dual':
            dual_heads()
        elif args.phase == 'uatd-cls':
            train_global(RUN / 'uatd', 'mid')
        elif args.phase in ('nksid-mean', 'nksid-cls'):
            nksid.RUN = RUN / 'nksid'
            nksid.train_head('mid_' + args.phase.split('-')[1])
        else:
            close()
        status = 'PASS'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - PROCESS_STARTED
        measured = resources()
        receipt = {'phase': args.phase, 'status': status, 'pid': os.getpid(), 'seconds': elapsed,
                   'total_compute_seconds': prior + elapsed, 'resources': measured}
        with ledger_path.open('a') as handle:
            handle.write(json.dumps(receipt, allow_nan=False) + '\n')
        write_json(RUN / (args.phase + '_receipt.json'), receipt)
        assert same_phase + elapsed <= LIMITS[args.phase] and prior + elapsed <= 2820
        check_resources()
    if args.phase == 'close':
        write_json(RUN / 'completion.json', {'status': 'COMPLETE_SINGLE_FIXED_ENCODER_MIDPOINT_FIFTEEN_CELLS',
                   'protocol_sha256': digest(RUN / 'protocol.json'), 'results_sha256': digest(RUN / 'results.json'),
                   'verification_sha256': digest(RUN / 'verification.json'), 'compute_ledger_sha256': digest(ledger_path),
                   'total_compute_seconds': prior + elapsed, 'resources': measured})


if __name__ == '__main__':
    main()
