"""One bounded frozen Marine pixel-transfer readout, with a source-supervised C reference."""

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
from runs.readout_preserving_adapter import make_adapter, token_delta
from sonar_encoder.training.model import build_encoder, prepare_images
from sonar_encoder.training.readout import (confusion_matrix, grayscale, head_logits, load_targets,
                                            metrics_from_confusion, state_digest, write_json)
from sonar_encoder.training.train import file_digest

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'marine_frozen_transfer_v1'
DATA = ROOT / 'runs/data/readout_v1/marine'
REAL = BASE / 'ssl_real_only_v1'
KEEP = BASE / 'ssl_real_global_keep_amp_resume_v1'
CAPACITY = BASE / 'readout_preserving_adapter_v1'
C_BEST = CAPACITY / 'heads_c/best_dense.pt'
C_FILE_SHA = '656deee5eaf56aa8c44082ab5e640d43e92ed0d7b69090ab22a6cda09eb675e3'
C_STATE_SHA = 'ce76f697944c70254742c10fb42e65022a156c0b6f19843641004da75f78c537'
BODY_SHA = {'t0': 'b9fc4072daeecd3250c87c4f4b6994248f5a84526d3e62effa5bb050dbbf973e',
            'real': '4744af7045878201960f67be2fd475b43680f3ad43394502efbb086821473849',
            'keep': '7d78671e534fbade6e3c900c877ebb005364098895ab20fb314bcf1a8436ee14'}
LIMITS = {'prepare': 180, 'extract': 600, 't0': 900, 'real': 900, 'keep': 900, 'c': 900, 'close': 240}


def projected_c_field(field, adapter):
    return field + token_delta(adapter, field, True)


def runtime():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Keep the original native CUDA bilinear backward instead of the slower deterministic decomposition.
    torch.use_deterministic_algorithms(False)


def resources():
    record = {'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
              'disk_bytes': sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())}
    assert record['max_rss_bytes'] <= 8 * 1024**3, record
    assert record['max_gpu_reserved_bytes'] <= 8 * 1024**3, record
    assert record['disk_bytes'] <= 2 * 1024**3, record
    return record


def prepare():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    norm = json.loads((DATA / 'normalization.json').read_text())
    assert file_digest(DATA / 'manifest.json') == norm['source_manifest_sha256'] == '7946445ef9d0c70e1651de7fb0e65731f0d0576f6ccbc83572a6fdee73ab6725'
    assert norm['mean'] == .1453277000149234 and norm['std'] == .15094323542837218
    ids = {split: [row['row_id'] for row in manifest[split]] for split in ('train', 'validation')}
    assert [len(ids[s]) for s in ids] == [1000, 251]
    assert all(row['shape'] == [480, 320] for s in ids for row in manifest[s])
    inputs = [Path(__file__), ROOT / 'tests/test_marine_frozen_transfer.py', DATA / 'manifest.json', DATA / 'normalization.json',
              ROOT / 'sonar_encoder/training/model.py', ROOT / 'sonar_encoder/training/readout.py',
              ROOT / 'sonar_encoder/training/frontends.py', ROOT / 'sonar_encoder/training/train.py',
              ROOT / 'runs/readout_preserving_adapter.py', C_BEST,
              RUN / 'cpu_red.log', RUN / 'cpu_green.log', RUN / 'source_review_receipt.json']
    for path, expected in manifest['source_manifest_hashes'].items():
        assert file_digest(Path(path)) == expected
        inputs.append(Path(path))
    origins = {'t0': {'state_sha256': BODY_SHA['t0'], 'checkpoint': None, 'conversion': 'Official RGB patch projection SUM to one channel, original R010 gain1.'}}
    for state, parent, arm, expected_file in (
            ('real', REAL, 'real', '0765ded329d6dbbf9e89204ffc1c3cd61ac81d1024b99f3de125734f41e2257c'),
            ('keep', KEEP, 'real_keep', 'b7d9d8d9d13c7c03c814bbffa951eb4942593aa57646ef4a9fcf214a6f6a0061')):
        report = json.loads((parent / arm / 'report1000.json').read_text())
        verified = json.loads((parent / 'verification.json').read_text())
        closed = json.loads((parent / 'final_closure.json').read_text())
        assert file_digest(parent / 'results.json') == verified['results_sha256'] == closed['results_sha256']
        assert file_digest(parent / 'verification.json') == closed['verification_sha256']
        assert report['step'] == 1000 and report['last_sha256'] == expected_file
        checkpoint = parent / arm / 'checkpoints/last.pt'
        assert file_digest(checkpoint) == expected_file
        bound = verified['closed_inputs_sha256'] if state == 'real' else verified['closed_outputs_sha256']
        assert bound[str(checkpoint)] == expected_file
        origins[state] = {'state_sha256': BODY_SHA[state], 'checkpoint': str(checkpoint),
                          'checkpoint_sha256': expected_file, 'parent_protocol_sha256': file_digest(parent / 'protocol.json'),
                          'success': report['step'], 'attempts': report['attempts']}
        inputs += [checkpoint, parent / 'protocol.json', parent / arm / 'report1000.json',
                   parent / 'results.json', parent / 'verification.json', parent / 'completion.json', parent / 'final_closure.json']
    cverified = json.loads((CAPACITY / 'verification.json').read_text())
    assert file_digest(C_BEST) == cverified['closed_output_sha256'][str(C_BEST)] == C_FILE_SHA
    saved_c = torch.load(C_BEST, map_location='cpu', weights_only=True)
    adapter = make_adapter()
    adapter.load_state_dict(saved_c['adapter'], strict=True)
    assert saved_c['epoch'] == 25 and state_digest(adapter.state_dict()) == C_STATE_SHA
    inputs += [CAPACITY / 'protocol.json', CAPACITY / 'results.json', CAPACITY / 'verification.json',
               CAPACITY / 'final_closure.json', CAPACITY / 'heads_c/report.json']
    config = json.loads((REAL / 'protocol.json').read_text())['config']
    assert config['code_dir'] == json.loads((KEEP / 'protocol.json').read_text())['config']['code_dir']
    assert config['weights_path'] == json.loads((KEEP / 'protocol.json').read_text())['config']['weights_path']
    inputs += [Path(config['weights_path']), *sorted(Path(config['code_dir']).rglob('*.py'))]
    torch.manual_seed(0)
    initial_head = state_digest(nn.Conv2d(384, 12, 1, bias=True).state_dict())
    generator, order = torch.Generator().manual_seed(0), hashlib.sha256()
    for _ in range(100):
        order.update(torch.randperm(1000, generator=generator).numpy().tobytes())
    payload = 3 * 1251 * 384 * 256 * 4
    predictions = 4 * 251 * 480 * 320
    temporary = 251 * 480 * 320
    static_disk = payload + 6 * 128 + predictions + temporary + 32 * 1024**2
    assert payload == 1475739648 and static_disk < 2 * 1024**3
    for path in (Path(__file__), ROOT / 'tests/test_marine_frozen_transfer.py'):
        shutil.copyfile(path, RUN / path.name)
    protocol = {'scope': 'One frozen pixel-transfer development readout; ordinary baselines, not a new method or confirmatory/scene-independent test.',
                'manifest': str(DATA / 'manifest.json'), 'normalization': str(DATA / 'normalization.json'),
                'row_ids': ids, 'class_names': manifest['class_names'], 'origins': origins,
                'official_config': {'code_dir': config['code_dir'], 'weights_path': config['weights_path']},
                'C': {'best_file': str(C_BEST), 'best_file_sha256': C_FILE_SHA, 'selected_source_epoch': 25,
                      'adapter_sha256': C_STATE_SHA, 'field': 'T0 post-final-LN CHW field + token_delta(hC,field,True), FP32.',
                      'source_supervision': '6051 UATD frame/box labels; frozen supervised capacity reference, not equal unlabeled data/budget.',
                      'trained_on_Marine': False, 'UATD_dense_head_used': False, 'global_results_fabricated': False},
                'input': 'PIL L uint8/255 -> one log1p -> fixed1000 train-only mean/std -> complete bilinearAA224 align_cornersFalse/gain1; original PE/finalLN.',
                'native_GT': {'shape': [480, 320], 'classes': 12, 'background': 0, 'debris': list(range(1, 11)), 'wall': 11,
                              'resize': False, 'transpose': False, 'ignore_pixels': False},
                'extract': {'bodies': ['t0', 'real', 'keep'], 'batch_size': 16, 'sample_forwards': 3753,
                            'batch_forwards': 237, 'caches': 6, 'dtype': 'float32', 'shape': '[N,384,16,16]',
                            'C_cache': False, 'GT_cache': False, 'new_encoder_checkpoint': False},
                'heads': {'states': ['t0', 'real', 'keep', 'c'], 'structure': 'CPU seed0 Conv2d(384,12,1,biasTrue), no preceding Linear.',
                          'initial_sha256': initial_head, 'CPU_private_shuffle_seed': 0, 'private_order_sha256': order.hexdigest(),
                          'epochs': 100, 'updates': 6300, 'batch_size': 16, 'drop_last': False,
                          'optimizer': 'AdamW lr.001 wd.0001 betas(.9,.999) eps1e-8; constant; only head parameters.',
                          'loss': 'Unweighted mean CE over all native GT pixels, no label smoothing; logits bilinear native480x320 align_cornersFalse.',
                          'validation': 'All251 everyepoch; 12x12 integer confusion -> TP/max(TP+FP+FN,1). Absent class zero in fixed12 mean.',
                          'select': 'Earliest strict maximum mIoU12; debris10/perclass/support auxiliary; epoch100 actual log only unless selected best.',
                          'best_predictions': 'Native argmax uint8 [251,480,320]; original GT only referenced/in memory.'},
                'numerics': 'CUDA0 FP32/no AMP/TF32off/deterministicFalse/CUBLAS_WORKSPACE_CONFIG=:4096:8, CPU4, same allstates. Original native bilinear backward; no bitwise repeatability claim.',
                'static': {'feature_payload_bytes': payload, 'four_best_prediction_bytes': predictions,
                           'one_prediction_atomic_temporary_bytes': temporary, 'new_disk_upper_bound_bytes': static_disk,
                           'peak_CPU_basis': 'Onebody checkpoint/init and batch extraction; heads one492MB featurepair +192MB uint8 native GT and <1GiB temporary allowance.',
                           'peak_GPU_basis': 'One492MB featurepair+192MB uint8 GT, adapter/head, native16x12x480x320 logits118MB and <2GiB temporary allowance; <8GiB.'},
                'budgets_seconds': LIMITS, 'total_seconds': 4620, 'disk_limit_bytes': 2 * 1024**3,
                'RSS_GPU_reserved_limits_bytes': 8 * 1024**3,
                'inputs_sha256': {str(path): file_digest(path) for path in inputs},
                'stop': 'After four best independent NumPy scores and source/resource closure. No seed/HPO/scale/SSL/globalhead/outer617 or next phase.'}
    write_json(RUN / 'protocol.json', protocol)
    print(json.dumps({'status': 'FROZEN_BEFORE_FIRST_OFFICIAL_FORWARD', 'protocol_sha256': file_digest(RUN / 'protocol.json'),
                      'static_disk_upper_bound_bytes': static_disk, 'resources': resources()}), flush=True)


@torch.no_grad()
def extract():
    runtime()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest, norm = json.loads(Path(protocol['manifest']).read_text()), json.loads(Path(protocol['normalization']).read_text())
    config = protocol['official_config']
    for state in ('t0', 'real', 'keep'):
        torch.manual_seed(0)
        encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
        assert state_digest(encoder.state_dict()) == BODY_SHA['t0']
        origin = protocol['origins'][state]
        if state != 't0':
            path = Path(origin['checkpoint'])
            assert file_digest(path) == origin['checkpoint_sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['stream']['step'] == 1000 and saved['protocol_sha256'] == origin['parent_protocol_sha256']
            encoder.load_state_dict(saved['student'], strict=True)
            del saved
        before = state_digest(encoder.state_dict())
        assert before == origin['state_sha256']
        encoder.to('cuda:0').eval().requires_grad_(False)
        folder = RUN / state
        folder.mkdir()
        hashes = {}
        for split in ('train', 'validation'):
            rows = manifest[split]
            path = folder / f'{split}.npy'
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 384, 16, 16))
            for start in range(0, len(rows), 16):
                samples = []
                for row in rows[start:start + 16]:
                    if state == 't0':
                        assert file_digest(Path(row['image'])) == row['imageSHA256']
                    image = torch.from_numpy(grayscale(row['image'])).float().unsqueeze(0) / 255
                    assert image.shape == (1, 480, 320)
                    samples.append({'intensity': image, 'valid': torch.ones((480, 320), dtype=torch.bool)})
                images = prepare_images(samples, norm, None).to('cuda:0')
                field = encoder(images)['patch_tokens'].transpose(1, 2).reshape(len(samples), 384, 16, 16)
                assert field.dtype == torch.float32 and torch.isfinite(field).all()
                array[start:start + len(samples)] = field.cpu().numpy()
                resources()
                if start % 400 == 0:
                    print(json.dumps({'state': state, 'split': split, 'rows_done': start + len(samples)}), flush=True)
            array.flush()
            del array
            hashes[path.name] = file_digest(path)
        after = state_digest(encoder.state_dict())
        assert before == after
        write_json(folder / 'cache_manifest.json', {'protocol_sha256': file_digest(RUN / 'protocol.json'),
                   'encoder_state_sha256_before': before, 'encoder_state_sha256_after': after, 'source': origin,
                   'row_ids': protocol['row_ids'], 'normalization_sha256': file_digest(Path(protocol['normalization'])),
                   'cache_files_sha256': hashes, 'sample_forwards': 1251, 'batch_forwards': 79, 'resources': resources()})
        del encoder
        torch.cuda.empty_cache()
    write_json(RUN / 'extraction.json', {'sample_forwards': 3753, 'batch_forwards': 237, 'C_extra_forwards': 0,
               'GT_disk_copies': 0, 'new_encoder_checkpoints': 0, 'resources': resources()})


@torch.no_grad()
def evaluate(head, features, targets, adapter):
    confusion = torch.zeros((12, 12), dtype=torch.long, device='cuda:0')
    predictions = np.empty((251, 480, 320), dtype=np.uint8)
    head.eval()
    for start in range(0, 251, 16):
        field = features[start:start + 16]
        if adapter is not None:
            field = projected_c_field(field, adapter)
        gt = targets[start:start + 16]
        pred = head_logits(head, field, gt, 'marine').argmax(dim=1)
        confusion += confusion_matrix(gt, pred, 12)
        predictions[start:start + len(pred)] = pred.to(torch.uint8).cpu().numpy()
    return metrics_from_confusion(confusion.cpu(), 'marine'), predictions


def train_head(state):
    runtime()
    started = time.monotonic()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    parent = RUN / ('t0' if state == 'c' else state)
    cache = json.loads((parent / 'cache_manifest.json').read_text())
    assert cache['row_ids'] == protocol['row_ids'] and cache['protocol_sha256'] == file_digest(RUN / 'protocol.json')
    features = {}
    targets = {}
    for split in ('train', 'validation'):
        path = parent / f'{split}.npy'
        assert file_digest(path) == cache['cache_files_sha256'][path.name]
        features[split] = torch.from_numpy(np.load(path)).to('cuda:0')
        assert features[split].shape == (len(manifest[split]), 384, 16, 16)
        native = load_targets(manifest[split], 'marine')
        assert native.dtype == torch.uint8 and native.shape == (len(manifest[split]), 480, 320) and native.max() <= 11
        targets[split] = native.to('cuda:0')
    adapter = None
    if state == 'c':
        saved = torch.load(C_BEST, map_location='cpu', weights_only=True)
        adapter = make_adapter()
        adapter.load_state_dict(saved['adapter'], strict=True)
        assert state_digest(adapter.state_dict()) == C_STATE_SHA
        adapter.to('cuda:0').eval().requires_grad_(False)
    torch.manual_seed(0)
    head = nn.Conv2d(384, 12, 1, bias=True)
    initial = state_digest(head.state_dict())
    assert initial == protocol['heads']['initial_sha256']
    head.to('cuda:0')
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8)
    generator, order_hash = torch.Generator().manual_seed(0), hashlib.sha256()
    folder = RUN / ('head_' + state)
    folder.mkdir()
    best_score, best_epoch, updates = -1., 0, 0
    with (folder / 'metrics.jsonl').open('w') as log:
        for epoch in range(1, 101):
            head.train()
            order = torch.randperm(1000, generator=generator)
            order_hash.update(order.numpy().tobytes())
            total_loss = 0.
            for indices in order.split(16):
                field = features['train'][indices]
                if state == 'c':
                    with torch.no_grad():
                        field = projected_c_field(field, adapter)
                gt = targets['train'][indices]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head_logits(head, field, gt, 'marine'), gt.long())
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                updates += 1
                resources()
            metrics, predictions = evaluate(head, features['validation'], targets['validation'], adapter)
            entry = {'epoch': epoch, 'updates': updates, 'train_loss': total_loss / 1000, 'validation': metrics}
            log.write(json.dumps(entry, allow_nan=False) + '\n')
            log.flush()
            if metrics['miou12'] > best_score:
                best_score, best_epoch = metrics['miou12'], epoch
                saved = {'head': head.state_dict(), 'epoch': epoch, 'metrics': metrics, 'state': state,
                         'protocol_sha256': file_digest(RUN / 'protocol.json'), 'row_ids': protocol['row_ids']['validation']}
                temporary = folder / 'best.tmp'
                torch.save(saved, temporary)
                temporary.replace(folder / 'best.pt')
                temporary = folder / 'best_validation_predictions.tmp'
                with temporary.open('wb') as output:
                    np.save(output, predictions)
                temporary.replace(folder / 'best_validation_predictions.npy')
            resources()
            if epoch % 10 == 0:
                print(json.dumps({'state': state, 'epoch': epoch, 'miou12': metrics['miou12'], 'best_epoch': best_epoch}), flush=True)
    assert updates == 6300 and order_hash.hexdigest() == protocol['heads']['private_order_sha256']
    if state == 'c':
        assert state_digest(adapter.state_dict()) == C_STATE_SHA
    write_json(folder / 'report.json', {'state': state, 'epochs': 100, 'updates': updates, 'best_epoch': best_epoch,
               'best_miou12': best_score, 'epoch100': metrics, 'epoch100_saved_predictions': best_epoch == 100,
               'initial_head_sha256': initial, 'actual_private_order_sha256': order_hash.hexdigest(),
               'optimizer_only_head': True, 'C_adapter_before_after_sha256': C_STATE_SHA if state == 'c' else None,
               'encoder_loaded_in_head_phase': False, 'cache_manifest_sha256': file_digest(parent / 'cache_manifest.json'),
               'best_head_sha256': file_digest(folder / 'best.pt'),
               'best_predictions_sha256': file_digest(folder / 'best_validation_predictions.npy'),
               'wall_seconds': time.monotonic() - started, 'resources': resources()})


def close():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    gt = load_targets(manifest['validation'], 'marine').numpy()
    for split in ('train', 'validation'):
        for row in manifest[split]:
            assert file_digest(Path(row['mask'])) == row['maskSHA256']
    states, checks, outputs = {}, {}, {}
    for state in ('t0', 'real', 'keep', 'c'):
        folder = RUN / ('head_' + state)
        report = json.loads((folder / 'report.json').read_text())
        saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        prediction = np.load(folder / 'best_validation_predictions.npy', mmap_mode='r')
        assert prediction.dtype == np.uint8 and prediction.shape == (251, 480, 320) and prediction.max() <= 11
        assert saved['row_ids'] == protocol['row_ids']['validation'] and saved['protocol_sha256'] == file_digest(RUN / 'protocol.json')
        matrix = np.zeros((12, 12), dtype=np.int64)
        for index in range(251):
            matrix += np.bincount(gt[index].astype(np.int64).ravel() * 12 + prediction[index].ravel(), minlength=144).reshape(12, 12)
        tp, support, guessed = matrix.diagonal().astype(np.float64), matrix.sum(1), matrix.sum(0)
        iou = tp / np.maximum(support + guessed - tp, 1)
        independent = {'confusion_matrix': matrix.tolist(), 'support': support.tolist(), 'per_class_iou': iou.tolist(),
                       'miou12': float(iou.mean()), 'debris_miou10': float(iou[1:11].mean())}
        assert np.array_equal(matrix, np.array(saved['metrics']['confusion_matrix']))
        assert np.array_equal(support, np.array(saved['metrics']['support']))
        error = max(float(np.max(np.abs(np.array(independent[k]) - np.array(saved['metrics'][k])))) for k in ('miou12', 'debris_miou10', 'per_class_iou'))
        assert error <= 1e-12
        logs = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert len(logs) == 100 and all(row['epoch'] == i and row['updates'] == i * 63 for i, row in enumerate(logs, 1))
        selected = max(logs, key=lambda row: row['validation']['miou12'])
        assert selected['epoch'] == saved['epoch'] == report['best_epoch'] and selected['validation'] == saved['metrics']
        assert logs[-1]['validation'] == report['epoch100']
        assert report['initial_head_sha256'] == protocol['heads']['initial_sha256']
        assert report['actual_private_order_sha256'] == protocol['heads']['private_order_sha256']
        assert report['best_head_sha256'] == file_digest(folder / 'best.pt')
        assert report['best_predictions_sha256'] == file_digest(folder / 'best_validation_predictions.npy')
        states[state] = {'best': independent, 'best_epoch': saved['epoch'], 'epoch100': report['epoch100'],
                         'epoch100_independent_predictions': saved['epoch'] == 100,
                         'supervised_source_capacity_reference': state == 'c'}
        checks[state] = {'status': 'PASS_INDEPENDENT_NATIVE_NUMPY_GT_CONFUSION_SUPPORT_IOU',
                         'confusion_and_support_exact': True, 'max_float64_IoU_abs_error': error}
        for path in folder.iterdir():
            outputs[str(path)] = file_digest(path)
    for state in ('t0', 'real', 'keep'):
        cache = json.loads((RUN / state / 'cache_manifest.json').read_text())
        assert cache['encoder_state_sha256_before'] == cache['encoder_state_sha256_after'] == BODY_SHA[state]
        assert cache['row_ids'] == protocol['row_ids'] and cache['normalization_sha256'] == file_digest(Path(protocol['normalization']))
        for name, expected in cache['cache_files_sha256'].items():
            assert file_digest(RUN / state / name) == expected
            outputs[str(RUN / state / name)] = expected
        outputs[str(RUN / state / 'cache_manifest.json')] = file_digest(RUN / state / 'cache_manifest.json')
    for path, expected in protocol['inputs_sha256'].items():
        assert file_digest(Path(path)) == expected, path
    differences = {a + '_minus_' + b: {when: {key: states[a][when][key] - states[b][when][key]
                       for key in ('miou12', 'debris_miou10')} for when in ('best', 'epoch100')}
                   for a, b in (('real', 't0'), ('keep', 't0'), ('keep', 'real'), ('c', 't0'))}
    write_json(RUN / 'results.json', {'status': 'COMPLETE_FIXED_FOUR_MARINE_PIXEL_TRANSFER_CELLS',
               'states': states, 'differences': differences, 'primary': 'mIoU12 including background/wall; no interface/model selection.',
               'C_supervised_reference_separate': True, 'new_encoder_sample_forwards': 3753, 'new_encoder_batch_forwards': 237,
               'new_heads': 4, 'GT_copied_to_disk': False, 'new_outer617_evaluated': False,
               'epoch100_evidence': 'Actual100 logs; independent predictions only when selected best100.',
               'scope': protocol['scope'], 'old_NO_GO_unchanged': True, 'next_experiment_started': False, 'resources': resources()})
    write_json(RUN / 'verification.json', {'status': 'PASS_FOUR_NATIVE_SAVED_BEST_NUMPY_SCORES_WITH_BINDINGS',
               'checks': checks, 'protocol_sha256': file_digest(RUN / 'protocol.json'),
               'results_sha256': file_digest(RUN / 'results.json'), 'closed_inputs_sha256': protocol['inputs_sha256'],
               'all_1251_train_validation_mask_file_SHA_match_manifest': True,
               'closed_outputs_sha256': outputs})
    print(json.dumps({'status': 'FOUR_CELLS_CLOSED', 'differences': differences, 'resources': resources()}), flush=True)


def pipeline():
    for number, phase in enumerate(('prepare', 'extract', 't0', 'real', 'keep', 'c', 'close')):
        records = [json.loads(x) for x in (RUN / 'compute_ledger.jsonl').read_text().splitlines()]
        remaining = min(4620 - sum(x['seconds'] for x in records), LIMITS[phase] - sum(x['seconds'] for x in records if x['group'] == phase))
        assert remaining > 0
        path = RUN / f'{number:02d}_{phase}.log'
        began, peak_rss, peak_disk, violation = time.monotonic(), 0, 0, None
        with path.open('w') as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), phase], cwd=ROOT,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'phase': phase, 'PID': process.pid, 'remaining_seconds': remaining, 'log': str(path)}), flush=True)
            try:
                while process.poll() is None:
                    if time.monotonic() - began >= remaining:
                        violation = 'CUMULATIVE_TIME_BUDGET'
                    else:
                        try:
                            status = Path(f'/proc/{process.pid}/status').read_text()
                        except FileNotFoundError:
                            continue
                        values = [int(x.split()[1]) * 1024 for x in status.splitlines() if x.startswith('VmHWM:')]
                        if values:
                            peak_rss = max(peak_rss, values[0])
                        disk = 0
                        for file in RUN.rglob('*'):
                            try:
                                if file.is_file():
                                    disk += file.stat().st_size
                            except FileNotFoundError:
                                continue
                        peak_disk = max(peak_disk, disk)
                        if peak_rss > 8 * 1024**3:
                            violation = 'RSS_BUDGET'
                        if disk > 2 * 1024**3:
                            violation = 'DISK_BUDGET'
                    if violation:
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
                record = {'phase': phase, 'group': phase, 'PID': process.pid, 'seconds': time.monotonic() - began,
                          'returncode': process.returncode, 'budget_violation': violation, 'log_sha256': file_digest(path),
                          'monitored_peak_rss_bytes': peak_rss, 'monitored_disk_peak_bytes': peak_disk}
                with (RUN / 'compute_ledger.jsonl').open('a') as ledger:
                    ledger.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            assert process.returncode == 0 and violation is None, f'Stopped at {phase}; preserve {path}'
    records = [json.loads(x) for x in (RUN / 'compute_ledger.jsonl').read_text().splitlines()]
    write_json(RUN / 'completion.json', {'status': 'COMPLETE_BOUNDED_FOUR_MARINE_PIXEL_TRANSFER_CELLS',
               'total_compute_seconds': sum(x['seconds'] for x in records),
               'stage_seconds': {g: sum(x['seconds'] for x in records if x['group'] == g) for g in LIMITS},
               'ledger_sha256': file_digest(RUN / 'compute_ledger.jsonl'), 'results_sha256': file_digest(RUN / 'results.json'),
               'verification_sha256': file_digest(RUN / 'verification.json'), 'resources': resources(), 'next_phase_started': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pipeline', 'prepare', 'extract', 't0', 'real', 'keep', 'c', 'close'))
    args = parser.parse_args()
    if args.phase == 'pipeline':
        pipeline()
    elif args.phase == 'prepare':
        prepare()
    elif args.phase == 'extract':
        extract()
    elif args.phase == 'close':
        close()
    else:
        train_head(args.phase)
