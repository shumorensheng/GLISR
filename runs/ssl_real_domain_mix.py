"""Fixed S/MIX training-data intervention under the reviewed R010 contract."""

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs.recompute_uatd_area_results import compare_metrics, dense_metrics, digest, global_metrics
from runs.ssl_global_keep import (BASE, DATA, SSL_MANIFEST, SSL_NORM, UATD, disk_bytes,
                                 initialize_training, next_indices, restore_training, training_state)
from sonar_encoder.training.data import SonarDataset
from sonar_encoder.training.model import build_encoder, make_patch_mask, masked_latent_loss, prepare_native
from sonar_encoder.training.readout import grayscale, state_digest
from sonar_encoder.training.train import manifest_digest, optimizer_update, validate
from sonar_encoder.training.uatd_readout import native_sample

CONTRACT = BASE / 'real_domain_ssl_contract_v1'
PARENT = BASE / 'ssl_global_keep_v1'
INITIAL = 'b9fc4072daeecd3250c87c4f4b6994248f5a84526d3e62effa5bb050dbbf973e'
PREDICTOR = '4991a00886f80b0392ad455da46111f79ea89302a291595a83412be91a058dc7'


def native_mask(mask, height, width):
    rows = torch.arange(height) * 16 // height
    columns = torch.arange(width) * 16 // width
    return mask.reshape(16, 16)[rows[:, None], columns[None, :]]


def prepare_views(samples, normalizations, masks):
    images = []
    for index, (sample, normalization) in enumerate(zip(samples, normalizations)):
        image = prepare_native(sample, normalization, None)
        if masks is not None:
            height, width = image.shape[-2:]
            image.masked_fill_(native_mask(masks[index], height, width).unsqueeze(0), 0)
        images.append(F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear',
                                    align_corners=False, antialias=True)[0])
    return torch.stack(images)


def new_stream(sim_count, real_count, sim_rng, real_rng):
    return {'step': 0, 'attempts': 0, 'success_trace': '0' * 64, 'attempt_trace': '0' * 64,
            'sim': {'epoch': 0, 'sample_cursor': 0, 'epoch_order': torch.randperm(sim_count, generator=sim_rng)},
            'real': {'epoch': 0, 'sample_cursor': 0, 'epoch_order': torch.randperm(real_count, generator=real_rng)}}


def next_microbatch(stream, sim_count, real_count, sim_rng, real_rng, mask_rng, arm):
    candidates = next_indices(stream['sim'], sim_count, 8, sim_rng)
    masks = make_patch_mask(8, mask_rng)
    if arm == 'sim':
        actual = [('sim', index) for index in candidates]
    else:
        reals = next_indices(stream['real'], real_count, 4, real_rng)
        actual = [('sim', index) for index in candidates[:4]] + [('real', index) for index in reals]
    return candidates, actual, masks


def real_sample(row):
    image = grayscale(Path(row['image']))
    return {'sample_id': row['row_id'], 'intensity': torch.from_numpy(image).float().unsqueeze(0) / 255,
            'valid': torch.ones(image.shape, dtype=torch.bool)}


def prepare(run):
    started = time.monotonic()
    contract = json.loads((CONTRACT / 'contract_v1.json').read_text())
    assert digest(CONTRACT / 'contract_v1.json') == '3558017534561c2dcd85d4184b2f913e2e4c690f2145771e1801629b2af8cc6b'
    for path, expected in contract['inputs_sha256'].items():
        assert digest(Path(path)) == expected
    subset = json.loads((CONTRACT / 'real_train_subset.json').read_text())
    assert len(subset['rows']) == 1024
    parent = json.loads((PARENT / 'protocol.json').read_text())
    bindings = dict(parent['inputs_sha256'])
    # Rebind only the new driver/test; every original shared source must stay identical.
    bindings.update({str(path): digest(path) for path in
                    (Path(__file__), ROOT / 'tests/test_ssl_real_domain_mix.py',
                     CONTRACT / 'contract_v1.json', CONTRACT / 'real_train_subset.json',
                     CONTRACT / 'advisor_review_receipt.json', run / 'cpu_tests.log',
                     PARENT / 'protocol.json', PARENT / 'verification.json', PARENT / 'results.json',
                     PARENT / 'epoch100_evidence_clarification.json')})
    assert all(digest(Path(path)) == expected for path, expected in bindings.items())
    count, total, square = 0, np.float64(0), np.float64(0)
    for row in subset['rows']:
        image = grayscale(Path(row['image']))
        assert list(image.shape) == row['shape'][:2]
        values = np.log1p(image.astype(np.float64) / 255)
        count += values.size
        total += values.sum(dtype=np.float64)
        square += np.square(values).sum(dtype=np.float64)
    assert count == subset['normalization_population_native_pixels'] == 1253128704
    real_norm = {'mean': float(total / count), 'std': float(np.sqrt(square / count - (total / count)**2)),
                 'count': count, 'sum_FP64': float(total), 'sumsq_FP64': float(square),
                 'subset_sha256': digest(CONTRACT / 'real_train_subset.json'),
                 'row_ids_sha256': subset['row_ids_sha256'], 'population': 'Only selected1024 complete native L/255 log1p pixels'}
    (run / 'real_ssl_normalization.json').write_text(json.dumps(real_norm, indent=2) + '\n')
    bindings[str(run / 'real_ssl_normalization.json')] = digest(run / 'real_ssl_normalization.json')
    t0 = PARENT / 't0'
    cache = json.loads((t0 / 'cache_manifest.json').read_text())
    heads_report = json.loads((t0 / 'heads_t0/report.json').read_text())
    assert json.loads((t0 / 'protocol.json').read_text())['heads'] == parent['heads']
    assert cache['checkpoint_sha256'] is None
    assert cache['initial_state_sha256'] == cache['model_state_sha256_before'] == cache['model_state_sha256_after'] == INITIAL
    assert cache['normalization_sha256'] == digest(UATD / 'normalization.json')
    assert cache['protocol_sha256'] == digest(t0 / 'protocol.json')
    assert cache['parent_protocol_sha256'] == digest(PARENT / 'protocol.json')
    task = json.loads((UATD / 'manifest.json').read_text())
    for split in ('train', 'validation'):
        path = t0 / f't0_{split}.npy'
        assert digest(path) == cache['cache_files_sha256'][path.name]
        array = np.load(path, mmap_mode='r')
        assert array.shape == (len(task[split]), 384, 16, 16) and array.dtype == np.float32
        assert cache['row_ids'][split] == [row['row_id'] for row in task[split]]
        bindings[str(path)] = cache['cache_files_sha256'][path.name]
    for name in ('protocol.json', 'cache_manifest.json', 'heads_t0/report.json', 'heads_t0/metrics.jsonl',
                 'heads_t0/best_global.pt', 'heads_t0/best_dense.pt',
                 'heads_t0/best_global_validation_predictions.pt', 'heads_t0/best_dense_validation_predictions.pt'):
        path = t0 / name
        bindings[str(path)] = digest(path)
    for task_name in ('global', 'dense'):
        assert bindings[str(t0 / f'heads_t0/best_{task_name}.pt')] == heads_report['best_heads_sha256'][task_name]
        assert bindings[str(t0 / f'heads_t0/best_{task_name}_validation_predictions.pt')] == heads_report['best_predictions_sha256'][task_name]
    protocol = {'scope': contract['question'], 'design_contract_sha256': digest(CONTRACT / 'contract_v1.json'),
                'config': parent['config'], 'heads': parent['heads'], 'inputs_sha256': bindings,
                'ssl_manifest': str(SSL_MANIFEST), 'ssl_normalization': str(SSL_NORM), 'data_root': str(DATA),
                'real_subset': str(CONTRACT / 'real_train_subset.json'),
                'real_ssl_normalization': str(run / 'real_ssl_normalization.json'),
                'prepared_manifest': str(UATD / 'manifest.json'), 'task_normalization': str(UATD / 'normalization.json'),
                'feature_states': ['sim', 'mix'], 'feature_batch': 16, 'T0_origin': str(t0),
                'budgets': {**contract['budgets'], 'head_seconds_per_arm': contract['budgets']['head_seconds_per_pair']},
                'expected_student_sha256': INITIAL, 'expected_predictor_sha256': PREDICTOR,
                'native_mask': contract['native_mask'], 'sampling': contract['sampling'],
                'optimization': contract['optimization'], 'metrics': contract['metrics'], 'limits': contract['limits'],
                'T0_identity': 'Conditionally reusable parentT0; exact official initialization verified during50; downstream uses original native_sample/forward_native(task norm,None), ordinaryAA224/gain1/FP32/officialPE and patchlayout. Original shared sources, parent protocols/cache/head provenance bound.',
                'stop': 'Two step1000 students, two new global+dense pairs, six NumPy best scores; no third extraction/Test/ratio/seed/HPO.'}
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for arm in ('sim', 'mix'):
        folder = run / arm
        folder.mkdir()
        (folder / 'checkpoints').mkdir()
        child = {'scope': protocol['scope'], 'prepared_manifest': protocol['prepared_manifest'],
                 'heads': protocol['heads'], 'budgets': protocol['budgets'], 'feature_state': arm,
                 'inputs_sha256': {str(path): digest(path) for path in
                                  (run / 'protocol.json', UATD / 'manifest.json',
                                   ROOT / 'runs/train_uatd_area_heads.py', ROOT / 'sonar_encoder/training/uatd_head.py')}}
        (folder / 'protocol.json').write_text(json.dumps(child, indent=2) + '\n')
    for path in (Path(__file__), ROOT / 'tests/test_ssl_real_domain_mix.py'):
        shutil.copyfile(path, run / path.name)
    receipt = {'status': 'FROZEN_BEFORE_OFFICIAL_MODEL_COMPUTATION', 'input_bindings': len(bindings),
               'real_ssl_statistics': real_norm, 'CPU_interfaces_passed': 3,
               'wall_seconds': time.monotonic() - started,
               'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    assert receipt['wall_seconds'] < 300 and receipt['max_rss_bytes'] < 8 * 1024**3
    (run / 'prepare_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


def train_prefix(run, arm, stop_after):
    started = time.monotonic()
    protocol = json.loads((run / 'protocol.json').read_text())
    config = protocol['config']
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models = initialize_training(config)
    student, teacher, initial, predictor, optimizer, scaler, sim_rng, mask_rng = models
    initial_states = {'student': state_digest(student.state_dict()), 'teacher': state_digest(teacher.state_dict()),
                      'predictor': state_digest(predictor.state_dict())}
    assert initial_states == {'student': INITIAL, 'teacher': INITIAL, 'predictor': PREDICTOR}
    del initial, models
    real_rng = torch.Generator().manual_seed(3)
    folder = run / arm
    dataset = SonarDataset(SSL_MANIFEST, DATA, 'ssl_train')
    validation = SonarDataset(SSL_MANIFEST, DATA, 'ssl_validation')
    real_rows = json.loads(Path(protocol['real_subset']).read_text())['rows']
    sim_norm = json.loads(SSL_NORM.read_text())
    real_norm = json.loads(Path(protocol['real_ssl_normalization']).read_text())
    if stop_after == 50:
        stream = new_stream(len(dataset), len(real_rows), sim_rng, real_rng)
        previous_seconds = 0.
    else:
        assert json.loads((run / 'gate50.json').read_text())['status'] == 'PASS_PAIRED_50_AND_RESOURCE_PROJECTION'
        previous_seconds = json.loads((folder / 'report50.json').read_text())['cumulative_wall_seconds']
        saved = torch.load(folder / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert saved['initial_sha256'] == INITIAL and saved['protocol_sha256'] == digest(run / 'protocol.json')
        assert saved['stream']['step'] == 50
        restore_training(saved, student, teacher, predictor, optimizer, scaler, sim_rng, mask_rng, config['device'])
        real_rng.set_state(saved['real_rng'])
        stream = saved['stream']
        del saved
    student.train()
    torch.cuda.reset_peak_memory_stats()
    windows, window_seconds, window_updates = [], 0., 0
    consecutive_skips = 0
    with (folder / 'batches.jsonl').open('a', buffering=1) as batches, (folder / 'metrics.jsonl').open('a', buffering=1) as log:
        while stream['step'] < stop_after:
            step_started = time.monotonic()
            step = stream['step']
            factor = (step + 1) / config['warmup_steps']
            tau = 1 - (1 - config['ema_start']) * (1 + math.cos(math.pi * step / config['steps'])) / 2
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            actual_slots, candidate_ids, paired_ids, mask_hash = [], [], [], hashlib.sha256()
            masked_value = 0.
            for micro in range(config['accumulation']):
                candidates, actual, masks = next_microbatch(stream, len(dataset), len(real_rows), sim_rng, real_rng, mask_rng, arm)
                candidate_ids.extend(dataset.frames[index]['sample_id'] for index in candidates)
                samples, norms = [], []
                for slot, (domain, index) in enumerate(actual):
                    sample = dataset[index] if domain == 'sim' else real_sample(real_rows[index])
                    samples.append(sample)
                    norms.append(sim_norm if domain == 'sim' else real_norm)
                    actual_slots.append({'micro': micro, 'slot': slot, 'domain': domain, 'actual_id': sample['sample_id'],
                                         'masked_token_indices': masks[slot].nonzero().flatten().tolist()})
                    if slot < 4:
                        paired_ids.append(sample['sample_id'])
                mask_hash.update(masks.numpy().tobytes())
                with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                    with torch.no_grad():
                        target = teacher(prepare_views(samples, norms, None).to(config['device']))['patch_tokens']
                    prediction = predictor(student(prepare_views(samples, norms, masks).to(config['device']),
                                                   masks.to(config['device']))['patch_tokens'])
                    loss = masked_latent_loss(prediction, target, masks.to(config['device']))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite SSL loss at {arm}/{step}')
                masked_value += float(loss.detach()) / config['accumulation']
                scaler.scale(loss / config['accumulation']).backward()
            updated, norm = optimizer_update(student, teacher, optimizer, scaler, tau, config['clip_grad'])
            stream['attempts'] += 1
            batch = {'attempt': stream['attempts'], 'step_before': step, 'actual_slots': actual_slots,
                     'candidate_sim_ids_not_exposure': candidate_ids, 'paired_sim_ids': paired_ids,
                     'mask_sha256': mask_hash.hexdigest(), 'updated': updated, 'lr_factor': factor, 'tau': tau,
                     'grad_norm': norm, 'masked_loss': masked_value, 'amp_scale': scaler.get_scale()}
            identity = json.dumps([paired_ids, batch['mask_sha256']], separators=(',', ':')).encode()
            stream['attempt_trace'] = hashlib.sha256(bytes.fromhex(stream['attempt_trace']) + identity + bytes([updated])).hexdigest()
            batches.write(json.dumps(batch) + '\n')
            if not updated:
                consecutive_skips += 1
                if consecutive_skips == 3:
                    raise FloatingPointError('Three AMP overflows; preserve attempt evidence and consult Astra.')
                continue
            consecutive_skips = 0
            stream['step'] += 1
            stream['success_trace'] = hashlib.sha256(bytes.fromhex(stream['success_trace']) + identity).hexdigest()
            window_seconds += time.monotonic() - step_started
            window_updates += 1
            if stream['step'] % config['log_every'] == 0:
                row = {'step': stream['step'], 'seconds': window_seconds, 'updates': window_updates,
                       'masked_loss': masked_value, 'grad_norm': norm, 'lr_factor': factor, 'tau': tau}
                log.write(json.dumps(row) + '\n')
                print(json.dumps({'arm': arm, **row}), flush=True)
                windows.append(row)
                window_seconds, window_updates = 0., 0
            if stream['step'] % 500 == 0 and stream['step'] < stop_after:
                health = validate(student, teacher, predictor, validation, sim_norm, config)
                log.write(json.dumps({'kind': 'validation', 'step': stream['step'], 'metrics': health}) + '\n')
                print(json.dumps({'arm': arm, 'step': stream['step'], 'health': health}), flush=True)
            assert previous_seconds + time.monotonic() - started < 1800
    # The engineering50 gate has no task or health-performance selection.
    health = validate(student, teacher, predictor, validation, sim_norm, config) if stop_after == 1000 else None
    checkpoint = training_state(student, teacher, predictor, optimizer, scaler, sim_rng, mask_rng, stream, config['device'])
    checkpoint.update(real_rng=real_rng.get_state(), initial_sha256=INITIAL, resolved_config=config,
                      protocol_sha256=digest(run / 'protocol.json'))
    temporary = folder / 'checkpoints/last.tmp'
    torch.save(checkpoint, temporary)
    atomic_peak_bytes = disk_bytes(run)
    assert atomic_peak_bytes < protocol['budgets']['new_peak_bytes']
    temporary.replace(folder / 'checkpoints/last.pt')
    report = {'status': 'COMPLETE_PREFIX', 'arm': arm, 'step': stream['step'], 'attempts': stream['attempts'],
              'success_trace': stream['success_trace'], 'attempt_trace': stream['attempt_trace'],
              'initial_states_sha256': initial_states, 'last_sha256': digest(folder / 'checkpoints/last.pt'),
              'validation_original_masked': health,
              'last30_seconds_per_update': sum(row['seconds'] for row in windows[-3:]) / sum(row['updates'] for row in windows[-3:]),
              'cumulative_wall_seconds': previous_seconds + time.monotonic() - started,
              'resources': {'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved(), 'atomic_peak_disk_bytes': atomic_peak_bytes}}
    assert report['cumulative_wall_seconds'] < 1800 and report['resources']['max_rss_bytes'] < 8 * 1024**3
    assert report['resources']['max_gpu_reserved_bytes'] < 8 * 1024**3
    (folder / f'report{stop_after}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def check_prefix(run, step):
    reports = {arm: json.loads((run / arm / f'report{step}.json').read_text()) for arm in ('sim', 'mix')}
    protocol = json.loads((run / 'protocol.json').read_text())
    assert reports['sim']['initial_states_sha256'] == reports['mix']['initial_states_sha256']
    paired, exposures = {}, {}
    for arm in ('sim', 'mix'):
        report = reports[arm]
        assert report['step'] == step and digest(run / arm / 'checkpoints/last.pt') == report['last_sha256']
        batches = [json.loads(line) for line in (run / arm / 'batches.jsonl').read_text().splitlines()]
        successful = [batch for batch in batches if batch['updated']]
        assert len(successful) == step and len(batches) == report['attempts']
        paired[arm] = [(batch['attempt'], batch['paired_sim_ids'], batch['mask_sha256']) for batch in successful]
        counts = {kind: {domain: Counter() for domain in ('sim', 'real')} for kind in ('attempted', 'successful')}
        chain = '0' * 64
        for batch in batches:
            reconstructed = torch.zeros(32, 256, dtype=torch.bool)
            assert len(batch['actual_slots']) == 32
            for number, slot in enumerate(batch['actual_slots']):
                assert slot['micro'] == number // 8 and slot['slot'] == number % 8
                assert slot['domain'] == ('real' if arm == 'mix' and number % 8 >= 4 else 'sim')
                indices = slot['masked_token_indices']
                assert len(indices) == len(set(indices)) == 192
                reconstructed[number, indices] = True
                if slot['domain'] == 'sim':
                    assert slot['actual_id'] == batch['candidate_sim_ids_not_exposure'][number]
                counts['attempted'][slot['domain']][slot['actual_id']] += 1
                if batch['updated']:
                    counts['successful'][slot['domain']][slot['actual_id']] += 1
            assert hashlib.sha256(reconstructed.numpy().tobytes()).hexdigest() == batch['mask_sha256']
            assert batch['paired_sim_ids'] == [slot['actual_id'] for slot in batch['actual_slots'] if slot['slot'] < 4]
            if batch['updated']:
                identity = json.dumps([batch['paired_sim_ids'], batch['mask_sha256']], separators=(',', ':')).encode()
                chain = hashlib.sha256(bytes.fromhex(chain) + identity).hexdigest()
        assert chain == report['success_trace']
        for number, batch in enumerate(successful):
            assert batch['step_before'] == number and batch['lr_factor'] == (number + 1) / protocol['config']['warmup_steps']
            assert batch['tau'] == 1 - (1 - protocol['config']['ema_start']) * (1 + math.cos(math.pi * number / protocol['config']['steps'])) / 2
        exposures[arm] = {kind: {domain: {'exposures': sum(counter.values()), 'unique': len(counter),
                                        'repeated_exposures': sum(counter.values()) - len(counter),
                                        'per_actual_id_counts': dict(counter)}
                                for domain, counter in domains.items()} for kind, domains in counts.items()}
        assert sum(value['exposures'] for value in exposures[arm]['successful'].values()) == step * 32
    assert paired['sim'] == paired['mix'], 'AMP successful pairing disrupted: stop dependencies and consult Astra.'
    projections = {arm: reports[arm]['cumulative_wall_seconds'] + 950 * reports[arm]['last30_seconds_per_update'] + 180
                   for arm in ('sim', 'mix')} if step == 50 else {}
    if step == 50:
        assert max(projections.values()) <= 1800
    record = {'status': 'PASS_PAIRED_50_AND_RESOURCE_PROJECTION' if step == 50 else 'PASS_PAIRED_1000_PREFIXES',
              'step': step, 'success_trace_sha256': reports['sim']['success_trace'], 'projected_seconds': projections,
              'exposures': exposures, 'reports_sha256': {arm: digest(run / arm / f'report{step}.json') for arm in reports},
              'no_loss_or_task_performance_gate': True}
    (run / f'gate{step}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'exposures'}), flush=True)


@torch.no_grad()
def extract(run, state):
    started = time.monotonic()
    protocol = json.loads((run / 'protocol.json').read_text())
    assert json.loads((run / 'gate1000.json').read_text())['status'] == 'PASS_PAIRED_1000_PREFIXES'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = protocol['config']
    torch.manual_seed(config['seed'])
    encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
    initial = state_digest(encoder.state_dict())
    reference = json.loads((run / 'sim/report1000.json').read_text())['initial_states_sha256']['student']
    assert initial == reference
    checkpoint_sha = None
    if state != 't0':
        checkpoint_path = run / state / 'checkpoints/last.pt'
        checkpoint_sha = digest(checkpoint_path)
        report = json.loads((run / state / 'report1000.json').read_text())
        assert checkpoint_sha == report['last_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['stream']['step'] == 1000 and checkpoint['initial_sha256'] == initial
        assert checkpoint['protocol_sha256'] == digest(run / 'protocol.json')
        encoder.load_state_dict(checkpoint['student'])
        del checkpoint
    encoder = encoder.to('cuda:0').eval().requires_grad_(False)
    before = state_digest(encoder.state_dict())
    torch.cuda.reset_peak_memory_stats()
    normalization = json.loads(Path(protocol['task_normalization']).read_text())
    prepared = json.loads(Path(protocol['prepared_manifest']).read_text())
    folder = run / state
    cache_hashes, row_ids, forwards = {}, {}, 0
    prior_seconds = sum(json.loads((run / earlier / 'cache_manifest.json').read_text())['resources']['wall_seconds']
                        for earlier in protocol['feature_states'][:protocol['feature_states'].index(state)])
    for split in ('train', 'validation'):
        rows = prepared[split]
        path = folder / f'{state}_{split}.npy'
        features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 384, 16, 16))
        batch_size = protocol['feature_batch']
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            tokens = encoder.forward_native([native_sample(row) for row in batch], normalization, None)['patch_tokens']
            assert tokens.shape == (len(batch), 256, 384) and torch.isfinite(tokens).all()
            features[start:start + len(batch)] = tokens.transpose(1, 2).reshape(-1, 384, 16, 16).cpu().numpy()
            forwards += len(batch)
            if start % (batch_size * 50) == 0:
                print(json.dumps({'state': state, 'split': split, 'rows_done': start + len(batch), 'rows_total': len(rows)}), flush=True)
            assert prior_seconds + time.monotonic() - started < protocol['budgets']['new_feature_seconds_total']
        features.flush()
        del features
        cache_hashes[path.name] = digest(path)
        row_ids[split] = [row['row_id'] for row in rows]
        assert disk_bytes(run) < protocol['budgets']['new_peak_bytes']
    after = state_digest(encoder.state_dict())
    assert before == after and forwards == 7556
    resources = {'wall_seconds': time.monotonic() - started,
                 'cumulative_feature_seconds': prior_seconds + time.monotonic() - started,
                 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved(), 'disk_bytes_after': disk_bytes(run)}
    assert resources['cumulative_feature_seconds'] < 1200 and resources['max_rss_bytes'] < 8 * 1024**3
    assert resources['max_gpu_reserved_bytes'] < 8 * 1024**3
    record = {'status': 'COMPLETE_FULL_IMAGE_FROZEN_FEATURES', 'state': state,
              'protocol_sha256': digest(folder / 'protocol.json'), 'parent_protocol_sha256': digest(run / 'protocol.json'),
              'initial_state_sha256': initial, 'checkpoint_sha256': checkpoint_sha,
              'model_state_sha256_before': before, 'model_state_sha256_after': after,
              'cache_files_sha256': cache_hashes, 'row_ids': row_ids, 'new_network_forwards': forwards,
              'normalization_sha256': digest(Path(protocol['task_normalization'])),
              'precision': 'FP32 TF32off; ordinary R010 normalization/AA224, no extra gain or radial conversion.', 'resources': resources}
    (folder / 'cache_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'state': state, 'status': record['status'], 'model_state_sha256': after, 'resources': resources}), flush=True)



def score(run):
    started = time.monotonic()
    torch.set_num_threads(4)
    protocol = json.loads((run / 'protocol.json').read_text())
    bindings = {}
    for p, expected in protocol['inputs_sha256'].items():
        actual = digest(Path(p))
        assert actual == expected, p
        bindings[p] = actual
    assert json.loads((run / 'gate1000.json').read_text())['status'] == 'PASS_PAIRED_1000_PREFIXES'
    manifest = json.loads(Path(protocol['prepared_manifest']).read_text())
    validation_ids = [row['row_id'] for row in manifest['validation']]
    boxes = [np.asarray(row['boxes'], dtype=np.float32) for row in manifest['validation']]
    labels = [np.asarray(row['labels'], dtype=np.int64) for row in manifest['validation']]
    states, checks, initial_heads = {}, {}, {}
    for state in ('t0', 'sim', 'mix'):
        folder = Path(protocol['T0_origin']) if state == 't0' else run / state
        output = folder / f'heads_{state}'
        child = json.loads((folder / 'protocol.json').read_text())
        cache = json.loads((folder / 'cache_manifest.json').read_text())
        report = json.loads((output / 'report.json').read_text())
        log = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        assert report['status'] == 'COMPLETE_FIXED_DEVELOPMENT_HEADS'
        assert report['arm'] == state and report['epochs'] == len(log) == 100 and report['updates'] == 19000
        assert report['class_names'] == manifest['class_names'] and report['final_test_evaluated'] is False
        assert cache['protocol_sha256'] == digest(folder / 'protocol.json')
        assert cache['parent_protocol_sha256'] == digest(PARENT / 'protocol.json' if state == 't0' else run / 'protocol.json')
        assert cache['model_state_sha256_before'] == cache['model_state_sha256_after']
        assert cache['new_network_forwards'] == 7556
        provenance = {'arm': state, 'protocol_sha256': digest(folder / 'protocol.json'),
                      'cache_manifest_sha256': digest(folder / 'cache_manifest.json'),
                      'encoder_state_sha256': cache['model_state_sha256_after'], 'encoder_parameters_in_optimizer': False}
        assert report['provenance'] == provenance
        for p, expected in child['inputs_sha256'].items():
            assert digest(Path(p)) == expected
        for split in ('train', 'validation'):
            path = folder / f'{state}_{split}.npy'
            assert digest(path) == cache['cache_files_sha256'][path.name]
            array = np.load(path, mmap_mode='r')
            assert array.shape == (len(manifest[split]), 384, 16, 16) and array.dtype == np.float32
            assert cache['row_ids'][split] == [row['row_id'] for row in manifest[split]]
            bindings[str(path)] = cache['cache_files_sha256'][path.name]
        for epoch, row in enumerate(log, 1):
            assert row['epoch'] == epoch and row['updates'] == epoch * 190
            assert ('validation' in row) == (epoch % 5 == 0)
        assert report['epoch100_metrics'] == log[-1]['validation']
        evaluated = [row for row in log if 'validation' in row]
        metrics, checks[state] = {}, {}
        for task, key in (('global', 'map'), ('dense', 'map50')):
            best = {'score': -1., 'epoch': 0}
            for row in evaluated:
                if row['validation'][task][key] > best['score']:
                    best = {'score': row['validation'][task][key], 'epoch': row['epoch']}
            assert report['selected'][task] == best
            path, prediction_path = output / f'best_{task}.pt', output / f'best_{task}_validation_predictions.pt'
            assert digest(path) == report['best_heads_sha256'][task]
            assert digest(prediction_path) == report['best_predictions_sha256'][task]
            saved = torch.load(path, map_location='cpu', weights_only=True)
            predictions = torch.load(prediction_path, map_location='cpu', weights_only=True)
            expected = log[best['epoch'] - 1]['validation'][task]
            assert saved['epoch'] == best['epoch'] and saved['metrics'] == expected and saved['provenance'] == provenance
            assert predictions['row_ids'] == validation_ids
            if task == 'global':
                metrics[task] = global_metrics(predictions['predictions'].numpy(), labels)
            else:
                metrics[task] = dense_metrics([{k: v.numpy() for k, v in p.items()} for p in predictions['predictions']], boxes, labels)
            checks[state][task] = compare_metrics(metrics[task], expected)
            assert all(v['pass'] for v in checks[state][task].values())
            bindings[str(path)], bindings[str(prediction_path)] = digest(path), digest(prediction_path)
        initial_heads[state] = report['initial_head_sha256']
        states[state] = {'selected': report['selected'], 'best': metrics, 'epoch100_log_only': report['epoch100_metrics'],
                         'encoder_state_sha256': cache['model_state_sha256_after']}
        for path in (folder / 'protocol.json', folder / 'cache_manifest.json', output / 'report.json', output / 'metrics.jsonl'):
            bindings[str(path)] = digest(path)
    assert initial_heads['t0'] == initial_heads['sim'] == initial_heads['mix']
    deltas = {}
    for comparison, a, b in (('S-T0', 'sim', 't0'), ('MIX-T0', 'mix', 't0'), ('MIX-S', 'mix', 'sim')):
        deltas[comparison] = {name: {task: states[a][source][task][key] - states[b][source][task][key]
                                    for task, key in (('global', 'map'), ('dense', 'map50'))}
                              for name, source in (('best', 'best'), ('epoch100_log_only', 'epoch100_log_only'))}
    for arm in ('sim', 'mix'):
        report = json.loads((run / arm / 'report1000.json').read_text())
        assert digest(run / arm / 'checkpoints/last.pt') == report['last_sha256']
        for name in ('report50.json', 'report1000.json', 'batches.jsonl', 'metrics.jsonl', 'checkpoints/last.pt'):
            path = run / arm / name
            bindings[str(path)] = digest(path)
    for name in ('protocol.json', 'gate50.json', 'gate1000.json', Path(__file__).name,
                 'test_ssl_real_domain_mix.py', 'cpu_tests.log', 'prepare_receipt.json', 'real_ssl_normalization.json'):
        path = run / name
        bindings[str(path)] = digest(path)
    assert digest(run / Path(__file__).name) == protocol['inputs_sha256'][str(Path(__file__))]
    assert digest(run / 'test_ssl_real_domain_mix.py') == protocol['inputs_sha256'][str(ROOT / 'tests/test_ssl_real_domain_mix.py')]
    resources = {'score_wall_seconds': time.monotonic() - started,
                 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'new_retained_bytes_before_score_record': disk_bytes(run)}
    ledger = [json.loads(line) for line in (run / 'compute_ledger.jsonl').read_text().splitlines()]
    resources['total_recorded_compute_seconds'] = sum(row['seconds'] for row in ledger) + resources['score_wall_seconds']
    resources['time_accounting'] = 'Includes prior failedCPU/tests/prepare/gates; final external process+closing total is in completion.json.'
    assert resources['score_wall_seconds'] < 180 and resources['max_rss_bytes'] < 8 * 1024**3
    assert resources['new_retained_bytes_before_score_record'] < 8 * 1024**3 and resources['total_recorded_compute_seconds'] < 7200
    results = {'status': 'COMPLETE_BOUNDED_REAL_DOMAIN_SSL_MIX', 'scope': protocol['scope'], 'states': states, 'deltas': deltas,
               'initial_heads_sha256': initial_heads, 'resources': resources, 'limits': protocol['limits'], 'data_intervention': 'Fixed50% real training images plus subset domain normalization; not pure appearance causality.',
               'epoch100_evidence': 'Logged metrics auxiliary; only saved best predictions independently recomputed; no extra epoch100 checkpoint.',
               'old_M2_NO_GO_unchanged': True, 'new_test_evaluation': False, 'next_stage_automatically_started': False}
    (run / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    verification = {'status': 'PASS_NUMPY_SAVED_BEST_AND_CLOSED_INPUT_BINDINGS', 'metric_checks': checks,
                    'closed_inputs_sha256': bindings, 'results_sha256': digest(run / 'results.json'),
                    'successful_pair_gate_sha256': digest(run / 'gate1000.json'),
                    'limits': 'Code/payload/provenance integrity and savedbest scoring; not scientific effectiveness or scene-independent replication.'}
    (run / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps({'status': results['status'], 'deltas': deltas, 'resources': resources,
                      'verification': verification['status'], 'input_bindings': len(bindings)}), flush=True)


def pipeline(run):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    run.mkdir()
    groups = {'prepare': 300, 'ssl_sim': 1800, 'ssl_mix': 1800, 'features': 1200,
              'heads_sim': 900, 'heads_mix': 900, 'score': 180, 'closing': 120}
    prior = {'group': 'prepare', 'phase': 'initial_CPU_RED_and_GREEN_and_full_driver_check', 'seconds': 5.214665,
             'note': 'Old1028 mask regression failed as intended; three new CPU interfaces passed twice. No model.'}
    (run / 'compute_ledger.jsonl').write_text(json.dumps(prior) + '\n')
    steps = [('cpu', 'prepare', []), ('prepare', 'prepare', [])]
    steps += [('train', f'ssl_{arm}', ['--arm', arm, '--stop-after', '50']) for arm in ('sim', 'mix')]
    steps += [('gate', 'closing', ['--stop-after', '50'])]
    steps += [('train', f'ssl_{arm}', ['--arm', arm, '--stop-after', '1000']) for arm in ('sim', 'mix')]
    steps += [('gate', 'closing', ['--stop-after', '1000'])]
    steps += [('extract', 'features', ['--arm', arm]) for arm in ('sim', 'mix')]
    steps += [('heads', f'heads_{arm}', ['--arm', arm]) for arm in ('sim', 'mix')]
    steps += [('score', 'score', [])]
    for number, (phase, group, extra) in enumerate(steps):
        records = [json.loads(line) for line in (run / 'compute_ledger.jsonl').read_text().splitlines()]
        used = sum(row['seconds'] for row in records)
        remaining = min(7200 - used, groups[group] - sum(row['seconds'] for row in records if row['group'] == group))
        assert remaining > 0
        command = [sys.executable, '-m', 'pytest', 'tests/test_ssl_real_domain_mix.py', '-q'] if phase == 'cpu' else [
            sys.executable, str(Path(__file__)), phase, '--run', str(run), *extra]
        log_path = run / ('cpu_tests.log' if phase == 'cpu' else f'{number:02d}_{phase}_{group}.log')
        print(json.dumps({'phase': phase, 'group': group, 'extra': extra, 'hard_remaining_seconds': remaining,
                          'log': str(log_path)}), flush=True)
        started = time.monotonic()
        with log_path.open('w') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            violation = None
            try:
                while process.poll() is None:
                    if time.monotonic() - started >= remaining:
                        violation = 'CUMULATIVE_TIME_BUDGET'
                    else:
                        try:
                            status = Path(f'/proc/{process.pid}/status').read_text()
                        except FileNotFoundError:
                            continue  # Normal exit can occur between poll and /proc read.
                        rss = next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmHWM:')), 0)
                        if rss > 8 * 1024**3:
                            violation = 'RSS_BUDGET'
                        live_bytes = 0
                        for path in run.rglob('*'):
                            try:
                                if path.is_file():
                                    live_bytes += path.stat().st_size
                            except FileNotFoundError:
                                continue  # The child atomically replaces last.tmp while monitored.
                        if live_bytes > 8 * 1024**3:
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
                record = {'phase': phase, 'group': group, 'command': command, 'seconds': time.monotonic() - started,
                          'returncode': process.returncode, 'budget_violation': violation, 'log_sha256': digest(log_path),
                          'new_disk_bytes_after': disk_bytes(run)}
                with (run / 'compute_ledger.jsonl').open('a') as ledger:
                    ledger.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        assert process.returncode == 0 and violation is None, f'Stopped at {phase}: see {log_path}'
    records = [json.loads(line) for line in (run / 'compute_ledger.jsonl').read_text().splitlines()]
    total = sum(row['seconds'] for row in records)
    assert total < 7200 and disk_bytes(run) < 8 * 1024**3
    completion = {'status': 'COMPLETE_WITH_ALL_CUMULATIVE_BUDGETS', 'compute_seconds_including_failed_CPU': total,
                  'stage_seconds': {group: sum(row['seconds'] for row in records if row['group'] == group) for group in groups},
                  'new_disk_bytes': disk_bytes(run), 'ledger_sha256': digest(run / 'compute_ledger.jsonl'),
                  'results_sha256': digest(run / 'results.json'), 'verification_sha256': digest(run / 'verification.json'),
                  'next_stage_started': False}
    (run / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pipeline', 'prepare', 'train', 'gate', 'extract', 'heads', 'score'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=('sim', 'mix'))
    parser.add_argument('--stop-after', type=int, choices=(50, 1000))
    args = parser.parse_args()
    if args.phase == 'pipeline':
        pipeline(args.run)
    elif args.phase == 'prepare':
        prepare(args.run)
    elif args.phase == 'train':
        train_prefix(args.run, args.arm, args.stop_after)
    elif args.phase == 'gate':
        check_prefix(args.run, args.stop_after)
    elif args.phase == 'extract':
        extract(args.run, args.arm)
    elif args.phase == 'heads':
        from runs.train_uatd_area_heads import main
        main(args.run / args.arm, args.arm, 'cuda:0')
    else:
        score(args.run)
