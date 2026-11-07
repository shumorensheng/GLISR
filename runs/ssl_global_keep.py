"""Bounded R010 masked SSL with a frozen initial global-feature anchor."""

import os
import argparse
import copy
import hashlib
import json
import math
import random
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs.recompute_uatd_area_results import compare_metrics, dense_metrics, digest, global_metrics
from sonar_encoder.training.data import SonarDataset
from sonar_encoder.training.model import build_encoder, make_patch_mask, masked_latent_loss
from sonar_encoder.training.readout import state_digest
from sonar_encoder.training.train import manifest_digest, optimizer_update, validate
from sonar_encoder.training.uatd_readout import native_sample

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
DATA = Path(os.environ.get('GLISR_OCEANGYM_DATA', 'data/oceangym'))
SSL_MANIFEST = ROOT / 'runs/data/real_ssl_v1/manifest.json'
SSL_NORM = ROOT / 'runs/data/real_ssl_v1/normalization.json'
UATD = ROOT / 'runs/data/uatd_readout_v1'


def initialize_training(config):
    device = torch.device(config['device'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    student = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010').to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    initial = copy.deepcopy(student).eval().requires_grad_(False)
    predictor = nn.Sequential(nn.Linear(student.embed_dim, student.embed_dim * 2), nn.GELU(),
                              nn.Linear(student.embed_dim * 2, student.embed_dim)).to(device)
    groups = defaultdict(list)
    for name, parameter in student.named_parameters():
        rate = config['frontend_lr'] if 'patch_embed' in name or 'mask_token' in name else config['trunk_lr']
        groups[(rate, config['weight_decay'] if parameter.ndim > 1 else 0.)].append(parameter)
    for parameter in predictor.parameters():
        groups[(config['frontend_lr'], config['weight_decay'] if parameter.ndim > 1 else 0.)].append(parameter)
    optimizer = torch.optim.AdamW([{'params': values, 'lr': rate, 'initial_lr': rate, 'weight_decay': decay}
                                  for (rate, decay), values in groups.items()], betas=(.9, .999))
    scaler = torch.amp.GradScaler(device.type, enabled=config['amp'])
    return (student, teacher, initial, predictor, optimizer, scaler,
            torch.Generator().manual_seed(config['seed'] + 1), torch.Generator().manual_seed(config['seed'] + 2))


def training_terms(student, teacher, initial, predictor, samples, normalization, masks, keep):
    with torch.no_grad():
        target = teacher.forward_native(samples, normalization, None)['patch_tokens']
        if keep:
            reference = initial.forward_native(samples, normalization, None)['patch_tokens'].float().mean(1)
    prediction = predictor(student.forward_native(samples, normalization, masks)['patch_tokens'])
    masked = masked_latent_loss(prediction, target, masks.to(prediction.device))
    if keep:
        complete = student.forward_native(samples, normalization, None)['patch_tokens'].float().mean(1)
        preservation = F.smooth_l1_loss(complete, reference, beta=1.)
    else:
        preservation = masked.new_zeros(())
    return masked, preservation, prediction


def next_indices(stream, sample_count, batch_size, sample_rng):
    indices = []
    for _ in range(batch_size):
        if stream['sample_cursor'] == len(stream['epoch_order']):
            stream['epoch'] += 1
            stream['epoch_order'] = torch.randperm(sample_count, generator=sample_rng)
            stream['sample_cursor'] = 0
        indices.append(int(stream['epoch_order'][stream['sample_cursor']]))
        stream['sample_cursor'] += 1
    return indices


def training_state(student, teacher, predictor, optimizer, scaler, sample_rng, mask_rng, stream, device):
    saved = {'student': student.state_dict(), 'teacher': teacher.state_dict(), 'predictor': predictor.state_dict(),
             'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'stream': stream,
             'sample_rng': sample_rng.get_state(), 'mask_rng': mask_rng.get_state(),
             'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
    if torch.device(device).type == 'cuda':
        saved['cuda_rng'] = torch.cuda.get_rng_state_all()
    return saved


def restore_training(saved, student, teacher, predictor, optimizer, scaler, sample_rng, mask_rng, device):
    for name, model in (('student', student), ('teacher', teacher), ('predictor', predictor)):
        model.load_state_dict(saved[name])
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    sample_rng.set_state(saved['sample_rng'])
    mask_rng.set_state(saved['mask_rng'])
    torch.set_rng_state(saved['torch_rng'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def disk_bytes(run):
    return sum(p.stat().st_size for p in run.rglob('*') if p.is_file())


def prepare(run):
    run.mkdir()
    config = json.loads((ROOT / 'runs/r010-seed0-v1/resolved_config.json').read_text())
    config.update(variant='R010', device='cuda:0', code_dir=str(ROOT / config['code_dir']),
                  weights_path=str(ROOT / config['weights_path']))
    manifest = json.loads(SSL_MANIFEST.read_text())
    normalization = json.loads(SSL_NORM.read_text())
    assert normalization['source_manifest_sha256'] == manifest_digest(manifest)
    task = json.loads((UATD / 'manifest.json').read_text())
    task_norm = json.loads((UATD / 'normalization.json').read_text())
    assert task_norm['source_manifest_sha256'] == digest(UATD / 'manifest.json')
    frames = [r for r in manifest['frames'] if r['usage'] in ('ssl_train', 'ssl_validation')]
    bindings = {str(DATA / r['npz_relpath']): r['sha256'] for r in frames}
    bindings.update({str(Path(r['image'])): r['image_sha256'] for split in ('train', 'validation') for r in task[split]})
    metadata = {DATA / r['metadata_relpath'] for r in frames}
    sources = [Path(__file__), ROOT / 'tests/test_ssl_global_keep.py',
               ROOT / 'runs/r010-seed0-v1/resolved_config.json', SSL_MANIFEST, SSL_NORM,
               UATD / 'manifest.json', UATD / 'normalization.json', Path(config['weights_path']),
               ROOT / 'runs/train_uatd_area_heads.py', ROOT / 'runs/recompute_uatd_area_results.py',
               *[ROOT / 'sonar_encoder/training' / name for name in
                 ('data.py', 'model.py', 'frontends.py', 'train.py', 'readout.py', 'uatd_head.py', 'uatd_readout.py', 'artifacts.py')],
               *sorted(Path(config['code_dir']).rglob('*.py')),
               Path('/tmp/pmca_ssl_keep_design_astra_review.txt'), Path('/tmp/pmca_ssl_keep_cpu_tests.log'), *sorted(metadata)]
    bindings.update({str(p): digest(p) for p in sources})
    assert all(digest(Path(p)) == expected for p, expected in bindings.items())
    heads = json.loads((BASE / 'uatd_area_readout_v1/protocol.json').read_text())['heads']
    protocol = {'scope': 'Two R010 1000-successful-update prefixes, only objective factor0/1, three-state UATD development readout, no old Test/10k extension.',
                'config': config, 'ssl_manifest': str(SSL_MANIFEST), 'ssl_normalization': str(SSL_NORM), 'data_root': str(DATA),
                'prepared_manifest': str(UATD / 'manifest.json'), 'task_normalization': str(UATD / 'normalization.json'),
                'heads': heads, 'inputs_sha256': bindings, 'arms': {'masked': 0., 'keep': 1.},
                'loss': 'Original masked SmoothL1 beta1; keep=SmoothL1(final-LN complete patchtokens.float().mean256, frozenE0 same complete input), meanB/channel384, beta1,lambda1,no projector. Both student branches independent.',
                'initial_reference': 'Reconstruct R010 from bound official source/weights before loading latest; E0 independent/frozen, not EMA/optimizer, no initial or E0 checkpoint archive.',
                'precision': 'Original FP16 autocast/GradScaler for SSL, losses FP32; features and heads FP32/TF32off. Original R010 no gain1.5, no window/unit-conversion.',
                'prefix': 'Original10k LR/EMA schedule and1000warmup;50 counted in1000, gate engineering only, student1000 fixed, no loss selection.',
                'pairing': 'Actual successful IDs/mask traces must match at50 and1000; AMPskip consumes data as original; mismatch stops dependencies without silently changingAMP/batches.',
                'budgets': {'cpu_threads': 4, 'gpu_index': 0, 'ssl_seconds_per_arm': 1800, 'feature_seconds_total': 1800,
                            'head_seconds_per_arm': 900, 'score_seconds': 300, 'total_compute_seconds': 10800,
                            'rss_bytes': 8 * 1024**3, 'gpu_bytes': 8 * 1024**3, 'new_peak_bytes': 12 * 1024**3},
                'smoke_projection': 'prior50 wall +950*mean last30 successful-step wall +180s overhead margin <=1800; hard cumulative run budget remains1800 across resumes.',
                'feature_states': ['t0', 'masked', 'keep'], 'feature_batch': 16, 'max_feature_sample_forwards': 22668,
                'metrics': 'Three absolute globalAP/denseAP50, M-T0/K-T0/K-M separate, best main/epoch100 auxiliary, no aggregate/adaptive denominator or added pass threshold.',
                'limits': ['Single map/head/SSL seed0 development only; additional full-view computation and gradient supervision, not equalFLOPs.',
                           'Keep loss low/mean equality does not demonstrate semantic or local-information preservation.',
                           'No claim fixing old10k drop unless corresponding short-prefix pattern appears; no physics/generalencoder claim.',
                           'Preserving global while losing dense adaptation is a tradeoff, not joint benefit.'],
                'stop': 'Two step1000 students +reconstructedT0, three global/dense pairs and savedbest independent scores; no extra seeds/scales/lambda/structure/Test/10k.'}
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for state in protocol['feature_states']:
        folder = run / state
        folder.mkdir()
        child = {'scope': protocol['scope'], 'prepared_manifest': protocol['prepared_manifest'], 'heads': heads,
                 'budgets': protocol['budgets'], 'feature_state': state,
                 'inputs_sha256': {str(p): digest(p) for p in (run / 'protocol.json', UATD / 'manifest.json',
                                   ROOT / 'runs/train_uatd_area_heads.py', ROOT / 'sonar_encoder/training/uatd_head.py')}}
        (folder / 'protocol.json').write_text(json.dumps(child, indent=2) + '\n')
    for p in (Path(__file__), ROOT / 'tests/test_ssl_global_keep.py', Path('/tmp/pmca_ssl_keep_design_astra_review.txt'),
              Path('/tmp/pmca_ssl_keep_cpu_tests.log')):
        shutil.copyfile(p, run / p.name)
    print(json.dumps({'status': 'FROZEN_BEFORE_OFFICIAL_MODEL_COMPUTATION', 'input_bindings': len(bindings),
                      'ssl_train': sum(r['usage'] == 'ssl_train' for r in frames),
                      'ssl_validation': sum(r['usage'] == 'ssl_validation' for r in frames)}), flush=True)


def train_prefix(run, arm, stop_after):
    started = time.monotonic()
    protocol = json.loads((run / 'protocol.json').read_text())
    config = protocol['config']
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    folder = run / arm
    if stop_after == 50:
        (folder / 'checkpoints').mkdir()
        previous_seconds = 0.
    else:
        gate = json.loads((run / 'gate50.json').read_text())
        assert gate['status'] == 'PASS_MATCHED_50_AND_RESOURCE_PROJECTION'
        previous_seconds = json.loads((folder / 'report50.json').read_text())['cumulative_wall_seconds']
    models = initialize_training(config)
    student, teacher, initial, predictor, optimizer, scaler, sample_rng, mask_rng = models
    initial_hash = state_digest(initial.state_dict())
    initial_states = {'student': state_digest(student.state_dict()), 'teacher': state_digest(teacher.state_dict()),
                      'predictor': state_digest(predictor.state_dict())}
    dataset = SonarDataset(Path(protocol['ssl_manifest']), Path(protocol['data_root']), 'ssl_train')
    validation = SonarDataset(Path(protocol['ssl_manifest']), Path(protocol['data_root']), 'ssl_validation')
    normalization = json.loads(Path(protocol['ssl_normalization']).read_text())
    if stop_after == 50:
        stream = {'step': 0, 'epoch': 0, 'sample_cursor': 0, 'epoch_order': torch.randperm(len(dataset), generator=sample_rng),
                  'attempts': 0, 'success_trace': '0' * 64, 'attempt_trace': '0' * 64}
    else:
        saved = torch.load(folder / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert saved['initial_sha256'] == initial_hash and saved['protocol_sha256'] == digest(run / 'protocol.json')
        assert saved['stream']['step'] == 50
        restore_training(saved, student, teacher, predictor, optimizer, scaler, sample_rng, mask_rng, config['device'])
        stream = saved['stream']
        del saved
    student.train()
    torch.cuda.reset_peak_memory_stats()
    windows, window_seconds, window_updates = [], 0., 0
    skipped = 0
    with (folder / 'batches.jsonl').open('a', buffering=1) as batches, (folder / 'metrics.jsonl').open('a', buffering=1) as log:
        while stream['step'] < stop_after:
            step_started = time.monotonic()
            step = stream['step']
            factor = (step + 1) / config['warmup_steps']
            tau = 1 - (1 - config['ema_start']) * (1 + math.cos(math.pi * step / config['steps'])) / 2
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            ids, mask_hash = [], hashlib.sha256()
            masked_value, keep_value = 0., 0.
            for _ in range(config['accumulation']):
                indices = next_indices(stream, len(dataset), config['micro_batch'], sample_rng)
                samples = [dataset[i] for i in indices]
                ids.extend(sample['sample_id'] for sample in samples)
                masks = make_patch_mask(len(samples), mask_rng)
                mask_hash.update(masks.numpy().tobytes())
                with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                    masked, preservation, prediction = training_terms(student, teacher, initial, predictor, samples, normalization, masks,
                                                                        protocol['arms'][arm] == 1.)
                loss = masked + preservation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step{step}: {ids}')
                masked_value += float(masked.detach()) / config['accumulation']
                keep_value += float(preservation.detach()) / config['accumulation']
                scaler.scale(loss / config['accumulation']).backward()
            updated, norm = optimizer_update(student, teacher, optimizer, scaler, tau, config['clip_grad'])
            stream['attempts'] += 1
            batch = {'attempt': stream['attempts'], 'step_before': step, 'sample_ids': ids, 'mask_sha256': mask_hash.hexdigest(),
                     'updated': updated, 'lr_factor': factor, 'tau': tau, 'grad_norm': norm,
                     'masked_loss': masked_value, 'keep_loss': keep_value, 'amp_scale': scaler.get_scale()}
            identity = json.dumps([ids, batch['mask_sha256']], separators=(',', ':')).encode()
            stream['attempt_trace'] = hashlib.sha256(bytes.fromhex(stream['attempt_trace']) + identity + bytes([updated])).hexdigest()
            batches.write(json.dumps(batch) + '\n')
            if not updated:
                skipped += 1
                if skipped >= 3:
                    raise FloatingPointError('Three consecutive AMP overflows; stopped without changing precision/data.')
                continue
            if not math.isfinite(norm):
                raise FloatingPointError(f'Nonfinite successful gradient at step{step}')
            skipped = 0
            stream['success_trace'] = hashlib.sha256(bytes.fromhex(stream['success_trace']) + identity).hexdigest()
            stream['step'] += 1
            window_seconds += time.monotonic() - step_started
            window_updates += 1
            if stream['step'] % config['log_every'] == 0:
                row = {'step': stream['step'], 'seconds': window_seconds, 'updates': window_updates,
                       'masked_loss': masked_value, 'keep_loss': keep_value, 'grad_norm': norm,
                       'lr_factor': factor, 'tau': tau, 'success_trace': stream['success_trace']}
                log.write(json.dumps(row) + '\n')
                print(json.dumps({'arm': arm, **row}), flush=True)
                windows.append(row)
                window_seconds, window_updates = 0., 0
            if stream['step'] % config['validation_every'] == 0 and stream['step'] < stop_after:
                validation_metrics = validate(student, teacher, predictor, validation, normalization, config)
                log.write(json.dumps({'kind': 'validation', 'step': stream['step'], 'metrics': validation_metrics}) + '\n')
                print(json.dumps({'arm': arm, 'step': stream['step'], 'validation': validation_metrics}), flush=True)
            assert previous_seconds + time.monotonic() - started < 1800
    metrics = validate(student, teacher, predictor, validation, normalization, config)
    assert state_digest(initial.state_dict()) == initial_hash
    checkpoint = training_state(student, teacher, predictor, optimizer, scaler, sample_rng, mask_rng, stream, config['device'])
    checkpoint.update(initial_sha256=initial_hash, resolved_config=config, protocol_sha256=digest(run / 'protocol.json'))
    temporary = folder / 'checkpoints/last.tmp'
    torch.save(checkpoint, temporary)
    atomic_peak_bytes = disk_bytes(run)
    assert atomic_peak_bytes < protocol['budgets']['new_peak_bytes']
    temporary.replace(folder / 'checkpoints/last.pt')
    del checkpoint
    report = {'status': 'COMPLETE_PREFIX', 'arm': arm, 'step': stream['step'], 'attempts': stream['attempts'],
              'success_trace': stream['success_trace'], 'attempt_trace': stream['attempt_trace'],
              'initial_states_sha256': initial_states, 'initial_reference_sha256_before': initial_hash,
              'initial_reference_sha256_after': state_digest(initial.state_dict()),
              'last_sha256': digest(folder / 'checkpoints/last.pt'), 'validation_original_masked': metrics,
              'last30_seconds_per_update': sum(x['seconds'] for x in windows[-3:]) / sum(x['updates'] for x in windows[-3:]),
              'cumulative_wall_seconds': previous_seconds + time.monotonic() - started,
              'resources': {'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved(), 'atomic_peak_disk_bytes': atomic_peak_bytes}}
    assert report['cumulative_wall_seconds'] < 1800 and report['resources']['max_rss_bytes'] < 8 * 1024**3
    assert report['resources']['max_gpu_reserved_bytes'] < 8 * 1024**3
    (folder / f'report{stop_after}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def check_prefix(run, step):
    protocol = json.loads((run / 'protocol.json').read_text())
    reports = {arm: json.loads((run / arm / f'report{step}.json').read_text()) for arm in ('masked', 'keep')}
    assert reports['masked']['initial_states_sha256'] == reports['keep']['initial_states_sha256']
    traces = {}
    for arm in ('masked', 'keep'):
        report = reports[arm]
        assert report['step'] == step and report['initial_reference_sha256_before'] == report['initial_reference_sha256_after']
        assert digest(run / arm / 'checkpoints/last.pt') == report['last_sha256']
        batches = [json.loads(line) for line in (run / arm / 'batches.jsonl').read_text().splitlines()]
        successful = [b for b in batches if b['updated']]
        assert len(successful) == step and len(batches) == report['attempts']
        traces[arm] = [(b['sample_ids'], b['mask_sha256']) for b in successful]
        chain = '0' * 64
        for number, batch in enumerate(successful):
            assert batch['step_before'] == number
            assert batch['lr_factor'] == (number + 1) / protocol['config']['warmup_steps']
            expected_tau = 1 - (1 - protocol['config']['ema_start']) * (1 + math.cos(math.pi * number / protocol['config']['steps'])) / 2
            assert batch['tau'] == expected_tau
            identity = json.dumps([batch['sample_ids'], batch['mask_sha256']], separators=(',', ':')).encode()
            chain = hashlib.sha256(bytes.fromhex(chain) + identity).hexdigest()
        assert chain == report['success_trace']
    assert traces['masked'] == traces['keep']
    projections = {arm: reports[arm]['cumulative_wall_seconds'] + 950 * reports[arm]['last30_seconds_per_update'] + 180
                   for arm in ('masked', 'keep')} if step == 50 else {}
    if step == 50:
        assert max(projections.values()) <= 1800
    record = {'status': 'PASS_MATCHED_50_AND_RESOURCE_PROJECTION' if step == 50 else 'PASS_MATCHED_1000_PREFIXES',
              'step': step, 'reports_sha256': {arm: digest(run / arm / f'report{step}.json') for arm in reports},
              'success_trace_sha256': reports['masked']['success_trace'], 'projected_seconds': projections,
              'no_loss_or_task_performance_gate': True}
    (run / f'gate{step}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


@torch.no_grad()
def extract(run, state):
    started = time.monotonic()
    protocol = json.loads((run / 'protocol.json').read_text())
    assert json.loads((run / 'gate1000.json').read_text())['status'] == 'PASS_MATCHED_1000_PREFIXES'
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
    reference = json.loads((run / 'masked/report1000.json').read_text())['initial_states_sha256']['student']
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
            assert prior_seconds + time.monotonic() - started < protocol['budgets']['feature_seconds_total']
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
    assert resources['cumulative_feature_seconds'] < 1800 and resources['max_rss_bytes'] < 8 * 1024**3
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


def heads(run, state):
    from runs.train_uatd_area_heads import main

    main(run / state, state, 'cuda:0')


def score(run):
    started = time.monotonic()
    torch.set_num_threads(4)
    protocol = json.loads((run / 'protocol.json').read_text())
    bindings = {}
    for p, expected in protocol['inputs_sha256'].items():
        actual = digest(Path(p))
        assert actual == expected, p
        bindings[p] = actual
    assert json.loads((run / 'gate1000.json').read_text())['status'] == 'PASS_MATCHED_1000_PREFIXES'
    manifest = json.loads(Path(protocol['prepared_manifest']).read_text())
    validation_ids = [row['row_id'] for row in manifest['validation']]
    boxes = [np.asarray(row['boxes'], dtype=np.float32) for row in manifest['validation']]
    labels = [np.asarray(row['labels'], dtype=np.int64) for row in manifest['validation']]
    states, checks, initial_heads = {}, {}, {}
    for state in protocol['feature_states']:
        folder, output = run / state, run / state / f'heads_{state}'
        child = json.loads((folder / 'protocol.json').read_text())
        cache = json.loads((folder / 'cache_manifest.json').read_text())
        report = json.loads((output / 'report.json').read_text())
        log = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        assert report['status'] == 'COMPLETE_FIXED_DEVELOPMENT_HEADS'
        assert report['arm'] == state and report['epochs'] == len(log) == 100 and report['updates'] == 19000
        assert report['class_names'] == manifest['class_names'] and report['final_test_evaluated'] is False
        assert cache['protocol_sha256'] == digest(folder / 'protocol.json')
        assert cache['parent_protocol_sha256'] == digest(run / 'protocol.json')
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
    assert initial_heads['t0'] == initial_heads['masked'] == initial_heads['keep']
    deltas = {}
    for comparison, a, b in (('M-T0', 'masked', 't0'), ('K-T0', 'keep', 't0'), ('K-M', 'keep', 'masked')):
        deltas[comparison] = {name: {task: states[a][source][task][key] - states[b][source][task][key]
                                    for task, key in (('global', 'map'), ('dense', 'map50'))}
                              for name, source in (('best', 'best'), ('epoch100_log_only', 'epoch100_log_only'))}
    for arm in ('masked', 'keep'):
        report = json.loads((run / arm / 'report1000.json').read_text())
        assert digest(run / arm / 'checkpoints/last.pt') == report['last_sha256']
        for name in ('report50.json', 'report1000.json', 'batches.jsonl', 'metrics.jsonl', 'checkpoints/last.pt'):
            path = run / arm / name
            bindings[str(path)] = digest(path)
    for name in ('protocol.json', 'gate50.json', 'gate1000.json', Path(__file__).name,
                 'test_ssl_global_keep.py', 'pmca_ssl_keep_design_astra_review.txt', 'pmca_ssl_keep_cpu_tests.log'):
        path = run / name
        bindings[str(path)] = digest(path)
    assert digest(run / Path(__file__).name) == protocol['inputs_sha256'][str(Path(__file__))]
    assert digest(run / 'test_ssl_global_keep.py') == protocol['inputs_sha256'][str(ROOT / 'tests/test_ssl_global_keep.py')]
    resources = {'score_wall_seconds': time.monotonic() - started,
                 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'new_retained_bytes_before_score_record': disk_bytes(run)}
    resources['total_recorded_compute_seconds'] = resources['score_wall_seconds'] + sum(
        json.loads((run / arm / 'report1000.json').read_text())['cumulative_wall_seconds'] for arm in ('masked', 'keep')) + sum(
        json.loads((run / state / 'cache_manifest.json').read_text())['resources']['wall_seconds'] +
        json.loads((run / state / f'heads_{state}/report.json').read_text())['resources']['wall_seconds'] for state in states)
    assert resources['score_wall_seconds'] < 300 and resources['max_rss_bytes'] < 8 * 1024**3
    assert resources['new_retained_bytes_before_score_record'] < 12 * 1024**3 and resources['total_recorded_compute_seconds'] < 10800
    results = {'status': 'COMPLETE_BOUNDED_SSL_GLOBAL_KEEP', 'scope': protocol['scope'], 'states': states, 'deltas': deltas,
               'initial_heads_sha256': initial_heads, 'resources': resources, 'limits': protocol['limits'],
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    stages = parser.add_subparsers(dest='action', required=True)
    stages.add_parser('prepare')
    train_parser = stages.add_parser('train')
    train_parser.add_argument('--arm', choices=('masked', 'keep'), required=True)
    train_parser.add_argument('--until', type=int, choices=(50, 1000), required=True)
    gate_parser = stages.add_parser('check')
    gate_parser.add_argument('--step', type=int, choices=(50, 1000), required=True)
    for stage in ('extract', 'heads'):
        state_parser = stages.add_parser(stage)
        state_parser.add_argument('--state', choices=('t0', 'masked', 'keep'), required=True)
    stages.add_parser('score')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.run)
    elif args.action == 'train':
        train_prefix(args.run, args.arm, args.until)
    elif args.action == 'check':
        check_prefix(args.run, args.step)
    elif args.action == 'extract':
        extract(args.run, args.state)
    elif args.action == 'heads':
        heads(args.run, args.state)
    else:
        score(args.run)
