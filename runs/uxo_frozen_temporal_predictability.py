"""Fixed E0 UXO features and four record-balanced affine temporal controls."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'uxo_frozen_temporal_predictability_v1'
AUTH = ROOT / '.aris/traces/experiment-plan/2026-10-08_uxo_temporal_predictability/contract.json'
AUTH_SHA = 'c83f2738227b6b2f474c8f45c0d37dec7dae89438fee48cbe6d6d51f1278bcf2'
INITIAL = 'b9fc4072daeecd3250c87c4f4b6994248f5a84526d3e62effa5bb050dbbf973e'
SALT = 'pmca-uxo-temporal-target-hash-cycle-v1'
LIMITS = {'prepare': 1200, 'extract': 3600, 'fit': 300, 'score': 600, 'close': 600}
PHASE_START = time.monotonic()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')


def encode_id(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode() + b'\n'


def id_digest(rows):
    return hashlib.sha256(b''.join(encode_id(row) for row in rows)).hexdigest()


def ledger():
    return [json.loads(line) for line in (RUN / 'compute_ledger.jsonl').read_text().splitlines()]


def resource_check(phase):
    rows = ledger()
    elapsed = time.monotonic() - PHASE_START
    group = ('score', 'close') if phase in ('score', 'close') else (phase,)
    charged = sum(r['elapsed_seconds'] for r in rows)
    previous_group = sum(r['elapsed_seconds'] for r in rows if r['phase'] in group)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = sys.modules['torch'].cuda.max_memory_reserved(0) if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized() else 0
    disk = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    assert charged + elapsed <= 7200 and previous_group + elapsed <= LIMITS[phase], (phase, charged, elapsed)
    assert rss <= 8 * 1024**3 and gpu <= 8 * 1024**3 and disk <= 256 * 1024**2, (rss, gpu, disk)
    return {'RSS_peak_bytes': rss, 'GPU_reserved_peak_bytes': gpu, 'new_disk_bytes': disk,
            'current_phase_seconds': elapsed, 'closed_charged_seconds': charged}


def fit_affine(record_pairs):
    count = len(record_pairs)
    mu_x = np.mean([x.mean(0) for x, _ in record_pairs], axis=0)
    mu_y = np.mean([y.mean(0) for _, y in record_pairs], axis=0)
    width = len(mu_x)
    cxx = np.zeros((width, width), dtype=np.float64)
    cxy = np.zeros_like(cxx)
    for x, y in record_pairs:
        xc, yc = x - mu_x, y - mu_y
        cxx += (xc.T @ xc) / (count * len(x))
        cxy += (xc.T @ yc) / (count * len(x))
    trace = np.trace(cxx)
    assert np.isfinite(trace) and trace > 0
    lam = .001 * trace / width
    normal = cxx + lam * np.eye(width)
    a = np.linalg.solve(normal, cxy)
    b = mu_y - mu_x @ a
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(normal @ a, cxy, rtol=1e-9, atol=1e-12)
    return {'A': a, 'b': b, 'mu_x': mu_x, 'mu_y': mu_y, 'cxx': cxx, 'cxy': cxy,
            'lambda': lam, 'normal_equation_maxabs': float(np.abs(normal @ a - cxy).max())}


def cycle_donors(record_id, keys, offset):
    m = max(len(keys) - 8, 0)
    order = sorted(range(m), key=lambda i: (
        hashlib.sha256((SALT + '\0' + record_id + '\0' + str(offset) + '\0' + keys[i]).encode()).hexdigest(), keys[i]))
    donors = np.arange(m)
    for j, recipient in enumerate(order):
        donors[recipient] = order[(j + 1) % m]
    return donors


def pair_rows(record_id, keys, offset, shuffled=False):
    m = max(len(keys) - 8, 0)
    donors = cycle_donors(record_id, keys, offset) if shuffled else np.arange(m)
    return [[record_id, keys[i], keys[int(donors[i]) + offset], offset] for i in range(m)]


def prepare():
    assert digest(AUTH) == AUTH_SHA
    auth = json.loads(AUTH.read_text())
    definition_path = Path(auth['definition']['path'])
    assert digest(definition_path) == auth['definition']['sha256']
    definition = json.loads(definition_path.read_text())
    manifest_path = Path(auth['inputs']['source_manifest'])
    train_path = Path(auth['inputs']['original_train_list'])
    known = {r['path']: r['sha256'] for r in definition['sources']}
    assert digest(manifest_path) == known[str(manifest_path)]
    assert digest(train_path) == known[str(train_path)]
    all_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    by_path = {r['native_npz']: r for r in all_rows}
    rows = [by_path[p] for p in train_path.read_text().splitlines()]
    split = definition['prospective_record_partition']
    fit_ids, dev_ids = set(split['fitting_ids_rank_order']), set(split['development_ids_rank_order'])
    assert len(fit_ids) == 60 and len(dev_ids) == 14 and fit_ids.isdisjoint(dev_ids)
    assert {r['recording'] for r in rows} == fit_ids | dev_ids
    conversion_path = manifest_path.parents[3] / 'audit/uxo/conversion_verified.json'
    conversion = {r['recording']: r for r in json.loads(conversion_path.read_text())['recordings']}
    norm_path = Path(auth['encoder']['normalization']['path'])
    assert digest(norm_path) == auth['encoder']['normalization']['sha256']
    norm = json.loads(norm_path.read_text())
    assert (norm['mean'], norm['std']) == (.025372744258233756, .036703725756021106)
    records, cursors = [], {'fitting': 0, 'development': 0}
    row_hash = {role: hashlib.sha256() for role in cursors}
    natural = {role: {d: hashlib.sha256() for d in (1, 8)} for role in cursors}
    shuffled = {d: hashlib.sha256() for d in (1, 8)}
    exact_current = {d: 0 for d in (1, 8)}
    batches = 0
    for row in sorted(rows, key=lambda r: r['recording']):
        rid = row['recording']
        keys = sorted(row['frame_keys'], key=lambda k: int(k.removeprefix('frame_')))
        assert len(keys) == row['frames'] and len(set(int(k[6:]) for k in keys)) == len(keys)
        role = 'fitting' if rid in fit_ids else 'development'
        path = Path(auth['inputs']['current_payload_path_rule'].format(record_id=rid))
        before = path.stat()
        actual_sha = digest(path)
        after = path.stat()
        old = conversion[rid]
        assert actual_sha == old['npz_sha256'] and before.st_size == old['npz_bytes']
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        members = []
        with zipfile.ZipFile(path) as z:
            for key in keys:
                info = z.getinfo(key + '.npy')
                with z.open(info) as f:
                    version = np.lib.format.read_magic(f)
                    shape, fortran, dtype = np.lib.format._read_array_header(f, version)
                assert dtype == np.dtype('uint8') and len(shape) == 2 and all(v > 0 for v in shape)
                members.append([key, list(shape), info.CRC, info.file_size, info.compress_size])
                row_hash[role].update(encode_id([rid, key]))
        record = {'record_id': rid, 'role': role, 'frames': len(keys), 'cache_start': cursors[role],
                  'archive_path': str(path), 'archive_bytes': before.st_size, 'archive_mtime_ns': before.st_mtime_ns,
                  'archive_sha256': actual_sha, 'archive_identity_origin': 'matched existing local conversion receipt, not author checksum',
                  'members': members}
        for d in (1, 8):
            natural_rows = pair_rows(rid, keys, d)
            for item in natural_rows:
                natural[role][d].update(encode_id(item))
            if role == 'fitting':
                shuffled_rows = pair_rows(rid, keys, d, True)
                assert sorted(x[2] for x in natural_rows) == sorted(x[2] for x in shuffled_rows)
                assert all(a[2] != b[2] for a, b in zip(natural_rows, shuffled_rows))
                for item in shuffled_rows:
                    shuffled[d].update(encode_id(item))
                    exact_current[d] += item[1] == item[2]
        cursors[role] += len(keys)
        batches += math.ceil(len(keys) / 16)
        records.append(record)
        resource_check('prepare')
        print('BOUND_ARCHIVE', len(records), rid, before.st_size, flush=True)
    assert cursors == {'fitting': 48163, 'development': 11607}
    assert exact_current == {1: 60, 8: 56}
    for role in cursors:
        for d in (1, 8):
            assert natural[role][d].hexdigest() == definition['pair_universe']['natural_population_sha256'][role][str(d)]
    for d in (1, 8):
        assert shuffled[d].hexdigest() == definition['hash_cycle_training_control']['map_population_sha256'][str(d)]
    write_json(RUN / 'records.json', records)
    code_paths = [Path(__file__), ROOT / 'runs/uxo_frozen_temporal_predictability_score.py',
                  ROOT / 'tests/test_uxo_frozen_temporal_predictability.py', ROOT / 'sonar_encoder/training/model.py',
                  ROOT / 'sonar_encoder/training/frontends.py', ROOT / 'sonar_encoder/training/readout.py']
    (RUN / 'source_snapshot').mkdir()
    for path in code_paths:
        (RUN / 'source_snapshot' / path.name).write_bytes(path.read_bytes())
    import torch
    protocol = {'unit': auth['unit'], 'authorization_sha256': AUTH_SHA, 'definition_path': str(definition_path),
                'definition_sha256': auth['definition']['sha256'], 'resources': auth['resources'],
                'scientific_contract': auth, 'records_sha256': digest(RUN / 'records.json'),
                'cache_row_ids_sha256': {role: h.hexdigest() for role, h in row_hash.items()},
                'counts': {'record_count': 74, 'sample_forwards': 59770, 'batches': batches,
                           'role_frames': cursors, 'fit_pairs_each_offset': 47683, 'dev_pairs_each_offset': 11495},
                'source_sha256': {str(path): digest(path) for path in code_paths},
                'input_sha256': {str(path): digest(path) for path in (AUTH, definition_path, manifest_path, train_path, norm_path, conversion_path)},
                'normalization': {'mean': norm['mean'], 'std': norm['std']},
                'runtime': {'python': sys.executable, 'python_version': sys.version, 'torch': torch.__version__,
                            'torch_cuda': torch.version.cuda, 'torch_file': torch.__file__, 'numpy': np.__version__,
                            'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                            'num_threads': 4, 'TF32': False, 'AMP': False, 'deterministic_algorithms': False},
                'static_disk': {'vector_payload_bytes': 91806720, 'hard_cap_bytes': 256 * 1024**2,
                                'planning_upper_bound_bytes': 192 * 1024**2, 'no_new_images_weights_or_patch_fields': True},
                'archive_identity_rehash_at_close': False,
                'review_sha256': digest(RUN / 'preflight/source_review.json')}
    assert sys.executable == sys.executable
    write_json(RUN / 'protocol.json', protocol)
    write_json(RUN / 'prepare_receipt.json', {'status': 'PASS_INPUT_SOURCE_RUNTIME_FROZEN', 'protocol_sha256': digest(RUN / 'protocol.json'),
               'archive_count': 74, 'samples': 59770, 'batches': batches, 'GPU_initialized': torch.cuda.is_initialized(),
               'array_payloads_decoded': 0, 'resources': resource_check('prepare')})


def gpu_entry():
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    gpu = next(line.split(', ') for line in query.splitlines() if line.startswith('0, '))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    conflicts, foreign, disappeared = [], [], []
    for line in apps.splitlines():
        uuid, pid, executable, memory = line.split(', ')
        if uuid == gpu[1] and int(pid) != os.getpid():
            proc = Path('/proc') / pid
            try:
                uid = proc.stat().st_uid
                if uid != os.getuid():
                    foreign.append({'pid': int(pid), 'UID': uid, 'executable': executable, 'GPU_memory_MiB': memory})
                elif (proc / 'cwd').resolve(strict=True) == ROOT:
                    conflicts.append(int(pid))
            except FileNotFoundError:
                disappeared.append(int(pid))
    evidence = {'gpu_query': query, 'process_metadata': apps, 'foreign_UIDs': foreign,
                'same_project_conflicts': conflicts, 'disappeared': disappeared,
                'free_bytes': int(gpu[2]) * 1024**2}
    write_json(RUN / 'gpu_entry.json', evidence)
    assert evidence['free_bytes'] >= 24 * 1024**3 and not conflicts, ('DEFERRED_GPU_ENTRY', evidence)


def extract():
    gpu_entry()
    import torch
    from sonar_encoder.training.model import build_encoder, prepare_images
    from sonar_encoder.training.readout import state_digest
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('highest')
    p = json.loads((RUN / 'protocol.json').read_text())
    records = json.loads((RUN / 'records.json').read_text())
    assert digest(RUN / 'records.json') == p['records_sha256']
    for path, expected in p['source_sha256'].items():
        assert digest(path) == expected
    torch.manual_seed(0)
    model = build_encoder(Path(p['scientific_contract']['encoder']['code_dir']), Path(p['scientific_contract']['encoder']['weights']), 'R010')
    before = state_digest(model.state_dict())
    assert before == INITIAL
    model.eval().requires_grad_(False).to('cuda:0')
    assert all(v.dtype == torch.float32 for v in model.parameters())
    loaded_sources = {str(Path(m.__file__).resolve()): digest(m.__file__) for m in sys.modules.values()
                      if hasattr(m, '__file__') and m.__file__ and str(m.__file__).startswith(p['scientific_contract']['encoder']['code_dir']) and str(m.__file__).endswith('.py')}
    write_json(RUN / 'encoder_entry_receipt.json', {'protocol_sha256': digest(RUN / 'protocol.json'),
               'state_sha256': before, 'loaded_official_source_sha256': loaded_sources,
               'model_parameters': sum(v.numel() for v in model.parameters()), 'gradients_enabled_on_parameters': any(v.requires_grad for v in model.parameters()),
               'AMP_enabled': torch.is_autocast_enabled('cuda'), 'TF32_matmul': torch.backends.cuda.matmul.allow_tf32,
               'TF32_cudnn': torch.backends.cudnn.allow_tf32, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'drop_path': [getattr(b.drop_path1, 'drop_prob', 0.) for b in model.backbone.blocks], 'resources': resource_check('extract')})
    caches = {role: np.lib.format.open_memmap(RUN / filename, mode='w+', dtype=np.float32, shape=(n, 384))
              for role, filename, n in [('fitting', 'fit.npy', 48163), ('development', 'dev.npy', 11607)]}
    batches, samples, prefix_times = 0, 0, []
    with torch.inference_mode():
        for record in records:
            path = Path(record['archive_path'])
            st = path.stat()
            assert (st.st_size, st.st_mtime_ns) == (record['archive_bytes'], record['archive_mtime_ns'])
            with zipfile.ZipFile(path) as z:
                for begin in range(0, record['frames'], 16):
                    started = time.monotonic()
                    members = record['members'][begin:begin + 16]
                    views = []
                    for key, shape, crc, usize, csize in members:
                        info = z.getinfo(key + '.npy')
                        assert (info.CRC, info.file_size, info.compress_size) == (crc, usize, csize)
                        array = np.lib.format.read_array(io.BytesIO(z.read(info)), allow_pickle=False)
                        assert array.dtype == np.uint8 and list(array.shape) == shape
                        views.append({'intensity': torch.from_numpy(array.astype(np.float32)).unsqueeze(0) / 255.,
                                      'valid': torch.ones(shape, dtype=torch.bool)})
                    images = prepare_images(views, p['normalization'], None).to('cuda:0')
                    vectors = model(images)['patch_tokens'].float().mean(dim=1).cpu().numpy()
                    assert vectors.shape == (len(members), 384) and np.isfinite(vectors).all()
                    start = record['cache_start'] + begin
                    caches[record['role']][start:start + len(members)] = vectors
                    caches[record['role']].flush()
                    samples += len(members)
                    batches += 1
                    duration = time.monotonic() - started
                    if batches <= 4:
                        prefix_times.append(duration)
                    observed = resource_check('extract')
                    if batches == 4:
                        assert samples == 64
                        projected = 1.5 * np.mean(prefix_times) * (p['counts']['batches'] - 4)
                        elapsed = time.monotonic() - PHASE_START
                        available_encode = 3600 - elapsed
                        available_global = 7200 - observed['closed_charged_seconds'] - elapsed - 300 - 600
                        assert projected <= min(available_encode, available_global), (projected, available_encode, available_global)
                        write_json(RUN / 'prefix_gate.json', {'status': 'PASS_ENGINEERING_AUTOMATIC_CONTINUATION',
                                   'samples_counted_in_formal_total': 64, 'normal_batches': 4, 'batch_wall_seconds': prefix_times,
                                   'remaining_batches': p['counts']['batches'] - 4, 'factor': 1.5,
                                   'conditional_remaining_projection_seconds': float(projected), 'available_encode_seconds': available_encode,
                                   'available_global_after_fit_close_reserve_seconds': available_global, 'state_sha256': before, 'resources': observed})
                        print('PREFIX64_PASS_AUTOMATIC_CONTINUATION', flush=True)
            print('ENCODED_RECORD', record['record_id'], 'samples', samples, 'batches', batches, flush=True)
    for cache in caches.values():
        cache.flush()
    assert samples == 59770 and batches == p['counts']['batches']
    after = state_digest(model.state_dict())
    assert after == before == INITIAL
    write_json(RUN / 'extraction_receipt.json', {'status': 'PASS_COMPLETE_FROZEN_E0_EXTRACTION',
               'samples': samples, 'batches': batches, 'state_before_sha256': before, 'state_after_sha256': after,
               'cache_sha256': {name: digest(RUN / name) for name in ('fit.npy', 'dev.npy')},
               'cache_row_ids_sha256': p['cache_row_ids_sha256'], 'resources': resource_check('extract')})


def fit():
    p = json.loads((RUN / 'protocol.json').read_text())
    records = json.loads((RUN / 'records.json').read_text())
    cache = np.load(RUN / 'fit.npy', mmap_mode='r', allow_pickle=False)
    maps, reports = {}, {}
    for d in (1, 8):
        for condition in ('ordered', 'shuffled'):
            record_pairs = []
            for r in records:
                if r['role'] == 'fitting':
                    z = np.asarray(cache[r['cache_start']:r['cache_start'] + r['frames']], dtype=np.float64)
                    m = r['frames'] - 8
                    donors = cycle_donors(r['record_id'], [v[0] for v in r['members']], d) if condition == 'shuffled' else np.arange(m)
                    record_pairs.append((z[:m], z[donors + d]))
            fitted = fit_affine(record_pairs)
            prefix = f'{condition}_{d}'
            for key in ('A', 'b', 'mu_y', 'cxy'):
                maps[prefix + '_' + key] = fitted[key]
            if condition == 'ordered' and d == 1:
                maps['cxx'], maps['mu_x'] = fitted['cxx'], fitted['mu_x']
            assert np.array_equal(maps['cxx'], fitted['cxx']) and np.array_equal(maps['mu_x'], fitted['mu_x'])
            maps[prefix + '_lambda'] = np.array(fitted['lambda'], dtype=np.float64)
            reports[prefix] = {'lambda': fitted['lambda'], 'normal_equation_maxabs': fitted['normal_equation_maxabs']}
            resource_check('fit')
            print('AFFINE_FIT_COMPLETE', prefix, flush=True)
    np.savez(RUN / 'maps.npz', **maps)
    for d in (1, 8):
        reports[str(d) + '_natural_shuffle_target_mean_maxabs'] = float(np.abs(maps[f'ordered_{d}_mu_y'] - maps[f'shuffled_{d}_mu_y']).max())
    dev = np.load(RUN / 'dev.npy', mmap_mode='r', allow_pickle=False)
    scores = {'offsets': {}}
    for d in (1, 8):
        rule_rows = {key: [] for key in ('copy', 'fit_mean', 'ordered_affine', 'shuffled_affine')}
        for r in records:
            if r['role'] == 'development':
                z = np.asarray(dev[r['cache_start']:r['cache_start'] + r['frames']], dtype=np.float64)
                m = r['frames'] - 8
                x, y = z[:m], z[d:d + m]
                predictions = {'copy': x, 'fit_mean': np.broadcast_to(maps[f'ordered_{d}_mu_y'], y.shape),
                               'ordered_affine': x @ maps[f'ordered_{d}_A'] + maps[f'ordered_{d}_b'],
                               'shuffled_affine': x @ maps[f'shuffled_{d}_A'] + maps[f'shuffled_{d}_b']}
                for rule, prediction in predictions.items():
                    error = float(np.square(prediction - y).mean())
                    assert np.isfinite(error)
                    rule_rows[rule].append({'record_id': r['record_id'], 'pairs': m, 'mse': error})
        values = {rule: {'records': rr, 'macro_record_mse': float(np.mean([v['mse'] for v in rr])),
                         'pair_weighted_mse': float(sum(v['pairs'] * v['mse'] for v in rr) / sum(v['pairs'] for v in rr))}
                  for rule, rr in rule_rows.items()}
        ordered = values['ordered_affine']['macro_record_mse']
        scores['offsets'][str(d)] = {'rules': values, 'differences': {
            'ordered_minus_copy': ordered - values['copy']['macro_record_mse'],
            'ordered_minus_fit_mean': ordered - values['fit_mean']['macro_record_mse'],
            'ordered_minus_shuffled': ordered - values['shuffled_affine']['macro_record_mse']},
            'paired_record_differences': [
                {'record_id': row['record_id'], 'pairs': row['pairs'],
                 'ordered_minus_copy': row['mse'] - values['copy']['records'][i]['mse'],
                 'ordered_minus_fit_mean': row['mse'] - values['fit_mean']['records'][i]['mse'],
                 'ordered_minus_shuffled': row['mse'] - values['shuffled_affine']['records'][i]['mse']}
                for i, row in enumerate(values['ordered_affine']['records'])]}
        resource_check('fit')
    write_json(RUN / 'scores.json', scores)
    write_json(RUN / 'fit_results.json', {'status': 'PASS_FOUR_FIXED_FP64_MAPS', 'maps_file': 'maps.npz',
               'map_sha256': digest(RUN / 'maps.npz'), 'map_reports': reports,
               'source_target': 'frozen E0 representation, not acoustic/semantic ground truth', 'resources': resource_check('fit')})


def close():
    p = json.loads((RUN / 'protocol.json').read_text())
    scores = json.loads((RUN / 'scores.json').read_text())
    verification = json.loads((RUN / 'verification.json').read_text())
    for path, expected in p['source_sha256'].items():
        assert digest(path) == expected
    resource_check('close')
    write_json(RUN / 'results.json', {'status': 'NUMERIC_COMPLETE_FROZEN_FEATURE_PREDICTABILITY_ONLY',
               'target': 'T0 b9fc final-LN patch mean', 'offsets': scores['offsets'],
               'primary': '14-record unweighted mean of within-record vector MSE/384, separately per offset',
               'auxiliary': 'pair-count weighted error, separately per offset',
               'source_context': '60/14 original-train record development split; not cold/object/site independent',
               'scientific_limits': 'No encoder/predictor neural training, semantic/physical GT, transfer or novelty inference; affine negativity cannot exclude other temporal models.'})
    write_json(RUN / 'close_receipt.json', {'status': 'SCORING_AND_SOURCE_CLOSED',
               'protocol_sha256': digest(RUN / 'protocol.json'), 'results_sha256': digest(RUN / 'results.json'),
               'verification_sha256': digest(RUN / 'verification.json'), 'independent_status': verification['status'],
               'archive_inputs': 'bound once in prepare to exact existing local conversion SHA; no close archive-byte scan',
               'resources': resource_check('close')})


def pipeline():
    phases = [('prepare', 'runs.uxo_frozen_temporal_predictability'), ('extract', 'runs.uxo_frozen_temporal_predictability'),
              ('fit', 'runs.uxo_frozen_temporal_predictability'), ('score', 'runs.uxo_frozen_temporal_predictability_score'),
              ('close', 'runs.uxo_frozen_temporal_predictability')]
    for phase, module in phases:
        rows = ledger()
        group = ('score', 'close') if phase in ('score', 'close') else (phase,)
        allowed = min(7200 - sum(r['elapsed_seconds'] for r in rows),
                      LIMITS[phase] - sum(r['elapsed_seconds'] for r in rows if r['phase'] in group))
        assert allowed > 0
        command = [sys.executable, '-m', module] + ([] if phase == 'score' else [phase])
        start = time.monotonic()
        with (RUN / (phase + '.log')).open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'phase': phase, 'pid': process.pid, 'log': str(RUN / (phase + '.log')), 'allowed_seconds': allowed}), flush=True)
            violation = None
            while process.poll() is None:
                if time.monotonic() - start > allowed:
                    violation = 'PHASE_OR_GLOBAL_TIME_CAP'
                    os.killpg(process.pid, signal.SIGTERM)
                    break
                time.sleep(1)
            code = process.wait()
        row = {'phase': phase, 'returncode': code, 'pid': process.pid,
               'elapsed_seconds': time.monotonic() - start, 'budget_violation': violation,
               'closed_log_sha256': digest(RUN / (phase + '.log'))}
        with (RUN / 'compute_ledger.jsonl').open('a') as output:
            output.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if code != 0 or violation:
            write_json(RUN / 'failure_closure.json', {'status': 'STOP_NO_AUTOMATIC_RETRY', 'failed_phase': row, 'ledger': ledger()})
            return
    write_json(RUN / 'completion.json', {'status': 'COMPLETE', 'formal_phases': ledger(),
               'charged_seconds': sum(r['elapsed_seconds'] for r in ledger()),
               'all_five_formal_phases_exit0': True, 'extra_encoder_forwards_or_training': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'extract', 'fit', 'close', 'pipeline'])
    phase = parser.parse_args().phase
    globals()[phase]()
