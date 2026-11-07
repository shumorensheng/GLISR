"""Bounded CANOE revisit retrieval with three frozen encoders and no task fitting."""

import argparse
import ast
import hashlib
import io
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
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs.cls_global_readout import read_cls_and_patches
from runs.encoder_midpoint_readout import build_midpoint, CHECKPOINT, REAL, REAL_STATE
from runs.recompute_uatd_area_results import digest
from runs.ssl_global_keep import BASE, disk_bytes
from runs.ssl_real_domain_mix import INITIAL
from sonar_encoder.training.model import build_encoder
from sonar_encoder.training.readout import state_digest, write_json

RUN = BASE / 'canoe_frozen_retrieval_v1'
ASSETS = BASE / 'canoe_protocol_assets_v1'
INPUT = ASSETS / 'bounded_input_v1'
MID = BASE / 'encoder_midpoint_readout_v1'
MID_STATE = '6c4e2561f41beba73206a93122dd676bf46fce7cdfe9152b9974dc6590e2a639'
STATES = {'t0': INITIAL, 'real': REAL_STATE, 'mid': MID_STATE}
TASK_NORMALIZATION = {'mean': .025567574079009666, 'std': .03710258238121667}
LIMITS = {'prepare': 180, 'extract': 900, 'close': 300}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def author_rotation():
    # Reuse the frozen author's four direction functions without importing its unrelated dependencies.
    tree = ast.parse((ASSETS / 'official_utils.py.txt').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('C1', 'C2', 'C3', 'zyx_ang_rad_to_C')]
    assert len(functions) == 4
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'official_utils.py.txt', 'exec'), namespace)
    return namespace['zyx_ang_rad_to_C']


def build_positives(gallery, query):
    rotate = author_rotation()
    rotations = []
    positions = []
    timestamps = []
    for rows in (gallery, query):
        rotations.append(np.stack([rotate(x_rad=float(row['raw_pose_fields']['ang_x']),
                                         y_rad=float(row['raw_pose_fields']['ang_y']),
                                         z_rad=float(row['raw_pose_fields']['ang_z'])) for row in rows]))
        positions.append(np.array([[float(row['raw_pose_fields'][axis]) for axis in ('x', 'y')] for row in rows]))
        timestamps.append(np.array([row['timestamp_microseconds'] for row in rows], dtype=np.int64))
    distance = np.linalg.norm(positions[1][:, None, :] - positions[0][None, :, :], axis=2)
    # trace(R_g.T @ R_q) is the Frobenius inner product of the two rotations.
    trace = np.einsum('qij,gij->qg', rotations[1], rotations[0])
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    gap = np.abs(timestamps[1][:, None] - timestamps[0][None, :])
    return (distance <= 5.) & (angle <= np.deg2rad(15.)) & (gap >= 60_000_000)


def intensity_sample(raw):
    assert raw.shape == (380, 512) and raw.dtype == np.uint8
    values = torch.from_numpy(raw[2:, :].copy()).float().unsqueeze(0) / 255
    return {'intensity': values, 'valid': torch.ones((378, 512), dtype=torch.bool)}


def read_vectors(encoder, samples):
    cls, patches = read_cls_and_patches(encoder, samples, TASK_NORMALIZATION)
    return {'mean': patches.mean(dim=1), 'cls': cls}


def expected_top1(query, gallery, positives):
    q = query.astype(np.float64)
    g = gallery.astype(np.float64)
    qnorm = np.linalg.norm(q, axis=1)
    gnorm = np.linalg.norm(g, axis=1)
    assert np.isfinite(q).all() and np.isfinite(g).all()
    assert (qnorm > 0).all() and (gnorm > 0).all()
    q /= qnorm[:, None]
    g /= gnorm[:, None]
    records = []
    for index, vector in enumerate(q):
        similarities = g @ vector
        maximum = similarities.max()
        ties = similarities == maximum
        count = int(ties.sum())
        hits = int(np.count_nonzero(ties & positives[index]))
        records.append({'pos_count': int(positives[index].sum()), 'top_score': float(maximum),
                        'top_tie_count': count, 'top_positive_count': hits, 'score': hits / count})
    return records


def resources():
    return {'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            'disk_bytes_observed': disk_bytes(RUN)}


def check_resources():
    measured = resources()
    assert measured['max_rss_bytes'] <= 8 * 1024**3
    assert measured['max_gpu_reserved_bytes'] <= 8 * 1024**3
    assert measured['disk_bytes_observed'] <= 64 * 1024**2
    return measured


def input_rows():
    rows = jsonl(INPUT / 'file_manifest.jsonl')
    poses = {row['row_id']: row for row in jsonl(INPUT / 'raw_pose_targets.jsonl')}
    splits = {role: [row for row in rows if row['role'] == role] for role in ('gallery', 'query')}
    ordered = {}
    for role, batch in splits.items():
        ordered[role] = [poses[row['row_id']] for row in batch]
        for row, pose in zip(batch, ordered[role]):
            assert row['timestamp_microseconds'] == pose['timestamp_microseconds'] == int(pose['raw_pose_fields']['utc_micro'])
    assert [len(splits[role]) for role in ('gallery', 'query')] == [1311, 1136]
    return splits, ordered


def prepare():
    original = json.loads((INPUT / 'protocol.json').read_text())
    receipt = json.loads((INPUT / 'receipt.json').read_text())
    downloaded = json.loads((INPUT / 'download_summary.json').read_text())
    checked = json.loads((INPUT / 'post_completion_check.json').read_text())
    assert downloaded['status'] == 'PASS_DATA_PREPARATION_ONLY' and checked['status'] == 'PASS'
    assert downloaded['state']['successes'] == receipt['png_success_count'] == 2447
    assert digest(INPUT / 'protocol.json') == receipt['protocol_sha256']
    for key in ('file_manifest', 'raw_pose_targets', 'input_decode_audit'):
        assert digest(INPUT / original[key]['path']) == original[key]['sha256'] == receipt[key + '_sha256']
    assert digest(INPUT / 'download_summary.json') == receipt['download_summary_sha256']
    assert digest(INPUT / 'post_completion_check.json') == receipt['post_completion_check_sha256']
    for binding in original['source_bindings']:
        assert digest(Path(binding['path'])) == binding['sha256']
    rows, poses = input_rows()
    positives = build_positives(poses['gallery'], poses['query'])
    counts = positives.sum(axis=1)
    feasibility = json.loads((ASSETS / 'feasibility.json').read_text())['feasibility_counts']
    assert int((counts > 0).sum()) == feasibility['query_with_proposed_positive'] == 1136
    assert [int(counts.min()), float(np.median(counts)), int(counts.max())] == feasibility['positive_candidates_count_quantiles_p0_p50_p100'] == [1, 10, 28]
    np.save(RUN / 'positives.npy', positives)
    ids = {role: [row['row_id'] for row in split] for role, split in rows.items()}
    write_json(RUN / 'row_ids.json', ids)
    parents = {}
    inputs = [Path(__file__), ROOT / 'tests/test_canoe_frozen_retrieval.py',
              RUN / 'preflight_receipt.json', RUN / 'cpu_tests.log', RUN / 'source_review_receipt.json',
              ROOT / 'runs/encoder_midpoint_readout.py', ROOT / 'runs/cls_global_readout.py',
              ROOT / 'runs/recompute_uatd_area_results.py', ROOT / 'runs/ssl_global_keep.py',
              ROOT / 'runs/ssl_real_domain_mix.py', *[ROOT / 'sonar_encoder/training' / name for name in
                  ('model.py', 'frontends.py', 'readout.py', 'data.py', 'train.py', 'artifacts.py')],
              *[INPUT / name for name in ('protocol.json', 'file_manifest.jsonl', 'raw_pose_targets.jsonl',
                  'input_decode_audit.json', 'receipt.json', 'download_summary.json', 'post_completion_check.json')],
              ASSETS / 'feasibility.json', ASSETS / 'official_utils.py.txt', ASSETS / 'official_DATA_REFERENCE.md',
              CHECKPOINT, REAL / 'real/report1000.json']
    inputs.extend(Path(binding['path']) for binding in original['source_bindings'])
    for parent in (REAL, MID):
        verification = json.loads((parent / 'verification.json').read_text())
        completion = json.loads((parent / 'completion.json').read_text())
        assert digest(parent / 'results.json') == verification['results_sha256'] == completion['results_sha256']
        assert digest(parent / 'verification.json') == completion['verification_sha256']
        inputs.extend(parent / name for name in ('protocol.json', 'results.json', 'verification.json', 'completion.json', 'final_closure.json'))
        parents[parent.name] = {'results_sha256': digest(parent / 'results.json'), 'final_closure_sha256': digest(parent / 'final_closure.json')}
    mid = json.loads((MID / 'protocol.json').read_text())
    assert mid['encoder_state_sha256'] == MID_STATE and mid['encoder_state_tensor_count'] == 175
    assert digest(CHECKPOINT) == mid['parents']['real']['checkpoint_sha256']
    assert mid['inputs_sha256'][str(ROOT / 'runs/encoder_midpoint_readout.py')] == digest(ROOT / 'runs/encoder_midpoint_readout.py')
    inputs.extend([Path(mid['config']['weights_path']), *sorted(Path(mid['config']['code_dir']).rglob('*.py'))])
    vector_bytes = 3 * 2 * 2447 * 384 * 4 + 12 * 128
    positive_bytes = 1136 * 1311 + 128
    reserve = 16 * 1024**2
    assert vector_bytes == 22553088 and vector_bytes + positive_bytes + reserve < 64 * 1024**2
    protocol = {'question': 'How does REAL1000 behave on head-free revisit-place retrieval beyond box labels?',
                'scope': 'Exploratory cross-data-source development on two same-site same-day CANOE traversals; not official benchmark, cross-site, acoustic pixel correspondence or universal encoder proof.',
                'input_root': str(INPUT), 'source_population': {'gallery': 1311, 'query': 1136, 'sampling': 'Previously frozen first PNG in each 1-second bucket; no re-selection.'},
                'input_sha256': {str(path): digest(path) for path in inputs}, 'parent_closed_sources': parents,
                'config': mid['config'], 'states': STATES, 'mid_arithmetic': mid['arithmetic'], 'mid_state_tensors': 175,
                'preprocessing': {'raw_shape': [380, 512], 'intensity': 'uint8 raw[2:,:] FP32/255, no bearing header/time/pose/file identity/range/FOV channel.',
                    'native_shape': [378, 512], 'task_normalization': TASK_NORMALIZATION,
                    'steps': 'Once log1p -> fixed UATD6051 tasknorm -> bilinearAA224 align_cornersFalse/gain1; complete unmasked eval FP32 TF32off originalPE; valid rectangular processing support.'},
                'features': {'batch': 16, 'sample_forwards': 7341, 'batch_forwards': 459,
                    'interfaces': 'Same forward finalLN CLS384 and GPU FP32 patch_tokens.mean(dim1); twelve FP32 vector NPY only; no patch grids.'},
                'GT': {'horizontal_ENU_ref_m_max': 5., 'full_rotation_geodesic_deg_max': 15., 'absolute_timestamp_gap_s_min': 60.,
                    'rotation': 'Author C1(ang_x) @ C2(ang_y) @ C3(ang_z), radians, sensor->ENU_ref; no second extrinsic or per-sequence zero.',
                    'trace_clamp': 'acos clip[-1,1] only for floating trace range.', 'covered_queries': 1136,
                    'positive_count_min_median_max': [1, 10, 28], 'total_positive_pairs': int(counts.sum()),
                    'positive_sha256': digest(RUN / 'positives.npy'), 'row_ids_sha256': digest(RUN / 'row_ids.json'),
                    'limits': 'Operational pose label; spatial-but-orientation-ineligible pairs are not positives, not verified semantic negatives.'},
                'scoring': {'primary': 'mean expected R@1', 'auxiliary': 'CLS expected R@1',
                    'definition': 'FP32 vectors -> CPU FP64 -> each own L2 normalization -> cosine over all1311gallery; exact maximum ties only, score=top_positive_count/top_tie_count; average all1136queries.',
                    'chance': float(np.mean(counts.astype(np.float64) / 1311)),
                    'undefined': 'Stop on zero or nonfinite vector; no filtering or denominator change.',
                    'verification': 'Independent NumPy per-query implementation from saved vectors and original poses; GT/counts/IDs exact, scores absolute1e-12; report cosine maxabs for different reduction.'},
                'runtime': {'cpu_threads': 4, 'gpu': 0, 'FP32': True, 'TF32': False, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']},
                'budgets': {'phase_seconds': LIMITS, 'total_seconds': 1380, 'rss_bytes': 8 * 1024**3, 'gpu_reserved_bytes': 8 * 1024**3, 'new_disk_peak_bytes': 64 * 1024**2},
                'static_disk': {'twelve_vectors_with_headers_bytes': vector_bytes, 'positive_matrix_with_header_bytes': positive_bytes,
                    'perquery_rows_logs_temporary_reserve_bytes': reserve, 'conservative_peak_upper_bound_bytes': vector_bytes + positive_bytes + reserve},
                'limits': ['One route/site/day and two traversals; no official benchmark or cross-site/acquisition-independent claim.',
                    'No heads/SSL/optimizers/epoch selection, CANOE statistics fitting/whitening/PCA, pose gallery prefilter, alpha/seed/threshold search or ensemble.',
                    'No downloads/image copies/new checkpoints/full similarity cache/other datasets; old NO_GO/C1/C2 remain unchanged.'],
                'stop': 'Six fixed scores, independent NumPy verification and source/resource closure, then report to Astra and stop.'}
    write_json(RUN / 'protocol.json', protocol)
    (RUN / 'code_snapshot').mkdir()
    for path in (Path(__file__), ROOT / 'tests/test_canoe_frozen_retrieval.py'):
        shutil.copyfile(path, RUN / 'code_snapshot' / path.name)
    print(json.dumps({'status': 'FROZEN_BEFORE_ENCODER_FORWARD_AND_SCORING', 'protocol_sha256': digest(RUN / 'protocol.json'),
                      'GT': protocol['GT'], 'chance': protocol['scoring']['chance'], 'static_disk': protocol['static_disk']}), flush=True)


@torch.no_grad()
def extract():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    rows, _ = input_rows()
    ids = json.loads((RUN / 'row_ids.json').read_text())
    reports = {}
    samples_seen = batches_seen = 0
    for state, expected in STATES.items():
        config = protocol['config']
        if state == 'mid':
            encoder, actual, count = build_midpoint(config)
            assert count == 175
        else:
            torch.manual_seed(0)
            encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
            assert state_digest(encoder.state_dict()) == INITIAL
            if state == 'real':
                saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
                assert saved['stream']['step'] == 1000 and saved['initial_sha256'] == INITIAL
                assert saved['protocol_sha256'] == digest(REAL / 'protocol.json')
                encoder.load_state_dict(saved['student'])
                del saved
            actual = state_digest(encoder.state_dict())
        assert actual == expected == protocol['states'][state]
        encoder.to('cuda:0').eval().requires_grad_(False)
        folder = RUN / state
        folder.mkdir()
        hashes = {}
        for role in ('gallery', 'query'):
            split = rows[role]
            assert [row['row_id'] for row in split] == ids[role]
            arrays = {name: np.lib.format.open_memmap(folder / f'{name}_{role}.npy', mode='w+', dtype=np.float32,
                                                     shape=(len(split), 384)) for name in ('mean', 'cls')}
            for start in range(0, len(split), 16):
                batch = split[start:start + 16]
                samples = []
                for row in batch:
                    data = (INPUT / row['relative_png_path']).read_bytes()
                    assert len(data) == row['expected_bytes']
                    assert hashlib.md5(data).hexdigest() == row['expected_singlepart_etag_md5']
                    with Image.open(io.BytesIO(data)) as image:
                        assert image.mode == 'L'
                        raw = np.array(image)
                    samples.append(intensity_sample(raw))
                vectors = read_vectors(encoder, samples)
                for name, vector in vectors.items():
                    assert vector.dtype == torch.float32 and vector.shape == (len(batch), 384)
                    assert torch.isfinite(vector).all()
                    arrays[name][start:start + len(batch)] = vector.cpu().numpy()
                samples_seen += len(batch)
                batches_seen += 1
                check_resources()
                if start % 800 == 0:
                    print(json.dumps({'state': state, 'role': role, 'rows_done': start + len(batch), 'sample_forwards': samples_seen}), flush=True)
            for name, array in arrays.items():
                array.flush()
                path = folder / f'{name}_{role}.npy'
                hashes[str(path)] = digest(path)
            del arrays, vectors
        after = state_digest(encoder.state_dict())
        assert after == actual
        reports[state] = {'state_sha256_before': actual, 'state_sha256_after': after, 'row_ids': ids,
                         'sample_forwards': 2447, 'batch_forwards': 153, 'actual_PNG_size_MD5_decode_checks': 2447,
                         'cache_sha256': hashes, 'resources': check_resources()}
        write_json(folder / 'cache_manifest.json', reports[state])
        del encoder
    assert samples_seen == 7341 and batches_seen == 459
    write_json(RUN / 'extraction.json', {'protocol_sha256': digest(RUN / 'protocol.json'), 'states': reports,
               'sample_forwards': samples_seen, 'batch_forwards': batches_seen, 'resources': check_resources()})


def close():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    rows, poses = input_rows()
    ids = json.loads((RUN / 'row_ids.json').read_text())
    assert ids == {role: [row['row_id'] for row in split] for role, split in rows.items()}
    positives = np.load(RUN / 'positives.npy')
    assert digest(RUN / 'positives.npy') == protocol['GT']['positive_sha256']
    assert digest(RUN / 'row_ids.json') == protocol['GT']['row_ids_sha256']
    # Independent GT: literal author matrices, per-query R_g.T @ R_q traces.
    rotations = {}
    xy = {}
    stamps = {}
    for role in ('gallery', 'query'):
        matrices = []
        for row in poses[role]:
            p = row['raw_pose_fields']
            x, y, z = [float(p['ang_' + axis]) for axis in 'xyz']
            cx, cy, cz = np.cos([x, y, z])
            sx, sy, sz = np.sin([x, y, z])
            matrices.append(np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]]) @
                            np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]]) @
                            np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]]))
        rotations[role] = np.array(matrices)
        xy[role] = np.array([[float(r['raw_pose_fields'][axis]) for axis in ('x', 'y')] for r in poses[role]])
        stamps[role] = np.array([r['timestamp_microseconds'] for r in poses[role]], dtype=np.int64)
    independent_GT = np.empty_like(positives)
    for index in range(1136):
        distances = np.sqrt(np.sum((xy['gallery'] - xy['query'][index])**2, axis=1))
        relative = rotations['gallery'].transpose(0, 2, 1) @ rotations['query'][index]
        angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
        independent_GT[index] = (distances <= 5.) & (angle <= np.pi / 12) & (np.abs(stamps['gallery'] - stamps['query'][index]) >= 60_000_000)
    assert np.array_equal(independent_GT, positives)
    extraction = json.loads((RUN / 'extraction.json').read_text())
    assert extraction['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert extraction['sample_forwards'] == 7341 and extraction['batch_forwards'] == 459
    outputs = {str(RUN / name): digest(RUN / name) for name in ('positives.npy', 'row_ids.json', 'extraction.json')}
    scores = {}
    checks = {}
    for state in STATES:
        report = extraction['states'][state]
        assert report['state_sha256_before'] == report['state_sha256_after'] == STATES[state]
        assert report['row_ids'] == ids
        outputs[str(RUN / state / 'cache_manifest.json')] = digest(RUN / state / 'cache_manifest.json')
        scores[state] = {}
        for interface in ('mean', 'cls'):
            arrays = {}
            for role in ('gallery', 'query'):
                path = RUN / state / f'{interface}_{role}.npy'
                assert digest(path) == report['cache_sha256'][str(path)]
                arrays[role] = np.load(path)
                assert arrays[role].shape == (len(ids[role]), 384) and arrays[role].dtype == np.float32
                outputs[str(path)] = digest(path)
            record = expected_top1(arrays['query'], arrays['gallery'], positives)
            for row, row_id in zip(record, ids['query']):
                row['query_row_id'] = row_id
            path = RUN / state / f'{interface}_perquery.jsonl'
            path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in record))
            outputs[str(path)] = digest(path)
            # Read saved records back; independently normalize with explicit sums and einsum cosine.
            saved_records = jsonl(path)
            primary_q = arrays['query'].astype(np.float64)
            primary_g = arrays['gallery'].astype(np.float64)
            primary_q /= np.linalg.norm(primary_q, axis=1)[:, None]
            primary_g /= np.linalg.norm(primary_g, axis=1)[:, None]
            q = arrays['query'].astype(np.float64)
            g = arrays['gallery'].astype(np.float64)
            q /= np.sqrt(np.sum(q * q, axis=1))[:, None]
            g /= np.sqrt(np.sum(g * g, axis=1))[:, None]
            del arrays
            independent_scores = []
            max_cosine_error = 0.
            max_score_error = 0.
            for index, saved in enumerate(saved_records):
                assert saved['query_row_id'] == ids['query'][index]
                similarities = np.einsum('ij,j->i', g, q[index])
                primary_similarities = primary_g @ primary_q[index]
                max_cosine_error = max(max_cosine_error, float(np.max(np.abs(similarities - primary_similarities))))
                top = similarities.max()
                selected = similarities == top
                tie_count = int(selected.sum())
                positive_count = int(np.count_nonzero(selected & independent_GT[index]))
                score = positive_count / tie_count
                assert saved['pos_count'] == int(independent_GT[index].sum())
                assert saved['top_tie_count'] == tie_count and saved['top_positive_count'] == positive_count
                assert abs(saved['top_score'] - float(top)) <= 1e-12
                max_score_error = max(max_score_error, abs(saved['score'] - score))
                independent_scores.append(score)
            aggregate = float(np.mean([row['score'] for row in saved_records]))
            independent = float(np.sum(independent_scores, dtype=np.float64) / 1136)
            assert max_score_error <= 1e-12 and abs(aggregate - independent) <= 1e-12
            scores[state][interface] = {'expected_R_at_1': aggregate, 'query_count': 1136,
                                        'perquery': str(path), 'top_tie_count_max': max(r['top_tie_count'] for r in record)}
            checks[state + '_' + interface] = {'score_abs_error': abs(aggregate - independent),
                'perquery_score_max_abs_error': max_score_error, 'count_and_ID_exact': True,
                'cosine_reduction_max_abs_difference': max_cosine_error, 'independent_expected_R_at_1': independent}
    for path, expected in protocol['input_sha256'].items():
        assert digest(Path(path)) == expected, path
    differences = {label: {interface: scores[a][interface]['expected_R_at_1'] - scores[b][interface]['expected_R_at_1']
                          for interface in ('mean', 'cls')} for label, a, b in
                   (('REAL_minus_T0', 'real', 't0'), ('MID_minus_T0', 'mid', 't0'), ('MID_minus_REAL', 'mid', 'real'))}
    write_json(RUN / 'results.json', {'scope': protocol['scope'], 'states': scores, 'differences_within_interface': differences,
               'primary': 'mean', 'fixed_auxiliary': 'cls', 'analytic_chance': protocol['scoring']['chance'],
               'GT': protocol['GT'], 'sample_forwards': 7341, 'batch_forwards': 459, 'new_heads_or_updates': 0, 'limits': protocol['limits']})
    write_json(RUN / 'verification.json', {'status': 'PASS_SIX_INDEPENDENT_NUMPY_SCORES_GT_COUNTS_IDS_AND_BINDINGS',
               'checks': checks, 'independent_GT_exact': True, 'results_sha256': digest(RUN / 'results.json'),
               'protocol_sha256': digest(RUN / 'protocol.json'), 'closed_input_sha256': protocol['input_sha256'], 'closed_output_sha256': outputs})
    print(json.dumps({'status': 'SIX_SCORES_CLOSED', 'states': scores, 'differences': differences,
                      'chance': protocol['scoring']['chance'], 'resources': check_resources()}), flush=True)


def timeout_phase(signum, frame):
    raise TimeoutError('Frozen CANOE retrieval compute budget exhausted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=tuple(LIMITS))
    args = parser.parse_args()
    ledger_path = RUN / 'compute_ledger.jsonl'
    ledger = jsonl(ledger_path)
    prior = sum(row['seconds'] for row in ledger)
    phase_prior = sum(row['seconds'] for row in ledger if row['phase'] == args.phase)
    available = min(LIMITS[args.phase] - phase_prior, 1380 - prior)
    signal.signal(signal.SIGALRM, timeout_phase)
    signal.setitimer(signal.ITIMER_REAL, available - (time.monotonic() - PROCESS_STARTED))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(json.dumps({'phase': args.phase, 'pid': os.getpid(), 'prior_compute_seconds': prior, 'available_seconds': available}), flush=True)
    status = 'FAILED'
    try:
        {'prepare': prepare, 'extract': extract, 'close': close}[args.phase]()
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
        assert elapsed + phase_prior <= LIMITS[args.phase] and elapsed + prior <= 1380
        check_resources()
    if args.phase == 'close':
        write_json(RUN / 'completion.json', {'status': 'COMPLETE_SIX_FROZEN_CANOE_RETRIEVAL_CELLS',
                   'protocol_sha256': digest(RUN / 'protocol.json'), 'results_sha256': digest(RUN / 'results.json'),
                   'verification_sha256': digest(RUN / 'verification.json'), 'compute_ledger_sha256': digest(ledger_path),
                   'total_compute_seconds': prior + elapsed, 'resources': measured})


if __name__ == '__main__':
    main()
