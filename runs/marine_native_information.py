"""One frozen Marine comparison of native versus interpolated tiled input."""
import argparse
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from sonar_encoder.training.model import build_encoder, prepare_native
from sonar_encoder.training.readout import head_logits, load_targets, state_digest, write_json
from runs.marine_frozen_transfer import evaluate

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'marine_native_information_v1'
PARENT = BASE / 'marine_frozen_transfer_v1'
AUTH = ROOT / '.aris/traces/experiment-plan/2026-10-08_marine_native_information/contract.json'
AUTH_SHA = 'a72e4a7aa4993f0e7794e040042fe4e069c22bdd001dd10e4b29ba83697018de'
INITIAL = 'b9fc4072daeecd3250c87c4f4b6994248f5a84526d3e62effa5bb050dbbf973e'
STATES = ('T0_224', 'interp_tiles', 'native_tiles')
LIMITS = {'prepare': 600, 'extract': 900, 'head_T0_224': 900, 'head_interp_tiles': 900,
          'head_native_tiles': 900, 'score_close': 600}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): result.update(block)
    return result.hexdigest()


def runtime():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('highest')


def ledger():
    return [json.loads(x) for x in (RUN / 'compute_ledger.jsonl').read_text().splitlines()]


def remaining(group):
    rows = ledger()
    return min(7200 - sum(r['seconds'] for r in rows), LIMITS[group] - sum(r['seconds'] for r in rows if r['group'] == group))


def resources():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = torch.cuda.max_memory_reserved(0) if torch.cuda.is_initialized() else 0
    disk = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
    assert rss <= 16 * 1024**3 and gpu <= 12 * 1024**3 and disk <= 8 * 1024**3, (rss, gpu, disk)
    return {'max_rss_bytes': rss, 'max_gpu_reserved_bytes': gpu, 'new_disk_bytes': disk,
            'readonly_parent_cache_links_excluded': True}


def make_paths(sample, normalization):
    native = prepare_native(sample, normalization, None)
    c224 = F.interpolate(native[None], size=(224, 224), mode='bilinear', align_corners=False, antialias=True)[0]
    interp = F.interpolate(c224[None], size=(448, 448), mode='bilinear', align_corners=False, antialias=True)[0]
    direct = F.interpolate(native[None], size=(448, 448), mode='bilinear', align_corners=False, antialias=True)[0]
    return interp, direct


def tile_frames(frames):
    return torch.stack([frames[..., r:r + 224, c:c + 224]
                        for r, c in ((0, 0), (0, 224), (224, 0), (224, 224))], dim=1).flatten(0, 1)


def stitch_tokens(tokens):
    count, _, channels = tokens.shape
    tiles = tokens.transpose(1, 2).reshape(count // 4, 4, channels, 16, 16)
    field = torch.empty((count // 4, channels, 32, 32), dtype=tokens.dtype, device=tokens.device)
    for i, (r, c) in enumerate(((0, 0), (0, 16), (16, 0), (16, 16))):
        field[..., r:r + 16, c:c + 16] = tiles[:, i]
    return field


def prepare():
    runtime()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0' and os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert digest(AUTH) == AUTH_SHA
    auth = json.loads(AUTH.read_text())
    population = auth['population']
    manifest_path, norm_path = Path(population['prepared_manifest']), Path(population['normalization'])
    assert digest(manifest_path) == population['manifest_sha256']
    manifest, norm = json.loads(manifest_path.read_text()), json.loads(norm_path.read_text())
    assert norm['source_manifest_sha256'] == population['manifest_sha256']
    assert (norm['mean'], norm['std']) == (.1453277000149234, .15094323542837218)
    ids = {s: [r['row_id'] for r in manifest[s]] for s in ('train', 'validation')}
    assert [len(ids[s]) for s in ids] == [1000, 251]
    assert all(r['shape'] == [480, 320] for s in ids for r in manifest[s])
    parent = json.loads((PARENT / 'protocol.json').read_text())
    parent_closed = json.loads((PARENT / 'final_closure.json').read_text())
    parent_verified = json.loads((PARENT / 'verification.json').read_text())
    cache_path = PARENT / 't0/cache_manifest.json'
    cache = json.loads(cache_path.read_text())
    assert digest(PARENT / 'protocol.json') == parent_closed['protocol_sha256'] == cache['protocol_sha256']
    assert digest(PARENT / 'verification.json') == parent_closed['verification_sha256']
    assert parent_closed['results_sha256'] == parent_verified['results_sha256']
    assert digest(cache_path) == parent_verified['closed_outputs_sha256'][str(cache_path)]
    assert cache['row_ids'] == parent['row_ids'] == ids
    assert cache['normalization_sha256'] == digest(norm_path) == parent['inputs_sha256'][str(norm_path)]
    assert digest(manifest_path) == parent['inputs_sha256'][str(manifest_path)]
    assert cache['encoder_state_sha256_before'] == cache['encoder_state_sha256_after'] == parent['origins']['t0']['state_sha256'] == INITIAL
    (RUN / 'T0_224').mkdir()
    t0_bindings = {}
    for split in ids:
        source = PARENT / 't0' / (split + '.npy')
        actual = digest(source)
        assert actual == cache['cache_files_sha256'][source.name] == parent_verified['closed_outputs_sha256'][str(source)]
        array = np.load(source, mmap_mode='r', allow_pickle=False)
        assert array.shape == (len(ids[split]), 384, 16, 16) and array.dtype == np.float32
        link = RUN / 'T0_224' / source.name
        link.symlink_to(source)
        t0_bindings[split] = {'path': str(link), 'parent_path': str(source), 'sha256': actual,
                              'shape': list(array.shape), 'dtype': 'float32', 'parent_bytes': source.stat().st_size,
                              'actual_full_parent_cache_hash_reads': 1}
    for state in ('interp_tiles', 'native_tiles'): (RUN / state).mkdir()
    for split in ids:
        for row in manifest[split]: assert digest(row['mask']) == row['maskSHA256']
    torch.manual_seed(0)
    initial_head = state_digest(nn.Conv2d(384, 12, 1, bias=True).state_dict())
    assert initial_head == auth['head_protocol']['expected_initial_sha256']
    generator, order = torch.Generator().manual_seed(0), hashlib.sha256()
    for _ in range(100): order.update(torch.randperm(1000, generator=generator).numpy().tobytes())
    assert order.hexdigest() == auth['head_protocol']['expected_actual_order_sha256']
    paths = [Path(__file__), ROOT / 'tests/test_marine_native_information.py',
             ROOT / 'runs/marine_frozen_transfer.py', ROOT / 'sonar_encoder/training/model.py',
             ROOT / 'sonar_encoder/training/readout.py', ROOT / 'sonar_encoder/training/frontends.py']
    for path, expected in auth['ROOT_read_source_sha256'].items(): assert digest(path) == expected
    code_dir = Path(parent['official_config']['code_dir'])
    official_sources = [code_dir / 'dinov2/hub/backbones.py', code_dir / 'dinov2/models/vision_transformer.py',
                        code_dir / 'dinov2/models/__init__.py', *sorted((code_dir / 'dinov2/layers').glob('*.py'))]
    official_bindings = {str(p): digest(p) for p in official_sources}
    for path, actual in official_bindings.items(): assert actual == parent['inputs_sha256'][path]
    (RUN / 'source').mkdir()
    for path in paths: (RUN / 'source' / path.name).write_bytes(path.read_bytes())
    write_json(RUN / 'protocol.json', {'unit': RUN.name, 'authorization_sha256': AUTH_SHA, 'contract': auth,
               'manifest': str(manifest_path), 'normalization': str(norm_path), 'normalization_sha256': digest(norm_path),
               'manifest_sha256': digest(manifest_path), 'row_ids': ids, 'class_names': manifest['class_names'],
               'official_config': parent['official_config'], 'initial_encoder_sha256': INITIAL,
               'heads': {'initial_sha256': initial_head, 'private_order_sha256': order.hexdigest()},
               'arms': {s: {'grid': 16 if s == 'T0_224' else 32} for s in STATES},
               'T0_cache_reuse': t0_bindings, 'parent_protocol_sha256': parent_closed['protocol_sha256'],
               'parent_final_closure_sha256': digest(PARENT / 'final_closure.json'),
               'parent_cache_manifest_sha256': digest(cache_path), 'parent_verification_sha256': digest(PARENT / 'verification.json'),
               'official_source_sha256': official_bindings,
               'official_weight_sha256_inherited_not_new_archive_hash': parent['inputs_sha256'][parent['official_config']['weights_path']],
               'source_sha256': {str(p): digest(p) for p in paths},
               'runtime': {'python': sys.executable, 'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                           'torch_file': torch.__file__, 'numpy': np.__version__,
                           'torchvision_distribution_version': importlib.metadata.version('torchvision'),
                           'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                           'CPU_threads': torch.get_num_threads(), 'TF32': torch.backends.cuda.matmul.allow_tf32,
                           'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
               'static_new_peak_bytes': 5 * 1024**3, 'hard_new_disk_cap_bytes': 8 * 1024**3,
               'review_sha256': digest(RUN / 'preflight/source_review.json'),
               'parent_weight_archive_cache_close_rehash': False})
    assert sys.executable == sys.executable
    assert not torch.cuda.is_initialized()
    write_json(RUN / 'prepare_receipt.json', {'status': 'ALL_THREE_PATHS_HEADS_SCORES_FROZEN_BEFORE_FIRST_IMAGE_FORWARD',
               'protocol_sha256': digest(RUN / 'protocol.json'), 'native_population': [1000, 251],
               'T0_existing_cache_identity_closed': True, 'new_T0_encoder_forwards': 0,
               'head_initial_sha256': initial_head, 'actual_order_expected_sha256': order.hexdigest(), 'resources': resources()})


def gpu_entry(phase):
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
                if uid != os.getuid(): foreign.append({'pid': int(pid), 'UID': uid, 'executable': executable, 'memory_MiB': memory})
                elif (proc / 'cwd').resolve(strict=True) == ROOT: conflicts.append(int(pid))
            except FileNotFoundError: disappeared.append(int(pid))
    evidence = {'phase': phase, 'query': query, 'process_metadata': apps, 'foreign_UIDs': foreign,
                'project_conflicts': conflicts, 'disappeared': disappeared, 'free_bytes': int(gpu[2]) * 1024**2}
    with (RUN / 'gpu_entries.jsonl').open('a') as f: f.write(json.dumps(evidence) + '\n')
    return evidence['free_bytes'] >= 24 * 1024**3 and not conflicts


@torch.no_grad()
def extract():
    runtime()
    begun = time.monotonic()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    norm = json.loads(Path(protocol['normalization']).read_text())
    config = protocol['official_config']
    torch.manual_seed(0)
    encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
    before = state_digest(encoder.state_dict())
    assert before == INITIAL
    encoder.to('cuda:0').eval().requires_grad_(False)
    write_json(RUN / 'encoder_entry_receipt.json', {'protocol_sha256': digest(RUN / 'protocol.json'),
               'encoder_state_sha256': before, 'eval': not encoder.training,
               'parameter_requires_grad_any': any(v.requires_grad for v in encoder.parameters()),
               'model_dtype': str(next(encoder.parameters()).dtype), 'PE_shape': list(encoder.backbone.pos_embed.shape),
               'unchanged_per_tile224_PE': True, 'resources': resources()})
    row_count, sample_count, batch_count = 0, 0, 0
    cache_metadata = {state: {} for state in ('interp_tiles', 'native_tiles')}
    for split in ('train', 'validation'):
        rows = manifest[split]
        arrays = {state: np.lib.format.open_memmap(RUN / state / (split + '.npy'), mode='w+', dtype=np.float32,
                                                  shape=(len(rows), 384, 32, 32)) for state in cache_metadata}
        for start in range(0, len(rows), 4):
            batch_start = time.monotonic()
            rowbatch = rows[start:start + 4]
            paths = {'interp_tiles': [], 'native_tiles': []}
            for row in rowbatch:
                raw = Path(row['image']).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == row['imageSHA256']
                with Image.open(io.BytesIO(raw)) as image: gray = np.array(image.convert('L'), dtype=np.uint8)
                assert gray.shape == (480, 320)
                sample = {'intensity': torch.from_numpy(gray).float().unsqueeze(0) / 255,
                          'valid': torch.ones((480, 320), dtype=torch.bool)}
                interp, native = make_paths(sample, norm)
                paths['interp_tiles'].append(interp)
                paths['native_tiles'].append(native)
            for state in cache_metadata:
                tiles = tile_frames(torch.stack(paths[state])).to('cuda:0')
                tokens = encoder(tiles)['patch_tokens']
                field = stitch_tokens(tokens)
                assert field.shape == (len(rowbatch), 384, 32, 32) and field.dtype == torch.float32 and torch.isfinite(field).all()
                arrays[state][start:start + len(rowbatch)] = field.cpu().numpy()
                sample_count += len(tiles)
                batch_count += 1
                resources()
            row_count += len(rowbatch)
            for array in arrays.values(): array.flush()
            if row_count == 4:
                prefix_seconds = time.monotonic() - batch_start
                projected = 1.5 * prefix_seconds * (313 - 1)
                reserve = 2700 + 600
                assert projected <= remaining('extract') - (time.monotonic() - begun)
                assert projected + reserve <= 7200 - sum(r['seconds'] for r in ledger()) - (time.monotonic() - begun)
                write_json(RUN / 'prefix_gate.json', {'status': 'FIRST_NORMAL4_IMAGES_TWO16_TILE_BATCHES_PASS_AUTOMATIC_CONTINUATION',
                           'native_rows_counted_in_total': 4, 'sample_forwards_counted_in_total': sample_count,
                           'encoder_calls_counted_in_total': batch_count, 'batch_wall_seconds_including_decode_bothpaths_andwrite': prefix_seconds,
                           'remaining_paired_row_batches': 312, 'projection_factor': 1.5,
                           'conditional_remaining_encode_seconds': projected, 'fixed_head_score_reserve_seconds': reserve,
                           'resources': resources()})
                print('NORMAL_PREFIX_PASS_AUTOMATIC_CONTINUATION', flush=True)
            if start % 100 == 0: print('TILED_ROWS', split, row_count, 'samples', sample_count, 'calls', batch_count, flush=True)
        for state, array in arrays.items():
            array.flush()
            path = RUN / state / (split + '.npy')
            cache_metadata[state][split] = {'path': str(path), 'sha256': digest(path),
                                             'shape': list(array.shape), 'row_ids': protocol['row_ids'][split]}
    assert row_count == 1251 and sample_count == 10008 and batch_count == 626
    after = state_digest(encoder.state_dict())
    assert before == after == INITIAL
    write_json(RUN / 'extraction.json', {'native_rows_decoded': row_count, 'sample_forwards': sample_count,
               'encoder_calls': batch_count, 'T0_extra_forwards': 0, 'encoder_state_sha256_before': before,
               'encoder_state_sha256_after': after, 'caches': cache_metadata, 'resources': resources()})


def atomic_predictions(path, predictions):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: np.save(f, predictions, allow_pickle=False)
    resources()
    temp.replace(path)


def train_head(state):
    runtime()
    begun = time.monotonic()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    grid = protocol['arms'][state]['grid']
    features, targets = {}, {}
    for split in ('train', 'validation'):
        path = RUN / state / (split + '.npy')
        features[split] = torch.from_numpy(np.load(path, allow_pickle=False)).to('cuda:0')
        assert features[split].shape == (len(manifest[split]), 384, grid, grid)
        native = load_targets(manifest[split], 'marine')
        assert native.dtype == torch.uint8 and native.shape == (len(manifest[split]), 480, 320) and native.max() <= 11
        targets[split] = native.to('cuda:0')
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
                field, gt = features['train'][indices], targets['train'][indices]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head_logits(head, field, gt, 'marine'), gt.long())
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in head.parameters())
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                updates += 1
                resources()
            metrics, predictions = evaluate(head, features['validation'], targets['validation'], None)
            entry = {'epoch': epoch, 'updates': updates, 'train_loss': total_loss / 1000, 'validation': metrics}
            log.write(json.dumps(entry, allow_nan=False) + '\n'); log.flush()
            if metrics['miou12'] > best_score:
                best_score, best_epoch, best_metrics = metrics['miou12'], epoch, metrics
                temp = folder / 'best.tmp'
                torch.save({'head': head.state_dict(), 'epoch': epoch, 'metrics': metrics, 'state': state,
                            'protocol_sha256': digest(RUN / 'protocol.json'), 'row_ids': protocol['row_ids']['validation']}, temp)
                resources(); temp.replace(folder / 'best.pt')
                atomic_predictions(folder / 'best_validation_predictions.npy', predictions)
            if epoch == 100:
                terminal_path = folder / ('best_validation_predictions.npy' if best_epoch == 100 else 'terminal_validation_predictions.npy')
                if best_epoch != 100: atomic_predictions(terminal_path, predictions)
            resources()
            if epoch % 10 == 0: print('HEAD_PROGRESS', state, epoch, updates, flush=True)
    assert updates == 6300 and order_hash.hexdigest() == protocol['heads']['private_order_sha256']
    write_json(folder / 'report.json', {'state': state, 'epochs': 100, 'updates': updates,
               'best_epoch': best_epoch, 'best': best_metrics, 'terminal': metrics,
               'initial_head_sha256': initial, 'actual_private_order_sha256': order_hash.hexdigest(),
               'optimizer_only_head': True, 'encoder_loaded_in_head_phase': False, 'feature_grid': grid,
               'best_head_sha256': digest(folder / 'best.pt'),
               'best_predictions_sha256': digest(folder / 'best_validation_predictions.npy'),
               'terminal_prediction_path': str(terminal_path), 'terminal_predictions_sha256': digest(terminal_path),
               'unique_predictions': 1 if best_epoch == 100 else 2,
               'terminal_weight_saved': best_epoch == 100, 'wall_seconds': time.monotonic() - begun, 'resources': resources()})


def independent_metrics(gt, prediction):
    matrix = np.zeros((12, 12), dtype=np.int64)
    for i in range(251):
        matrix += np.bincount(gt[i].astype(np.int64).ravel() * 12 + prediction[i].ravel(), minlength=144).reshape(12, 12)
    tp, support, guessed = matrix.diagonal().astype(np.float64), matrix.sum(1), matrix.sum(0)
    iou = tp / np.maximum(support + guessed - tp, 1)
    return {'confusion_matrix': matrix.tolist(), 'support': support.tolist(), 'per_class_iou': iou.tolist(),
            'miou12': float(iou.mean()), 'debris_miou10': float(iou[1:11].mean())}


def close():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    for row in manifest['validation']: assert digest(row['mask']) == row['maskSHA256']
    gt = load_targets(manifest['validation'], 'marine').numpy()
    states, checks = {}, []
    for state in STATES:
        folder = RUN / ('head_' + state)
        report = json.loads((folder / 'report.json').read_text())
        saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        assert saved['row_ids'] == protocol['row_ids']['validation'] and saved['protocol_sha256'] == digest(RUN / 'protocol.json')
        history = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert len(history) == 100 and all(v['epoch'] == i and v['updates'] == i * 63 for i, v in enumerate(history, 1))
        selected = max(history, key=lambda x: x['validation']['miou12'])
        assert selected['epoch'] == saved['epoch'] == report['best_epoch']
        assert selected['validation'] == saved['metrics'] == report['best'] and history[-1]['validation'] == report['terminal']
        assert report['initial_head_sha256'] == protocol['heads']['initial_sha256']
        assert report['actual_private_order_sha256'] == protocol['heads']['private_order_sha256']
        assert digest(folder / 'best.pt') == report['best_head_sha256']
        states[state] = {'best_epoch': report['best_epoch'], 'terminal_epoch': 100}
        for when, path, expected_sha in [('best', folder / 'best_validation_predictions.npy', report['best_predictions_sha256']),
                                         ('terminal', Path(report['terminal_prediction_path']), report['terminal_predictions_sha256'])]:
            assert digest(path) == expected_sha
            pred = np.load(path, mmap_mode='r', allow_pickle=False)
            assert pred.dtype == np.uint8 and pred.shape == (251, 480, 320) and pred.max() <= 11
            actual = independent_metrics(gt, pred)
            expected = report[when]
            assert np.array_equal(actual['confusion_matrix'], expected['confusion_matrix'])
            assert np.array_equal(actual['support'], expected['support'])
            error = max(float(np.abs(np.asarray(actual[k]) - np.asarray(expected[k])).max()) for k in ('miou12', 'debris_miou10', 'per_class_iou'))
            assert error <= 1e-12
            states[state][when] = actual
            checks.append({'state': state, 'when': when, 'prediction_path': str(path), 'prediction_sha256': expected_sha,
                           'status': 'PASS_INDEPENDENT_NUMPY_NATIVE_CONFUSION_SUPPORT_IOU', 'counts_support_exact': True, 'maxabs': error})
        resources()
    for path, expected in protocol['source_sha256'].items(): assert digest(path) == expected
    differences = {when: {a + '-'+b: {k: states[a][when][k] - states[b][when][k] for k in ('miou12', 'debris_miou10')}
                          for a, b in [('native_tiles', 'interp_tiles'), ('native_tiles', 'T0_224'), ('interp_tiles', 'T0_224')]}
                   for when in ('terminal', 'best')}
    write_json(RUN / 'results.json', {'status': 'COMPLETE_FROZEN_MARINE_INPUT_ACCESS_CONTROL', 'states': states, 'differences': differences,
               'primary': 'native_tiles-interp_tiles terminal100 mIoU12; all3absolute andT0-relative values shown',
               'auxiliary': 'strictbest/debris10/perclass/support', 'oldT0scores_historical_only_not_newbaseline': True,
               'new_T0_head_fit': True, 'new_encoder_sample_forwards': 10008, 'new_encoder_calls': 626,
               'native_rows_decoded': 1251, 'new_heads': 3, 'new_head_updates': 18900,
               'GT_diskcopy': False, 'outer617_accessed': False, 'scope': 'Repeateddevelopment image split; finitefrozenpath/task utility only, noBayesloss/causal/novelmethod claim.'})
    write_json(RUN / 'verification.json', {'status': 'PASS_SIX_LOGICAL_NATIVE_SAVED_PREDICTION_SCORES', 'checks': checks,
               'unique_prediction_artifacts': len({c['prediction_path'] for c in checks}),
               'results_sha256': digest(RUN / 'results.json'), 'protocol_sha256': digest(RUN / 'protocol.json'),
               'parent_cache_byte_hashes_once_prepare_only': True, 'resources': resources()})


def pipeline():
    for phase in ('prepare', 'extract', 'head_T0_224', 'head_interp_tiles', 'head_native_tiles', 'score_close'):
        if phase == 'extract' or phase.startswith('head_'):
            started = time.monotonic(); admitted = gpu_entry(phase)
            with (RUN / 'compute_ledger.jsonl').open('a') as f: f.write(json.dumps({'phase': 'GPU_entry', 'group': phase, 'seconds': time.monotonic() - started}) + '\n')
            if not admitted:
                write_json(RUN / 'deferred.json', {'status': 'DEFERRED_GPU_ENTRY_NO_QUEUE', 'phase': phase}); return
        allowed = remaining(phase)
        assert allowed > 0
        started = time.monotonic(); violation = None
        log = RUN / (phase + '.log')
        with log.open('w') as output:
            child = subprocess.Popen([sys.executable, '-m', 'runs.marine_native_information', phase], stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'phase': phase, 'pid': child.pid, 'log': str(log), 'seconds_available': allowed}), flush=True)
            while child.poll() is None:
                if time.monotonic() - started > allowed:
                    violation = 'TIME_CAP'; os.killpg(child.pid, signal.SIGTERM); break
                time.sleep(1)
            code = child.wait()
        row = {'phase': phase, 'group': phase, 'pid': child.pid, 'seconds': time.monotonic() - started,
               'exit_code': code, 'budget_violation': violation, 'closed_log_sha256': digest(log)}
        with (RUN / 'compute_ledger.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        if code != 0 or violation:
            write_json(RUN / 'failure_closure.json', {'status': 'STOP_NO_AUTOMATIC_RETRY', 'failed_phase': row, 'ledger': ledger()}); return
    write_json(RUN / 'completion.json', {'status': 'COMPLETE_STOP_NO_NEW_PHASE', 'charged_seconds': sum(r['seconds'] for r in ledger()),
               'new_encoder_samples': 10008, 'new_encoder_calls': 626, 'new_head_updates': 18900,
               'results_sha256': digest(RUN / 'results.json'), 'verification_sha256': digest(RUN / 'verification.json'),
               'formal_phase_exits': [r for r in ledger() if 'exit_code' in r], 'resources': resources()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'extract', 'head_T0_224', 'head_interp_tiles', 'head_native_tiles', 'score_close', 'pipeline'))
    phase = parser.parse_args().phase
    if phase.startswith('head_'): train_head(phase.removeprefix('head_'))
    elif phase == 'score_close': close()
    else: {'prepare': prepare, 'extract': extract, 'pipeline': pipeline}[phase]()
