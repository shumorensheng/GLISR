"""Fixed same-224-observation tiled FSOD detection readout."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
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
from torchvision.ops import roi_align

from runs.marine_native_information import digest, runtime, stitch_tokens, tile_frames
from runs.shared_supervised_readout import (
    CATEGORIES, COUNTS, independent_ap50, make_fsod_head, prediction_list, prediction_rows,
)
from runs.shared_supervised_json_readout import canonical_evaluate
from runs.cfc_independent_ap50 import read_predictions
from runs.cfc_frozen_detector import scale_boxes
from sonar_encoder.training.model import build_encoder, prepare_images
from sonar_encoder.training.readout import state_digest, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'interp_tiles_fsod_transfer_v1'
AUTH = ROOT / '.aris/traces/experiment-plan/2026-10-08_interp_tiles_fsod_transfer/contract.json'
AUTH_SHA = 'b078fb2583538de248acb1e880030de03e88331e2ace8a3736f12eb8235fb249'
INITIAL = 'b9fc4072daeecd3250c87c4f4b6994248f5a84526d3e62effa5bb050dbbf973e'
LIMITS = {'prepare': 600, 'extract': 900, 'head': 4500, 'score_close': 600}
PEAK_DISK = 0


class TiledRoIPool(nn.Module):
    def forward(self, features, boxes, image_shapes):
        return roi_align(features['0'], boxes, output_size=(7, 7), spatial_scale=1 / 7,
                         sampling_ratio=2, aligned=True)


def resources():
    global PEAK_DISK
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = torch.cuda.max_memory_reserved(0) if torch.cuda.is_initialized() else 0
    disk = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
    PEAK_DISK = max(PEAK_DISK, disk)
    assert rss <= 16 * 1024**3 and gpu <= 12 * 1024**3 and disk <= 8 * 1024**3, (rss, gpu, disk)
    return {'max_RSS_bytes': rss, 'max_GPU_reserved_bytes': gpu, 'disk_bytes': disk,
            'maximum_observed_disk_bytes_in_process': PEAK_DISK}


def ledger():
    return [json.loads(x) for x in (RUN / 'compute_ledger.jsonl').read_text().splitlines()]


def remaining(group):
    rows = ledger()
    return min(7200 - sum(v['seconds'] for v in rows), LIMITS[group] - sum(v['seconds'] for v in rows if v['group'] == group))


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
    record = {'phase': phase, 'query': query, 'nvidia_process_metadata': apps, 'foreign_UIDs': foreign,
              'same_project_conflicts': conflicts, 'disappeared': disappeared, 'free_bytes': int(gpu[2]) * 1024**2}
    with (RUN / 'gpu_entries.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
    return record['free_bytes'] >= 24 * 1024**3 and not conflicts


def prepare():
    runtime()
    assert sys.executable == sys.executable
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0' and os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert digest(AUTH) == AUTH_SHA
    auth = json.loads(AUTH.read_text())
    parent_path = Path(auth['data']['source_prepare_run']) / 'protocol.json'
    parent_bytes = parent_path.read_bytes()
    assert hashlib.sha256(parent_bytes).hexdigest() == auth['data']['reference_protocol_sha256']
    parent = json.loads(parent_bytes)
    target = parent['targets']['fsod']
    rows = target['rows']
    assert [len(rows[s]) for s in ('train', 'validation')] == [2357, 785]
    assert all([v['width'], v['height']] == [2008, 1112] for s in rows for v in rows[s])
    assert (target['normalization']['mean'], target['normalization']['std'], target['normalization']['pixel_count']) == (.24604009070235597, .09830033220343315, 5262935872)
    actual_counts = {}
    coco_bindings = {}
    for split, key in [('train', 'training_coco'), ('validation', 'validation_coco')]:
        path = Path(target[key]); blob = path.read_bytes()
        coco_bindings[str(path)] = hashlib.sha256(blob).hexdigest()
        assert coco_bindings[str(path)] == parent['inputs_sha256'][str(path)]
        coco = json.loads(blob)
        assert [v['id'] for v in coco['categories']] == CATEGORIES
        assert [v['id'] for v in coco['images']] == [v['image_id'] for v in rows[split]]
        by_image = {v['image_id']: [] for v in rows[split]}
        for annotation in coco['annotations']: by_image[annotation['image_id']].append(annotation)
        assert all(by_image[v['image_id']] == v['annotations'] for v in rows[split])
        actual_counts[split] = {'empty': sum(len(v['annotations']) == 0 for v in rows[split]),
                                'multi': sum(len(v['annotations']) > 1 for v in rows[split])}
    assert actual_counts == {'train': {'empty': 92, 'multi': 97}, 'validation': {'empty': 44, 'multi': 29}}
    reference = Path(auth['data']['reference_run'])
    assert digest(reference / 'final_closure.json') == auth['data']['reference_final_closure_sha256']
    closed = json.loads((reference / 'final_closure.json').read_text())
    assert closed['bindings_sha256'][str(reference / 'results.json')] == auth['data']['reference_results_sha256']
    target_results_path = reference / 'target_results.json'
    assert digest(target_results_path) == closed['bindings_sha256'][str(target_results_path)]
    ref = json.loads(target_results_path.read_text())['states']['T0']['fsod']
    assert ref['terminal']['AP50'] == ref['best']['AP50'] == .8222198114343945 and ref['best_epoch'] == 60
    receipt_path = reference / 'readouts/T0/fsod/score_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['result'] == ref and receipt['status'] == 'PASS_INDEPENDENT_NUMPY_COUNTS_AND_METRICS'
    report_path, extraction_path = receipt_path.parent / 'report.json', receipt_path.parent / 'extraction.json'
    assert digest(report_path) == receipt['report_sha256'] and digest(extraction_path) == receipt['extraction_sha256']
    prior = json.loads(extraction_path.read_text())
    assert prior['encoder_before_sha256'] == prior['encoder_after_sha256'] == INITIAL
    assert receipt['protocol_sha256'] == auth['data']['reference_protocol_sha256']
    for split in rows:
        assert prior['caches'][split]['row_ids'] == [v['image_id'] for v in rows[split]]
    marine = Path(auth['selection']['Marine_run'])
    assert digest(marine / 'protocol.json') == auth['selection']['Marine_protocol_sha256']
    assert digest(marine / 'results.json') == auth['selection']['Marine_results_sha256']
    marine_protocol = json.loads((marine / 'protocol.json').read_text())
    official_sources = marine_protocol['official_source_sha256']
    for path, expected in official_sources.items(): assert digest(path) == expected
    torch.manual_seed(0)
    head = make_fsod_head()
    initial_head = state_digest(head.state_dict())
    head.roi_heads.box_roi_pool = TiledRoIPool()
    assert state_digest(head.state_dict()) == initial_head == target['initial_head_sha256'] == auth['training']['expected_initial_sha256']
    assert sum(v.numel() for v in head.parameters()) == target['actual_head_parameter_count'] == 21730946
    generator, order = torch.Generator().manual_seed(0), hashlib.sha256()
    for _ in range(60): order.update(torch.randperm(2357, generator=generator).numpy().tobytes())
    assert order.hexdigest() == 'c661968727e847e8d7ebb2bb5c52fe21120c1eb8daab7c9c96feb83e4f9f1733'
    paths = [Path(__file__), ROOT / 'tests/test_interp_tiles_fsod_transfer.py', ROOT / 'runs/marine_native_information.py',
             ROOT / 'runs/shared_supervised_readout.py', ROOT / 'runs/shared_supervised_json_readout.py',
             ROOT / 'runs/cfc_frozen_detector.py', ROOT / 'runs/cfc_independent_ap50.py',
             ROOT / 'sonar_encoder/training/model.py', ROOT / 'sonar_encoder/training/readout.py', ROOT / 'sonar_encoder/training/frontends.py']
    for path in paths:
        if str(path) in marine_protocol['source_sha256']: assert digest(path) == marine_protocol['source_sha256'][str(path)]
    (RUN / 'source').mkdir()
    for path in paths: (RUN / 'source' / path.name).write_bytes(path.read_bytes())
    (RUN / 'head').mkdir()
    write_json(RUN / 'protocol.json', {'unit': RUN.name, 'authorization_sha256': AUTH_SHA, 'contract': auth,
        'target': target, 'row_ids': {s: [v['image_id'] for v in rows[s]] for s in rows}, 'inputs_sha256': coco_bindings,
        'initial_encoder_sha256': INITIAL, 'official_config': {'code_dir': parent['code_dir'], 'weights_path': parent['encoder_weights_path']},
        'official_weight_sha256_inherited_not_new_file_hash': parent['encoder_weights_sha256'], 'official_source_sha256': official_sources,
        'initial_head_sha256': initial_head, 'head_parameter_count': 21730946, 'private_order_sha256': order.hexdigest(),
        'reference_T0': {'terminal_AP50': ref['terminal']['AP50'], 'best_AP50': ref['best']['AP50'], 'best_epoch': 60,
            'origin': 'inherited closed score only, not newly trained/extracted/scored', 'parent_protocol_sha256': hashlib.sha256(parent_bytes).hexdigest(),
            'final_closure_sha256': digest(reference / 'final_closure.json'), 'target_results_sha256': digest(target_results_path),
            'score_receipt_sha256': digest(receipt_path), 'report_sha256': receipt['report_sha256'], 'extraction_sha256': receipt['extraction_sha256'],
            'best_head_file_sha256_inherited_not_rehashed': receipt['best_head_file_sha256'], 'encoder_state_sha256': INITIAL},
        'FSOD_counts': actual_counts, 'feature_grid': 32, 'detector_canvas': 224, 'RPN_stride': 7, 'ROI_scale': '1/7',
        'source_sha256': {str(p): digest(p) for p in paths}, 'review_sha256': digest(RUN / 'preflight/source_review.json'),
        'runtime': {'python': sys.executable, 'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda, 'torch_file': torch.__file__,
            'numpy': np.__version__, 'torchvision_metadata_version': importlib.metadata.version('torchvision'),
            'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'CPU_threads': torch.get_num_threads(), 'TF32': torch.backends.cuda.matmul.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
        'scoring_actual_source_math': 'Original FSOD independent_ap50: stored native bbox FP32, promoted to FP64 for NumPy IoU/AP matching; canonical pycocotools unchanged. No newly substituted FP32 matcher.',
        'static_new_peak_planning_bytes': 6 * 1024**3, 'parent_cache_weight_close_rehash': False})
    assert not torch.cuda.is_initialized()
    write_json(RUN / 'prepare_receipt.json', {'status': 'FIXED_ONE_PATH_HEAD_TARGETS_FROZEN_BEFORE_FIRST_IMAGE_FORWARD',
        'protocol_sha256': digest(RUN / 'protocol.json'), 'initial_head_sha256': initial_head,
        'head_parameters': 21730946, 'RPN_stride': 7, 'ROI_spatial_scale': '1/7', 'population_counts': actual_counts, 'resources': resources()})


@torch.no_grad()
def extract():
    runtime()
    begun = time.monotonic()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    target = protocol['target']; config = protocol['official_config']
    torch.manual_seed(0)
    encoder = build_encoder(Path(config['code_dir']), Path(config['weights_path']), 'R010')
    before = state_digest(encoder.state_dict()); assert before == INITIAL
    encoder.to('cuda:0').eval().requires_grad_(False)
    write_json(RUN / 'encoder_entry_receipt.json', {'protocol_sha256': digest(RUN / 'protocol.json'), 'actual_initial_state_sha256': before,
        'model_dtype': str(next(encoder.parameters()).dtype), 'eval': not encoder.training, 'requires_grad_any': any(v.requires_grad for v in encoder.parameters()),
        'actual_original_PE_shape': list(encoder.backbone.pos_embed.shape), 'per_tile_input': [1, 224, 224], 'resources': resources()})
    caches, samples, calls, decoded = {}, 0, 0, 0
    for split in ('train', 'validation'):
        rows = target['rows'][split]
        path = RUN / (split + '.npy')
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 384, 32, 32))
        for start in range(0, len(rows), 4):
            started = time.monotonic()
            views = []
            for row in rows[start:start + 4]:
                raw = Path(row['image']).read_bytes(); assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
                with Image.open(io.BytesIO(raw)) as image: gray = np.array(image.convert('L'), dtype=np.uint8)
                assert gray.shape == (row['height'], row['width'])
                sample = {'intensity': torch.from_numpy(gray).float().unsqueeze(0) / 255, 'valid': torch.ones(gray.shape, dtype=torch.bool)}
                c224 = prepare_images([sample], target['normalization'], None)
                views.append(F.interpolate(c224, size=(448, 448), mode='bilinear', align_corners=False, antialias=True)[0])
            tiles = tile_frames(torch.stack(views)).to('cuda:0')
            field = stitch_tokens(encoder(tiles)['patch_tokens'])
            assert field.shape == (len(views), 384, 32, 32) and field.dtype == torch.float32 and torch.isfinite(field).all()
            array[start:start + len(views)] = field.cpu().numpy(); array.flush()
            samples += len(tiles); calls += 1; decoded += len(views)
            resources()
            if calls == 1:
                seconds = time.monotonic() - started
                projected = 1.5 * seconds * 786
                assert projected <= remaining('extract') - (time.monotonic() - begun)
                assert projected + 4500 + 600 <= 7200 - sum(v['seconds'] for v in ledger()) - (time.monotonic() - begun)
                write_json(RUN / 'encoder_prefix_gate.json', {'status': 'FIRST_NORMAL_BATCH_PASS_AUTOMATIC_CONTINUATION',
                    'counted_rows': decoded, 'counted_tile_samples': samples, 'counted_calls': calls, 'batch_seconds': seconds,
                    'remaining_calls': 786, 'projection_factor': 1.5, 'conditional_remaining_encoding_seconds': projected,
                    'shape': list(field.shape), 'native_statistics': target['normalization'], 'resources': resources()})
            if start % 100 == 0: print('TILED_FSOD', split, start + len(views), 'samples', samples, 'calls', calls, flush=True)
        caches[split] = {'path': str(path), 'sha256': digest(path), 'shape': list(array.shape), 'row_ids': protocol['row_ids'][split]}
    after = state_digest(encoder.state_dict())
    assert before == after == INITIAL and (decoded, samples, calls) == (3142, 12568, 787)
    write_json(RUN / 'extraction.json', {'protocol_sha256': digest(RUN / 'protocol.json'), 'caches': caches,
        'encoder_before_sha256': before, 'encoder_after_sha256': after, 'native_rows_once_decoded': decoded,
        'encoder_samples': samples, 'encoder_calls': calls, 'old_T0_forward': 0, 'resources': resources()})


def atomic_torch(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp); resources(); temp.replace(path)


def atomic_predictions(path, values):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: np.save(f, values, allow_pickle=False)
    resources(); temp.replace(path)


@torch.no_grad()
def evaluate(head, array, rows, coco):
    head.eval()
    outputs = []
    for start in range(0, len(rows), 8):
        features = torch.from_numpy(np.array(array[start:start + 8], copy=True)).to('cuda:0')
        outputs.append(prediction_rows(head(features), rows[start:start + 8]))
        resources()
    predictions = np.concatenate(outputs)
    return canonical_evaluate(coco, prediction_list(predictions)), predictions


def train_head():
    runtime()
    begun = time.monotonic()
    protocol = json.loads((RUN / 'protocol.json').read_text()); target = protocol['target']
    extraction = json.loads((RUN / 'extraction.json').read_text())
    assert extraction['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert extraction['encoder_before_sha256'] == extraction['encoder_after_sha256'] == INITIAL
    arrays = {}
    for split in ('train', 'validation'):
        cache = extraction['caches'][split]
        assert cache['row_ids'] == protocol['row_ids'][split]
        arrays[split] = np.load(cache['path'], mmap_mode='r', allow_pickle=False)
        assert arrays[split].dtype == np.float32 and list(arrays[split].shape) == cache['shape']
    targets = []
    for row in target['rows']['train']:
        annotations = row['annotations']
        boxes = torch.tensor([a['bbox'] for a in annotations], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        targets.append({'boxes': scale_boxes(boxes, (row['height'], row['width']), (224, 224)),
                        'labels': torch.tensor([a['category_id'] for a in annotations], dtype=torch.int64)})
    coco = json.loads(Path(target['validation_coco']).read_text())
    torch.manual_seed(0)
    head = make_fsod_head(); head.roi_heads.box_roi_pool = TiledRoIPool()
    initial = state_digest(head.state_dict()); assert initial == protocol['initial_head_sha256']
    head.to('cuda:0')
    optimizer = torch.optim.SGD(head.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
    generator, order_hash = torch.Generator().manual_seed(0), hashlib.sha256()
    folder = RUN / 'head'; protocol_sha = digest(RUN / 'protocol.json')
    selector = {'best_score': -1., 'best_epoch': 0}
    stream = {'epoch': 0, 'cursor': 0, 'updates': 0, 'order': torch.empty(0, dtype=torch.int64), 'loss_sum': 0.}
    witness = {}

    def anchor_hook(module, inputs, outputs):
        centers = (outputs[0][:, :2] + outputs[0][:, 2:]) / 2
        witness.update(image_canvas=list(inputs[0].tensors.shape[-2:]), feature_grid=list(inputs[1][0].shape[-2:]),
                       anchor_count=len(outputs[0]), origin=centers[0].detach().cpu().tolist(),
                       horizontal_stride=(centers[15] - centers[0]).detach().cpu().tolist(),
                       vertical_stride=(centers[32 * 15] - centers[0]).detach().cpu().tolist())

    hook = head.rpn.anchor_generator.register_forward_hook(anchor_hook)

    def save_boundary(failure):
        atomic_torch(folder / 'last_head.pt', {'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'stream': stream, 'selector': selector, 'private_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(0), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'head_state_sha256': state_digest(head.state_dict()), 'protocol_sha256': protocol_sha,
            'failure_not_safe_for_automatic_resume': failure is not None, 'failure': failure})

    def interrupted(signum, frame):
        raise InterruptedError('Time/resource stop; no automatic resume')

    signal.signal(signal.SIGTERM, interrupted)
    prefix_step_seconds = []
    try:
        save_boundary(None)
        with (folder / 'metrics.jsonl').open('w') as log:
            for epoch in range(1, 61):
                head.train()
                order = torch.randperm(2357, generator=generator); order_hash.update(order.numpy().tobytes())
                stream.update(epoch=epoch, cursor=0, order=order, loss_sum=0.)
                lr = .005 if epoch <= 40 else (.0005 if epoch <= 55 else .00005)
                for group in optimizer.param_groups: group['lr'] = lr
                for indices in order.split(8):
                    started = time.monotonic()
                    features = torch.from_numpy(np.array(arrays['train'][indices.numpy()], copy=True)).to('cuda:0')
                    current_targets = [{key: value.to('cuda:0') for key, value in targets[i].items()} for i in indices.tolist()]
                    optimizer.zero_grad(set_to_none=True)
                    loss = sum(head(features, current_targets).values())
                    assert torch.isfinite(loss)
                    loss.backward(); assert all(torch.isfinite(p.grad).all() for p in head.parameters())
                    optimizer.step(); assert all(torch.isfinite(p).all() for p in head.parameters())
                    stream.update(cursor=stream['cursor'] + len(indices), updates=stream['updates'] + 1,
                                  loss_sum=stream['loss_sum'] + loss.item() * len(indices))
                    if stream['updates'] == 1:
                        hook.remove()
                        assert witness == {'image_canvas': [224, 224], 'feature_grid': [32, 32], 'anchor_count': 15360,
                                           'origin': [0., 0.], 'horizontal_stride': [7., 0.], 'vertical_stride': [0., 7.]}
                    resources()
                    if stream['updates'] <= 100: prefix_step_seconds.append(time.monotonic() - started)
                    if stream['updates'] == 100:
                        projected = 1.5 * sum(prefix_step_seconds) / 100 * (17700 - 100) + 600
                        assert projected <= remaining('head') - (time.monotonic() - begun)
                        assert projected <= 7200 - sum(v['seconds'] for v in ledger()) - (time.monotonic() - begun)
                        save_boundary(None)
                        write_json(RUN / 'head_prefix_gate.json', {'status': 'FIRST100_REAL_UPDATES_PASS_AUTOMATIC_CONTINUATION',
                            'updates_counted_inside17700': 100, 'runtime_anchor_witness': witness, 'ROI_spatial_scale': '1/7',
                            'ROI_output_size': [7, 7], 'ROI_sampling_ratio': 2, 'ROI_aligned': True, 'head_parameter_count': sum(v.numel() for v in head.parameters()),
                            'actual_first100_train_seconds': sum(prefix_step_seconds), 'remaining_updates': 17600,
                            'projection_factor': 1.5, 'fixed_validation_IO_reserve_seconds': 600,
                            'conditional_remaining_head_seconds': projected, 'no_extra_evaluation_or_AP_probe': True, 'resources': resources()})
                    if stream['updates'] % 500 == 0: print('HEAD_FSOD', epoch, stream['updates'], flush=True)
                entry = {'epoch': epoch, 'updates': stream['updates'], 'LR': lr, 'train_loss': stream['loss_sum'] / 2357}
                if epoch % 5 == 0:
                    metrics, predictions = evaluate(head, arrays['validation'], target['rows']['validation'], coco)
                    entry['validation'] = metrics
                    if metrics['AP50'] > selector['best_score']:
                        selector.update(best_score=metrics['AP50'], best_epoch=epoch, best_metrics=metrics,
                                        best_head_state_sha256=state_digest(head.state_dict()))
                        atomic_torch(folder / 'best_head.pt', {'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            'epoch': epoch, 'validation': metrics, 'head_state_sha256': selector['best_head_state_sha256'],
                            'initial_head_sha256': initial, 'encoder_state_sha256': INITIAL, 'protocol_sha256': protocol_sha})
                        atomic_predictions(folder / 'best_validation.npy', predictions)
                    if epoch == 60:
                        terminal_metrics, terminal_state = metrics, state_digest(head.state_dict())
                        terminal_path = folder / ('best_validation.npy' if selector['best_epoch'] == 60 else 'terminal_validation.npy')
                        if selector['best_epoch'] != 60: atomic_predictions(terminal_path, predictions)
                log.write(json.dumps(entry, allow_nan=False) + '\n'); log.flush()
                save_boundary(None)
        assert stream['updates'] == 17700 and order_hash.hexdigest() == protocol['private_order_sha256']
        write_json(folder / 'report.json', {'epochs': 60, 'updates': stream['updates'], 'initial_head_sha256': initial,
            'private_order_sha256': order_hash.hexdigest(), 'best_epoch': selector['best_epoch'], 'best': selector['best_metrics'],
            'terminal': terminal_metrics, 'best_head_state_sha256': selector['best_head_state_sha256'], 'terminal_head_state_sha256': terminal_state,
            'best_head_file_sha256': digest(folder / 'best_head.pt'), 'best_prediction': str(folder / 'best_validation.npy'),
            'terminal_prediction': str(terminal_path), 'best_prediction_sha256': digest(folder / 'best_validation.npy'),
            'terminal_prediction_sha256': digest(terminal_path), 'protocol_sha256': protocol_sha,
            'logical_endpoints': 2, 'unique_prediction_artifacts': 1 if selector['best_epoch'] == 60 else 2,
            'terminal_weight_retained': selector['best_epoch'] == 60, 'resources': resources()})
    except Exception as error:
        failure = {'exception_type': type(error).__name__, 'message': str(error), 'epoch': stream['epoch'],
                   'cursor': stream['cursor'], 'completed_updates': stream['updates'], 'failure_not_safe_for_automatic_resume': True}
        save_boundary(failure); write_json(folder / 'failure.json', failure)
        raise


def close():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    report = json.loads((RUN / 'head/report.json').read_text())
    folder = RUN / 'head'
    assert report['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert report['initial_head_sha256'] == protocol['initial_head_sha256'] and report['private_order_sha256'] == protocol['private_order_sha256']
    saved = torch.load(folder / 'best_head.pt', map_location='cpu', weights_only=True)
    assert digest(folder / 'best_head.pt') == report['best_head_file_sha256']
    assert saved['protocol_sha256'] == report['protocol_sha256'] and saved['epoch'] == report['best_epoch']
    assert state_digest(saved['head']) == report['best_head_state_sha256'] == saved['head_state_sha256']
    history = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines()]
    assert len(history) == 60 and all(v['epoch'] == i and v['updates'] == i * 295 for i, v in enumerate(history, 1))
    evaluated = [v for v in history if 'validation' in v]
    assert [v['epoch'] for v in evaluated] == list(range(5, 61, 5))
    selected = max(evaluated, key=lambda x: x['validation']['AP50'])
    assert selected['epoch'] == report['best_epoch'] and selected['validation'] == saved['validation'] == report['best']
    assert history[-1]['validation'] == report['terminal']
    coco_path = Path(protocol['target']['validation_coco'])
    assert digest(coco_path) == protocol['inputs_sha256'][str(coco_path)]
    coco = json.loads(coco_path.read_text()); scored, checks = {}, []
    for when in ('terminal', 'best'):
        path = Path(report[when + '_prediction']); assert digest(path) == report[when + '_prediction_sha256']
        actual = independent_ap50(coco, read_predictions(path)); expected = report[when]
        assert actual['category_ids'] == expected['category_ids'] == CATEGORIES and abs(actual['AP50'] - expected['AP50']) <= 1e-12
        for key in COUNTS: assert actual[key] == expected[key]
        for cat in map(str, CATEGORIES):
            assert abs(actual['per_class'][cat]['AP50'] - expected['per_class'][cat]['AP50']) <= 1e-12
            for key in (*COUNTS, 'category_id', 'ordering'): assert actual['per_class'][cat][key] == expected['per_class'][cat][key]
        scored[when] = actual
        checks.append({'when': when, 'prediction_path': str(path), 'prediction_sha256': digest(path),
                       'status': 'PASS_INDEPENDENT_NUMPY_AP50_COUNTS_SUPPORT_ORDERING', 'AP50_maxabs': abs(actual['AP50'] - expected['AP50'])})
    for path, expected in protocol['source_sha256'].items(): assert digest(path) == expected
    last_path = folder / 'last_head.pt'; last_sha = digest(last_path)
    last = torch.load(last_path, map_location='cpu', weights_only=False)
    assert last['protocol_sha256'] == report['protocol_sha256'] and last['stream']['updates'] == 17700
    assert state_digest(last['head']) == last['head_state_sha256'] == report['terminal_head_state_sha256']
    write_json(RUN / 'score_receipt.json', {'status': 'PASS_TWO_LOGICAL_SAVED_PREDICTION_SCORES', 'checks': checks,
        'best_epoch': report['best_epoch'], 'unique_prediction_artifacts': report['unique_prediction_artifacts'],
        'removed_new_rolling_path': str(last_path), 'removed_new_rolling_sha256': last_sha, 'parent_cleanup': False, 'resources': resources()})
    del last; last_path.unlink()
    write_json(RUN / 'results.json', {'status': 'COMPLETE_FIXED_FSOD_INTERP_PATH', 'interp_tiles': scored, 'best_epoch': report['best_epoch'],
        'T0_reference': protocol['reference_T0'], 'difference_from_inherited_T0': {w: scored[w]['AP50'] - protocol['reference_T0'][w + '_AP50'] for w in scored},
        'primary': 'terminal60 AP50; best auxiliary', 'new_encoder_samples': 12568, 'new_encoder_calls': 787, 'new_head_updates': 17700,
        'scope': 'Repeateddevelopment split; layout selected using Marine; bundled grid/anchor density/context/compute. No cold-transfer, new-method or causal claim.'})
    write_json(RUN / 'verification.json', {'status': 'PASS_TWO_LOGICAL_INDEPENDENT_SCORES', 'checks': checks,
        'results_sha256': digest(RUN / 'results.json'), 'protocol_sha256': digest(RUN / 'protocol.json'),
        'old_T0_inherited_not_newly_scored': True, 'new_cache_hash_at_extraction_only': True, 'resources': resources()})


def pipeline():
    for phase in ('prepare', 'extract', 'head', 'score_close'):
        if phase in ('extract', 'head'):
            started = time.monotonic(); admitted = gpu_entry(phase)
            with (RUN / 'compute_ledger.jsonl').open('a') as f: f.write(json.dumps({'phase': 'GPU_entry', 'group': phase, 'seconds': time.monotonic() - started}) + '\n')
            if not admitted:
                write_json(RUN / 'deferred.json', {'status': 'DEFERRED_GPU_ENTRY_NO_AUTOMATIC_QUEUE', 'phase': phase}); return
        allowed = remaining(phase); assert allowed > 0
        started, violation = time.monotonic(), None
        log = RUN / (phase + '.log')
        with log.open('w') as output:
            child = subprocess.Popen([sys.executable, '-m', 'runs.interp_tiles_fsod_transfer', phase], stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'phase': phase, 'pid': child.pid, 'log': str(log), 'seconds_available': allowed}), flush=True)
            while child.poll() is None:
                if time.monotonic() - started > allowed:
                    violation = 'TIME_CAP'; os.killpg(child.pid, signal.SIGTERM); break
                time.sleep(1)
            code = child.wait()
        record = {'phase': phase, 'group': phase, 'pid': child.pid, 'seconds': time.monotonic() - started,
                  'exit_code': code, 'budget_violation': violation, 'closed_log_sha256': digest(log)}
        with (RUN / 'compute_ledger.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
        if code != 0 or violation:
            write_json(RUN / 'failure_closure.json', {'status': 'STOP_NO_AUTOMATIC_RETRY', 'failed_phase': record, 'ledger': ledger()}); return
    write_json(RUN / 'completion.json', {'status': 'COMPLETE_STOP_NO_NEW_PHASE', 'charged_seconds': sum(v['seconds'] for v in ledger()),
        'formal_phase_exits': [v for v in ledger() if 'exit_code' in v], 'resources': resources(),
        'results_sha256': digest(RUN / 'results.json'), 'verification_sha256': digest(RUN / 'verification.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'extract', 'head', 'score_close', 'pipeline'))
    phase = parser.parse_args().phase
    {'prepare': prepare, 'extract': extract, 'head': train_head, 'score_close': close, 'pipeline': pipeline}[phase]()
