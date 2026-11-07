"""Finite mature-tracker comparison using existing native-frame predictions."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.ops import nms

from runs import cfc26_mature_trackers as mature

BASE = mature.RUN.parent
RUN = BASE/'cfc26_tracking_flexible_v1'
AUTHOR = BASE/'cfc26_author_weights_eval_v1'
OFFICIAL = BASE/'cfc26_official_alignment_v1/official_source/CFC26-main'
write = mature.write


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Flexible-tracker unit already prepared')
    detectors = {'author_public': {'val': str(AUTHOR/'author_val_predictions.pt'),
                                  'test': str(AUTHOR/'author_test_predictions.pt')}}
    sources = [
        ('GLISR_ROI512', 'cfc26_glisr_rgb_adaptation_v1/RGB512', 'val_epoch20', 'test_terminal'),
        ('GLISR_ROI384', 'cfc26_glisr_rgb_adaptation_v1/RGB384', 'val_epoch15', 'test_best'),
        ('GLISR_YOLO512', 'cfc26_glisr_yolo_adaptation_v1/GLISR_YOLO512', 'val_epoch15', 'test_best'),
        ('GLISR_source_YOLO256', 'cfc26_glisr_fresh_yolo256_v1/FRESH_GLISR_YOLO256', 'val_epoch20', 'test_terminal'),
        ('GLISR_background256', 'cfc26_glisr_background_adaptation_v1/BACKGROUND256', 'val_epoch10', 'test_terminal')]
    for key, folder, val, test in sources:
        detectors[key] = {'val': str(BASE/folder/val/'predictions.pt'),
                          'test': str(BASE/folder/test/'predictions.pt')}
    for paths in detectors.values():
        for path in paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    settings = []
    for threshold in [.3, .5, .7, .9]:
        settings.append({'tracker': 'ByteTrack', 'threshold': threshold, 'use_byte': True})
        for use_byte in [False, True]:
            settings.append({'tracker': 'OC_SORT', 'threshold': threshold, 'use_byte': use_byte,
                             'iou_threshold': .3, 'min_hits': 3})
    protocol = {
        'status': 'FIXED_BEFORE_NEW_ASSOCIATION', 'created_unix': time.time(),
        'user_scope_override': 'User explicitly removed the requirement to retain identical tracking/count code: 就不用限制沿用一样的追踪和计数代码. AP50 is not an optimization or acceptance endpoint.',
        'goal': 'Improve GLISR tracking toward author public weights; HOTA primary, IDF1/MOTA together, counting separately',
        'detectors': detectors, 'settings': settings,
        'input_endpoint_selection': 'Highest old official-validationHOTA among each completed condition retained endpoints; source_YOLO256 uses terminal20, not the old AP-selectedepoch5. No newtest association used for selection',
        'shared_inputs': 'Original native frames/GT/split/rowIDs unchanged; sourceconfidence>.1, class1, clipboxes to nativeHW, NMS.25/max300, no image/model forward',
        'association': 'Actual existing official ByteTrack and OC-SORT sources; ByteTrack buffer30/match.8/frame_rate30; OC-SORT max_age30/min_hits3/IoU.3/delta_t3/inertia.2, optional original use_byte',
        'track_retention': 'Offline remove tracks with fewer than5 emitted observations within each actualsequence. No GT-dependent retention or gap interpolation.',
        'validation': 'All12settings for all6detectors on all1000valframes:72 CPU association evaluations, every direction retained',
        'selection': 'Perdetector/pertracker family highest validationHOTA, then IDF1, then MOTA; first declared setting wins remainingties. Lock all12selected pipelines before newtest association. Also lock overall GLISR and author pipelines using validation only.',
        'test': 'All2000testframes for12selected pipelines; original official baseline reference retained separately. No test-based re-selection.',
        'counting': 'Keep the original officialLOI counter initially. Reuse originalGT/frame directory links; no duplicatedGT/images. NativeMOTxywh rounding3decimals; conf1 is a format placeholder for trackeroutputs lacking confidence, not a predicted probability.',
        'budgets': {'CPU_workers': 4, 'CPU_threads_each': 2, 'global_seconds': 1200,
                    'each_detector_seconds': 300, 'RSS_each_GiB': 3, 'new_disk_GiB': 1,
                    'GPU_allocations': 0, 'new_training': 0, 'new_downloads': 0},
        'stop': 'Actual wall/RSS/code failure is visible; no automatic retry, new settings or test-based tuning',
        'population_scope': 'Existing repeatedly used local10k snapshot; this is development optimization, not a fresh blind or acquisition-independent test'}
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    write(RUN/'protocol.json', protocol)


def load_inputs(key, split, protocol):
    rows = mature.tracking.load_rows(split)
    source = Path(protocol['detectors'][key][split])
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if payload['row_ids'] != [r['row_id'] for r in rows]:
        raise RuntimeError(f'Prediction/GT row identities differ: {key}/{split}')
    detections = []
    for row, det in zip(rows, payload['native_detections']):
        boxes = torch.as_tensor(det['boxes'], dtype=torch.float32).clone()
        scores = torch.as_tensor(det['scores'], dtype=torch.float32)
        h, w = row['actual_shape_HWC'][:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        valid = ((scores > .1) & (torch.as_tensor(det['labels']) == 1) &
                 (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        boxes, scores = boxes[valid], scores[valid]
        keep = nms(boxes, scores, .25)[:300]
        detections.append(np.column_stack((boxes[keep].numpy(), scores[keep].numpy())).astype(np.float64))
    return rows, detections, source


def associate(rows, detections, setting):
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import BaseTrack
    from trackers.ocsort_tracker.ocsort import OCSort, KalmanBoxTracker
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['sequence_id']].append(i)
    tracks = [None]*len(rows)
    for indices in groups.values():
        if setting['tracker'] == 'ByteTrack_birth':
            BaseTrack._count = 0
            engine = BYTETracker(SimpleNamespace(track_thresh=setting['threshold'], track_buffer=30,
                                 match_thresh=setting['match_thresh'], mot20=True), frame_rate=30)
            engine.det_thresh = setting['new_track_thresh']
        elif setting['tracker'] == 'ByteTrack':
            BaseTrack._count = 0
            engine = BYTETracker(SimpleNamespace(track_thresh=setting['threshold'], track_buffer=30,
                                                  match_thresh=.8, mot20=False), frame_rate=30)
        else:
            KalmanBoxTracker.count = 0
            engine = OCSort(det_thresh=setting['threshold'], max_age=30, min_hits=setting['min_hits'],
                            iou_threshold=setting['iou_threshold'], delta_t=3, asso_func='iou', inertia=.2,
                            use_byte=setting['use_byte'])
        for i in indices:
            hw = tuple(rows[i]['actual_shape_HWC'][:2])
            emitted = engine.update(detections[i].copy(), hw, hw)
            if setting['tracker'] in ('ByteTrack', 'ByteTrack_birth'):
                tracks[i] = {'boxes': np.asarray([t.tlbr for t in emitted]).reshape(-1, 4),
                             'ids': np.asarray([t.track_id for t in emitted], dtype=np.int64)}
            else:
                tracks[i] = {'boxes': emitted[:, :4].copy(), 'ids': emitted[:, 4].astype(np.int64)}
        counts = Counter(int(identity) for i in indices for identity in tracks[i]['ids'])
        for i in indices:
            keep = np.array([counts[int(identity)] >= 5 for identity in tracks[i]['ids']], dtype=bool)
            tracks[i] = {'boxes': tracks[i]['boxes'][keep], 'ids': tracks[i]['ids'][keep]}
    return tracks


def count_tracks(rows, tracks, folder, split):
    for name in ['gt', 'frames']:
        (folder/name).symlink_to(AUTHOR/split/'author_fisheye'/name, target_is_directory=True)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['sequence_id']].append(i)
    for sequence, indices in groups.items():
        location, clip, segment = sequence.split('/')
        name = clip+'_'+segment
        target = folder/'infer'/location/name
        target.mkdir(parents=True)
        with (target/(name+'.txt')).open('w') as f:
            for i in indices:
                for box, identity in zip(tracks[i]['boxes'], tracks[i]['ids']):
                    x, y, x2, y2 = box
                    f.write(f'{rows[i]["frame_index"]+1},{identity},{x:.3f},{y:.3f},{x2-x:.3f},{y2-y:.3f},1,-1,-1,-1\n')
    with (folder/'official_count_stdout.log').open('w') as log:
        subprocess.run([sys.executable, str(OFFICIAL/'scripts/eval_count_infer.py'),
            '--infer_dir', str(folder/'infer'), '--gt_dir', str(folder/'gt'),
            '--frames_dir', str(folder/'frames'), '--output_dir', str(folder/'count'),
            '--count_protocol', 'LOI'], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    import pandas as pd
    result = pd.read_csv(folder/'count/eval_summary.csv').replace({np.nan: None}).to_dict('records')
    if sum(int(x['n_clips']) for x in result if x['location'] != 'ALL') != len(groups):
        raise RuntimeError('Official count evaluator did not include every originalsegment')
    return result


def evaluate_detector(key, split, settings, run):
    RUN = Path(run)
    torch.set_num_threads(2)
    protocol = json.loads((RUN/'protocol.json').read_text())
    rows, detections, source = load_inputs(key, split, protocol)
    started = time.monotonic()
    results = []
    for setting in settings:
        index = protocol['settings'].index(setting)
        folder = RUN/split/key/f'config_{index:02}'
        folder.mkdir(parents=True)
        tracks = associate(rows, detections, setting)
        metrics = mature.tracking.score(rows, tracks)
        record = {'condition': key, 'split': split, 'setting': setting, 'config_index': index,
                  'tracking': metrics, 'input_predictions': str(source), 'worker_PID': os.getpid()}
        if split == 'test':
            torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': tracks,
                        'setting': setting, 'input_predictions': str(source)}, folder/'tracks.pt')
            record['official_LOI_counts'] = count_tracks(rows, tracks, folder, 'test')
        record['elapsed_detector_seconds'] = time.monotonic()-started
        record['max_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if record['elapsed_detector_seconds'] > 300 or record['max_RSS_bytes'] > 3*1024**3:
            raise RuntimeError(f'Actual CPU evaluation resource cap: {key}/{split}')
        write(folder/'result.json', record)
        results.append(record)
    return results


def pipeline():
    protocol = json.loads((RUN/'protocol.json').read_text())
    tracker_families = list(dict.fromkeys(s['tracker'] for s in protocol['settings']))
    val_count = len(protocol['detectors'])*len(protocol['settings'])
    test_count = len(protocol['detectors'])*len(tracker_families)
    write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'started_unix': time.time(),
                                     'GPU_allocations': 0, 'validation_jobs': val_count, 'test_jobs': test_count})
    validation = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(evaluate_detector, key, 'val', protocol['settings'], str(RUN)): key
                   for key in protocol['detectors']}
        for future in as_completed(futures):
            validation.extend(future.result())
            write(RUN/'progress.json', {'phase': 'validation', 'complete': len(validation), 'total': val_count})
    write(RUN/'all_validation_results.json', validation)
    selected = []
    for key in protocol['detectors']:
        for tracker in tracker_families:
            choices = sorted([r for r in validation if r['condition'] == key and r['setting']['tracker'] == tracker],
                             key=lambda r: r['config_index'])
            selected.append(max(choices, key=lambda r: tuple(r['tracking'][k] for k in ['HOTA', 'IDF1', 'MOTA'])))
    best_author = max([r for r in selected if r['condition'] == 'author_public'], key=lambda r: r['tracking']['HOTA'])
    best_glisr = max([r for r in selected if r['condition'] != 'author_public'], key=lambda r: r['tracking']['HOTA'])
    write(RUN/'selection.json', {'locked_before_new_test_association': True, 'created_unix': time.time(),
        'selected': selected, 'primary_author': best_author, 'primary_GLISR': best_glisr})
    test = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(evaluate_detector, key, 'test', [r['setting'] for r in selected if r['condition'] == key], str(RUN)): key
                   for key in protocol['detectors']}
        for future in as_completed(futures):
            test.extend(future.result())
            write(RUN/'progress.json', {'phase': 'test', 'complete': len(test), 'total': test_count})
    new_bytes = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
    if new_bytes > 1024**3:
        raise RuntimeError(f'Actual new-entity disk cap: {new_bytes}')
    write(RUN/'results.json', {'status': 'COMPLETE', 'test_results': test,
                              'selection': str(RUN/'selection.json'), 'model_forwards': 0})
    write(RUN/'completion.json', {'status': 'COMPLETE', 'validation_evaluations': len(validation),
          'test_evaluations': len(test), 'GPU_allocations': 0, 'new_training': 0,
          'source_predictions_unchanged': True, 'new_entity_bytes_before_summary': new_bytes,
          'CUDA_initialized': torch.cuda.is_initialized()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        pipeline()
