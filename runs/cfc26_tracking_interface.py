"""CFC26 fixed detector + IoU tracking interface, preparation and official scoring.

All inference boxes are native xyxy. Detector targets alone use stretched 224x224.
No labels enter association. Empty frames and sequence boundaries are retained.
"""
import os
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
DATA = BASE / 'cfc26_tracking_10000_v1'
RUN = BASE / 'cfc26_tracking_models_v1'
VENDOR = (Path(__file__).resolve().parents[1] / 'third_party') / 'TrackEval'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_rows(split):
    rows = [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text().splitlines()]
    return sorted(rows, key=lambda row: (row['sequence_id'], row['frame_index']))


def target(row):
    import torch
    boxes = np.asarray([ann['bbox'] for ann in row['annotations']], dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    height, width = row['actual_shape_HWC'][:2]
    boxes *= np.array([224 / width, 224 / height] * 2, dtype=np.float32)
    return {'boxes': torch.from_numpy(boxes), 'labels': torch.ones(len(boxes), dtype=torch.int64)}


def native_detections(detection, row):
    """Convert one detector result from 224 canvas to native coordinates once."""
    height, width = row['actual_shape_HWC'][:2]
    return {'boxes': detection['boxes'].detach().cpu().numpy() * np.array([width / 224, height / 224] * 2),
            'scores': detection['scores'].detach().cpu().numpy(),
            'labels': detection['labels'].detach().cpu().numpy()}


def make_head(siglip=False):
    """Fresh fish+background head: 384x32x32 at stride7, or native SigLIP768x14x14."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    if siglip:
        from runs.siglip2_frozen_components import SiglipDetectorHead
        head = SiglipDetectorHead()
    else:
        from runs.cfc_frozen_detector import FrozenDetectorHead
        from runs.interp_tiles_fsod_transfer import TiledRoIPool
        head = FrozenDetectorHead()
        head.roi_heads.box_roi_pool = TiledRoIPool()
    head.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    return head


def box_iou(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, 4), np.asarray(b, dtype=np.float64).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    union = (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - intersection
    return intersection / union


def associate_sequence(detections):
    """Consecutive ordered frames; allow two missing frames, emit observed fish only.

    Gating precedes assignment: forbidden edges cannot displace valid matches.
    IDs start at 1 per sequence and increase monotonically, including after expiry.
    """
    tracks, next_id, output = {}, 1, []
    for frame, detection in enumerate(detections):
        keep = (np.asarray(detection['scores']) >= .25) & (np.asarray(detection['labels']) == 1)
        boxes = np.asarray(detection['boxes'], dtype=np.float64).reshape(-1, 4)[keep]
        scores = np.asarray(detection['scores'])[keep]
        tracks = {identity: state for identity, state in tracks.items() if frame - state[1] <= 3}
        identities = list(tracks)
        assigned = np.zeros(len(boxes), dtype=np.int64)
        if identities and len(boxes):
            similarity = box_iou([tracks[identity][0] for identity in identities], boxes)
            cost = np.where(similarity >= .3, 1 - similarity, 1e6)
            ii, jj = linear_sum_assignment(cost)
            for i, j in zip(ii, jj):
                if similarity[i, j] >= .3:
                    assigned[j] = identities[i]
        for j, box in enumerate(boxes):
            if assigned[j] == 0:
                assigned[j] = next_id
                next_id += 1
            tracks[int(assigned[j])] = (box, frame)
        output.append({'boxes': boxes, 'ids': assigned, 'scores': scores})
    return output


def associate(rows, detections):
    """Input/result aligned to rows, resetting state for every continuous sequence."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['sequence_id']].append(i)
    output = [None] * len(rows)
    for indices in groups.values():
        indices.sort(key=lambda i: rows[i]['frame_index'])
        for i, prediction in zip(indices, associate_sequence([detections[i] for i in indices])):
            output[i] = prediction
    return output


def metric_data(rows, predictions):
    """Official TrackEval preprocessed single-class sequence contract.

    CFC labels are original fish boxes, with no ignore/distractor classes. IDs are
    remapped only for TrackEval's contiguous-index requirement, within sequence.
    """
    gt_ids, tracker_ids, similarity = [], [], []
    gt_map = {identity: i for i, identity in enumerate(sorted({ann['track_id'] for row in rows for ann in row['annotations']}))}
    pred_map = {identity: i for i, identity in enumerate(sorted({int(identity) for pred in predictions for identity in pred['ids']}))}
    for row, pred in zip(rows, predictions):
        gt_boxes = np.asarray([ann['bbox'] for ann in row['annotations']], dtype=np.float64).reshape(-1, 4)
        gt_boxes[:, 2:] += gt_boxes[:, :2]
        gt_ids.append(np.array([gt_map[ann['track_id']] for ann in row['annotations']], dtype=np.int64))
        tracker_ids.append(np.array([pred_map[int(identity)] for identity in pred['ids']], dtype=np.int64))
        similarity.append(box_iou(gt_boxes, pred['boxes']))
    return {'gt_ids': gt_ids, 'tracker_ids': tracker_ids, 'similarity_scores': similarity,
            'num_gt_ids': len(gt_map), 'num_tracker_ids': len(pred_map), 'num_timesteps': len(rows),
            'num_gt_dets': sum(map(len, gt_ids)), 'num_tracker_dets': sum(map(len, tracker_ids))}


def score(rows, predictions):
    sys.path.insert(0, str(VENDOR))
    from trackeval.metrics import HOTA, CLEAR, Identity
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['sequence_id']].append(i)
    metrics = [HOTA(), CLEAR({'PRINT_CONFIG': False}), Identity({'PRINT_CONFIG': False})]
    results = {metric.get_name(): {} for metric in metrics}
    for sequence, indices in groups.items():
        indices.sort(key=lambda i: rows[i]['frame_index'])
        data = metric_data([rows[i] for i in indices], [predictions[i] for i in indices])
        for metric in metrics:
            results[metric.get_name()][sequence] = metric.eval_sequence(data)
    combined = {metric.get_name(): metric.combine_sequences(results[metric.get_name()]) for metric in metrics}
    return {'HOTA': float(combined['HOTA']['HOTA'].mean()), 'IDF1': float(combined['Identity']['IDF1']),
            'MOTA': float(combined['CLEAR']['MOTA']), 'IDs': int(combined['CLEAR']['IDSW']),
            'FP': int(combined['CLEAR']['CLR_FP']), 'FN': int(combined['CLEAR']['CLR_FN']),
            'GT': sum(len(row['annotations']) for row in rows), 'frames': len(rows), 'sequences': len(groups),
            'HOTA_alpha': metrics[0].array_labels.tolist(),
            'HOTA_by_alpha': combined['HOTA']['HOTA'].tolist(),
            'metric_implementation': 'official JonathonLuiten/TrackEval; sequence-combined single fish class; HOTA/IDF1 fractions; MOTA can be negative'}


def prepare():
    import torch
    from torch.nn import functional as F
    start = time.monotonic()
    torch.set_num_threads(4)
    RUN.mkdir(parents=True, exist_ok=True)
    all_rows = {split: load_rows(split) for split in ('train', 'val', 'test')}
    count, total, squared = 0, 0., 0.
    for row in all_rows['train']:
        with Image.open(row['path']) as image:
            values = np.log1p(np.asarray(image.convert('L'), dtype=np.float64) / 255.)
        count += values.size
        total += values.sum(dtype=np.float64)
        squared += np.square(values).sum(dtype=np.float64)
        if time.monotonic() - start > 1800:
            raise TimeoutError('30-minute CPU preparation cap')
    normalization = {'mean': total / count, 'std': float(np.sqrt(squared / count - (total / count) ** 2)),
                     'pixel_count': count, 'fit_frames': len(all_rows['train']), 'fit_split': 'train',
                     'definition': 'FP64 native pixel-weighted population log1p(PIL L uint8/255); train only'}
    write_json(RUN / 'normalization.json', normalization)
    cache_bytes = sum(len(rows) for rows in all_rows.values()) * 224 * 224 * 4
    if cache_bytes > 4 * 1024**3:
        raise RuntimeError('4 GiB preparation cache cap exceeded')
    for split, rows in all_rows.items():
        output = np.lib.format.open_memmap(RUN / f'{split}_224.npy', mode='w+', dtype=np.float32,
                                           shape=(len(rows), 1, 224, 224))
        for i, row in enumerate(rows):
            with Image.open(row['path']) as image:
                gray = np.asarray(image.convert('L'), dtype=np.uint8).copy()
            native = torch.from_numpy(gray).float()[None, None] / 255.
            native = (native.log1p() - normalization['mean']) / normalization['std']
            output[i] = F.interpolate(native, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)[0].numpy()
            if time.monotonic() - start > 1800:
                raise TimeoutError('30-minute CPU preparation cap')
        output.flush()
        with (RUN / f'{split}_rows.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        del output
    write_json(RUN / 'prepare_receipt.json', {'status': 'CPU_PREPARED_NO_TRAINING', 'seconds': time.monotonic() - start,
                'split_frames': {split: len(rows) for split, rows in all_rows.items()},
                'normalization': normalization, 'cache_bytes': cache_bytes,
                'cache_definition': 'FP32 [N,1,224,224] mmap; native log1p normalization then torch antialiased bilinear stretch',
                'SigLIP2': 'Raw source PIL L→RGB, official SiglipImageProcessor resize224 bilinear, rescale1/255, mean/std .5. Does not use normalized cache. Complete preprocessing+encoder baseline, not normalization-matched encoder attribution.',
                'target_transform': 'Original xywh converted to xyxy then x*=224/native_width,y*=224/native_height; predictions inverse transformed for tracking/scoring. No crop or frame removal.',
                'source_exposure': 'CFC26 prior diagnostic history; Eel/Nanaimo absent in compared six-interface GLISR CFC22 Kenai source definition; fish already seen; no globally unseen or foundation-exclusion claim'})


def self_test():
    def detection(boxes, scores=None):
        return {'boxes': np.asarray(boxes, dtype=float).reshape(-1, 4),
                'scores': np.asarray(scores if scores is not None else [1.] * len(boxes)), 'labels': np.ones(len(boxes), dtype=int)}
    # Order swaps while objects approach/cross in x with separated y lanes.
    frames = [detection([[0, 0, 10, 10], [8, 20, 18, 30]]),
              detection([[5, 20, 15, 30], [3, 0, 13, 10]]),
              detection([[6, 0, 16, 10], [2, 20, 12, 30]])]
    crossing = associate_sequence(frames)
    assert [p['ids'].tolist() for p in crossing] == [[1, 2], [2, 1], [1, 2]]
    empty = detection([])
    bridge = associate_sequence([frames[0], empty, empty, frames[0], empty, empty, empty, frames[0]])
    assert bridge[3]['ids'].tolist() == [1, 2] and bridge[7]['ids'].tolist() == [3, 4]
    assert len(associate_sequence([detection([[0, 0, 1, 1]], [.249])])[0]['ids']) == 0
    rows = [{'sequence_id': 'known', 'frame_index': t, 'annotations': [{'track_id': 7, 'bbox': [0, 0, 10, 10]}]} for t in range(3)]
    perfect = associate_sequence([detection([[0, 0, 10, 10]])] * 3)
    perfect_score = score(rows, perfect)
    assert perfect_score['HOTA'] == perfect_score['IDF1'] == perfect_score['MOTA'] == 1.
    imperfect = associate_sequence([detection([[0, 0, 10, 10]]), empty,
                                    detection([[0, 0, 10, 10], [100, 100, 110, 110]])])
    measured = score(rows, imperfect)
    # GT=3, TP=2, FN=1, FP=1, no switch; exact association quality=2/3.
    assert measured['FP'] == measured['FN'] == 1 and measured['IDs'] == 0
    assert np.isclose(measured['MOTA'], 1 / 3) and np.isclose(measured['IDF1'], 2 / 3)
    assert np.isclose(measured['HOTA'], np.sqrt((2 / 4) * (2 / 3)))
    switched = [dict(p, ids=np.array([i + 1])) for i, p in enumerate(perfect)]
    switches = score(rows, switched)
    assert switches['IDs'] == 2 and np.isclose(switches['IDF1'], 1 / 3)
    # Real original train GT exercises xywh, sparse track IDs and empty frames.
    train = load_rows('train')
    sequence = next(row['sequence_id'] for row in train if row['annotations'])
    real_rows = [row for row in train if row['sequence_id'] == sequence]
    canonical = []
    for row in real_rows:
        boxes = np.asarray([ann['bbox'] for ann in row['annotations']], dtype=float).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        canonical.append({'boxes': boxes, 'ids': np.array([ann['track_id'] for ann in row['annotations']])})
    real_score = score(real_rows, canonical)
    assert real_score['HOTA'] == real_score['IDF1'] == real_score['MOTA'] == 1.
    result = {'real_train_GT_identity_scoring': real_score,
              'real_train_GT_sequence': sequence,
              'crossing_order': [p['ids'].tolist() for p in crossing], 'two_missed_frames_rejoined': True,
              'expired_ids_not_reused': True, 'perfect': perfect_score, 'miss_and_false_positive': measured,
              'known_switches': switches, 'official_revision': ((VENDOR/'GIT_COMMIT').read_text().strip() if (VENDOR/'GIT_COMMIT').is_file() else subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()),
              'vendor_patch': 'Removed NumPy aliases np.float→float,np.int→int for installed NumPy; algorithms unchanged'}
    write_json(RUN / 'interface_self_test.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'self-test'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        print(json.dumps(self_test(), indent=2))
