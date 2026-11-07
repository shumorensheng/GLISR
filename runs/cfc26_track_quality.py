"""Train-only track quality regression on a fixed, frozen GLISR endpoint."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
from torch import nn
from torchvision.ops import nms

from runs import cfc26_location_motion as location
from runs import cfc26_glisr_native128 as native

rgb = native.rgb
RUN = location.BASE/'cfc26_track_quality_v1'
KEY = 'GLISR_native128_terminal'
INIT = native.RUN/native.ARM/'latest.pt'
write = location.write
FEATURES = ['log_observations', 'coverage', 'logit_mean', 'logit_std', 'logit_q10',
            'logit_q90', 'relative_motion_log1p', 'net_over_path', 'relative_step_mean',
            'relative_step_std', 'consecutive_IoU_mean', 'log_width_std', 'log_height_std',
            'log_aspect_median', 'relative_diagonal_median', 'is_nanaimo']


def prepare():
    RUN.mkdir(exist_ok=False)
    p = json.loads((location.RUN/'protocol.json').read_text())
    rows = rgb.tracking.load_rows('train')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['sequence_id']].append(i)
    indices = []
    for group in groups.values():
        start = (len(group)-32)//2
        indices.extend(group[start:start+32])
    selected = [rows[i] for i in indices]
    views = sum(len(rgb.origins(r['actual_shape_HWC'][0], 128))*len(rgb.origins(r['actual_shape_HWC'][1], 128)) for r in selected)
    calls = sum(int(np.ceil(len(rgb.origins(r['actual_shape_HWC'][0], 128))*len(rgb.origins(r['actual_shape_HWC'][1], 128))/16)) for r in selected)
    saved = torch.load(INIT, map_location='cpu', weights_only=False)
    identity = {'checkpoint': str(INIT), 'epoch': saved['progress']['epoch'], 'updates': saved['progress']['updates'],
                'projection_shape': list(saved['model']['encoder.backbone.patch_embed.proj.weight'].shape), 'head_state_keys': len(saved['head'])}
    del saved
    p.update({
        'status': 'FIXED_BEFORE_FIRST_NEW_IMAGE_FORWARD', 'created_unix': time.time(),
        'user_scope_override': 'Different tracking/counting code explicitly allowed. Original data/GT/evaluator preserved; AP50 not an objective.',
        'source_model': identity,
        'training_rows': {'indices': indices, 'row_ids': [r['row_id'] for r in selected], 'frames': len(selected), 'sequences': len(groups),
                          'selection': 'Exactly middle32consecutive released frames of each of26originaltrainsegments; no GT/performance-dependent selection.',
                          'native128_window_views': views, 'batch16_calls': calls},
        'image_inference': 'Actual immutable Native128 terminal10, GLISR soleimageencoder/nativeRGB128/stride96+last/global224+local448/ImageNet/FP32/noAMP/TF32off/B16. Same publishedYOLOtaskneck/Detect, no teacher network. Train selected832frames only; val/test reuse saved predictions.',
        'motion_rule': 'location',
        'features': FEATURES,
        'quality_target': 'Original trainGT only, framewise one-to-one Hungarian IoU>=0.5 matching of alreadyassociated observed boxes, allGT retained. Pertrack target=matched observations/observations, sample_weight=observations. This is a frame-quality training surrogate, not a new evaluation metric.',
        'estimator': {'class': 'sklearn.ensemble.HistGradientBoostingRegressor', 'loss': 'squared_error', 'learning_rate': .1,
                      'max_iter': 100, 'max_leaf_nodes': 7, 'min_samples_leaf': 10, 'l2_regularization': 1., 'early_stopping': False, 'random_state': 0},
        'thresholds': [None, .25, .5, .75],
        'validation': 'One fitted estimator, identity versus three predeclared quality thresholds on all1000originalvalframes. HOTA-first and balancedworstpositive3metricdeficit roles locked before test. Existing identity must reproduce prior fullvalidation vector.',
        'test': 'Selected unique thresholds/all2000originaltestframes; retain full HOTA/IDF1/MOTA/FP/FN/IDs and officialLOI counting. No test labels used in quality features, fitting or threshold selection.',
        'scope': 'Conventional supervised tracker-output quality adaptation; known-location inputs and offline wholetrack descriptors explicit. Same original repeatedly used local10k snapshot, no independent holdout claim or encoder-only gain.',
        'budgets': {'global_seconds': 3600, 'train_inference_seconds': 2400, 'CPU_threads': 4, 'RSS_GiB': 16, 'GPU_reserved_GiB': 16,
                    'new_disk_MiB': 128, 'physical_GPU': 0, 'minimum_free_GiB': 24},
        'stop': 'Only one fixed estimator/100iterations/three thresholds. Actualtime/resource/code/numeric errors visible, no automatic retries or additional sampling/model/thresholds. No newsource or neural training/downloads.'})
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    shutil.copyfile(location.__file__, RUN/'location_snapshot.py')
    write(RUN/'protocol.json', p)
    nodes = [
        {'id': 'title', 'label': 'GLISR：训练集监督的轨迹质量评分', 'x': 650, 'y': 30, 'width': 1200, 'height': 40, 'font_size': 22, 'fill': '#FFFFFF', 'stroke': '#FFFFFF'},
        {'id': 'train', 'label': '26 个训练片段\n每段固定连续 32 帧\n原图与原始 GT', 'x': 160, 'y': 150, 'width': 240, 'height': 100},
        {'id': 'frozen', 'label': '现有 GLISR 模型冻结\n原生 128 滑窗推理\n不更新图像编码器', 'x': 480, 'y': 150, 'width': 260, 'height': 100, 'fill': '#EAF3FF'},
        {'id': 'tracks', 'label': '现有关联与运动过滤\n置信度 / 运动 / 尺寸\n持续性等轨迹描述', 'x': 820, 'y': 150, 'width': 270, 'height': 100},
        {'id': 'fit', 'label': '训练 GT 匹配比例监督\n固定小型回归树评分器\n只拟合轨迹质量', 'x': 1130, 'y': 150, 'width': 260, 'height': 100, 'fill': '#E9F7EF'},
        {'id': 'pred', 'label': '保存的验证 / 测试检测\n不重新编码图像', 'x': 230, 'y': 350, 'width': 310, 'height': 95},
        {'id': 'apply', 'label': '同一轨迹特征 → 质量评分\n验证锁定阈值 → 整轨迹保留', 'x': 720, 'y': 350, 'width': 370, 'height': 95, 'fill': '#E9F7EF'},
        {'id': 'eval', 'label': '全部原 GT 评价\nHOTA / IDF1 / MOTA\n官方 LOI 计数', 'x': 1130, 'y': 350, 'width': 260, 'height': 95}]
    edges = [{'from': a, 'to': b} for a, b in [('train', 'frozen'), ('frozen', 'tracks'), ('tracks', 'fit'), ('pred', 'apply'), ('fit', 'apply'), ('apply', 'eval')]]
    write(RUN/'architecture_spec.json', {'canvas': {'width': 1300, 'height': 440}, 'nodes': nodes, 'edges': edges})


def guard(deadline):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    gpu = torch.cuda.max_memory_reserved()
    if time.time() > deadline or rss > 16*1024**3 or gpu > 16*1024**3:
        raise RuntimeError(f'Trackquality actual resource cap: RSS={rss},GPU={gpu}')
    return {'RSS_bytes': rss, 'GPU_reserved_bytes': gpu}


@torch.no_grad()
def infer_training(p, deadline):
    rgb.native.setup(0)
    source = json.loads(rgb.PARENT.read_text())
    model, _ = rgb.native.construct('Rtau', 0, source)
    model.encoder.backbone.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14)
    head = native.parent.YOLOTaskHead()
    saved = torch.load(INIT, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    head.load_state_dict(saved['head'], strict=True)
    del saved
    model.eval().requires_grad_(False).cuda()
    head.eval().requires_grad_(False).cuda()
    versions = [v._version for m in (model, head) for v in (*m.parameters(), *m.buffers())]
    original = rgb.tracking.load_rows('train')
    rows = [original[i] for i in p['training_rows']['indices']]
    predictions, calls, views = [], 0, 0
    start = time.monotonic()
    for i, row in enumerate(rows):
        h, w = row['actual_shape_HWC'][:2]
        with Image.open(row['path']) as image:
            pixels = np.array(image.convert('RGB'), copy=True)
        positions = [(x, y) for y in rgb.origins(h, 128) for x in rgb.origins(w, 128)]
        boxes, scores = [], []
        for offset in range(0, len(positions), 16):
            current = positions[offset:offset+16]
            g, local = rgb.views([pixels[y:y+128, x:x+128] for x, y in current])
            output = head(model.field(g, head, 'detection', high_resolution=local))
            for d, (x, y) in zip(output, current):
                boxes.append(d['boxes']*(128/224)+d['boxes'].new_tensor([x, y, x, y]))
                scores.append(d['scores'])
            calls += 1
            views += len(current)
            guard(deadline)
        boxes, scores = torch.cat(boxes), torch.cat(scores)
        keep = nms(boxes, scores, .25)[:300]
        predictions.append({'boxes': boxes[keep].float().cpu().numpy(), 'scores': scores[keep].float().cpu().numpy(),
                            'labels': np.ones(len(keep), dtype=np.int64)})
        if (i+1) % 32 == 0:
            receipt = {'phase': 'train_inference', 'frames': i+1, 'total_frames': len(rows), 'calls': calls,
                       'views': views, 'seconds': time.monotonic()-start, **guard(deadline)}
            write(RUN/'progress.json', receipt)
            print(json.dumps(receipt), flush=True)
    unchanged = versions == [v._version for m in (model, head) for v in (*m.parameters(), *m.buffers())]
    if not unchanged or calls != p['training_rows']['batch16_calls'] or views != p['training_rows']['native128_window_views']:
        raise RuntimeError('Frozen train extraction identity/count witness failed')
    path = RUN/'train_predictions.pt'
    torch.save({'row_ids': [r['row_id'] for r in rows], 'native_detections': predictions, 'split': 'train_subset'}, path)
    write(RUN/'inference_receipt.json', {'status': 'COMPLETE', 'frames': len(rows), 'calls': calls, 'views': views,
          'seconds': time.monotonic()-start, 'student_parameter_buffer_versions_unchanged': unchanged,
          'new_encoder_updates': 0, 'teacher_forwards': 0, **guard(deadline)})
    del model, head
    torch.cuda.empty_cache()
    return rows


def associate(rows, detections, p):
    tracks = location.motion.flexible.associate(rows, detections, p['association'][KEY])
    tracks, _ = location.motion.confirmation.confirmed(rows, tracks, p['confirmations'][KEY])
    return location.filtered(rows, tracks, location.motion.descriptors(rows, tracks), 'location', p)


def examples(rows, detections, tracks, training):
    grouped = defaultdict(list)
    for row, det, track in zip(rows, detections, tracks):
        boxes = track['boxes']
        matches = np.zeros(len(boxes), dtype=bool)
        if training and len(boxes) and row['annotations']:
            gt = np.asarray([a['bbox'] for a in row['annotations']], dtype=np.float64)
            gt[:, 2:] += gt[:, :2]
            overlap = rgb.tracking.box_iou(gt, boxes)
            ii, jj = linear_sum_assignment(np.where(overlap >= .5, 1-overlap, 1e6))
            matches[jj[overlap[ii, jj] >= .5]] = True
        if len(boxes):
            overlap = rgb.tracking.box_iou(boxes, det[:, :4])
            index = overlap.argmax(1)
            if (overlap[np.arange(len(boxes)), index] < .999999).any():
                raise RuntimeError('Observed track box no longer identifies an original detection confidence')
            confidence = det[index, 4]
        for j, (box, identity) in enumerate(zip(boxes, track['ids'])):
            grouped[(row['sequence_id'], int(identity))].append((row['frame_index'], box, float(confidence[j]), bool(matches[j]), row['actual_shape_HWC'][:2]))
    keys, features, quality, weights = [], [], [], []
    for key, values in grouped.items():
        frame = np.asarray([v[0] for v in values])
        boxes = np.stack([v[1] for v in values])
        logit = np.log(np.clip([v[2] for v in values], 1e-6, 1-1e-6)/(1-np.clip([v[2] for v in values], 1e-6, 1-1e-6)))
        wh = boxes[:, 2:]-boxes[:, :2]
        centers = (boxes[:, 2:]+boxes[:, :2])/2
        diagonal = np.median(np.linalg.norm(wh, axis=1))
        lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        steps = lengths/np.diff(frame)/diagonal
        overlap = np.diag(rgb.tracking.box_iou(boxes[:-1], boxes[1:]))
        log_wh = np.log(wh)
        vector = [np.log1p(len(values)), len(values)/(frame[-1]-frame[0]+1), logit.mean(), logit.std(), *np.quantile(logit, [.1, .9]),
                  np.log1p(location.motion.relative_motion(np.column_stack((centers, wh)))),
                  np.linalg.norm(centers[-1]-centers[0])/(lengths.sum()+1e-12), steps.mean(), steps.std(), overlap.mean(),
                  *log_wh.std(0), np.median(log_wh[:, 0]-log_wh[:, 1]), diagonal/np.linalg.norm(values[0][4]), float(key[0].split('/')[0] == 'nanaimo')]
        keys.append(key); features.append(vector)
        quality.append(np.mean([v[3] for v in values])); weights.append(len(values))
    return keys, np.asarray(features), np.asarray(quality), np.asarray(weights)


def retained(rows, tracks, estimates, threshold):
    if threshold is None:
        return tracks
    output = []
    for row, track in zip(rows, tracks):
        keep = np.asarray([estimates[(row['sequence_id'], int(i))] >= threshold for i in track['ids']], dtype=bool)
        output.append({'boxes': track['boxes'][keep], 'ids': track['ids'][keep]})
    return output


def evaluate(p, estimator, deadline):
    rows, det, _ = location.motion.flexible.load_inputs(KEY, 'val', p)
    original = associate(rows, det, p)
    keys, features, _, _ = examples(rows, det, original, False)
    estimates = dict(zip(keys, estimator.predict(features)))
    validation = []
    expected = json.loads((location.RUN/'selection.json').read_text())['all_validation']
    identity = next(r['tracking'] for r in expected if r['condition'] == KEY and r['rule'] == 'location')
    for threshold in p['thresholds']:
        tracks = retained(rows, original, estimates, threshold)
        metrics = rgb.tracking.score(rows, tracks)
        if threshold is None and max(abs(metrics[m]-identity[m]) for m in location.KEYS) > 1e-12:
            raise RuntimeError('Identity quality control did not reproduce prior validation')
        validation.append({'threshold': threshold, 'tracking': metrics})
    ref = p['reference_validation']
    hota = max(validation, key=lambda r: tuple(r['tracking'][m] for m in location.KEYS))
    balanced = max(validation, key=lambda r: (-max(max(0, ref[m]-r['tracking'][m]) for m in location.KEYS), *(r['tracking'][m] for m in location.KEYS)))
    selected = [{'policy': name, 'validation': value} for name, value in [('HOTA_first', hota), ('balanced_tracking', balanced)]]
    write(RUN/'selection.json', {'locked_before_new_test_quality_evaluation': True, 'created_unix': time.time(),
                               'all_validation': validation, 'selected_roles': selected})
    rows, det, _ = location.motion.flexible.load_inputs(KEY, 'test', p)
    original = associate(rows, det, p)
    keys, features, _, _ = examples(rows, det, original, False)
    estimates = dict(zip(keys, estimator.predict(features)))
    tests = []
    for threshold in dict.fromkeys(r['validation']['threshold'] for r in selected):
        tracks = retained(rows, original, estimates, threshold)
        folder = RUN/'test'/('identity' if threshold is None else f'quality_{threshold:g}')
        folder.mkdir(parents=True)
        torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': tracks, 'quality_scores': estimates, 'threshold': threshold}, folder/'tracks.pt')
        result = {'threshold': threshold, 'tracking': rgb.tracking.score(rows, tracks),
                  'official_LOI_counts': location.motion.flexible.count_tracks(rows, tracks, folder, 'test')}
        write(folder/'result.json', result); tests.append(result)
        guard(deadline)
    roles = [{**r, 'test': next(t for t in tests if t['threshold'] == r['validation']['threshold'])} for r in selected]
    write(RUN/'results.json', {'status': 'COMPLETE', 'selected_roles': roles, 'unique_test_results': tests,
          'author_reference_retained': str(location.BASE/'cfc26_track_confirmation_v1/results.json'), 'AP50_computed': False})


def pipeline():
    p = json.loads((RUN/'protocol.json').read_text())
    info = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    if int(info.split(',')[0]) < 24576:
        raise RuntimeError(f'GPU0 actual admission failed: {info}')
    start = time.monotonic()
    deadline = time.time()+3600
    write(RUN/'launch_receipt.json', {'PID': os.getpid(), 'physical_GPU': 0, 'started_unix': time.time(), 'deadline_unix': deadline, 'admission': info})
    rows = infer_training(p, min(deadline, time.time()+2400))
    # Existing loader requires the full split. Here the fixed train subset is the explicit external boundary.
    subset = torch.load(RUN/'train_predictions.pt', map_location='cpu', weights_only=False)
    if subset['row_ids'] != [r['row_id'] for r in rows]:
        raise RuntimeError('Fixed train quality row identities differ')
    detections = []
    for row, d in zip(rows, subset['native_detections']):
        boxes = torch.as_tensor(d['boxes']).clone(); scores = torch.as_tensor(d['scores'])
        h, w = row['actual_shape_HWC'][:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        valid = (scores > .1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        keep = nms(boxes, scores, .25)[:300]
        detections.append(np.column_stack((boxes[keep].numpy(), scores[keep].numpy())).astype(np.float64))
    tracks = associate(rows, detections, p)
    keys, x, y, weight = examples(rows, detections, tracks, True)
    np.savez_compressed(RUN/'train_track_examples.npz', features=x, quality=y, observations=weight,
                        sequence=np.asarray([k[0] for k in keys]), track_id=np.asarray([k[1] for k in keys]))
    estimator = HistGradientBoostingRegressor(**{k: v for k, v in p['estimator'].items() if k != 'class'})
    estimator.fit(x, y, sample_weight=weight)
    with (RUN/'quality_estimator.pkl').open('wb') as f:
        pickle.dump(estimator, f)
    write(RUN/'fit_receipt.json', {'status': 'COMPLETE', 'track_examples': len(y), 'matched_observation_fraction': float(np.average(y, weights=weight)),
          'zero_quality_tracks': int((y == 0).sum()), 'majority_correct_tracks': int((y >= .5).sum()),
          'fit_iterations': estimator.n_iter_, 'feature_names': FEATURES, 'new_encoder_updates': 0})
    evaluate(p, estimator, deadline)
    size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
    if size > 128*1024**2:
        raise RuntimeError('Actual track quality disk cap')
    write(RUN/'completion.json', {'status': 'COMPLETE', 'seconds': time.monotonic()-start, **guard(deadline),
          'new_entity_bytes': size, 'new_encoder_updates': 0, 'new_neural_training': 0, 'teacher_forwards': 0,
          'train_frames_encoded': len(rows), 'val_test_new_neural_forwards': 0, 'quality_fit_iterations': estimator.n_iter_})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        pipeline()
