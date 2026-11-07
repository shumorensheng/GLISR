"""One CPU observation-quality filter with an explicit separate LOI branch."""
import argparse
from collections import Counter, defaultdict
import json
import os
import pickle
import resource
import shutil
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

from runs import cfc26_track_quality as quality

RUN = quality.location.BASE/'cfc26_observation_quality_v1'
PREVIOUS = quality.location.BASE/'cfc26_track_quality_fulltrain_v1'
KEY = quality.KEY
PHOTO = [f'{a}_channel_{b}' for a in ['box_mean', 'box_std', 'box_q90', 'box_minus_context_mean'] for b in range(3)]
LOCAL = ['observation_logit', 'center_x_over_W', 'center_y_over_H', 'log_width', 'log_height',
         'log_width_minus_track_median', 'log_height_minus_track_median', 'previous_observation_IoU',
         'next_observation_IoU', 'previous_relative_step']
write = quality.write


def guard(deadline):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if time.time() > deadline or rss > 3*1024**3:
        raise RuntimeError(f'Actual observation-quality CPU cap: RSS={rss}')
    return rss


def prepare():
    RUN.mkdir(exist_ok=False)
    parent = json.loads((PREVIOUS/'protocol.json').read_text())
    if json.loads((quality.location.BASE/'cfc26_track_quality_appearance_v1/completion.json').read_text())['status'] != 'COMPLETE':
        raise RuntimeError('Preceding appearance control incomplete')
    p = {k: parent[k] for k in ['association', 'confirmations', 'global_cutoff', 'location_priors', 'reference_validation', 'source_model', 'thresholds']}
    p.update({'status': 'FIXED_BEFORE_NEW_PIXEL_READ_OR_FIT', 'created_unix': time.time(),
        'detectors': {KEY: {**parent['detectors'][KEY], 'train': str(PREVIOUS/'train_predictions.pt')}},
        'features': quality.FEATURES+PHOTO+LOCAL,
        'features_definition': 'Original16whole-trackfeatures broadcast to eachobservedbox,12currentbox/contextRGBstatistics identicaltoprecedingcontrol,10currentgeometry/confidence/neighbourfeatures. Completeofflinepredictedtrack only; endpointneighbours are self, previousstep normalizedbytrackmediandiagonal. NoGT in featurecalculation.',
        'training_target': 'Original7000trainGT, one-to-oneframeHungarianIoU>=.5 originalobservedboxes. Onebinarytarget perobservation, unitweight. Pertrack mean must exactly reproduce473previousrealqualitytargets. No newlabels or ignoredGT.',
        'estimator': {'loss': 'log_loss', 'learning_rate': .1, 'max_iter': 100, 'max_leaf_nodes': 7, 'min_samples_leaf': 10,
                      'l2_regularization': 1., 'early_stopping': False, 'random_state': 0},
        'thresholds': parent['thresholds'],
        'tracking_output': 'Keepindividualobservations abovequalitythreshold, preserve originalnativeboxes/IDs; retaintrackswithatleast5remainingobservations, sameexistingminimum. No interpolation/relinking/boxrefinement/GTfilter.',
        'count_output': 'Separateexplicitbranch: officialLOIcounter consumes originalmotion-qualifiedcompletepredictedtracks BEFORE observationqualityfilter. QualityIoU fidelity andlinecrossingquantity are differentoutputrequirements. Samecountbranch forallworkingpoints, not permetricwinnercombination.',
        'validation': 'All1000originalframes, identity/.25/.5/.75 only. LockHOTA-first, balanced3metric, MOTA-first rolesbeforetest. Originalidentityvalidationmustreproduceexistinglocationprofile. Fixedcountbranch evaluatedonce.',
        'test': 'All2000originalframes. Eachselectedunique trackingthreshold scoredonce; oneactualsharedcountbranch. Saveboth actualtrackingandcountinputtracks. Reportfullvectors/roleswithoutmixing.',
        'scope': 'Supervisedofflineknownlocationpostprocessing, sameimmutableGLISR detections/neuralweights. RawRGBappearancefeatures andCPUclassificationexplicit; noencoder-onlygain or freshblindholdout claim.',
        'budgets': {'global_seconds': 600, 'CPU_threads': 4, 'RSS_GiB': 3, 'disk_MiB': 64, 'GPU_allocations': 0, 'new_neural_forwards': 0},
        'stop': 'Onefixed38featureclassifier,100iterations,3thresholds plusidentity. Actualtime/RSS/disk/codefailurevisible; no automatic extra features/thresholds/retraining/downloads.'})
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    shutil.copyfile(quality.__file__, RUN/'track_engine_snapshot.py')
    write(RUN/'protocol.json', p)
    nodes = [
        {'id': 'title', 'label': 'GLISR：逐观测质量过滤与独立过线计数', 'x': 680, 'y': 30, 'width': 1280, 'height': 40, 'font_size': 22, 'fill': '#FFFFFF', 'stroke': '#FFFFFF'},
        {'id': 'input', 'label': '已有 GLISR 检测缓存\n原始存储 RGB 图像\n无新增神经网络前向', 'x': 190, 'y': 155, 'width': 310, 'height': 115, 'fill': '#EAF3FF'},
        {'id': 'tracks', 'label': '原关联 + 地点运动过滤\n完整预测轨迹 / 原生框坐标', 'x': 630, 'y': 155, 'width': 350, 'height': 105},
        {'id': 'label', 'label': '仅训练：逐帧真实 GT 匹配\n7000 帧 / 473 条原轨迹\n100 次迭代小型分类器', 'x': 1100, 'y': 155, 'width': 350, 'height': 115},
        {'id': 'quality', 'label': '每次观测 38 项特征\n局部外观 / 运动 / 轨迹上下文\n验证锁定阈值 → 保留观测', 'x': 390, 'y': 365, 'width': 420, 'height': 120, 'fill': '#E9F7EF'},
        {'id': 'mot', 'label': '跟踪输出：质量过滤后的观测\n原 ID 不变 / 至少 5 次保留观测\n原 GT：HOTA / IDF1 / MOTA', 'x': 970, 'y': 365, 'width': 450, 'height': 120},
        {'id': 'count', 'label': '计数输出：运动过滤后的完整轨迹 → 官方 LOI 计数\n此分支在逐观测质量过滤之前分出，所有工作点共享', 'x': 680, 'y': 560, 'width': 1140, 'height': 80, 'fill': '#FFF5E5'}]
    edges = [{'from': a, 'to': b} for a, b in [('input', 'tracks'), ('tracks', 'quality'), ('label', 'quality'), ('quality', 'mot'), ('tracks', 'count')]]
    write(RUN/'architecture_spec.json', {'canvas': {'width': 1360, 'height': 630}, 'nodes': nodes, 'edges': edges})


def observation_features(rows, detections, tracks, training, stage, deadline):
    keys, base, means, _ = quality.examples(rows, detections, tracks, training)
    context = dict(zip(keys, base))
    grouped = defaultdict(list)
    for i, (row, track) in enumerate(zip(rows, tracks)):
        for box, identity in zip(track['boxes'], track['ids']):
            grouped[(row['sequence_id'], int(identity))].append((i, box))
    local = {}
    for key, values in grouped.items():
        boxes = np.stack([v[1] for v in values])
        wh = boxes[:, 2:]-boxes[:, :2]
        log_wh = np.log(wh)
        centers = (boxes[:, 2:]+boxes[:, :2])/2
        diagonal = np.median(np.linalg.norm(wh, axis=1))
        for j, (row_i, box) in enumerate(values):
            prev, nxt = max(0, j-1), min(len(values)-1, j+1)
            iou_previous = quality.rgb.tracking.box_iou([box], [boxes[prev]])[0, 0]
            iou_next = quality.rgb.tracking.box_iou([box], [boxes[nxt]])[0, 0]
            local[(row_i, key[1])] = np.r_[log_wh[j], log_wh[j]-np.median(log_wh, axis=0), iou_previous, iou_next,
                np.linalg.norm(centers[j]-centers[prev])/diagonal]
    features, labels, row_index, identities = [], [], [], []
    observed_target = defaultdict(list)
    pixel_frames = 0
    begun = time.monotonic()
    for i, (row, det, track) in enumerate(zip(rows, detections, tracks)):
        boxes = track['boxes']
        matched = np.zeros(len(boxes), dtype=bool)
        if training and len(boxes) and row['annotations']:
            gt = np.asarray([a['bbox'] for a in row['annotations']], dtype=np.float64)
            gt[:, 2:] += gt[:, :2]
            overlap = quality.rgb.tracking.box_iou(gt, boxes)
            ii, jj = linear_sum_assignment(np.where(overlap >= .5, 1-overlap, 1e6))
            matched[jj[overlap[ii, jj] >= .5]] = True
        if len(boxes):
            with Image.open(row['path']) as image:
                pixels = np.asarray(image.convert('RGB'), dtype=np.float32)/255
            h, w = pixels.shape[:2]
            overlap = quality.rgb.tracking.box_iou(boxes, det[:, :4])
            confidence = det[overlap.argmax(1), 4]
            pixel_frames += 1
            for j, (box, identity) in enumerate(zip(boxes, track['ids'])):
                identity = int(identity)
                key = (row['sequence_id'], identity)
                x1, y1 = np.floor(box[:2]).astype(int)
                x2, y2 = np.ceil(box[2:]).astype(int)
                region = pixels[y1:y2, x1:x2].reshape(-1, 3)
                center = (box[:2]+box[2:])/2
                wh = box[2:]-box[:2]
                a, b = np.maximum(0, np.floor(center-wh)).astype(int)
                c, d = np.minimum([w, h], np.ceil(center+wh)).astype(int)
                surround = pixels[b:d, a:c].reshape(-1, 3)
                photo = np.r_[region.mean(0), region.std(0), np.quantile(region, .9, axis=0), region.mean(0)-surround.mean(0)]
                score = np.clip(confidence[j], 1e-6, 1-1e-6)
                features.append(np.r_[context[key], photo, np.log(score/(1-score)), center/[w, h], local[(i, identity)]])
                labels.append(matched[j]); row_index.append(i); identities.append(identity)
                observed_target[key].append(bool(matched[j]))
        if (i+1) % 256 == 0:
            guard(deadline)
            write(RUN/'progress.json', {'stage': stage, 'frames': i+1, 'total_frames': len(rows), 'seconds': time.monotonic()-begun, 'pixel_frames': pixel_frames})
    x = np.stack(features)
    if training:
        if not np.array_equal(np.asarray([np.mean(observed_target[k]) for k in keys]), means):
            raise RuntimeError('Observation labels do not reproduce original473track quality targets')
        with np.load(PREVIOUS/'train_track_examples.npz') as previous:
            if keys != list(zip(previous['sequence'].tolist(), previous['track_id'].tolist())) or not np.array_equal(means, previous['quality']):
                raise RuntimeError('Observationquality changed source GT target identities')
    cache = {'features': x, 'row_index': np.asarray(row_index, dtype=np.int32), 'track_id': np.asarray(identities, dtype=np.int32)}
    if training:
        cache['GT_matched'] = np.asarray(labels, dtype=bool)
    np.savez_compressed(RUN/f'{stage}_features.npz', **cache)
    write(RUN/f'{stage}_feature_receipt.json', {'stage': stage, 'frames': len(rows), 'observations': len(x), 'features': x.shape[1],
          'pixel_frames': pixel_frames, 'seconds': time.monotonic()-begun, 'original_train_track_targets_exact': training, 'RSS_bytes': guard(deadline)})
    return x, np.asarray(labels, dtype=bool)


def retained(rows, tracks, probabilities, threshold):
    if threshold is None:
        return tracks
    output = []
    cursor = 0
    counts = Counter()
    for row, track in zip(rows, tracks):
        n = len(track['ids'])
        keep = probabilities[cursor:cursor+n] >= threshold
        cursor += n
        value = {'boxes': track['boxes'][keep], 'ids': track['ids'][keep]}
        output.append(value)
        counts.update((row['sequence_id'], int(i)) for i in value['ids'])
    for row, track in zip(rows, output):
        keep = np.asarray([counts[(row['sequence_id'], int(i))] >= 5 for i in track['ids']], dtype=bool)
        track['boxes'], track['ids'] = track['boxes'][keep], track['ids'][keep]
    return output


def pipeline():
    torch.set_num_threads(4)
    p = json.loads((RUN/'protocol.json').read_text())
    start = time.monotonic(); deadline = time.time()+600
    write(RUN/'launch_receipt.json', {'PID': os.getpid(), 'started_unix': time.time(), 'deadline_unix': deadline, 'GPU_allocations': 0})
    rows, det, _ = quality.location.motion.flexible.load_inputs(KEY, 'train', p)
    tracks = quality.associate(rows, det, p)
    x, y = observation_features(rows, det, tracks, True, 'train', deadline)
    estimator = HistGradientBoostingClassifier(**p['estimator'])
    estimator.fit(x, y)
    with (RUN/'quality_estimator.pkl').open('wb') as f:
        pickle.dump(estimator, f)
    write(RUN/'fit_receipt.json', {'status': 'COMPLETE', 'observations': len(x), 'GT_matched': int(y.sum()), 'features': x.shape[1],
          'iterations': estimator.n_iter_, 'original473track_targets_exact': True, 'new_neural_training': 0, 'RSS_bytes': guard(deadline)})
    rows, det, _ = quality.location.motion.flexible.load_inputs(KEY, 'val', p)
    original = quality.associate(rows, det, p)
    x, _ = observation_features(rows, det, original, False, 'val', deadline)
    probability = estimator.predict_proba(x)[:, 1]
    validation = []
    previous = json.loads((quality.location.RUN/'selection.json').read_text())['all_validation']
    expected = next(r['tracking'] for r in previous if r['condition'] == KEY and r['rule'] == 'location')
    for threshold in p['thresholds']:
        current = retained(rows, original, probability, threshold)
        metrics = quality.rgb.tracking.score(rows, current)
        if threshold is None and max(abs(metrics[k]-expected[k]) for k in quality.location.KEYS) > 1e-12:
            raise RuntimeError('Observationidentity didnot reproduce previous fullvalidation')
        validation.append({'threshold': threshold, 'tracking': metrics})
    count_val = RUN/'val_count_base'; count_val.mkdir()
    validation_counts = quality.location.motion.flexible.count_tracks(rows, original, count_val, 'val')
    reference = p['reference_validation']; keys = quality.location.KEYS
    hota = max(validation, key=lambda r: tuple(r['tracking'][k] for k in keys))
    balanced = max(validation, key=lambda r: (-max(max(0, reference[k]-r['tracking'][k]) for k in keys), *(r['tracking'][k] for k in keys)))
    mota = max(validation, key=lambda r: tuple(r['tracking'][k] for k in ['MOTA', 'HOTA', 'IDF1']))
    selected = [{'policy': name, 'validation': value} for name, value in [('HOTA_first', hota), ('balanced_tracking', balanced), ('MOTA_first', mota)]]
    write(RUN/'selection.json', {'locked_before_test': True, 'created_unix': time.time(), 'all_validation': validation,
                               'selected_roles': selected, 'fixed_count_branch_validation': validation_counts})
    rows, det, _ = quality.location.motion.flexible.load_inputs(KEY, 'test', p)
    original = quality.associate(rows, det, p)
    x, _ = observation_features(rows, det, original, False, 'test', deadline)
    probability = estimator.predict_proba(x)[:, 1]
    count_folder = RUN/'test_count_base'; count_folder.mkdir()
    torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': original, 'role': 'count_before_observation_filter'}, count_folder/'count_input_tracks.pt')
    counts = quality.location.motion.flexible.count_tracks(rows, original, count_folder, 'test')
    tests = []
    for threshold in dict.fromkeys(r['validation']['threshold'] for r in selected):
        current = retained(rows, original, probability, threshold)
        folder = RUN/'test'/('identity' if threshold is None else f'quality_{threshold:g}'); folder.mkdir(parents=True)
        torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': current, 'threshold': threshold}, folder/'tracking_output.pt')
        result = {'threshold': threshold, 'tracking': quality.rgb.tracking.score(rows, current), 'official_LOI_counts': counts,
                  'count_input': str(count_folder/'count_input_tracks.pt'), 'count_branch': 'prefilter complete motion-qualified tracks, sharedallthresholds'}
        write(folder/'result.json', result); tests.append(result); guard(deadline)
    roles = [{**r, 'test': next(t for t in tests if t['threshold'] == r['validation']['threshold'])} for r in selected]
    write(RUN/'results.json', {'status': 'COMPLETE', 'selected_roles': roles, 'unique_test_tracking_results': tests,
          'unique_count_test_branches': 1, 'author_reference': str(quality.location.BASE/'cfc26_track_confirmation_v1/results.json'), 'AP50_computed': False})
    size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
    if size > 64*1024**2:
        raise RuntimeError('Actual observation-quality disk cap')
    write(RUN/'completion.json', {'status': 'COMPLETE', 'seconds': time.monotonic()-start, 'RSS_bytes': guard(deadline),
          'CUDA_initialized': torch.cuda.is_initialized(), 'new_neural_forwards': 0, 'new_neural_training': 0,
          'train_observations': int(len(y)), 'feature_count': x.shape[1], 'new_entity_bytes': size,
          'unique_test_tracking_results': len(tests), 'logical_roles': len(roles), 'unique_count_test_branches': 1})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        pipeline()
