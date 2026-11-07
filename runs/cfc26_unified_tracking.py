"""Apply the delivered, fixed tracking pipeline to retained CFC26 predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import time

import numpy as np
import torch

from runs import cfc26_observation_quality as observation

BASE = observation.quality.location.BASE
RUN = BASE/'cfc26_unified_tracking_v1'
PROFILE = BASE/'cfc26_tracking_best_available_v1/best_observation_profile.json'
write = observation.write


def prepare():
    RUN.mkdir(exist_ok=False)
    profile = json.loads(PROFILE.read_text())
    postprocess = json.loads(Path(profile['postprocess_protocol']).read_text())
    jobs, missing = [], []
    for root, pattern in [('glisr_supplement_20261104_v1', 'CFC26_*'),
                          ('cfc26_structural_ablations_v1', '*_s[012]')]:
        for folder in sorted((BASE/root).glob(pattern)):
            if not folder.is_dir():
                continue
            method, seed = folder.name.removeprefix('CFC26_').rsplit('_s', 1)
            prediction = folder/'test_terminal_predictions.pt'
            record = {'id': folder.name, 'method': method, 'seed': int(seed),
                      'endpoint': 'terminal60', 'predictions': str(prediction),
                      'training_run': str(folder), 'status': 'REUSE' if prediction.exists() else 'BLOCKED_MISSING_TERMINAL_PREDICTIONS'}
            (jobs if prediction.exists() else missing).append(record)
    jobs.extend([
        {'id': 'GLISR_native128_terminal10_s0', 'method': 'GLISR_native128', 'seed': 0,
         'endpoint': 'terminal10_after_previous20',
         'predictions': profile['neural_detection']['source_test_predictions'],
         'training_run': str(Path(profile['neural_checkpoint']).parent), 'status': 'REUSE'},
        {'id': 'author_public', 'method': 'author_public', 'seed': None,
         'endpoint': 'published_weights',
         'predictions': str(BASE/'cfc26_author_weights_eval_v1/author_test_predictions.pt'),
         'training_run': 'Published author weights; membership/version qualifications retained', 'status': 'REUSE'},
    ])
    protocol = {
        'status': 'FIXED_BEFORE_EVALUATION', 'created_unix': time.time(),
        'profile': str(PROFILE), 'jobs': jobs, 'missing': missing,
        'association': postprocess['association']['GLISR_native128_terminal'],
        'confirmation': postprocess['confirmations']['GLISR_native128_terminal'],
        'location_priors': postprocess['location_priors'],
        'quality_estimator': profile['quality_estimator'],
        'quality_threshold': profile['quality_threshold'],
        'minimum_observations': 5,
        'quality_definition': 'Same frozen 38-feature/31-leaf classifier and threshold .30 for EVERY condition. No per-model fitting, tuning or test-dependent selection. Classifier was trained on original GLISR_native128 training predictions; transfer to other detector outputs is explicit.',
        'count_branch': 'Official LOI counting from complete location-motion-qualified tracks before quality filtering; actual count evaluation for every condition.',
        'data': str(BASE/'cfc26_tracking_10000_v1'),
        'evaluation': 'All original 2000 test frames/13 segments/9000 GT observations. Original TrackEval HOTA/IDF1/MOTA/FP/FN/IDs; AP50 omitted. Reused local development population, not fresh blind data.',
        'report': 'One unified table, every seed retained, terminal endpoints only. Neural input/head/teacher/training differences retained in row provenance; unified postprocessing does not make these encoder-only causal comparisons. Existing best-only weights are not relabeled terminal.',
        'budget': {'global_seconds': 1800, 'worker_seconds': 300, 'CPU_workers': 4,
                   'CPU_threads_each': 2, 'RSS_each_GiB': 3, 'new_disk_GiB': 1,
                   'GPU_allocations': 0, 'new_neural_forwards': 0, 'new_training': 0},
        'stop': 'Exactly listed available terminal predictions; missing row stays missing. No threshold/tracker grid, training, downloads, retry or altered GT. Failures retain logs and do not erase completed rows.',
    }
    write(RUN/'protocol.json', protocol)
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    write(RUN/'run_manifest.json', {'jobs': jobs, 'missing': missing})


def evaluate(job, deadline):
    torch.set_num_threads(2)
    p = json.loads((RUN/'protocol.json').read_text())
    folder = RUN/job['id']
    folder.mkdir()
    begun = time.monotonic()
    write(folder/'launch.json', {'PID': os.getpid(), 'deadline_unix': deadline, 'job': job})
    flexible = observation.quality.location.motion.flexible
    condition = {'detectors': {job['id']: {'test': job['predictions']}}}
    rows, detections, _ = flexible.load_inputs(job['id'], 'test', condition)
    tracks = flexible.associate(rows, detections, p['association'])
    tracks, _ = observation.quality.location.motion.confirmation.confirmed(rows, tracks, p['confirmation'])
    original_metrics = observation.quality.rgb.tracking.score(rows, tracks)
    motion = observation.quality.location.motion.descriptors(rows, tracks)
    count_tracks = observation.quality.location.filtered(rows, tracks, motion, 'location', p)
    observation.RUN = folder
    observations = sum(len(t['ids']) for t in count_tracks)
    if observations:
        x, _ = observation.observation_features(rows, detections, count_tracks, False, 'test', deadline)
        with Path(p['quality_estimator']).open('rb') as f:
            estimator = pickle.load(f)
        probabilities = estimator.predict_proba(x)[:, 1]
    else:
        probabilities = np.empty(0, dtype=np.float64)
    filtered = observation.retained(rows, count_tracks, probabilities, p['quality_threshold'])
    metrics = observation.quality.rgb.tracking.score(rows, filtered)
    counter = folder/'count'; counter.mkdir()
    counts = flexible.count_tracks(rows, count_tracks, counter, 'test')
    torch.save({'row_ids': [r['row_id'] for r in rows], 'tracking_output': filtered,
                'count_input_tracks': count_tracks, 'quality_scores': probabilities,
                'source_predictions': job['predictions']}, folder/'actual_outputs.pt')
    if metrics['frames'] != 2000 or metrics['sequences'] != 13 or metrics['GT'] != 9000:
        raise RuntimeError('Original CFC26 evaluation population differs')
    rss = observation.guard(deadline)
    result = {**job, 'status': 'COMPLETE', 'tracking': metrics, 'official_LOI_counts': counts,
              'association_before_motion_and_quality': original_metrics,
              'postprocess': str(RUN/'protocol.json'), 'prequality_observations': observations,
              'postquality_observations': sum(len(t['ids']) for t in filtered),
              'PID': os.getpid(), 'seconds': time.monotonic()-begun, 'RSS_bytes': rss,
              'CUDA_initialized': torch.cuda.is_initialized(), 'new_neural_forwards': 0}
    write(folder/'results.json', result)
    print(json.dumps({'complete': job['id'], **{k: metrics[k] for k in ['HOTA', 'IDF1', 'MOTA']}}), flush=True)
    return result


def pipeline():
    p = json.loads((RUN/'protocol.json').read_text())
    begun = time.monotonic(); deadline = time.time()+p['budget']['global_seconds']
    write(RUN/'launch_receipt.json', {'supervisor_PID': os.getpid(), 'started_unix': time.time(),
          'deadline_unix': deadline, 'jobs': len(p['jobs']), 'missing': p['missing'],
          'parallel_CPU_workers': 4, 'GPU_allocations': 0})
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        pending = {pool.submit(evaluate, job, min(deadline, time.time()+300)): job for job in p['jobs'][:4]}
        remaining = iter(p['jobs'][4:])
        while pending:
            future = next(as_completed(pending))
            job = pending.pop(future)
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({'job': job, 'error': repr(error)})
            write(RUN/'queue_state.json', {'complete': [r['id'] for r in results], 'failures': failures,
                  'active': [j['id'] for j in pending.values()], 'expected': len(p['jobs'])})
            next_job = next(remaining, None)
            if next_job is not None and time.time() < deadline:
                pending[pool.submit(evaluate, next_job, min(deadline, time.time()+300))] = next_job
    results.sort(key=lambda r: r['id'])
    complete = len(results) == len(p['jobs']) and not failures
    write(RUN/'results.json', {'status': 'COMPLETE_AVAILABLE_SCOPE' if complete else 'PARTIAL_OR_FAILED',
          'results': results, 'failures': failures, 'missing': p['missing'], 'protocol': str(RUN/'protocol.json')})
    lines = ['# CFC26 统一改进跟踪流程', '', '|模型|seed|端点|HOTA%|IDF1%|MOTA%|计数nMAE|',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in results:
        m = r['tracking']; count = next(c for c in r['official_LOI_counts'] if c['location'] == 'ALL')
        lines.append(f"|{r['method']}|{r['seed']}|{r['endpoint']}|{100*m['HOTA']:.4f}|{100*m['IDF1']:.4f}|{100*m['MOTA']:.4f}|{count['nMAE']:.6f}|")
    lines.extend(['', '所有行使用同一冻结后处理。原模型输入、任务头、教师与训练历史仍保留在各行来源中。',
                  '缺少末轮预测的项目：'+', '.join(j['id'] for j in p['missing'])])
    (RUN/'summary.md').write_text('\n'.join(lines)+'\n')
    size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
    if size > 1024**3:
        raise RuntimeError('Unified tracking disk budget exceeded')
    write(RUN/'completion.json', {'status': 'COMPLETE_AVAILABLE_SCOPE' if complete else 'PARTIAL_OR_FAILED',
          'complete_jobs': len(results), 'requested_available_jobs': len(p['jobs']),
          'missing': p['missing'], 'failures': failures, 'seconds': time.monotonic()-begun,
          'new_entity_bytes': size, 'new_training': 0, 'GPU_allocations': 0})
    if not complete:
        raise RuntimeError('Some declared unified tracking jobs did not complete; see results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        pipeline()
