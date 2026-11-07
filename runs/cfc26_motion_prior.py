"""One train-derived low-motion prediction filter, with all original GT kept."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch

from runs import cfc26_tracking_flexible as flexible
from runs import cfc26_track_confirmation as confirmation

BASE = flexible.BASE
RUN = BASE/'cfc26_motion_prior_v1'
write = flexible.write
METRICS = ['HOTA', 'IDF1', 'MOTA']


def relative_motion(points):
    z = np.asarray(points, dtype=np.float64)
    radius = np.quantile(np.linalg.norm(z[:, :2]-np.median(z[:, :2], axis=0), axis=1), .9)
    return float(radius/np.median(np.linalg.norm(z[:, 2:], axis=1)))


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Motionprior unit already prepared')
    if json.loads((BASE/'cfc26_glisr_norm_trainneck_v1/completion.json').read_text())['status'] != 'COMPLETE':
        raise RuntimeError('Previousfixedneuralunit not complete')
    grouped = defaultdict(list)
    for row in flexible.mature.tracking.load_rows('train'):
        for ann in row['annotations']:
            x, y, w, h = ann['bbox']
            grouped[(row['sequence_id'], ann['track_id'])].append((x+w/2, y+h/2, w, h))
    training_motion = [(relative_motion(p), len(p)) for p in grouped.values() if len(p) >= 5]
    value, weight = np.asarray(training_motion, dtype=np.float64).T
    order = np.argsort(value, kind='stable')
    cumulative = np.cumsum(weight[order])/weight.sum()
    cutoff = float(value[order[np.searchsorted(cumulative, .01)]])
    prior = {'definition': '90percentile centerdistance from median center / medianbboxdiagonal, nativepublishedpixels; no physicalspeed/FPS interpretation',
             'cutoff': cutoff, 'weighted_quantile': .01, 'eligible_GT_tracks': len(value),
             'eligible_GT_observations': int(weight.sum()),
             'training_observation_fraction_strictly_below_cutoff': float(weight[value < cutoff].sum()/weight.sum()),
             'scope': 'Trainmetadata only. QuietfisharevalidGT; filteringpredictionscanloseGTmatches, measuredagainstALLunchangedval/testGT.'}
    write(RUN/'train_prior.json', prior)
    old = json.loads((BASE/'cfc26_native128_tracker_selection_v1/results.json').read_text())
    role = next(r for r in old['selected_roles'] if r['condition'] == 'GLISR_KD_train_neck_terminal20' and r['policy'] == 'balanced_tracking')
    old_protocol = json.loads((BASE/'cfc26_native128_tracker_selection_v1/protocol.json').read_text())
    reference_selection = json.loads((BASE/'cfc26_track_confirmation_v1/selection.json').read_text())['selected']['author_public']
    reference_test = next(r for r in json.loads((BASE/'cfc26_track_confirmation_v1/results.json').read_text())['test_results'] if r['condition'] == 'author_public')
    detectors = {'author_public': old_protocol['detectors']['author_public'],
                 'GLISR_native256_balanced': old_protocol['detectors']['GLISR_KD_train_neck_terminal20']}
    association = {'author_public': reference_test['association'],
                   'GLISR_native256_balanced': role['test']['setting']}
    confirmations = {'author_public': reference_test['confirmation'],
                     'GLISR_native256_balanced': {'min_consecutive': 1, 'min_coverage': 0.}}
    validation_expected = {'author_public': reference_selection['tracking'],
                           'GLISR_native256_balanced': role['validation']['tracking']}
    folders = {
        'GLISR_native128_terminal': (BASE/'cfc26_glisr_native128_v1/GLISR_native128', 10),
        'GLISR_hard_windows_terminal': (BASE/'cfc26_glisr_hard_windows_v1/GLISR_hard_windows128', 10),
        'GLISR_trainnorm_frozen_terminal': (BASE/'cfc26_glisr_channel_norm_v1/KD_frozen_trainnorm', 20),
        'GLISR_trainnorm_train_terminal': (BASE/'cfc26_glisr_norm_trainneck_v1/KD_train_neck_trainnorm', 20)}
    for key, (folder, epoch) in folders.items():
        data = json.loads((folder/'results.json').read_text())
        validation = next(r['result'] for r in data['validation'] if r['epoch'] == epoch)
        detectors[key] = {'val': validation['source_predictions'], 'test': data['terminal_test']['source_predictions']}
        association[key] = data['terminal_test']['tracking_configuration']
        confirmations[key] = {'min_consecutive': 1, 'min_coverage': 0.}
        validation_expected[key] = validation['tracking']
    protocol = {'status': 'FIXED_BEFORE_NEW_VALIDATION_FILTERING', 'created_unix': time.time(),
        'detectors': detectors, 'association': association, 'confirmations': confirmations,
        'expected_identity_validation': validation_expected,
        'motion_cutoffs': [0., cutoff], 'reference_validation': reference_selection['tracking'],
        'definition': 'Originalfixedassociation/confirmation first, thenretainwholepredictedtracks withrelative_motion>=cutoff. Identitycutoff0 versusonefixedtrainpriorcutoff only. No boxmovement/GTremoval/gapfill/sitefitting/parametergrid.',
        'validation': 'Sixretainedpipelines*2rules/all1000valframes=12scores. Identitymustreproduceoriginalmetrics. SelectHOTA-first andbalancedworstpositive3metricdeficit roles, lockedbeforetest.',
        'test': 'Atmost12logicalroles onall2000testframes; eachuniquecondition/cutoffscoredonce. OriginalGT/rows/IDs andLOIcounter, noNN/imageforward or AP50.',
        'selection': 'HOTA-first max(HOTA,IDF1,MOTA); balancedminimaxpositivedeficit versusonefixedstrongauthorvalidationvector, thenHOTA/IDF1/MOTA. Reportallrawmetricswithoutmixingcolumns.',
        'budgets': {'global_seconds': 600, 'CPU_threads': 4, 'RSS_GiB': 3, 'new_disk_MiB': 64,
                    'new_GPU_allocations': 0, 'new_neural_training': 0, 'new_model_forwards': 0},
        'stop': 'Actualwall/memory/codefailurevisible; noautomaticretry/newcutoff/GTfilter or NNjob.',
        'interpretation': 'Conservativeempiricalpredictionprior, not proofthatstationaryfisharebackground. Offlinecomplete-trackdescriptor; not online/speed or acquisitionindependenceclaim.'}
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    shutil.copyfile(flexible.__file__, RUN/'association_snapshot.py')
    write(RUN/'protocol.json', protocol)


def descriptors(rows, tracks):
    grouped = defaultdict(list)
    for row, predicted in zip(rows, tracks):
        for box, identity in zip(predicted['boxes'], predicted['ids']):
            x, y, x2, y2 = box
            grouped[(row['sequence_id'], int(identity))].append(((x+x2)/2, (y+y2)/2, x2-x, y2-y))
    return {key: relative_motion(points) for key, points in grouped.items()}


def filtered(rows, tracks, motion, cutoff):
    output = []
    for row, predicted in zip(rows, tracks):
        keep = np.asarray([motion[(row['sequence_id'], int(i))] >= cutoff for i in predicted['ids']], dtype=bool)
        output.append({'boxes': predicted['boxes'][keep].copy(), 'ids': predicted['ids'][keep].copy()})
    return output


def pipeline():
    torch.set_num_threads(4)
    p = json.loads((RUN/'protocol.json').read_text())
    begun = time.monotonic()
    write(RUN/'launch_receipt.json', {'PID': os.getpid(), 'started_unix': time.time(),
                                    'new_GPU_allocations': 0,
                                    'validation_jobs': len(p['detectors'])*len(p['motion_cutoffs'])})
    validation = []
    for key in p['detectors']:
        rows, detections, _ = flexible.load_inputs(key, 'val', p)
        original = flexible.associate(rows, detections, p['association'][key])
        original, _ = confirmation.confirmed(rows, original, p['confirmations'][key])
        motion = descriptors(rows, original)
        for index, cutoff in enumerate(p['motion_cutoffs']):
            tracks = filtered(rows, original, motion, cutoff)
            metrics = flexible.mature.tracking.score(rows, tracks)
            if index == 0 and max(abs(metrics[m]-p['expected_identity_validation'][key][m]) for m in METRICS) > 1e-12:
                raise RuntimeError('Motionidentity didnot reproduceoriginalvalidation')
            validation.append({'condition': key, 'rule_index': index, 'cutoff': cutoff, 'tracking': metrics})
    selected = []
    reference = p['reference_validation']
    for key in p['detectors']:
        choices = [r for r in validation if r['condition'] == key]
        hota = max(choices, key=lambda r: tuple(r['tracking'][m] for m in METRICS))
        balanced = max(choices, key=lambda r: (-max(max(0, reference[m]-r['tracking'][m]) for m in METRICS),
                                               *(r['tracking'][m] for m in METRICS)))
        for policy, chosen in [('HOTA_first', hota), ('balanced_tracking', balanced)]:
            selected.append({'condition': key, 'policy': policy, 'validation': chosen})
    write(RUN/'selection.json', {'locked_before_test_association': True, 'created_unix': time.time(),
                                'selected_roles': selected, 'all_validation': validation})
    test = []
    for key in p['detectors']:
        rows, detections, _ = flexible.load_inputs(key, 'test', p)
        original = flexible.associate(rows, detections, p['association'][key])
        original, _ = confirmation.confirmed(rows, original, p['confirmations'][key])
        motion = descriptors(rows, original)
        indices = list(dict.fromkeys(r['validation']['rule_index'] for r in selected if r['condition'] == key))
        for index in indices:
            cutoff = p['motion_cutoffs'][index]
            tracks = filtered(rows, original, motion, cutoff)
            folder = RUN/'test'/key/f'rule_{index}'
            folder.mkdir(parents=True)
            torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': tracks, 'cutoff': cutoff}, folder/'tracks.pt')
            record = {'condition': key, 'rule_index': index, 'cutoff': cutoff,
                      'tracking': flexible.mature.tracking.score(rows, tracks),
                      'official_LOI_counts': flexible.count_tracks(rows, tracks, folder, 'test')}
            test.append(record);write(folder/'result.json', record)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if time.monotonic()-begun > 600 or rss > 3*1024**3:
            raise RuntimeError('ActualmotionpriorCPUresourcecap')
    roles = [{**r, 'test': next(t for t in test if t['condition'] == r['condition'] and
                               t['rule_index'] == r['validation']['rule_index'])} for r in selected]
    size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
    if size > 64*1024**2:
        raise RuntimeError('Motionpriorretaineddiskcap')
    write(RUN/'results.json', {'status': 'COMPLETE', 'selected_roles': roles, 'unique_test_results': test,
                              'AP50_computed': False})
    write(RUN/'completion.json', {'status': 'COMPLETE', 'validation_evaluations': len(validation),
          'unique_test_evaluations': len(test), 'logical_roles': len(roles),
          'new_GPU_allocations': 0, 'new_neural_training': 0,
          'train_fitted_motion_cutoffs': len(p['motion_cutoffs'])-1,
          'CUDA_initialized': torch.cuda.is_initialized(),
          'seconds': time.monotonic()-begun, 'RSS_bytes': rss, 'new_entity_bytes': size})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        pipeline()
