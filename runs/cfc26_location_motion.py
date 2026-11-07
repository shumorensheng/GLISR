"""One location-conditioned motion calibration; shared image model unchanged."""
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

from runs import cfc26_motion_prior as motion

BASE = motion.BASE
RUN = BASE/'cfc26_location_motion_v1'
KEYS = ['HOTA', 'IDF1', 'MOTA']
write = motion.write


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Locationmotion unit already prepared')
    prior = BASE/'cfc26_motion_prior_v1'
    p = json.loads((prior/'protocol.json').read_text())
    grouped = defaultdict(lambda: defaultdict(list))
    for row in motion.flexible.mature.tracking.load_rows('train'):
        location = row['sequence_id'].split('/')[0]
        for ann in row['annotations']:
            x, y, w, h = ann['bbox']
            grouped[location][(row['sequence_id'], ann['track_id'])].append((x+w/2, y+h/2, w, h))
    fitted = {}
    for location, tracks in grouped.items():
        data = np.asarray([(motion.relative_motion(v), len(v)) for v in tracks.values() if len(v) >= 5])
        values, weights = data.T
        order = np.argsort(values, kind='stable')
        cumulative = np.cumsum(weights[order])/weights.sum()
        cutoff = float(values[order[np.searchsorted(cumulative, .01)]])
        fitted[location] = {'cutoff': cutoff, 'eligible_tracks': len(values),
                            'eligible_observations': int(weights.sum()),
                            'training_observation_fraction_below': float(weights[values < cutoff].sum()/weights.sum())}
    conditions = ['author_public', 'GLISR_native256_balanced', 'GLISR_native128_terminal']
    p.update({'status': 'FIXED_BEFORE_NEW_VALIDATION_FILTERING', 'created_unix': time.time(),
        'detectors': {k: p['detectors'][k] for k in conditions},
        'association': {k: p['association'][k] for k in conditions},
        'confirmations': {k: p['confirmations'][k] for k in conditions},
        'expected_identity_validation': {k: p['expected_identity_validation'][k] for k in conditions},
        'rules': ['identity', 'global', 'location'],
        'global_cutoff': json.loads((prior/'train_prior.json').read_text())['cutoff'],
        'location_priors': fitted,
        'definition': 'Same empiricalweightedlowest1percenttrainmotionquantile, fittedseparatelyfor existingEel/Nanaimolocationmetadata. Ruleidentity/global/locationonly; sameimageweightsandfixedassociation. QuietGTfishvalid, ALLoriginalGTscored. Knownlocationrequired; no unknownsitefallback/physicalspeedclaim.',
        'validation': 'Threeexistingpipelines*3rules/full1000validationframes=9. HOTA-firstandbalanced3metricrolesselectedbeforetest; noassociation/cutoffgridexpansion.',
        'test': 'Atmost6logicalroles/alloriginal2000frames, uniquecondition/rulescoredonce. AllmetricsandLOIretained; no newneuralforward/training/pixels orGTmodification.',
        'scope': 'Locationmetadata-conditioneddeploymentprior, not image-onlyencoding or independentunseensitegeneralization. Sourceencoder/head shared acrosslocations.',
        'stop': 'Onlyidentity/global/locationrules, actualCPUwall/memory/codefailurevisible; noautomaticnewquantile/location-specifictrainingorarchitecture.'})
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    shutil.copyfile(motion.__file__, RUN/'motion_engine_snapshot.py')
    write(RUN/'train_location_priors.json', fitted)
    write(RUN/'protocol.json', p)


def filtered(rows, tracks, descriptor, rule, p):
    output = []
    for row, t in zip(rows, tracks):
        if rule == 'identity':
            cutoff = 0.
        elif rule == 'global':
            cutoff = p['global_cutoff']
        else:
            cutoff = p['location_priors'][row['sequence_id'].split('/')[0]]['cutoff']
        keep = np.asarray([descriptor[(row['sequence_id'], int(i))] >= cutoff for i in t['ids']], dtype=bool)
        output.append({'boxes': t['boxes'][keep].copy(), 'ids': t['ids'][keep].copy()})
    return output


def original_tracks(key, split, p):
    rows, detections, _ = motion.flexible.load_inputs(key, split, p)
    tracks = motion.flexible.associate(rows, detections, p['association'][key])
    tracks, _ = motion.confirmation.confirmed(rows, tracks, p['confirmations'][key])
    return rows, tracks


def pipeline():
    torch.set_num_threads(4)
    p = json.loads((RUN/'protocol.json').read_text())
    begun = time.monotonic()
    write(RUN/'launch_receipt.json', {'PID': os.getpid(), 'started_unix': time.time(),
                                    'validation_jobs': 9, 'new_GPU_allocations': 0})
    validation = []
    for key in p['detectors']:
        rows, original = original_tracks(key, 'val', p)
        descriptor = motion.descriptors(rows, original)
        for index, rule in enumerate(p['rules']):
            tracks = filtered(rows, original, descriptor, rule, p)
            metrics = motion.flexible.mature.tracking.score(rows, tracks)
            if index == 0 and max(abs(metrics[m]-p['expected_identity_validation'][key][m]) for m in KEYS) > 1e-12:
                raise RuntimeError('Locationidentity didnot reproduceoriginalvalidation')
            validation.append({'condition': key, 'rule': rule, 'tracking': metrics})
    selected = []
    reference = p['reference_validation']
    for key in p['detectors']:
        choices = [r for r in validation if r['condition'] == key]
        hota = max(choices, key=lambda r: tuple(r['tracking'][m] for m in KEYS))
        balanced = max(choices, key=lambda r: (-max(max(0, reference[m]-r['tracking'][m]) for m in KEYS),
                                               *(r['tracking'][m] for m in KEYS)))
        for policy, chosen in [('HOTA_first', hota), ('balanced_tracking', balanced)]:
            selected.append({'condition': key, 'policy': policy, 'validation': chosen})
    write(RUN/'selection.json', {'locked_before_test_association': True, 'created_unix': time.time(),
                                'selected_roles': selected, 'all_validation': validation})
    test = []
    for key in p['detectors']:
        rows, original = original_tracks(key, 'test', p)
        descriptor = motion.descriptors(rows, original)
        rules = list(dict.fromkeys(r['validation']['rule'] for r in selected if r['condition'] == key))
        for rule in rules:
            tracks = filtered(rows, original, descriptor, rule, p)
            folder = RUN/'test'/key/rule
            folder.mkdir(parents=True)
            torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': tracks, 'rule': rule}, folder/'tracks.pt')
            record = {'condition': key, 'rule': rule, 'tracking': motion.flexible.mature.tracking.score(rows, tracks),
                      'official_LOI_counts': motion.flexible.count_tracks(rows, tracks, folder, 'test')}
            test.append(record);write(folder/'result.json', record)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if time.monotonic()-begun > 600 or rss > 3*1024**3:
            raise RuntimeError('ActuallocationmotionCPUresourcecap')
    roles = [{**r, 'test': next(t for t in test if t['condition'] == r['condition'] and
                               t['rule'] == r['validation']['rule'])} for r in selected]
    size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
    if size > 64*1024**2:
        raise RuntimeError('Locationmotionretaineddiskcap')
    write(RUN/'results.json', {'status': 'COMPLETE', 'selected_roles': roles,
                              'unique_test_results': test, 'AP50_computed': False})
    write(RUN/'completion.json', {'status': 'COMPLETE', 'validation_evaluations': len(validation),
          'unique_test_evaluations': len(test), 'logical_roles': len(roles), 'new_GPU_allocations': 0,
          'new_neural_training': 0, 'train_fitted_motion_cutoffs': len(p['location_priors']),
          'CUDA_initialized': torch.cuda.is_initialized(), 'seconds': time.monotonic()-begun,
          'RSS_bytes': rss, 'new_entity_bytes': size})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        pipeline()
