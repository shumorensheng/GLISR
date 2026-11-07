"""Finite temporal-confirmation controls on already-completed fish detections."""
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

BASE = flexible.BASE
RUN = BASE/'cfc26_track_confirmation_v1'
PARENT = BASE/'cfc26_teacher_tracker_selection_v1'
write = flexible.write


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Track confirmation unit already prepared')
    p = json.loads((PARENT/'protocol.json').read_text())
    old = json.loads((PARENT/'selection.json').read_text())
    p.update({'status': 'FIXED_BEFORE_NEW_TRACK_FILTER_EVALUATION', 'created_unix': time.time(),
        'association': {r['condition']: r['setting'] for r in old['selected']},
        'baseline_validation': {r['condition']: r['tracking'] for r in old['selected']},
        'confirmation': [{'min_consecutive': run, 'min_coverage': coverage}
                         for run in [1, 3] for coverage in [0., .5]],
        'definition': 'Within eachactualsequence/track, coverage=observations/(lastreleasedframe-firstreleasedframe+1), consecutive=longest run of releasedindices differingby1. Retainwholeoriginaltrack only if bothminima met; no boxmovement/gapfill/frame/GTalteration. Existingobservedlength>=5 is unchanged.',
        'scope': 'Four teacherendpoints plus authorpublic detector, previous validation-selected association fixed. Offlinecomplete-trackfilter; not real-timeor actualacquisition-FPS claim.',
        'purpose': 'Finite temporalconfirmation control on unchanged savedpredictions; no model or associationparameter changes.',
        'selection': 'Perdetector highestvalidationHOTA thenIDF1 thenMOTA among4confirmationrules; declaredorder ties. Lock before newtest association. OverallGLISRselected by the samevalidationtuple.',
        'validation': 'Exactly4confirmationrules*5detectors/full1000validationframes=20. Identityrule must reproduce previouslyretainedvalidationmetrics. Select perdetector byHOTA/IDF1/MOTA, declaredorder ties, before newtestassociation.',
        'test': 'Five selected pipelines/full2000testframes, originalGT/IDs/splits; preservealltracking/countingvectors and comparewithunchangedauthorreference.',
        'new_training': 0, 'new_model_forwards': 0, 'new_GPU_allocations': 0,
        'budgets': {'CPU_threads': 4, 'global_seconds': 300, 'RSS_GiB': 3, 'new_disk_MiB': 64},
        'stop': 'Actualwall/memory/codefailure visible, no automatic retry/newrules/test-based selection',
        'interpretation': 'Tests consecutiveconfirmation andtrackoccupancy as a precisioncontrol; does notassume statictracksarebackground. Allsignedresultsretained.'})
    shutil.copyfile(__file__, RUN/'entry_snapshot.py')
    shutil.copyfile(flexible.__file__, RUN/'association_snapshot.py')
    write(RUN/'protocol.json', p)


def confirmed(rows, tracks, rule):
    observations = defaultdict(list)
    for row, t in zip(rows, tracks):
        for identity in t['ids']:
            observations[(row['sequence_id'], int(identity))].append(row['frame_index'])
    accepted = set()
    lengths = []
    for key, frames in observations.items():
        ordered = np.asarray(sorted(frames))
        coverage = len(ordered)/(ordered[-1]-ordered[0]+1)
        starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 1)+1, len(ordered)]
        longest = int(np.diff(starts).max())
        keep = longest >= rule['min_consecutive'] and coverage >= rule['min_coverage']
        if keep:
            accepted.add(key)
        lengths.append({'sequence': key[0], 'track': key[1], 'observations': len(ordered),
                        'coverage': float(coverage), 'longest_consecutive': longest, 'accepted': bool(keep)})
    output = []
    for row, t in zip(rows, tracks):
        keep = np.asarray([(row['sequence_id'], int(i)) in accepted for i in t['ids']], dtype=bool)
        output.append({'boxes': t['boxes'][keep].copy(), 'ids': t['ids'][keep].copy()})
    return output, lengths


def pipeline():
    torch.set_num_threads(4)
    p = json.loads((RUN/'protocol.json').read_text())
    begun = time.monotonic()
    write(RUN/'launch_receipt.json', {'PID': os.getpid(), 'started_unix': time.time(),
                                    'new_GPU_allocations': 0, 'validation_jobs': 20, 'test_jobs': 5})
    validation = []
    for key in p['detectors']:
        rows, detections, _ = flexible.load_inputs(key, 'val', p)
        original = flexible.associate(rows, detections, p['association'][key])
        for index, rule in enumerate(p['confirmation']):
            tracks, _ = confirmed(rows, original, rule)
            measured = flexible.mature.tracking.score(rows, tracks)
            difference = max(abs(measured[k]-p['baseline_validation'][key][k]) for k in ['HOTA', 'IDF1', 'MOTA'])
            if index == 0 and difference > 1e-12:
                raise RuntimeError('Identityconfirmation didnot reproduce originalvalidation')
            validation.append({'condition': key, 'rule': rule, 'tracking': measured,
                               'baseline_difference': difference if index == 0 else None})
        write(RUN/'progress.json', {'phase': 'validation', 'completed': len(validation), 'total': 20})
    selected = {key: max([r for r in validation if r['condition'] == key],
                         key=lambda x: tuple(x['tracking'][m] for m in ['HOTA', 'IDF1', 'MOTA']))
                for key in p['detectors']}
    primary = max([r for r in selected.values() if r['condition'] != 'author_public'],
                  key=lambda x: tuple(x['tracking'][m] for m in ['HOTA', 'IDF1', 'MOTA']))
    write(RUN/'selection.json', {'locked_before_test_association': True, 'created_unix': time.time(),
          'selected': selected, 'primary_GLISR': primary, 'all_validation': validation})
    results = []
    for key, chosen in selected.items():
        rows, detections, _ = flexible.load_inputs(key, 'test', p)
        original = flexible.associate(rows, detections, p['association'][key])
        tracks, qualities = confirmed(rows, original, chosen['rule'])
        folder = RUN/'test'/key
        folder.mkdir(parents=True)
        torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': tracks,
                    'association': p['association'][key], 'confirmation': chosen['rule']}, folder/'tracks.pt')
        write(folder/'track_confirmation.json', qualities)
        result = {'condition': key, 'confirmation': chosen['rule'], 'association': p['association'][key],
                  'tracking': flexible.mature.tracking.score(rows, tracks),
                  'official_LOI_counts': flexible.count_tracks(rows, tracks, folder, 'test')}
        results.append(result)
        write(folder/'result.json', result)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if time.monotonic()-begun > 300 or rss > 3*1024**3:
            raise RuntimeError('Actual CPUconfirmation resourcecap')
    size = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
    if size > 64*1024**2:
        raise RuntimeError('Confirmation retained disk cap')
    write(RUN/'results.json', {'status': 'COMPLETE', 'test_results': results,
          'validation_selected_primary_GLISR': primary, 'AP50_computed': False})
    write(RUN/'completion.json', {'status': 'COMPLETE', 'validation_evaluations': 20,
          'test_evaluations': 5, 'new_GPU_allocations': 0, 'new_training': 0,
          'seconds': time.monotonic()-begun, 'RSS_bytes': rss, 'new_entity_bytes': size})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        pipeline()
