"""Prepare existing datasets for the user-authorized shared-GSR expansion."""
from collections import defaultdict
import os
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
REAL = BASE.parent/'incoming/real_fls'
DATA = REAL/'all_candidates_20260927/task_validation'
RUN = BASE/'gsr_multidata_expansion_v1'


def relocated(value):
    return str(REAL/value.split('/pmca_real_dataset/', 1)[1])


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    begun = time.monotonic()
    torch.set_num_threads(4)
    RUN.mkdir(exist_ok=True)
    tasks = {}
    parent = json.loads((BASE/'gsr_shared_complete_v1/protocol.json').read_text())
    prepared = json.loads(Path(parent['downstream_data']).read_text())
    for name in ('FSOD', 'Marine', 'UATD', 'NKSID'):
        src = prepared['inputs'][name]
        views = (parent['views'][name] if name != 'NKSID' else
                 {s: str(BASE/'downstream_vit_finetuning_v1/inputs/NKSID'/(s+'.npy')) for s in ('train', 'validation')})
        tasks[name] = {'kind': 'classification' if name == 'NKSID' else 'segmentation' if name == 'Marine' else 'detection',
            'role': 'source_and_downstream' if name in ('FSOD', 'Marine') else 'downstream_only',
            'classes': 8 if name == 'NKSID' else 12 if name == 'Marine' else 10,
            'rows': src['rows'], 'normalization': src['normalization'], 'views': views,
            'existing_population_and_input_reused': True}
        if name == 'FSOD':
            tasks[name]['GT_paths'] = src['GT_paths']
    for name, rel, label_key in (('Xing_detection', 'xing_full/detection_manifest.jsonl', 'label'),
                                 ('LSOD', 'lsod_task_ready/without_conflicting_boxes.jsonl', 'class')):
        rows = jsonl(DATA/rel)
        names = sorted({b[label_key] for r in rows for b in r['boxes']})
        normalized = defaultdict(list)
        for i, r in enumerate(rows):
            boxes = [{'bbox': b['xyxy'] if name == 'Xing_detection' else b['xyxy_original'],
                      'class_id': names.index(b[label_key])+1} for b in r['boxes']]
            h, w = (r['height'], r['width']) if name == 'Xing_detection' else r['shape'][:2]
            normalized['validation' if r['split'] == 'val' else r['split']].append(
                {'row_id': r['stem'] if name == 'Xing_detection' else r['id'], 'image_id': i+1,
                 'image': relocated(r['image_path']), 'height': h, 'width': w, 'boxes_xyxy': boxes})
        tasks[name] = {'kind': 'detection', 'role': 'source_and_downstream', 'classes': len(names),
            'class_names': {str(i+1): n for i, n in enumerate(names)}, 'rows': dict(normalized),
            'source_manifest': str(DATA/rel), 'annotation_policy': 'Original float xyxy retained; no clipping/correction. LSOD uses existing separate conflict-excluded4756 variant.'}
    rows = jsonl(DATA/'xing_full/segmentation_manifest.jsonl')
    normalized = defaultdict(list)
    for r in rows:
        normalized['train' if r['split'] == 'train' else 'validation'].append(
            {'row_id': r['stem'], 'image': relocated(r['image_path']), 'original_mask': relocated(r['mask_path']),
             'width': r['image_size'][0], 'height': r['image_size'][1]})
    tasks['Xing_segmentation'] = {'kind': 'segmentation', 'role': 'source_and_downstream', 'classes': 3,
        'class_names': {'0': 'literal_black', '1': 'literal_cyan_51_221_255', '2': 'literal_pink_250_5_255'},
        'rows': dict(normalized), 'source_manifest': str(DATA/'xing_full/segmentation_manifest.jsonl'),
        'annotation_policy': 'Predict literal annotation colors; no physical-background/exhaustiveness claim or mask-to-box constraint. Detection/segmentation overlap keeps original common split.'}
    cfc = json.loads((BASE/'teacher_context_cfc_frozen_transfer_v1/protocol.json').read_text())
    normalized = {}
    coco_paths = {'train': BASE/'cfc22_protocol_assets_v1/kenai-train-subsampled.original.json',
                  'validation': REAL/'cfc22_trainval/kenai-val.original.json'}
    for split in ('train', 'validation'):
        coco = json.loads(coco_paths[split].read_text()); annotations = defaultdict(list)
        for a in coco['annotations']:
            annotations[a['image_id']].append(a)
        rows = json.loads(Path(cfc['manifests'][split]).read_text())
        normalized[split] = [{'row_id': r['official_filename'], 'image_id': r['image_id'], 'image': r['path'],
                              'height': r['height'], 'width': r['width'], 'annotations': annotations[r['image_id']]} for r in rows]
    tasks['CFC22'] = {'kind': 'detection', 'role': 'source_and_downstream', 'classes': 1,
        'class_names': {'1': 'fish'}, 'rows': normalized, 'GT_paths': {k: str(v) for k, v in coco_paths.items()},
        'normalization': {k: cfc['normalization'][k] for k in ('mean', 'std')},
        'annotation_policy': 'Original subsampled76619/train and30454/val; empty/multibox/negative-y boxes retained. Channel excluded from source and development.'}
    rows = jsonl(DATA/'high_resolution_full/manifest.jsonl')
    train = [r for r in rows if r['split'] == 'train']
    order = np.random.RandomState(42).permutation(len(train)); internal_val = set(order[:69].tolist())
    normalized = defaultdict(list)
    for i, r in enumerate(train):
        normalized['validation' if i in internal_val else 'train'].append(
            {'row_id': r['stem'], 'image': relocated(r['image_path']), 'height': r['height'], 'width': r['width'], 'polygons': r['polygons']})
    normalized['official_validation'] = [{'row_id': r['stem'], 'image': relocated(r['image_path']), 'height': r['height'],
                                         'width': r['width'], 'polygons': r['polygons']} for r in rows if r['split'] == 'val']
    tasks['HighResolution'] = {'kind': 'segmentation', 'role': 'downstream_only', 'classes': 2,
        'class_names': {'0': 'obstacle', '1': 'ground'}, 'rows': dict(normalized),
        'source_manifest': str(DATA/'high_resolution_full/manifest.jsonl'),
        'annotation_policy': 'Project semantic rasterization of original polygons. Unannotated and conflicting overlap pixels ignored255, not invented background. Internal273/69 from official342train; publisher39val reserved fixed-endpoint additional evaluation, not epoch selection; image-level only.'}
    rows = jsonl(DATA/'deeper_diver_full/manifest.jsonl'); normalized = defaultdict(list)
    for r in rows:
        split = 'train' if r['sequence'] in ('set_1', 'set_2') else 'validation' if r['sequence'] == 'set_3' else 'test'
        normalized[split].append({'row_id': r['sequence']+'/'+str(r['frameIndex']), 'image': relocated(r['image']),
            'sequence': r['sequence'], 'height': r['shape'][0], 'width': r['shape'][1], 'point_xy': [r['x'], r['y']]})
    tasks['Diver'] = {'kind': 'point_regression', 'role': 'downstream_only', 'rows': dict(normalized),
        'source_manifest': str(DATA/'deeper_diver_full/manifest.jsonl'),
        'annotation_policy': 'Project sequence split set1+2 fit/set3 development/set4 final. Original single positive diver point, normalized XY regression/native-pixel error; no fabricated boxes/masks/negative labels or physical-object independence.'}
    rows = jsonl(DATA/'cfc26_mini_task_ready/verified_intersection.jsonl'); normalized = defaultdict(list)
    for i, r in enumerate(rows):
        normalized['validation' if r['split'] == 'val' else r['split']].append(
            {'row_id': r['clip']+'/'+str(r['image_id_within_clip']), 'image_id': i+1, 'image': relocated(r['image_path']),
             'height': r['height'], 'width': r['width'], 'annotations': r['annotations']})
    tasks['CFC26_Mini'] = {'kind': 'detection', 'role': 'downstream_diagnostic_only', 'classes': 1,
        'class_names': {'1': 'fish'}, 'rows': dict(normalized),
        'source_manifest': str(DATA/'cfc26_mini_task_ready/verified_intersection.jsonl'),
        'annotation_policy': 'Only existing verified215-frame intersection71/69/75; demonstration diagnostic, not fullCFC26/SOTA/independent capture benchmark.'}
    for name, task in tasks.items():
        for rows in task['rows'].values():
            for row in rows:
                if not Path(row['image']).is_file():
                    raise FileNotFoundError(row['image'])
        task['counts'] = {s: len(r) for s, r in task['rows'].items()}
    view_bytes = sum((task['counts']['train']+task['counts']['validation'])*224*224*4
                     for task in tasks.values() if not task.get('existing_population_and_input_reused', False))
    if view_bytes > 48*1024**3:
        raise RuntimeError('Expanded CPU input views exceed48GiB allocation')
    definition = {'status': 'FINITE_CPU_PREPARATION_BEFORE_EXPANDED_MODEL_PROTOCOL', 'created_unix': time.time(),
        'source_tasks': ['FSOD', 'Marine', 'Xing_detection', 'Xing_segmentation', 'LSOD', 'CFC22'],
        'downstream_only_tasks': ['UATD', 'NKSID', 'HighResolution', 'Diver'],
        'diagnostic_only': ['CFC26_Mini', 'CANOE_revisit', 'UXO_temporal'],
        'CPU_prepare_cap_seconds': 3600, 'new_disk_cap_bytes': 48*1024**3,
        'planned_new_view_payload_bytes': view_bytes, 'tasks': tasks,
        'CANOE': {'role': 'downstream_only_retrieval', 'population': 'existing1311gallery/1136query; original header-exclusion/pose-positive definition',
                  'input_receipt': str(BASE/'canoe_protocol_assets_v1/bounded_input_v1/receipt.json'), 'source_training': False},
        'UXO': {'role': 'downstream_temporal_proxy_diagnostic', 'population': 'existing59770frames/60fit14dev within original74train records',
                'definition': str(BASE/'uxo_frozen_temporal_predictability_v1/protocol.json'), 'no_sonar_object_GT_claim': True, 'source_training': False},
        'simulation': {'role': 'retained_separate_auxiliary_candidate', 'reason': 'Existing intensity/pose/config sources available; no unapproved semantic conversion or new simulator/capture expansion. A concrete auxiliary objective must be frozen before adding source loss.'},
        'not_ready': {'RGBS50': 'Archive listings only, byte access/index not closed; no prepared training frames.',
                      'SOLAQUA': 'Partial selected payload and declared index display only, geometry/pairing unclosed.',
                      'Marine_classification_HDF5': 'Three releases are not independent images; cross-version/cross-role identity must be fixed before combining.'},
        'scientific_scope': 'Expand original fullGSR, not replace it with ordinary pretraining. Original DINO fixed during source stage; keep shared selector/fusion for each independent target. Non-source labels never enter source/selector fitting. Same source snapshot in RAM per new unit, no trained weights on disk. Prior frozen three-task run unchanged. Expanded source will be a fresh fit, not recovery of unavailable disk weights. No source/run/SOTA success inferred from preparation.'}
    write(RUN/'data_plan.json', definition)
    for name, task in tasks.items():
        if task.get('existing_population_and_input_reused', False):
            continue
        folder = RUN/'inputs'/name; folder.mkdir(parents=True)
        task['views'] = {}
        total = squared = 0.; pixels = 0
        for split in ('train', 'validation'):
            rows = task['rows'][split]
            path = folder/(split+'.npy')
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 1, 224, 224))
            for i, row in enumerate(rows):
                with Image.open(row['image']) as image:
                    raw = np.array(image.convert('L'), dtype=np.uint8)
                intensity = np.log1p(raw.astype(np.float32)/255.)
                if split == 'train' and 'normalization' not in task:
                    values = np.log1p(raw.astype(np.float64)/255.)
                    total += values.sum(); squared += np.square(values).sum(); pixels += values.size
                array[i] = F.interpolate(torch.from_numpy(intensity)[None, None], (224, 224), mode='bilinear', align_corners=False, antialias=True)[0].numpy()
                if i % 256 == 0:
                    print(json.dumps({'task': name, 'split': split, 'prepared': i+1, 'rows': len(rows), 'wall_seconds': time.monotonic()-begun}), flush=True)
                    if time.monotonic()-begun > 3600:
                        raise TimeoutError('Expanded data CPU preparation one-hour cap')
            array.flush(); task['views'][split] = str(path)
        if 'normalization' not in task:
            mean = total/pixels
            task['normalization'] = {'mean': float(mean), 'std': float(np.sqrt(squared/pixels-mean**2)),
                'native_train_pixels': pixels, 'definition': 'FP64 native PIL-L/255 log1p pixel-weighted training-only population.'}
        for path in task['views'].values():
            array = np.load(path, mmap_mode='r+')
            for start in range(0, len(array), 64):
                array[start:start+64] = (array[start:start+64]-task['normalization']['mean'])/task['normalization']['std']
            array.flush()
        write(RUN/'data_preparation_progress.json', {'last_complete_dataset': name, 'wall_seconds': time.monotonic()-begun})
    definition['status'] = 'COMPLETE_CPU_INPUTS_MODEL_EXECUTION_PROTOCOL_PENDING'
    definition['wall_seconds'] = time.monotonic()-begun
    definition['new_disk_bytes'] = sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())
    write(RUN/'data_manifest.json', definition)
    write(RUN/'cpu_prepare_receipt.json', {'status': definition['status'], 'counts': {n: t['counts'] for n, t in tasks.items()},
        'wall_seconds': definition['wall_seconds'], 'new_disk_bytes': definition['new_disk_bytes'], 'model_constructions': 0,
        'model_forwards': 0, 'new_SHA_or_smoke_tests': 0, 'CPU_threads': 4,
        'unmaterialized_final_splits': ['LSOD/test', 'Diver/test', 'CFC26_Mini/test', 'HighResolution/official_validation'],
        'derived_masks': 'Mask rasterization is deferred until the task model protocol is explicit; original palette/polygons are retained in data_manifest.'})


if __name__ == '__main__':
    main()
