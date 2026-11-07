"""Fixed background-aware GLISR adaptation with unchanged official tracking."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from runs import cfc26_glisr_yolo_adaptation as parent

BASE = parent.BASE
RUN = BASE/'cfc26_glisr_background_adaptation_v1'
ARM = 'BACKGROUND256'
INIT = BASE/'cfc26_glisr_yolo_adaptation_v1/GLISR_YOLO512/best.pt'
EPOCHS = 10
UPDATES = 8750
PREVIOUS_RUNS = ('cfc26_glisr_rgb_adaptation_v1',
                 'cfc26_glisr_yolo_adaptation_v1',
                 'cfc26_glisr_fresh_yolo256_v1')


def prepare_plan(rows, folder):
    generator = torch.Generator().manual_seed(0)
    order = np.stack([torch.randperm(len(rows), generator=generator).numpy()
                      for _ in range(EPOCHS)])
    rng = np.random.RandomState(1731)
    xy = np.empty((EPOCHS, len(rows), 2), dtype=np.int32)
    flips = np.empty((EPOCHS, len(rows)), dtype=bool)
    selected = np.full((EPOCHS, len(rows)), -1, dtype=np.int16)
    empty_counts = []
    for i, row in enumerate(rows):
        h, w = row['actual_shape_HWC'][:2]
        positions = np.array([(x, y) for y in parent.rgb.origins(h, 256)
                              for x in parent.rgb.origins(w, 256)], dtype=np.int32)
        boxes = np.asarray([a['bbox'] for a in row['annotations']], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        shifted = np.clip(boxes[None]-np.tile(positions, (1, 2))[:, None], 0, 256)
        positive = ((shifted[:, :, 2] > shifted[:, :, 0]) &
                    (shifted[:, :, 3] > shifted[:, :, 1])).any(1)
        empty = positions[~positive]
        if len(empty) == 0:
            raise RuntimeError(f'No original-GT-empty256 window: {row["row_id"]}')
        empty_counts.append(len(empty))
        for epoch in range(EPOCHS):
            if len(boxes) and rng.rand() < .25:
                index = rng.randint(len(boxes))
                box = boxes[index]
                cx = (max(0, box[0])+min(w, box[2]))/2
                cy = (max(0, box[1])+min(h, box[3]))/2
                x = rng.randint(max(0, math.ceil(cx-256)), min(w-256, math.floor(cx))+1)
                y = rng.randint(max(0, math.ceil(cy-256)), min(h-256, math.floor(cy))+1)
                selected[epoch, i] = index
                xy[epoch, i] = x, y
            else:
                xy[epoch, i] = empty[rng.randint(len(empty))]
            flips[epoch, i] = rng.rand() < .5
        actual = np.clip(boxes[None]-np.tile(xy[:, i], (1, 2))[:, None], 0, 256)
        actual_positive = ((actual[:, :, 2] > actual[:, :, 0]) &
                           (actual[:, :, 3] > actual[:, :, 1])).any(1)
        if not np.array_equal(actual_positive, selected[:, i] >= 0):
            raise RuntimeError(f'Crop policy and actual original-GT intersections differ: {row["row_id"]}')
    np.savez(folder/'augmentation_plan.npz', order=order.astype(np.int32),
             xy=xy, flip=flips, selected_GT=selected)
    parent.rgb.native.write(folder/'sampling_receipt.json', {
        'epochs': EPOCHS, 'frames_each': len(rows), 'tile': 256,
        'private_order_seed': 0, 'augmentation_seed': 1731,
        'positive_crop_count': int((selected >= 0).sum()),
        'negative_crop_count': int((selected < 0).sum()),
        'actual_GT_intersection_policy_checked_all_planned_crops': True,
        'minimum_empty_windows_per_train_image': min(empty_counts),
        'all_original_train_frames_retained_each_epoch': True})


def load_plan(rows, tile, folder):
    with np.load(folder/'augmentation_plan.npz', allow_pickle=False) as plan:
        return plan['order'], plan['xy'], plan['flip']


def construct(protocol):
    model, head = parent.construct(protocol)
    # This stage loads a genuinely trained RGB checkpoint, not the old gray source.
    model.encoder.backbone.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14)
    return model, head


def prepare():
    for name in PREVIOUS_RUNS:
        previous = BASE/name
        completion = json.loads((previous/'completion.json').read_text())
        if completion['status'] != 'COMPLETE':
            raise RuntimeError(f'Previous fixed wave not complete: {name}')
        launch = json.loads((previous/'launch_receipt.json').read_text())
        if Path('/proc', str(launch['supervisor_PID'])).exists():
            raise RuntimeError(f'Previous supervisor has not exited: {name}')
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Background-adaptation unit already prepared')
    folder = RUN/ARM
    folder.mkdir(exist_ok=True)
    prepare_plan(parent.rgb.tracking.load_rows('train'), folder)
    checkpoint = torch.load(INIT, map_location='cpu', weights_only=False)
    initialization = {'path': str(INIT), 'validation_selected_epoch': checkpoint['progress']['best_epoch'],
        'previous_stage_updates': checkpoint['progress']['updates'],
        'RGB_projection_shape': list(checkpoint['model']['encoder.backbone.patch_embed.proj.weight'].shape),
        'preserve_learned_RGB_projection': True, 'new_optimizer_state': True}
    del checkpoint
    protocol = json.loads((BASE/'cfc26_glisr_yolo_adaptation_v1/protocol.json').read_text())
    protocol.update({
        'status': 'FIXED_BEFORE_FIRST_MODEL_FORWARD', 'created_unix': time.time(),
        'candidates': {ARM: {'tile': 256, 'GPU': 0}},
        'initialization': initialization, 'initial_GLISR': str(INIT),
        'base_selection': 'Highest validationHOTA among retained endpoints of the preceding four fixed configurations: YOLO512 selectedepoch15, validationHOTA0.25664242737962806. No test-based initializer choice',
        'head': 'Existing GLISR field plus publishedYOLO neck/Detect10..24; no authorCNNforward',
        'updates': {'epochs_each': EPOCHS, 'batch': 8, 'updates_each': UPDATES,
                    'updates_total': UPDATES, 'seed': 0, 'new_source_training': 0},
        'augmentation': 'Each of the original7000train frames once/epoch. On GT-positive frames25percent GT-center crop; otherwise choose an actual original-GT-empty native256 window. Flip0.5; seed1731; all GT intersections retained, none relabelled or removed',
        'optimizer': 'Reset optimizers. BodyAdamW1e-5, YOLOheadSGD.001/momentum.9/wd.0005, GSRmodulesAdamW1e-4;200warmup then full8750cosine to1percent, gradientclip1',
        'selection': 'Validation at5/10, earliest strict highest HOTA determines best. IDF1/MOTA and counting retained; terminal10 and validation-selected best test only after10. AP50 does not determine selection or goal acceptance',
        'evaluation': 'Full original1000val afterepochs5/10; full original2000test only after10. Same native256 slidingcoverage, official tracker maxage20/minhits5/IoU.001, low>.1/high>.3/NMS.25/maxdet300 and originalGT LOI counter. No physical-length filtering without calibration',
        'goal': 'Same official tracker/counter/thresholds; improve HOTA/IDF1/MOTA toward author public weight. AP50 retained by unchanged scorer only',
        'change_scope': 'Additional finite target adaptation: trainedRGB checkpoint retained, nativecrop256 and background-aware sampling. Not a single-factor mechanism ablation or source-generalization claim',
        'motivation_evidence': [str(BASE/'cfc26_glisr_rgb_adaptation_v1/crop_occupancy_analysis.json'),
                               str(BASE/'cfc26_glisr_yolo_adaptation_v1/validation_error_analysis.json')],
        'budgets': {'parallel_GPU_workers': 1, 'physical_GPUs': [0], 'minimum_free_GiB': 24,
                    'wall_each_seconds': 14400, 'global_seconds': 14400, 'RSS_each_GiB': 64,
                    'GPU_reserved_each_GiB': 24, 'retained_disk_total_GiB': 4, 'CPU_threads_each': 4},
        'stop': 'Actual wall/RSS/GPU caps or actual numeric/code failure stop visibly; no score-based early stop, LR/threshold rescue or automatic retry',
        'initialization_RGB_projection': 'Strictly load and retain the actual trained3channel RGB projection. Do not restore pretrained projection over it',
        'implementation': ['Existing GLISR+YOLO training/evaluation loop with one explicit preserve-RGB-projection switch',
                           'One fixed background-aware training plan; original tracker/counter untouched'],
        'other_active_units': 'All four previous fixed configurations must have completed and supervisors exited before this preparation',
        'source_training': 0, 'full_dataset_downloads': 0})
    del protocol['working_target_reference']['AP50']
    for source in [Path(__file__), Path(parent.__file__), Path(parent.rgb.__file__),
                   Path(parent.rgb.native.__file__), Path('runs/cfc26_glisr_yolo_head.py'),
                   Path('runs/glisr_supplement_variants.py'), Path('runs/cfc26_author_weights_eval.py')]:
        shutil.copyfile(source, RUN/(source.stem+'_snapshot.py'))
    parent.rgb.native.write(RUN/'protocol.json', protocol)
    print('Prepared one background-aware8750update tracking-target adaptation; no modelforward', flush=True)


def worker(deadline):
    parent.rgb.RUN = RUN
    parent.rgb.EPOCHS = EPOCHS
    parent.rgb.UPDATES = UPDATES
    parent.rgb.make_plan = load_plan
    parent.rgb.worker(ARM, deadline, construct, INIT, .001, 'HOTA', False, False, 'runs.cfc26_glisr_rgb_adaptation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'worker', 'pipeline'])
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'worker':
        worker(args.deadline)
    else:
        parent.RUN = RUN
        parent.ARM = ARM
        parent.pipeline('runs.cfc26_glisr_background_adaptation')
