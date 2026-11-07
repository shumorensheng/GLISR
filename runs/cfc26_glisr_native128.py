"""One bounded native128 target adaptation from the retained GLISR model."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from runs import cfc26_glisr_teacher_pyramid as teacher
from runs import cfc26_glisr_yolo_adaptation as parent

rgb = parent.rgb
BASE = parent.BASE
RUN = BASE/'cfc26_glisr_native128_v1'
ARM = 'GLISR_native128'
INIT = BASE/'cfc26_glisr_teacher_pyramid_v1/KD_train_neck/latest.pt'
EPOCHS, UPDATES, TILE = 10, 8750, 128
TRACKER = {'tracker': 'OC_SORT', 'threshold': .5, 'use_byte': True,
           'iou_threshold': .05, 'min_hits': 1}


class Native128TeacherHead(teacher.TeacherPyramidHead):
    def forward(self, field, targets=None):
        if self.training:
            self.native_teacher_input_shape = list(self.teacher_images.shape)
            self.teacher_images = F.interpolate(self.teacher_images, (256, 256), mode='bilinear',
                                                align_corners=False, antialias=True)
        return super().forward(field, targets)

    def teacher_report(self):
        report = super().teacher_report()
        report['native_teacher_input_shape'] = self.native_teacher_input_shape
        report['teacher_resize'] = 'OriginalnativeRGB128crop/255 -> FP32bilinearAA256; trainingonly'
        return report


def prepare_plan(rows, folder):
    generator = torch.Generator().manual_seed(0)
    order = np.stack([torch.randperm(len(rows), generator=generator).numpy() for _ in range(EPOCHS)])
    rng = np.random.RandomState(1731)
    xy = np.empty((EPOCHS, len(rows), 2), dtype=np.int32)
    flips = np.empty((EPOCHS, len(rows)), dtype=bool)
    selected = np.full((EPOCHS, len(rows)), -1, dtype=np.int16)
    empty_counts = []
    for i, row in enumerate(rows):
        h, w = row['actual_shape_HWC'][:2]
        positions = np.array([(x, y) for y in rgb.origins(h, TILE) for x in rgb.origins(w, TILE)], dtype=np.int32)
        boxes = np.asarray([a['bbox'] for a in row['annotations']], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        shifted = np.clip(boxes[None]-np.tile(positions, (1, 2))[:, None], 0, TILE)
        positive = ((shifted[:, :, 2] > shifted[:, :, 0]) & (shifted[:, :, 3] > shifted[:, :, 1])).any(1)
        empty = positions[~positive]
        if len(empty) == 0:
            raise RuntimeError(f'No originalGT-empty128window: {row["row_id"]}')
        empty_counts.append(len(empty))
        for epoch in range(EPOCHS):
            if len(boxes) and rng.rand() < .25:
                index = rng.randint(len(boxes))
                box = boxes[index]
                cx = (max(0, box[0])+min(w, box[2]))/2
                cy = (max(0, box[1])+min(h, box[3]))/2
                x = rng.randint(max(0, math.ceil(cx-TILE)), min(w-TILE, math.floor(cx))+1)
                y = rng.randint(max(0, math.ceil(cy-TILE)), min(h-TILE, math.floor(cy))+1)
                selected[epoch, i] = index
                xy[epoch, i] = x, y
            else:
                xy[epoch, i] = empty[rng.randint(len(empty))]
            flips[epoch, i] = rng.rand() < .5
        actual = np.clip(boxes[None]-np.tile(xy[:, i], (1, 2))[:, None], 0, TILE)
        actual_positive = ((actual[:, :, 2] > actual[:, :, 0]) & (actual[:, :, 3] > actual[:, :, 1])).any(1)
        if not np.array_equal(actual_positive, selected[:, i] >= 0):
            raise RuntimeError(f'Native128 sampling/GT intersection differs: {row["row_id"]}')
    np.savez(folder/'augmentation_plan.npz', order=order.astype(np.int32), xy=xy, flip=flips, selected_GT=selected)
    rgb.native.write(folder/'sampling_receipt.json', {'epochs': EPOCHS, 'frames_each': len(rows), 'tile': TILE,
        'private_order_seed': 0, 'augmentation_seed': 1731,
        'positive_crop_count': int((selected >= 0).sum()), 'negative_crop_count': int((selected < 0).sum()),
        'minimum_empty_windows_per_train_image': min(empty_counts),
        'actual_originalGT_intersections_checked': True, 'all_train_frames_each_epoch': True})


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Native128 unit already prepared')
    for name in ['cfc26_glisr_roi_refinement_v1', 'cfc26_projection_bn_calibration_v1',
                 'cfc26_track_confirmation_v1']:
        if json.loads((BASE/name/'completion.json').read_text())['status'] != 'COMPLETE':
            raise RuntimeError(f'Previous bounded unit not complete: {name}')
    folder = RUN/ARM
    folder.mkdir()
    rows = rgb.tracking.load_rows('train')
    prepare_plan(rows, folder)
    saved = torch.load(INIT, map_location='cpu', weights_only=False)
    identity = {'path': str(INIT), 'previous_target_updates': saved['progress']['updates'],
                'previous_target_epoch': saved['progress']['epoch'],
                'RGB_projection_shape': list(saved['model']['encoder.backbone.patch_embed.proj.weight'].shape)}
    del saved
    wh = np.asarray([a['bbox'][2:] for row in rows for a in row['annotations']], dtype=np.float64)
    geometry = {'split': 'train', 'GT_boxes': len(wh),
                'height_quantiles_10_25_50_75_90': np.quantile(wh[:, 1], [.1, .25, .5, .75, .9]).tolist(),
                'height_lt_8_fraction': float((wh[:, 1] < 8).mean()),
                'height_lt_16_fraction': float((wh[:, 1] < 16).mean()),
                'WH_both_lte128_fraction': float((wh.max(1) <= 128).mean()),
                'interpretation': 'Smallboxesrelative to the old8nativepixel localpatch motivate this finite higherresolution path. Not a samplingtheorem or causaldiagnosis.'}
    rgb.native.write(RUN/'train_box_geometry.json', geometry)
    views = {}
    for split in ['val', 'test']:
        counts = [len(rgb.origins(r['actual_shape_HWC'][0], TILE))*len(rgb.origins(r['actual_shape_HWC'][1], TILE))
                  for r in rgb.tracking.load_rows(split)]
        views[split] = {'images': len(counts), 'window_views': sum(counts),
                        'batches16': sum(math.ceil(x/16) for x in counts)}
    reference = next(r for r in json.loads((BASE/'cfc26_track_confirmation_v1/results.json').read_text())['test_results']
                     if r['condition'] == 'author_public')
    protocol = json.loads((teacher.RUN/'protocol.json').read_text())
    protocol.update({
        'status': 'FIXED_BEFORE_FIRST_MODEL_FORWARD', 'created_unix': time.time(),
        'candidates': {ARM: {'tile': TILE, 'GPU': 0, 'freeze_neck': False}},
        'initialization': identity, 'initial_GLISR': str(INIT),
        'base_selection': 'Previously validation-selected KD_train_neck terminal20 under its locked deployment recipe. Reuse that retainedinitializer; not the rejectedRoI/BN/boxcalibration models.',
        'architecture': 'UnchangedGLISR Rtau.8/K64..256/global12/local6/memory3+6/gate with originalYOLOneck/Detect. Sourceimageencoding/countergeometryunchanged; this stage changes nativewindow256->128, and furtheradapts to it.',
        'preprocessing': 'OriginalRGBnative128crop/255 -> independent224/448 bilinearAA/ImageNetnormalizedstudent views. Preserve learned3channelRGBprojection from initializer. No logtransform/channelpermutation or learnednormalization.',
        'initialization_RGB_projection': 'Strictlypreserve actualtrained3channelRGBprojection from prior20epochmodel; no pretrainedprojection restoration.',
        'neck_control': 'Single arm only: originalstudentYOLOneck/Detect andBN train, withbody/projections/GSRmodules; no frozenneck comparison inthisunit.',
        'cpu_boundary_receipt_scope': 'Historicalparent256teacher-interfacecheck only. New128->256teacherresize alignment is measured in the firstcountednormaltrainingstep, not claimed as a newCPUtest.',
        'teacher': 'Same frozenauthorCNN0..9/taps4/6/9/trainingonly/eval/noGrad/unregistered; actualnativeRGB128crop/255 is bilinearAAupsampledto256 for matchingP3/P4/P5canvas. No teacher inference/checkpointcopies.',
        'updates': {'epochs_each': EPOCHS, 'batch': 8, 'updates_each': UPDATES, 'updates_total': UPDATES,
                    'seed': 0, 'new_source_training': 0, 'inherited_target_updates': identity['previous_target_updates']},
        'augmentation': 'Eachoriginal7000trainframesonce/epoch/privateorder0;25percentGTcenter crop on positiveframes, otherwise actualGT-empty128window; flip.5/augmentation1731. Newstage sampler resets, not seamlessresume.',
        'optimizer': 'Resetmoments, preserve allstudentweights. BodyAdamW1e-5/headSGD.001/.9/.0005/GSRAdamW1e-4;200warmup/full8750cosine1percentfloor/clipgrad1, nativeYOLOGTloss+same.25normalizedteacherMSE.',
        'evaluation': 'Full1000val atnewstage5/10;full2000testterminal10 andvalidation-bestafter10. Same nativeaspect/noGTinferencewindowselection, native128slidingstride96+lastorigin; same NMS.25/max300/OC-SORT high.5/IoU.05/minhits1/age30/usebyte/observedlength5/originalLOI.',
        'inference_view_plan_each_endpoint': views,
        'selection': 'EarlieststrictmaximumvalidationHOTAonly, terminalprimary/bestauxiliary. IDF1/MOTA/counting fullyretained; noAP50, no source-loss or test-based stopping.',
        'tracking_configuration': TRACKER, 'score_module': 'runs.cfc26_glisr_native128',
        'working_target_reference': {k: reference['tracking'][k] for k in ['HOTA', 'IDF1', 'MOTA']},
        'reference_count_nMAE': reference['official_LOI_counts'][0]['nMAE'],
        'reference_evidence': str(BASE/'cfc26_track_confirmation_v1/results.json'),
        'change_scope': 'One finite native128+additionaltarget-adaptation path, not a singlefactorcausalablation. Smallercontext/morewindows/extra8750updates/teacherresize all explicit. Originaldata/GT/splits/populationremainunchanged; no newsource training or fullofficialdownload.',
        'cost_projection': 'Metadata-only exactviews/batches above. Two val passes andup to twotest passes cost66320batch16calls, versus10900 for oldoneval+onetest. Oldpairedinfer1061s includes2048calibrationviews andtwoheads; rough conditionalprojection plus8750training/cache/score fits10800s butactualsharedload/activation not promised. No forecast-only stop.',
        'budgets': {'parallel_GPU_workers': 1, 'physical_GPUs': [0], 'minimum_free_GiB': 24,
                    'wall_each_seconds': 10800, 'global_seconds': 10800, 'RSS_each_GiB': 64,
                    'GPU_reserved_each_GiB': 24, 'retained_disk_total_GiB': 4, 'CPU_threads_each': 4},
        'stop': 'Fixed10newepochs8750updates irrespectiveloss/resultdirection. Actualwall/RSS/GPU/numeric/codefailurestopvisibly;noautomaticretry/LR/threshold/native-sizechanges. No newqueuedmodels.',
        'other_active_units': 'All previousfixedGPUandCPUunitscomplete; onlythisone targetadaptation. No formal latencybenchmark while sharedGPUload.',
        'user_scope_override': 'Identicaltracker/countcode restriction removed; focustrackingmetricsnotAP50. Strongauthorreferencepreserved.'})
    for source in [Path(__file__), Path(teacher.__file__), Path(parent.__file__), Path(rgb.__file__),
                   Path('runs/cfc26_glisr_teacher_head.py'), Path('runs/cfc26_glisr_yolo_head.py')]:
        shutil.copyfile(source, RUN/(source.stem+'_snapshot.py'))
    rgb.native.write(RUN/'protocol.json', protocol)


def worker(deadline):
    rgb.RUN = RUN
    rgb.EPOCHS, rgb.UPDATES = EPOCHS, UPDATES
    rgb.make_plan = teacher.background.load_plan
    def construct(protocol):
        model, _ = rgb.native.construct('Rtau', 0, protocol)
        model.encoder.backbone.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14)
        return model, Native128TeacherHead(False)
    rgb.worker(ARM, deadline, construct, INIT, .001, 'HOTA', False, True, 'runs.cfc26_glisr_native128')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'worker', 'pipeline', 'score'])
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'worker':
        worker(args.deadline)
    elif args.phase == 'score':
        teacher.TRACKER = TRACKER
        teacher.score(args.folder)
    else:
        parent.RUN, parent.ARM = RUN, ARM
        parent.pipeline('runs.cfc26_glisr_native128', 0)
