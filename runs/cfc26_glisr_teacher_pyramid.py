"""Paired training-only feature transfer to GLISR, scored as fish tracking."""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from runs import cfc26_glisr_rgb_adaptation as rgb
from runs import cfc26_glisr_background_adaptation as background
from runs.cfc26_glisr_teacher_head import TeacherPyramidHead
from runs import cfc26_tracking_flexible as flexible

BASE = rgb.BASE
RUN = BASE/'cfc26_glisr_teacher_pyramid_v1'
INIT = BASE/'cfc26_glisr_fresh_yolo256_v1/initialization.pt'
ARMS = {'KD_frozen_neck': {'tile': 256, 'GPU': 0, 'freeze_neck': True},
        'KD_train_neck': {'tile': 256, 'GPU': 2, 'freeze_neck': False}}
TRACKER = {'tracker': 'OC_SORT', 'threshold': .3, 'use_byte': True,
           'iou_threshold': .001, 'min_hits': 1}


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Teacher-pyramid unit already prepared')
    background.EPOCHS = 20
    shared = RUN/'shared_plan'
    shared.mkdir(exist_ok=True)
    background.prepare_plan(rgb.tracking.load_rows('train'), shared)
    for arm in ARMS:
        folder = RUN/arm
        folder.mkdir(exist_ok=True)
        for name in ['augmentation_plan.npz', 'sampling_receipt.json']:
            (folder/name).symlink_to(shared/name)
    saved = torch.load(INIT, map_location='cpu', weights_only=False)
    initial = {'path': str(INIT), 'previous_target_updates': saved['progress']['updates'],
               'shared_student_head_keys': len(saved['head']),
               'source_projection_before_RGB_restoration': list(saved['model']['encoder.backbone.patch_embed.proj.weight'].shape)}
    del saved
    fresh = json.loads((BASE/'cfc26_glisr_fresh_yolo256_v1/protocol.json').read_text())
    protocol = json.loads((BASE/'cfc26_glisr_background_adaptation_v1/protocol.json').read_text())
    author = next(r for r in json.loads((BASE/'cfc26_tracking_fish_motion_v1/results.json').read_text())['test_results']
                  if r['condition'] == 'author_public')
    protocol.update({
        'status': 'FIXED_BEFORE_FIRST_MODEL_FORWARD', 'created_unix': time.time(),
        'candidates': ARMS, 'initialization': initial, 'initial_GLISR': str(INIT),
        'base_selection': 'Common actual shared six-interface GLISR source6000, originalDINO frozen during source; no CFC26 target updates in initializer. Same original publishedYOLOneck/Detect and seed0 newprojections for botharms.',
        'source_recipe_evidence': fresh['source_recipe_evidence'],
        'initialization_RGB_projection': 'Restore the original pretrainedDINO RGB projection from the shared1channel source before either newtarget run; exact same rule botharms',
        'architecture': 'Same GLISR Rtau .8/K64..256/global12/local6/memory3+6/gate and YOLOhead. Teacher only during training; no authorCNN imageencoding or teacher at inference.',
        'head': 'StudentGLISR field/P3P4P5 projections plus publishedYOLOneck/Detect. Separate frozenCNN featureteacher executes only in training.',
        'new_task_head': 'AuthorCNN0..9 absent from student inference branch; a separate frozen copy supplies training features. No teacherbranch in student checkpoint.',
        'teacher': 'Actual published authorCNN layers0..9, taps4/6/9; FP32/eval/noGrad/trainable0, unregistered from studentoptimizers/checkpoints; actual nativeRGB256crop/255 input. No teacherboxes or pseudolabels.',
        'teacher_loss': 'Original nativeYOLO GTloss +.25/3 * sum_l MSE(studentP_l,teacherP_l)/(mean(teacherP_l^2)+1e-6), l=P3/P4/P5. No coefficient sweep.',
        'neck_control': 'Only paired difference: freeze pretrained neck+Detect weights and neckBNstatistics versus train them. Projections, entireGLISR and fusion train in both; Detect remains in training mode for GTloss. Frozenparams excluded from optimizer.',
        'change_scope': 'Finite teacher-transfer target adaptation/control, not a newalgorithm or originalGLISR-only baseline. Extra fish-taskpretraining and teachercompute explicit.',
        'augmentation': 'Same common original7000trainframes once/epoch, privateorder0/augmentation1731;25percentGT-center sampling on GTpositive frames, otherwise exactoriginalGT-empty256window; sameoriginalGT/flip0.5, no changedlabels.',
        'updates': {'epochs_each': 20, 'batch': 8, 'updates_each': 17500, 'updates_total': 35000,
                    'seed': 0, 'new_source_training': 0},
        'optimizer': 'BodyAdamW1e-5; studentheadSGD.001/.9/.0005 on actualtrainableparams only; GSRmodulesAdamW1e-4.200warmup thenfull17500cosine1percentfloor/gradientclip1. Botharms match.',
        'evaluation': 'Full1000val at5/10/15/20. New independentCPUscore uses lockedOC-SORT high.3/low>.1/IoU.001/min_hits1/age30/use_byteTrue, observedtracklength>=5, nativeclippedboxes/NMS.25/max300. OfficialLOI counter and originalGT unchanged. AP50 notcomputed.',
        'selection': 'Earliest strict highest validationHOTA only. All IDF1/MOTA/counting results retained. Terminal20 and valHOTA-best evaluated on fulloriginal2000test only after20. No test-based selection or midrunqualitychanges.',
        'score_module': 'runs.cfc26_glisr_teacher_pyramid', 'tracking_configuration': TRACKER,
        'working_target_reference': {k: author['tracking'][k] for k in ['HOTA', 'IDF1', 'MOTA']},
        'reference_count_nMAE': author['official_LOI_counts'][0]['nMAE'],
        'reference_evidence': str(BASE/'cfc26_tracking_fish_motion_v1/results.json'),
        'cpu_boundary_receipt': str(RUN/'cpu_receipt.json'),
        'cpu_boundary_receipt_scope': 'Previously completed single CPU teacher/student real-interface check; not GPU/performance validation. Unfrozen constructor mode checked without secondforward.',
        'extra_pretraining': 'Published author fish detector supplies both frozen teacherCNN and initial taskneck/Detect. Its original trainingmembership is unknown; not unseen-pretraining or independent generalization evidence.',
        'budgets': {'parallel_GPU_workers': 2, 'physical_GPUs': [0, 2], 'minimum_free_GiB': 24,
                    'wall_each_seconds': 14400, 'global_seconds': 14400, 'RSS_each_GiB': 64,
                    'GPU_reserved_each_GiB': 24, 'retained_disk_total_GiB': 8, 'CPU_threads_each': 4},
        'other_active_units': 'All priorfixedGPU and CPUtracking units closed; only these two predeclared teacher-transfer arms. No next training is queued.',
        'stop': 'Actualwall/RSS/GPUcap or actualnumeric/code error stops affectedworker visibly; otherdeclaredarm completes itsfixedscope. No automaticretry, LR/lambda/precision/threshold/seed changes.',
        'user_scope_override': 'User allowed changing tracker/count pipeline; AP50 ignored. Current unit fixes one maturetracker for training selection and fair authorreference.'})
    for source in [Path(__file__), Path(rgb.__file__), Path(background.__file__), Path(flexible.__file__),
                   Path('runs/cfc26_glisr_teacher_head.py'), Path('runs/cfc26_glisr_yolo_head.py'),
                   Path('runs/cfc26_glisr_yolo_adaptation.py'), Path('runs/glisr_supplement_variants.py')]:
        shutil.copyfile(source, RUN/(source.stem+'_snapshot.py'))
    rgb.native.write(RUN/'protocol.json', protocol)
    print('Prepared two fixed17500update teacher-transfer arms; no GPU/model forward', flush=True)


def worker(arm, deadline):
    rgb.RUN = RUN
    rgb.EPOCHS = 20
    rgb.UPDATES = 17500
    rgb.make_plan = background.load_plan
    def construct(parent):
        model, _ = rgb.native.construct('Rtau', 0, parent)
        return model, TeacherPyramidHead(ARMS[arm]['freeze_neck'])
    rgb.worker(arm, deadline, construct, INIT, .001, 'HOTA', True, True,
               'runs.cfc26_glisr_teacher_pyramid')


def score(folder):
    config = json.loads((folder/'protocol.json').read_text())
    split = config['splits'][0]
    condition = next(iter(config['detectors']))
    torch.set_num_threads(4)
    rows, detections, source = flexible.load_inputs(condition, split, config)
    tracks = flexible.associate(rows, detections, TRACKER)
    output = folder/'tracking'
    output.mkdir(exist_ok=True)
    torch.save({'row_ids': [r['row_id'] for r in rows], 'tracks': tracks,
                'tracker': TRACKER, 'source_predictions': str(source)}, output/'tracks.pt')
    counts = flexible.count_tracks(rows, tracks, output, split)
    result = {'condition': condition, 'split': split, 'frames': len(rows),
              'tracking': rgb.tracking.score(rows, tracks), 'official_LOI_counts': counts,
              'source_predictions': str(source), 'tracking_configuration': TRACKER,
              'detection': {'AP50': None, 'status': 'NOT_COMPUTED_NOT_AN_OBJECTIVE'},
              'new_model_forwards_in_scoring': 0}
    rgb.native.write(folder/'results.json', {'status': 'COMPLETE', 'results': [result]})
    print(json.dumps({'split': split, **{k: result['tracking'][k] for k in ['HOTA', 'IDF1', 'MOTA']},
                      'count': counts}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'pipeline', 'worker', 'score'])
    parser.add_argument('--arm', choices=list(ARMS))
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'worker':
        worker(args.arm, args.deadline)
    elif args.phase == 'score':
        score(args.folder)
    else:
        rgb.RUN = RUN
        rgb.pipeline('runs.cfc26_glisr_teacher_pyramid')
