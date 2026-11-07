"""Author deployment weights and existing detectors with official Fisheye tracking."""
import os
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
RUN = BASE / 'cfc26_author_weights_eval_v1'
ALIGN = (Path(__file__).resolve().parents[1] / 'third_party')
FISHEYE = ALIGN / 'fisheye'
YOLO = ALIGN / 'yolov5'
sys.path[:0] = [p for p in [str(RUN/'dependencies'), str(YOLO), str(FISHEYE), str((Path(__file__).resolve().parents[1] / 'third_party/deps'))] if Path(p).exists()]

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import scale_boxes
from fisheye.boxes import non_max_suppression
from fisheye.configs.inference import TrackerConfig
from fisheye.track.tracker import run_tracker
from fisheye.track.kalman_tracker import KalmanBoxTracker
from fisheye.format import tracker_output_to_dict_rows, dict_rows_to_mot_format
from fisheye.export.inference import MOTExporter
from runs import cfc26_tracking_interface as tracking


def write(path, data):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def infer():
    config = json.loads((RUN/'protocol.json').read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from models.experimental import attempt_load
    model = attempt_load(str(RUN/'assets/cfc_detect_yolov5m_v0.pt'),device='cpu',fuse=True)
    identity = {'model_class': str(type(model)), 'names': model.names,
                'parameters': sum(p.numel() for p in model.parameters()),
                'loader':'Existing official YOLOv5 v7.0 attempt_load, native inference fusion',
                'classes': model.model[-1].nc, 'stride': model.stride.tolist()}
    if identity['classes'] != 1:
        raise RuntimeError(f'Author asset is not the expected single fish detector: {identity}')
    write(RUN/'actual_model.json', identity)
    model = model.float().eval().cuda().half()
    begun = time.monotonic()
    frames = calls = 0
    for split in config['splits']:
        rows = tracking.load_rows(split)
        low_all, high_all = [], []
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row['sequence_id']].append(i)
        for ids in groups.values():
            for offset in range(0, len(ids), config['batch']):
                batch_ids = ids[offset:offset+config['batch']]
                images, shapes = [], []
                for i in batch_ids:
                    row = rows[i]
                    image = cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)
                    h, w = image.shape[:2]
                    if [h, w] != row['actual_shape_HWC'][:2]:
                        raise RuntimeError('Native image/GT size mismatch')
                    ratio = 896 / max(h, w)
                    resized = cv2.resize(image, (int(w*ratio), int(h*ratio)),
                                         interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
                    rh, rw = resized.shape[:2]
                    target = np.ceil(np.array([h/w, 1] if h <= w else [1, w/h])*896/64 + .5).astype(int)*64
                    resized, _, pad = letterbox(resized, target, auto=False, scaleup=False, stride=64)
                    images.append(torch.from_numpy(np.ascontiguousarray(resized.transpose(2,0,1))))
                    shapes.append(((h,w), ((rh/h,rw/w),pad)))
                batch = torch.stack(images).cuda().half()/255
                raw = model(batch)[0]
                # The source width test is merely positivity when max_length=0.
                # Use a dimensionless positive factor; do not invent physical meters.
                low = non_max_suppression(raw.float().cpu(), pix2width=1., conf=.1, iou=.25, max_length=0)
                high = non_max_suppression(raw.float().cpu(), pix2width=1., conf=.3, iou=.25, max_length=0)
                for outputs, destination in [(low,low_all), (high,high_all)]:
                    for det, ((h,w), ratio_pad) in zip(outputs, shapes):
                        scale_boxes(batch.shape[2:], det[:,:4], (h,w), ratio_pad)
                        destination.append({'boxes': det[:,:4].numpy(), 'scores': det[:,4].numpy(),
                                            'labels': np.ones(len(det), dtype=np.int64)})
                frames += len(batch_ids)
                calls += 1
                if calls == 1:
                    write(RUN/'first_counted_batch.json', {'split':split,'row_ids':[rows[i]['row_id'] for i in batch_ids],
                          'actual_input_shape':list(batch.shape),'output_shape':list(raw.shape),
                          'low_detection_counts':[len(x) for x in low], 'high_detection_counts':[len(x) for x in high],
                          'first_batch_included_in_total':True})
                if calls % 20 == 0:
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                    reserved = torch.cuda.max_memory_reserved()
                    if rss > config['budgets']['RSS_bytes'] or reserved > config['budgets']['GPU_reserved_bytes']:
                        raise RuntimeError(f'Inference resource cap: RSS={rss}, GPU={reserved}')
                    write(RUN/'progress.json', {'phase':'author_inference','frames':frames,'calls':calls,
                          'seconds':time.monotonic()-begun,'RSS_bytes':rss,'GPU_reserved_bytes':reserved})
                    print('inference',split,frames,calls,flush=True)
        if len(low_all) != len(rows):
            raise RuntimeError('Prediction population changed')
        torch.save({'row_ids':[r['row_id'] for r in rows], 'native_detections':low_all,
                    'high_detections':high_all,'split':split,'weights':str(RUN/'assets/cfc_detect_yolov5m_v0.pt')},
                   RUN/f'author_{split}_predictions.pt')
    write(RUN/'inference_receipt.json', {'status':'COMPLETE','frames':frames,'calls':calls,
          'seconds':time.monotonic()-begun, 'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'GPU_reserved_bytes':torch.cuda.max_memory_reserved(),'training_updates':0})


def score(run):
    RUN = Path(run)
    from runs.cfc26_small_target_optimization import detection_scores
    torch.set_num_threads(4)
    config = json.loads((RUN/'protocol.json').read_text())
    summaries = []
    for split in config['splits']:
        rows = tracking.load_rows(split)
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row['sequence_id']].append(i)
        for condition, sources in config['detectors'].items():
            saved = torch.load(sources[split],map_location='cpu',weights_only=False)
            if saved['row_ids'] != [r['row_id'] for r in rows]:
                raise RuntimeError(f'{condition} row identities differ')
            output = RUN/split/condition
            output.mkdir(parents=True,exist_ok=True)
            clean = []
            for row,det in zip(rows,saved['native_detections']):
                boxes = torch.as_tensor(det['boxes'],dtype=torch.float32)
                scores = torch.as_tensor(det['scores'],dtype=torch.float32)
                h,w = row['actual_shape_HWC'][:2]
                boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,w)
                boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,h)
                valid = ((scores > .1) & (torch.as_tensor(det['labels']) == 1) &
                         (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1]))
                boxes,scores = boxes[valid],scores[valid]
                keep = nms(boxes,scores,.25)[:300]
                clean.append({'boxes':boxes[keep].numpy(),'scores':scores[keep].numpy(),
                              'labels':np.ones(len(keep),dtype=np.int64)})
            tracks = [None]*len(rows)
            sequence_map = []
            for sequence, ids in groups.items():
                local = [rows[i] for i in ids]
                frame_indices = [r['frame_index'] for r in local]
                if frame_indices != list(range(frame_indices[0],frame_indices[0]+len(ids))):
                    raise RuntimeError('Selected segment is not consecutive in released frame indices')
                h,w = local[0]['actual_shape_HWC'][:2]
                low, high = {}, {}
                for frame, i in enumerate(ids):
                    d = clean[i]
                    low[frame] = np.column_stack((d['boxes']/np.array([w,h,w,h]),d['scores']))
                    if condition == 'author_fisheye':
                        d = saved['high_detections'][i]
                        high[frame] = np.column_stack((d['boxes']/np.array([w,h,w,h]),d['scores']))
                    else:
                        high[frame] = low[frame][low[frame][:,4] > .3]
                KalmanBoxTracker.count = 0
                tracked = run_tracker(low, high, image_meter_width=0, image_meter_height=0,
                          tracking_config=TrackerConfig(max_age=20,min_hits=5,iou_threshold=.001),min_length=-1)
                for i, frame in zip(ids, tracked.frames):
                    tracks[i] = {'boxes':np.array([f.bbox for f in frame.fish]).reshape(-1,4)*np.array([w,h,w,h]),
                                 'ids':np.array([f.id for f in frame.fish],dtype=np.int64)}
                location,clip,run = sequence.split('/')
                name = clip+'_'+run
                pred_dir = output/'infer'/location/name
                gt_dir = output/'gt'/location/name
                frame_link = output/'frames'/location/name
                pred_dir.mkdir(parents=True,exist_ok=True)
                gt_dir.mkdir(parents=True,exist_ok=True)
                frame_link.parent.mkdir(parents=True,exist_ok=True)
                frame_link.symlink_to(Path(local[0]['path']).parent, target_is_directory=True)
                mot_rows = dict_rows_to_mot_format(tracker_output_to_dict_rows(asdict(tracked)),w,h,frame_indices[0])
                MOTExporter(str(pred_dir),filename=name).export(mot_rows)
                with (gt_dir/'gt.txt').open('w') as f:
                    for row in local:
                        for ann in row['annotations']:
                            x,y,bw,bh = ann['bbox']
                            f.write(f'{row["frame_index"]+1},{ann["track_id"]},{x},{y},{bw},{bh},1,-1,-1,-1\n')
                sequence_map.append({'sequence_id':sequence,'export_name':name,'frames':len(ids),
                                     'original_frame_first':frame_indices[0],'original_frame_last':frame_indices[-1]})
            torch.save({'row_ids':saved['row_ids'],'tracks':tracks},output/'tracks.pt')
            write(output/'sequence_map.json',sequence_map)
            with (output/'official_count_stdout.log').open('w') as log:
                command=[sys.executable,str(ALIGN/'CFC26-main/scripts/eval_count_infer.py'),
                         '--infer_dir',str(output/'infer'),'--gt_dir',str(output/'gt'),
                         '--frames_dir',str(output/'frames'),'--output_dir',str(output/'count'), '--count_protocol','LOI']
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            import pandas as pd
            count_rows = pd.read_csv(output/'count/eval_summary.csv').replace({np.nan:None}).to_dict('records')
            counted = sum(int(x['n_clips']) for x in count_rows if x['location'] != 'ALL')
            if counted != len(groups):
                raise RuntimeError(f'Official evaluator skipped segments: {counted}/{len(groups)}')
            summary = {'condition':condition,'split':split,'frames':len(rows),'sequences':len(groups),
                       'detection':detection_scores(rows,clean),'tracking':tracking.score(rows,tracks),
                       'official_LOI_counts':count_rows,'source_predictions':sources[split]}
            summary['by_location'] = {}
            for location in sorted({r['location'] for r in rows}):
                selected = [i for i,r in enumerate(rows) if r['location'] == location]
                local_rows = [rows[i] for i in selected]
                summary['by_location'][location] = {
                    'frames':len(selected),
                    'detection':detection_scores(local_rows,[clean[i] for i in selected]),
                    'tracking':tracking.score(local_rows,[tracks[i] for i in selected])}
            write(output/'result.json',summary)
            summaries.append(summary)
            print(json.dumps({'condition':condition,'split':split,'AP50':summary['detection']['AP50'],
                              'HOTA':summary['tracking']['HOTA'],'count':count_rows}),flush=True)
    write(RUN/'results.json',{'status':'COMPLETE','results':summaries,'protocol':str(RUN/'protocol.json'),
          'scope':'Fixed local CFC26 predictions, original annotations, shared image-only Fisheye tracking and official LOI counting; population, model and input details in protocol'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['infer','score'])
    args = parser.parse_args()
    if args.phase == 'infer':
        infer()
    else:
        score(RUN)
