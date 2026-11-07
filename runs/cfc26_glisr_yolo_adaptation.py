"""GLISR image encoding with a pretrained YOLO task neck/head, fixed tracking."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from torch import nn

from runs.cfc26_glisr_yolo_head import GLISRYOLOHead
from runs import cfc26_glisr_rgb_adaptation as rgb

BASE=rgb.BASE
RUN=BASE/'cfc26_glisr_yolo_adaptation_v1'
ARM='GLISR_YOLO512'
ROOT=Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))


class YOLOTaskHead(nn.Module):
    """Use the existing crop loop's 224-coordinate task-head interface."""
    def __init__(self):
        super().__init__()
        self.core=GLISRYOLOHead()

    def forward(self,field,targets=None):
        raw=self.core(field)
        if self.training:
            labels=[]
            for i,target in enumerate(targets):
                boxes=target['boxes']/224
                xywh=torch.cat(((boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]),dim=1)
                labels.append(torch.cat((boxes.new_full((len(boxes),1),i),
                                         boxes.new_zeros((len(boxes),1)),xywh),dim=1))
            loss,components=self.core.loss(raw,torch.cat(labels))
            self.last_loss_components=components.detach()
            return {'native_yolo_loss':loss}
        from runs.cfc26_author_weights_eval import non_max_suppression
        detected=non_max_suppression(raw[0],pix2width=1.,conf=.05,iou=.25,max_length=0,max_det=300)
        return [{'boxes':d[:,:4]*(224/256),'scores':d[:,4],
                 'labels':torch.ones(len(d),device=d.device,dtype=torch.int64)} for d in detected]


def construct(parent):
    model,_=rgb.native.construct('Rtau',0,parent)
    return model,YOLOTaskHead()


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Independent YOLO-head unit already prepared')
    torch.set_num_threads(4);torch.manual_seed(0)
    old=torch.load(rgb.INIT,map_location='cpu',weights_only=False)
    head=YOLOTaskHead()
    torch.save({'model':old['model'],'head':head.state_dict(),'progress':old['progress']},RUN/'initialization.pt')
    protocol=json.loads((rgb.RUN/'protocol.json').read_text())
    protocol.update({'created_unix':time.time(),'candidates':{ARM:{'tile':512,'GPU':0}},
        'status':'FIXED_BEFORE_MODEL_FORWARD','initialization':str(RUN/'initialization.pt'),
        'initial_GLISR':str(rgb.INIT),'new_task_head':'GLISR32field -> independentP3/P4/P5 projections -> actual author YOLO neck/Detect modules10..24; originalCNN0..9 absent and never forwarded',
        'extra_pretraining':'Published author fish-task neck/Detect weights retained; this is GLISR+pretrainedYOLOhead adaptation, not original-GLISR-only evidence',
        'head_parameters':sum(p.numel() for p in head.parameters()),
        'coordinates':'YOLO decodes native logical256; wrapper exposes224coords to unchanged native crop/evaluation code. Normalized training GT is derived from original224task targets',
        'updates':{'epochs_each':20,'batch':8,'updates_each':17500,'updates_total':17500,'seed':0,'new_source_training':0},
        'optimizer':'Same bodyAdamW1e-5/moduleAdamW1e-4; YOLOneck/headSGD peak.005,momentum.9,wd.0005. All200warmup/cosine1%floor/gradientclip1. Native batch-scaled YOLO CIoU+objectnessBCE; singleclass cls loss0',
        'budgets':{'parallel_GPU_workers':1,'physical_GPUs':[0],'minimum_free_GiB':24,'wall_each_seconds':14400,'global_seconds':14400,'RSS_each_GiB':64,'GPU_reserved_each_GiB':24,'retained_disk_total_GiB':4,'CPU_threads_each':4},
        'selection':'Fixed val epochs5/10/15/20, earliest strict highest validationHOTA; all endpoints retained. Original test only after fixed20, terminal and valbest both reported',
        'other_active_units':'The two RGB384/512 original-head jobs continue their unchanged frozen protocols; none interrupted or shortened'})
    rgb.native.write(RUN/'protocol.json',protocol)
    print('Prepared one17,500update task-head adaptation; no GPU/model forward',flush=True)


def worker(deadline):
    rgb.RUN=RUN
    rgb.worker(ARM,deadline,construct,RUN/'initialization.pt',.005,'HOTA',True,False,'runs.cfc26_glisr_rgb_adaptation')


def pipeline(worker_module,gpu):
    protocol=json.loads((RUN/'protocol.json').read_text())
    info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True).strip()
    if int(info.split(',')[0])<24576:raise RuntimeError(f'GPU{gpu} admission failed:{info}')
    deadline=time.time()+protocol['budgets']['global_seconds']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'OPENBLAS_NUM_THREADS':'4','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1','YOLOv5_AUTOINSTALL':'False',
         'CUBLAS_WORKSPACE_CONFIG':':4096:8','WANDB_MODE':'disabled'}
    started=time.monotonic()
    with (RUN/'stdout.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u','-m',worker_module,'worker',
                '--deadline',str(deadline)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rgb.native.write(RUN/'launch_receipt.json',{'supervisor_PID':os.getpid(),'worker_PID':child.pid,
            'physical_GPU':gpu,'admission':info,'deadline_unix':deadline,'protocol':str(RUN/'protocol.json')})
        try:code=child.wait(timeout=protocol['budgets']['wall_each_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    rgb.native.write(RUN/'actual_wait.json',{'PID':child.pid,'actual_wait_exit':code,'wall_seconds':time.monotonic()-started})
    rgb.native.write(RUN/'completion.json',{'status':'COMPLETE' if code==0 else 'FAILED',
        'actual_wait_exit':code,'source_CNN_inference_forwards':0,'new_source_training':0})
    if code!=0:sys.exit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','worker','pipeline'])
    parser.add_argument('--deadline',type=float);args=parser.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='worker':worker(args.deadline)
    else:pipeline('runs.cfc26_glisr_yolo_adaptation',0)
