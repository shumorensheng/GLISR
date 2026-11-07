"""GLISR field adapter with the author's pretrained CFC26 YOLOv5 neck/head.

This adds fish-task pretraining from the released detector; it is not an
original-GLISR-only benchmark. The author's image backbone is never retained
or forwarded. Image encoding remains entirely in GLISR.
"""
from pathlib import Path
import os
import sys

BASE = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))
YOLO_RUN = BASE / 'cfc26_yolov5m_rgb896_v1'
YOLO_SOURCE = (Path(__file__).resolve().parents[1] / 'third_party') / 'yolov5'
AUTHOR_WEIGHTS = BASE / 'cfc26_author_weights_eval_v1/assets/cfc_detect_yolov5m_v0.pt'
sys.path[:0] = [p for p in [str(YOLO_RUN / 'dependencies'), str(YOLO_SOURCE)] if Path(p).exists()]

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from models.common import Conv
from models.yolo import BaseModel
from utils.loss import ComputeLoss


class GLISRYOLOHead(BaseModel):
    """Consume [B,384,32,32]; decode xywh in a logical 256x256 canvas.

    In training, forward returns three native YOLO logit tensors. In eval,
    it returns (predictions, logits), where predictions is [B,4032,6] with
    columns (cx,cy,w,h,objectness,class_probability), before NMS.
    BaseModel supplies native Detect stride/grid device and dtype handling.
    """
    canvas_size = 256

    def __init__(self, weights_path=AUTHOR_WEIGHTS):
        super().__init__()
        source = torch.load(weights_path, map_location='cpu', weights_only=False)['model']
        self.model = nn.ModuleList(list(source.model)[10:]).float().requires_grad_(True)
        self.p3 = Conv(384, 192, 1)
        self.p4 = Conv(384, 384, 1)
        self.p5 = Conv(384, 768, 1)
        detect = self.model[-1]
        self.nc = detect.nc
        self.hyp = yaml.safe_load((YOLO_SOURCE / 'data/hyps/hyp.scratch-low.yaml').read_text())
        self.hyp['box'] *= 3 / detect.nl
        self.hyp['cls'] *= self.nc / 80 * 3 / detect.nl
        self.hyp['obj'] *= (self.canvas_size / 640) ** 2 * 3 / detect.nl
        self.train()

    def forward(self, field):
        # Stand in for the author's P3/P4/SPPF outputs, using only GLISR features.
        saved = {4: self.p3(field),
                 6: self.p4(F.avg_pool2d(field, 2)),
                 9: self.p5(F.avg_pool2d(field, 4))}
        self.pyramid_features = (saved[4], saved[6], saved[9])
        x = saved[9]
        for layer in self.model:
            if layer.f != -1:
                x = (saved[layer.f] if isinstance(layer.f, int) else
                     [x if index == -1 else saved[index] for index in layer.f])
            x = layer(x)
            saved[layer.i] = x
        return x

    def loss(self, raw_training_outputs, targets):
        """Targets: [N,6] (image_index,class=0,normalized cx,cy,w,h).

        Normalize against the full logical canvas, including any image padding.
        Native loss returns (batch_scaled_total, detached [box,obj,cls]); cls
        is zero for this single-class head. Build the criterion on the current
        device because native ComputeLoss keeps device-specific tensors.
        """
        return ComputeLoss(self)(raw_training_outputs, targets)
