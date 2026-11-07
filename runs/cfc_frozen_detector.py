from collections import OrderedDict

from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import roi_align


class FixedRoIPool(nn.Module):
    def forward(self, features, boxes, image_shapes):
        return roi_align(
            features["0"], boxes, output_size=(7, 7), spatial_scale=1 / 14,
            sampling_ratio=2, aligned=True,
        )


class FrozenDetectorHead(nn.Module):
    def __init__(self):
        super().__init__()
        anchors = AnchorGenerator(((9, 18, 36, 72, 143),), ((.5, 1., 2.),))
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(384, 15), fg_iou_thresh=.7, bg_iou_thresh=.3,
            batch_size_per_image=256, positive_fraction=.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=.7, score_thresh=0.,
        )
        self.roi_heads = RoIHeads(
            FixedRoIPool(), TwoMLPHead(384 * 7 * 7, 1024), FastRCNNPredictor(1024, 2),
            fg_iou_thresh=.5, bg_iou_thresh=.5, batch_size_per_image=512,
            positive_fraction=.25, bbox_reg_weights=(10., 10., 5., 5.),
            score_thresh=.05, nms_thresh=.5, detections_per_img=100,
        )

    def forward(self, patch_features, targets=None):
        batch_size = patch_features.shape[0]
        image_shapes = [(224, 224)] * batch_size
        images = ImageList(patch_features.new_zeros(batch_size, 1, 224, 224), image_shapes)
        features = OrderedDict([("0", patch_features)])
        proposals, rpn_losses = self.rpn(images, features, targets)
        detections, roi_losses = self.roi_heads(features, proposals, image_shapes, targets)
        return {**rpn_losses, **roi_losses} if self.training else detections


def scale_boxes(boxes, source_hw, target_hw):
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    return boxes * boxes.new_tensor([
        target_w / source_w, target_h / source_h,
        target_w / source_w, target_h / source_h,
    ])
