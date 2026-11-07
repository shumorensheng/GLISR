"""Pinned native SigLIP2 vision, preprocessing, and 768-channel target heads."""

import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import roi_align

from runs.cfc_frozen_detector import FrozenDetectorHead


class SiglipFixedRoIPool(nn.Module):
    def forward(self, features, boxes, image_shapes):
        return roi_align(features["0"], boxes, output_size=(7, 7), spatial_scale=1 / 16,
                         sampling_ratio=2, aligned=True)


class SiglipDetectorHead(FrozenDetectorHead):
    def __init__(self):
        nn.Module.__init__(self)
        anchors = AnchorGenerator(((9, 18, 36, 72, 143),), ((.5, 1., 2.),))
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(768, 15), fg_iou_thresh=.7, bg_iou_thresh=.3,
            batch_size_per_image=256, positive_fraction=.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=.7, score_thresh=0.,
        )
        self.roi_heads = RoIHeads(
            SiglipFixedRoIPool(), TwoMLPHead(768 * 7 * 7, 1024), FastRCNNPredictor(1024, 11),
            fg_iou_thresh=.5, bg_iou_thresh=.5, batch_size_per_image=512,
            positive_fraction=.25, bbox_reg_weights=(10., 10., 5., 5.),
            score_thresh=.05, nms_thresh=.5, detections_per_img=100,
        )


def make_detector_head():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return SiglipDetectorHead()


def make_nksid_head():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return nn.Linear(768, 8)


def make_processor(assetdir):
    from transformers import SiglipImageProcessor

    processor = SiglipImageProcessor.from_pretrained(assetdir, local_files_only=True)
    assert processor.do_resize and processor.size == {"height": 224, "width": 224}
    assert processor.resample == 2 and processor.do_rescale and processor.rescale_factor == 1 / 255
    assert processor.do_normalize and processor.image_mean == [.5] * 3 and processor.image_std == [.5] * 3
    assert processor.do_convert_rgb is None
    return processor


def process_grayscale(processor, images):
    repeated_rgb = [image.convert("L").convert("RGB") for image in images]
    return processor(images=repeated_rgb, return_tensors="pt")["pixel_values"]


def route_outputs(outputs):
    patches, pooler = outputs.last_hidden_state, outputs.pooler_output
    assert patches.shape[1:] == (196, 768) and pooler.shape == (patches.shape[0], 768)
    assert patches.dtype == pooler.dtype == torch.float32
    return {"fsod": patches.transpose(1, 2).reshape(-1, 768, 14, 14), "nksid": pooler}


def load_frozen_vision(assetdir):
    from transformers import SiglipVisionConfig, SiglipVisionModel

    assetdir = Path(assetdir)
    original = json.loads((assetdir / "config.json").read_text())
    assert original["model_type"] == "siglip"
    config = SiglipVisionConfig(**original["vision_config"])
    assert config.hidden_size == 768 and config.patch_size == 16 and config.image_size == 224
    model, info = SiglipVisionModel.from_pretrained(
        assetdir, config=config, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, use_safetensors=True, output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["mismatched_keys"] and not info["error_msgs"], info
    assert all(key.startswith("text_model.") or key in ("logit_scale", "logit_bias")
               for key in info["unexpected_keys"]), info["unexpected_keys"]
    total = sum(parameter.numel() for parameter in model.parameters())
    map_count = sum(parameter.numel() for parameter in model.vision_model.head.parameters())
    assert (total, total - map_count, map_count) == (92884224, 85797120, 7087104)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    model.eval().requires_grad_(False)
    return model, info
