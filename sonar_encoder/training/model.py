"""Single-channel DINOv2 baseline and native-grid masked prediction."""

import os
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .frontends import NativeStem


def make_patch_mask(batch_size: int, generator: torch.Generator) -> Tensor:
    return torch.stack(
        [torch.randperm(256, generator=generator) < 192 for _ in range(batch_size)]
    )


def prepare_native(sample: dict, normalization: dict, mask: Tensor | None) -> Tensor:
    intensity = sample["intensity"]
    valid = sample["valid"]
    image = torch.zeros_like(intensity)
    image[:, valid] = (intensity[:, valid].log1p() - normalization["mean"]) / normalization["std"]
    if mask is not None:
        height, width = intensity.shape[-2:]
        native_mask = mask.reshape(16, 16).repeat_interleave(height // 16, 0).repeat_interleave(width // 16, 1)
        image.masked_fill_(native_mask.unsqueeze(0), 0)
    return image


def prepare_images(samples: list[dict], normalization: dict, masks: Tensor | None) -> Tensor:
    images = []
    for index, sample in enumerate(samples):
        image = prepare_native(sample, normalization, masks[index] if masks is not None else None)
        images.append(
            F.interpolate(
                image.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        )
    return torch.stack(images)


class SingleChannelEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, variant: str):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        rgb_projection = backbone.patch_embed.proj
        projection = nn.Conv2d(
            1,
            rgb_projection.out_channels,
            kernel_size=rgb_projection.kernel_size,
            stride=rgb_projection.stride,
        )
        with torch.no_grad():
            projection.weight.copy_(rgb_projection.weight.sum(dim=1, keepdim=True))
            projection.bias.copy_(rgb_projection.bias)
        backbone.patch_embed.proj = projection
        self.variant = variant
        if variant != "R010":
            # Added stem parameters must not change the shared predictor initialization.
            with torch.random.fork_rng(devices=[]):
                self.stem = NativeStem(variant)

    def forward(self, images: Tensor, masks: Tensor | None = None) -> dict[str, Tensor]:
        patch_tokens = self.backbone.forward_features(images, masks=masks)[
            "x_norm_patchtokens"
        ]
        return {
            "patch_tokens": patch_tokens,
            "global_features": patch_tokens.mean(dim=1),
        }

    def forward_native(self, samples: list[dict], normalization: dict, masks: Tensor | None):
        device = self.backbone.patch_embed.proj.weight.device
        device_masks = masks.to(device) if masks is not None else None
        if self.variant == "R010":
            return self(prepare_images(samples, normalization, masks).to(device), device_masks)
        images, positions = [], []
        for index, sample in enumerate(samples):
            image = prepare_native(sample, normalization, masks[index] if masks is not None else None).to(device)
            valid = sample["valid"].to(device)
            if self.variant in ("R011", "R012", "R038"):
                image = self.stem(image, valid)
            else:
                image = self.stem(image, valid, sample["range_m"].to(device), sample["bearing_rad"].to(device),
                                  (sample["configuration"]["RangeMin"], sample["configuration"]["RangeMax"]))
            images.append(F.interpolate(image[None], size=(224, 224), mode="bilinear",
                                        align_corners=False, antialias=True)[0])
            if self.variant == "R011":
                positions.append(self.stem.polar_pe(sample["range_m"].to(device), sample["bearing_rad"].to(device)))
        images = torch.stack(images)
        if self.variant != "R011":
            return self(images, device_masks)
        tokens = self.backbone.prepare_tokens_with_masks(images, device_masks)
        tokens = torch.cat([tokens[:, :1], tokens[:, 1:] + torch.stack(positions)], dim=1)
        for block in self.backbone.blocks:
            tokens = block(tokens)
        # The fixed official ViT-S/14 checkpoint has no register tokens.
        patches = self.backbone.norm(tokens)[:, 1:]
        return {"patch_tokens": patches, "global_features": patches.mean(1)}


def build_encoder(code_dir: Path, weights_path: Path, variant: str) -> nn.Module:
    os.environ["XFORMERS_DISABLED"] = "1"
    backbone = torch.hub.load(
        str(code_dir), "dinov2_vits14", source="local", pretrained=False
    )
    backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return SingleChannelEncoder(backbone, variant)


def masked_latent_loss(prediction: Tensor, target: Tensor, masks: Tensor) -> Tensor:
    return F.smooth_l1_loss(
        prediction.float()[masks], target.detach().float()[masks], beta=1.0
    )


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, tau: float) -> None:
    for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
        teacher_parameter.mul_(tau).add_(student_parameter, alpha=1.0 - tau)
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers()):
        teacher_buffer.copy_(student_buffer)
