import copy

import pytest
import torch
from torch import nn

from sonar_encoder.training.model import SingleChannelEncoder, make_patch_mask, prepare_images


class ContextBlock(nn.Module):
    def forward(self, tokens):
        return tokens + tokens.mean(1, keepdim=True)


class SmallBackbone(nn.Module):
    """The actual token interfaces with cheap image-dependent context mixing."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 384
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14)
        self.mask_token = nn.Parameter(torch.randn(1, 384))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 384))
        self.blocks = nn.ModuleList([ContextBlock()])
        self.norm = nn.LayerNorm(384)

    def prepare_tokens_with_masks(self, images, masks):
        tokens = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        if masks is not None:
            tokens = torch.where(masks[:, :, None], self.mask_token, tokens)
        return torch.cat([self.cls_token.expand(len(images), -1, -1), tokens], dim=1)

    def forward_features(self, images, masks=None):
        tokens = self.prepare_tokens_with_masks(images, masks)
        for block in self.blocks:
            tokens = block(tokens)
        return {"x_norm_patchtokens": self.norm(tokens)[:, 1:]}


def samples_and_masks():
    samples = []
    for height, width in [(32, 64), (64, 32)]:
        samples.append({"intensity": torch.rand(1, height, width),
                        "valid": torch.ones(height, width, dtype=torch.bool),
                        "range_m": torch.linspace(0.6, 10, height),
                        "bearing_rad": torch.linspace(0.7, -0.7, width),
                        "configuration": {"RangeMin": 0.5, "RangeMax": 10.1}})
    return samples, make_patch_mask(2, torch.Generator().manual_seed(7))


@pytest.mark.parametrize("variant", ["R013", "R030", "R031", "R032", "R036", "R037", "R038", "R039"])
def test_frontend_initialization_preserves_shared_predictor_rng(variant):
    torch.manual_seed(11)
    backbone = SmallBackbone()
    state = torch.get_rng_state()
    SingleChannelEncoder(copy.deepcopy(backbone), "R010")
    baseline_predictor = nn.Linear(384, 768)
    torch.set_rng_state(state)
    SingleChannelEncoder(copy.deepcopy(backbone), variant)
    native_predictor = nn.Linear(384, 768)
    torch.testing.assert_close(native_predictor.weight, baseline_predictor.weight, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["R011", "R012", "R013", "R020", "R021", "R030", "R031", "R032",
                                     "R036", "R037", "R038", "R039"])
def test_native_encoder_preserves_initial_response_and_masks_before_active_stem(variant):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    backbone = SmallBackbone()
    baseline = SingleChannelEncoder(copy.deepcopy(backbone), "R010")
    encoder = SingleChannelEncoder(backbone, variant)
    samples, masks = samples_and_masks()
    normalization = {"mean": 0.2, "std": 0.1}
    expected = baseline(prepare_images(samples, normalization, masks), masks)
    initial = encoder.forward_native(samples, normalization, masks)
    torch.testing.assert_close(initial["patch_tokens"], expected["patch_tokens"], rtol=0, atol=0)

    with torch.no_grad():
        if variant in ("R011", "R036"):
            encoder.stem.gamma.fill_(0.2)
            if variant == "R011":
                encoder.stem.polar_pe.scale.fill_(0.1)
        else:
            encoder.stem.residual_scale.fill_(0.1)
            if variant == "R038":
                encoder.stem.gamma.fill_(0.2)
        if variant in ("R013", "R021", "R030", "R031", "R032", "R037", "R039"):
            for gate in encoder.stem.gates:
                gate[2].weight.fill_(0.1)
        if variant == "R037":
            encoder.stem.offset_head[2].weight.fill_(0.1)
    changed = copy.deepcopy(samples)
    for i, sample in enumerate(changed):
        height, width = sample["intensity"].shape[-2:]
        native_mask = masks[i].reshape(16, 16).repeat_interleave(height // 16, 0).repeat_interleave(width // 16, 1)
        sample["intensity"][:, native_mask] += 10
    student = encoder.forward_native(samples, normalization, masks)
    changed_student = encoder.forward_native(changed, normalization, masks)
    torch.testing.assert_close(student["patch_tokens"], changed_student["patch_tokens"], rtol=0, atol=0)
    assert not torch.allclose(encoder.forward_native(samples, normalization, None)["patch_tokens"],
                              encoder.forward_native(changed, normalization, None)["patch_tokens"])
    assert student["patch_tokens"].shape == (2, 256, 384)
    student["patch_tokens"].square().mean().backward()
    assert encoder.backbone.patch_embed.proj.weight.grad is not None
