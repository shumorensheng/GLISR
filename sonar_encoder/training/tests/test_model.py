import copy

import pytest
import torch
from torch import nn

from sonar_encoder.training.frontends import NativeStem
from sonar_encoder.training.model import (
    SingleChannelEncoder,
    build_encoder,
    make_patch_mask,
    masked_latent_loss,
    prepare_images,
    update_teacher,
)


def test_patch_masks_have_fixed_count_and_reproduce_generator_state():
    generator = torch.Generator().manual_seed(7)
    state = generator.get_state()
    masks = make_patch_mask(4, generator)

    assert masks.shape == (4, 256)
    assert masks.dtype == torch.bool
    assert masks.device.type == "cpu"
    assert torch.equal(masks.sum(dim=1), torch.full((4,), 192))
    assert not torch.equal(masks[0], masks[1])
    generator.set_state(state)
    assert torch.equal(masks, make_patch_mask(4, generator))


@pytest.mark.parametrize("height,width", [(256, 128), (1024, 512)])
def test_native_mask_blocks_changes_before_downsampling_or_upsampling(height, width):
    intensity = torch.linspace(0, 4, height * width).reshape(1, height, width)
    masks = torch.zeros(1, 256, dtype=torch.bool)
    masks[:, 0::2] = True
    changed = intensity.clone()
    changed.reshape(1, 16, height // 16, 16, width // 16)[:, :, :, 0::2, :] += 100
    sample = {"intensity": intensity, "valid": torch.ones(height, width, dtype=torch.bool)}
    changed_sample = {**sample, "intensity": changed}
    normalization = {"mean": 0.7, "std": 0.5}

    original_student = prepare_images([sample], normalization, masks)
    changed_student = prepare_images([changed_sample], normalization, masks)
    assert original_student.shape == (1, 1, 224, 224)
    assert original_student.dtype == torch.float32
    assert original_student.device.type == "cpu"
    assert torch.equal(original_student, changed_student)
    assert not torch.equal(
        prepare_images([sample], normalization, None),
        prepare_images([changed_sample], normalization, None),
    )


def test_prepare_images_normalizes_valid_pixels_and_handles_native_shapes():
    samples = []
    for height, width in [(256, 128), (1024, 512)]:
        intensity = torch.full((1, height, width), 3.0)
        valid = torch.ones(height, width, dtype=torch.bool)
        valid[: height // 2] = False
        intensity[:, : height // 2] = -10.0
        samples.append({"intensity": intensity, "valid": valid})

    images = prepare_images(samples, {"mean": 0.0, "std": 2.0}, None)

    assert images.shape == (2, 1, 224, 224)
    assert torch.isfinite(images).all()
    torch.testing.assert_close(images[:, :, :100], torch.zeros(2, 1, 100, 224))
    torch.testing.assert_close(
        images[:, :, 124:], torch.full((2, 1, 100, 224), 0.6931471805599453)
    )


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 4
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.mask_token = nn.Parameter(torch.full((1, 4), 0.5))
        self.trunk = nn.Linear(4, 4)

    def forward_features(self, images, masks=None):
        patches = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        if masks is not None:
            patches = torch.where(masks.unsqueeze(-1), self.mask_token, patches)
        return {"x_norm_patchtokens": self.trunk(patches)}


def test_single_channel_projection_preserves_rgb_response_and_global_readout():
    backbone = TinyBackbone()
    intensity = torch.randn(2, 1, 8, 8)
    rgb_response = backbone.patch_embed.proj(intensity.repeat(1, 3, 1, 1))
    encoder = SingleChannelEncoder(backbone, "R010")

    assert encoder.embed_dim == 4
    torch.testing.assert_close(backbone.patch_embed.proj(intensity), rgb_response)
    masks = torch.zeros(2, 16, dtype=torch.bool)
    masks[:, 0] = True
    outputs = encoder(intensity, masks)
    expected = backbone.forward_features(intensity, masks)["x_norm_patchtokens"]
    torch.testing.assert_close(outputs["patch_tokens"], expected)
    torch.testing.assert_close(outputs["global_features"], expected.mean(dim=1))


def test_build_encoder_loads_pretrained_trunk_before_projection_conversion(tmp_path, monkeypatch):
    pretrained = TinyBackbone()
    weights_path = tmp_path / "weights.pth"
    torch.save(pretrained.state_dict(), weights_path)
    code_dir = tmp_path / "dinov2"

    def load_local(repo_or_dir, model, *, source, pretrained):
        assert repo_or_dir == str(code_dir)
        assert model == "dinov2_vits14"
        assert source == "local"
        assert pretrained is False
        return TinyBackbone()

    monkeypatch.setattr(torch.hub, "load", load_local)
    encoder = build_encoder(code_dir, weights_path, "R010")

    torch.testing.assert_close(encoder.backbone.trunk.weight, pretrained.trunk.weight)
    torch.testing.assert_close(encoder.backbone.mask_token, pretrained.mask_token)
    torch.testing.assert_close(
        encoder.backbone.patch_embed.proj.weight,
        pretrained.patch_embed.proj.weight.sum(dim=1, keepdim=True),
    )
    torch.testing.assert_close(
        encoder.backbone.patch_embed.proj.bias, pretrained.patch_embed.proj.bias
    )


def test_masked_latent_loss_is_fp32_masked_only_and_stops_target_gradient():
    prediction = torch.tensor([[[2.0, -2.0], [50.0, -50.0]]], dtype=torch.float16)
    prediction.requires_grad_()
    target = torch.zeros_like(prediction, requires_grad=True)
    masks = torch.tensor([[True, False]])

    loss = masked_latent_loss(prediction, target, masks)
    assert loss.dtype == torch.float32
    assert loss.item() == 1.5
    altered = prediction.detach().clone()
    altered[:, 1] += 100
    assert masked_latent_loss(altered, target, masks).item() == loss.item()
    altered[:, 0] = 0
    assert masked_latent_loss(altered, target, masks).item() == 0
    loss.backward()
    assert target.grad is None
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0
    assert torch.count_nonzero(prediction.grad[:, 0]) == 2


def test_teacher_ema_updates_parameters_without_gradients():
    student = nn.Linear(2, 2)
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(2)
        for parameter in teacher.parameters():
            parameter.fill_(10)

    update_teacher(student, teacher, 0.75)

    for parameter in teacher.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 8))
        assert parameter.grad is None
    for parameter in student.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2))


def test_teacher_copies_all_r036_bn_statistics_and_integer_counters_after_parameter_ema():
    student = NativeStem("R036")
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    student(torch.arange(81).float().reshape(1, 9, 9), torch.ones(9, 9, dtype=torch.bool),
            torch.linspace(0.5, 5, 9), torch.linspace(0.8, -0.8, 9))
    expected = {name: buffer.clone() for name, buffer in student.named_buffers()}

    update_teacher(student, teacher, 0.99)

    for name, buffer in teacher.named_buffers():
        torch.testing.assert_close(buffer, expected[name], rtol=0, atol=0)
    assert all(layer.num_batches_tracked == 1 for layer in teacher.modules() if isinstance(layer, nn.BatchNorm2d))
    assert not teacher.training
