import copy
import math

import pytest
import torch
from torch import nn

from sonar_encoder.training.frontends import NativeStem
from sonar_encoder.training.train import optimizer_update


@pytest.mark.parametrize("variant,parameters", [("R036", 41121), ("R037", 1816),
                                               ("R038", 334), ("R039", 334)])
def test_strong_controls_have_frozen_capacity_and_zero_scale_identity(variant, parameters):
    torch.set_num_threads(1)
    stem = NativeStem(variant)
    z = torch.randn(1, 33, 41)
    valid = torch.ones(33, 41, dtype=torch.bool)
    valid[10, 10] = False
    z[:, ~valid] = 0
    ranges = torch.linspace(0.25, 5, 33)
    bearings = torch.linspace(0.8, -0.8, 41)

    assert torch.equal(stem(z, valid, ranges, bearings, (0.1, 5.1)), z)
    assert sum(p.numel() for p in stem.parameters() if p.requires_grad) == parameters


@pytest.mark.parametrize("variant", ["R037", "R039"])
def test_new_sampling_controls_preserve_r021_kernel_and_gate_initialization(variant):
    torch.manual_seed(11)
    baseline = NativeStem("R021")
    torch.manual_seed(11)
    stem = NativeStem(variant)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(stem.state_dict()[name], value, rtol=0, atol=0)


def test_r036_relative_xy_weights_and_channel_major_neighbor_products():
    stem = NativeStem("R036").eval()
    z = (torch.arange(49).reshape(1, 7, 7).float() + 1) / 10
    ranges = torch.linspace(1, 2.5, 7)
    bearings = torch.linspace(0.3, -0.3, 7)
    valid = torch.ones(7, 7, dtype=torch.bool)
    products = []
    hook = stem.aggregation[0].register_forward_pre_hook(lambda _, args: products.append(args[0].detach()))
    with torch.no_grad():
        for parameter in stem.parameters():
            parameter.zero_()
        for layer in stem.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.fill_(1)
                layer.running_var.fill_(1 - layer.eps)
        stem.input_features[0].weight[:, 0, 1, 1] = torch.arange(1, 65)
        stem.geometry_mlp[0].weight[0, 0] = 1
        stem.geometry_mlp[0].bias[0] = 1
        stem.geometry_mlp[0].weight[1, 1] = 1
        stem.geometry_mlp[0].bias[1] = 1
        stem.geometry_mlp[2].weight[:, 0] = torch.arange(1, 65)
        stem.geometry_mlp[2].weight[:, 1] = 2 * torch.arange(1, 65)
        stem.geometry_mlp[2].weight[-1, :2] *= -1
        stem.geometry_mlp[2].bias.fill_(3)
        stem.aggregation[2].weight[0, 9 * 2 + 0, 0, 0] = 2
        stem.aggregation[2].weight[0, 9 * 5 + 7, 0, 0] = 3
        stem.aggregation[5].weight[0, 0, 0, 0] = 1
        stem.gamma.fill_(1)

    output = stem(z, valid, ranges, bearings)
    hook.remove()

    expected = torch.empty(64, 9)
    for channel in range(64):
        for neighbor, (di, dj) in enumerate((di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)):
            dx = ranges[3 + di] * bearings[3 + dj].cos() - ranges[3] * bearings[3].cos()
            dy = ranges[3 + di] * bearings[3 + dj].sin() - ranges[3] * bearings[3].sin()
            sign = -1 if channel == 63 else 1
            expected[channel, neighbor] = (channel + 1) * z[0, 3 + di, 3 + dj] * (sign * (channel + 1) * (3 + dx + 2 * dy) + 3)
    torch.testing.assert_close(products[0][0, :, 3, 3], expected.flatten())
    torch.testing.assert_close(output[0, 3, 3], z[0, 3, 3] + 2 * expected[2, 0] + 3 * expected[5, 7])
    assert len([layer for layer in stem.modules() if isinstance(layer, nn.BatchNorm2d)]) == 3


def test_r036_complete_local_support_radius_floor_and_live_residual_scale():
    torch.manual_seed(7)
    stem = NativeStem("R036").eval()
    z = torch.ones(1, 9, 9)
    ranges = torch.arange(9).float() / 10 + 0.25
    bearings = torch.linspace(0.8, -0.8, 9)
    valid = torch.ones(9, 9, dtype=torch.bool)
    stem(z, valid, ranges, bearings).sum().backward()
    assert stem.gamma.grad.abs() > 0
    assert torch.count_nonzero(stem.aggregation[-1].weight) > 0
    with torch.no_grad():
        stem.gamma.fill_(1)
    output = stem(z, valid, ranges, bearings)
    assert output[0, 3, 3] != 1
    assert output[0, 2, 3] == 1
    assert output[0, 3, 1] == 1
    assert output[0, 3, 7] == 1
    valid[1, 1] = False
    assert stem(z, valid, ranges, bearings)[0, 3, 3] == 1


def test_r037_actual_center_displacements_are_differentiable_and_require_head_support():
    ranges = torch.arange(65).float() / 8 + 1
    bearings = 0.8 - torch.arange(129).float() / 50
    z = (100 * torch.arange(65)[:, None] + torch.arange(129)[None, :]).float()[None]
    valid = torch.ones(65, 129, dtype=torch.bool)
    torch.manual_seed(11)
    baseline = NativeStem("R021")
    torch.manual_seed(11)
    stem = NativeStem("R037")
    with torch.no_grad():
        for model in (baseline, stem):
            model.weights.zero_()
            model.weights[:, 2] = 1
            model.residual_scale.fill_(1)
    torch.testing.assert_close(stem(z, valid, ranges, bearings, (0.9, 9.1))[0, 32, 64],
                               baseline(z, valid, ranges, bearings, (0.9, 9.1))[0, 32, 64], rtol=0, atol=0)
    with torch.no_grad():
        stem.offset_head[2].bias[[2, 7]] = math.atanh(0.5)

    output = stem(z, valid, ranges, bearings, (0.9, 9.1))
    torch.testing.assert_close(output[0, 32, 64], torch.tensor(6626.75))
    output[0, 32, 64].backward()
    torch.testing.assert_close(stem.offset_head[2].bias.grad[2], torch.tensor(150.0))
    torch.testing.assert_close(stem.offset_head[2].bias.grad[7], torch.tensor(-1.875), rtol=1e-5, atol=1e-5)
    valid[31, 63] = False
    assert stem(z, valid, ranges, bearings, (0.9, 9.1))[0, 32, 64] == z[0, 32, 64]


def test_r037_support_tracks_all_displaced_taps_even_zero_weight_taps():
    stem = NativeStem("R037")
    ranges = torch.arange(33).float() / 8 + 0.25
    bearings = torch.linspace(1, -1, 65)
    z = torch.ones(1, 33, 65)
    valid = torch.ones(33, 65, dtype=torch.bool)
    with torch.no_grad():
        stem.weights.zero_()
        stem.weights[0, 2] = 1
        stem.residual_scale.copy_(torch.tensor([1., 0.]))
    assert stem(z, valid, ranges, bearings, (0.1, 4.5))[0, 5, 32] == 1.5
    with torch.no_grad():
        stem.offset_head[2].bias[0] = -math.atanh(0.9)
    assert stem(z, valid, ranges, bearings, (0.1, 4.5))[0, 5, 32] == 1
    with torch.no_grad():
        stem.offset_head[2].bias.zero_()
    valid[2, 33] = False
    assert stem(z, valid, ranges, bearings, (0.1, 4.5))[0, 5, 32] == 1


def test_r038_pointwise_intensity_mlp_is_active_after_zero_gamma_without_metadata():
    stem = NativeStem("R038")
    z = torch.ones(1, 9, 9)
    valid = torch.ones(9, 9, dtype=torch.bool)
    optimizer = torch.optim.SGD(stem.parameters(), lr=0.01)
    stem(z, valid).sum().backward()
    assert stem.gamma.grad.abs() > 0
    assert torch.count_nonzero(stem.intensity_mlp[2].weight) > 0
    assert torch.count_nonzero(stem.intensity_mlp[0].weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad()
    stem(z, valid).sum().backward()
    for parameter in stem.intensity_mlp.parameters():
        assert torch.count_nonzero(parameter.grad) > 0
    with torch.no_grad():
        stem.residual_scale.zero_()
        stem.intensity_mlp[0].weight.zero_()
        stem.intensity_mlp[0].bias.zero_()
        stem.intensity_mlp[0].weight[0, 0, 0, 0] = 1
        stem.intensity_mlp[2].weight.zero_()
        stem.intensity_mlp[2].weight[0, 0, 0, 0] = 2
        stem.gamma.fill_(1)
    expected = z + 2 * torch.nn.functional.gelu(z)
    torch.testing.assert_close(stem(z, valid), expected)
    assert torch.equal(stem(z, valid, torch.arange(9), torch.arange(9), (0, 9)), stem(z, valid))


def test_r039_fixed_noninteger_bin_interpolation_ignores_recipient_scale_but_keeps_orientation():
    stem = NativeStem("R039")
    with torch.no_grad():
        stem.weights.zero_()
        stem.weights[:, -1] = 1
        stem.residual_scale.fill_(1)
    z = (100 * torch.arange(41)[:, None] + torch.arange(41)[None, :]).float()[None]
    valid = torch.ones(41, 41, dtype=torch.bool)
    ranges = torch.arange(41).float() / 5 + 1
    bearings = 0.4 - torch.arange(41).float() / 50

    output = stem(z, valid, ranges, bearings, (0.9, 9.1))
    torch.testing.assert_close(output[0, 20, 20], torch.tensor(4692.4353))
    changed_config = stem(z, valid, ranges * 2, bearings * 3, (1.8, 18.2))
    torch.testing.assert_close(changed_config, output, rtol=0, atol=0)
    reversed_output = stem(z, valid, ranges, -bearings, (0.9, 9.1))
    torch.testing.assert_close(reversed_output[0, 20, 20], torch.tensor(4700.3853))
    torch.testing.assert_close(output[0, 5, 20], torch.tensor(776.025))
    assert stem(z, valid, ranges / 20, bearings, (0.045, 0.455))[0, 20, 20] == z[0, 20, 20]
    valid[34, 21] = False
    torch.testing.assert_close(stem(z, valid, ranges, bearings, (0.9, 9.1))[0, 20, 20], torch.tensor(3026.025))


@pytest.mark.parametrize("overflow", [False, True])
def test_bn_teacher_buffers_copy_only_when_optimizer_update_succeeds(overflow):
    student = nn.BatchNorm2d(2)
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    before = [buffer.clone() for buffer in teacher.buffers()]
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    output = student(torch.arange(32).float().reshape(1, 2, 4, 4))
    scaler.scale(output.sum() * (float("inf") if overflow else 1)).backward()

    updated, _ = optimizer_update(student, teacher, optimizer, scaler, 0.9, 1)

    assert updated == (not overflow)
    for original, student_buffer, teacher_buffer in zip(before, student.buffers(), teacher.buffers()):
        torch.testing.assert_close(teacher_buffer, original if overflow else student_buffer, rtol=0, atol=0)
