import math

import pytest
import torch

from sonar_encoder.training.frontends import NativeStem, PolarPE, footprint_descriptor


@pytest.mark.parametrize(
    "variant,parameters", [("R012", 12), ("R013", 334), ("R020", 12), ("R021", 334),
                           ("R030", 334), ("R031", 167), ("R032", 167), ("R011", 14978)]
)
def test_native_stems_start_as_identity_with_declared_parameter_budget(variant, parameters):
    stem = NativeStem(variant)
    z = torch.randn(1, 32, 32)
    valid = torch.ones(32, 32, dtype=torch.bool)
    ranges = torch.linspace(0.5, 5, 32)
    bearings = torch.linspace(0.8, -0.8, 32)

    output = stem(z, valid, ranges, bearings, (0.4, 5.1))

    assert torch.equal(output, z)
    assert sum(p.numel() for p in stem.parameters()) == parameters
    assert sum(p.numel() for p in stem.parameters() if p.requires_grad) == parameters


@pytest.mark.parametrize("variant,axis", [("R030", None), ("R031", 0), ("R032", 1)])
def test_trained_controls_use_corresponding_r021_initial_parameters(variant, axis):
    torch.manual_seed(11)
    baseline = NativeStem("R021")
    torch.manual_seed(11)
    stem = NativeStem(variant)

    if axis is None:
        for name, value in stem.state_dict().items():
            torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
    else:
        torch.testing.assert_close(stem.weights[0], baseline.weights[axis], rtol=0, atol=0)
        torch.testing.assert_close(stem.residual_scale[0], baseline.residual_scale[axis], rtol=0, atol=0)
        for name, value in stem.gates[0].state_dict().items():
            torch.testing.assert_close(value, baseline.gates[axis].state_dict()[name], rtol=0, atol=0)
        assert stem.weights.shape == (1, 5)
        assert stem.residual_scale.shape == (1,)
        assert len(stem.gates) == 1


def test_r030_uses_average_of_both_gates_for_each_physical_axis():
    ranges = torch.linspace(0.5, 2.5, 5)
    bearings = torch.linspace(0.5, -0.5, 5)
    z = (2 * ranges[:, None] + 6 * bearings[None, :] + 5)[None]
    stem = NativeStem("R030")
    with torch.no_grad():
        stem.weights.copy_(torch.tensor([[-2, -1, 0, 1, 2], [1, 0, 0, 0, -1]]))
        stem.residual_scale.fill_(1)
        stem.gates[0][2].bias.fill_(math.log(0.2 / 0.8))
        stem.gates[1][2].bias.fill_(math.log(0.8 / 0.2))

    output = stem(z, torch.ones(5, 5, dtype=torch.bool), ranges, bearings, (0.25, 2.75))

    torch.testing.assert_close(output[0, 2, 0], torch.tensor(13.5))
    torch.testing.assert_close(output[0, 4, 2], torch.tensor(8.8))
    torch.testing.assert_close(output[0, 2, 2], torch.tensor(8.5))


@pytest.mark.parametrize("variant,radial", [("R031", True), ("R032", False)])
def test_trained_single_axis_control_has_only_its_active_physical_update(variant, radial):
    ranges = torch.linspace(0.5, 2.5, 5)
    bearings = torch.linspace(0.5, -0.5, 5)
    z = (2 * ranges[:, None] + 6 * bearings[None, :] + 5)[None]
    stem = NativeStem(variant)
    with torch.no_grad():
        stem.weights[0].copy_(torch.tensor([-2, -1, 0, 1, 2] if radial else [1, 0, 0, 0, -1]))
        stem.residual_scale.fill_(1)

    output = stem(z, torch.ones(5, 5, dtype=torch.bool), ranges, bearings, (0.25, 2.75))

    torch.testing.assert_close(output[0, 2, 2], torch.tensor(10.5 if radial else 6.0))
    torch.testing.assert_close(output[0, 4, 2], torch.tensor(10.0 if radial else 8.8))
    torch.testing.assert_close(output[0, 2, 0], torch.tensor(13.5 if radial else 11.0))


def test_fixed_bin_axial_updates_are_parallel_and_do_not_read_calibration():
    stem = NativeStem("R012")
    with torch.no_grad():
        stem.weights.zero_()
        stem.weights[:, -1] = 1
        stem.residual_scale.fill_(1)
    z = (100 * torch.arange(9)[:, None] + torch.arange(9)[None, :]).float()[None]
    valid = torch.ones(9, 9, dtype=torch.bool)

    output = stem(z, valid)

    assert output[0, 5, 5] == 1717
    assert output[0, 0, 5] == 12
    assert output[0, 5, 8] == 1216
    assert output[0, 8, 8] == 808
    assert torch.equal(
        output,
        stem(z, valid, torch.linspace(10, 30, 9), torch.linspace(-1, 1, 9), (9, 31)),
    )
    valid[7, 5] = False
    assert stem(z, valid)[0, 5, 5] == 1012
    assert stem(z, valid)[0, 7, 5] == 0


@pytest.mark.parametrize("points,query", [(5, 2), (9, 4)])
def test_metric_sampler_matches_an_affine_field_on_two_descending_bearing_grids(points, query):
    ranges = torch.linspace(0.5, 2.5, points)
    bearings = torch.linspace(0.5, -0.5, points)
    z = (2 * ranges[:, None] + 6 * bearings[None, :] + 5)[None]
    stem = NativeStem("R020")
    with torch.no_grad():
        stem.weights.copy_(torch.tensor([[-2, -1, 0, 1, 2], [1, 0, 0, 0, -1]]))
        stem.residual_scale.fill_(1)

    output = stem(z, torch.ones(points, points, dtype=torch.bool), ranges, bearings)

    torch.testing.assert_close(output[0, query, query], torch.tensor(9.0))


def test_metric_boundary_disables_only_the_unsafe_branch_and_requires_all_four_nodes():
    ranges = torch.linspace(0.5, 2.5, 5)
    bearings = torch.linspace(0.5, -0.5, 5)
    z = (2 * ranges[:, None] + 6 * bearings[None, :] + 5)[None]
    valid = torch.ones(5, 5, dtype=torch.bool)
    stem = NativeStem("R020")
    with torch.no_grad():
        stem.weights.copy_(torch.tensor([[-2, -1, 0, 1, 2], [1, 0, 0, 0, -1]]))
        stem.residual_scale.fill_(1)

    torch.testing.assert_close(stem(z, valid, ranges, bearings)[0, 4, 2], torch.tensor(7.6))
    valid[3, 2] = False
    assert stem(z, valid, ranges, bearings)[0, 4, 2] == 10


def test_metric_radius_floor_and_zero_filled_mask_do_not_redefine_physical_validity():
    ranges = torch.tensor([0.25, 0.5, 0.75, 1.0, 1.25])
    bearings = torch.linspace(2, -2, 9)
    z = torch.ones(1, 5, 9)
    z[0, 2, 4] = 0
    valid = torch.ones(5, 9, dtype=torch.bool)
    stem = NativeStem("R020")
    with torch.no_grad():
        stem.weights.zero_()
        stem.weights[:, -1] = 1
        stem.residual_scale.fill_(1)

    output = stem(z, valid, ranges, bearings)

    assert output[0, 0, 4] == 1
    assert output[0, 2, 4] > 1
    assert valid.all()


def test_footprint_descriptor_uses_acquisition_bounds_positive_spacing_and_research_sign():
    ranges = torch.linspace(0.5, 2.5, 5)
    bearings = torch.linspace(0.5, -0.5, 5)
    valid = torch.ones(5, 5, dtype=torch.bool)

    descriptor = footprint_descriptor(ranges, bearings, valid, (0.25, 2.75))

    expected = torch.tensor([0.3, 0.2474039593, 0.9689124217, -0.6931471806,
                             -1.3862943611, -2.0794415417, -0.6931471806, 1])
    torch.testing.assert_close(descriptor[1, 1], expected)
    stem = NativeStem("R013")
    assert torch.equal(stem.gates[0](descriptor), torch.full((5, 5, 1), 0.5))
    with torch.no_grad():
        for parameter in stem.gates[0].parameters():
            parameter.zero_()
        stem.gates[0][0].weight[0, 6] = 1
        stem.gates[0][2].weight[0, 0] = 1
    torch.testing.assert_close(stem.gates[0](descriptor)[1, 1, 0], torch.tensor(0.4577998373))


@pytest.mark.parametrize("variant", ["R012", "R013", "R020", "R021", "R030", "R031", "R032", "R011"])
def test_zero_residual_starts_identity_then_enables_branch_gradients_after_one_update(variant):
    torch.manual_seed(3)
    stem = NativeStem(variant)
    z = torch.rand(1, 32, 32) + 0.5
    valid = torch.ones(32, 32, dtype=torch.bool)
    ranges = torch.linspace(1, 5, 32)
    bearings = torch.linspace(1, -1, 32)
    optimizer = torch.optim.SGD(stem.parameters(), lr=0.1)
    branch = stem.convs[0].weight if variant == "R011" else stem.weights

    stem(z, valid, ranges, bearings, (0.9, 5.1)).square().mean().backward()
    assert torch.count_nonzero(branch.grad) == 0
    optimizer.step()
    optimizer.zero_grad()
    stem(z, valid, ranges, bearings, (0.9, 5.1)).square().mean().backward()

    assert torch.isfinite(branch.grad).all()
    assert torch.count_nonzero(branch.grad) > 0
    if variant in ("R013", "R021", "R030", "R031", "R032"):
        for gate in stem.gates:
            assert torch.count_nonzero(gate[2].weight.grad) > 0
    if variant in ("R030", "R031", "R032"):
        optimizer.step()
        optimizer.zero_grad()
        stem(z, valid, ranges, bearings, (0.9, 5.1)).square().mean().backward()
        for parameter in stem.parameters():
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0


def test_r011_multiscale_residual_uses_complete_radius_six_support():
    stem = NativeStem("R011")
    z = torch.ones(1, 17, 17)
    valid = torch.ones(17, 17, dtype=torch.bool)
    with torch.no_grad():
        for parameter in stem.convs.parameters():
            parameter.fill_(0.1)
        stem.gamma.fill_(1)

    output = stem(z, valid)

    assert output[0, 6, 6] > 1
    assert output[0, 5, 6] == 1
    assert output[0, 6, 5] == 1
    valid[0, 0] = False
    assert stem(z, valid)[0, 6, 6] == 1
    assert stem(z, valid)[0, 0, 0] == 0


def test_polar_pe_uses_native_block_centers_physical_units_and_zero_initial_scale():
    pe = PolarPE()
    ranges = torch.arange(32).float() + 0.5
    bearings = torch.linspace(0.31, -0.31, 32)
    assert torch.equal(pe(ranges, bearings), torch.zeros(256, 384))
    with torch.no_grad():
        for network in (pe.range_mlp, pe.bearing_mlp):
            for parameter in network.parameters():
                parameter.zero_()
            network[0].weight[0, 0] = 1
            network[2].weight[0, 0] = 1
        pe.scale.fill_(1)
    output = pe(ranges, bearings)
    gelu = lambda value: value * (1 + math.erf(value / math.sqrt(2))) / 2
    torch.testing.assert_close(output[0, 0], torch.tensor(gelu(1 / 50) + gelu(0.30)))
    torch.testing.assert_close(output[16, 0], torch.tensor(gelu(3 / 50) + gelu(0.30)))
    torch.testing.assert_close(output[15, 0], torch.tensor(gelu(1 / 50) + gelu(-0.30)))
    assert torch.count_nonzero(output[:, 1:]) == 0


def test_polar_pe_keeps_all_rows_for_uatd_nondivisible_native_height():
    pe = PolarPE()
    ranges = (torch.arange(1028).float() + 0.5) * (10 / 1028)
    bearings = (torch.arange(512).float() + 0.5) * (-2 / 512) + 1
    with torch.no_grad():
        for parameter in pe.parameters():
            parameter.zero_()
        pe.range_mlp[0].weight[0, 0] = 1
        pe.range_mlp[2].weight[0, 0] = 1
        pe.scale.fill_(1)
    output = pe(ranges, bearings).reshape(16, 16, 384)
    centers = (torch.arange(16).float() + 0.5) * (10 / 16) / 50
    expected = torch.nn.functional.gelu(centers)
    torch.testing.assert_close(output[:, 0, 0], expected)
