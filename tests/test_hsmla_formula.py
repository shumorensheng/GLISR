"""Independent dense reference for the small-grid HSMLA equations."""
import copy

import torch
from torch.nn import functional as F

from runs.hsmla_formula import HSMLABlock


def direct_attention(module, field):
    b, c, h, w = field.shape
    n, d = h * w, c // module.heads
    raw = module.qkv(field)
    multi = sum(conv(raw) for conv in module.multiscale)
    q, k, v = raw.reshape(b, 3, module.heads, d, n).transpose(-1, -2).unbind(1)
    qh, kh, vh = multi.reshape(b, 3, module.heads, d, n).transpose(-1, -2).unbind(1)
    scores = qh.relu() @ kh.relu().transpose(-1, -2)
    global_output = (scores @ vh) / scores.sum(-1, keepdim=True).clamp_min(1e-6)
    positions = torch.arange(n)
    local = (((positions[:, None] // w - positions[None, :] // w).abs() <= 3)
             & ((positions[:, None] % w - positions[None, :] % w).abs() <= 3))
    local_scores = scores * local
    local_linear = (local_scores @ vh) / local_scores.sum(-1, keepdim=True).clamp_min(1e-6)
    soft_scores = (q @ k.transpose(-1, -2)) / d ** 0.5
    local_soft = soft_scores.masked_fill(~local, -torch.inf).softmax(-1) @ v
    gate = F.avg_pool2d(module.gate(field), 4).sigmoid()
    tokens = gate.repeat_interleave(4, 2).repeat_interleave(4, 3).reshape(b, 1, n, 1)
    if not module.training:
        tokens = (tokens > 0.15).to(field.dtype)
    output = global_output + tokens * (local_soft - local_linear)
    return module.projection(output.transpose(-1, -2).reshape(b, c, h, w)), gate


def test_soft_block_matches_dense_equations_and_gradients():
    torch.manual_seed(41)
    model = HSMLABlock(channels=12, heads=3).double()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 12, 8, 12, dtype=torch.double, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    result = model(x)
    normalized = reference.norm1(xr.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    attention, gate = direct_attention(reference, normalized)
    expected = xr + attention
    expected = expected + reference.depthwise(reference.norm2(expected.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    expected = expected + reference.ffn(reference.norm3(expected.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    reg = (0.01 * (gate.mean((1, 2, 3)) - reference.rho_logit.sigmoid()).abs()
           + 0.005 * ((gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().sum((1, 2, 3))
                      + (gate[:, :, 1:, :] - gate[:, :, :-1, :]).abs().sum((1, 2, 3)))).mean()
    torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(model.regularization_loss, reg)
    assert model.last_local_query_count == 192
    (result.square().mean() + model.regularization_loss).backward()
    (expected.square().mean() + reg).backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-10, rtol=1e-8)
    for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=1e-10, rtol=1e-8)
    assert model.gate.weight.grad.abs().sum() > 0
    assert model.rho_logit.grad.abs() > 0


def test_hard_gate_computes_only_selected_queries_and_matches_dense_reference():
    torch.manual_seed(42)
    model = HSMLABlock(channels=12, heads=3).double().eval()
    x = torch.randn(1, 12, 4, 8, dtype=torch.double)
    x[:, 0, :, :4] = -4
    x[:, 0, :, 4:] = 4
    calls = []
    original = model._local_correction

    def record_queries(*args):
        calls.append(args[6].clone())
        return original(*args)

    model._local_correction = record_queries
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.bias.zero_()
        model.gate.weight[0, 0, 1, 1] = 1
        actual = model._attention(x)
        expected, _ = direct_attention(model, x)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        selected = torch.cat(calls)
        assert selected.numel() == model.last_local_query_count == 16
        assert (selected % 8 >= 4).all()
        assert model.last_gate_fraction == 0.5
        calls.clear()
        model.gate.weight.zero_()
        model.gate.bias.fill_(-10)
        actual = model._attention(x)
        expected, _ = direct_attention(model, x)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        assert model.last_local_query_count == 0
        assert calls == []
