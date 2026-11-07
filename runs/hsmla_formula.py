"""Paper-equation HSMLA adaptation (arXiv:2608.07616v2), not released weights.

One Eq. (28) block on a BCHW feature field. The shared gate follows
Algorithm 1's Conv(X) -> tile average -> sigmoid, rather than head gates.
Plain PyTorch gathers/checkpointing do not reproduce the paper's CUDA kernel.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class HSMLABlock(nn.Module):
    def __init__(self, channels=384, heads=6):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.multiscale = nn.ModuleList([
            nn.Conv2d(3 * channels, 3 * channels, k, padding=k // 2,
                      groups=3 * channels) for k in (3, 5, 7)
        ])
        self.gate = nn.Conv2d(channels, 1, 3, padding=1)
        self.rho_logit = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.projection = nn.Conv2d(channels, channels, 1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 4 * channels, 1), nn.GELU(),
                                 nn.Conv2d(4 * channels, channels, 1))
        self.regularization_loss = None
        self.last_gate_fraction = None
        self.last_local_query_count = 0

    def _local_correction(self, q, k, v, qh, kh, vh, queries, height, width):
        # Queries are compacted across the batch; no full-grid unfolded K/V.
        n = height * width
        position = queries % n
        offsets = torch.arange(-3, 4, device=q.device)
        row = position[:, None, None] // width + offsets[None, :, None]
        col = position[:, None, None] % width + offsets[None, None, :]
        valid = ((row >= 0) & (row < height) & (col >= 0) & (col < width)).flatten(1)
        neighbors = (row.clamp(0, height - 1) * width + col.clamp(0, width - 1)).flatten(1)
        neighbors = neighbors + (queries // n)[:, None] * n
        scores = torch.einsum('qhd,qwhd->qhw', q[queries], k[neighbors]) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        soft = torch.einsum('qhw,qwhd->qhd', weights, v[neighbors])
        local_weights = torch.einsum('qhd,qwhd->qhw', qh[queries], kh[neighbors])
        local_weights = local_weights * valid[:, None, :]
        linear = torch.einsum('qhw,qwhd->qhd', local_weights, vh[neighbors])
        linear = linear / local_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        return soft - linear

    def _attention(self, field):
        batch, channels, height, width = field.shape
        n, dim = height * width, channels // self.heads
        raw = self.qkv(field)
        enriched = sum(conv(raw) for conv in self.multiscale)
        q, k, v = raw.reshape(batch, 3, self.heads, dim, n).unbind(1)
        qh, kh, vh = enriched.reshape(batch, 3, self.heads, dim, n).unbind(1)
        qh, kh = qh.relu(), kh.relu()
        aggregate = torch.einsum('bhdn,bhen->bhde', kh, vh)
        numerator = torch.einsum('bhdn,bhde->bhne', qh, aggregate)
        denominator = torch.einsum('bhdn,bhd->bhn', qh, kh.sum(-1)).clamp_min(1e-6)
        linear = (numerator / denominator[..., None]).permute(0, 2, 1, 3).reshape(batch * n, self.heads, dim)

        gate = F.avg_pool2d(self.gate(field), 4, stride=4).sigmoid()
        budget = (gate.mean((1, 2, 3)) - self.rho_logit.sigmoid()).abs()
        smooth = ((gate[:, :, 1:, :] - gate[:, :, :-1, :]).abs().sum((1, 2, 3))
                  + (gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().sum((1, 2, 3)))
        self.regularization_loss = (0.01 * budget + 0.005 * smooth).mean()
        self.last_gate_fraction = (gate.detach() > 0.15).float().mean()
        token_gate = gate.repeat_interleave(4, 2).repeat_interleave(4, 3).reshape(-1)
        queries = (torch.arange(batch * n, device=field.device) if self.training
                   else torch.nonzero(token_gate > 0.15, as_tuple=True)[0])
        self.last_local_query_count = queries.numel()
        flattened = [x.permute(0, 3, 1, 2).reshape(batch * n, self.heads, dim)
                     for x in (q, k, v, qh, kh, vh)]
        corrections = []
        for chunk in queries.split(128):
            if chunk.numel() == 0:
                continue
            args = (*flattened, chunk, height, width)
            if self.training and torch.is_grad_enabled():
                correction = checkpoint(self._local_correction, *args, use_reentrant=False)
            else:
                correction = self._local_correction(*args)
            if self.training:
                correction = correction * token_gate[chunk, None, None]
            corrections.append(correction)
        if corrections:
            linear = linear.index_add(0, queries, torch.cat(corrections))
        result = linear.reshape(batch, n, channels).transpose(1, 2).reshape_as(field)
        return self.projection(result)

    def forward(self, field):
        normalized = self.norm1(field.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        field = field + self._attention(normalized)
        normalized = self.norm2(field.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        field = field + self.depthwise(normalized)
        normalized = self.norm3(field.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return field + self.ffn(normalized)
