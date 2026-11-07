"""Single-channel native stems; callers mask intensity before any spatial mixing."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def footprint_descriptor(range_m, bearing_rad, valid, range_bounds):
    """Research-order descriptor with sampling-scale area, not acoustic PSF area."""
    height, width = valid.shape
    r_min, r_max = range_bounds
    dr = (range_m[1] - range_m[0]).abs()
    dtheta = (bearing_rad[1] - bearing_rad[0]).abs()
    x = dr.log().expand(height, width)
    y = (range_m[:, None] * dtheta).log().expand(height, width)
    return torch.stack(
        [
            ((range_m[:, None] - r_min) / (r_max - r_min)).expand(height, width),
            bearing_rad.sin()[None].expand(height, width),
            bearing_rad.cos()[None].expand(height, width),
            x,
            y,
            x + y,
            y - x,
            valid.to(range_m.dtype),
        ],
        dim=-1,
    )


def _metric_update(z, valid, range_m, bearing_rad, weights, radial):
    height, width = valid.shape
    offsets = torch.arange(-2, 3, device=z.device, dtype=range_m.dtype)[:, None, None] / 4
    rows = torch.arange(height, device=z.device, dtype=range_m.dtype)[None, :, None]
    cols = torch.arange(width, device=z.device, dtype=range_m.dtype)[None, None, :]
    if radial:
        rows = rows + offsets / (range_m[1] - range_m[0])
    else:
        cols = cols + offsets / (range_m[None, :, None] * (bearing_rad[1] - bearing_rad[0]))
    rows = rows.expand(5, height, width)
    cols = cols.expand(5, height, width)
    return _bilinear_update(z, valid, range_m, weights, rows, cols)


def _bilinear_update(z, valid, range_m, weights, rows, cols):
    height, width = valid.shape
    inside = (rows >= 0) & (rows <= height - 1) & (cols >= 0) & (cols <= width - 1)

    # Clamp addresses only; queries outside the hull never contribute an update.
    i = rows.floor().long().clamp(0, height - 2)
    j = cols.floor().long().clamp(0, width - 2)
    s, t = rows - i, cols - j
    stencil_valid = valid[i, j] & valid[i + 1, j] & valid[i, j + 1] & valid[i + 1, j + 1]
    safe = (inside & stencil_valid).all(dim=0) & valid & (range_m[:, None] >= 0.5)
    samples = (
        (1 - s) * (1 - t) * z[0, i, j]
        + s * (1 - t) * z[0, i + 1, j]
        + (1 - s) * t * z[0, i, j + 1]
        + s * t * z[0, i + 1, j + 1]
    )
    update = (weights[:, None, None] * samples).sum(dim=0)
    return update.masked_fill(~safe, 0)


class PolarPE(nn.Module):
    """Explicit R011 adaptation: physical native-block centers, 384-dimensional PE."""

    def __init__(self):
        super().__init__()
        self.range_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 384))
        self.bearing_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 384))
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, range_m, bearing_rad):
        ranges = F.interpolate(range_m[None, None], size=16, mode="linear", align_corners=False)[0, 0, :, None] / 50
        bearings = F.interpolate(bearing_rad[None, None], size=16, mode="linear", align_corners=False)[0, 0, :, None]
        encoding = self.range_mlp(ranges)[:, None] + self.bearing_mlp(bearings)[None]
        return self.scale * encoding.reshape(256, 384)


class NativeStem(nn.Module):
    """Keep echo embedding and output single-channel; preserve the R010 projection.

    Inputs are standardized, already masked z[1,H,W] and physical valid[H,W].
    All tensors reside on the model device. R013 and metric variants read calibration;
    gated variants also take acquisition-edge range_bounds=(RangeMin, RangeMax).
    """

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        if variant == "R011":
            self.convs = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(16, 16, 5, padding=2, groups=16, bias=False),
                nn.Conv2d(16, 16, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(16, 16, 7, padding=3, groups=16, bias=False),
                nn.Conv2d(16, 16, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(16, 1, 1, bias=False),
            )
            self.gamma = nn.Parameter(torch.zeros(()))
            self.polar_pe = PolarPE()
        elif variant == "R036":
            self.input_features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.geometry_mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64))
            self.aggregation = nn.Sequential(
                nn.BatchNorm2d(576), nn.ReLU(), nn.Conv2d(576, 64, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 1, bias=False),
            )
            self.gamma = nn.Parameter(torch.zeros(()))
        elif variant in ("R012", "R013", "R020", "R021", "R030", "R037", "R038", "R039"):
            self.weights = nn.Parameter(torch.randn(2, 5) / math.sqrt(5))
            self.residual_scale = nn.Parameter(torch.zeros(2))
            if variant in ("R013", "R021", "R030", "R037", "R039"):
                self.gates = nn.ModuleList(
                    nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
                    for _ in range(2)
                )
                for gate in self.gates:
                    nn.init.zeros_(gate[2].weight)
                    nn.init.zeros_(gate[2].bias)
            if variant == "R037":
                self.offset_head = nn.Sequential(
                    nn.Conv2d(9, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 10, 1),
                )
                nn.init.zeros_(self.offset_head[2].weight)
                nn.init.zeros_(self.offset_head[2].bias)
            elif variant == "R038":
                self.intensity_mlp = nn.Sequential(
                    nn.Conv2d(1, 107, 1), nn.GELU(), nn.Conv2d(107, 1, 1, bias=False),
                )
                self.gamma = nn.Parameter(torch.zeros(()))
        elif variant in ("R031", "R032"):
            # Draw both R021 branches in order, then register only the active axis.
            weights = torch.randn(2, 5) / math.sqrt(5)
            gates = [nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
                     for _ in range(2)]
            for gate in gates:
                nn.init.zeros_(gate[2].weight)
                nn.init.zeros_(gate[2].bias)
            axis = 0 if variant == "R031" else 1
            self.weights = nn.Parameter(weights[axis:axis + 1].clone())
            self.residual_scale = nn.Parameter(torch.zeros(1))
            self.gates = nn.ModuleList([gates[axis]])
        else:
            raise ValueError(f"Unknown native stem variant: {variant}")

    def forward(self, z, valid, range_m=None, bearing_rad=None, range_bounds=None):
        if self.variant == "R011":
            coverage = F.conv2d(valid.float()[None, None], z.new_ones(1, 1, 13, 13), padding=6)
            update = self.convs(z[None])[0].masked_fill(coverage[0] != 169, 0)
            return (z + self.gamma * update).masked_fill(~valid[None], 0)

        if self.variant == "R036":
            height, width = valid.shape
            xy = range_m[None, :, None] * torch.stack([bearing_rad.cos(), bearing_rad.sin()])[:, None, :]
            relative_xy = F.unfold(xy[None], 3, padding=1).reshape(1, 2, 9, height, width) - xy[None, :, None]
            dynamic_weights = self.geometry_mlp(relative_xy.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            features = self.input_features(z[None])
            neighbors = F.unfold(features, 3, padding=1).reshape(1, 64, 9, height, width)
            # Match RangeDet's channel-major ordering: product channel 9*c+k.
            products = (neighbors * dynamic_weights).reshape(1, 576, height, width)
            update = self.aggregation(products)[0]
            coverage = F.conv2d(valid.float()[None, None], z.new_ones(1, 1, 5, 5), padding=2)[0, 0]
            safe = (coverage == 25) & (range_m[:, None] >= 0.5)
            # This bounds the local gather only; training BN uses whole-image statistics.
            update = update.masked_fill(~safe[None], 0)
            return (z + self.gamma * update).masked_fill(~valid[None], 0)

        if self.variant in ("R013", "R021", "R030", "R031", "R032", "R037", "R039"):
            descriptor = footprint_descriptor(range_m, bearing_rad, valid, range_bounds)
        if self.variant in ("R031", "R032"):
            update = _metric_update(z, valid, range_m, bearing_rad, self.weights[0], self.variant == "R031")
            update = update * self.gates[0](descriptor).squeeze(-1)
            return (z + self.residual_scale[0] * update[None]).masked_fill(~valid[None], 0)
        if self.variant == "R030":
            common_gate = (self.gates[0](descriptor) + self.gates[1](descriptor)).squeeze(-1) / 2
        if self.variant in ("R037", "R039"):
            height, width = valid.shape
            rows = torch.arange(height, device=z.device, dtype=range_m.dtype)[None, :, None]
            cols = torch.arange(width, device=z.device, dtype=range_m.dtype)[None, None, :]
            offsets = torch.arange(-2, 3, device=z.device, dtype=range_m.dtype)[:, None, None]
            if self.variant == "R037":
                head_input = torch.cat([z, descriptor.permute(2, 0, 1)], dim=0)[None]
                displacements = offsets[None] / 4 + 0.5 * self.offset_head(head_input).reshape(2, 5, height, width).tanh()
                coverage = F.conv2d(valid.float()[None, None], z.new_ones(1, 1, 3, 3), padding=1)[0, 0]
        result = z
        for branch in range(2):
            if self.variant in ("R012", "R013", "R038"):
                shape, padding = ((5, 1), (2, 0)) if branch == 0 else ((1, 5), (0, 2))
                kernel = self.weights[branch].reshape(1, 1, *shape)
                update = F.conv2d(z[None], kernel, padding=padding)[0, 0]
                coverage = F.conv2d(valid.float()[None, None], torch.ones_like(kernel), padding=padding)
                update = update.masked_fill((coverage[0, 0] != 5) | ~valid, 0)
            elif self.variant in ("R037", "R039"):
                query_rows, query_cols = rows, cols
                if self.variant == "R037":
                    if branch == 0:
                        query_rows = rows + displacements[branch] / (range_m[1] - range_m[0])
                    else:
                        query_cols = cols + displacements[branch] / (range_m[None, :, None] * (bearing_rad[1] - bearing_rad[0]))
                elif branch == 0:
                    query_rows = rows + offsets * 6.564102564102564
                else:
                    query_cols = cols + offsets * 3.974991749417094 * (bearing_rad[1] - bearing_rad[0]).sign()
                update = _bilinear_update(z, valid, range_m, self.weights[branch],
                                          query_rows.expand(5, height, width), query_cols.expand(5, height, width))
                if self.variant == "R037":
                    update = update.masked_fill(coverage != 9, 0)
            else:
                update = _metric_update(z, valid, range_m, bearing_rad, self.weights[branch], branch == 0)
            if self.variant == "R030":
                update = update * common_gate
            elif self.variant in ("R013", "R021", "R037", "R039"):
                update = update * self.gates[branch](descriptor).squeeze(-1)
            result = result + self.residual_scale[branch] * update[None]
        if self.variant == "R038":
            result = result + self.gamma * self.intensity_mlp(z[None])[0]
        return result.masked_fill(~valid[None], 0)
