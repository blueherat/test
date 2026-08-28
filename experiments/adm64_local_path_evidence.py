#!/usr/bin/env python3
"""Predictable spatially localized alternatives for ADM path evidence.

The first ADM screen tilts the full 64x64 state with a cross-scale score
difference.  A small malformed limb or fused boundary can be diluted by
unrelated, perfectly ordinary pixels.  This module defines a theory-valid
diagnostic alternative that chooses the highest-energy *contiguous tile*
before the current transition noise is drawn and applies the Gaussian tilt
only inside that tile.

For a baseline transition ``P=N(mu, diag(sigma**2))`` and a predictable score
difference ``theta``, let ``w=sigma*theta``.  A fixed grid partitions the
spatial plane.  The tile with largest ``sum(w**2)`` is selected using only the
current history, ties are resolved in row-major order, and

    delta = gamma * sigma**2 * mask * theta,

where ``gamma`` caps ``0.5*||gamma*mask*w||**2``.  Conditional on the history
this still defines a normalized same-covariance Gaussian Q, so

    log(Q/P) = (gamma*mask*w)^T epsilon
               - 0.5*||gamma*mask*w||**2

is an exact finite-step likelihood-ratio increment.  Selecting a tile after
seeing ``epsilon`` would be invalid and is deliberately impossible through
this interface.

This is an exploratory Q modification, not a post-hoc relabeling of the
locked global primary detector.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

try:
    from .adm64_path_evidence import LogLikelihoodIncrement, same_covariance_log_lr_from_noise
except ImportError:  # pragma: no cover - CLI execution.
    from adm64_path_evidence import LogLikelihoodIncrement, same_covariance_log_lr_from_noise


@dataclass(frozen=True)
class PredictableTileShift:
    """A batch of pre-innovation, spatially localized Gaussian mean shifts."""

    mean_shift: torch.Tensor
    raw_whitened_shift: torch.Tensor
    whitened_shift: torch.Tensor
    scale: torch.Tensor
    raw_kl: torch.Tensor
    applied_kl: torch.Tensor
    grid_size: int
    tile_index: torch.Tensor
    tile_bounds_yxyx: torch.Tensor
    selected_energy_fraction: torch.Tensor


def _validate_inputs(
    score_difference: torch.Tensor,
    standard_deviation: torch.Tensor,
    grid_size: int,
    max_conditional_kl: float,
) -> None:
    if score_difference.shape != standard_deviation.shape:
        raise ValueError("score difference and standard deviation shapes must match")
    if score_difference.ndim != 4:
        raise ValueError("expected tensors with shape [batch, channels, height, width]")
    if not score_difference.is_floating_point() or not standard_deviation.is_floating_point():
        raise TypeError("score difference and standard deviation must be floating point")
    if not torch.isfinite(score_difference).all() or not torch.isfinite(standard_deviation).all():
        raise ValueError("inputs must be finite")
    if not torch.all(standard_deviation > 0):
        raise ValueError("standard deviations must be strictly positive")
    if not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    height, width = score_difference.shape[-2:]
    if height % grid_size or width % grid_size:
        raise ValueError("height and width must be divisible by grid_size")
    if not math.isfinite(max_conditional_kl) or max_conditional_kl <= 0:
        raise ValueError("max_conditional_kl must be finite and strictly positive")


def predictable_max_energy_tile_shift(
    score_difference: torch.Tensor,
    standard_deviation: torch.Tensor,
    *,
    grid_size: int,
    max_conditional_kl: float,
) -> PredictableTileShift:
    """Select one contiguous tile per sample and construct a KL-capped tilt.

    The function accepts no innovation/noise argument.  Call it before drawing
    the current P transition noise, then pass ``result.whitened_shift`` and the
    realized noise to :func:`localized_log_lr_from_noise`.
    """

    _validate_inputs(
        score_difference, standard_deviation, grid_size, max_conditional_kl
    )
    score64 = score_difference.to(torch.float64)
    sigma64 = standard_deviation.to(torch.float64)
    raw_global = sigma64 * score64
    batch, _, height, width = raw_global.shape
    tile_height = height // grid_size
    tile_width = width // grid_size

    # [B, C, grid_y, tile_y, grid_x, tile_x] -> energy [B, grid_y, grid_x].
    tiled = raw_global.reshape(
        batch, raw_global.shape[1], grid_size, tile_height, grid_size, tile_width
    )
    energies = tiled.square().sum(dim=(1, 3, 5))
    flattened = energies.reshape(batch, -1)
    tile_index = flattened.argmax(dim=1)

    masked = torch.zeros_like(raw_global)
    bounds = torch.empty((batch, 4), dtype=torch.int64, device=raw_global.device)
    selected_energy = torch.empty(batch, dtype=torch.float64, device=raw_global.device)
    for sample_index in range(batch):
        index = int(tile_index[sample_index].item())
        tile_y, tile_x = divmod(index, grid_size)
        y0, y1 = tile_y * tile_height, (tile_y + 1) * tile_height
        x0, x1 = tile_x * tile_width, (tile_x + 1) * tile_width
        masked[sample_index, :, y0:y1, x0:x1] = raw_global[
            sample_index, :, y0:y1, x0:x1
        ]
        bounds[sample_index] = torch.tensor(
            [y0, x0, y1, x1], dtype=torch.int64, device=raw_global.device
        )
        selected_energy[sample_index] = flattened[sample_index, index]

    total_energy = flattened.sum(dim=1)
    energy_fraction = torch.zeros_like(total_energy)
    positive_total = total_energy > 0
    energy_fraction[positive_total] = (
        selected_energy[positive_total] / total_energy[positive_total]
    )

    raw_kl = 0.5 * masked.reshape(batch, -1).square().sum(dim=1)
    scale = torch.ones_like(raw_kl)
    positive_kl = raw_kl > 0
    cap = torch.full_like(raw_kl, float(max_conditional_kl))
    scale[positive_kl] = torch.minimum(
        scale[positive_kl], torch.sqrt(cap[positive_kl] / raw_kl[positive_kl])
    )
    expanded_scale = scale.reshape(batch, 1, 1, 1)
    whitened = masked * expanded_scale
    applied_kl = 0.5 * whitened.reshape(batch, -1).square().sum(dim=1)
    mean_shift = sigma64 * whitened
    return PredictableTileShift(
        mean_shift=mean_shift,
        raw_whitened_shift=masked,
        whitened_shift=whitened,
        scale=scale,
        raw_kl=raw_kl,
        applied_kl=applied_kl,
        grid_size=grid_size,
        tile_index=tile_index,
        tile_bounds_yxyx=bounds,
        selected_energy_fraction=energy_fraction,
    )


def localized_log_lr_from_noise(
    shift: PredictableTileShift,
    sampled_noise: torch.Tensor,
) -> LogLikelihoodIncrement:
    """Evaluate the already-selected localized Q/P increment after sampling."""

    return same_covariance_log_lr_from_noise(shift.whitened_shift, sampled_noise)


def run_self_test() -> None:
    torch.manual_seed(17)
    theta = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    sigma = torch.full_like(theta, 0.25)
    theta[0, :, 0:4, 4:8] = 5.0
    theta[1, :, 4:8, 0:4] = -3.0
    selected = predictable_max_energy_tile_shift(
        theta, sigma, grid_size=2, max_conditional_kl=0.2
    )
    assert selected.tile_index.tolist() == [1, 2]
    assert selected.tile_bounds_yxyx.tolist() == [[0, 4, 4, 8], [4, 0, 8, 4]]
    assert torch.allclose(selected.selected_energy_fraction, torch.ones(2, dtype=torch.float64))
    assert torch.all(selected.applied_kl <= 0.2 + 1e-12)
    outside = selected.raw_whitened_shift.clone()
    outside[0, :, 0:4, 4:8] = 0
    outside[1, :, 4:8, 0:4] = 0
    assert torch.count_nonzero(outside) == 0

    # The tile choice is independent of the subsequently supplied innovation.
    choice_before = selected.tile_index.clone()
    for _ in range(8):
        noise = torch.randn_like(theta)
        increment = localized_log_lr_from_noise(selected, noise)
        assert torch.isfinite(increment.value).all()
        assert torch.equal(selected.tile_index, choice_before)

    # Monte Carlo calibration for a fixed predictable localized shift.
    one = predictable_max_energy_tile_shift(
        theta[:1] * 0.08, sigma[:1], grid_size=2, max_conditional_kl=0.02
    )
    generator = torch.Generator().manual_seed(1234)
    values = []
    for _ in range(20_000):
        noise = torch.randn(theta[:1].shape, generator=generator)
        values.append(torch.exp(localized_log_lr_from_noise(one, noise).value))
    empirical_mean = float(torch.cat(values).mean().item())
    if abs(empirical_mean - 1.0) > 0.015:
        raise AssertionError(f"localized E calibration failed: {empirical_mean}")

    # Row-major tie breaking is deterministic and does not inspect noise.
    tied = predictable_max_energy_tile_shift(
        torch.ones((1, 1, 4, 4)),
        torch.ones((1, 1, 4, 4)),
        grid_size=2,
        max_conditional_kl=1.0,
    )
    assert tied.tile_index.item() == 0
    print(
        "self-test passed: predictable contiguous-tile selection, KL cap, "
        f"and Monte Carlo E calibration ({empirical_mean:.6f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is available; sampler integration is a separate runner")
    run_self_test()


if __name__ == "__main__":
    main()
