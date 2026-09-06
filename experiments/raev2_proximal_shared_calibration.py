"""Exact projection into the spatially shared channel-affine convex cone.

The input contains complete coordinate-wise TRAINING moments. Pooling includes
between-position means; it does not treat spatial positions as independent
images or assume a spatially invariant target distribution.
"""
import torch

from experiments.raev2_proximal_calibration import DiagonalFit


def pool_channel_fit(fit: DiagonalFit) -> DiagonalFit:
    if fit.center.ndim != 3:
        raise ValueError("expected channel-height-width moments")
    m, c, v, cross = (getattr(fit, key).double() for key in
                      ("center", "offset", "variance", "covariance"))
    mean_m = m.mean((-2, -1), keepdim=True)
    mean_c = c.mean((-2, -1), keepdim=True)
    variance = (v + (m-mean_m).square()).mean((-2, -1), keepdim=True)
    covariance = (cross + (m-mean_m)*(c-mean_c)).mean((-2, -1), keepdim=True)
    slope = torch.where(variance > 0,
                        covariance/torch.where(variance > 0, variance, torch.ones_like(variance)),
                        torch.zeros_like(variance)).clamp_min(0)
    expand = lambda value: value.expand_as(m)
    return DiagonalFit(*(expand(value) for value in
                         (mean_m, mean_c, slope, variance, covariance)), count=fit.count)
