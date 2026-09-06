"""One global convex radial correction, with no fitted translation.

For P=T(z_t), Y=z_s let B=E||Y||² and N=E<(P-Y),Y>.
The nonnegative cone fit is a=max(N/B,0), giving prox(P)=P/(1+a).
Its population version decreases actual same-input marginal W2 for arbitrary
finite-second-moment distributions. There is no Gaussian or zero-mean premise.
Finite estimation and rollout distribution shift remain explicit limitations.
"""
import torch

from experiments.raev2_proximal_calibration import DiagonalFit


def pool_global_fit(fit: DiagonalFit) -> DiagonalFit:
    m, c, v, cross = (getattr(fit, key).double() for key in
                      ("center", "offset", "variance", "covariance"))
    target_second_moment = (v+m.square()).mean()
    residual_cross_moment = (cross+m*c).mean()
    if not bool(torch.isfinite(target_second_moment) & torch.isfinite(residual_cross_moment)):
        raise ValueError("nonfinite training moments")
    slope = (residual_cross_moment/target_second_moment).clamp_min(0) if target_second_moment > 0 else torch.zeros_like(target_second_moment)
    zero = torch.zeros_like(slope)
    expand = lambda value: value.expand_as(m)
    return DiagonalFit(*(expand(value) for value in
                         (zero, zero, slope, target_second_moment, residual_cross_moment)), count=fit.count)
