"""Data-calibrated nonexpansive corrections of a frozen Euler map.

Fit a diagonal convex quadratic gradient h(y)=offset+slope*(y-center)
to the step residual T(z_t)-z_s, using the TRUE NEXT state y=z_s.
The slope is constrained nonnegative.  This is a least-squares cone
projection, with no guidance multiplier or independently chosen time window.

For any such h, prox_h(T(z_t))-z_s=(r-h(z_s))/(1+slope).
Consequently its squared coupling error is bounded by ||r-h(z_s)||²,
and the correction map is nonexpansive. Population cone fitting reduces
that upper bound; finite-sample fitting must be checked on independent data.
These statements do not assert monotone FID for a finite fitted model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DiagonalFit:
    center: Tensor
    offset: Tensor
    slope: Tensor
    variance: Tensor
    covariance: Tensor
    count: int

    def state_dict(self) -> dict:
        return {"center": self.center, "offset": self.offset, "slope": self.slope,
                "variance": self.variance, "covariance": self.covariance, "count": self.count}

    @classmethod
    def from_state_dict(cls, state: dict) -> "DiagonalFit":
        return cls(**{key: state[key] for key in (
            "center", "offset", "slope", "variance", "covariance", "count")})

    def to(self, device=None, dtype=torch.float32) -> "DiagonalFit":
        return DiagonalFit(**{key: getattr(self, key).to(device=device, dtype=dtype)
                              for key in ("center", "offset", "slope", "variance", "covariance")},
                           count=self.count)


class DiagonalMoments:
    """Mergeable, FP64 centered sufficient statistics of (next state, error)."""
    def __init__(self):
        self.count = 0
        self.mean_state = self.mean_error = self.state_m2 = self.cross_m2 = None

    def update(self, state: Tensor, error: Tensor) -> None:
        if state.shape != error.shape or state.ndim < 2 or state.shape[0] == 0:
            raise ValueError("expected matching nonempty batched state and error")
        if state.device != error.device:
            raise ValueError("state and error must share a device")
        if self.count and (state.shape[1:] != self.mean_state.shape or state.device != self.mean_state.device):
            raise ValueError("moment shape and device must remain fixed")
        x, e = state.detach().double(), error.detach().double()
        n = len(x)
        mx, me = x.mean(0), e.mean(0)
        xc, ec = x-mx, e-me
        xx, xe = xc.square().sum(0), (xc*ec).sum(0)
        if not self.count:
            self.count = n
            self.mean_state, self.mean_error = mx, me
            self.state_m2, self.cross_m2 = xx, xe
            return
        total = self.count+n
        dx, de = mx-self.mean_state, me-self.mean_error
        weight = self.count*n/total
        self.state_m2 += xx+weight*dx.square()
        self.cross_m2 += xe+weight*dx*de
        self.mean_state += (n/total)*dx
        self.mean_error += (n/total)*de
        self.count = total

    def fit(self) -> DiagonalFit:
        if self.count < 2:
            raise ValueError("at least two samples required for calibration")
        variance = (self.state_m2/self.count).clamp_min(0)
        covariance = self.cross_m2/self.count
        safe_variance = torch.where(variance > 0, variance, torch.ones_like(variance))
        slope = torch.where(variance > 0, covariance/safe_variance, torch.zeros_like(variance)).clamp_min(0)
        fit = DiagonalFit(self.mean_state.clone(), self.mean_error.clone(), slope,
                          variance, covariance, self.count)
        if any(not bool(torch.isfinite(getattr(fit, key)).all())
               for key in ("center", "offset", "slope", "variance", "covariance")):
            raise FloatingPointError("nonfinite calibration statistics")
        return fit


def evaluate_correction(value: Tensor, fit: DiagonalFit) -> Tensor:
    """Evaluate h at a state, using the state's precision (FP32 in sampling)."""
    if value.shape[1:] != fit.center.shape:
        raise ValueError("state does not match calibrated coordinates")
    center, offset, slope = (getattr(fit, key).to(value) for key in ("center", "offset", "slope"))
    return offset+slope*(value-center)


def apply_proximal(value: Tensor, fit: DiagonalFit) -> Tensor:
    """Return (I+h)^-1(value), the exact proximal map of the fitted quadratic."""
    if value.shape[1:] != fit.center.shape:
        raise ValueError("state does not match calibrated coordinates")
    center, offset, slope = (getattr(fit, key).to(value) for key in ("center", "offset", "slope"))
    return (value-offset+slope*center)/(1+slope)


def heldout_step_metrics(predicted_next: Tensor, true_next: Tensor, fit: DiagonalFit) -> dict[str, Tensor]:
    """Per-image metrics; positive risk_gain/J indicate lower coupling error.

    Mathematical nonexpansivity guarantees risk_gain >= J for any frozen fit
    with nonnegative slopes, even when J itself is negative on held-out data.
    Reduce across coordinates only; independent images are statistical units.
    """
    if predicted_next.shape != true_next.shape:
        raise ValueError("predicted and true next states must agree in shape")
    p, y = predicted_next.float(), true_next.float()
    residual = p-y
    correction = evaluate_correction(y, fit)
    corrected = apply_proximal(p, fit)
    mean = lambda v: v.flatten(1).double().mean(1)
    risk_before = mean(residual.square())
    risk_after = mean((corrected-y).square())
    certificate = mean(2*correction*residual-correction.square())
    return {"risk_before": risk_before, "risk_after": risk_after,
            "risk_gain": risk_before-risk_after, "J": certificate,
            "nonexpansive_slack": risk_before-risk_after-certificate,
            "h_target_mse": mean(correction.square()),
            "applied_correction_mse": mean((corrected-p).square())}
