"""Shared affine guidance and Euclidean energy objectives, without a sampler.

The initial hypothesis uses raw predicted-clean channel values, with no feature
normalization, bandwidth, gain cap, or time schedule. This parameterization and
these distribution objectives alone provide no claim of an FID improvement.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AffineTokenGuidance(nn.Module):
    """One affine scalar per spatial token, shared across all tokens and steps.

    For channel vectors F_p and B_p, a_p = b + u^T F_p + v^T B_p, and
    G_p = G_official,p + a_p (F_p - B_p). Default C=1024 gives 2049 parameters,
    all initialized to zero. Raw F/B values are the current modeling assumption.

    Inputs are matching finite [N,C,H,W] tensors. The caller supplies the exact
    official prediction, including its existing guidance window; this module
    has no time input or window. The learned correction applies wherever called.
    Parameters must be moved to the input device/dtype, e.g. gate.to(full).
    Neither F/B nor the official prediction is detached, so state derivatives
    propagate through every supplied prediction as well as through the gate.
    """

    def __init__(self, channels: int = 1024) -> None:
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.u = nn.Parameter(torch.zeros(channels))
        self.v = nn.Parameter(torch.zeros(channels))
        self.b = nn.Parameter(torch.zeros(()))

    def _validate(self, full: Tensor, base: Tensor) -> None:
        if not isinstance(full, Tensor) or not isinstance(base, Tensor):
            raise TypeError("full and base must be tensors")
        if full.ndim != 4 or any(size == 0 for size in full.shape):
            raise ValueError("predictions must be nonempty [N,C,H,W] tensors")
        if full.shape != base.shape or full.shape[1] != self.channels:
            raise ValueError("full/base shapes must match the configured channel count")
        if not full.is_floating_point() or not base.is_floating_point():
            raise TypeError("predictions must be floating-point tensors")
        if full.dtype != base.dtype or full.device != base.device:
            raise ValueError("full/base must have the same dtype and device")
        if any(p.dtype != full.dtype or p.device != full.device for p in self.parameters()):
            raise ValueError("gate parameters must match prediction dtype and device")

    def gate(self, full: Tensor, base: Tensor) -> Tensor:
        """Return differentiable scalar gates of shape [N,1,H,W]."""
        self._validate(full, base)
        return ((full * self.u[None, :, None, None]).sum(1, keepdim=True)
                + (base * self.v[None, :, None, None]).sum(1, keepdim=True)
                + self.b)

    def forward(self, full: Tensor, base: Tensor, official: Tensor) -> Tensor:
        coefficient = self.gate(full, base)
        if not isinstance(official, Tensor):
            raise TypeError("official must be a tensor")
        if (official.shape != full.shape or official.dtype != full.dtype
                or official.device != full.device):
            raise ValueError("official must match prediction shape, dtype and device")
        return official + coefficient * (full - base)


def _feature_pair(generated: Tensor, real: Tensor, *, minimum_real: int) -> tuple[Tensor, Tensor]:
    if not isinstance(generated, Tensor) or not isinstance(real, Tensor):
        raise TypeError("generated and real features must be tensors")
    if generated.ndim != 2 or real.ndim != 2 or generated.shape[1] == 0:
        raise ValueError("features must have nonempty [samples,features] shapes")
    if generated.shape[1] != real.shape[1]:
        raise ValueError("generated/real feature dimensions must match")
    if generated.shape[0] < 2 or real.shape[0] < minimum_real:
        raise ValueError(f"requires at least 2 generated and {minimum_real} real samples")
    if not generated.is_floating_point() or not real.is_floating_point():
        raise TypeError("features must be floating-point tensors")
    if generated.device != real.device or generated.dtype != real.dtype:
        raise ValueError("generated/real features must have the same dtype and device")
    if not bool(torch.isfinite(generated).all()) or not bool(torch.isfinite(real).all()):
        raise ValueError("features must be finite")
    # CPU cdist lacks half kernels; accumulation in FP32 also avoids half sums.
    # Tensor conversion retains the graph to the original feature producers.
    if generated.dtype in (torch.float16, torch.bfloat16):
        generated, real = generated.float(), real.float()
    return generated, real


def _distances(first: Tensor, second: Tensor) -> Tensor:
    # Avoid the norm-squared matmul identity: roundoff there can give a nonzero
    # self-distance and corrupt diagonal exclusion and gradients near equality.
    return torch.cdist(first, second, p=2, compute_mode="donot_use_mm_for_euclid_dist")


def _within_mean(features: Tensor) -> Tensor:
    distances = _distances(features, features)
    count = len(features)
    return (distances.sum() - distances.diagonal().sum()) / (count * (count - 1))


def energy_distance_u_statistic(generated: Tensor, real: Tensor) -> Tensor:
    """Full pooled energy U-statistic, allowing N != M with N,M >= 2.

    2 mean_ij ||g_i-r_j|| - mean_i!=j ||g_i-g_j|| - mean_i!=j ||r_i-r_j||.

    Unbiasedness refers to independent iid samples from two pooled populations.
    A class-stratified sample, or one generated image per class, is not thereby
    a conditional-energy estimator. The finite-sample U-statistic can be
    negative; it is not clipped. All three terms retain their input gradients.
    """
    generated, real = _feature_pair(generated, real, minimum_real=2)
    return 2 * _distances(generated, real).mean() - _within_mean(generated) - _within_mean(real)


def _labels(labels: Tensor, features: Tensor, name: str) -> Tensor:
    if not isinstance(labels, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if labels.shape != (len(features),):
        raise ValueError(f"{name} must have one label per feature row")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must contain integer class labels")
    return labels.to(device=features.device)


def conditional_energy_training_objective(
    generated: Tensor, generated_labels: Tensor, real: Tensor, real_labels: Tensor,
) -> Tensor:
    """Uniform average over generated classes of energy's trainable terms.

    For each represented class c, require n_c >= 2 and m_c >= 1, and use
    2 mean_ij ||g_ci-r_cj|| - mean_i!=j ||g_ci-g_cj||. Extra real-only classes
    are unused. Missing real counterparts or one generated image in a class
    are errors. Different class sample counts do not change class weights.

    The real-real constant is intentionally omitted: this scalar is NOT the
    full conditional energy distance. A single iid real sample per class can
    give an unbiased gradient in expectation when independent of generated
    draws; a fixed reused real bank does not make an exact-population claim.
    No input feature is detached, including real; freezing its producer is
    the caller's responsibility. No bandwidth, normalization, or gain is used.
    """
    generated, real = _feature_pair(generated, real, minimum_real=1)
    generated_labels = _labels(generated_labels, generated, "generated_labels")
    real_labels = _labels(real_labels, real, "real_labels")
    objectives = []
    for label in torch.unique(generated_labels).tolist():
        gen_class = generated[generated_labels == label]
        real_class = real[real_labels == label]
        if len(gen_class) < 2:
            raise ValueError(f"class {label} requires at least 2 generated samples")
        if len(real_class) < 1:
            raise ValueError(f"class {label} requires at least 1 real sample")
        objectives.append(2 * _distances(gen_class, real_class).mean() - _within_mean(gen_class))
    return torch.stack(objectives).mean()
