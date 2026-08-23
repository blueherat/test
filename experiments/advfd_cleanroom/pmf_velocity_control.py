"""Pure local flow-matching objective for the released pMF model.

This module deliberately contains no Frechet-distance feature extractor or
adaptive critic.  It is the ordinary supervised continuation control used to
separate AdvFD gains from gains caused by seeing more ImageNet batches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VelocityControlProtocol:
    path: str = "z_t=(1-t)*x0+t*noise"
    target: str = "(z_t-x0)/clamp(t,t_eps,1)"
    prediction: str = "pMF u_fn at h=0, omega=1, interval=[0,1]"
    loss: str = "mean squared velocity error"
    timestep_sampling: str = "pMF logit-normal sample_t"
    classifier_free_dropout: str = "released pMF cond_drop"
    feature_distance_loss: bool = False
    adaptive_feature_critic: bool = False
    generated_image_training: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def pmf_velocity_target(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    *,
    t_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the released pMF linear path and its capped velocity target."""

    if x0.shape != noise.shape:
        raise ValueError(f"x0/noise shape mismatch: {x0.shape} vs {noise.shape}")
    if t.ndim != 4 or t.shape[0] != x0.shape[0] or t.shape[1:] != (1, 1, 1):
        raise ValueError(f"expected t shape [B,1,1,1], got {tuple(t.shape)}")
    if not 0.0 < float(t_eps) <= 1.0:
        raise ValueError("t_eps must be in (0, 1]")

    z_t = (1.0 - t) * x0 + t * noise
    target = (z_t - x0) / t.clamp(float(t_eps), 1.0)
    return z_t, target


def local_pmf_conditioning(t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the non-guided, zero-interval conditioning for local velocity."""

    return {
        "h": torch.zeros_like(t),
        "omega": torch.ones_like(t),
        "t_min": torch.zeros_like(t),
        "t_max": torch.ones_like(t),
    }


class PMFVelocityObjective(nn.Module):
    """DDP-compatible wrapper around a released pMF denoiser."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images_01: torch.Tensor,
        labels: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        apply_label_dropout: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x0 = images_01.mul(2.0).sub(1.0)
        batch_size = x0.shape[0]
        if t is None:
            t = self.model.sample_t(batch_size, x0.device)
        if noise is None:
            noise = torch.randn_like(x0) * float(self.model.noise_scale)

        z_t, target = pmf_velocity_target(
            x0,
            noise,
            t,
            t_eps=float(self.model.t_eps),
        )
        labels_used = labels
        if apply_label_dropout:
            labels_used, _ = self.model.cond_drop(target, target, labels)

        cond = local_pmf_conditioning(t)
        prediction = self.model.u_fn(
            z_t,
            t,
            cond["h"],
            cond["omega"],
            cond["t_min"],
            cond["t_max"],
            labels_used,
        )[0]
        residual = prediction.float() - target.float()
        loss = residual.square().mean()
        metrics = {
            "velocity_mse": loss.detach(),
            "velocity_rmse": loss.detach().sqrt(),
            "prediction_rms": prediction.float().square().mean().detach().sqrt(),
            "target_rms": target.float().square().mean().detach().sqrt(),
            "t_mean": t.float().mean().detach(),
            "t_min": t.float().amin().detach(),
            "t_max": t.float().amax().detach(),
            "label_drop_rate": labels_used.eq(self.model.num_classes).float().mean().detach(),
        }
        return loss, metrics
