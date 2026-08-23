from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.advfd_cleanroom.pmf_velocity_control import (
    PMFVelocityObjective,
    VelocityControlProtocol,
    local_pmf_conditioning,
    pmf_velocity_target,
)
from experiments.advfd_cleanroom.train_pmf_velocity_continuation import (
    save_checkpoint,
)
from experiments.advfd_cleanroom.eval_pmf_velocity_mse import (
    continuation_unseen_indices,
)


def test_pmf_velocity_target_matches_released_capped_conversion() -> None:
    x0 = torch.tensor([[[[-1.0, 0.5]]], [[[0.25, -0.75]]]])
    noise = torch.tensor([[[[0.5, -0.5]]], [[[1.0, 0.0]]]])
    t = torch.tensor([0.25, 0.01]).view(2, 1, 1, 1)
    z_t, target = pmf_velocity_target(x0, noise, t, t_eps=0.05)
    expected_z = (1.0 - t) * x0 + t * noise
    expected_target = (expected_z - x0) / t.clamp(0.05, 1.0)
    torch.testing.assert_close(z_t, expected_z)
    torch.testing.assert_close(target, expected_target)


def test_local_conditioning_is_h_zero_and_cfg_one() -> None:
    t = torch.rand(3, 1, 1, 1)
    cond = local_pmf_conditioning(t)
    torch.testing.assert_close(cond["h"], torch.zeros_like(t))
    torch.testing.assert_close(cond["omega"], torch.ones_like(t))
    torch.testing.assert_close(cond["t_min"], torch.zeros_like(t))
    torch.testing.assert_close(cond["t_max"], torch.ones_like(t))


class _PerfectCleanModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.noise_scale = 1.0
        self.t_eps = 0.05
        self.num_classes = 1000
        self.seen = None

    def cond_drop(self, target, _guided, labels):
        return labels, target

    def u_fn(self, z_t, t, h, omega, t_min, t_max, labels):
        del labels
        self.seen = SimpleNamespace(h=h, omega=omega, t_min=t_min, t_max=t_max)
        # The test supplies x0=2*images-1=0.  A perfect clean prediction then
        # converts to the released pMF velocity as z_t/clamp(t,t_eps).
        return z_t / t.clamp(self.t_eps, 1.0), torch.zeros_like(z_t)


def test_perfect_clean_prediction_has_zero_velocity_loss() -> None:
    model = _PerfectCleanModel()
    objective = PMFVelocityObjective(model)
    images = torch.full((2, 3, 2, 2), 0.5)
    labels = torch.tensor([1, 2])
    t = torch.tensor([0.2, 0.01]).view(2, 1, 1, 1)
    noise = torch.randn_like(images)
    loss, metrics = objective(
        images,
        labels,
        t=t,
        noise=noise,
        apply_label_dropout=False,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    torch.testing.assert_close(metrics["velocity_mse"], torch.zeros_like(loss), atol=1e-7, rtol=0)
    torch.testing.assert_close(model.seen.h, torch.zeros_like(t))
    torch.testing.assert_close(model.seen.omega, torch.ones_like(t))


def test_protocol_explicitly_excludes_fd_and_adaptive_critic() -> None:
    protocol = VelocityControlProtocol().to_dict()
    assert protocol["feature_distance_loss"] is False
    assert protocol["adaptive_feature_critic"] is False
    assert protocol["generated_image_training"] is False
    assert protocol["loss"] == "mean squared velocity error"


class _TinyEMA:
    def state_dict(self):
        return {"shadows": {"test": {"weight": torch.ones(1)}}}


def test_checkpoint_uses_public_step_semantics(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    path = tmp_path / "step_0005000.pth"
    save_checkpoint(
        path,
        model=model,
        ema_model=_TinyEMA(),
        optimizer=optimizer,
        step=5000,
        samples_seen=360_000,
        elapsed_seconds=1.0,
        protocol=VelocityControlProtocol().to_dict(),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 4999
    assert checkpoint["current_step"] == 5000


def test_continuation_unseen_bank_excludes_consumed_ddp_prefix() -> None:
    dataset_size = 23
    world_size = 4
    consumed = 12
    generator = torch.Generator().manual_seed(7)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    total_size = (dataset_size // world_size) * world_size
    expected = permutation[:total_size][consumed:consumed + 5]
    actual = continuation_unseen_indices(
        dataset_size=dataset_size,
        samples=5,
        samples_seen=consumed,
        sampler_seed=7,
        world_size=world_size,
    )
    assert actual == expected
    assert set(actual).isdisjoint(permutation[:consumed])
