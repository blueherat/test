import torch

from experiments.nonlinear_fm_whitening_toy import (
    MixtureFMConfig,
    ResidualMLP,
    sample_fm_batch,
)
from experiments.nonlinear_raw_velocity_gap import (
    _weighted_raw_loss,
    raw_rollout,
)


def test_gamma_zero_raw_loss_is_plain_velocity_mse():
    problem = MixtureFMConfig(
        variance=(0.2, 2.0),
        bimodal_fraction=(0.0, 0.9),
        decoder_gain=(1.0, 1.0),
    )
    batch = sample_fm_batch(
        problem,
        32,
        "cpu",
        torch.Generator().manual_seed(0),
    )
    prediction = torch.randn(batch["velocity"].shape, generator=torch.Generator().manual_seed(1))
    actual = _weighted_raw_loss(
        prediction,
        batch,
        gamma=0.0,
        damping=1e-4,
        normalizer=1.0,
    )
    expected = (prediction - batch["velocity"]).square().mean()
    assert torch.allclose(actual, expected)


def test_raw_rollout_is_finite_on_cpu():
    problem = MixtureFMConfig(
        variance=(0.2, 2.0),
        bimodal_fraction=(0.0, 0.9),
        decoder_gain=(1.0, 1.0),
    )
    model = ResidualMLP(problem.dimension, hidden_size=8, depth=1)
    samples = raw_rollout(
        model,
        problem,
        sample_count=16,
        ode_steps=10,
        seed=2,
        device=torch.device("cpu"),
    )
    assert samples.shape == (16, 2)
    assert torch.isfinite(samples).all()
