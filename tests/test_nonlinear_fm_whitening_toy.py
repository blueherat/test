import numpy as np
import torch

from experiments.nonlinear_fm_whitening_toy import (
    MixtureFMConfig,
    NeuralTrainConfig,
    fm_statistics,
    residual_weight_normalizer,
    sample_fm_batch,
    train_neural_model,
)


def test_gaussian_coordinates_have_zero_nonlinear_residual_mean():
    problem = MixtureFMConfig(
        variance=(0.2, 2.0),
        bimodal_fraction=(0.0, 0.0),
        decoder_gain=(1.0, 1.0),
    )
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((128, 2), generator=generator)
    t = torch.rand((128, 1), generator=generator)
    statistics = fm_statistics(x, t, problem)
    assert torch.allclose(
        statistics["conditional_residual"],
        torch.zeros_like(x),
        atol=2e-6,
        rtol=2e-6,
    )


def test_mixture_conditional_velocity_matches_explicit_posterior_average():
    problem = MixtureFMConfig(
        variance=(0.7, 3.0),
        bimodal_fraction=(0.35, 0.92),
        decoder_gain=(1.0, 1.0),
    )
    x = torch.tensor(
        [[-2.1, 0.4], [-0.3, 1.7], [0.0, -1.2], [1.6, 3.0]],
        dtype=torch.float64,
    )
    t = torch.tensor([[0.17], [0.43], [0.68], [0.89]], dtype=torch.float64)
    actual = fm_statistics(x, t, problem)["conditional_velocity"]

    variance = torch.tensor(problem.variance, dtype=torch.float64)[None]
    fraction = torch.tensor(problem.bimodal_fraction, dtype=torch.float64)[None]
    mixture_mean = torch.sqrt(variance * fraction)
    component_variance = variance * (1.0 - fraction)
    a, b = 1.0 - t, t
    input_variance = a.square() * component_variance + b.square()

    conditional_by_sign = []
    log_weights = []
    for sign in (-1.0, 1.0):
        signed_mean = sign * mixture_mean
        centered = x - a * signed_mean
        posterior_z = signed_mean + a * component_variance / input_variance * centered
        posterior_noise = b / input_variance * centered
        conditional_by_sign.append(posterior_noise - posterior_z)
        log_weights.append(-0.5 * centered.square() / input_variance)
    weights = torch.softmax(torch.stack(log_weights, dim=-1), dim=-1)
    expected = torch.sum(torch.stack(conditional_by_sign, dim=-1) * weights, dim=-1)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_linear_skip_residual_is_orthogonal_and_has_predicted_variance():
    problem = MixtureFMConfig()
    count = 200_000
    generator = torch.Generator().manual_seed(1)
    batch = sample_fm_batch(problem, count, "cpu", generator)
    selected = torch.isclose(batch["t"][:, 0], batch["t"][:, 0])
    assert selected.all()

    # Use a fixed time so the population covariance has one analytic value.
    tensors = sample_fm_batch(problem, count, "cpu", torch.Generator().manual_seed(2))
    fixed_t = torch.full((count, 1), 0.63)
    a, b = 1.0 - fixed_t, fixed_t
    x = a * tensors["latent"] + b * tensors["noise"]
    velocity = tensors["noise"] - tensors["latent"]
    statistics = fm_statistics(x, fixed_t, problem)
    residual = velocity - statistics["linear_skip"] * x
    cross = torch.mean(residual * x, dim=0)
    empirical_variance = torch.mean(residual.square(), dim=0)
    predicted_variance = statistics["residual_variance"][0]

    assert torch.max(torch.abs(cross)).item() < 0.025
    assert torch.allclose(empirical_variance, predicted_variance, rtol=0.025, atol=0.01)


def test_weight_normalizer_has_unit_expected_weight():
    problem = MixtureFMConfig()
    gamma = 0.65
    damping = 1e-4
    normalizer = residual_weight_normalizer(problem, gamma, damping)
    t = torch.linspace(problem.t_min, problem.t_max, 4097, dtype=torch.float64)[:, None]
    variance = torch.tensor(problem.variance, dtype=torch.float64)[None]
    residual_variance = variance / ((1.0 - t).square() * variance + t.square())
    weight = torch.pow(residual_variance + damping, -gamma) / normalizer
    assert np.isclose(float(weight.mean()), 1.0, atol=1e-12)


def test_mlp_and_mini_dit_training_smoke_on_cpu():
    problem = MixtureFMConfig(
        variance=(0.2, 2.0),
        bimodal_fraction=(0.0, 0.9),
        decoder_gain=(2.0, 1.0),
    )
    for architecture in ("mlp", "mini_dit"):
        config = NeuralTrainConfig(
            architecture=architecture,
            gamma=0.5,
            batch_size=16,
            steps=4,
            learning_rate=1e-3,
            hidden_size=16,
            depth=1,
            num_heads=4,
            eval_every=2,
            eval_count=32,
            device="cpu",
            seed=3,
        )
        run = train_neural_model(problem, config)
        assert len(run.history) == 3
        assert np.isfinite(run.history["excess_mse"]).all()
        assert np.isfinite(run.summary["final_excess_mse"])
