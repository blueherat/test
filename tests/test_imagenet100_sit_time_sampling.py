from __future__ import annotations

import torch

from experiments.imagenet100_sit_time_sampling import (
    sample_time_values,
    time_distribution_metadata,
    validate_time_sampling,
)


def test_uniform_sampler_is_the_original_torch_rand_path() -> None:
    expected_generator = torch.Generator().manual_seed(31)
    actual_generator = torch.Generator().manual_seed(31)
    expected = torch.rand(64, generator=expected_generator)
    actual = sample_time_values(
        64,
        device="cpu",
        time_sampler="uniform",
        generator=actual_generator,
    )
    assert torch.equal(actual, expected)


def test_logit_normal_sampler_matches_official_jit_formula_exactly() -> None:
    expected_generator = torch.Generator().manual_seed(37)
    actual_generator = torch.Generator().manual_seed(37)
    expected = torch.sigmoid(torch.randn(64, generator=expected_generator) * 0.8 - 0.8)
    actual = sample_time_values(
        64,
        device="cpu",
        time_sampler="logit_normal",
        logit_mean=-0.8,
        logit_std=0.8,
        generator=actual_generator,
    )
    assert torch.equal(actual, expected)
    assert bool(((actual > 0.0) & (actual < 1.0)).all())


def test_jit_distribution_rarely_reaches_the_clamped_x_endpoint() -> None:
    generator = torch.Generator().manual_seed(41)
    times = sample_time_values(
        1_000_000,
        device="cpu",
        time_sampler="logit_normal",
        logit_mean=-0.8,
        logit_std=0.8,
        generator=generator,
    )
    assert int((times > 0.95).sum()) < 10
    weight = times.new_tensor(1.0) / (1.0 - times).clamp_min(0.05).square()
    assert 2.9 < float(weight.mean()) < 3.1


def test_time_sampler_validation_and_metadata() -> None:
    validate_time_sampling("uniform", -0.8, 0.8)
    metadata = time_distribution_metadata("logit_normal", -0.8, 0.8)
    assert metadata["name"] == "logit_normal"
    assert metadata["logit_mean"] == -0.8
    assert metadata["logit_std"] == 0.8
    for sampler, mean, std in (("bad", -0.8, 0.8), ("uniform", -0.8, 0.0)):
        try:
            validate_time_sampling(sampler, mean, std)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid time-sampling configuration was accepted")
