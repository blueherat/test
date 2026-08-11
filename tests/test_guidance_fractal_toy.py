from __future__ import annotations

import torch

from experiments.run_guidance_fractal_toy import (
    InternalScoreView,
    InternalToyModel,
    ag_toy,
    evaluate_samples,
    log_probabilities,
)


def test_internal_model_scores_have_expected_shapes() -> None:
    model = InternalToyModel(
        hidden_dim=8, num_layers=2, intermediate_after=1
    ).eval()
    x = torch.randn(5, 2)
    sigma = torch.full((5,), 0.5)
    intermediate, final = model.scores(x, sigma)
    assert intermediate.shape == x.shape
    assert final.shape == x.shape
    assert torch.isfinite(intermediate).all()
    assert torch.isfinite(final).all()


def test_internal_guidance_endpoints_are_exact() -> None:
    model = InternalToyModel(
        hidden_dim=8, num_layers=2, intermediate_after=1
    ).eval()
    with torch.no_grad():
        model.intermediate_gain.fill_(0.1)
        model.final_gain.fill_(0.2)
    x = torch.randn(7, 2)
    sigma = torch.full((7,), 0.3)
    intermediate, final = model.scores(x, sigma)
    output_weak = InternalScoreView(model, 0.0).score(x, sigma)
    output_strong = InternalScoreView(model, 1.0).score(x, sigma)
    assert torch.allclose(output_weak, intermediate)
    assert torch.allclose(output_strong, final)


def test_internal_guidance_is_affine_extrapolation() -> None:
    model = InternalToyModel(
        hidden_dim=8, num_layers=2, intermediate_after=1
    ).eval()
    with torch.no_grad():
        model.intermediate_gain.fill_(0.1)
        model.final_gain.fill_(0.2)
    x = torch.randn(7, 2)
    sigma = torch.full((7,), 0.3)
    intermediate, final = model.scores(x, sigma)
    output = InternalScoreView(model, 2.0).score(x, sigma)
    assert torch.allclose(output, intermediate + 2.0 * (final - intermediate))


def test_evaluation_rejects_nonfinite_samples() -> None:
    import numpy as np
    import pytest

    distribution = ag_toy.gt("A", torch.device("cpu"))
    reference = np.zeros((8, 2), dtype=np.float32)
    invalid = reference.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="invalid"):
        evaluate_samples(
            samples={"reference": reference, "invalid": invalid},
            reference_name="reference",
            distribution=distribution,
            seed=0,
        )


def test_stable_mixture_logp_stays_finite_for_far_outliers() -> None:
    import numpy as np

    distribution = ag_toy.gt("A", torch.device("cpu"))
    samples = np.asarray([[1e3, -1e3], [-1e3, 1e3]], dtype=np.float32)
    values = log_probabilities(distribution, samples)
    assert np.isfinite(values).all()


def test_stable_mixture_logp_matches_official_on_distribution() -> None:
    import numpy as np

    distribution = ag_toy.gt("A", torch.device("cpu"))
    generator = torch.Generator().manual_seed(7)
    samples = distribution.sample(32, generator=generator)
    stable = log_probabilities(distribution, samples.numpy())
    official = distribution.logp(samples).numpy()
    assert np.allclose(stable, official, atol=2e-5, rtol=2e-5)
