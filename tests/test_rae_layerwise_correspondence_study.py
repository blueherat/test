import torch

from baselines.dinov2_token_diagnostics import P
from experiments.rae_layerwise_correspondence_study import (
    effective_rank_from_gram,
    layer_pair_metrics,
    residual_energy_fraction,
    residual_spatial_gram,
)


def _permutations(tokens: int):
    return [torch.arange(tokens - 1, -1, -1), torch.roll(torch.arange(tokens), 1)]


def test_perfect_spatial_equivariance_has_exact_correspondence():
    generator = torch.Generator().manual_seed(4)
    base = torch.randn((3, 8, 4, 4), generator=generator)
    target = P(base, "flip_h")
    metrics = layer_pair_metrics(base, target, "flip_h", _permutations(16))

    assert float(metrics["residual_direct_error"].max()) < 1e-7
    torch.testing.assert_close(metrics["diag_cosine"], torch.ones(3))
    torch.testing.assert_close(metrics["exact_match_rate"], torch.ones(3))
    torch.testing.assert_close(metrics["within_1_rate"], torch.ones(3))
    assert float(metrics["random_permutation_error"].min()) > 0.5


def test_spatially_homogeneous_tokens_are_exposed_by_noncollapse_controls():
    value = torch.randn((2, 6, 1, 1), generator=torch.Generator().manual_seed(5))
    base = value.expand(-1, -1, 4, 4).clone()
    target = P(base, "rot90")
    metrics = layer_pair_metrics(base, target, "rot90", _permutations(16))
    gram = residual_spatial_gram(base)

    assert float(residual_energy_fraction(base).max()) < 1e-12
    assert effective_rank_from_gram(gram) == 0.0
    assert float(metrics["residual_direct_error"].max()) == 0.0
    # Relative centered errors are ill-conditioned when residual energy is zero;
    # rank/energy and correspondence controls must reject this false appearance.
    assert torch.isfinite(metrics["random_permutation_error"]).all()
    torch.testing.assert_close(
        metrics["exact_match_rate"],
        metrics["random_exact_match_rate"],
    )


def test_effective_rank_matches_identity_gram():
    gram = torch.eye(9, dtype=torch.float64)
    assert abs(effective_rank_from_gram(gram) - 9.0) < 1e-10
