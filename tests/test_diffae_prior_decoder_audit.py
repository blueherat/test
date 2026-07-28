import torch

from experiments.diffae_prior_decoder_audit import (
    covariance_relative_error,
    extract_complete_ema_state,
    fit_full_covariance_gaussian,
    feature_distribution_metrics,
    kernel_inception_distance,
    latent_distribution_metrics,
    mean_cosine_distance,
    sample_matched_gaussian,
    score_decoded_feature_sets,
    select_empirical_latents,
    standardized_sliced_wasserstein,
)


def test_empirical_selection_is_seeded_without_replacement():
    values = torch.arange(400, dtype=torch.float32).reshape(100, 4)
    first, first_indices = select_empirical_latents(values, 20, seed=7)
    second, second_indices = select_empirical_latents(values, 20, seed=7)
    assert torch.equal(first, second)
    assert torch.equal(first_indices, second_indices)
    assert len(torch.unique(first_indices)) == 20


def test_full_covariance_gaussian_recovers_correlated_moments():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(20_000, 3, generator=generator)
    transform = torch.tensor([[2.0, 0.0, 0.0], [1.2, 0.7, 0.0], [-0.4, 0.3, 0.2]])
    values = base @ transform.T + torch.tensor([2.0, -1.0, 0.5])
    mean = values.mean(0)
    std = values.std(0, unbiased=False)
    normalized_mean, factor = fit_full_covariance_gaussian(values, mean, std)
    samples = sample_matched_gaussian(
        20_000, normalized_mean, factor, mean, std, seed=13
    )
    assert torch.linalg.norm(samples.mean(0) - values.mean(0)) < 0.05
    assert covariance_relative_error(values, samples) < 0.04


def test_distribution_metrics_are_neutral_for_exact_match():
    values = torch.randn(500, 12, generator=torch.Generator().manual_seed(17))
    assert standardized_sliced_wasserstein(values, values.clone(), seed=19) == 0.0
    assert covariance_relative_error(values, values.clone()) == 0.0
    metrics = latent_distribution_metrics(values, values.clone(), seed=23)
    assert metrics["standardized_swd"] == 0.0
    assert metrics["mean_shift_relative"] == 0.0
    assert metrics["covariance_relative_error"] == 0.0
    assert 0.40 <= metrics["c2st_accuracy"] <= 0.60


def test_checkpoint_extractor_requires_encoder_decoder_and_prior():
    expected = {
        "encoder.weight": torch.zeros(2, 2),
        "latent_net.weight": torch.zeros(2, 2),
        "input_blocks.0.weight": torch.zeros(2, 2),
        "output_blocks.0.weight": torch.zeros(2, 2),
    }
    checkpoint = {
        "global_step": 9,
        "state_dict": {f"ema_model.{key}": value for key, value in expected.items()},
    }
    extracted, metadata = extract_complete_ema_state(checkpoint, expected)
    assert extracted.keys() == expected.keys()
    assert metadata["global_step"] == 9
    broken = {"state_dict": {"ema_model.encoder.weight": torch.zeros(2, 2)}}
    try:
        extract_complete_ema_state(broken, expected)
    except RuntimeError as error:
        assert "incomplete DiffAE EMA checkpoint" in str(error)
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def test_feature_metrics_detect_a_large_distribution_shift():
    generator = torch.Generator().manual_seed(29)
    reference = torch.randn(600, 32, generator=generator)
    exact_kid, _ = kernel_inception_distance(reference, reference.clone(), seed=31)
    shifted = reference + 2.0
    shifted_kid, _ = kernel_inception_distance(reference, shifted, seed=31)
    metrics = feature_distribution_metrics(reference, shifted, seed=37)
    assert shifted_kid > exact_kid
    assert metrics["standardized_swd"] > 1.0
    assert metrics["c2st_accuracy"] > 0.9


def test_feature_gap_scoring_uses_paired_randomness_and_orders_quality():
    generator = torch.Generator().manual_seed(41)
    real = torch.randn(600, 32, generator=generator)
    decoded = {
        "empirical": real + 0.1 * torch.randn(600, 32, generator=generator),
        "prior": real + 0.8,
        "gaussian": real + 1.6,
    }
    first_comparisons, first_gaps = score_decoded_feature_sets(real, decoded, seed=43)
    second_comparisons, second_gaps = score_decoded_feature_sets(real, decoded, seed=43)
    assert first_comparisons == second_comparisons
    assert first_gaps == second_gaps
    assert first_gaps["prior_gap_kid_mean"] > 0
    assert first_gaps["gaussian_gap_kid_mean"] > first_gaps["prior_gap_kid_mean"]


def test_paired_cosine_distance_is_zero_only_for_identical_features():
    values = torch.randn(100, 16, generator=torch.Generator().manual_seed(47))
    assert abs(mean_cosine_distance(values, values.clone())) < 1e-7
    assert mean_cosine_distance(values, values.roll(1, dims=0)) > 0.5
