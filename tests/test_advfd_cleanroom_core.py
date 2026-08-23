import math

import torch

from experiments.advfd_cleanroom.core import (
    EMAMomentTracker,
    StreamingMomentAccumulator,
    batch_moments,
    calibrate_features,
    calibrate_moments,
    fit_calibration,
    fit_calibration_from_moments,
    frechet_from_features,
    project_moments,
)


def _features(seed: int, samples: int = 512, dimension: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        samples, dimension, generator=generator, dtype=torch.float64
    )


def test_identical_features_have_zero_frechet_distance() -> None:
    features = _features(1)
    distance = frechet_from_features(features, features)
    assert distance.mean.item() == 0.0
    assert distance.covariance.item() < 1e-10


def test_translation_only_changes_mean_component() -> None:
    features = _features(2)
    shift = torch.tensor([0.5, -0.2, 0.1, 0.0, 0.3], dtype=torch.float64)
    distance = frechet_from_features(features, features + shift)
    assert torch.allclose(distance.mean, shift.square().sum(), atol=1e-12)
    assert distance.covariance.item() < 1e-10


def test_raw_frechet_has_quadratic_scaling_degeneracy() -> None:
    real = _features(3)
    generated = 1.2 * _features(4) + 0.25
    original = frechet_from_features(real, generated).total
    scaled = frechet_from_features(7.0 * real, 7.0 * generated).total
    assert torch.allclose(scaled, 49.0 * original, rtol=1e-9, atol=1e-9)


def test_real_whitening_is_invariant_to_common_invertible_affine_map() -> None:
    real = _features(5, samples=2048, dimension=4)
    generated = 1.15 * _features(6, samples=2048, dimension=4) + 0.2
    matrix = torch.tensor(
        [
            [1.5, 0.2, -0.1, 0.3],
            [0.1, 0.8, 0.4, -0.2],
            [0.0, -0.3, 1.2, 0.1],
            [0.2, 0.1, 0.0, 0.9],
        ],
        dtype=torch.float64,
    )
    offset = torch.tensor([0.3, -0.4, 0.2, 1.0], dtype=torch.float64)

    calibrated = calibrate_features(
        real, generated, mode="real", epsilon=1e-10
    )
    transformed = calibrate_features(
        real @ matrix + offset,
        generated @ matrix + offset,
        mode="real",
        epsilon=1e-10,
    )
    first = frechet_from_features(*calibrated).total
    second = frechet_from_features(*transformed).total
    assert torch.allclose(first, second, rtol=2e-7, atol=2e-7)


def test_pooled_whitening_is_invariant_to_common_invertible_affine_map() -> None:
    real = _features(7, samples=2048, dimension=4)
    generated = 0.85 * _features(8, samples=2048, dimension=4) - 0.15
    matrix = torch.tensor(
        [
            [1.1, 0.2, 0.0, 0.1],
            [-0.1, 0.9, 0.3, 0.2],
            [0.2, 0.0, 1.3, -0.2],
            [0.1, -0.2, 0.1, 0.7],
        ],
        dtype=torch.float64,
    )
    offset = torch.tensor([-0.5, 0.1, 0.4, 0.2], dtype=torch.float64)

    calibrated = calibrate_features(
        real, generated, mode="pooled", epsilon=1e-10
    )
    transformed = calibrate_features(
        real @ matrix + offset,
        generated @ matrix + offset,
        mode="pooled",
        epsilon=1e-10,
    )
    first = frechet_from_features(*calibrated).total
    second = frechet_from_features(*transformed).total
    assert torch.allclose(first, second, rtol=2e-7, atol=2e-7)


def test_ema_history_is_detached_but_current_batch_keeps_gradients() -> None:
    tracker = EMAMomentTracker(decay=0.9)
    tracker.initialize(_features(9, samples=64, dimension=3))
    current = _features(10, samples=32, dimension=3).requires_grad_(True)
    effective = tracker.update(current)
    loss = effective.mean.square().sum() + effective.covariance.square().sum()
    loss.backward()
    assert current.grad is not None
    assert current.grad.abs().sum().item() > 0.0
    state = tracker.state_dict()
    assert not state["mean"].requires_grad
    assert not state["second"].requires_grad


def test_real_whitened_mean_supremum_equals_pearson_chi_square() -> None:
    p = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    q = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)
    ratio_residual = q / p - 1.0
    chi_square = (p * ratio_residual.square()).sum()
    witness = ratio_residual / chi_square.sqrt()
    assert abs((p * witness).sum().item()) < 1e-12
    assert torch.allclose((p * witness.square()).sum(), p.new_tensor(1.0))
    assert torch.allclose((q * witness).sum().square(), chi_square)


def test_fisher_pooled_supremum_is_symmetric_and_support_safe() -> None:
    p = torch.tensor([1.0, 0.0], dtype=torch.float64)
    q = torch.tensor([0.0, 1.0], dtype=torch.float64)
    mixture = 0.5 * (p + q)
    ratio_witness = (p - q) / mixture
    fisher_squared = ((p - q).square() / mixture).sum()
    witness = ratio_witness / fisher_squared.sqrt()
    assert torch.allclose(
        (mixture * witness.square()).sum(), mixture.new_tensor(1.0)
    )
    assert torch.allclose(((p - q) * witness).sum().square(), fisher_squared)
    assert fisher_squared.item() == 4.0

    # Real-only normalization leaves the fake-only value unconstrained.
    for fake_value in (10.0, 100.0, 1000.0):
        real_normalized_objective = (1.0 - fake_value) ** 2
        pooled_rayleigh = (1.0 - fake_value) ** 2 / (
            0.5 * (1.0 + fake_value**2)
        )
        assert real_normalized_objective > fake_value
        assert pooled_rayleigh <= 4.0 + 1e-12


def test_batch_moments_use_population_covariance() -> None:
    features = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    moments = batch_moments(features)
    assert moments.mean.item() == 1.0
    assert moments.second.item() == 2.0
    assert moments.covariance.item() == 1.0


def test_real_whitening_makes_real_moments_identity() -> None:
    real = _features(11, samples=1024, dimension=4)
    generated = _features(12, samples=1024, dimension=4)
    calibrated_real, _ = calibrate_features(
        real, generated, mode="real", epsilon=1e-10
    )
    moments = batch_moments(calibrated_real)
    assert moments.mean.norm().item() < 1e-10
    assert math.isclose(
        moments.covariance.trace().item(), 4.0, rel_tol=1e-8, abs_tol=1e-8
    )


def test_fitted_calibration_can_be_reused_on_heldout_features() -> None:
    real = _features(13, samples=256, dimension=3)
    generated = _features(14, samples=256, dimension=3) + 0.3
    heldout = _features(15, samples=64, dimension=3)
    calibration = fit_calibration(
        real, generated, mode="real", epsilon=1e-8
    )
    assert calibration.apply(heldout).shape == heldout.shape
    assert not calibration.center.requires_grad
    assert not calibration.transform.requires_grad


def test_moment_space_projection_matches_projected_features() -> None:
    features = _features(16, samples=128, dimension=5)
    projection = _features(17, samples=5, dimension=3)
    expected = batch_moments(features @ projection)
    actual = project_moments(batch_moments(features), projection)
    assert torch.allclose(actual.mean, expected.mean, atol=1e-12)
    assert torch.allclose(actual.second, expected.second, atol=1e-12)
    assert torch.allclose(actual.covariance, expected.covariance, atol=1e-12)


def test_moment_space_calibration_matches_calibrated_features() -> None:
    real = _features(18, samples=512, dimension=4)
    generated = 1.1 * _features(19, samples=512, dimension=4) + 0.2
    calibration = fit_calibration_from_moments(
        batch_moments(real),
        batch_moments(generated),
        mode="real",
        epsilon=1e-8,
    )
    expected = batch_moments(calibration.apply(generated))
    actual = calibrate_moments(batch_moments(generated), calibration)
    assert torch.allclose(actual.mean, expected.mean, atol=1e-10)
    assert torch.allclose(actual.covariance, expected.covariance, atol=1e-10)


def test_ema_preview_does_not_advance_until_commit() -> None:
    tracker = EMAMomentTracker(decay=0.8)
    tracker.initialize(_features(20, samples=64, dimension=3))
    state_before = tracker.state_dict()
    current = _features(21, samples=32, dimension=3).requires_grad_(True)
    preview = tracker.preview(current)
    state_after_preview = tracker.state_dict()
    assert torch.equal(state_before["mean"], state_after_preview["mean"])
    assert torch.equal(state_before["second"], state_after_preview["second"])
    preview.mean.sum().backward()
    assert current.grad is not None and current.grad.abs().sum().item() > 0.0
    tracker.commit(preview)
    state_after_commit = tracker.state_dict()
    assert torch.equal(preview.mean.detach(), state_after_commit["mean"])
    assert torch.equal(preview.second.detach(), state_after_commit["second"])


def test_streaming_moments_match_concatenated_features() -> None:
    first = _features(22, samples=37, dimension=6)
    second = _features(23, samples=53, dimension=6)
    accumulator = StreamingMomentAccumulator(dtype=torch.float64)
    accumulator.update(first[:11])
    accumulator.update(first[11:])
    accumulator.update(second)
    actual = accumulator.moments(dtype=torch.float64)
    expected = batch_moments(torch.cat([first, second], dim=0))
    assert accumulator.count == 90
    assert torch.allclose(actual.mean, expected.mean, atol=1e-12)
    assert torch.allclose(actual.second, expected.second, atol=1e-12)
    assert torch.allclose(actual.covariance, expected.covariance, atol=1e-12)
