import math

import torch

from experiments.latent_transport_audit_metrics import (
    AnisotropicChannelTransform,
    IdentityLatentTransform,
    LatentSketch,
    SignedChannelOrthogonalTransform,
    apply_linear_or_jvp,
    bootstrap_spearman,
    knn_overlap,
    local_velocity_ambiguity,
    projected_shared_class_viv,
    sliced_wasserstein_1,
    spearman_correlation,
)
from experiments.latent_transport_paths import bridge_commutation_defect


def _latent(seed=0, shape=(12, 8, 4, 4), dtype=torch.float64):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_controlled_linear_maps_are_exactly_invertible_and_chord_preserving():
    data = _latent(1)
    noise = _latent(2)
    time = torch.linspace(0.05, 0.95, len(data), dtype=data.dtype)
    transforms = (
        IdentityLatentTransform(),
        SignedChannelOrthogonalTransform(8, seed=3),
        AnisotropicChannelTransform(8, condition_number=8.0, seed=4),
    )
    for transform in transforms:
        torch.testing.assert_close(transform.inverse(transform(data)), data)
        defect = bridge_commutation_defect(data, noise, time, transform)
        assert float(defect.max()) < 1e-12


def test_signed_permutation_preserves_norm_and_anisotropic_map_has_requested_condition():
    value = _latent(5)
    orthogonal = SignedChannelOrthogonalTransform(8, seed=6)
    torch.testing.assert_close(
        orthogonal(value).flatten(1).norm(dim=1),
        value.flatten(1).norm(dim=1),
    )
    anisotropic = AnisotropicChannelTransform(8, condition_number=5.0, seed=7)
    observed = float(anisotropic.scales.max() / anisotropic.scales.min())
    assert math.isclose(observed, 5.0, rel_tol=1e-6)
    assert abs(float(torch.log(anisotropic.scales).sum())) < 1e-6


def test_latent_sketch_is_linear_and_standardizes_iid_noise_scale():
    first = _latent(8, shape=(2048, 8, 8, 8), dtype=torch.float32)
    second = _latent(9, shape=first.shape, dtype=torch.float32)
    sketch = LatentSketch(8, projected_channels=4, spatial_size=2, seed=10)
    torch.testing.assert_close(
        sketch(0.3 * first + 0.7 * second),
        0.3 * sketch(first) + 0.7 * sketch(second),
        atol=2e-6,
        rtol=2e-6,
    )
    assert abs(float(sketch(first).std(unbiased=False)) - 1.0) < 0.05


def test_projected_shared_class_viv_matches_isotropic_closed_form():
    generator = torch.Generator().manual_seed(11)
    classes = 8
    per_class = 512
    dimension = 6
    labels = torch.arange(classes).repeat_interleave(per_class)
    means = torch.randn((classes, dimension), generator=generator) * 3.0
    values = means[labels] + torch.randn(
        (classes * per_class, dimension), generator=generator
    )
    metrics = projected_shared_class_viv(values, labels)
    expected_per_dim = math.pi / 2.0
    assert abs(metrics["projected_viv_per_dim"] - expected_per_dim) < 0.08
    assert metrics["degrees_of_freedom"] == classes * (per_class - 1)


def test_local_velocity_ambiguity_detects_predictable_field():
    state = torch.linspace(-2.0, 2.0, 256).unsqueeze(1)
    state = torch.cat((state, state.square()), dim=1)
    velocity = torch.stack((2.0 * state[:, 0], -state[:, 0]), dim=1)
    predictable = local_velocity_ambiguity(state, velocity, neighbors=4)
    shuffled = local_velocity_ambiguity(
        state,
        velocity[torch.randperm(len(velocity), generator=torch.Generator().manual_seed(12))],
        neighbors=4,
    )
    assert predictable["ambiguity_ratio"] < 0.01
    assert shuffled["ambiguity_ratio"] > 0.5


def test_knn_overlap_and_sliced_wasserstein_controls():
    generator = torch.Generator().manual_seed(13)
    value = torch.randn((256, 12), generator=generator)
    overlap = knn_overlap(value, value.clone(), neighbors=8)
    assert overlap["recall"] == 1.0
    assert overlap["jaccard"] == 1.0
    assert sliced_wasserstein_1(value, value.clone(), seed=14) == 0.0
    shifted = value + 2.0
    assert sliced_wasserstein_1(value, shifted, seed=14) > 0.1


def test_spearman_and_bootstrap_are_directionally_correct():
    left = [0, 1, 2, 3, 4, 5]
    right = [0, 2, 1, 4, 3, 6]
    assert spearman_correlation(left, right) > 0.8
    result = bootstrap_spearman(left, right, resamples=500, seed=15)
    assert result.correlation > 0.8
    assert result.ci_low > 0.0


def test_linear_jvp_shortcut_matches_transform_direction():
    point = _latent(16)
    direction = _latent(17)
    transform = AnisotropicChannelTransform(8, condition_number=3.0, seed=18)
    value, tangent = apply_linear_or_jvp(transform, point, direction)
    torch.testing.assert_close(value, transform(point))
    torch.testing.assert_close(tangent, transform(direction))
