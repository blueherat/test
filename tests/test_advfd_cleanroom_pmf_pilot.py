from types import SimpleNamespace

import torch

from experiments.advfd_cleanroom.core import batch_moments
from experiments.advfd_cleanroom.run_pmf_pilot import (
    adaptive_components,
    adaptive_scale,
    balanced_evaluation_labels,
    covariance_diagnostics,
    evaluation_artifact_names,
    fixed_projection,
    per_sample_evaluation_noise,
    quantize_unit_images,
    resolve_adaptive_feature_dim,
    resolve_adaptive_eval_samples,
    scheduled_lr,
)


def test_scaled_adaptive_schedule_has_explicit_start_and_warmup() -> None:
    args = SimpleNamespace(
        adaptive_start=10,
        adaptive_warmup=4,
        adaptive_weight=0.05,
    )
    assert adaptive_scale(args, 9) == 0.0
    assert adaptive_scale(args, 10) == 0.0125
    assert adaptive_scale(args, 13) == 0.05
    assert adaptive_scale(args, 100) == 0.05


def test_cosine_schedule_starts_at_peak_without_warmup_and_ends_at_zero() -> None:
    assert scheduled_lr(1e-6, 0, total=5, warmup=0) == 1e-6
    assert scheduled_lr(1e-6, 4, total=5, warmup=0) == 0.0


def test_short_prefix_uses_full_schedule_horizon() -> None:
    assert scheduled_lr(1e-6, 999, total=125_000, warmup=6_250) == 1.6e-7
    assert scheduled_lr(1e-6, 6_249, total=125_000, warmup=6_250) == 1e-6


def test_evaluation_artifact_names_preserve_default_and_isolate_tagged_runs() -> None:
    assert evaluation_artifact_names(None) == ("evaluation.json", "samples_grid.png")
    assert evaluation_artifact_names("step000040_5k") == (
        "evaluation_step000040_5k.json",
        "samples_grid_step000040_5k.png",
    )


def test_evaluation_artifact_names_reject_path_like_tags() -> None:
    for tag in ("", "../escape", "nested/tag", "has space"):
        try:
            evaluation_artifact_names(tag)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid evaluation tag: {tag!r}")


def test_adaptive_eval_count_defaults_to_full_eval_and_supports_strict_subset() -> None:
    assert resolve_adaptive_eval_samples(5000, None) == 5000
    assert resolve_adaptive_eval_samples(5000, 512) == 512
    for requested in (0, 5001):
        try:
            resolve_adaptive_eval_samples(5000, requested)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid adaptive sample count: {requested}")


def test_full_dimension_projection_is_exact_identity() -> None:
    projection = fixed_projection(5, 5, seed=123, device=torch.device("cpu"))
    torch.testing.assert_close(projection, torch.eye(5), rtol=0, atol=0)


def test_separate_adaptive_dimension_requires_full_static_features() -> None:
    assert resolve_adaptive_feature_dim(2048, 512) == 512
    assert resolve_adaptive_feature_dim(64, None) == 64
    try:
        resolve_adaptive_feature_dim(64, 32)
    except ValueError:
        pass
    else:
        raise AssertionError("separate adaptive projection must use a full warm-start")


def test_balanced_evaluation_labels_match_pmf_class_major_order() -> None:
    labels = balanced_evaluation_labels(
        start=3,
        count=6,
        total=20,
        num_classes=4,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1, 1, 1]))


def test_per_sample_evaluation_noise_is_batch_partition_invariant() -> None:
    first = per_sample_evaluation_noise(
        start=0,
        count=4,
        sample_shape=(2, 3),
        initial_seed=42,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    split = torch.cat(
        [
            per_sample_evaluation_noise(
                start=0,
                count=1,
                sample_shape=(2, 3),
                initial_seed=42,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
            per_sample_evaluation_noise(
                start=1,
                count=3,
                sample_shape=(2, 3),
                initial_seed=42,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
        ]
    )
    torch.testing.assert_close(first, split, rtol=0, atol=0)


def test_quantize_unit_images_matches_uint8_round_trip() -> None:
    images = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1])
    expected = torch.tensor([0.0, 0.0, 128.0 / 255.0, 1.0, 1.0])
    torch.testing.assert_close(quantize_unit_images(images), expected)


def test_real_whitened_adaptive_components_are_finite_and_differentiable() -> None:
    generator = torch.Generator().manual_seed(7)
    real_features = torch.randn(128, 6, generator=generator)
    fake_features = (
        1.1 * torch.randn(128, 6, generator=generator) + 0.2
    ).requires_grad_(True)
    components, calibrated_real, calibrated_fake = adaptive_components(
        batch_moments(real_features),
        batch_moments(fake_features),
        variant="real",
    )
    components.total.backward()
    assert torch.isfinite(components.total)
    assert fake_features.grad is not None
    assert fake_features.grad.abs().sum() > 0
    assert calibrated_real.mean.shape == calibrated_fake.mean.shape == (6,)


def test_float64_diagnostic_moments_survive_large_common_feature_offset() -> None:
    generator = torch.Generator().manual_seed(19)
    features = 10_000.0 + torch.randn(4096, 8, generator=generator)
    features64 = features.to(torch.float64)

    moments = batch_moments(features64)
    centered = features64 - features64.mean(dim=0)
    expected_covariance = centered.mT @ centered / features64.shape[0]

    torch.testing.assert_close(
        moments.covariance,
        expected_covariance,
        rtol=1e-6,
        atol=1e-6,
    )
    diagnostics = covariance_diagnostics(moments.covariance)
    assert diagnostics["minimum_eigenvalue"] > 0.0
    assert diagnostics["negative_eigenvalues"] == 0
