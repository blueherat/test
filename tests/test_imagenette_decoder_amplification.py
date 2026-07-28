import math

import pandas as pd
import torch

from experiments.analyze_imagenette_decoder_amplification import (
    condition_angles,
    decoder_forward_with_embedding,
    geodesic_step,
    hungarian_sphere_match,
    project_condition_sphere,
    random_tangent,
    tangent_toward,
)
from experiments.imagenette_noise_responsibility import ImagenetteConditionalUNet
from experiments.analyze_imagenette_decoder_witness_gap import grouped_domain_split
from experiments.summarize_imagenette_decoder_amplification import (
    evaluate_gates,
    prediction_table,
)


def test_condition_sphere_and_geodesic_step_preserve_constraints():
    generator = torch.Generator().manual_seed(11)
    base = project_condition_sphere(torch.randn(8, 32, generator=generator))
    target = project_condition_sphere(torch.randn(8, 32, generator=generator))
    tangent = tangent_toward(base, target)
    stepped = geodesic_step(base, tangent, 0.15)

    torch.testing.assert_close(base.mean(dim=1), torch.zeros(8), atol=2e-7, rtol=0)
    torch.testing.assert_close(
        base.square().mean(dim=1).sqrt(), torch.ones(8), atol=2e-7, rtol=0
    )
    torch.testing.assert_close(stepped.mean(dim=1), torch.zeros(8), atol=2e-7, rtol=0)
    torch.testing.assert_close(
        stepped.square().mean(dim=1).sqrt(), torch.ones(8), atol=2e-7, rtol=0
    )
    torch.testing.assert_close(
        condition_angles(base, stepped),
        torch.full((8,), 0.15),
        atol=2e-6,
        rtol=0,
    )


def test_random_tangent_is_centered_unit_and_orthogonal():
    base = project_condition_sphere(torch.randn(7, 24, generator=torch.Generator().manual_seed(3)))
    tangent = random_tangent(base, seed=17)
    base_unit = torch.nn.functional.normalize(base, dim=1)

    torch.testing.assert_close(tangent.mean(dim=1), torch.zeros(7), atol=2e-7, rtol=0)
    torch.testing.assert_close(tangent.norm(dim=1), torch.ones(7), atol=2e-7, rtol=0)
    torch.testing.assert_close(
        (base_unit * tangent).sum(dim=1), torch.zeros(7), atol=2e-7, rtol=0
    )


def test_hungarian_sphere_match_recovers_a_permutation():
    base = project_condition_sphere(torch.randn(9, 16, generator=torch.Generator().manual_seed(5)))
    permutation = torch.tensor([5, 0, 8, 2, 7, 1, 4, 6, 3])
    target = base[permutation]
    matched, indices = hungarian_sphere_match(base, target)

    torch.testing.assert_close(matched, base, atol=0, rtol=0)
    torch.testing.assert_close(target[indices], base, atol=0, rtol=0)


def test_external_condition_forward_matches_original_decoder_forward():
    torch.manual_seed(23)
    model = ImagenetteConditionalUNet(latent_dim=6, width=8).eval()
    value = torch.randn(3, 3, 16, 16)
    time = torch.tensor([0.2, 0.5, 0.9])
    condition = torch.randn(3, 6)
    embedding = model.condition_embedding(condition)

    expected = model(value, time, condition)
    observed = decoder_forward_with_embedding(model, value, time, embedding)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)


def test_tangent_step_moves_toward_target_for_a_small_angle():
    base = project_condition_sphere(torch.randn(6, 20, generator=torch.Generator().manual_seed(29)))
    target = project_condition_sphere(torch.randn(6, 20, generator=torch.Generator().manual_seed(31)))
    tangent = tangent_toward(base, target)
    original_angle = condition_angles(base, target)
    step_angle = torch.minimum(original_angle * 0.25, torch.full_like(original_angle, 0.1))
    radius = math.sqrt(base.shape[1])
    base_unit = torch.nn.functional.normalize(base.double(), dim=1)
    moved = radius * (
        step_angle.cos()[:, None] * base_unit
        + step_angle.sin()[:, None] * tangent.double()
    )
    moved = project_condition_sphere(moved.float())

    assert bool((condition_angles(moved, target) < original_angle).all())


def test_preregistered_gates_accept_a_decoder_aligned_synthetic_pattern():
    rows = []
    response = {16: 1.0, 64: 1.15, 256: 1.5}
    gap = {16: 2.0, 64: 8.0, 256: 20.0}
    for seed in range(5):
        for latent_dim in (16, 64, 256):
            prior_response = response[latent_dim] * (1.0 + 0.01 * seed)
            weighted = gap[latent_dim] + 0.05 * seed
            row = {
                "latent_dim": latent_dim,
                "frozen_seed": seed,
                "count": 256,
                "fixed_angle": 0.15,
                "modeling_gap": gap[latent_dim] + 0.1 * seed,
                "condition_prior_latent_sliced_wasserstein": 0.01 * math.sqrt(latent_dim),
                "condition_prior_matched_angle_mean": weighted / prior_response * 0.15,
                "decoder_weighted_mismatch": weighted,
                "prior_direction_feature_rms_mean": prior_response,
                "prior_direction_pixel_rms_mean": 0.1 * prior_response,
                "feature_rms_alignment_ratio": 1.2 if latent_dim == 256 else 1.0,
                "feature_rms_manifold_ratio": 1.2 if latent_dim == 256 else 1.0,
                "frozen_decoder_matches_formal": True,
            }
            for name in (
                "base",
                "prior_direction",
                "empirical_direction",
                "random_direction",
                "prior_endpoint",
            ):
                row[f"{name}_condition_abs_mean_max"] = 1e-7
                row[f"{name}_condition_rms_max_error"] = 1e-7
            for name in ("prior_direction", "empirical_direction", "random_direction"):
                row[f"{name}_fixed_angle_max_error"] = 1e-7
            for time in ("0p9", "0p5", "0p1"):
                row[f"prior_direction_velocity_rms_t{time}_mean"] = prior_response
            rows.append(row)
    table = pd.DataFrame(rows)
    prediction, _ = prediction_table(table)
    gates, paired = evaluate_gates(table, prediction)

    assert len(paired) == 5
    assert gates["implementation_audit"]
    assert gates["gate1_capacity_response"]
    assert gates["gate2_prior_direction_alignment"]
    assert gates["gate3_secondary_response"]
    assert gates["gate4_decoder_weighted_prediction"]
    assert gates["decoder_amplified_mismatch_supported"]


def test_grouped_domain_split_keeps_paired_noise_indices_together():
    index = torch.arange(40, dtype=torch.float64).numpy()[:, None]
    real = torch.from_numpy(index).repeat(1, 3).numpy()
    generated = real.copy()
    generated[:, 2] = 1.0
    train_x, train_y, test_x, test_y = grouped_domain_split(
        real, generated, seed=101, test_fraction=0.25
    )

    train_indices = set(train_x[:, 0].astype(int).tolist())
    test_indices = set(test_x[:, 0].astype(int).tolist())
    assert train_indices.isdisjoint(test_indices)
    assert train_indices | test_indices == set(range(40))
    assert int((train_y == 0).sum()) == int((train_y == 1).sum())
    assert int((test_y == 0).sum()) == int((test_y == 1).sum())
