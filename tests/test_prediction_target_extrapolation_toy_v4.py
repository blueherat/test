from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    DenoiseMLP,
    clean_from_output,
    direct_target,
    guided_clean,
    sample_condition,
    sample_conditions_batched,
    sample_mixture_conditions,
    velocity_from_output,
)
from experiments.summarize_prediction_target_toy_v4 import (
    aggregate_baseline_regimes,
    aggregate_contrasts,
    build_baseline_regimes,
    build_contrasts,
)


def test_prediction_target_conversions_recover_same_paired_clean_and_velocity() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 11, generator=generator)
    eps = torch.randn(3, 11, generator=generator)
    t = torch.tensor([0.1, 0.4, 0.9])
    x_t = (1.0 - t[:, None]) * x + t[:, None] * eps

    for target in ("x", "v", "eps"):
        output = direct_target(x, eps, target)
        recovered_x = clean_from_output(output, x_t, t, target, 1e-6)
        recovered_v = velocity_from_output(output, x_t, t, target, 1e-6)
        torch.testing.assert_close(recovered_x, x, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(recovered_v, eps - x, atol=2e-6, rtol=2e-6)


def test_equivalent_population_clean_estimate_has_zero_prediction_axis_gap() -> None:
    generator = torch.Generator().manual_seed(11)
    x_t = torch.randn(4, 9, generator=generator)
    x_bar = torch.randn(4, 9, generator=generator)
    t = torch.tensor([0.15, 0.35, 0.65, 0.85])

    outputs = {
        "x": x_bar,
        "v": (x_t - x_bar) / t[:, None],
        "eps": (x_t - (1.0 - t[:, None]) * x_bar) / t[:, None],
    }
    clean = {
        target: clean_from_output(output, x_t, t, target, 1e-6)
        for target, output in outputs.items()
    }
    torch.testing.assert_close(clean["x"], clean["v"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(clean["x"], clean["eps"], atol=2e-6, rtol=2e-6)


def test_curved_embedding_round_trip_and_tangent_normal_split() -> None:
    embedding = CurvedEmbedding(
        16,
        curvature=0.5,
        frequency_scale=4.0,
        seed=17,
        device=torch.device("cpu"),
        scale_mode="unit_rms",
    )
    generator = torch.Generator().manual_seed(19)
    u = torch.randn(5, 2, generator=generator)
    x = embedding.embed(u)
    torch.testing.assert_close(embedding.decode_intrinsic(x), u, atol=2e-5, rtol=2e-5)

    vec = torch.randn(5, 16, generator=generator)
    tangent, normal = embedding.split_tangent_normal(vec, u)
    tangent_gram, normal_gram = embedding.split_tangent_normal_gram(vec, u)
    torch.testing.assert_close(tangent + normal, vec, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(tangent_gram, tangent, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(normal_gram, normal, atol=2e-5, rtol=2e-5)
    inner = (tangent.double() * normal.double()).sum(dim=1)
    torch.testing.assert_close(inner, torch.zeros_like(inner), atol=2e-5, rtol=0)


class _CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, state: torch.Tensor, _time: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(state)


def test_guided_clean_only_evaluates_required_prediction_heads() -> None:
    embedding = CurvedEmbedding(
        4,
        curvature=0.0,
        frequency_scale=2.0,
        seed=23,
        device=torch.device("cpu"),
        scale_mode="constant_norm",
    )
    models = {target: _CountingModel() for target in ("x", "v", "eps")}
    state = torch.randn(2, 4)
    t = torch.full((2,), 0.5)

    guided_clean(
        models=models,
        embedding=embedding,
        state=state,
        t=t,
        kind="x",
        strength=0.0,
        clip=0.02,
    )
    assert [models[key].calls for key in ("x", "v", "eps")] == [1, 0, 0]

    guided_clean(
        models=models,
        embedding=embedding,
        state=state,
        t=t,
        kind="xv",
        strength=-0.03,
        clip=0.02,
    )
    assert [models[key].calls for key in ("x", "v", "eps")] == [2, 1, 0]


def test_grouped_mixture_sampling_matches_individual_trajectories() -> None:
    torch.manual_seed(29)
    embedding = CurvedEmbedding(
        8,
        curvature=0.3,
        frequency_scale=3.0,
        seed=31,
        device=torch.device("cpu"),
        scale_mode="unit_rms",
    )
    models = {
        target: DenoiseMLP(8, hidden=16, depth=3, time_dim=4)
        for target in ("x", "v", "eps")
    }
    strengths = (-0.1, 0.03, 0.1)
    common = {
        "models": models,
        "embedding": embedding,
        "sample_count": 12,
        "sample_batch_size": 4,
        "sample_steps": 5,
        "t_max": 0.9,
        "t_min": 0.1,
        "clip": 0.02,
        "seed": 37,
        "device": torch.device("cpu"),
    }
    grouped = sample_mixture_conditions(
        kind="xv", strengths=strengths, **common
    )
    individual = [
        sample_condition(
            oracle=models["x"], kind="xv", strength=strength, **common
        )
        for strength in strengths
    ]
    for batched, single in zip(grouped, individual):
        np.testing.assert_allclose(batched, single, rtol=2e-6, atol=2e-6)


def test_general_batched_sampling_matches_every_individual_condition() -> None:
    torch.manual_seed(41)
    embedding = CurvedEmbedding(
        8,
        curvature=0.4,
        frequency_scale=3.0,
        seed=43,
        device=torch.device("cpu"),
        scale_mode="unit_rms",
    )
    models = {
        target: DenoiseMLP(8, hidden=16, depth=3, time_dim=4)
        for target in ("x", "v", "eps")
    }
    conditions = [
        ("x", 0.0),
        ("v", 0.0),
        ("eps", 0.0),
        ("xv", 0.1),
        ("xeps", -0.03),
        ("xv_norm", 0.02),
        ("xv_tangent", 0.1),
        ("xv_normal", 0.1),
    ]
    common = {
        "models": models,
        "embedding": embedding,
        "sample_count": 12,
        "sample_batch_size": 4,
        "sample_steps": 5,
        "t_max": 0.9,
        "t_min": 0.1,
        "clip": 0.02,
        "seed": 47,
        "device": torch.device("cpu"),
    }
    batched = sample_conditions_batched(conditions=conditions, **common)
    individual = [
        sample_condition(
            oracle=models["x"], kind=kind, strength=strength, **common
        )
        for kind, strength in conditions
    ]
    for combined, single in zip(batched, individual):
        np.testing.assert_allclose(combined, single, rtol=3e-6, atol=3e-6)


def test_summary_distinguishes_interpolation_from_extrapolation() -> None:
    common = {
        "seed": 1,
        "D": 8,
        "curvature": 0.5,
        "hidden": 16,
        "loss_space": "v",
    }
    generation = pd.DataFrame(
        [
            {
                **common,
                "condition": "x",
                "kind": "x",
                "strength": 0.0,
                "swd_2d": 0.2,
                "swd_delta_vs_x_ci_low": 0.0,
                "swd_delta_vs_x_ci_high": 0.0,
                "mmd_2d": 0.02,
                "manifold_consistency_rms": 0.01,
            },
            {
                **common,
                "condition": "xv_gm0p03",
                "kind": "xv",
                "strength": -0.03,
                "swd_2d": 0.18,
                "swd_delta_vs_x_ci_low": -0.03,
                "swd_delta_vs_x_ci_high": -0.01,
                "mmd_2d": 0.01,
                "manifold_consistency_rms": 0.0105,
            },
            {
                **common,
                "condition": "xv_g0p03",
                "kind": "xv",
                "strength": 0.03,
                "swd_2d": 0.22,
                "swd_delta_vs_x_ci_low": 0.01,
                "swd_delta_vs_x_ci_high": 0.03,
                "mmd_2d": 0.03,
                "manifold_consistency_rms": 0.03,
            },
        ]
    )
    contrasts = build_contrasts(generation)
    assert contrasts.set_index("condition").loc["xv_gm0p03", "operation"] == "interpolation"
    assert contrasts.set_index("condition").loc["xv_g0p03", "operation"] == "extrapolation"
    aggregate = aggregate_contrasts(contrasts).set_index("condition")
    assert aggregate.loc["xv_gm0p03", "mean_delta_swd_vs_x"] < 0
    assert aggregate.loc["xv_gm0p03", "mean_delta_mmd_vs_x"] < 0
    assert aggregate.loc["xv_gm0p03", "joint_distribution_pass_seed_fraction"] == 1
    assert aggregate.loc["xv_g0p03", "mean_delta_swd_vs_x"] > 0
    assert aggregate.loc["xv_g0p03", "joint_distribution_pass_seed_fraction"] == 0


def test_baseline_regime_summary_checks_x_before_extrapolation() -> None:
    rows = []
    for seed, x_swd, v_swd in ((1, 0.20, 0.30), (2, 0.22, 0.28)):
        common = {
            "seed": seed,
            "D": 8,
            "curvature": 0.5,
            "hidden": 16,
            "loss_space": "v",
        }
        rows.extend(
            [
                {
                    **common,
                    "condition": "x",
                    "swd_2d": x_swd,
                    "mmd_2d": 0.01,
                    "manifold_consistency_rms": 0.02,
                },
                {
                    **common,
                    "condition": "v",
                    "swd_2d": v_swd,
                    "mmd_2d": 0.03,
                    "manifold_consistency_rms": 0.08,
                },
            ]
        )
    regimes = build_baseline_regimes(pd.DataFrame(rows))
    assert regimes["x_better_swd"].all()
    assert regimes["x_better_mmd"].all()
    aggregate = aggregate_baseline_regimes(regimes).iloc[0]
    assert aggregate["x_better_swd_seed_fraction"] == 1
    assert aggregate["x_better_mmd_seed_fraction"] == 1
    assert np.isclose(aggregate["mean_x_manifold_rms"], 0.02)
    assert np.isclose(aggregate["mean_v_over_x_manifold_rms"], 4.0)
