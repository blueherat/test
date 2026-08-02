from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments.run_raev2_distribution_auc import (
    MomentAccumulator,
    SamplerStateRecorder,
    bootstrap_auc_delta,
    build_audit_time_rows,
    build_requested_labels,
    class_group_split,
    fit_diagonal_lda,
    match_requested_times,
    moment_distance_metrics,
    paired_auc,
    screening_conclusion,
    shifted_solver_grid,
)


def test_shifted_grid_and_time_matching_use_actual_solver_steps() -> None:
    grid = shifted_solver_grid(num_steps=100, shift=8.0)
    matched = match_requested_times((0.2, 0.4, 0.6, 0.8, 1.0), grid)

    assert grid[0].item() == 1.0
    assert grid[-1].item() == 0.0
    assert [row["requested_time"] for row in matched] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert matched[-1]["solver_index"] == 0
    assert matched[-1]["actual_time"] == 1.0
    assert len({row["solver_index"] for row in matched}) == len(matched)


def test_endpoint_time_is_not_treated_as_a_model_evaluation() -> None:
    grid = shifted_solver_grid(num_steps=100, shift=8.0)
    matched = build_audit_time_rows((0.0, 0.2, 1.0), grid, num_steps=100)
    endpoint = matched[0]
    assert endpoint == {
        "requested_time": 0.0,
        "solver_index": 100,
        "actual_time": 0.0,
        "absolute_time_error": 0.0,
    }
    assert [row["requested_time"] for row in matched[1:]] == [0.2, 1.0]


def test_class_split_has_no_class_leakage() -> None:
    labels = build_requested_labels(sample_count=2000, num_classes=1000)
    test_mask = class_group_split(labels, test_fraction=0.2, seed=7)

    train_classes = set(labels[~test_mask].tolist())
    test_classes = set(labels[test_mask].tolist())
    assert train_classes.isdisjoint(test_classes)
    assert len(train_classes) == 800
    assert len(test_classes) == 200
    assert int(test_mask.sum()) == 400


def test_diagonal_lda_is_heldout_linear_discriminator() -> None:
    generator = torch.Generator().manual_seed(11)
    negative_train = torch.randn(128, 8, generator=generator)
    positive_train = torch.randn(128, 8, generator=generator) + 1.5
    negative_test = torch.randn(64, 8, generator=generator)
    positive_test = torch.randn(64, 8, generator=generator) + 1.5

    negative_stats = MomentAccumulator()
    positive_stats = MomentAccumulator()
    negative_stats.update(negative_train)
    positive_stats.update(positive_train)
    weight, intercept, ridge = fit_diagonal_lda(
        negative_stats, positive_stats, ridge_ratio=1e-4
    )

    negative_scores = negative_test.matmul(weight).add(intercept).numpy()
    positive_scores = positive_test.matmul(weight).add(intercept).numpy()
    assert paired_auc(negative_scores, positive_scores) > 0.99
    assert ridge > 0
    assert torch.allclose(weight.norm(), torch.tensor(1.0), atol=1e-6)


def test_moment_distance_separates_mean_and_variance_changes() -> None:
    reference = MomentAccumulator()
    shifted = MomentAccumulator()
    reference.update(torch.tensor([[-1.0, -2.0], [1.0, 2.0]]))
    shifted.update(torch.tensor([[-1.0, -4.0], [3.0, 4.0]]))
    metrics = moment_distance_metrics(reference, shifted)
    assert metrics["mean_shift_rms"] > 0
    assert metrics["diagonal_variance_relative_l2"] > 0
    assert metrics["mean_variance_ratio"] > 1


def test_sampler_recorder_captures_model_inputs_not_outputs() -> None:
    captures: list[tuple[float, torch.Tensor]] = []

    def model_fn(x: torch.Tensor, t: torch.Tensor, **_kwargs) -> torch.Tensor:
        return x + 100.0

    matched = [
        {"requested_time": 1.0, "solver_index": 0, "actual_time": 1.0},
        {"requested_time": 0.5, "solver_index": 1, "actual_time": 0.5},
    ]
    recorder = SamplerStateRecorder(
        model_fn,
        matched,
        real_batch_size=2,
        callback=lambda key, value: captures.append((key, value.clone())),
    )
    x0 = torch.arange(12, dtype=torch.float32).view(4, 3)
    x1 = x0 + 1.0
    assert torch.equal(recorder(x0, torch.ones(4)), x0 + 100.0)
    assert torch.equal(recorder(x1, torch.full((4,), 0.5)), x1 + 100.0)
    recorder.validate(expected_calls=2)

    assert [key for key, _ in captures] == [1.0, 0.5]
    assert torch.equal(captures[0][1], x0[:2])
    assert torch.equal(captures[1][1], x1[:2])


def test_bootstrap_delta_compares_separate_probe_scores() -> None:
    p_full = np.array([-2.0, -1.0, -0.5, 0.0])
    q_full = np.array([0.5, 1.0, 1.5, 2.0])
    p_ig = np.array([-0.1, 0.0, 0.1, 0.2])
    q_ig = np.array([-0.05, 0.05, 0.15, 0.25])

    low, high = bootstrap_auc_delta(
        p_full,
        q_full,
        p_ig,
        q_ig,
        repeats=100,
        seed=3,
    )
    assert low <= high
    assert high <= 0.0


def test_screening_prioritizes_sign_reversal_over_majority_count() -> None:
    deltas = pd.DataFrame(
        {
            "requested_time": [0.2, 0.4, 0.6, 0.8, 1.0],
            "delta_ci_low": [0.03, -0.04, -0.07, -0.06, 0.0],
            "delta_ci_high": [0.11, 0.04, -0.001, -0.007, 0.0],
        }
    )
    conclusion = screening_conclusion(deltas, heldout_pairs=200)
    assert conclusion.startswith("phase-dependent sign reversal")
