from __future__ import annotations

import numpy as np
import torch

from experiments.run_imagenet100_sit_finite_guidance import (
    _aggregate_tangent_frozen,
    build_parser,
)
from experiments.summarize_imagenet100_sit_tangent_frozen import _bootstrap_mean


def _synthetic_shard() -> dict[str, object]:
    baseline = torch.zeros(2, 1, 2)
    tangent = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
    gammas = torch.tensor([-0.01, 0.0, 0.01, 1.0])
    responses = torch.stack([float(gamma) * tangent for gamma in gammas])
    return {
        "gammas": gammas,
        "baseline": baseline,
        "feedback_baseline": baseline.clone(),
        "tangent": tangent,
        "frozen": responses,
        "closed": responses.clone(),
    }


def test_tangent_frozen_aggregation_accepts_exact_linear_response() -> None:
    rows, summary = _aggregate_tangent_frozen([_synthetic_shard()], 0.01)

    assert summary["central_difference_pass"]
    assert summary["largest_passing_positive_frozen_gamma"] == 1.0
    assert summary["frozen_at_gamma_one"]["cosine_mean"] == 1.0
    assert summary["frozen_at_gamma_one"]["relative_residual_mean"] == 0.0
    assert {row["response"] for row in rows} == {"frozen", "closed"}


def test_parser_exposes_800k_tangent_frozen_protocol() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--study", "tangent_frozen", "--direction", "v500"]
    )

    assert args.study == "tangent_frozen"
    assert args.direction == "v500"
    assert args.v500_checkpoint.name == "step_00500000.pt"


def test_bootstrap_summary_is_exact_for_constant_input() -> None:
    summary = _bootstrap_mean(
        np.ones(8, dtype=np.float64),
        np.random.default_rng(7),
        reps=100,
    )

    assert summary["mean"] == 1.0
    assert summary["ci95"] == [1.0, 1.0]
    assert summary["probability_gt_zero"] == 1.0
