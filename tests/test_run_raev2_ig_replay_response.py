from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.run_raev2_ig_replay_response import choose_gap, feedback_metrics


def test_choose_gap_separates_recursive_and_replay_branches() -> None:
    current = torch.tensor([[1.0], [2.0]])
    replay = torch.tensor([[3.0], [4.0]])
    result = choose_gap(current, replay, ("recursive", "replay"))
    assert torch.equal(result, torch.tensor([[1.0], [4.0]]))


def test_feedback_metrics_are_zero_when_replay_matches_recursive() -> None:
    baseline = np.zeros((3, 2), dtype=np.float64)
    positive = np.ones((3, 2), dtype=np.float64)
    negative = -positive
    metrics = feedback_metrics(
        baseline,
        positive,
        negative,
        positive,
        negative,
        gamma=0.5,
    )
    assert np.allclose(metrics["feedback_difference_per_gamma"], 0.0)
    assert np.allclose(metrics["feedback_fraction"], 0.0)
    assert np.allclose(metrics["recursive_replay_cosine"], 1.0)


def test_choose_gap_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="modes"):
        choose_gap(torch.ones(1, 1), torch.ones(1, 1), ("unknown",))
