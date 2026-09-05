import numpy as np
import pytest
import torch
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_generated_classification import (
    balanced_labels,
    summarize_logits,
)


def test_balanced_labels_follow_sampler_protocol() -> None:
    assert balanced_labels(5, 3).tolist() == [0, 1, 2, 0, 1]


def test_summarize_logits_reports_requested_class_accuracy() -> None:
    logits = torch.tensor([[4.0, 0.0, -1.0], [0.0, 3.0, 1.0]])
    labels = torch.tensor([0, 2])
    metrics = summarize_logits(logits, labels)
    assert metrics["top1_accuracy"] == pytest.approx(0.5)
    assert metrics["top5_accuracy"] == pytest.approx(1.0)
    assert metrics["occupied_top1_classes"] == pytest.approx(2.0)
    assert np.isfinite(metrics["target_log_probability_mean"])


def test_summarize_logits_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        summarize_logits(torch.zeros(2, 3), torch.zeros(1, dtype=torch.long))
