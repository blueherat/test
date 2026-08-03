from __future__ import annotations

import numpy as np

from experiments.decode_raev2_ig_window_endpoints import classification_metrics, local_ids


def test_local_ids_partition_samples() -> None:
    joined = np.concatenate([local_ids(8, rank, 3) for rank in range(3)])
    np.testing.assert_array_equal(np.sort(joined), np.arange(8))


def test_classification_metrics_reward_correct_confident_logits() -> None:
    logits = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    result = classification_metrics(logits, labels)
    assert result["top1_accuracy"] == 1.0
    assert result["true_class_log_probability"] > -0.01
    assert result["maximum_probability"] > 0.99
