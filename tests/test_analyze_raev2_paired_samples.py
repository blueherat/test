import numpy as np
import pytest

from experiments.analyze_raev2_paired_samples import paired_metrics


def test_paired_metrics_distinguishes_flow_and_opposing_increment() -> None:
    origin = np.full((3, 2, 2, 3), 100, dtype=np.uint8)
    flow = np.full((3, 2, 2, 3), 110, dtype=np.uint8)
    same_as_flow = flow.copy()
    opposite = np.full((3, 2, 2, 3), 105, dtype=np.uint8)

    control_metrics = paired_metrics(
        origin=origin,
        control=flow,
        candidate=same_as_flow,
        chunk_size=2,
    )
    opposing_metrics = paired_metrics(
        origin=origin,
        control=flow,
        candidate=opposite,
        chunk_size=2,
    )

    assert control_metrics["mae_to_flow_mean"] == pytest.approx(0.0)
    assert control_metrics["total_update_cosine_with_flow_mean"] == pytest.approx(1.0)
    assert np.isnan(control_metrics["lpl_increment_cosine_with_flow_mean"])
    assert opposing_metrics["total_update_cosine_with_flow_mean"] == pytest.approx(1.0)
    assert opposing_metrics["lpl_increment_cosine_with_flow_mean"] == pytest.approx(-1.0)
    assert opposing_metrics["mae_to_flow_over_flow_update"] == pytest.approx(0.5)


def test_paired_metrics_rejects_shape_mismatch() -> None:
    origin = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="identical shapes"):
        paired_metrics(
            origin=origin,
            control=origin,
            candidate=np.zeros((1, 2, 2, 3), dtype=np.uint8),
        )
