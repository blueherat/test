import math

import torch

from experiments.advfd_cleanroom.audit_pmf_generator_mode_coverage import (
    class_occupancy,
    fixed_space_coverage_metrics,
    manifold_radii,
)


def test_identical_cloud_has_full_precision_recall_and_coverage() -> None:
    points = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    radii = manifold_radii(points, neighborhood=1, batch_size=2)
    metrics = fixed_space_coverage_metrics(
        points,
        points,
        generated_radii=radii,
        reference_radii=radii,
        neighborhood=1,
        batch_size=2,
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["coverage"] == 1.0
    assert metrics["density"] >= 1.0


def test_collapsed_cloud_loses_recall() -> None:
    reference = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    generated = torch.tensor([[0.0], [0.01], [0.02], [0.03]])
    reference_radii = manifold_radii(reference, neighborhood=1, batch_size=2)
    generated_radii = manifold_radii(generated, neighborhood=1, batch_size=2)
    metrics = fixed_space_coverage_metrics(
        generated,
        reference,
        generated_radii=generated_radii,
        reference_radii=reference_radii,
        neighborhood=1,
        batch_size=2,
    )
    assert metrics["recall"] < 1.0
    assert metrics["coverage"] < 1.0


def test_class_occupancy_detects_single_class_collapse() -> None:
    logits = torch.full((8, 4), -10.0)
    logits[:, 2] = 10.0
    result = class_occupancy(logits)
    assert result["occupied_top1_classes"] == 1
    assert result["max_top1_class_fraction"] == 1.0
    assert math.isclose(result["top1_class_entropy_normalized"], 0.0, abs_tol=1e-10)
