from __future__ import annotations

import torch

from experiments.compute_raev2_predicted_clean_precision_recall import (
    manifold_coverage,
    manifold_radii,
    precision_recall,
)


def test_manifold_radii_exclude_self_neighbor() -> None:
    features = torch.tensor([[0.0], [1.0], [10.0]])
    radii = manifold_radii(features, neighborhood=1, batch_size=2)
    torch.testing.assert_close(radii, torch.tensor([1.0, 1.0, 9.0]))


def test_manifold_coverage_uses_union_of_reference_balls() -> None:
    centers = torch.tensor([[0.0], [1.0], [10.0]])
    radii = manifold_radii(centers, neighborhood=1, batch_size=2)
    queries = torch.tensor([[0.5], [20.0]])
    assert manifold_coverage(queries, centers, radii, batch_size=1) == 0.5


def test_identical_feature_sets_have_perfect_precision_and_recall() -> None:
    features = torch.tensor([[0.0], [1.0], [2.0], [4.0]])
    radii = manifold_radii(features, neighborhood=1, batch_size=2)
    precision, recall, _ = precision_recall(
        features,
        features,
        radii,
        neighborhood=1,
        batch_size=2,
    )
    assert precision == 1.0
    assert recall == 1.0
