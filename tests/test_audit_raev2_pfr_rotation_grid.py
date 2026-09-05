from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_pfr_rotation_grid import direction_metrics  # noqa: E402


def test_direction_metrics_distinguish_parallel_and_orthogonal_change() -> None:
    depth = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    revision = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    metrics = direction_metrics(depth, revision)
    torch.testing.assert_close(
        metrics["revision_orthogonal_energy_fraction"], torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(
        metrics["additive_rotation_degrees"], torch.tensor([0.0, 45.0])
    )
    torch.testing.assert_close(
        metrics["orthogonal_rotation_degrees"], torch.tensor([0.0, 45.0])
    )
