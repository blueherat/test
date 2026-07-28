from __future__ import annotations

import torch

from experiments.run_rae_predictability_lpl_proxy_gate import build_block_metric


def test_block_metric_is_positive_and_trace_normalized() -> None:
    basis = torch.eye(16)
    blocks = {
        f"fractional_{start:03d}_{start + 1:03d}": basis[:, start : start + 2]
        for start in range(0, 16, 2)
    }
    weights = {name: 0.4 + index * 0.05 for index, name in enumerate(blocks)}
    metric = build_block_metric(blocks, weights, floor=0.3)
    assert torch.linalg.eigvalsh(metric).min() > 0.0
    torch.testing.assert_close(metric.diagonal().mean(), torch.tensor(1.0))
