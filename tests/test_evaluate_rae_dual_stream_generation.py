from __future__ import annotations

import pandas as pd

from experiments.evaluate_rae_dual_stream_generation import build_gate


def test_full_dual_stream_gate_requires_all_three_fid_comparisons() -> None:
    table = pd.DataFrame(
        [
            {"condition": "static", "frechet_inception_distance": 100.0, "kernel_inception_distance_mean": 0.10},
            {"condition": "semantic_only", "frechet_inception_distance": 110.0, "kernel_inception_distance_mean": 0.11},
            {"condition": "paired_detail", "frechet_inception_distance": 90.0, "kernel_inception_distance_mean": 0.09},
            {"condition": "shuffled_detail", "frechet_inception_distance": 105.0, "kernel_inception_distance_mean": 0.105},
        ]
    )
    gate = build_gate(table)
    assert gate["gate_pass"] is True
    assert gate["decision"] == "continue_dual_stream"
