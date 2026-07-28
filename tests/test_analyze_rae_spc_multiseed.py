from __future__ import annotations

import pandas as pd

from experiments.analyze_rae_spc_multiseed import (
    exact_sign_flip_pvalue,
    paired_table,
    summarize,
)


def _table(deltas: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for seed, (fid_delta, kid_delta) in enumerate(deltas):
        rows.extend(
            [
                {
                    "seed": seed,
                    "condition": "static",
                    "frechet_inception_distance": 100.0,
                    "kernel_inception_distance_mean": 0.1,
                    "inception_score_mean": 5.0,
                },
                {
                    "seed": seed,
                    "condition": "spc",
                    "frechet_inception_distance": 100.0 + fid_delta,
                    "kernel_inception_distance_mean": 0.1 + kid_delta,
                    "inception_score_mean": 5.2,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_generation_gate_passes_consistent_effect() -> None:
    summary = summarize(paired_table(_table([(-10, -0.01)] * 5)))
    assert summary["both_fid_kid_better_count"] == 5
    assert summary["generation_gate_pass"]


def test_generation_gate_rejects_inconsistent_effect() -> None:
    summary = summarize(
        paired_table(_table([(-10, -0.01), (-8, -0.01), (2, 0.01), (4, 0.01), (5, 0.02)]))
    )
    assert not summary["generation_gate_pass"]


def test_exact_sign_flip_reaches_five_seed_resolution() -> None:
    assert exact_sign_flip_pvalue([-1, -1, -1, -1, -1]) == 1 / 32
