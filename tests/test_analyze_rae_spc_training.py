from __future__ import annotations

import pandas as pd

from experiments.analyze_rae_spc_training import audit_pairing, window_rows


def _pair(mismatch: bool = False) -> pd.DataFrame:
    rows = []
    for condition in ("static", "spc"):
        for step in (2000, 2010, 3000, 4000, 5000):
            rows.append(
                {
                    "condition": condition,
                    "step": step,
                    "target_energy": 2.0 + (0.1 if mismatch and condition == "spc" else 0.0),
                    "semantic_energy": 1.0,
                    "detail_energy": 0.1,
                    "mean_time": 0.5,
                    "lr": 1e-4,
                    "loss": 0.9 if condition == "static" else 0.8,
                    "grad_norm": 0.3,
                }
            )
    return pd.DataFrame(rows)


def test_pairing_audit_requires_exact_model_independent_rows() -> None:
    assert audit_pairing(_pair(), 2000)["model_independent_stream_exact"]
    assert not audit_pairing(_pair(mismatch=True), 2000)["model_independent_stream_exact"]


def test_window_rows_preserve_paired_loss_direction() -> None:
    rows = window_rows(_pair(), seed=11, switch_step=2000, endpoint=5000)
    assert len(rows) == 3
    assert all(row["spc_minus_static_loss"] < 0 for row in rows)
