from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
SPEC = importlib.util.spec_from_file_location(
    "dual_spiral_summary",
    EXPERIMENTS / "summarize_dual_target_closed_loop_spiral_toy.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _endpoint_rows() -> pd.DataFrame:
    rows = []
    conditions = [
        "Reference_resample",
        "D0_x_shared",
        "D0_eps_shared",
        "D3_oracle_bayes_gate",
    ]
    for seed in (1, 2):
        for condition_index, condition in enumerate(conditions):
            value = 0.1 + 0.01 * seed + 0.02 * condition_index
            rows.append(
                {
                    "seed": seed,
                    "ambient_dim": 2,
                    "hidden_dim": 8,
                    "condition": condition,
                    "swd_2d": value,
                    "swd_fullD": value * 2,
                    "ridge_distance_mean": value,
                    "ridge_width_ratio": 1.0 + value,
                    "arc_hist_tv": value,
                    "ambient_surface_rms": value,
                }
            )
    return pd.DataFrame(rows)


def test_endpoint_summary_keeps_seed_count() -> None:
    summary = MODULE.summarize_endpoint(_endpoint_rows())
    assert set(summary["seeds"]) == {2}
    assert "swd_fullD_mean" in summary


def test_mechanism_table_uses_common_head_controls() -> None:
    endpoint = _endpoint_rows()
    cross_rows = []
    for seed in (1, 2):
        for condition in ("D1_gate_on_D0", "D2_gate_on_D0", "D4_gate_on_D0"):
            cross_rows.append(
                {
                    "seed": seed,
                    "ambient_dim": 2,
                    "condition": condition,
                    "swd_2d": 0.2,
                    "swd_fullD": 0.4,
                }
            )
    mechanism = MODULE.build_mechanism_table(endpoint, pd.DataFrame(cross_rows))
    assert len(mechanism) == 2
    assert set(mechanism["D2_gate_on_D0_swd_fullD"]) == {0.4}
    assert mechanism["oracle_over_best_shared_fullD"].notna().all()
