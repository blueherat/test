from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audit_imagenet100_sit_nfe import summarize_rows as summarize_batch_nfe
from experiments.summarize_imagenet100_sit_dual_mechanism import (
    fid_row,
    nfe_rows,
    validation_tables,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_nfe_summary_uses_batch_grain_and_preserves_rank_evidence(tmp_path: Path) -> None:
    write_json(
        tmp_path / "sampling_manifest.json",
        {
            "checkpoint_step": 10,
            "per_rank_batch_size": 4,
            "world_size": 2,
            "requested_samples": 14,
            "checkpoint_sha256": "baseline-hash",
        },
    )
    for rank, total_nfe in ((0, 30), (1, 34)):
        write_json(
            tmp_path / f"rank_{rank:02d}.json",
            {
                "rank": rank,
                "sample_count": 8,
                "total_nfe_across_batches": total_nfe,
                "elapsed_seconds": 2.0 + rank,
            },
        )
    rows, summary = nfe_rows(
        tmp_path,
        model_family="dual",
        checkpoint_step=10,
        mode="dynamic",
    )
    assert [row["mean_nfe_per_batch"] for row in rows] == [15.0, 17.0]
    assert summary["mean_nfe_per_batch"] == 16.0
    assert summary["padded_sample_count"] == 16
    assert "not recoverable" in str(summary["distribution_limitation"])


def test_fid_row_checks_checkpoint_and_carries_protocol_fields(tmp_path: Path) -> None:
    write_json(
        tmp_path / "sampling_manifest.json",
        {
            "checkpoint_step": 20,
            "requested_samples": 5000,
            "guidance": False,
            "checkpoint_sha256": "dual-hash",
        },
    )
    write_json(
        tmp_path / "fid5k_adm_results.json",
        {"fid": 7.0, "sfid": 8.0, "inception_score": 9.0},
    )
    row = fid_row(
        tmp_path,
        model_family="dual",
        checkpoint_step=20,
        mode="x",
    )
    assert row["sample_count"] == 5000
    assert row["guidance"] is False
    assert row["fid"] == 7.0


def test_validation_tables_derive_gate_std_from_first_two_moments(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    validation = {
        "clean_mse": 1.0,
        "epsilon_mse": 2.0,
        "gate_loss": 3.0,
        "gate_mean": 0.4,
        "gate_std": 0.2,
        "velocity_x_mse": 4.0,
        "velocity_epsilon_mse": 5.0,
        "velocity_dynamic_mse": 6.0,
        "time_bins": [
            {
                "t_min": 0.0,
                "t_max": 0.5,
                "count": 8,
                "gate_mean": 0.4,
                "gate_square": 0.2,
                "velocity_x_mse": 4.0,
                "velocity_epsilon_mse": 5.0,
                "velocity_dynamic_mse": 6.0,
            }
        ],
    }
    metrics.write_text(
        json.dumps(
            {"step": 100, "raw_validation": validation, "ema_validation": validation}
        )
        + "\n",
        encoding="utf-8",
    )
    summaries, bins = validation_tables(metrics, (100,))
    assert len(summaries) == 2
    assert len(bins) == 2
    assert bins[0]["gate_std_derived"] == pytest.approx(0.2)


def test_batch_nfe_summary_reports_full_distribution_quantiles() -> None:
    rows = [
        {"mode": mode, "nfe": value}
        for mode, values in {
            "velocity": (10, 12, 14),
            "x": (20, 22, 24),
            "epsilon": (30, 32, 34),
            "dynamic": (40, 42, 44),
        }.items()
        for value in values
    ]
    summaries = summarize_batch_nfe(rows)
    assert [row["mode"] for row in summaries] == [
        "velocity",
        "x",
        "epsilon",
        "dynamic",
    ]
    assert summaries[0]["mean_nfe"] == 12.0
    assert summaries[3]["q50_nfe"] == 42.0
