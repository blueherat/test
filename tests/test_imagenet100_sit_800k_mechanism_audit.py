from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.summarize_imagenet100_sit_800k_mechanism_audit as audit


def _row(scale: float, *, fid: float = 60.0) -> dict:
    return {
        "scale": scale,
        "fid": fid,
        "sfid": 68.0,
        "inception_score": 27.0,
        "sampling_elapsed_seconds_max_rank": 10.0,
        "checkpoint_step": 800_000,
        "other_checkpoint_step": 400_000,
        "num_samples": 5_000,
        "total_nfe": 100,
        "total_model_forwards": 200,
        "noise_fingerprint": "noise",
        "label_fingerprint": "labels",
    }


def _write_series(root: Path, name: str, rows: list[dict]) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "field_control_fid5k.json").write_text(
        json.dumps({"rows": rows}),
        encoding="utf-8",
    )


def test_800k_loader_uses_requested_weak_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "MECHANISM_COUNTS", {"floor_only": 1})
    _write_series(tmp_path, "floor_only", [_row(-1.0)])
    _write_series(
        tmp_path,
        "same_target_v400",
        [_row(scale) for scale in (1.0, -0.2, -0.5, -0.75, -1.0)],
    )

    rows, missing = audit.load_rows(tmp_path, (400,), allow_incomplete=False)

    assert missing == []
    assert len(rows) == 6
    assert {row["series"] for row in rows} == {"floor_only", "same_target_v400"}


def test_800k_loader_rejects_incomplete_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "MECHANISM_COUNTS", {})
    _write_series(tmp_path, "same_target_v400", [_row(1.0)])

    with pytest.raises(ValueError, match="expected 5 rows"):
        audit.load_rows(tmp_path, (400,), allow_incomplete=False)
