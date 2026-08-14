from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.summarize_imagenet100_sit_800k_compact_replication import (
    select_matched_weak,
    summarize_final,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _static_row(
    scale: float,
    fid: float,
    *,
    other_step: int,
    fingerprint: str = "paired",
) -> dict:
    return {
        "scale": scale,
        "fid": fid,
        "sfid": fid + 1,
        "inception_score": 20.0,
        "checkpoint_step": 800_000,
        "other_checkpoint_step": other_step,
        "noise_fingerprint": fingerprint,
        "label_fingerprint": fingerprint + "-labels",
        "total_nfe": 100,
        "sampling_peak_memory_mib": 1000,
        "fid_peak_memory_mib": 2000,
    }


def _static_payload(rows: list[dict], checkpoint: Path) -> dict:
    return {"other": {"checkpoint": str(checkpoint)}, "rows": rows}


def test_quality_match_selects_closest_endpoint(tmp_path: Path) -> None:
    x_checkpoint = _write(tmp_path / "x.pt", {})
    v400 = _write(tmp_path / "v400.pt", {})
    v500 = _write(tmp_path / "v500.pt", {})
    x_summary = _write(
        tmp_path / "x.json",
        _static_payload([_static_row(1, 64.0, other_step=800_000)], x_checkpoint),
    )
    v400_summary = _write(
        tmp_path / "v400.json",
        _static_payload([_static_row(1, 67.0, other_step=400_000)], v400),
    )
    v500_summary = _write(
        tmp_path / "v500.json",
        _static_payload([_static_row(1, 64.5, other_step=500_000)], v500),
    )
    args = argparse.Namespace(
        x_endpoint_summary=x_summary,
        v_candidate=[(400, v400_summary), (500, v500_summary)],
        output=tmp_path / "selection.json",
    )

    selected = select_matched_weak(args)

    assert selected["matched"]["label"] == "v500"
    assert selected["matched"]["absolute_fid_gap_to_x800"] == pytest.approx(0.5)


def _frozen_payload(fid: float, other_step: int, fingerprint: str) -> dict:
    return {
        "fid": fid,
        "sfid": fid + 1,
        "inception_score": 20.0,
        "noise_fingerprint": fingerprint,
        "label_fingerprint": fingerprint + "-labels",
        "anchor": {"checkpoint_step": 800_000},
        "other": {"checkpoint_step": other_step},
        "total_nfe": 100,
        "sampling_peak_memory_mib": 1000,
        "fid_peak_memory_mib": 2000,
    }


def test_final_summary_verifies_pairing_and_retention(tmp_path: Path) -> None:
    selection = _write(
        tmp_path / "quality_match.json",
        {"matched": {"checkpoint_step": 500_000, "label": "v500"}},
    )
    for seed in (0, 1):
        fingerprint = f"seed-{seed}"
        root = tmp_path / f"seed{seed}"
        _write(
            root / "x_pair/field_control_fid5k.json",
            {
                "rows": [
                    _static_row(0, 60.0, other_step=800_000, fingerprint=fingerprint),
                    _static_row(-1, 54.0, other_step=800_000, fingerprint=fingerprint),
                ]
            },
        )
        _write(
            root / "vweak_closed/field_control_fid5k.json",
            {
                "rows": [
                    _static_row(-1, 55.0, other_step=500_000, fingerprint=fingerprint)
                ]
            },
        )
        _write(
            root / "x_frozen/frozen_guidance_fid5k.json",
            _frozen_payload(56.0, 800_000, fingerprint),
        )
        _write(
            root / "vweak_frozen/frozen_guidance_fid5k.json",
            _frozen_payload(56.5, 500_000, fingerprint),
        )

    result = summarize_final(argparse.Namespace(root=tmp_path, selection=selection))

    assert result["acceptance"]["passed"] is True
    assert result["seed_metrics"][0]["x_frozen_retention"] == pytest.approx(4 / 6)
    assert result["seed_metrics"][0]["vweak_frozen_retention"] == pytest.approx(3.5 / 5)


def test_final_summary_rejects_unpaired_condition(tmp_path: Path) -> None:
    selection = _write(
        tmp_path / "quality_match.json",
        {"matched": {"checkpoint_step": 500_000, "label": "v500"}},
    )
    root = tmp_path / "seed0"
    _write(
        root / "x_pair/field_control_fid5k.json",
        {
            "rows": [
                _static_row(0, 60.0, other_step=800_000),
                _static_row(-1, 54.0, other_step=800_000),
            ]
        },
    )
    _write(
        root / "vweak_closed/field_control_fid5k.json",
        {"rows": [_static_row(-1, 55.0, other_step=500_000)]},
    )
    _write(
        root / "x_frozen/frozen_guidance_fid5k.json",
        _frozen_payload(56.0, 800_000, "different"),
    )
    _write(
        root / "vweak_frozen/frozen_guidance_fid5k.json",
        _frozen_payload(56.5, 500_000, "paired"),
    )

    with pytest.raises(ValueError, match="not noise/label paired"):
        summarize_final(argparse.Namespace(root=tmp_path, selection=selection))
