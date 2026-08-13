from __future__ import annotations

import json

import pytest

import experiments.summarize_imagenet100_sit_400k_mechanism_audit as audit


def make_row(*, scale: float = -1.0, noise: str = "noise", labels: str = "labels") -> dict:
    return {
        "series": "synthetic",
        "scale": scale,
        "fid": 60.0,
        "sfid": 68.0,
        "inception_score": 27.0,
        "sampling_elapsed_seconds_max_rank": 10.0,
        "checkpoint_step": 400_000,
        "other_checkpoint_step": 300_000,
        "num_samples": 5_000,
        "total_nfe": 100,
        "total_model_forwards": 200,
        "noise_fingerprint": noise,
        "label_fingerprint": labels,
    }


def test_verify_pairing_rejects_nonpaired_noise_or_wrong_sample_count() -> None:
    reference = make_row(scale=0.0)
    audit.verify_pairing([make_row()], reference_row=reference)

    with pytest.raises(ValueError, match="noise fingerprint"):
        audit.verify_pairing([make_row(noise="different")], reference_row=reference)
    wrong_count = make_row()
    wrong_count["num_samples"] = 4_999
    with pytest.raises(ValueError, match="not a 5K result"):
        audit.verify_pairing([wrong_count], reference_row=reference)


def test_load_audit_rows_enforces_registered_series_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "EXPECTED_AUDIT_COUNTS", {"floor_only": 2})
    result_dir = tmp_path / "floor_only"
    result_dir.mkdir()
    (result_dir / "field_control_fid5k.json").write_text(
        json.dumps({"rows": [make_row(scale=-0.5), make_row(scale=-1.0)]}),
        encoding="utf-8",
    )

    rows, missing = audit.load_audit_rows(tmp_path, allow_incomplete=False)

    assert missing == []
    assert [row["series"] for row in rows] == ["floor_only", "floor_only"]

    (result_dir / "field_control_fid5k.json").write_text(
        json.dumps({"rows": [make_row()]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected 2 rows"):
        audit.load_audit_rows(tmp_path, allow_incomplete=False)


def test_incomplete_audit_is_explicitly_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "EXPECTED_AUDIT_COUNTS", {"floor_only": 1})

    rows, missing = audit.load_audit_rows(tmp_path, allow_incomplete=True)

    assert rows == []
    assert missing == ["floor_only"]
    with pytest.raises(FileNotFoundError, match="incomplete 400K audit"):
        audit.load_audit_rows(tmp_path, allow_incomplete=False)


def test_write_csv_keeps_a_header_when_no_results_exist(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    audit.write_csv(path, [])
    assert path.read_text(encoding="utf-8").strip() == "series"


def test_complete_summary_figure_can_be_rendered(tmp_path) -> None:
    full_rows = [
        audit.normalize_row(make_row(scale=scale), series="x400_full_pair")
        for scale in (-1.0, -0.75, -0.5, -0.2, 0.0, 1.0)
    ]
    for index, row in enumerate(full_rows):
        row["fid"] = 59.0 + index
    rows = [
        audit.normalize_row(make_row(scale=-1.0), series="floor_only"),
        audit.normalize_row(make_row(scale=-1.0), series="orthogonal_pair"),
    ]
    for series in ("same_target_v240", "same_target_v270", "same_target_v300"):
        for scale in (1.0, -0.2, -0.5, -0.75, -1.0):
            rows.append(audit.normalize_row(make_row(scale=scale), series=series))

    output = tmp_path / "summary.png"
    audit.plot_summary(output, rows, full_rows, baseline_fid=64.0)

    assert output.is_file()
    assert output.stat().st_size > 1_000
