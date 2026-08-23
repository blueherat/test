import csv
import json
from pathlib import Path

from experiments.advfd_cleanroom.summarize_pmf_prefix import evaluation_rows


def _write_json(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluation_rows_join_training_metrics_and_compute_paired_delta(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "base" / "evaluation_paired5k.json",
        fid_inception_2048_small_sample=10.0,
        fid_inception_2048_float64=10.1,
        balanced_eval_labels=True,
        per_sample_eval_noise=True,
        quantize_eval_images=True,
    )
    _write_json(
        tmp_path / "static" / "evaluation_step001000_paired5k.json",
        fid_inception_2048_small_sample=9.5,
        fid_inception_2048_float64=9.4,
        balanced_eval_labels=True,
        per_sample_eval_noise=True,
        quantize_eval_images=True,
    )
    metrics = tmp_path / "static" / "train_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "static_fd"])
        writer.writeheader()
        writer.writerow({"step": 1000, "static_fd": 3.0})

    rows = evaluation_rows(tmp_path)
    static = next(row for row in rows if row["variant"] == "static")
    assert static["step"] == 1000
    assert static["fid_5k"] == 9.4
    assert static["fid_numeric"] == "cpu_float64"
    assert abs(static["delta_vs_base"] + 0.7) < 1e-12
    assert static["static_fd_train"] == "3.0"
