import csv
import json
from pathlib import Path

import pytest

from experiments.advfd_cleanroom.summarize_official_fdr3 import summarize_csv


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "fd", "fdr", "n"))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_csv_uses_official_fdr3_normalizers(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    write_rows(
        path,
        [
            {"model": "convnext", "fd": 56.87, "fdr": 1.0, "n": 5000},
            {"model": "dinov2_cls", "fd": 28.38, "fdr": 2.0, "n": 5000},
            {"model": "clip_cls", "fd": 16.8, "fdr": 3.0, "n": 5000},
        ],
    )

    summary = summarize_csv("static", path)

    assert summary["num_images"] == 5000
    assert summary["convnext_fdr"] == pytest.approx(1.0)
    assert summary["dinov2_fdr"] == pytest.approx(2.0)
    assert summary["clip_fdr"] == pytest.approx(3.0)
    assert summary["fdr3"] == pytest.approx(2.0)


def test_summarize_csv_rejects_missing_encoder(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    write_rows(
        path,
        [
            {"model": "convnext", "fd": 56.87, "fdr": 1.0, "n": 5000},
            {"model": "dinov2_cls", "fd": 14.19, "fdr": 1.0, "n": 5000},
        ],
    )

    with pytest.raises(ValueError, match="clip_cls"):
        summarize_csv("advfd", path)
