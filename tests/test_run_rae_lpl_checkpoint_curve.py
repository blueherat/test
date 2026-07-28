import json
from pathlib import Path

import pytest

from experiments.run_rae_lpl_checkpoint_curve import (
    evaluation_rows,
    parse_endpoints,
    plot_curve,
    summarize_curve,
)


def _row(branch: str, endpoint: int, fid: float, kid: float, score: float) -> dict:
    return {
        "branch": branch,
        "endpoint": endpoint,
        "sample_count": 1000,
        "sampling_seed": 7,
        "sampling_steps": 50,
        "sampling_processes": 4,
        "per_process_batch": 4,
        "label_sampler_version": "interleaved-v3-provenance",
        "reference_sha256": "reference",
        "sampling_provenance_protocol": "strict-sampling-provenance-v1",
        "endpoint_checkpoint_sha256": f"endpoint-{branch}",
        "sampling_checkpoint_sha256": f"sampling-{branch}",
        "sampling_provenance_sha256": f"provenance-{branch}",
        "sample_npz_sha256": f"samples-{branch}",
        "frechet_inception_distance": fid,
        "kernel_inception_distance_mean": kid,
        "inception_score_mean": score,
    }


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_endpoints_requires_unique_increasing_values() -> None:
    assert parse_endpoints("500,1000,1500") == (500, 1000, 1500)
    with pytest.raises(Exception):
        parse_endpoints("1000,500")
    with pytest.raises(Exception):
        parse_endpoints("500,500")


def test_summarize_curve_sorts_and_computes_paired_deltas(tmp_path: Path) -> None:
    late = _write(
        tmp_path / "late.json",
        [_row("flow", 1000, 20.0, 0.006, 70.0), _row("lpl", 1000, 18.0, 0.005, 72.0)],
    )
    early = _write(
        tmp_path / "early.json",
        [_row("flow", 500, 19.0, 0.005, 71.0), _row("lpl", 500, 18.5, 0.004, 73.0)],
    )
    official = _write(
        tmp_path / "official.json",
        [_row("official", 0, 19.5, 0.0045, 74.0)],
    )

    summary = summarize_curve([late, early], official_evaluation=official)

    assert [row["endpoint"] for row in summary["rows"]] == [500, 1000]
    assert summary["rows"][1]["lpl_minus_flow_fid"] == pytest.approx(-2.0)
    assert summary["rows"][1]["lpl_minus_official_fid"] == pytest.approx(-1.5)


def test_evaluation_rows_rejects_unpaired_sampling(tmp_path: Path) -> None:
    flow = _row("flow", 500, 20.0, 0.006, 70.0)
    lpl = _row("lpl", 500, 18.0, 0.005, 72.0)
    lpl["sampling_seed"] = 8
    path = _write(tmp_path / "unpaired.json", [flow, lpl])

    with pytest.raises(ValueError, match="unpaired fields"):
        evaluation_rows(path)


def test_plot_curve_writes_nonempty_png(tmp_path: Path) -> None:
    rows = [
        {
            "endpoint": endpoint,
            "flow_fid": 20.0 + index,
            "lpl_fid": 19.0 + index,
            "flow_kid": 0.006 + index * 0.0001,
            "lpl_kid": 0.005 + index * 0.0001,
            "flow_is": 70.0 - index,
            "lpl_is": 72.0 - index,
        }
        for index, endpoint in enumerate((500, 1000, 1500))
    ]
    output = tmp_path / "curve.png"

    plot_curve(rows, output)

    assert output.stat().st_size > 10_000
