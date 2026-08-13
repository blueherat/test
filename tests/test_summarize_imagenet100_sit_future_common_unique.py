from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.summarize_imagenet100_sit_future_common_unique as summary_module


def _write_condition(path: Path, fid: float, fingerprint: str = "paired") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "fid5k_adm_results.json").write_text(
        json.dumps({"fid": fid, "sfid": fid + 1, "inception_score": 10 - fid}),
        encoding="utf-8",
    )
    (path / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "requested_samples": 5_000,
                "rank_noise_sha256": [fingerprint],
                "rank_label_sha256": [fingerprint],
                "formula": "test",
                "total_nfe": 1,
                "total_model_forwards": 3,
            }
        ),
        encoding="utf-8",
    )


def _populate(base: Path, root: Path) -> None:
    audit = base / "fid5k_step400k_floor_audit_seed0"
    directories = (
        base / "fid5k_static_pair_v_to_jit_x_step400000_seed0/static_s0",
        audit / "orthogonal_pair/orthogonal_pair_sm1",
        root / "x_common_on_v/x_common_on_v_s1",
        root / "x_unique_to_v/x_unique_to_v_s1",
        audit / "v270_direction_decomposition/orthogonal_pair/orthogonal_pair_sm1",
        root / "v_common_on_x/v_common_on_x_s1",
        root / "v_unique_to_x/v_unique_to_x_s1",
    )
    for index, path in enumerate(directories):
        _write_condition(path, fid=10.0 - index)


def test_summary_requires_and_preserves_exact_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base"
    root = base / "fid5k_step400k_floor_audit_seed0/common_unique_x400_v270"
    _populate(base, root)
    monkeypatch.setattr(summary_module, "BASE", base)

    table = summary_module.summarize(root)

    assert len(table) == 7
    assert float(table.loc[table.condition == "v_unique_to_x", "fid_gain_vs_v400"].iloc[0]) == 6.0

    mismatched = root / "v_unique_to_x/v_unique_to_x_s1/sampling_manifest.json"
    payload = json.loads(mismatched.read_text(encoding="utf-8"))
    payload["rank_noise_sha256"] = ["different"]
    mismatched.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identical initial noise"):
        summary_module.summarize(root)
