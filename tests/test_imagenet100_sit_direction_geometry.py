from __future__ import annotations

import json

import torch

from experiments.analyze_imagenet100_sit_400k_direction_geometry import (
    direction_metrics,
)
from experiments.summarize_imagenet100_sit_400k_direction_comparison import (
    load_condition,
)


def test_direction_metrics_identifies_matching_orthogonal_directions() -> None:
    anchor = torch.tensor([[[[2.0, 0.0]]], [[[0.0, 3.0]]]])
    orthogonal = torch.tensor([[[[0.0, 1.0]]], [[[1.0, 0.0]]]])
    x_other = anchor - (0.5 * anchor + orthogonal)
    v_other = anchor - (-0.25 * anchor + 2.0 * orthogonal)

    metrics = direction_metrics(anchor, x_other, v_other)

    torch.testing.assert_close(
        metrics["orthogonal_cosine"], torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["v270_over_x_orthogonal_rms"],
        torch.full((2,), 2.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["x_orthogonal_energy_fraction"],
        torch.tensor([0.5, 4.0 / 13.0], dtype=torch.float64),
    )


def test_direction_metrics_preserves_opposite_orientation() -> None:
    anchor = torch.tensor([[[[1.0, 0.0]]]])
    x_other = anchor - torch.tensor([[[[0.0, 1.0]]]])
    v_other = anchor - torch.tensor([[[[0.0, -3.0]]]])

    metrics = direction_metrics(anchor, x_other, v_other)

    torch.testing.assert_close(
        metrics["orthogonal_cosine"], torch.tensor([-1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["orthogonal_overlap_cos2"], torch.tensor([1.0], dtype=torch.float64)
    )


def test_direction_summary_keeps_unrecorded_nfe_unknown(tmp_path) -> None:
    (tmp_path / "fid5k_adm_results.json").write_text(
        json.dumps({"fid": 60.0, "sfid": 68.0, "inception_score": 27.0}),
        encoding="utf-8",
    )
    manifest = {
        "requested_samples": 5_000,
        "rank_noise_sha256": ["noise"],
        "rank_label_sha256": ["labels"],
    }
    manifest_path = tmp_path / "sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    row = load_condition("baseline", "baseline", tmp_path)
    assert row["total_nfe"] is None

    manifest["rank_sampling_stats"] = [{"nfe": 11}, {"nfe": 13}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    row = load_condition("baseline", "baseline", tmp_path)
    assert row["total_nfe"] == 24
