from __future__ import annotations

import json

import numpy as np

from experiments.run_raev2_decoded_distribution_audit import feature_statistics
from experiments.run_raev2_reconstruction_fid_audit import (
    RECONSTRUCTION_KEY,
    compute_audit_metrics,
    load_decoded_protocol,
    paired_feature_metrics,
)


def test_identical_source_and_reconstruction_have_zero_matched_fid() -> None:
    rng = np.random.default_rng(17)
    source = rng.normal(size=(32, 8)).astype(np.float32)
    reference = feature_statistics(rng.normal(size=(40, 8)).astype(np.float32))
    metrics = compute_audit_metrics(source, source.copy(), reference)
    assert metrics["fid_reconstruction_to_matched_source"] < 1e-8
    assert np.isclose(
        metrics["fid_source_to_official"],
        metrics["fid_reconstruction_to_official"],
    )
    assert np.isclose(metrics["paired_feature_cosine_mean"], 1.0)


def test_paired_feature_metrics_detect_a_shift() -> None:
    source = np.eye(4, dtype=np.float32)
    shifted = source + 0.25
    metrics = paired_feature_metrics(source, shifted)
    assert metrics["paired_feature_rmse"] > 0
    assert 0 < metrics["paired_feature_cosine_mean"] < 1


def test_load_decoded_protocol_orders_interleaved_shards(tmp_path) -> None:
    manifest = {
        "protocol": "raev2_decoded_distribution_audit_v1",
        "inception_feature": "2048",
        "samples": 4,
        "world_size": 2,
        "seed": 3,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for rank, ids in enumerate((np.array([0, 2]), np.array([1, 3]))):
        np.savez(
            tmp_path / f"decoded_features_rank{rank:02d}.npz",
            ids=ids,
            labels=np.array([10, 12]) if rank == 0 else np.array([11, 13]),
            **{RECONSTRUCTION_KEY: np.zeros((2, 2048), dtype=np.float32)},
        )
    loaded_manifest, ids, labels = load_decoded_protocol(tmp_path)
    assert loaded_manifest["seed"] == 3
    assert np.array_equal(ids, np.arange(4))
    assert np.array_equal(labels, np.array([10, 11, 12, 13]))
