import numpy as np
import pytest

from experiments.summarize_sit_ig_direction_specificity import (
    build_summary_manifest,
    summarize,
    validate_compatible_manifests,
)


def compatible_manifest(seed):
    return {
        "seed": seed,
        "checkpoint_sha256": "abc",
        "model_name": "SiT-XL/2",
        "encoder_depth": 8,
        "state_key": "ema",
        "latent_size": [4, 32, 32],
        "num_steps": 50,
        "solver_grid": [1.0, 0.0],
        "gamma": 0.01,
        "probe_count": 4,
        "conditions": [{"name": "baseline"}],
        "cfg_scale": 1.0,
        "sampler": "Euler",
    }


def test_manifest_validation_requires_distinct_compatible_seeds():
    validate_compatible_manifests([compatible_manifest(1), compatible_manifest(2)])
    with pytest.raises(RuntimeError, match="distinct"):
        validate_compatible_manifests([compatible_manifest(1), compatible_manifest(1)])
    changed = compatible_manifest(2)
    changed["probe_count"] = 8
    with pytest.raises(RuntimeError, match="probe_count"):
        validate_compatible_manifests([compatible_manifest(1), changed])


def test_summarize_constant_values():
    row = summarize(np.full(16, 2.5), repeats=100, seed=3)
    assert row["mean"] == pytest.approx(2.5)
    assert row["ci_low"] == pytest.approx(2.5)
    assert row["ci_high"] == pytest.approx(2.5)


def test_summary_manifest_marks_completed_artifact(tmp_path):
    manifests = [compatible_manifest(1), compatible_manifest(2)]
    for manifest in manifests:
        manifest["samples"] = 64
    result = build_summary_manifest(
        manifests,
        [tmp_path / "a", tmp_path / "b"],
        bootstrap_repeats=5000,
        seed=9,
    )
    assert result["status"] == "complete"
    assert result["total_samples"] == 128
