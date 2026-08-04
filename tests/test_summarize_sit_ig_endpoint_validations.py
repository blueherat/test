import numpy as np
import pytest

from experiments.summarize_sit_ig_endpoint_validations import (
    pooled_row,
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
        "windows": [[0, 1]],
        "schedules": [{"name": "baseline"}],
        "cfg_scale": 1.0,
        "sampler": "Euler",
    }


def test_manifest_validation_requires_distinct_compatible_seeds():
    validate_compatible_manifests([compatible_manifest(1), compatible_manifest(2)])
    with pytest.raises(RuntimeError, match="distinct"):
        validate_compatible_manifests([compatible_manifest(1), compatible_manifest(1)])
    changed = compatible_manifest(2)
    changed["num_steps"] = 100
    with pytest.raises(RuntimeError, match="num_steps"):
        validate_compatible_manifests([compatible_manifest(1), changed])


def test_pooled_row_bootstraps_each_metric():
    row = pooled_row(
        {"response": np.ones(16), "ratio": np.full(16, 2.0)},
        repeats=100,
        seed=3,
    )
    assert row["response_mean"] == pytest.approx(1.0)
    assert row["response_ci_low"] == pytest.approx(1.0)
    assert row["ratio_mean"] == pytest.approx(2.0)
