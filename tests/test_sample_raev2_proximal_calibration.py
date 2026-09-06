from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_proximal_calibration import DiagonalFit
from experiments.sample_raev2_proximal_calibration import (
    CALIBRATION_PROTOCOL, GLOBAL_CALIBRATION_PROTOCOL, GLOBAL_SOURCE_FILES,
    SHARED_CALIBRATION_PROTOCOL, SHARED_SOURCE_FILES,
    finish_step, official_euler_step, parse_args, validate_calibration_archive,
)


def calibration_fixture():
    grid = torch.linspace(1, 0, 101).tolist()
    shape = (2, 1, 1)
    zeros = torch.zeros(shape)
    fit = DiagonalFit(zeros, zeros, zeros, zeros + 1, zeros, 1000).state_dict()
    request = {"protocol": CALIBRATION_PROTOCOL, "num_steps": 100, "config_sha256": "config",
               "checkpoint_sha256": "checkpoint", "normalization_stats_sha256": "stats", "time_grid": grid,
               "precision": "bf16", "batch_size": 8, "ig_scale": 1.78, "ig_interval": [0.1, 1.0],
               "fit_input": "true_next_state", "fit_error": "frozen_euler_next_minus_true_next",
               "forward_layout": "single_conditional_batch", "guidance_and_step_arithmetic": "fp32",
               "applied_coefficient_arithmetic": "fp32", "correction_multiplier": 1.0,
               "correction_time_selection": "none", "source_sha256": {"helper": "helper-sha"}}
    archive = {"source_requests": [request], "steps": {i: fit for i in range(100)},
               "applied_fp32_steps": {i: fit for i in range(100)}}
    arguments = dict(config_sha256="config", checkpoint_sha256="checkpoint", normalization_stats_sha256="stats",
                     time_grid=grid, latent_shape=shape, precision="bf16", batch_size=8,
                     expected_source_sha256={"helper": "helper-sha"})
    return archive, arguments


def test_disabled_proximal_is_exact_identity_and_zero_fit_matches_official_step():
    state = torch.tensor([[[[1.0]], [[2.0]]]])
    labels = torch.tensor([4])
    calls = []

    def model(x, t, context, attn_mask):
        calls.append((x, context))
        return x * 0.5, x * 0.25

    predicted = official_euler_step(model, state, labels, 0.7, 0.6)
    assert len(calls) == 1 and calls[0][0].shape == state.shape
    assert torch.equal(calls[0][1], labels)
    assert finish_step(predicted, mode="official", fit=None) is predicted
    archive, arguments = calibration_fixture()
    fit = validate_calibration_archive(archive, **arguments)[0]
    assert torch.equal(finish_step(predicted, mode="proximal", fit=fit), predicted)
    with pytest.raises(ValueError, match="every step"):
        finish_step(predicted, mode="proximal", fit=None)


@pytest.mark.parametrize("change", ("missing", "extra"))
def test_calibration_requires_all_100_steps_without_extra_or_missing_entries(change):
    archive, arguments = calibration_fixture()
    if change == "missing":
        del archive["applied_fp32_steps"][99]
    else:
        archive["applied_fp32_steps"][100] = archive["applied_fp32_steps"][0]
    with pytest.raises(ValueError, match="100 official steps"):
        validate_calibration_archive(archive, **arguments)


@pytest.mark.parametrize("key", ("config_sha256", "checkpoint_sha256", "normalization_stats_sha256", "precision", "batch_size"))
def test_calibration_refuses_protocol_or_hash_mismatch(key):
    archive, arguments = calibration_fixture()
    archive["source_requests"][0][key] = "different"
    with pytest.raises(ValueError, match=key):
        validate_calibration_archive(archive, **arguments)


def test_calibration_refuses_changed_helper_and_nonconvex_slope():
    archive, arguments = calibration_fixture()
    archive["source_requests"][0]["source_sha256"]["helper"] = "changed"
    with pytest.raises(ValueError, match="source hash"):
        validate_calibration_archive(archive, **arguments)
    archive, arguments = calibration_fixture()
    archive["applied_fp32_steps"][0] = copy.deepcopy(archive["applied_fp32_steps"][0])
    archive["applied_fp32_steps"][0]["slope"].fill_(-0.1)
    with pytest.raises(ValueError, match="convex"):
        validate_calibration_archive(archive, **arguments)


def test_only_proximal_cli_requires_calibration(tmp_path):
    assert parse_args(["--output-dir", str(tmp_path), "--mode", "official"]).calibration is None
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--mode", "proximal"])


def shared_calibration_fixture():
    archive, arguments = calibration_fixture()
    request = archive["source_requests"][0]
    request.update(protocol=SHARED_CALIBRATION_PROTOCOL, fit_structure="channel_shared_affine",
                   parent_calibration_path="/audit/parent/calibration.pt", parent_calibration_sha256="parent-sha")
    shared_hashes = {path: f"shared-sha-{index}" for index, path in enumerate(SHARED_SOURCE_FILES)}
    request["source_sha256"].update(shared_hashes)
    arguments["expected_source_sha256"].update(shared_hashes)
    return archive, arguments


def test_shared_protocol_keeps_full_grid_and_explicit_source_chain():
    archive, arguments = shared_calibration_fixture()
    assert set(validate_calibration_archive(archive, **arguments)) == set(range(100))
    del archive["source_requests"][0]["source_sha256"][SHARED_SOURCE_FILES[0]]
    with pytest.raises(ValueError, match="source hash"):
        validate_calibration_archive(archive, **arguments)


@pytest.mark.parametrize("field", ("parent_calibration_sha256", "parent_calibration_path", "fit_structure"))
def test_shared_protocol_rejects_missing_parent_or_undefined_structure(field):
    archive, arguments = shared_calibration_fixture()
    del archive["source_requests"][0][field]
    with pytest.raises(ValueError, match="derived calibration"):
        validate_calibration_archive(archive, **arguments)


def test_shared_protocol_requires_local_hash_validation_for_both_added_sources():
    archive, arguments = shared_calibration_fixture()
    del arguments["expected_source_sha256"][SHARED_SOURCE_FILES[1]]
    with pytest.raises(ValueError, match="source hash is required"):
        validate_calibration_archive(archive, **arguments)


def global_calibration_fixture():
    archive, arguments = calibration_fixture()
    request = archive["source_requests"][0]
    request.update(protocol=GLOBAL_CALIBRATION_PROTOCOL, fit_structure="global_zero_anchor",
                   parent_calibration_path="/audit/parent/calibration.pt", parent_calibration_sha256="parent-sha")
    hashes = {path: f"global-sha-{index}" for index, path in enumerate(GLOBAL_SOURCE_FILES)}
    request["source_sha256"].update(hashes)
    arguments["expected_source_sha256"].update(hashes)
    return archive, arguments


def test_global_zero_anchor_protocol_accepts_full_grid_with_expanded_scalar_coefficients():
    archive, arguments = global_calibration_fixture()
    scalar = torch.tensor(0.25)
    for fit in archive["applied_fp32_steps"].values():
        fit["slope"] = scalar.expand(arguments["latent_shape"])
    fits = validate_calibration_archive(archive, **arguments)
    assert len(fits) == 100
    value = torch.tensor([[[[1.0]], [[2.0]]]])
    assert torch.equal(finish_step(value, mode="proximal", fit=fits[0]), value / 1.25)


@pytest.mark.parametrize("field", ("center", "offset", "slope"))
def test_global_protocol_refuses_nonzero_anchor_or_spatial_slope(field):
    archive, arguments = global_calibration_fixture()
    archive["applied_fp32_steps"][0] = copy.deepcopy(archive["applied_fp32_steps"][0])
    archive["applied_fp32_steps"][0][field].flatten()[0] = 0.2
    with pytest.raises(ValueError, match="global_zero_anchor"):
        validate_calibration_archive(archive, **arguments)


def test_global_protocol_refuses_missing_source_and_structure_search():
    archive, arguments = global_calibration_fixture()
    del archive["source_requests"][0]["source_sha256"][GLOBAL_SOURCE_FILES[0]]
    with pytest.raises(ValueError, match="source hash"):
        validate_calibration_archive(archive, **arguments)
    archive, arguments = global_calibration_fixture()
    archive["source_requests"][0]["fit_structure"] = "global_with_fitted_offset"
    with pytest.raises(ValueError, match="fixed global_zero_anchor"):
        validate_calibration_archive(archive, **arguments)
