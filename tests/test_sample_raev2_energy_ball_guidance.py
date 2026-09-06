from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sample_raev2_energy_ball_guidance import (
    CALIBRATION_PROTOCOL, clean_second_moment_from_parent, cohort_euler_step_,
    coordinate_second_moment, initialize_cohort, parse_args, project_cohort_, target_second_moment,
    validate_pair,
)
from experiments.sample_raev2_proximal_calibration import official_euler_step


def test_initial_noise_preserves_official_draw_shapes_and_byte_hashes():
    state, labels, evidence = initialize_cohort(sample_count=17, latent_shape=(2, 3, 3),
                                               batch_size=8, seed=71, device=torch.device("cpu"))
    generator = torch.Generator(device="cpu").manual_seed(71)
    pieces = [torch.randn(min(8, 17-start), 2, 3, 3, generator=generator) for start in range(0, 17, 8)]
    original = torch.cat(pieces)
    assert torch.equal(state, original)
    assert torch.equal(labels, torch.arange(17))
    assert evidence["noise_sha256"] == hashlib.sha256(original.numpy().tobytes()).hexdigest()
    assert evidence["labels_sha256"] == hashlib.sha256(labels.numpy().tobytes()).hexdigest()
    assert [b["stop"]-b["start"] for b in evidence["batches"]] == [8, 8, 1]


def test_time_outer_loop_is_exactly_paired_with_batch_outer_official_loop():
    state, labels, _ = initialize_cohort(sample_count=16, latent_shape=(2, 2, 2),
                                        batch_size=8, seed=19, device=torch.device("cpu"))
    grid = torch.linspace(1, 0, 101).tolist()

    def model(x, t, context, attn_mask):
        # This model even depends on the other images in its microbatch:
        # accidentally forwarding the whole cohort would change the result.
        full = x * 0.4 + x.mean(0, keepdim=True) * 0.2
        full = full + t[:, None, None, None] * context[:, None, None, None] * 0.001
        return full, full * 0.8

    original = []
    for start in range(0, len(state), 8):
        batch = state[start:start+8].clone()
        for current, following in zip(grid[:-1], grid[1:]):
            batch = official_euler_step(model, batch, labels[start:start+8], current, following)
        original.append(batch)
    cohort = state.clone()
    for current, following in zip(grid[:-1], grid[1:]):
        cohort_euler_step_(model, cohort, labels, current, following, batch_size=8)
        project_cohort_(cohort, 0.0, mode="official", batch_size=8)
    assert torch.equal(cohort, torch.cat(original))


def test_projection_is_one_shared_scalar_for_actual_whole_cohort():
    state = torch.tensor([[1.0, 2.0], [8.0, 9.0]])
    before = state.clone()
    metrics = project_cohort_(state, 2.0, mode="energy_ball", batch_size=1)
    expected = math.sqrt(2.0 / float(before.double().square().mean()))
    assert metrics["lambda_ideal_fp64"] == expected
    assert torch.equal(state, before * metrics["lambda_applied_fp32"])
    assert metrics["active"]
    assert abs(coordinate_second_moment(state, 1) - 2.0) < 5e-7
    assert metrics["cohort_correction_rms"] > 0


def test_inactive_and_zero_cohorts_are_exact_identity_without_expansion():
    for state, bound in [(torch.zeros(4, 2), 0.0), (torch.ones(4, 2), 4.0)]:
        original = state.clone()
        metrics = project_cohort_(state, bound, mode="energy_ball", batch_size=3)
        assert torch.equal(state, original)
        assert metrics["lambda_applied_fp32"] == 1.0
        assert metrics["cohort_correction_rms"] == 0.0
    value = torch.ones(2, 3)
    project_cohort_(value, 0.0, mode="energy_ball", batch_size=1)
    assert not value.any()


def parent_fixture():
    center = torch.tensor([1.0, 2.0], dtype=torch.float64)
    variance = torch.tensor([3.0, 4.0], dtype=torch.float64)
    fit = {"center": center, "variance": variance, "count": 1000}
    request = {"protocol": CALIBRATION_PROTOCOL, "fit_input": "true_next_state",
               "fit_error": "frozen_euler_next_minus_true_next", "samples": 1000,
               "time_grid": torch.linspace(1, 0, 101).tolist()}
    return {"steps": {i: fit for i in range(100)}, "source_requests": [request]}


def test_clean_energy_uses_uncentered_final_target_moments_and_bridge_endpoints():
    archive = parent_fixture()
    m2 = clean_second_moment_from_parent(archive)
    assert m2 == 6.0
    assert target_second_moment(0.0, m2) == m2
    assert target_second_moment(1.0, m2) == 1.0
    assert target_second_moment(0.5, m2) == 1.75


@pytest.mark.parametrize("corrupt", ("grid", "fit_input", "steps", "precision", "count"))
def test_clean_energy_refuses_wrong_parent_target_or_moments(corrupt):
    archive = copy.deepcopy(parent_fixture())
    if corrupt == "grid":
        archive["source_requests"][0]["time_grid"][-1] = 0.1
    elif corrupt == "fit_input":
        archive["source_requests"][0]["fit_input"] = "actual_rollout"
    elif corrupt == "steps":
        del archive["steps"][2]
    elif corrupt == "precision":
        archive["steps"][99]["variance"] = archive["steps"][99]["variance"].float()
    else:
        archive["steps"][99]["count"] = 1024
    with pytest.raises(ValueError):
        clean_second_moment_from_parent(archive)


def test_cli_allows_16_image_loop_identity_smoke_but_no_guidance_tuning(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path), "--mode", "official", "--sample-count", "16"])
    assert args.sample_count == 16 and args.batch_size == 8
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--mode", "energy_ball", "--scale", "1.2"])


@pytest.mark.parametrize("corrupt", (None, "mode", "source", "decoder"))
def test_pair_reference_checks_official_mode_and_shared_implementation(tmp_path, corrupt):
    keys = ("seed", "sample_count", "batch_size", "precision", "config_sha256", "checkpoint_sha256",
            "normalization_stats_sha256", "time_grid", "tf32", "guidance_and_step_arithmetic",
            "pixel_arithmetic", "forward_layout")
    request = {key: key for key in keys}
    request.update(mode="official", source_sha256={"shared_helper": "sha"}, decoder_artifacts={"decoder": "sha"})
    baseline = copy.deepcopy(request)
    if corrupt == "mode":
        baseline["mode"] = "proximal"
    elif corrupt == "source":
        baseline["source_sha256"]["shared_helper"] = "changed"
    elif corrupt == "decoder":
        baseline["decoder_artifacts"]["decoder"] = "changed"
    evidence = {"noise_sha256": "noise", "labels_sha256": "labels"}
    (tmp_path / "request.json").write_text(json.dumps(baseline))
    (tmp_path / "summary.json").write_text(json.dumps({"complete": True, "mode": "official", **evidence}))
    if corrupt:
        with pytest.raises(ValueError):
            validate_pair(request, evidence, tmp_path)
    else:
        assert validate_pair(request, evidence, tmp_path)["noise_and_labels_match"]
