from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import experiments.run_imagenet100_sit_fid_curve as fid_curve
from experiments.run_imagenet100_sit_fid_curve import (
    absolute_without_resolving_symlinks,
    fid_environment,
    parse_gpu_indices,
    parse_nvidia_memory_mib,
    run_logged,
    parse_steps,
    save_summary,
    valid_fid_artifact,
    valid_sampling_artifact,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_adm_python_path_preserves_virtual_environment_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_bytes(b"")
    environment_bin = tmp_path / "adm-fid" / "bin"
    environment_bin.mkdir(parents=True)
    entry = environment_bin / "python"
    entry.symlink_to(target)

    normalized = absolute_without_resolving_symlinks(entry)
    assert normalized == entry.absolute()
    assert normalized != entry.resolve()


def test_fid_environment_uses_one_gpu_without_mutating_sampling_env() -> None:
    sampling_env = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "EXISTING": "kept",
    }

    evaluation_env = fid_environment(
        sampling_env,
        cuda_visible_devices="2",
    )

    assert evaluation_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert evaluation_env["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert evaluation_env["EXISTING"] == "kept"
    assert sampling_env == {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "EXISTING": "kept",
    }


def test_sampling_and_fid_artifacts_require_exact_protocol(tmp_path: Path) -> None:
    sample = tmp_path / "samples_unguided_n5000.npz"
    reference = tmp_path / "reference.npz"
    sample.write_bytes(b"sample")
    reference.write_bytes(b"reference")
    checkpoint = {"step": 60_000, "checkpoint_sha256": "abc"}
    write_json(
        tmp_path / "sampling_resource_audit.json",
        {
            "monitored_gpu_indices": [0, 1, 2, 3],
            "memory_ceiling_mib": 9_216,
            "peak_memory_mib": {"0": 7_000, "1": 7_100, "2": 7_200, "3": 7_300},
            "return_code": 0,
            "violation": None,
        },
    )
    write_json(
        tmp_path / "sampling_manifest.json",
        {
            "checkpoint_sha256": "abc",
            "checkpoint_step": 60_000,
            "weights": "ema",
            "requested_samples": 5_000,
            "global_seed": 0,
            "world_size": 4,
            "per_rank_batch_size": 16,
            "vae_decode_batch_size": 4,
            "cuda_allocator_limit_gib": 7.5,
            "cfg_scale": 1.0,
            "guidance": False,
            "samples": str(sample),
        },
    )
    write_json(
        tmp_path / "fid_resource_audit.json",
        {
            "monitored_gpu_indices": [0],
            "memory_ceiling_mib": 9_216,
            "peak_memory_mib": {"0": 7_500},
            "return_code": 0,
            "violation": None,
        },
    )
    write_json(
        tmp_path / "fid5k_adm_results.json",
        {
            "reference": str(reference),
            "samples": str(sample),
            "batch_size": 8,
            "gpu_memory_fraction": 0.3,
            "fid": 12.0,
            "sfid": 10.0,
            "inception_score": 3.0,
        },
    )
    assert valid_sampling_artifact(
        tmp_path,
        checkpoint=checkpoint,
        num_samples=5_000,
        global_seed=0,
        world_size=4,
        per_rank_batch_size=16,
        vae_decode_batch_size=4,
        cuda_allocator_limit_gib=7.5,
        gpu_indices=[0, 1, 2, 3],
        memory_ceiling_mib=9_216,
    )
    assert valid_fid_artifact(
        tmp_path,
        reference=reference,
        num_samples=5_000,
        fid_batch_size=8,
        fid_gpu_memory_fraction=0.3,
        gpu_indices=[0],
        memory_ceiling_mib=9_216,
    )


def test_summary_detects_improving_tail(tmp_path: Path) -> None:
    rows = [
        {"step": 60_000, "fid": 100.0},
        {"step": 120_000, "fid": 80.0},
        {"step": 180_000, "fid": 70.0},
    ]
    summary = save_summary(rows, tmp_path)
    assert summary["improving_tail"] is True
    assert summary["latest_delta"] == -10.0
    assert summary["tail_linear_slope_fid_per_step"] < 0.0


def test_logit_normal_sampling_manifest_requires_training_distribution(tmp_path: Path) -> None:
    sample = tmp_path / "samples_unguided_n8.npz"
    sample.write_bytes(b"sample")
    write_json(
        tmp_path / "sampling_resource_audit.json",
        {
            "monitored_gpu_indices": [0],
            "memory_ceiling_mib": 9_216,
            "peak_memory_mib": {"0": 7_000},
            "return_code": 0,
            "violation": None,
        },
    )
    base_manifest = {
        "checkpoint_sha256": "logit-checkpoint",
        "checkpoint_step": 100_000,
        "weights": "ema",
        "requested_samples": 8,
        "global_seed": 0,
        "world_size": 1,
        "per_rank_batch_size": 8,
        "vae_decode_batch_size": 4,
        "cuda_allocator_limit_gib": 7.5,
        "cfg_scale": 1.0,
        "guidance": False,
        "prediction_target": "x",
        "loss_space": "velocity",
        "denominator_floor": 0.05,
        "samples": str(sample),
    }
    write_json(tmp_path / "sampling_manifest.json", base_manifest)
    checkpoint = {
        "protocol": "imagenet100_sit_single_target_linear_flow_v2",
        "step": 100_000,
        "checkpoint_sha256": "logit-checkpoint",
        "prediction_target": "x",
        "loss_space": "velocity",
        "denominator_floor": 0.05,
        "time_sampler": "logit_normal",
        "time_logit_mean": -0.8,
        "time_logit_std": 0.8,
    }
    with pytest.raises(ValueError, match="training_time_sampler"):
        valid_sampling_artifact(
            tmp_path,
            checkpoint=checkpoint,
            num_samples=8,
            global_seed=0,
            world_size=1,
            per_rank_batch_size=8,
            vae_decode_batch_size=4,
            cuda_allocator_limit_gib=7.5,
            gpu_indices=[0],
            memory_ceiling_mib=9_216,
        )
    base_manifest.update(
        {
            "training_time_sampler": "logit_normal",
            "training_time_logit_mean": -0.8,
            "training_time_logit_std": 0.8,
        }
    )
    write_json(tmp_path / "sampling_manifest.json", base_manifest)
    assert valid_sampling_artifact(
        tmp_path,
        checkpoint=checkpoint,
        num_samples=8,
        global_seed=0,
        world_size=1,
        per_rank_batch_size=8,
        vae_decode_batch_size=4,
        cuda_allocator_limit_gib=7.5,
        gpu_indices=[0],
        memory_ceiling_mib=9_216,
    )


def test_step_parser_sorts_and_rejects_duplicates() -> None:
    assert parse_steps("300000,60000,120000") == [60_000, 120_000, 300_000]
    try:
        parse_steps("60000,60000")
    except Exception as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate checkpoint steps must be rejected")


def test_gpu_and_nvidia_memory_parsers() -> None:
    assert parse_gpu_indices("3,1,0") == [3, 1, 0]
    assert parse_nvidia_memory_mib("0, 45\n1, 7123\n") == {0: 45, 1: 7123}


def test_run_logged_terminates_on_memory_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = iter(({0: 100}, {0: 9_500}))
    monkeypatch.setattr(fid_curve, "query_gpu_memory_mib", lambda: next(observations))
    audit_path = tmp_path / "resource.json"

    with pytest.raises(RuntimeError, match="memory safety guard"):
        run_logged(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            tmp_path / "process.log",
            env={},
            monitored_gpu_indices=[0],
            memory_ceiling_mib=9_216,
            memory_poll_interval=0.01,
            resource_audit_path=audit_path,
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["violation"]["reason"] == "gpu_memory_ceiling_reached"
    assert audit["peak_memory_mib"]["0"] == 9_500
