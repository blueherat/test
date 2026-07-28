import json
from pathlib import Path

import torch

from experiments.audit_rae_lpl_authenticity import (
    PAIRED_FINGERPRINT_KEYS,
    PAIRED_MANIFEST_KEYS,
    audit_pair,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_branch(path: Path, objective: str) -> None:
    path.mkdir()
    manifest = {key: f"value-{key}" for key in PAIRED_MANIFEST_KEYS}
    manifest.update(
        {
            "objective": objective,
            "dataset_split": "train",
            "evaluation_reference_loaded_by_trainer": False,
            "encoder_frozen": True,
            "decoder_frozen": True,
            "frozen_boundary_runtime_assertions": True,
            "optimizer_exactly_stage2_parameters": True,
            "dataset_files_asserted_train_only": True,
            "decoder_deterministic": True,
            "fresh_initialization": False,
            "resumed_from_branch_checkpoint": False,
            "resume_is_exact": True,
            "optimizer_state_at_branch_start": "fresh_shared",
            "method_identity": "RAE-adapted LPL",
            "paper_code_available": False,
            "pairing_scope": "fresh paired branches",
            "world_size": 1,
            "global_seed": 4101,
            "branch_start_step": 0,
            "endpoint_step": 1,
            "grad_accum_steps": 4,
            "lpl_weight": 0.0 if objective == "flow" else 0.25,
            "cross_normalization": (
                "none"
                if objective == "flow"
                else "differentiable prediction variance"
            ),
        }
    )
    fingerprint = {key: f"value-{key}" for key in PAIRED_FINGERPRINT_KEYS}
    _write_json(path / "manifest.json", manifest)
    _write_json(path / "pair_fingerprint.json", fingerprint)
    _write_json(
        path / "stream_audit_rank0.json",
        {
            "rank": 0,
            "objective": objective,
            "global_seed": 4101,
            "sha256": "same-stream",
            "microbatches": 4,
            "fields": [
                "dataset_index",
                "label",
                "augmented_image_stride32",
                "time",
                "noise_channels8_stride4",
            ],
            "gpu_memory": {
                "physical_free_at_end_mib": 12_000.0,
                "physical_total_mib": 24_000.0,
            },
        },
    )
    for filename in (
        "train_rae_strict_lpl.py",
        "rae_strict_lpl.py",
        "rae_lpl_detach_audit.py",
    ):
        (path / filename).write_text(f"{filename}\n", encoding="utf-8")
    lpl = 0.0 if objective == "flow" else 2.0
    weight = 0.0 if objective == "flow" else 0.25
    row = {
        "step": 1,
        "branch_update": 1,
        "total_loss": 1.0 + weight * lpl,
        "flow_loss": 1.0,
        "lpl_batch_contribution": lpl,
    }
    (path / "metrics.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    checkpoint = path / "checkpoints" / "step-0000001.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "step": 1,
            "branch_start_step": 0,
            "epoch": 0,
            "model": {},
            "ema": {},
            "optimizer": {},
            "scheduler": {},
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": [],
        },
        checkpoint,
    )


def test_authenticity_audit_accepts_a_strictly_paired_run(tmp_path: Path) -> None:
    flow = tmp_path / "flow"
    lpl = tmp_path / "lpl"
    _make_branch(flow, "flow")
    _make_branch(lpl, "full")

    result = audit_pair(flow, lpl)

    assert result["passed"]
    assert result["errors"] == []


def test_authenticity_audit_rejects_a_different_training_stream(
    tmp_path: Path,
) -> None:
    flow = tmp_path / "flow"
    lpl = tmp_path / "lpl"
    _make_branch(flow, "flow")
    _make_branch(lpl, "full")
    _write_json(
        lpl / "stream_audit_rank0.json",
        {
            "rank": 0,
            "objective": "full",
            "global_seed": 4101,
            "sha256": "different-stream",
            "microbatches": 4,
            "fields": [
                "dataset_index",
                "label",
                "augmented_image_stride32",
                "time",
                "noise_channels8_stride4",
            ],
            "gpu_memory": {
                "physical_free_at_end_mib": 12_000.0,
                "physical_total_mib": 24_000.0,
            },
        },
    )

    result = audit_pair(flow, lpl)

    assert not result["passed"]
    assert "rank 0 data-stream SHA256 differs" in result["errors"]


def test_authenticity_audit_rejects_diverged_endpoint_rng(tmp_path: Path) -> None:
    flow = tmp_path / "flow"
    lpl = tmp_path / "lpl"
    _make_branch(flow, "flow")
    _make_branch(lpl, "full")
    checkpoint = lpl / "checkpoints" / "step-0000001.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["rng_cpu"] = state["rng_cpu"].clone()
    state["rng_cpu"][0] ^= 1
    torch.save(state, checkpoint)

    result = audit_pair(flow, lpl)

    assert not result["passed"]
    assert any(
        error.startswith("endpoint_checkpoint.rng_cpu_sha256")
        for error in result["errors"]
    )


def test_memory_smoke_can_explicitly_skip_endpoint_checkpoint(
    tmp_path: Path,
) -> None:
    flow = tmp_path / "flow"
    lpl = tmp_path / "lpl"
    _make_branch(flow, "flow")
    _make_branch(lpl, "full")
    (flow / "checkpoints" / "step-0000001.pt").unlink()
    (lpl / "checkpoints" / "step-0000001.pt").unlink()

    strict = audit_pair(flow, lpl)
    smoke = audit_pair(flow, lpl, require_endpoint_checkpoint=False)

    assert not strict["passed"]
    assert smoke["passed"]
    assert smoke["endpoint_checkpoints"]["flow"] == {"required": False}
