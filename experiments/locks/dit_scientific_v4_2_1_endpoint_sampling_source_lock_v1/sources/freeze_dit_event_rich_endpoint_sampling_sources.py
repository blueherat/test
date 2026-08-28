#!/usr/bin/env python3
"""Freeze scientific-v4.2.1 endpoint sampler sources, protocol, and assets.

The frozen sampler keeps the already reviewed pair-keyed singleton RNG and
84x12 discovery axis byte-for-byte.  This artifact is design-only and does
not authorize real sampling.  No endpoint label, review, metric, feature,
embedding, candidate value, or score is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import run_dit_event_rich_endpoint_screen as runner

sys.dont_write_bytecode = True


ROOT = runner.ROOT
DEFAULT_EVENT_PROTOCOL_LOCK = (
    ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_2_1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/locks/dit_scientific_v4_2_1_endpoint_sampling_source_lock_v1"
)
METHOD_LOCK = ROOT / "experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_2"
METHOD_IDENTITY = "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"
OLD_UNUSED_OUTPUT = (
    runner.DATA_ROOT / "cross_scale_evidence/dit_event_rich_endpoint_screen_v1"
)
OLD_SCIENTIFIC_V4_OUTPUT = (
    runner.DATA_ROOT / "cross_scale_evidence/dit_scientific_v4_endpoint_screen_v1"
)
SOURCE_PATHS = {
    "sample_dit_imagenet256_endpoint_pairs.py": (
        ROOT / "experiments/sample_dit_imagenet256_endpoint_pairs.py"
    ),
    "run_dit_event_rich_endpoint_screen.py": (
        ROOT / "experiments/run_dit_event_rich_endpoint_screen.py"
    ),
    "freeze_dit_event_rich_endpoint_sampling_sources.py": Path(__file__).resolve(),
    "reproduce_dit_imagenet256.py": (
        ROOT / "experiments/reproduce_dit_imagenet256.py"
    ),
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_module(path: Path) -> ModuleType:
    path = runner.require_regular(path, "strict reproduction source")
    spec = importlib.util.spec_from_file_location("_event_endpoint_freezer_strict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import strict helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_lock_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = runner.require_directory(root, "scientific-v4 confirmation protocol lock")
    protocol_path = runner.require_regular(root / "protocol.json", "event protocol")
    manifest_path = runner.require_regular(root / "manifest.json", "event manifest")
    completion_path = runner.require_regular(root / "completion.json", "event completion")
    protocol = runner.load_json(protocol_path)
    manifest = runner.load_json(manifest_path)
    completion = runner.load_json(completion_path)
    classes = runner.validate_event_protocol_snapshot(protocol)
    identity = protocol["identity_sha256"]
    manifest_identity = manifest.get("identity_sha256")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"event protocol lock contains a symlink: {path}")
        files.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": runner.sha256_file(path),
            }
        )
    if (
        not isinstance(manifest_identity, str)
        or runner.canonical_sha256(runner.without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != identity
        or manifest.get("files") != files
        or completion
        != {
            "complete": True,
            "protocol_identity_sha256": identity,
            "protocol_file_sha256": runner.sha256_file(protocol_path),
            "manifest_identity_sha256": manifest_identity,
            "manifest_file_sha256": runner.sha256_file(manifest_path),
            "ready_for_real_sampling": False,
        }
    ):
        raise RuntimeError("scientific-v4 confirmation protocol lock failed validation")
    return protocol, {
        "path": str(root),
        "identity_sha256": identity,
        "file_sha256": runner.sha256_file(protocol_path),
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": runner.sha256_file(manifest_path),
        "completion_file_sha256": runner.sha256_file(completion_path),
        "classes_ordered": list(classes),
    }


def method_lock_binding(
    event_protocol: Mapping[str, Any], root: Path = METHOD_LOCK
) -> dict[str, Any]:
    root = runner.require_directory(root, "method-v2.2 lock")
    manifest_path = runner.require_regular(root / "manifest.json", "method manifest")
    completion_path = runner.require_regular(root / "completion.json", "method completion")
    protocol_path = runner.require_regular(root / "protocol.json", "method protocol")
    power_path = runner.require_regular(
        root / "matched_q_conditional_power_gate.json", "method power gate"
    )
    adaptive_path = runner.require_regular(
        root / "adaptive_predictable_null_audit.json", "method null audit"
    )
    manifest = runner.load_json(manifest_path)
    completion = runner.load_json(completion_path)
    identity = manifest.get("identity_sha256")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"method-v2.2 lock contains a symlink: {path}")
        if not path.is_file():
            continue
        if path.parent == root and path.name in {"manifest.json", "completion.json"}:
            continue
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": runner.sha256_file(path),
            }
        )
    expected_completion = {
        "schema_version": 2,
        "identity_sha256": identity,
        "manifest_sha256": runner.sha256_file(manifest_path),
        "files_sha256": manifest.get("files_sha256"),
        "file_count": len(records),
        "execution_ready": False,
    }
    event_method = event_protocol.get("method_lock", {})
    if (
        identity != METHOD_IDENTITY
        or runner.canonical_sha256(runner.without_identity(manifest)) != identity
        or manifest.get("status") != "METHOD_PROTOCOL_FROZEN_EXECUTION_BLOCKED"
        or manifest.get("execution_ready") is not False
        or manifest.get("files") != records
        or manifest.get("files_sha256") != runner.canonical_sha256(records)
        or completion != expected_completion
        or event_method.get("identity_sha256") != METHOD_IDENTITY
        or event_method.get("exact_identity_required") != METHOD_IDENTITY
        or event_method.get("protocol_file_sha256") != runner.sha256_file(protocol_path)
        or event_method.get("manifest_file_sha256") != runner.sha256_file(manifest_path)
        or event_method.get("completion_file_sha256") != runner.sha256_file(completion_path)
        or event_method.get("matched_q_power_gate_file_sha256")
        != runner.sha256_file(power_path)
        or event_method.get("adaptive_null_audit_file_sha256")
        != runner.sha256_file(adaptive_path)
    ):
        raise RuntimeError("scientific-v4.2.1 method-v2.2 binding failed validation")
    return {
        "path": str(root),
        "identity_sha256": identity,
        "protocol_file_sha256": runner.sha256_file(protocol_path),
        "manifest_file_sha256": runner.sha256_file(manifest_path),
        "completion_file_sha256": runner.sha256_file(completion_path),
        "matched_q_power_gate_file_sha256": runner.sha256_file(power_path),
        "adaptive_null_audit_file_sha256": runner.sha256_file(adaptive_path),
        "execution_ready": False,
    }


def source_records() -> dict[str, dict[str, Any]]:
    if set(SOURCE_PATHS) != set(runner.SOURCE_BASENAMES):
        raise RuntimeError("freezer and launcher source sets differ")
    records: dict[str, dict[str, Any]] = {}
    for basename, raw_path in SOURCE_PATHS.items():
        path = runner.require_regular(raw_path, f"sampling source {basename}")
        records[basename] = {
            "live_path_at_freeze": str(path),
            "sha256": runner.sha256_file(path),
        }
    return records


def validate_assets(
    dit_root: Path, checkpoint: Path, vae_snapshot: Path
) -> dict[str, Any]:
    strict = load_module(SOURCE_PATHS["reproduce_dit_imagenet256.py"])
    dit_root = runner.require_directory(dit_root, "DiT repository")
    checkpoint = runner.require_regular(checkpoint, "DiT checkpoint")
    vae_snapshot = runner.require_directory(vae_snapshot, "VAE snapshot")
    if (
        strict.MODEL_NAME != "DiT-XL/2"
        or strict.NUM_SAMPLING_STEPS != 250
        or strict.CFG_SCALE != 4.0
        or strict.VAE_KIND != "mse"
        or strict.VAE_SCALING_FACTOR != 0.18215
    ):
        raise RuntimeError("strict DiT scientific constants changed")
    return {
        "dit_repository": strict.validate_repository(dit_root, checkpoint),
        "checkpoint": strict.validate_checkpoint(checkpoint),
        "vae_snapshot": strict.validate_vae_snapshot(vae_snapshot),
    }


def build_protocol(
    event_binding: Mapping[str, Any],
    method_binding: Mapping[str, Any],
    sources: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    classes = tuple(int(value) for value in event_binding["classes_ordered"])
    axis = runner.pair_axis(classes)
    shards = runner.logical_assignments(classes)
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "SCIENTIFIC_V4_2_1_ENDPOINT_SOURCE_FROZEN_EXECUTION_NOT_READY",
        "execution_ready": False,
        "real_endpoint_outputs_present_at_freeze": False,
        "real_expert_label_review_or_consensus_results_present": False,
        "event_protocol": {
            key: value for key, value in event_binding.items() if key != "classes_ordered"
        },
        "method_lock": dict(method_binding),
        "scientific_contract": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "sampling_steps": 250,
            "clip_denoised": False,
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "vae": "mse",
            "vae_scaling_factor": 0.18215,
            "classes_ordered": list(classes),
            "global_seeds": list(runner.EXPECTED_SEEDS),
            "pair_axis_order": "seed-major, frozen-class-roster-minor",
            "pair_count": len(axis),
            "endpoint_only": True,
            "trace_saved": False,
            "quality_score": None,
            "selection": None,
            "intervention": None,
        },
        "rng_contract": {
            "unit": "(global_seed,class_id)",
            "domain": runner.RNG_DOMAIN,
            "derivation": (
                "uint64_be(first_8_bytes(SHA256(ASCII(domain + NUL + global_seed "
                "+ NUL + class_id)))) mod 2^63"
            ),
            "manual_seed_timing": (
                "after frozen model/VAE load, immediately before singleton initial latent"
            ),
            "classes_per_sampler_invocation": 1,
            "same_global_seed_classes_share_initial_noise": False,
            "same_global_seed_classes_share_transition_innovations": False,
            "task_order_worker_shard_resume_invariant": True,
            "initial_noise_shape": [1, 4, 32, 32],
            "duplicated_cfg_state_shape": [2, 4, 32, 32],
            "transition_randn_like_calls": 250,
            "transition_noise_shape_each_call": [2, 4, 32, 32],
            "full_2B_randn_like_each_transition_including_t0": True,
            "terminal_t0_randn_consumed_then_masked": True,
            "second_half_transition_noises_consumed_then_state_discarded": True,
            "relation_to_third_pool": (
                "same singleton-or-batch first-half latent duplication, upstream CFG and full-2B "
                "ancestral transition semantics; pair-keyed reseeding replaces the third pool's "
                "ordered three-class batch RNG so endpoint pixels are shard/order invariant"
            ),
        },
        "execution_contract": {
            "logical_worker_count": runner.WORKER_COUNT,
            "pair_count_per_worker": runner.PAIRS_PER_WORKER,
            "assignment_kind": "four immutable contiguous 252-pair logical shards",
            "logical_assignment_is_not_scientific_rng_input": True,
            "physical_gpu_schedule_is_not_scientific_rng_input": True,
            "allowed_logical_worker_subset": [0, 1, 2, 3],
            "subset_rule": "any fixed nonempty duplicate-free subset of logical workers 0..3",
            "physical_gpu_rule": (
                "one to four currently free unique devices; multiple selected logical "
                "workers assigned to one device execute sequentially"
            ),
            "one_model_and_vae_load_per_logical_worker_invocation": True,
            "pool_validation_and_receipts_only_after_all_four_logical_shard_receipts": True,
            "ordered_logical_shards": [
                {
                    "logical_worker_index": index,
                    "first_pair": {"global_seed": shard[0][0], "class_id": shard[0][1]},
                    "last_pair": {"global_seed": shard[-1][0], "class_id": shard[-1][1]},
                    "pair_count": len(shard),
                }
                for index, shard in shards.items()
            ],
        },
        "output_contract": {
            "pair_directory_template": "pairs/seed{global_seed:04d}_class{class_id:04d}",
            "endpoint_filename": "endpoint.png",
            "endpoint_mode": "RGB",
            "endpoint_size": [256, 256],
            "files_per_pair": ["endpoint.png", "manifest.json", "completion.json"],
            "trajectory_or_latent_payload": None,
            "pair_and_pool_receipts_required": True,
            "logical_shard_receipt_required_before_pool_receipt": True,
            "all_endpoint_file_and_pixel_hashes_bound": True,
        },
        "resume_and_failure_contract": {
            "completed_pair_reuse": "full identity and endpoint file/pixel hash validation",
            "partial_or_changed_pair": "fail closed; preserve and refuse overwrite",
            "pool_receipts": "published only after all 1008 pair outputs fully revalidate",
            "automatic_deletion_or_quarantine": False,
        },
        "evidence_access_audit": {
            "labels_or_reviews_opened": False,
            "metrics_features_embeddings_or_scores_opened": False,
            "score_label_join_performed": False,
        },
        "external_evaluation_boundary": {
            "visual_labels_used_for_evaluation_cohort_enrichment": True,
            "class_selection_is_evaluation_event_enrichment_not_method": True,
            "visual_labels_scores_or_selected_class_rank_used_as_B_E_method_input": False,
            "endpoint_or_review_artifacts_are_forbidden_B_E_method_inputs": True,
            "B_E_replay_accepts_only_completed_preterminal_trace_schema": True,
            "B_E_calibration_accepts_only_frozen_label_free_internal_B_products": True,
            "Inception_DINO_FID_CLIP_or_other_external_representation_used": False,
        },
        "assets": dict(assets),
        "source_snapshots": dict(sources),
    }
    protocol["identity_sha256"] = runner.canonical_sha256(protocol)
    return protocol


def inspect_real_output_path(path: Path) -> dict[str, Any]:
    """Count prior real endpoint material without opening any endpoint pixels."""

    absolute = path.expanduser().absolute()
    if not os.path.lexists(absolute):
        return {
            "path": str(absolute),
            "state": "absent",
            "completed_pair_receipts": 0,
            "partial_pair_paths": 0,
            "endpoint_files": 0,
        }
    if not absolute.is_dir() or absolute.is_symlink():
        raise RuntimeError(f"real endpoint output path is indirect or not a directory: {absolute}")
    pairs = absolute / "pairs"
    if not os.path.lexists(pairs):
        return {
            "path": str(absolute),
            "state": "present_without_pairs",
            "completed_pair_receipts": 0,
            "partial_pair_paths": 0,
            "endpoint_files": 0,
        }
    if not pairs.is_dir() or pairs.is_symlink():
        raise RuntimeError(f"real endpoint pairs path is indirect or not a directory: {pairs}")
    completed = 0
    partial = 0
    endpoints = 0
    for pair in pairs.iterdir():
        if not pair.is_dir() or pair.is_symlink():
            partial += 1
            continue
        endpoint = pair / "endpoint.png"
        manifest = pair / "manifest.json"
        completion = pair / "completion.json"
        endpoints += int(endpoint.is_file() and not endpoint.is_symlink())
        if manifest.is_file() and completion.is_file() and not manifest.is_symlink() and not completion.is_symlink():
            completed += 1
        else:
            partial += 1
    return {
        "path": str(absolute),
        "state": "present",
        "completed_pair_receipts": completed,
        "partial_pair_paths": partial,
        "endpoint_files": endpoints,
    }


def pre_sampling_zero_audit() -> dict[str, Any]:
    rows = [
        inspect_real_output_path(OLD_UNUSED_OUTPUT),
        inspect_real_output_path(OLD_SCIENTIFIC_V4_OUTPUT),
        inspect_real_output_path(runner.DEFAULT_OUTPUT_ROOT),
    ]
    completed = sum(row["completed_pair_receipts"] for row in rows)
    partial = sum(row["partial_pair_paths"] for row in rows)
    endpoints = sum(row["endpoint_files"] for row in rows)
    if completed or partial or endpoints:
        raise RuntimeError(
            "real endpoint material already exists; refuse to claim a pre-sampling v4 freeze"
        )
    return {
        "audit_status": "ZERO_REAL_OUTPUTS_BEFORE_V4_2_1_SOURCE_FREEZE",
        "audited_paths": rows,
        "completed_real_endpoint_pairs": completed,
        "partial_real_endpoint_pairs": partial,
        "real_endpoint_files": endpoints,
        "real_sampling_started": False,
    }


def run_source_selftests() -> dict[str, Any]:
    runner.run_self_test()
    runner.run_smoke_test()
    return {
        "status": "PASS_SYNTHETIC_NO_REAL_DATA",
        "pair_rng_known_answers_checked": True,
        "logical_subset_and_physical_schedule_invariance_checked": True,
        "all_four_logical_shards_required_for_pool_receipts": True,
        "exact_tree_and_provenance_checked": True,
        "gpu_or_model_used": False,
    }


def freeze(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite source lock: {output}")
    event_protocol, event_binding = event_lock_binding(args.event_protocol_lock)
    method_binding = method_lock_binding(event_protocol)
    sources = source_records()
    checkpoint = args.checkpoint or args.dit_root / "pretrained_models" / runner.CHECKPOINT_FILENAME
    assets = validate_assets(args.dit_root, checkpoint, args.vae_snapshot)
    protocol = build_protocol(event_binding, method_binding, sources, assets)
    zero_audit = pre_sampling_zero_audit()
    selftest_receipt = run_source_selftests()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=str(output.parent))
    )
    try:
        (temporary / "sources").mkdir(exist_ok=False)
        for basename, source_path in SOURCE_PATHS.items():
            shutil.copyfile(source_path, temporary / "sources" / basename)
        shutil.copyfile(args.event_protocol_lock / "protocol.json", temporary / "event_protocol.json")
        write_json(temporary / "pre_sampling_zero_audit.json", zero_audit)
        write_json(temporary / "selftest_receipt.json", selftest_receipt)
        write_json(temporary / "sampling_protocol.json", protocol)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "sampling_protocol_identity_sha256": protocol["identity_sha256"],
            "event_protocol_identity_sha256": event_protocol["identity_sha256"],
            "files": runner.artifact_records(temporary),
        }
        manifest["identity_sha256"] = runner.canonical_sha256(manifest)
        write_json(temporary / "manifest.json", manifest)
        completion = {
            "complete": True,
            "sampling_protocol_identity_sha256": protocol["identity_sha256"],
            "sampling_protocol_file_sha256": runner.sha256_file(
                temporary / "sampling_protocol.json"
            ),
            "event_protocol_identity_sha256": event_protocol["identity_sha256"],
            "event_protocol_file_sha256": runner.sha256_file(
                temporary / "event_protocol.json"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": runner.sha256_file(temporary / "manifest.json"),
        }
        write_json(temporary / "completion.json", completion)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validated_protocol, _, _ = runner.validate_source_lock(output)
    if validated_protocol != protocol:
        raise RuntimeError("new source lock failed validation round trip")
    locked_launcher = output / "sources/run_dit_event_rich_endpoint_screen.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(locked_launcher),
            "--source-lock",
            str(output),
            "--validate-source-lock",
        ],
        cwd=Path(tempfile.gettempdir()),
        env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen endpoint launcher failed isolated provenance validation:\n"
            + completed.stdout
            + completed.stderr
        )
    with tempfile.TemporaryDirectory(prefix="dit-endpoint-lock-tamper-") as raw:
        poisoned = Path(raw) / "lock"
        shutil.copytree(output, poisoned)
        write_json(poisoned / "unexpected.json", {"poison": True})
        try:
            runner.validate_source_lock(poisoned)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("exact-tree self-test accepted an unlisted source-lock member")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-protocol-lock", type=Path, default=DEFAULT_EVENT_PROTOCOL_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dit-root", type=Path, default=runner.DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path, default=runner.DEFAULT_VAE)
    parser.add_argument("--validate-input-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.validate_input_only:
        protocol, binding = event_lock_binding(args.event_protocol_lock)
        method = method_lock_binding(protocol)
        print(
            json.dumps(
                {
                    "valid": True,
                    "event_protocol_identity_sha256": protocol["identity_sha256"],
                    "event_binding": binding,
                    "method_binding": method,
                    "real_sampling_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = freeze(args)
    print(json.dumps({"source_lock": str(output), "real_sampling_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
