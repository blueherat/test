#!/usr/bin/env python3
"""Freeze the v3 dynamic-trace, B/C-product, and two-stage evaluator sources.

This operation reads only the already frozen scientific protocol and endpoint
sampling source lock.  It does not accept an anchor plan, endpoint, trace,
review, label, candidate score, embedding, or external metric.  The resulting
source lock is executable infrastructure, not a result and not permission to
begin real sampling before the v3 readiness dependencies are satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from .dit_event_rich_dynamic_contract import (
        B_CANDIDATE,
        B_CHECKPOINTS,
        B_FEATURE,
        C_CANDIDATE,
        C_CHECKPOINTS,
        C_FEATURE,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        DEFAULT_ENDPOINT_SOURCE_LOCK,
        DEFAULT_EVENT_PROTOCOL_LOCK,
        artifact_records,
        canonical_sha256,
        load_json,
        require_directory,
        require_regular,
        sha256_file,
        validate_event_protocol,
        without_identity,
        write_json,
    )
except ImportError:
    from dit_event_rich_dynamic_contract import (  # type: ignore
        B_CANDIDATE,
        B_CHECKPOINTS,
        B_FEATURE,
        C_CANDIDATE,
        C_CHECKPOINTS,
        C_FEATURE,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        DEFAULT_ENDPOINT_SOURCE_LOCK,
        DEFAULT_EVENT_PROTOCOL_LOCK,
        artifact_records,
        canonical_sha256,
        load_json,
        require_directory,
        require_regular,
        sha256_file,
        validate_event_protocol,
        without_identity,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "dit_event_rich_dynamic_contract.py": ROOT
    / "experiments/dit_event_rich_dynamic_contract.py",
    "sample_dit_event_rich_dynamic_traces.py": ROOT
    / "experiments/sample_dit_event_rich_dynamic_traces.py",
    "extract_dit_event_rich_candidate_product.py": ROOT
    / "experiments/extract_dit_event_rich_candidate_product.py",
    "evaluate_dit_event_rich_dynamic_confirmation.py": ROOT
    / "experiments/evaluate_dit_event_rich_dynamic_confirmation.py",
    "freeze_dit_event_rich_dynamic_confirmation_sources.py": Path(__file__).resolve(),
}


def validate_endpoint_source_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "event-rich endpoint sampling source lock")
    protocol_path = require_regular(root / "sampling_protocol.json", "endpoint sampling protocol")
    manifest_path = require_regular(root / "manifest.json", "endpoint source manifest")
    completion_path = require_regular(root / "completion.json", "endpoint source completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    protocol_identity = protocol.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    if (
        canonical_sha256(without_identity(protocol)) != protocol_identity
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or protocol.get("status") != "FROZEN_BEFORE_EVENT_RICH_ENDPOINT_GPU_SAMPLING"
        or manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256") != protocol_identity
        or completion.get("complete") is not True
        or completion.get("sampling_protocol_identity_sha256") != protocol_identity
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("sampling_protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("endpoint sampling source lock identity changed")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("endpoint source-lock member list is malformed")
    by_name = {str(row.get("name")): row for row in rows}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "completion.json"}
    }
    if set(by_name) != observed:
        raise RuntimeError("endpoint source-lock member set changed")
    for name, row in by_name.items():
        path = require_regular(root / name, f"endpoint source member {name}")
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"endpoint source member changed: {name}")
    return protocol, manifest


def build_contract(
    *,
    event_lock: Path,
    event_protocol: Mapping[str, Any],
    endpoint_lock: Path,
    endpoint_protocol: Mapping[str, Any],
    endpoint_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_hashes = {
        name: {
            "live_path_at_freeze": str(require_regular(path, f"live source {name}")),
            "sha256": sha256_file(path),
        }
        for name, path in SOURCES.items()
    }
    strict_source = require_regular(
        endpoint_lock / "sources/reproduce_dit_imagenet256.py",
        "frozen strict reproduction helper",
    )
    source_hashes["reproduce_dit_imagenet256.py"] = {
        "inherited_from_endpoint_source_lock": str(strict_source),
        "sha256": sha256_file(strict_source),
    }
    contract: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_V3_DYNAMIC_CONFIRMATION_INFRASTRUCTURE_BEFORE_REAL_TRACE_SAMPLING",
        "scientific_version": "event-rich confirmation protocol v3; B/C only",
        "event_protocol": {
            "path": str(event_lock),
            "identity_sha256": event_protocol["identity_sha256"],
            "file_sha256": sha256_file(event_lock / "protocol.json"),
            "manifest_file_sha256": sha256_file(event_lock / "manifest.json"),
            "completion_file_sha256": sha256_file(event_lock / "completion.json"),
        },
        "upstream_endpoint_source_lock": {
            "path": str(endpoint_lock),
            "sampling_protocol_identity_sha256": endpoint_protocol["identity_sha256"],
            "manifest_identity_sha256": endpoint_manifest["identity_sha256"],
            "sampling_protocol_file_sha256": sha256_file(
                endpoint_lock / "sampling_protocol.json"
            ),
            "manifest_file_sha256": sha256_file(endpoint_lock / "manifest.json"),
        },
        "assets": endpoint_protocol["assets"],
        "sampler_contract": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "sampling_steps": 250,
            "clip_denoised": False,
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "classes_per_invocation": 1,
            "initial_latent_shape": [1, 4, 32, 32],
            "duplicated_cfg_state_shape": [2, 4, 32, 32],
            "rng_unit": ["global_seed", "class_id"],
            "rng_domain": "eqvae.dit.event-rich.endpoint.v1",
            "pair_seed_derivation": (
                "uint64_be(first_8_bytes(SHA256(ASCII(domain + NUL + global_seed + "
                "NUL + class_id)))) mod 2^63"
            ),
            "manual_seed_after_model_and_vae_load_immediately_before_initial_latent": True,
            "full_2B_randn_like_each_of_250_transitions_including_t0": True,
            "terminal_t0_draw_consumed_then_masked": True,
            "task_order_worker_shard_resume_invariant": True,
            "same_global_seed_classes_share_initial_or_transition_innovation": False,
            "vae": "stabilityai/sd-vae-ft-mse pinned snapshot, FP32",
            "vae_scaling_factor": 0.18215,
        },
        "dynamic_axis_contract": {
            "authority": "immutable selector anchor output validated against protocol v3",
            "active_classes": "exact active_union_classes; GO candidate union only; 0 or 6..12",
            "calibration_seeds": list(range(1100, 1120)),
            "confirmation_seeds": list(range(1200, 1328)),
            "calibration_rows_per_active_class": 20,
            "confirmation_rows_per_active_class": 128,
            "no_additional_class_seed_or_candidate_after_anchor_plan": True,
        },
        "minimum_trace_contract": {
            "endpoint_png": "RGB 256x256 for the independent blind label pipeline",
            "always": {"final_latent": [4, 32, 32]},
            B_CANDIDATE: {
                "only_for_B_scope_classes": True,
                "array": "b_pred_xstart",
                "shape": [len(B_CHECKPOINTS), 4, 32, 32],
                "sampling_steps": list(B_CHECKPOINTS),
            },
            C_CANDIDATE: {
                "only_for_C_scope_classes": True,
                "arrays": ["c_pred_xstart_c3", "c_alpha_bar"],
                "shapes": [[len(C_CHECKPOINTS), 32, 32], [len(C_CHECKPOINTS)]],
                "sampling_steps": list(C_CHECKPOINTS),
            },
            "full_state_mean_sigma_innovation_raw_cfg_arrays_saved": False,
            "labels_scores_embeddings_or_external_metrics_saved": False,
        },
        "candidate_product_contract": {
            "physical_separation": "one immutable root per candidate",
            "exact_columns": {
                B_CANDIDATE: ["phase", "global_seed", "class_id", B_FEATURE],
                C_CANDIDATE: ["phase", "global_seed", "class_id", C_FEATURE],
            },
            "legacy_placeholder_columns_forbidden": ["label", "raw_consensus_label"],
            "all_label_review_consensus_phenotype_columns_forbidden": True,
            "other_candidate_column_forbidden": True,
            "column_roles_and_formula_catalog_required": True,
        },
        "evaluation_contract": {
            "stage_A": (
                "opens only final-consensus manifest/completion and aggregate_counts; "
                "independently gates B>=15 blur events and C>=30 total clear-bad, each "
                "also requiring >=60 clean and >=3 event-bearing/comparable classes"
            ),
            "stage_B": (
                "separate process; opens evaluation_labels and only each authorized "
                "candidate's physically isolated single-score product"
            ),
            "gated_off_candidate": "raw permutation p=1; product path untouched",
            "primary_statistic": "within-class pair-count-weighted tie-aware directional ROC AUC",
            "permutation": {
                "draws": 100000,
                "rng": "numpy.default_rng(PCG64(seed=2026082801))",
                "unit": "one common intact 128-global-seed label/phenotype block permutation",
            },
            "multiple_testing": "Holm over exactly B and C",
            "thresholds": (
                "within candidate/class, 20 calibration scores: alpha0.10 is B upper "
                "19th or C lower 2nd order statistic; alpha0.05 is B upper 20th or C lower 1st; strict"
            ),
            "aggregate_output_only": True,
            "B_intervention_authority_only_if_all_original_gates_pass": True,
            "C_never_authorizes_intervention_by_itself": True,
        },
        "review_handoff_contract": {
            "artifact_kind": "EVENT_RICH_FINAL_CONSENSUS_LABEL_LOCK_V1",
            "stage_A_payload": "aggregate_counts.json only after manifest/completion",
            "stage_B_payload": "evaluation_labels.csv",
            "evaluation_label_columns": [
                "phase",
                "global_seed",
                "class_id",
                "final_severity",
                "blur_component",
            ],
            "aggregate_requires_per_class_clean_mild_clear_and_blur_counts": True,
            "labels_locked_before_candidate_products_open": True,
        },
        "method_boundary": {
            "internal_candidates": [B_CANDIDATE, C_CANDIDATE],
            "Inception_DINO_FID_embeddings_external_distances": (
                "forbidden as input, candidate, selection, gate, trigger, threshold, "
                "intervention signal, or claimed method"
            ),
            "endpoint_labels": "external evaluation only",
            "no_E_or_future_candidate_silently_added_to_v3": True,
        },
        "resume_and_integrity": {
            "no_overwrite": True,
            "completed_pairs_reused_only_after_full_hash_validation": True,
            "partial_or_changed_pair": "preserve and fail closed",
            "pool_finalization_requires_exact_dynamic_axis": True,
            "source_protocol_anchor_plan_pool_product_consensus_and_stage_receipt_identities_bound": True,
        },
        "execution_readiness": {
            "this_source_lock_alone_authorizes_real_sampling": False,
            "requires_protocol_v3_readiness_dependencies_and_immutable_anchor_plan": True,
        },
        "source_snapshots": source_hashes,
        "evidence_access_audit": {
            "anchor_plan_endpoint_trace_review_label_score_opened": False,
            "Inception_DINO_FID_embedding_or_external_distance_opened": False,
            "real_sampling_or_evaluation_run": False,
        },
    }
    contract["identity_sha256"] = canonical_sha256(contract)
    return contract


def validate_lock(root: Path) -> dict[str, Any]:
    root = require_directory(root, "dynamic confirmation source lock")
    contract_path = require_regular(root / "dynamic_contract.json", "dynamic contract")
    manifest_path = require_regular(root / "manifest.json", "dynamic source manifest")
    completion_path = require_regular(root / "completion.json", "dynamic source completion")
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    contract_identity = contract.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    if (
        canonical_sha256(without_identity(contract)) != contract_identity
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or contract.get("status")
        != "FROZEN_V3_DYNAMIC_CONFIRMATION_INFRASTRUCTURE_BEFORE_REAL_TRACE_SAMPLING"
        or manifest.get("status") != "complete"
        or manifest.get("dynamic_contract_identity_sha256") != contract_identity
        or manifest.get("event_protocol_identity_sha256")
        != contract.get("event_protocol", {}).get("identity_sha256")
        or completion.get("complete") is not True
        or completion.get("dynamic_contract_identity_sha256") != contract_identity
        or completion.get("dynamic_contract_file_sha256") != sha256_file(contract_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("dynamic confirmation source lock identity mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("dynamic source manifest file list malformed")
    by_name = {str(row.get("name")): row for row in rows}
    expected = {
        "dynamic_contract.json",
        *{f"sources/{name}" for name in SOURCES},
        "sources/reproduce_dit_imagenet256.py",
        "selftest_receipt.json",
    }
    if set(by_name) != expected or len(by_name) != len(rows):
        raise RuntimeError("dynamic source-lock member set changed")
    for name, row in by_name.items():
        path = require_regular(root / name, f"dynamic source member {name}")
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"dynamic source member changed: {name}")
    return contract


def run_frozen_selftests(staging: Path) -> dict[str, Any]:
    commands = [
        [sys.executable, str(staging / "sources/dit_event_rich_dynamic_contract.py")],
        [
            sys.executable,
            str(staging / "sources/sample_dit_event_rich_dynamic_traces.py"),
            "self-test",
        ],
        [
            sys.executable,
            str(staging / "sources/extract_dit_event_rich_candidate_product.py"),
            "--self-test",
        ],
        [
            sys.executable,
            str(staging / "sources/evaluate_dit_event_rich_dynamic_confirmation.py"),
            "self-test",
        ],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"frozen source self-test failed: {command}\n{completed.stderr}")
    return {
        "status": "all_frozen_source_selftests_passed_without_GPU_or_real_data",
        "tests": results,
    }


def freeze(args: argparse.Namespace) -> None:
    event_lock = require_directory(args.event_protocol_lock, "event protocol lock")
    event_protocol = validate_event_protocol(event_lock)
    endpoint_lock = require_directory(args.endpoint_source_lock, "endpoint source lock")
    endpoint_protocol, endpoint_manifest = validate_endpoint_source_lock(endpoint_lock)
    if endpoint_protocol.get("event_protocol", {}).get("identity_sha256") != event_protocol["identity_sha256"]:
        raise RuntimeError("endpoint sampler and dynamic infrastructure bind different event protocols")
    contract = build_contract(
        event_lock=event_lock,
        event_protocol=event_protocol,
        endpoint_lock=endpoint_lock,
        endpoint_protocol=endpoint_protocol,
        endpoint_manifest=endpoint_manifest,
    )
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite immutable source lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "dynamic_contract.json", contract)
        source_dir = staging / "sources"
        source_dir.mkdir()
        for name, path in SOURCES.items():
            shutil.copy2(require_regular(path, f"live source {name}"), source_dir / name)
        shutil.copy2(
            require_regular(
                endpoint_lock / "sources/reproduce_dit_imagenet256.py",
                "frozen strict helper",
            ),
            source_dir / "reproduce_dit_imagenet256.py",
        )
        receipt = run_frozen_selftests(staging)
        write_json(staging / "selftest_receipt.json", receipt)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "event_protocol_identity_sha256": event_protocol["identity_sha256"],
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "dynamic_contract_file_sha256": sha256_file(
                staging / "dynamic_contract.json"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
        }
        write_json(staging / "completion.json", completion)
        validate_lock(staging)
        os.replace(staging, output)
        validate_lock(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "frozen",
                "output": str(output),
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "real_sampling_run": False,
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-protocol-lock", type=Path, default=DEFAULT_EVENT_PROTOCOL_LOCK)
    parser.add_argument("--endpoint-source-lock", type=Path, default=DEFAULT_ENDPOINT_SOURCE_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    parser.add_argument("--validate", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.validate is not None:
        contract = validate_lock(args.validate)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "dynamic_contract_identity_sha256": contract["identity_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    freeze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
