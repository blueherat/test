#!/usr/bin/env python3
"""Freeze the scientific-v4.2.1 B/E dynamic pipeline before real execution.

This source-only operation reads frozen protocol/source locks and code.  It
does not accept or inspect a trace plan, endpoint, label, score, embedding, or
external metric.  The resulting lock deliberately keeps ``execution_ready``
false; a later non-overwriting authorization lock must bind the selector,
anchor, endpoint/review source locks, and all readiness receipts before GPU
sampling is lawful.
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
    from .dit_scientific_v4_be_contract import (
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        DEFAULT_METHOD_LOCK,
        DEFAULT_SCIENTIFIC_PROTOCOL_LOCK,
        METHOD_LOCK_ID,
        SCIENTIFIC_PROTOCOL_ID,
        artifact_records,
        canonical_sha256,
        load_json,
        require_directory,
        require_regular,
        sha256_file,
        validate_manifest_tree,
        validate_method_lock,
        validate_scientific_protocol,
        without_identity,
        write_json,
    )
except ImportError:
    from dit_scientific_v4_be_contract import (  # type: ignore
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        DEFAULT_METHOD_LOCK,
        DEFAULT_SCIENTIFIC_PROTOCOL_LOCK,
        METHOD_LOCK_ID,
        SCIENTIFIC_PROTOCOL_ID,
        artifact_records,
        canonical_sha256,
        load_json,
        require_directory,
        require_regular,
        sha256_file,
        validate_manifest_tree,
        validate_method_lock,
        validate_scientific_protocol,
        without_identity,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_scientific_v4_2_1_endpoint_sampling_source_lock_v1"
)
DEFAULT_REVIEW_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_scientific_v4_2_1_review_pipeline_source_lock_v1"
)
STATUS = "SCIENTIFIC_V4_2_1_B_E_DYNAMIC_SOURCES_FROZEN_EXECUTION_NOT_READY"
ARTIFACT_KIND = "SCIENTIFIC_V4_2_1_B_E_DYNAMIC_SOURCE_LOCK_V1"

LIVE_SOURCES = {
    "dit_scientific_v4_be_contract.py": ROOT
    / "experiments/dit_scientific_v4_be_contract.py",
    "sample_dit_scientific_v4_be_traces.py": ROOT
    / "experiments/sample_dit_scientific_v4_be_traces.py",
    "calibrate_dit_scientific_v4_be.py": ROOT
    / "experiments/calibrate_dit_scientific_v4_be.py",
    "extract_dit_scientific_v4_be_products.py": ROOT
    / "experiments/extract_dit_scientific_v4_be_products.py",
    "evaluate_dit_scientific_v4_be.py": ROOT
    / "experiments/evaluate_dit_scientific_v4_be.py",
    "selftest_dit_scientific_v4_be.py": ROOT
    / "experiments/selftest_dit_scientific_v4_be.py",
    "freeze_dit_scientific_v4_2_1_be_dynamic_sources.py": Path(__file__).resolve(),
}
METHOD_V1_SOURCE = "sources/observe_dit_blur_focused_eprocess.py"
METHOD_V2_SOURCE = "sources/observe_dit_blur_focused_eprocess_v2.py"


def _validate_named_record_tree(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("upstream source-lock file records are malformed")
    result = {str(row.get("name")): dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("upstream source lock repeats a filename")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"manifest.json", "completion.json"}
    }
    if set(result) != observed:
        raise RuntimeError("upstream source-lock exact tree changed")
    for name, row in result.items():
        if set(row) != {"name", "bytes", "sha256"}:
            raise RuntimeError(f"upstream source record schema changed: {name}")
        path = require_regular(root / name, f"upstream source member {name}")
        if row["bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise RuntimeError(f"upstream source member changed: {name}")
    return result


def validate_endpoint_source_lock(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "scientific-v4.2.1 endpoint source lock")
    protocol_path = require_regular(root / "sampling_protocol.json", "endpoint protocol")
    manifest_path = require_regular(root / "manifest.json", "endpoint manifest")
    completion_path = require_regular(root / "completion.json", "endpoint completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    protocol_identity = protocol.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    if (
        canonical_sha256(without_identity(protocol)) != protocol_identity
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256") != protocol_identity
        or manifest.get("event_protocol_identity_sha256") != SCIENTIFIC_PROTOCOL_ID
        or protocol.get("event_protocol", {}).get("identity_sha256")
        != SCIENTIFIC_PROTOCOL_ID
        or protocol.get("execution_ready") is not False
        or protocol.get("real_endpoint_outputs_present_at_freeze") is not False
        or protocol.get("real_expert_label_review_or_consensus_results_present")
        is not False
        or completion.get("complete") is not True
        or completion.get("sampling_protocol_identity_sha256") != protocol_identity
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("sampling_protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("endpoint v4.2.1 source-lock identity/boundary changed")
    records = _validate_named_record_tree(root, manifest)
    if "sources/reproduce_dit_imagenet256.py" not in records:
        raise RuntimeError("endpoint source lock lacks the strict DiT reproducer")
    return protocol, manifest, completion


def validate_review_source_lock(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "scientific-v4.2.1 review source lock")
    contract_path = require_regular(root / "review_contract.json", "review contract")
    manifest_path = require_regular(root / "manifest.json", "review source manifest")
    completion_path = require_regular(root / "completion.json", "review source completion")
    review = load_json(contract_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    review_identity = review.get("identity_sha256")
    nested_identity = manifest.get("identity")
    if not isinstance(nested_identity, dict):
        raise RuntimeError("review source manifest lacks nested identity")
    envelope_without = dict(manifest)
    envelope_identity = envelope_without.pop("manifest_identity_sha256", None)
    if (
        canonical_sha256(without_identity(review)) != review_identity
        or canonical_sha256(without_identity(nested_identity))
        != nested_identity.get("identity_sha256")
        or canonical_sha256(envelope_without) != envelope_identity
        or manifest.get("status") != "complete"
        or review.get("event_protocol_identity_sha256") != SCIENTIFIC_PROTOCOL_ID
        or review.get("ready_for_real_sampling") is not False
        or review.get("real_expert_or_reviewer_results_present") is not False
        or review.get("real_v4_endpoint_labels_consensus_or_production_results_present")
        is not False
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != envelope_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("review v4.2.1 source-lock identity/boundary changed")
    expected = artifact_records(root)
    if manifest.get("files") != expected:
        raise RuntimeError("review source-lock exact tree or hashes changed")
    return review, manifest, completion


def _source_record(path: Path, *, origin: str) -> dict[str, Any]:
    path = require_regular(path, origin)
    return {"origin": origin, "path_at_freeze": str(path), "sha256": sha256_file(path)}


def build_contract(
    *,
    method_lock: Path,
    method_manifest: Mapping[str, Any],
    scientific_lock: Path,
    scientific_manifest: Mapping[str, Any],
    scientific_protocol: Mapping[str, Any],
    endpoint_lock: Path,
    endpoint_protocol: Mapping[str, Any],
    endpoint_manifest: Mapping[str, Any],
    review_lock: Path,
    review_contract: Mapping[str, Any],
    review_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_snapshots = {
        name: _source_record(path, origin=f"live:{name}")
        for name, path in LIVE_SOURCES.items()
    }
    strict_path = endpoint_lock / "sources/reproduce_dit_imagenet256.py"
    v1_path = method_lock / METHOD_V1_SOURCE
    v2_path = method_lock / METHOD_V2_SOURCE
    source_snapshots.update(
        {
            "reproduce_dit_imagenet256.py": _source_record(
                strict_path, origin="endpoint-source-lock:strict-reproducer"
            ),
            "observe_dit_blur_focused_eprocess.py": _source_record(
                v1_path, origin="method-v2.2-lock:v1-dependency"
            ),
            "observe_dit_blur_focused_eprocess_v1.py": _source_record(
                v1_path, origin="method-v2.2-lock:v1-loader-alias"
            ),
            "observe_dit_blur_focused_eprocess_v2.py": _source_record(
                v2_path, origin="method-v2.2-lock:v2-core"
            ),
        }
    )
    contract: dict[str, Any] = {
        "schema_version": 1,
        "status": STATUS,
        "artifact_kind": ARTIFACT_KIND,
        "execution_ready": False,
        "scientific_revision": "v4.2.1",
        "scientific_protocol": {
            "path": str(scientific_lock),
            "identity_sha256": SCIENTIFIC_PROTOCOL_ID,
            "manifest_identity_sha256": scientific_manifest["identity_sha256"],
            "protocol_file_sha256": sha256_file(scientific_lock / "protocol.json"),
            "manifest_file_sha256": sha256_file(scientific_lock / "manifest.json"),
            "completion_file_sha256": sha256_file(scientific_lock / "completion.json"),
        },
        "method_lock": {
            "path": str(method_lock),
            "identity_sha256": METHOD_LOCK_ID,
            "manifest_file_sha256": sha256_file(method_lock / "manifest.json"),
            "protocol_file_sha256": sha256_file(method_lock / "protocol.json"),
            "completion_file_sha256": sha256_file(method_lock / "completion.json"),
            "matched_q_identity_sha256": method_manifest[
                "matched_q_power_gate_identity"
            ],
            "adaptive_null_identity_sha256": method_manifest[
                "adaptive_null_audit_identity"
            ],
        },
        "endpoint_sampling_source_lock": {
            "path": str(endpoint_lock),
            "sampling_protocol_identity_sha256": endpoint_protocol["identity_sha256"],
            "manifest_identity_sha256": endpoint_manifest["identity_sha256"],
            "sampling_protocol_file_sha256": sha256_file(
                endpoint_lock / "sampling_protocol.json"
            ),
            "manifest_file_sha256": sha256_file(endpoint_lock / "manifest.json"),
        },
        "review_pipeline_source_lock": {
            "path": str(review_lock),
            "review_contract_identity_sha256": review_contract["identity_sha256"],
            "source_lock_identity_sha256": review_manifest["identity_sha256"],
            "manifest_identity_sha256": review_manifest["manifest_identity_sha256"],
            "review_contract_file_sha256": sha256_file(
                review_lock / "review_contract.json"
            ),
            "manifest_file_sha256": sha256_file(review_lock / "manifest.json"),
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
            "rng_domain": "eqvae.dit.event-rich.endpoint.v1",
            "rng_unit": ["global_seed", "class_id"],
            "full_2B_randn_like_each_of_250_transitions_including_t0": True,
            "terminal_t0_draw_consumed_then_masked": True,
            "task_order_worker_shard_resume_invariant": True,
            "same_global_seed_classes_share_initial_or_transition_innovation": False,
            "vae_scaling_factor": 0.18215,
        },
        "physical_artifact_contract": {
            "method_trace_root": "method_traces; exact 888 calibration+confirmation pairs; no endpoint/review/label",
            "review_endpoint_root": "review_endpoints; exact 768 confirmation PNG pairs; no trace/B/E/calibration/score",
            "B_E_mechanics_E_no_gate_one_shot_G_products_are_distinct_roots": True,
            "E_mechanics_exact_payloads": [
                "internal_tracks.npz predictable mechanics whitelist only",
                "label_free_mechanics_audit.json",
            ],
            "E_mechanics_forbids_score_logE_innovation_endpoint_external_or_label_payload": True,
        },
        "method_input_firewall": {
            "B_E_calibration_gate_alarm_and_future_rollback_internal_only": True,
            "endpoint_labels_used_only_in_separate_evaluator_join": True,
            "FID_Inception_DINO_CLIP_embeddings_external_distances_forbidden": True,
            "external_visual_labels_used_for_evaluation_cohort_enrichment": True,
            "external_representations_used_for_cohort_selection": False,
            "external_inputs_used_by_B_or_E": False,
        },
        "frozen_evaluation_contract": {
            "prelabel_confirmation_mechanics_rows": 768,
            "primary_permutation_draws": 100000,
            "primary_permutation_seed": 2026082801,
            "Holm_family_exactly": ["B_persistence", "E_blur_gated_running_max_log"],
            "paired_bootstrap_seeds": {
                "E_minus_B": 2026082811,
                "E_minus_no_state_gate": 2026082812,
                "E_minus_prespecified_scalar_G": 2026082813,
                "E_minus_one_shot_descriptive_claim": 2026082814,
                "schedule_exact_conditional_descriptive": 2026082815,
            },
            "paired_bootstrap_draws": 100000,
            "hard_gate_lower_order_index": 4999,
            "one_shot_required_for_rollback": False,
            "schedule_exact_conditional_has_pass_fail_or_rollback_role": False,
            "alpha_0p10_is_overall_anytime_trigger_budget_not_clean_FPR": True,
        },
        "no_touch_contract": {
            "fixed_pair": "calibration seed 1100, selected_classes[0]",
            "all_seven_trace_arrays_endpoint_tensor_and_rng_boundaries_bitwise_bound": True,
            "manifest_bound_not_self_attested": True,
        },
        "execution_activation_requirements": {
            "this_source_lock_alone_authorizes_sampling": False,
            "required_before_any_execution": [
                "v4.2.1 selector and anchor plan lock",
                "v4.2.1 endpoint source lock",
                "v4.2.1 review source lock",
                "v4.2.1 scientific selftest lock",
                "new immutable global execution-authorization receipt binding all identities",
            ],
            "trace_plan_decision_go_alone_is_not_execution_authority": True,
        },
        "source_snapshots": source_snapshots,
        "evidence_access_audit": {
            "trace_plan_endpoint_trace_review_label_score_opened": False,
            "FID_Inception_DINO_CLIP_embedding_or_external_distance_opened": False,
            "real_GPU_sampling_or_evaluation_run": False,
        },
    }
    # Bind that the exact protocol passed validation while keeping the full
    # protocol outside this source artifact.
    contract["scientific_protocol_validated_identity_sha256"] = scientific_protocol[
        "identity_sha256"
    ]
    contract["identity_sha256"] = canonical_sha256(contract)
    return contract


def expected_payload_names() -> set[str]:
    return {
        "dynamic_contract.json",
        "selftest_receipt.json",
        *{f"sources/{name}" for name in LIVE_SOURCES},
        "sources/reproduce_dit_imagenet256.py",
        "sources/observe_dit_blur_focused_eprocess.py",
        "sources/observe_dit_blur_focused_eprocess_v1.py",
        "sources/observe_dit_blur_focused_eprocess_v2.py",
    }


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, completion = validate_manifest_tree(root)
    contract_path = require_regular(root / "dynamic_contract.json", "dynamic contract")
    contract = load_json(contract_path)
    identity = contract.get("identity_sha256")
    if (
        canonical_sha256(without_identity(contract)) != identity
        or contract.get("status") != STATUS
        or contract.get("artifact_kind") != ARTIFACT_KIND
        or contract.get("execution_ready") is not False
        or contract.get("scientific_protocol", {}).get("identity_sha256")
        != SCIENTIFIC_PROTOCOL_ID
        or contract.get("method_lock", {}).get("identity_sha256") != METHOD_LOCK_ID
        or contract.get("method_input_firewall", {}).get("external_inputs_used_by_B_or_E")
        is not False
        or contract.get("method_input_firewall", {}).get(
            "external_visual_labels_used_for_evaluation_cohort_enrichment"
        )
        is not True
        or contract.get("method_input_firewall", {}).get(
            "FID_Inception_DINO_CLIP_embeddings_external_distances_forbidden"
        )
        is not True
        or contract.get("execution_activation_requirements", {}).get(
            "this_source_lock_alone_authorizes_sampling"
        )
        is not False
        or manifest.get("artifact_kind") != ARTIFACT_KIND
        or manifest.get("dynamic_contract_identity_sha256") != identity
        or manifest.get("scientific_protocol_identity_sha256")
        != SCIENTIFIC_PROTOCOL_ID
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or completion.get("dynamic_contract_identity_sha256") != identity
        or completion.get("dynamic_contract_file_sha256") != sha256_file(contract_path)
    ):
        raise RuntimeError("v4.2.1 dynamic source-lock identity/boundary changed")
    if set(row["name"] for row in manifest["files"]) != expected_payload_names():
        raise RuntimeError("v4.2.1 dynamic source-lock payload set changed")
    snapshots = contract.get("source_snapshots")
    if not isinstance(snapshots, dict):
        raise RuntimeError("dynamic source lock lacks source snapshots")
    by_name = {row["name"]: row for row in manifest["files"]}
    for name, record in snapshots.items():
        frozen = f"sources/{name}"
        if by_name.get(frozen, {}).get("sha256") != record.get("sha256"):
            raise RuntimeError(f"frozen source differs from contract: {name}")
    return contract, manifest


def run_frozen_selftests(staging: Path) -> dict[str, Any]:
    source_dir = staging / "sources"
    commands = [
        [sys.executable, str(source_dir / "dit_scientific_v4_be_contract.py")],
        [sys.executable, str(source_dir / "selftest_dit_scientific_v4_be.py")],
        [sys.executable, str(source_dir / "sample_dit_scientific_v4_be_traces.py"), "--help"],
        [sys.executable, str(source_dir / "calibrate_dit_scientific_v4_be.py"), "--help"],
        [sys.executable, str(source_dir / "extract_dit_scientific_v4_be_products.py"), "--help"],
        [sys.executable, str(source_dir / "evaluate_dit_scientific_v4_be.py"), "--help"],
    ]
    environment = dict(os.environ)
    environment.update(
        {
            # Executing an absolute script gives it source_dir as sys.path[0].
            # Keep PYTHONPATH genuinely empty so no live repository module can
            # satisfy a fallback import.
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    results = []
    isolated_cwd = Path(tempfile.mkdtemp(prefix="v421-be-frozen-cwd-"))
    try:
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=isolated_cwd,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            result = {
                "command_basename": Path(command[1]).name,
                "arguments": command[2:],
                "returncode": completed.returncode,
                "stdout_sha256": canonical_sha256(completed.stdout),
                "stderr_sha256": canonical_sha256(completed.stderr),
            }
            results.append(result)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"frozen source selftest failed: {command}\n{completed.stderr}"
                )
    finally:
        shutil.rmtree(isolated_cwd, ignore_errors=True)
    return {
        "status": "PASS_CPU_SYNTHETIC_NO_GPU_NO_LABEL_NO_ENDPOINT",
        "test_count": len(results),
        "tests": results,
        "empty_external_cwd": True,
        "PYTHONPATH_empty_and_script_directory_is_frozen_sources": True,
        "real_GPU_sampling_run": False,
        "endpoint_label_score_embedding_opened": False,
    }


def run_post_envelope_loader_test(staging: Path) -> None:
    """Exercise the real source-lock loader after manifest/completion exist."""

    source_dir = (staging / "sources").resolve()
    code = "\n".join(
        [
            "import pathlib, sys, types",
            f"source_dir = pathlib.Path({str(source_dir)!r}).resolve()",
            f"lock_root = pathlib.Path({str(staging.resolve())!r}).resolve()",
            "sys.path.insert(0, str(source_dir))",
            "poison = types.ModuleType('observe_dit_blur_focused_eprocess')",
            "poison.TOKEN = 'LIVE_POISON_MUST_NOT_LOAD'",
            "sys.modules['observe_dit_blur_focused_eprocess'] = poison",
            "import sample_dit_scientific_v4_be_traces as sampler",
            "contract, manifest, strict, core = sampler.load_source_lock(lock_root)",
            "assert contract['execution_ready'] is False",
            "assert sys.modules['observe_dit_blur_focused_eprocess'] is poison",
            "assert pathlib.Path(core.__file__).resolve().parent == source_dir",
            "assert pathlib.Path(core.v1.__file__).resolve().parent == source_dir",
            "assert pathlib.Path(sampler.be_contract.__file__).resolve().parent == source_dir",
            "assert manifest['execution_ready'] is False",
        ]
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    isolated_cwd = Path(tempfile.mkdtemp(prefix="v421-be-loader-cwd-"))
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=isolated_cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(isolated_cwd, ignore_errors=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "post-envelope frozen loader/tamper test failed:\n" + completed.stderr
        )


def freeze(args: argparse.Namespace) -> None:
    method_lock = require_directory(args.method_lock, "method-v2.2 lock")
    method_manifest, _ = validate_method_lock(method_lock)
    scientific_lock = require_directory(args.scientific_protocol_lock, "v4.2.1 lock")
    scientific_manifest, scientific_protocol = validate_scientific_protocol(
        scientific_lock
    )
    endpoint_lock = require_directory(args.endpoint_source_lock, "v4.2.1 endpoint lock")
    endpoint_protocol, endpoint_manifest, _ = validate_endpoint_source_lock(endpoint_lock)
    review_lock = require_directory(args.review_source_lock, "v4.2.1 review lock")
    review_contract, review_manifest, _ = validate_review_source_lock(review_lock)
    contract = build_contract(
        method_lock=method_lock,
        method_manifest=method_manifest,
        scientific_lock=scientific_lock,
        scientific_manifest=scientific_manifest,
        scientific_protocol=scientific_protocol,
        endpoint_lock=endpoint_lock,
        endpoint_protocol=endpoint_protocol,
        endpoint_manifest=endpoint_manifest,
        review_lock=review_lock,
        review_contract=review_contract,
        review_manifest=review_manifest,
    )
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite immutable dynamic source lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "dynamic_contract.json", contract)
        source_dir = staging / "sources"
        source_dir.mkdir()
        for name, path in LIVE_SOURCES.items():
            shutil.copy2(require_regular(path, f"live source {name}"), source_dir / name)
        shutil.copy2(
            require_regular(
                endpoint_lock / "sources/reproduce_dit_imagenet256.py",
                "strict DiT reproducer",
            ),
            source_dir / "reproduce_dit_imagenet256.py",
        )
        v1 = require_regular(method_lock / METHOD_V1_SOURCE, "method v1 dependency")
        shutil.copy2(v1, source_dir / "observe_dit_blur_focused_eprocess.py")
        shutil.copy2(v1, source_dir / "observe_dit_blur_focused_eprocess_v1.py")
        shutil.copy2(
            require_regular(method_lock / METHOD_V2_SOURCE, "method v2.2 core"),
            source_dir / "observe_dit_blur_focused_eprocess_v2.py",
        )
        write_json(staging / "selftest_receipt.json", run_frozen_selftests(staging))
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "artifact_kind": ARTIFACT_KIND,
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": SCIENTIFIC_PROTOCOL_ID,
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "endpoint_source_manifest_identity_sha256": endpoint_manifest[
                "identity_sha256"
            ],
            "review_source_lock_identity_sha256": review_manifest[
                "identity_sha256"
            ],
            "execution_ready": False,
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "artifact_kind": ARTIFACT_KIND,
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "dynamic_contract_file_sha256": sha256_file(
                staging / "dynamic_contract.json"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "execution_ready": False,
        }
        write_json(staging / "completion.json", completion)
        validate_lock(staging)
        run_post_envelope_loader_test(staging)
        os.replace(staging, output)
        validate_lock(output)
        run_post_envelope_loader_test(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "frozen_execution_not_ready",
                "output": str(output),
                "dynamic_contract_identity_sha256": contract["identity_sha256"],
                "dynamic_source_manifest_identity_sha256": manifest[
                    "identity_sha256"
                ],
                "real_GPU_sampling_run": False,
                "labels_endpoints_scores_external_representations_opened": False,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method-lock", type=Path, default=DEFAULT_METHOD_LOCK)
    result.add_argument(
        "--scientific-protocol-lock",
        type=Path,
        default=DEFAULT_SCIENTIFIC_PROTOCOL_LOCK,
    )
    result.add_argument(
        "--endpoint-source-lock", type=Path, default=DEFAULT_ENDPOINT_SOURCE_LOCK
    )
    result.add_argument(
        "--review-source-lock", type=Path, default=DEFAULT_REVIEW_SOURCE_LOCK
    )
    result.add_argument("--output", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    result.add_argument("--validate", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.validate is not None:
        contract, manifest = validate_lock(args.validate)
        print(
            json.dumps(
                {
                    "status": "valid_execution_not_ready",
                    "dynamic_contract_identity_sha256": contract["identity_sha256"],
                    "manifest_identity_sha256": manifest["identity_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    freeze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
