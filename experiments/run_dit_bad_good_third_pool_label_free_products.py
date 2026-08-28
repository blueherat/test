#!/usr/bin/env python3
"""Produce the three frozen label-free products for the sealed DiT third pool.

The source/protocol lock is created before the pool is inspected.  After the
sampling runner publishes a complete pool receipt, ``--bind-pool`` performs a
full immutable validation of all 600 seed bundles and records their manifest,
completion, scientific-identity, and trace hashes.  ``--launch`` accepts only
that binding, builds the broad primary inventory first, then runs the frozen
visual and endpoint extractors concurrently on two distinct GPUs.

The endpoint Inception/DINO representations are external, terminal-only,
read-only secondary diagnostics.  They are never a method candidate, online
trigger, threshold, intervention input, or a way to rescue a failed internal
B/C trajectory hypothesis.  This launcher extracts embeddings only; it does
not compute endpoint distances, FID, AUC, or selection.

No label, review, consensus, screen result, candidate score, calibration,
threshold, or alert path is accepted by this interface.  Existing product
directories are reused only after full validation; invalid or partial paths are
preserved and refused, and no product is overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence


def find_repo_root(source: Path) -> Path:
    for candidate in source.resolve().parents:
        if (candidate / ".git").exists() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate repository root from {source}")


ROOT = find_repo_root(Path(__file__))
DATA_ROOT = Path("/data/users/zhoushunyu/eqvae")
DEFAULT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_label_free_products_source_lock_v1"
)
DEFAULT_POOL_BINDING = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_label_free_pool_binding_v1"
)
SAMPLING_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_sampling_source_lock_v1"
)
ENDPOINT_SECONDARY_LOCK = (
    ROOT / "experiments/locks/dit_third_pool_endpoint_secondary_protocol_lock_v1"
)
POOL_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/dit_bad_good_third_pool_v1_custom_traces_cfg_locked"
)
PRODUCT_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_confirmation_third_pool_v1"
)
PRIMARY_OUTPUT = PRODUCT_ROOT / "primary_label_free_v1"
VISUAL_OUTPUT = PRODUCT_ROOT / "predxstart_visual_label_free_v1"
ENDPOINT_OUTPUT = PRODUCT_ROOT / "endpoint_embeddings_label_free_v1"
RECEIPT_OUTPUT = PRODUCT_ROOT / "label_free_products_receipt_v1"

EXPECTED_SAMPLING_PROTOCOL_IDENTITY = (
    "330661e87de7846e1f590660f03ecef6270fa45e2f39c4fc54d992e3260950d8"
)
EXPECTED_SAMPLING_MANIFEST_IDENTITY = (
    "eae86d48c1c1b9c732fbeea4838b2418b9b7261b61db0355fd7306469f5b6df3"
)
EXPECTED_SAMPLING_RUNNER_SHA256 = (
    "aa1c9c906e610c1e4bef5bdd92218770760367494a9d4c72b489826f57b6b0c0"
)
EXPECTED_ENDPOINT_PROTOCOL_IDENTITY = (
    "575e7a8081144f17d16d99387561f950d7551017faf7d6b0a2b5d93a921a1bdd"
)
EXPECTED_ENDPOINT_MANIFEST_IDENTITY = (
    "905a634ba68e0f3277e7eae3f4218951593e2686da3deef203e55bac86cc8500"
)
EXPECTED_PRIMARY_SOURCE_SHA256 = (
    "acc348c7aa94ffe53d59ef4268c669fd948448de86f67202573bd04b69e9e129"
)
EXPECTED_VISUAL_SOURCE_SHA256 = (
    "452ae0e61fe36d027036e0d74c232fbcfbd7cb462d3749db92e062a104d0e398"
)
EXPECTED_ENDPOINT_SOURCE_SHA256 = (
    "5f3d52cb24e89e2c92639e70200f72f1e739344906550385c46ea2dfff343f8b"
)
EXPECTED_REPRODUCTION_SOURCE_SHA256 = (
    "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"
)
RESNET18_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
INCEPTION_SHA256 = (
    "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
)
VAE_FILE_SHA256 = {
    "config.json": "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e",
    "diffusion_pytorch_model.safetensors": (
        "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"
    ),
}
DINO_FILE_SHA256 = {
    "config.json": "03eee42f646659a9480f8911a81fdd81efeedd7ff39083c8e36398068daf72f5",
    "model.safetensors": (
        "edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b"
    ),
    "preprocessor_config.json": (
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
    ),
}

CLASSES = (207, 602, 795)
SEEDS = tuple(range(250, 850))
TRAJECTORY_COUNT = len(CLASSES) * len(SEEDS)
CHECKPOINTS = (69, 79, 89, 99, 109, 119, 129, 139, 149)
VISUAL_PHYSICAL_GPU = "0"
ENDPOINT_PHYSICAL_GPU = "3"
LOGICAL_DEVICE = "cuda:0"
VISUAL_DECODE_BATCH = 27
VISUAL_CLASSIFIER_BATCH = 64
ENDPOINT_BATCH = 16
PRIMARY_FEATURE = (
    "pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump"
)
VISUAL_FEATURE = "decoded_local_blur_severity__mean"

SOURCE_BASENAMES = (
    "run_dit_bad_good_third_pool_label_free_products.py",
    "freeze_dit_bad_good_third_pool_label_free_products.py",
    "run_dit_bad_good_third_pool.py",
    "analyze_dit_bad_good_custom_traces.py",
    "extract_dit_predxstart_visual_tracks.py",
    "extract_dit_endpoint_embeddings_label_free.py",
    "reproduce_dit_imagenet256.py",
)


def expected_products_contract() -> dict[str, Any]:
    return {
        "primary": {
            "experiment": "dit_bad_good_custom_trace_metric_discovery",
            "kind": "broad fixed label-free trajectory feature inventory",
            "required_feature": PRIMARY_FEATURE,
            "row_count": TRAJECTORY_COUNT,
            "source_inventory_is_only_downstream_trace_input": True,
            "consensus_argument_allowed": False,
        },
        "visual": {
            "experiment": "dit_predxstart_preterminal_visual_tracks_label_free",
            "required_feature": VISUAL_FEATURE,
            "sampling_checkpoints": list(CHECKPOINTS),
            "internal_timesteps": [249 - value for value in CHECKPOINTS],
            "row_count": TRAJECTORY_COUNT,
            "decoded_images_saved": False,
        },
        "endpoint": {
            "experiment": "dit_endpoint_embeddings_label_free_v1",
            "representations": {
                "inception_fid_pool2048": 2048,
                "dinov2_registers_large_cls1024": 1024,
            },
            "row_count": TRAJECTORY_COUNT,
            "distances_centroids_auc_or_selection_computed": False,
            "terminal_only": True,
            "role": "external_read_only_endpoint_secondary_diagnostic",
            "eligible_as_method_candidate_trigger_threshold_or_intervention": False,
            "may_rescue_or_replace_failed_internal_B_or_C_hypothesis": False,
        },
    }


def expected_role_boundaries() -> dict[str, Any]:
    return {
        "internal_trajectory_products": {
            "B": VISUAL_FEATURE,
            "C": PRIMARY_FEATURE,
            "scientific_role": "internal trajectory hypotheses",
        },
        "endpoint_E1_E2": {
            "representations": [
                "inception_fid_pool2048",
                "dinov2_registers_large_cls1024",
            ],
            "scientific_role": "external read-only terminal secondary diagnostics",
            "candidate_or_trigger": False,
            "threshold_or_intervention_input": False,
            "may_rescue_internal_metric_failure": False,
            "launcher_computes_distance_fid_auc_or_selection": False,
        },
        "cross_family_combination_allowed": False,
    }


def expected_resume_contract() -> dict[str, Any]:
    return {
        "overwrite_allowed": False,
        "existing_product_action": (
            "full immutable product validation, then reuse; otherwise fail closed"
        ),
        "partial_or_invalid_product_action": "preserve and refuse",
        "successful_parallel_sibling_may_be_reused_after_other_sibling_failure": True,
        "existing_complete_receipt_action": (
            "full pool, three-product, plan, log, and receipt revalidation"
        ),
    }


def expected_supervision_contract() -> dict[str, Any]:
    return {
        "labels_reviews_consensus_opened": False,
        "screen_results_opened": False,
        "preexisting_candidate_score_products_thresholds_or_alerts_opened": False,
        "score_label_join_performed": False,
        "raw_label_free_features_generated": True,
        "generated_raw_feature_values_read_only_for_full_product_validation": True,
        "selection_auc_or_thresholding_performed": False,
        "no_supervised_input_argument_exists": True,
    }


def expected_pool_binding_contract() -> dict[str, Any]:
    return {
        "binding_created_only_after_full_sampling_validate_complete_pool": True,
        "exact_recursive_pool_tree_required": True,
        "all_600_seed_identity_manifest_completion_and_trace_hashes_bound": True,
        "terminal_endpoint_hashes_bound_for_embedding_lineage": True,
        "pool_or_product_access_during_source_freeze": False,
    }


def expected_pool_access_audit() -> dict[str, bool]:
    return {
        "all_seed_manifests_completions_and_trace_payload_hashes_validated": True,
        "all_terminal_endpoint_byte_and_pixel_hashes_bound": True,
        "pool_exact_recursive_tree_validated": True,
        "labels_reviews_consensus_or_screen_results_opened": False,
        "preexisting_candidate_score_products_thresholds_or_alerts_opened": False,
        "score_label_join_performed": False,
    }


def expected_imported_helper_contract() -> dict[str, str]:
    return {
        "visual_imports_primary_validation": EXPECTED_PRIMARY_SOURCE_SHA256,
        "visual_imports_reproduction_validation": EXPECTED_REPRODUCTION_SOURCE_SHA256,
        "launcher_imports_sampling_pool_validator": EXPECTED_SAMPLING_RUNNER_SHA256,
    }


def expected_evidence_access_audit() -> dict[str, bool]:
    return {
        "sampling_source_protocol_including_phase1_binding_metadata_parsed": True,
        "linked_phase1_protocol_or_threshold_lock_files_opened": False,
        "endpoint_secondary_reference_files_read_bytewise_for_integrity_only": True,
        "endpoint_secondary_reference_arrays_or_rows_parsed": False,
        "sampling_pool_path_opened_statted_listed_or_hashed": False,
        "prospective_product_paths_opened_statted_listed_or_hashed": False,
        "labels_reviews_consensus_or_screen_results_opened": False,
        "preexisting_candidate_score_products_thresholds_or_alerts_opened": False,
        "score_label_join_performed": False,
        "source_protocol_locks_and_model_assets_validated": True,
    }


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def exclusive_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {path}")
    return path.resolve()


def require_hex64(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{description} must be a lowercase SHA-256")
    return value


def require_exact_path(path: Path, expected: Path, description: str) -> Path:
    """Compare lexical absolute paths before any filesystem access."""

    observed = path.expanduser().absolute()
    frozen = expected.expanduser().absolute()
    if observed != frozen:
        raise RuntimeError(f"{description} must equal frozen path {frozen}: {observed}")
    return frozen


def load_module(path: Path, name: str) -> ModuleType:
    path = require_regular(path, f"module source {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def exact_tree(
    root: Path, *, expected_files: set[str], expected_directories: set[str]
) -> None:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"tree contains a symlink: {path}")
        if path.is_file():
            files.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise RuntimeError(f"tree contains a special entry: {path}")
    if files != expected_files or directories != expected_directories:
        raise RuntimeError(
            f"tree changed: missing_files={sorted(expected_files-files)}, "
            f"extra_files={sorted(files-expected_files)}, "
            f"missing_dirs={sorted(expected_directories-directories)}, "
            f"extra_dirs={sorted(directories-expected_directories)}"
        )


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"artifact is a symlink: {path}")
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _expected_commands() -> dict[str, list[str]]:
    seed_csv = ",".join(str(value) for value in SEEDS)
    class_csv = ",".join(str(value) for value in CLASSES)
    checkpoint_csv = ",".join(str(value) for value in CHECKPOINTS)
    vae = (
        "/home/zhoushunyu/.cache/huggingface/hub/"
        "models--stabilityai--sd-vae-ft-mse/snapshots/"
        "31f26fdeee1355a5c34592e401dd41e45d25a493"
    )
    resnet = "/home/zhoushunyu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    inception = (
        "/home/zhoushunyu/.cache/torch/hub/checkpoints/"
        "pt_inception-2015-12-05-6726825d.pth"
    )
    dino = (
        "/home/zhoushunyu/.cache/huggingface/hub/"
        "models--facebook--dinov2-with-registers-large/snapshots/"
        "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
    )
    return {
        "primary": [
            sys.executable,
            str(ROOT / "experiments/analyze_dit_bad_good_custom_traces.py"),
            "--trace-root",
            str(POOL_ROOT),
            "--trace-glob",
            "third_pool_v1_seed*",
            "--expected-classes",
            class_csv,
            "--expected-seeds",
            seed_csv,
            "--output-dir",
            str(PRIMARY_OUTPUT),
        ],
        "visual": [
            sys.executable,
            str(ROOT / "experiments/extract_dit_predxstart_visual_tracks.py"),
            "--source-inventory",
            str(PRIMARY_OUTPUT / "source_inventory.json"),
            "--expected-seeds",
            seed_csv,
            "--expected-classes",
            class_csv,
            "--checkpoints",
            checkpoint_csv,
            "--vae-snapshot",
            vae,
            "--resnet18-weights",
            resnet,
            "--device",
            LOGICAL_DEVICE,
            "--decode-batch-size",
            str(VISUAL_DECODE_BATCH),
            "--classifier-batch-size",
            str(VISUAL_CLASSIFIER_BATCH),
            "--output-dir",
            str(VISUAL_OUTPUT),
        ],
        "endpoint": [
            sys.executable,
            str(ROOT / "experiments/extract_dit_endpoint_embeddings_label_free.py"),
            "--source-inventory",
            str(PRIMARY_OUTPUT / "source_inventory.json"),
            "--expected-seeds",
            "250:850",
            "--expected-classes",
            class_csv,
            "--inception-weights",
            inception,
            "--dino-snapshot",
            dino,
            "--device",
            LOGICAL_DEVICE,
            "--batch-size",
            str(ENDPOINT_BATCH),
            "--output-dir",
            str(ENDPOINT_OUTPUT),
        ],
    }


def _validate_small_lock(
    root: Path, *, artifact_kind: str, record_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, artifact_kind)
    exact_tree(
        root,
        expected_files={record_name, "manifest.json", "completion.json", "launcher_source.py"},
        expected_directories=set(),
    )
    record_path = require_regular(root / record_name, record_name)
    manifest_path = require_regular(root / "manifest.json", "binding manifest")
    completion_path = require_regular(root / "completion.json", "binding completion")
    source_path = require_regular(root / "launcher_source.py", "bound launcher source")
    record = load_json(record_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        canonical_sha256(without_identity(record)) != record.get("identity_sha256")
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("artifact_kind") != artifact_kind
        or manifest.get("status") != "complete"
        or manifest.get("primary_record_name") != record_name
        or manifest.get("primary_record_identity_sha256") != record.get("identity_sha256")
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("artifact_kind") != artifact_kind
        or completion.get("primary_record_identity_sha256") != record.get("identity_sha256")
        or completion.get("primary_record_file_sha256") != sha256_file(record_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or sha256_file(source_path) != record.get("implementation_source_sha256")
    ):
        raise RuntimeError(f"invalid immutable binding lock: {root}")
    return record, manifest


def _publish_small_lock(
    output: Path, *, artifact_kind: str, record_name: str, record: Mapping[str, Any]
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite binding lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / record_name, record)
        shutil.copy2(Path(__file__).resolve(), staging / "launcher_source.py")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": artifact_kind,
            "status": "complete",
            "primary_record_name": record_name,
            "primary_record_identity_sha256": record["identity_sha256"],
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "artifact_kind": artifact_kind,
                "primary_record_name": record_name,
                "primary_record_identity_sha256": record["identity_sha256"],
                "primary_record_file_sha256": sha256_file(staging / record_name),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            },
        )
        _validate_small_lock(staging, artifact_kind=artifact_kind, record_name=record_name)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def sampling_helper() -> ModuleType:
    path = ROOT / "experiments/run_dit_bad_good_third_pool.py"
    if sha256_file(path) != EXPECTED_SAMPLING_RUNNER_SHA256:
        raise RuntimeError("live sampling-pool validator differs from its SHA pin")
    return load_module(path, "_label_free_sampling_pool_validator")


def validate_sampling_source_lock_label_free(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate only the immutable sampling source lock, never its phase-1 links."""

    root = require_exact_path(root, SAMPLING_SOURCE_LOCK, "sampling source lock")
    root = require_directory(root, "sampling source lock")
    source_names = {
        "trace_dit_imagenet256_custom_batch.py",
        "sample_dit_imagenet256_custom.py",
        "reproduce_dit_imagenet256.py",
        "run_dit_bad_good_third_pool.py",
        "freeze_dit_bad_good_third_pool_sampling_sources.py",
    }
    exact_tree(
        root,
        expected_files={
            "sampling_protocol.json",
            "manifest.json",
            "completion.json",
            *(f"sources/{name}" for name in source_names),
        },
        expected_directories={"sources"},
    )
    protocol_path = require_regular(root / "sampling_protocol.json", "sampling protocol")
    manifest_path = require_regular(root / "manifest.json", "sampling source manifest")
    completion_path = require_regular(root / "completion.json", "sampling source completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        protocol.get("identity_sha256") != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or canonical_sha256(without_identity(protocol))
        != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or protocol.get("status") != "FROZEN_BEFORE_THIRD_POOL_GPU_SAMPLING"
        or manifest.get("identity_sha256") != EXPECTED_SAMPLING_MANIFEST_IDENTITY
        or canonical_sha256(without_identity(manifest))
        != EXPECTED_SAMPLING_MANIFEST_IDENTITY
        or manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256")
        != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("sampling_protocol_identity_sha256")
        != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or completion.get("sampling_protocol_file_sha256")
        != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256")
        != EXPECTED_SAMPLING_MANIFEST_IDENTITY
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("sampling source lock immutable envelope changed")
    source_records = protocol.get("source_snapshots")
    if not isinstance(source_records, dict) or set(source_records) != source_names:
        raise RuntimeError("sampling source snapshot family changed")
    for basename in sorted(source_names):
        snapshot = require_regular(root / "sources" / basename, f"sampling source {basename}")
        record = source_records[basename]
        if sha256_file(snapshot) != record.get("sha256"):
            raise RuntimeError(f"sampling source snapshot changed: {basename}")
    scientific = protocol.get("scientific_contract", {})
    if (
        scientific.get("classes_ordered") != list(CLASSES)
        or scientific.get("global_seeds") != list(SEEDS)
        or scientific.get("trajectory_count") != TRAJECTORY_COUNT
        or scientific.get("observation_only") is not True
        or scientific.get("quality_score") is not None
        or scientific.get("selection") is not None
        or scientific.get("intervention") is not None
    ):
        raise RuntimeError("sampling source lock cohort/observation contract changed")
    return protocol, manifest


def validate_endpoint_secondary_lock_label_free(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the endpoint protocol lock without parsing its frozen reference data."""

    root = require_exact_path(root, ENDPOINT_SECONDARY_LOCK, "endpoint secondary lock")
    root = require_directory(root, "endpoint secondary lock")
    source_names = {
        "audit_dit_endpoint_representation_distances.py",
        "evaluate_dit_bad_good_third_pool_confirmation.py",
        "evaluate_dit_third_pool_endpoint_secondary.py",
        "extract_dit_endpoint_embeddings_label_free.py",
        "freeze_dit_third_pool_endpoint_secondary.py",
    }
    exact_tree(
        root,
        expected_files={
            "protocol.json",
            "manifest.json",
            "completion.json",
            "reference_models.npz",
            "reference_summary.json",
            *(f"sources/{name}" for name in source_names),
        },
        expected_directories={"sources"},
    )
    protocol_path = require_regular(root / "protocol.json", "endpoint protocol")
    manifest_path = require_regular(root / "manifest.json", "endpoint protocol manifest")
    completion_path = require_regular(root / "completion.json", "endpoint protocol completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        protocol.get("identity_sha256") != EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or canonical_sha256(without_identity(protocol))
        != EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or manifest.get("identity_sha256") != EXPECTED_ENDPOINT_MANIFEST_IDENTITY
        or canonical_sha256(without_identity(manifest))
        != EXPECTED_ENDPOINT_MANIFEST_IDENTITY
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256")
        != EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256")
        != EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256")
        != EXPECTED_ENDPOINT_MANIFEST_IDENTITY
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("reference_models_file_sha256")
        != sha256_file(root / "reference_models.npz")
        or completion.get("reference_summary_file_sha256")
        != sha256_file(root / "reference_summary.json")
        or completion.get("third_pool_data_opened") is not False
    ):
        raise RuntimeError("endpoint secondary protocol lock immutable envelope changed")
    source_records = protocol.get("source_snapshots")
    if not isinstance(source_records, dict) or set(source_records) != source_names:
        raise RuntimeError("endpoint secondary source snapshot family changed")
    for basename in sorted(source_names):
        snapshot = require_regular(root / "sources" / basename, f"endpoint source {basename}")
        if sha256_file(snapshot) != source_records[basename].get("sha256"):
            raise RuntimeError(f"endpoint secondary source snapshot changed: {basename}")
    return protocol, manifest


def primary_helper() -> ModuleType:
    path = ROOT / "experiments/analyze_dit_bad_good_custom_traces.py"
    if sha256_file(path) != EXPECTED_PRIMARY_SOURCE_SHA256:
        raise RuntimeError("live primary extractor differs from its SHA pin")
    return load_module(path, "_third_pool_primary_label_free_helper")


def visual_helper() -> ModuleType:
    path = ROOT / "experiments/extract_dit_predxstart_visual_tracks.py"
    if sha256_file(path) != EXPECTED_VISUAL_SOURCE_SHA256:
        raise RuntimeError("live visual extractor differs from its SHA pin")
    return load_module(path, "_third_pool_visual_label_free_helper")


def endpoint_helper() -> ModuleType:
    path = ROOT / "experiments/extract_dit_endpoint_embeddings_label_free.py"
    if sha256_file(path) != EXPECTED_ENDPOINT_SOURCE_SHA256:
        raise RuntimeError("live endpoint extractor differs from its SHA pin")
    return load_module(path, "_third_pool_endpoint_label_free_helper")


def _validate_source_lock_contents(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "label-free product source lock")
    expected_files = {
        "product_protocol.json",
        "manifest.json",
        "completion.json",
        *(f"sources/{name}" for name in SOURCE_BASENAMES),
    }
    exact_tree(root, expected_files=expected_files, expected_directories={"sources"})
    protocol_path = require_regular(root / "product_protocol.json", "product protocol")
    manifest_path = require_regular(root / "manifest.json", "product source manifest")
    completion_path = require_regular(root / "completion.json", "product source completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        canonical_sha256(without_identity(protocol)) != protocol.get("identity_sha256")
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or protocol.get("schema_version") != 1
        or protocol.get("experiment")
        != "dit_bad_good_third_pool_label_free_products_v1"
        or protocol.get("status") != "FROZEN_BEFORE_POOL_SEAL_OR_PRODUCT_ACCESS"
        or manifest.get("schema_version") != 1
        or manifest.get("experiment")
        != "dit_bad_good_third_pool_label_free_products_source_lock_v1"
        or manifest.get("status") != "complete"
        or manifest.get("product_protocol_identity_sha256") != protocol.get("identity_sha256")
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("product_protocol_identity_sha256") != protocol.get("identity_sha256")
        or completion.get("product_protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("pool_or_product_access_performed") is not False
    ):
        raise RuntimeError("label-free product source lock envelope changed")

    cohort = protocol.get("cohort", {})
    foundations = protocol.get("foundation_identity_pins", {})
    outputs = protocol.get("outputs", {})
    execution = protocol.get("execution", {})
    if (
        tuple(cohort.get("classes_ordered", ())) != CLASSES
        or tuple(cohort.get("global_seeds", ())) != SEEDS
        or cohort.get("seed_count") != len(SEEDS)
        or cohort.get("trajectory_count") != TRAJECTORY_COUNT
        or foundations.get("sampling_protocol_identity_sha256")
        != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or foundations.get("sampling_manifest_identity_sha256")
        != EXPECTED_SAMPLING_MANIFEST_IDENTITY
        or foundations.get("sampling_runner_source_sha256")
        != EXPECTED_SAMPLING_RUNNER_SHA256
        or foundations.get("endpoint_secondary_protocol_identity_sha256")
        != EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or foundations.get("endpoint_secondary_manifest_identity_sha256")
        != EXPECTED_ENDPOINT_MANIFEST_IDENTITY
        or outputs
        != {
            "pool_root": str(POOL_ROOT),
            "primary": str(PRIMARY_OUTPUT),
            "visual": str(VISUAL_OUTPUT),
            "endpoint": str(ENDPOINT_OUTPUT),
            "completion_receipt": str(RECEIPT_OUTPUT),
            "pool_binding": str(DEFAULT_POOL_BINDING),
        }
        or protocol.get("products") != expected_products_contract()
        or protocol.get("role_boundaries") != expected_role_boundaries()
        or execution.get("python_executable") != sys.executable
        or execution.get("commands") != _expected_commands()
        or execution.get("dependency_order")
        != ["validate bound complete pool", "primary", "visual || endpoint", "receipt"]
        or execution.get("parallel_after_primary") != ["visual", "endpoint"]
        or execution.get("gpu_routing")
        != {
            "visual": {"physical_cuda_visible_devices": VISUAL_PHYSICAL_GPU, "logical_device": LOGICAL_DEVICE},
            "endpoint": {"physical_cuda_visible_devices": ENDPOINT_PHYSICAL_GPU, "logical_device": LOGICAL_DEVICE},
        }
        or execution.get("primary_is_completed_and_validated_before_children_start")
        is not True
    ):
        raise RuntimeError("label-free product scientific/execution contract changed")

    supervision = protocol.get("supervision_policy", {})
    resume = protocol.get("resume_and_overwrite", {})
    if (
        supervision != expected_supervision_contract()
        or resume != expected_resume_contract()
        or protocol.get("pool_binding_contract") != expected_pool_binding_contract()
        or protocol.get("imported_helper_sha256")
        != expected_imported_helper_contract()
        or protocol.get("evidence_access_audit") != expected_evidence_access_audit()
        or protocol.get("threat_model")
        != {
            "assumption": (
                "controlled static non-concurrently-rewritten local filesystem with "
                "Git or append-only chronology"
            ),
            "not_claimed": (
                "cryptographic authentication against malicious replacement and "
                "manual re-signing of a self-consistent artifact tree"
            ),
        }
    ):
        raise RuntimeError("label-free supervision/resume contract changed")

    source_records = protocol.get("source_snapshots", {})
    if set(source_records) != set(SOURCE_BASENAMES):
        raise RuntimeError("label-free source snapshot family changed")
    for basename in SOURCE_BASENAMES:
        path = require_regular(root / "sources" / basename, f"source {basename}")
        live = require_regular(ROOT / "experiments" / basename, f"live source {basename}")
        if (
            sha256_file(path) != source_records[basename].get("sha256")
            or path.stat().st_size != source_records[basename].get("bytes")
            or str(live) != source_records[basename].get("live_path_at_freeze")
            or sha256_file(live) != source_records[basename].get("sha256")
        ):
            raise RuntimeError(f"frozen product source changed: {basename}")
    known = {
        "run_dit_bad_good_third_pool.py": EXPECTED_SAMPLING_RUNNER_SHA256,
        "analyze_dit_bad_good_custom_traces.py": EXPECTED_PRIMARY_SOURCE_SHA256,
        "extract_dit_predxstart_visual_tracks.py": EXPECTED_VISUAL_SOURCE_SHA256,
        "extract_dit_endpoint_embeddings_label_free.py": EXPECTED_ENDPOINT_SOURCE_SHA256,
        "reproduce_dit_imagenet256.py": EXPECTED_REPRODUCTION_SOURCE_SHA256,
    }
    for basename, digest in known.items():
        if source_records[basename].get("sha256") != digest:
            raise RuntimeError(f"known helper pin changed: {basename}")
    invoked = Path(__file__).resolve()
    if (
        sha256_file(invoked) != source_records[invoked.name].get("sha256")
        or protocol.get("implementation_source_sha256")
        != source_records["run_dit_bad_good_third_pool_label_free_products.py"].get("sha256")
    ):
        raise RuntimeError("invoked product launcher differs from frozen source")

    sampling_protocol, sampling_manifest = validate_sampling_source_lock_label_free(
        SAMPLING_SOURCE_LOCK
    )
    if (
        sampling_protocol.get("identity_sha256") != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or sampling_manifest.get("identity_sha256") != EXPECTED_SAMPLING_MANIFEST_IDENTITY
        or foundations.get("sampling_source_lock_path") != str(SAMPLING_SOURCE_LOCK)
        or foundations.get("endpoint_secondary_lock_path")
        != str(ENDPOINT_SECONDARY_LOCK)
    ):
        raise RuntimeError("sampling source lock differs from product foundation")
    endpoint_protocol, endpoint_manifest = validate_endpoint_secondary_lock_label_free(
        ENDPOINT_SECONDARY_LOCK
    )
    if (
        endpoint_protocol.get("identity_sha256") != EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or endpoint_manifest.get("identity_sha256") != EXPECTED_ENDPOINT_MANIFEST_IDENTITY
    ):
        raise RuntimeError("endpoint secondary lock differs from product foundation")
    if protocol.get("assets", {}).get("vae") != sampling_protocol["assets"]["vae_snapshot"]:
        raise RuntimeError("product VAE asset differs from sampling-source foundation")
    _validate_assets(protocol.get("assets", {}))
    return protocol, manifest


def validate_source_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_exact_path(root, DEFAULT_SOURCE_LOCK, "product source lock")
    return _validate_source_lock_contents(root)


def _validate_bound_file(record: Mapping[str, Any], description: str) -> None:
    path = Path(str(record.get("path", ""))).expanduser().absolute()
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"{description} resolved asset is invalid: {resolved}")
    if (
        str(resolved) != record.get("resolved_path")
        or resolved.stat().st_size != record.get("bytes")
        or sha256_file(resolved) != record.get("sha256")
    ):
        raise RuntimeError(f"{description} asset changed")


def _validate_assets(assets: Mapping[str, Any]) -> None:
    if set(assets) != {"vae", "resnet18", "inception", "dinov2"}:
        raise RuntimeError("label-free asset family changed")
    vae = assets["vae"]
    dino = assets["dinov2"]
    expected_vae = (
        "/home/zhoushunyu/.cache/huggingface/hub/"
        "models--stabilityai--sd-vae-ft-mse/snapshots/"
        "31f26fdeee1355a5c34592e401dd41e45d25a493"
    )
    expected_resnet = (
        "/home/zhoushunyu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    )
    expected_inception = (
        "/home/zhoushunyu/.cache/torch/hub/checkpoints/"
        "pt_inception-2015-12-05-6726825d.pth"
    )
    expected_dino = (
        "/home/zhoushunyu/.cache/huggingface/hub/"
        "models--facebook--dinov2-with-registers-large/snapshots/"
        "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
    )
    if (
        vae.get("revision") != "31f26fdeee1355a5c34592e401dd41e45d25a493"
        or vae.get("snapshot") != expected_vae
        or dino.get("revision") != "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
        or dino.get("snapshot") != expected_dino
        or dino.get("model_id") != "facebook/dinov2-with-registers-large"
        or assets["resnet18"].get("path") != expected_resnet
        or assets["inception"].get("path") != expected_inception
        or len(vae.get("files", ())) != 2
        or len(dino.get("files", ())) != 3
        or assets["resnet18"].get("sha256") != RESNET18_SHA256
        or assets["inception"].get("sha256") != INCEPTION_SHA256
        or {item.get("name"): item.get("sha256") for item in vae.get("files", ())}
        != VAE_FILE_SHA256
        or {item.get("name"): item.get("sha256") for item in dino.get("files", ())}
        != DINO_FILE_SHA256
    ):
        raise RuntimeError("VAE/DINO asset contract changed")
    _validate_bound_file(assets["resnet18"], "ResNet-18")
    _validate_bound_file(assets["inception"], "Inception")
    for item in vae["files"]:
        _validate_bound_file(item, f"VAE {item.get('name')}")
    for item in dino["files"]:
        _validate_bound_file(item, f"DINO {item.get('name')}")


def _validate_pool_tree(pool: Path, sampling: ModuleType) -> None:
    manifest = load_json(pool / "pool_manifest.json")
    logs = manifest.get("runner_logs")
    if not isinstance(logs, list):
        raise RuntimeError("pool runner log records are missing")
    expected_directories = {"_runner_logs"}
    expected_files = {"execution_plan.json", "pool_manifest.json", "pool_completion.json"}
    for item in logs:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError("unsafe pool runner log member")
        expected_files.add(f"_runner_logs/{name}")
    for seed in SEEDS:
        base = f"third_pool_v1_seed{seed:03d}"
        expected_directories.update({base, f"{base}/images"})
        expected_files.update({f"{base}/manifest.json", f"{base}/completion.json"})
        expected_files.update(
            f"{base}/{relative}" for relative in sampling.EXPECTED_OUTPUT_RELATIVE_PATHS
        )
    exact_tree(
        pool,
        expected_files=expected_files,
        expected_directories=expected_directories,
    )


def validate_completed_pool() -> dict[str, Any]:
    sampling = sampling_helper()
    sampling_protocol, sampling_manifest = validate_sampling_source_lock_label_free(
        SAMPLING_SOURCE_LOCK
    )
    pool = require_directory(POOL_ROOT, "completed third sampling pool")
    plan = load_json(pool / "execution_plan.json")
    gpus = tuple(plan.get("gpus_ordered", ()))
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(isinstance(x, str) for x in gpus):
        raise RuntimeError("completed pool GPU execution plan changed")
    frozen_assets = sampling_protocol["assets"]
    expected_plan = sampling.build_plan(
        sampling_protocol,
        sampling_manifest,
        SAMPLING_SOURCE_LOCK,
        gpus,
        POOL_ROOT,
        Path(frozen_assets["dit_repository"]["root"]),
        Path(frozen_assets["checkpoint"]["path"]),
        Path(frozen_assets["vae_snapshot"]["snapshot"]),
    )
    if plan != expected_plan:
        raise RuntimeError("completed pool execution plan does not replay frozen source")
    _validate_pool_tree(pool, sampling)
    sampling.validate_complete_pool(pool, expected_plan, sampling_protocol, SAMPLING_SOURCE_LOCK)
    pool_manifest_path = require_regular(pool / "pool_manifest.json", "pool manifest")
    pool_completion_path = require_regular(pool / "pool_completion.json", "pool completion")
    pool_manifest = load_json(pool_manifest_path)
    pool_completion = load_json(pool_completion_path)
    seed_outputs = pool_manifest.get("seed_outputs")
    if (
        pool_manifest.get("identity_sha256") != pool_completion.get("pool_identity_sha256")
        or not isinstance(seed_outputs, list)
        or len(seed_outputs) != len(SEEDS)
        or [item.get("seed") for item in seed_outputs] != list(SEEDS)
    ):
        raise RuntimeError("completed pool identity/seed receipt family changed")
    endpoint_outputs: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = pool / f"third_pool_v1_seed{seed:03d}"
        seed_manifest = load_json(seed_root / "manifest.json")
        outputs = seed_manifest.get("outputs")
        if not isinstance(outputs, list):
            raise RuntimeError(f"seed {seed} endpoint lineage is missing")
        by_relative = {
            item.get("relative_path"): item for item in outputs if isinstance(item, dict)
        }
        for class_slot, class_id in enumerate(CLASSES):
            relative = f"images/{class_slot:02d}_class{class_id:04d}.png"
            item = by_relative.get(relative)
            if (
                not isinstance(item, dict)
                or item.get("mode") != "RGB"
                or item.get("size") != [256, 256]
                or require_hex64(item.get("sha256"), "endpoint byte hash")
                != item.get("sha256")
                or require_hex64(item.get("pixel_sha256"), "endpoint pixel hash")
                != item.get("pixel_sha256")
                or not isinstance(item.get("bytes"), int)
                or item["bytes"] <= 0
            ):
                raise RuntimeError(f"seed {seed} class {class_id} endpoint lineage changed")
            endpoint_outputs.append(
                {
                    "sample_index": len(endpoint_outputs),
                    "global_seed": seed,
                    "class_slot": class_slot,
                    "class_id": class_id,
                    "relative_path": relative,
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "pixel_sha256": item["pixel_sha256"],
                }
            )
    if len(endpoint_outputs) != TRAJECTORY_COUNT:
        raise RuntimeError("completed pool terminal endpoint axis is not exact")
    return {
        "path": str(pool),
        "pool_identity_sha256": pool_manifest["identity_sha256"],
        "pool_manifest_file_sha256": sha256_file(pool_manifest_path),
        "pool_completion_file_sha256": sha256_file(pool_completion_path),
        "execution_plan_file_sha256": sha256_file(pool / "execution_plan.json"),
        "sampling_protocol_identity_sha256": sampling_protocol["identity_sha256"],
        "sampling_manifest_identity_sha256": sampling_manifest["identity_sha256"],
        "seed_outputs": seed_outputs,
        "seed_lineage_identity_sha256": canonical_sha256(seed_outputs),
        "terminal_endpoint_outputs": endpoint_outputs,
        "terminal_endpoint_lineage_identity_sha256": canonical_sha256(endpoint_outputs),
        "seed_count": len(SEEDS),
        "trajectory_count": TRAJECTORY_COUNT,
    }


def bind_pool(source_lock: Path, output: Path) -> Path:
    source_lock = require_exact_path(source_lock, DEFAULT_SOURCE_LOCK, "product source lock")
    output = require_exact_path(output, DEFAULT_POOL_BINDING, "pool binding output")
    if os.path.lexists(output):
        validate_pool_binding(output, source_lock)
        return output
    protocol, source_manifest = validate_source_lock(source_lock)
    lineage = validate_completed_pool()
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "BOUND_ONLY_AFTER_FULL_COMPLETE_POOL_VALIDATION",
        "product_protocol_identity_sha256": protocol["identity_sha256"],
        "product_source_manifest_identity_sha256": source_manifest["identity_sha256"],
        "pool": lineage,
        "access_audit": expected_pool_access_audit(),
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    record["identity_sha256"] = canonical_sha256(record)
    return _publish_small_lock(
        output,
        artifact_kind="dit_bad_good_third_pool_label_free_pool_binding_v1",
        record_name="pool_binding.json",
        record=record,
    )


def validate_pool_binding(
    root: Path, source_lock: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_exact_path(root, DEFAULT_POOL_BINDING, "pool binding lock")
    source_lock = require_exact_path(source_lock, DEFAULT_SOURCE_LOCK, "product source lock")
    record, manifest = _validate_small_lock(
        root,
        artifact_kind="dit_bad_good_third_pool_label_free_pool_binding_v1",
        record_name="pool_binding.json",
    )
    protocol, source_manifest = validate_source_lock(source_lock)
    if (
        record.get("status") != "BOUND_ONLY_AFTER_FULL_COMPLETE_POOL_VALIDATION"
        or record.get("product_protocol_identity_sha256") != protocol["identity_sha256"]
        or record.get("product_source_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or record.get("implementation_source_sha256")
        != protocol["implementation_source_sha256"]
        or record.get("access_audit") != expected_pool_access_audit()
    ):
        raise RuntimeError("pool binding scientific/access contract changed")
    observed = validate_completed_pool()
    if observed != record.get("pool"):
        raise RuntimeError("completed sampling pool differs from immutable binding")
    return record, manifest


def _pool_by_seed(binding: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = binding["pool"]["seed_outputs"]
    return {int(item["seed"]): item for item in rows}


def _validate_trace_inventory(
    inventory: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    runs = inventory.get("trace_runs")
    if (
        inventory.get("ordered_classes") != list(CLASSES)
        or inventory.get("ordered_seeds") != list(SEEDS)
        or inventory.get("locked_consensus") is not None
        or not isinstance(runs, list)
        or len(runs) != len(SEEDS)
    ):
        raise RuntimeError("primary label-free source inventory axis/supervision changed")
    pool_by_seed = _pool_by_seed(binding)
    bound_pool_root = Path(str(binding["pool"]["path"]))
    for seed, run in zip(SEEDS, runs, strict=True):
        expected = pool_by_seed[seed]
        root = bound_pool_root / f"third_pool_v1_seed{seed:03d}"
        if (
            run.get("global_seed") != seed
            or run.get("classes") != list(CLASSES)
            or str(Path(str(run.get("root", ""))).absolute()) != str(root)
            or run.get("identity_sha256") != expected["identity_sha256"]
            or run.get("manifest_sha256") != expected["manifest_sha256"]
            or run.get("completion_sha256") != expected["completion_sha256"]
            or run.get("trace_sha256") != expected["trace_npz_sha256"]
        ):
            raise RuntimeError(f"label-free source inventory differs from pool at seed {seed}")


def _load_manifest_identity(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = require_regular(root / "manifest.json", "product manifest")
    completion_path = require_regular(root / "completion.json", "product completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    require_hex64(manifest.get("identity_sha256"), "product manifest identity")
    return manifest, completion


def _require_catalog_feature(path: Path, feature: str, latest_step: int) -> None:
    path = require_regular(path, "label-free feature catalog")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"feature catalog header is malformed: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise RuntimeError(f"feature catalog contains extra cells: {path}")
    matches = [row for row in rows if row.get("feature") == feature]
    if len(matches) != 1 or int(matches[0].get("latest_required_sampling_step", -1)) != latest_step:
        raise RuntimeError(f"required frozen feature/timing is absent: {feature}")


def validate_primary_product(binding: Mapping[str, Any]) -> dict[str, Any]:
    root = require_directory(PRIMARY_OUTPUT, "primary label-free product")
    helper = primary_helper()
    helper.validate_analysis_output(root)
    manifest, completion = _load_manifest_identity(root)
    summary = load_json(root / "summary.json")
    inventory = load_json(root / "source_inventory.json")
    if (
        manifest.get("analysis_source_sha256") != EXPECTED_PRIMARY_SOURCE_SHA256
        or manifest.get("experiment")
        != "dit_bad_good_custom_trace_metric_discovery"
        or manifest.get("status") != "complete"
        or summary.get("status") != "DISCOVERY_ONLY_NOT_AN_INTERVENTION_TRIGGER"
        or summary.get("sample_count") != TRAJECTORY_COUNT
        or summary.get("run_count") != len(SEEDS)
        or summary.get("ordered_classes") != list(CLASSES)
        or summary.get("ordered_seeds") != list(SEEDS)
        or summary.get("labels_joined") is not False
        or summary.get("label_counts") != {"unlabeled": TRAJECTORY_COUNT}
        or summary.get("univariate_result_count") != 0
        or inventory.get("analysis_source", {}).get("sha256")
        != EXPECTED_PRIMARY_SOURCE_SHA256
        or manifest.get("trace_identity_sha256_ordered")
        != [item["identity_sha256"] for item in binding["pool"]["seed_outputs"]]
    ):
        raise RuntimeError("primary label-free product scientific contract changed")
    _validate_trace_inventory(inventory, binding)
    _validate_product_sample_axis(root / "sample_features.csv")
    _require_catalog_feature(root / "feature_catalog.csv", PRIMARY_FEATURE, 149)
    return {
        "path": str(root),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "completion_file_sha256": sha256_file(root / "completion.json"),
        "source_inventory_file_sha256": sha256_file(root / "source_inventory.json"),
        "sample_features_file_sha256": sha256_file(root / "sample_features.csv"),
        "analysis_source_sha256": EXPECTED_PRIMARY_SOURCE_SHA256,
        "sample_count": TRAJECTORY_COUNT,
        "completion_payload_sha256": completion.get("payload_sha256"),
        "scientific_fingerprint_sha256": inventory.get(
            "scientific_fingerprint_sha256"
        ),
    }


def _validate_product_sample_axis(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_index", "global_seed", "class_slot", "class_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"product sample axis columns changed: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise RuntimeError(f"product sample axis contains extra CSV cells: {path}")
    if len(rows) != TRAJECTORY_COUNT:
        raise RuntimeError(f"product sample count changed: {path}")
    for index, row in enumerate(rows):
        seed = SEEDS[index // len(CLASSES)]
        slot = index % len(CLASSES)
        if (
            int(row["sample_index"]) != index
            or int(row["global_seed"]) != seed
            or int(row["class_slot"]) != slot
            or int(row["class_id"]) != CLASSES[slot]
        ):
            raise RuntimeError(f"product sample order changed at row {index}: {path}")


def _validate_primary_inventory_binding(
    item: Mapping[str, Any], primary: Mapping[str, Any]
) -> None:
    manifest_sha = item.get("manifest_sha256")
    if manifest_sha is None:
        manifest_sha = item.get("manifest_file_sha256")
    if (
        str(Path(str(item.get("path", ""))).absolute())
        != str(Path(primary["path"]) / "source_inventory.json")
        or item.get("sha256") != primary["source_inventory_file_sha256"]
        or manifest_sha != primary["manifest_file_sha256"]
        or item.get("completion_sha256") != primary["completion_file_sha256"]
        or item.get("trace_run_count") != len(SEEDS)
    ):
        raise RuntimeError("downstream product is not bound to exact primary inventory")


def validate_visual_product(
    binding: Mapping[str, Any], primary: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    root = require_directory(VISUAL_OUTPUT, "visual label-free product")
    helper = visual_helper()
    helper._validate_output(root)
    manifest, completion = _load_manifest_identity(root)
    members = manifest.get("files")
    if not isinstance(members, list) or not all(isinstance(item, dict) for item in members):
        raise RuntimeError("visual label-free manifest member list is malformed")
    member_names = [item.get("name") for item in members]
    if (
        any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            for name in member_names
        )
        or len(member_names) != len(set(member_names))
    ):
        raise RuntimeError("visual label-free manifest member names are unsafe/repeated")
    exact_tree(
        root,
        expected_files={"manifest.json", "completion.json", *member_names},
        expected_directories=set(),
    )
    summary = load_json(root / "summary.json")
    inventory = load_json(root / "source_inventory.json")
    provenance = load_json(root / "provenance.json")
    sources = inventory.get("input_label_free_source_inventories")
    if (
        manifest.get("analysis_source_sha256") != EXPECTED_VISUAL_SOURCE_SHA256
        or manifest.get("experiment")
        != "dit_predxstart_preterminal_visual_tracks_label_free"
        or manifest.get("status") != "complete"
        or summary.get("status") != "COMPLETE_LABEL_FREE_VISUAL_TRACK_EXTRACTION"
        or summary.get("trace_count") != len(SEEDS)
        or summary.get("sample_count") != TRAJECTORY_COUNT
        or summary.get("ordered_classes") != list(CLASSES)
        or summary.get("ordered_seeds") != list(SEEDS)
        or summary.get("selected_sampling_steps") != list(CHECKPOINTS)
        or summary.get("selected_internal_timesteps") != [249 - value for value in CHECKPOINTS]
        or summary.get("decoded_images_saved") is not False
        or summary.get("labels_read_or_emitted") is not False
        or inventory.get("ordered_classes") != list(CLASSES)
        or inventory.get("ordered_seeds") != list(SEEDS)
        or inventory.get("analysis_source", {}).get("sha256")
        != EXPECTED_VISUAL_SOURCE_SHA256
        or inventory.get("imported_validation_helper", {}).get("sha256")
        != EXPECTED_PRIMARY_SOURCE_SHA256
        or not isinstance(sources, list)
        or len(sources) != 1
        or provenance.get("device") != LOGICAL_DEVICE
        or provenance.get("selected_sampling_steps") != list(CHECKPOINTS)
        or provenance.get("decoded_images_saved") is not False
        or provenance.get("vae") != protocol["assets"]["vae"]
        or provenance.get("resnet18")
        != {
            "path": protocol["assets"]["resnet18"]["resolved_path"],
            "bytes": protocol["assets"]["resnet18"]["bytes"],
            "sha256": protocol["assets"]["resnet18"]["sha256"],
        }
        or provenance.get("supervision_audit")
        != helper.PROTOCOL["supervision_policy"]
        or load_json(root / "protocol_snapshot.json") != helper.PROTOCOL
    ):
        raise RuntimeError("visual label-free product scientific contract changed")
    _validate_primary_inventory_binding(sources[0], primary)
    _validate_trace_inventory(inventory, binding)
    _validate_product_sample_axis(root / "sample_features.csv")
    _require_catalog_feature(root / "feature_catalog.csv", VISUAL_FEATURE, 149)
    return {
        "path": str(root),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "completion_file_sha256": sha256_file(root / "completion.json"),
        "source_inventory_file_sha256": sha256_file(root / "source_inventory.json"),
        "sample_features_file_sha256": sha256_file(root / "sample_features.csv"),
        "analysis_source_sha256": EXPECTED_VISUAL_SOURCE_SHA256,
        "sample_count": TRAJECTORY_COUNT,
        "completion_payload_sha256": completion.get("payload_sha256"),
    }


def _validate_endpoint_sample_lineage(
    path: Path, binding: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    expected_fields = (
        "sample_index",
        "global_seed",
        "class_slot",
        "class_id",
        "trace_root",
        "trace_identity_sha256",
        "endpoint_png_path",
        "endpoint_sha256",
        "endpoint_pixel_sha256",
    )
    with require_regular(path, "endpoint sample index").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError("endpoint sample-index schema changed")
        rows = list(reader)
    endpoints = binding["pool"].get("terminal_endpoint_outputs")
    if not isinstance(endpoints, list) or len(endpoints) != TRAJECTORY_COUNT:
        raise RuntimeError("pool binding lacks exact terminal endpoint lineage")
    seed_outputs = _pool_by_seed(binding)
    bound_pool_root = Path(str(binding["pool"]["path"]))
    endpoint_hashes: list[str] = []
    trace_identities: list[str] = []
    for index, (row, expected) in enumerate(zip(rows, endpoints, strict=True)):
        seed = int(expected["global_seed"])
        trace_root = bound_pool_root / f"third_pool_v1_seed{seed:03d}"
        endpoint_path = trace_root / expected["relative_path"]
        trace_identity = seed_outputs[seed]["identity_sha256"]
        if (
            None in row
            or int(row["sample_index"]) != index
            or int(row["global_seed"]) != seed
            or int(row["class_slot"]) != expected["class_slot"]
            or int(row["class_id"]) != expected["class_id"]
            or str(Path(row["trace_root"]).absolute()) != str(trace_root)
            or row["trace_identity_sha256"] != trace_identity
            or str(Path(row["endpoint_png_path"]).absolute()) != str(endpoint_path)
            or row["endpoint_sha256"] != expected["sha256"]
            or row["endpoint_pixel_sha256"] != expected["pixel_sha256"]
        ):
            raise RuntimeError(f"endpoint product differs from sealed pool at row {index}")
        endpoint_hashes.append(row["endpoint_sha256"])
        trace_identities.append(trace_identity)
    return endpoint_hashes, list(dict.fromkeys(trace_identities))


def validate_endpoint_product(
    binding: Mapping[str, Any], primary: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    root = require_directory(ENDPOINT_OUTPUT, "endpoint label-free product")
    helper = endpoint_helper()
    receipt = helper.validate_output(root)
    manifest, completion = _load_manifest_identity(root)
    summary = load_json(root / "summary.json")
    inventory = load_json(root / "source_inventory.json")
    provenance = load_json(root / "provenance.json")
    sources = inventory.get("input_label_free_source_analyses")
    if (
        manifest.get("analysis_source_sha256") != EXPECTED_ENDPOINT_SOURCE_SHA256
        or summary.get("status") != "COMPLETE_LABEL_FREE_ENDPOINT_EMBEDDINGS"
        or summary.get("sample_count") != TRAJECTORY_COUNT
        or summary.get("seed_count") != len(SEEDS)
        or summary.get("ordered_classes") != list(CLASSES)
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("distances_or_scores_computed") is not False
        or summary.get("images_saved") is not False
        or summary.get("preterminal_actionable") is not False
        or inventory.get("ordered_classes") != list(CLASSES)
        or inventory.get("ordered_seeds") != list(SEEDS)
        or not isinstance(sources, list)
        or len(sources) != 1
        or provenance.get("supervision_audit", {}).get("labels_read_or_emitted") is not False
        or provenance.get("supervision_audit")
        != helper.PROTOCOL["supervision_policy"]
        or provenance.get("models")
        != {
            "inception": {
                "path": protocol["assets"]["inception"]["resolved_path"],
                "bytes": protocol["assets"]["inception"]["bytes"],
                "sha256": protocol["assets"]["inception"]["sha256"],
            },
            "dinov2": protocol["assets"]["dinov2"],
        }
        or provenance.get("runtime", {}).get("device") != LOGICAL_DEVICE
        or provenance.get("runtime", {}).get("dtype") != "float32"
        or provenance.get("runtime", {}).get("models_loaded_once") is not True
        or provenance.get("runtime", {}).get("offline_only") is not True
        or provenance.get("runtime", {}).get("deterministic_algorithms") is not True
        or provenance.get("runtime", {}).get("tf32") is not False
        or load_json(root / "protocol_snapshot.json") != helper.PROTOCOL
        or receipt.get("sample_count") != TRAJECTORY_COUNT
    ):
        raise RuntimeError("endpoint label-free product scientific contract changed")
    _validate_primary_inventory_binding(sources[0], primary)
    _validate_product_sample_axis(root / "sample_index.csv")
    endpoint_hashes, trace_identities = _validate_endpoint_sample_lineage(
        root / "sample_index.csv", binding
    )
    if (
        inventory.get("ordered_endpoint_sha256") != endpoint_hashes
        or inventory.get("ordered_trace_identity_sha256") != trace_identities
        or inventory.get("sampler_scientific_fingerprint_sha256")
        != primary["scientific_fingerprint_sha256"]
    ):
        raise RuntimeError("endpoint product source lineage differs from primary/pool")
    return {
        "path": str(root),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "completion_file_sha256": sha256_file(root / "completion.json"),
        "source_inventory_file_sha256": sha256_file(root / "source_inventory.json"),
        "embeddings_file_sha256": sha256_file(root / "embeddings.npz"),
        "sample_index_file_sha256": sha256_file(root / "sample_index.csv"),
        "analysis_source_sha256": EXPECTED_ENDPOINT_SOURCE_SHA256,
        "sample_count": TRAJECTORY_COUNT,
        "completion_payload_sha256": completion.get("payload_sha256"),
    }


def _validate_or_absent(
    path: Path,
    validator: Callable[[], dict[str, Any]],
    *,
    _exists: Callable[[Path], bool] = os.path.lexists,
) -> tuple[bool, dict[str, Any] | None]:
    if not _exists(path):
        return False, None
    return True, validator()


def _root_members_allowed() -> None:
    if not PRODUCT_ROOT.exists():
        return
    root = require_directory(PRODUCT_ROOT, "label-free product root")
    allowed = {
        "label_free_execution_plan.json",
        "_launcher_logs",
        PRIMARY_OUTPUT.name,
        VISUAL_OUTPUT.name,
        ENDPOINT_OUTPUT.name,
        RECEIPT_OUTPUT.name,
    }
    observed = {path.name for path in root.iterdir()}
    extra = observed - allowed
    if extra:
        raise RuntimeError(f"product root contains unexpected partial/evidence paths: {sorted(extra)}")
    for path in root.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"product root contains a symlink: {path}")
        if path.name == "label_free_execution_plan.json":
            if not path.is_file():
                raise RuntimeError("label-free execution plan is not a regular file")
        elif not path.is_dir():
            raise RuntimeError(f"product child must be a real directory: {path}")


def _next_log(name: str) -> Path:
    logs = PRODUCT_ROOT / "_launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if logs.is_symlink():
        raise RuntimeError("launcher log directory is a symlink")
    for attempt in range(1, 10_000):
        path = logs / f"{name}_attempt{attempt:04d}.log"
        if not os.path.lexists(path):
            return path
    raise RuntimeError(f"too many launcher attempts for {name}")


def _run_command(name: str, command: Sequence[str], physical_gpu: str | None) -> None:
    log_path = _next_log(name)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    if physical_gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = physical_gpu
    started = time.time()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "name": name,
                    "started_unix": started,
                    "command": list(command),
                    "physical_cuda_visible_devices": physical_gpu,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        finished = time.time()
        log.write(
            json.dumps(
                {
                    "finished_unix": finished,
                    "elapsed_seconds": finished - started,
                    "returncode": completed.returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed closed; preserve and inspect {log_path}")


def _log_records() -> list[dict[str, Any]]:
    logs = require_directory(PRODUCT_ROOT / "_launcher_logs", "launcher logs")
    records = []
    for path in sorted(logs.iterdir()):
        if not path.is_file() or path.is_symlink() or path.suffix != ".log":
            raise RuntimeError(f"unexpected launcher log entry: {path}")
        records.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def build_execution_plan(
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
    binding_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_LABEL_FREE_PRODUCT_EXECUTION_PLAN",
        "product_protocol_identity_sha256": protocol["identity_sha256"],
        "pool_binding_identity_sha256": binding["identity_sha256"],
        "pool_binding_manifest_identity_sha256": binding_manifest["identity_sha256"],
        "pool_identity_sha256": binding["pool"]["pool_identity_sha256"],
        "commands": _expected_commands(),
        "dependency_order": ["primary", "visual || endpoint", "receipt"],
        "gpu_routing": {
            "visual": VISUAL_PHYSICAL_GPU,
            "endpoint": ENDPOINT_PHYSICAL_GPU,
        },
        "outputs": {
            "primary": str(PRIMARY_OUTPUT),
            "visual": str(VISUAL_OUTPUT),
            "endpoint": str(ENDPOINT_OUTPUT),
            "receipt": str(RECEIPT_OUTPUT),
        },
        "labels_reviews_scores_opened": False,
        "preexisting_score_products_opened": False,
        "generated_products_read_only_for_full_validation": True,
        "selection_auc_or_thresholding_performed": False,
        "overwrite_allowed": False,
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }


def _publish_receipt(
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
    binding_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    products: Mapping[str, Any],
) -> Path:
    if os.path.lexists(RECEIPT_OUTPUT):
        raise RuntimeError(f"refusing to overwrite product receipt: {RECEIPT_OUTPUT}")
    staging = Path(tempfile.mkdtemp(prefix=f".{RECEIPT_OUTPUT.name}.tmp-", dir=PRODUCT_ROOT))
    try:
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE_THREE_LABEL_FREE_PRODUCTS",
            "product_protocol_identity_sha256": protocol["identity_sha256"],
            "pool_binding_identity_sha256": binding["identity_sha256"],
            "pool_binding_manifest_identity_sha256": binding_manifest["identity_sha256"],
            "pool_identity_sha256": binding["pool"]["pool_identity_sha256"],
            "execution_plan_file_sha256": sha256_file(
                PRODUCT_ROOT / "label_free_execution_plan.json"
            ),
            "products": dict(products),
            "launcher_logs": _log_records(),
            "execution": {
                "primary_completed_before_parallel_children": True,
                "visual_and_endpoint_allowed_to_run_in_parallel": True,
                "existing_products_reused_only_after_full_validation": True,
                "overwrite_performed": False,
            },
            "supervision_audit": {
                "labels_reviews_consensus_or_screen_results_opened": False,
                "preexisting_candidate_score_products_opened": False,
                "generated_raw_label_free_feature_values_read_for_validation": True,
                "calibration_thresholds_or_alerts_opened": False,
                "score_label_join_performed": False,
                "raw_label_free_feature_products_generated": True,
                "selection_or_auc_computed": False,
            },
            "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        result["identity_sha256"] = canonical_sha256(result)
        write_json(staging / "products.json", result)
        shutil.copy2(Path(__file__).resolve(), staging / "launcher_source.py")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": "dit_bad_good_third_pool_label_free_products_v1",
            "products_identity_sha256": result["identity_sha256"],
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "products_identity_sha256": result["identity_sha256"],
                "products_file_sha256": sha256_file(staging / "products.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            },
        )
        os.replace(staging, RECEIPT_OUTPUT)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RECEIPT_OUTPUT


def validate_receipt(
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
    binding_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    root = require_directory(RECEIPT_OUTPUT, "label-free product receipt")
    exact_tree(
        root,
        expected_files={
            "products.json",
            "launcher_source.py",
            "manifest.json",
            "completion.json",
        },
        expected_directories=set(),
    )
    result = load_json(root / "products.json")
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    products = {
        "primary": validate_primary_product(binding),
    }
    products["visual"] = validate_visual_product(
        binding, products["primary"], protocol
    )
    products["endpoint"] = validate_endpoint_product(
        binding, products["primary"], protocol
    )
    if (
        canonical_sha256(without_identity(result)) != result.get("identity_sha256")
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("files") != artifact_records(root)
        or manifest.get("status") != "complete"
        or manifest.get("experiment")
        != "dit_bad_good_third_pool_label_free_products_v1"
        or manifest.get("products_identity_sha256") != result.get("identity_sha256")
        or result.get("status") != "COMPLETE_THREE_LABEL_FREE_PRODUCTS"
        or result.get("product_protocol_identity_sha256") != protocol["identity_sha256"]
        or result.get("pool_binding_identity_sha256") != binding["identity_sha256"]
        or result.get("pool_binding_manifest_identity_sha256")
        != binding_manifest["identity_sha256"]
        or result.get("pool_identity_sha256")
        != binding["pool"]["pool_identity_sha256"]
        or result.get("execution_plan_file_sha256")
        != sha256_file(PRODUCT_ROOT / "label_free_execution_plan.json")
        or result.get("products") != products
        or result.get("launcher_logs") != _log_records()
        or result.get("execution")
        != {
            "primary_completed_before_parallel_children": True,
            "visual_and_endpoint_allowed_to_run_in_parallel": True,
            "existing_products_reused_only_after_full_validation": True,
            "overwrite_performed": False,
        }
        or result.get("supervision_audit")
        != {
            "labels_reviews_consensus_or_screen_results_opened": False,
            "preexisting_candidate_score_products_opened": False,
            "generated_raw_label_free_feature_values_read_for_validation": True,
            "calibration_thresholds_or_alerts_opened": False,
            "score_label_join_performed": False,
            "raw_label_free_feature_products_generated": True,
            "selection_or_auc_computed": False,
        }
        or result.get("implementation_source_sha256")
        != protocol["implementation_source_sha256"]
        or sha256_file(root / "launcher_source.py")
        != protocol["implementation_source_sha256"]
        or completion.get("complete") is not True
        or completion.get("products_identity_sha256") != result.get("identity_sha256")
        or completion.get("products_file_sha256") != sha256_file(root / "products.json")
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or load_json(PRODUCT_ROOT / "label_free_execution_plan.json") != plan
    ):
        raise RuntimeError("completed label-free product receipt failed validation")
    return result


def launch(source_lock: Path, pool_binding: Path) -> dict[str, Any]:
    protocol, _ = validate_source_lock(source_lock)
    binding, binding_manifest = validate_pool_binding(pool_binding, source_lock)
    plan = build_execution_plan(protocol, binding, binding_manifest)
    if not PRODUCT_ROOT.exists():
        PRODUCT_ROOT.mkdir(parents=True, exist_ok=False)
    _root_members_allowed()
    plan_path = PRODUCT_ROOT / "label_free_execution_plan.json"
    if plan_path.exists():
        if plan_path.is_symlink() or load_json(plan_path) != plan:
            raise RuntimeError("existing label-free execution plan changed")
    else:
        if any(PRODUCT_ROOT.iterdir()):
            raise RuntimeError("nonempty product root lacks the frozen execution plan")
        exclusive_json(plan_path, plan)
    (PRODUCT_ROOT / "_launcher_logs").mkdir(exist_ok=True)
    _root_members_allowed()

    if os.path.lexists(RECEIPT_OUTPUT):
        return validate_receipt(protocol, binding, binding_manifest, plan)

    commands = protocol["execution"]["commands"]
    primary_exists, primary = _validate_or_absent(
        PRIMARY_OUTPUT, lambda: validate_primary_product(binding)
    )
    if not primary_exists:
        _run_command("primary", commands["primary"], None)
        primary = validate_primary_product(binding)
    assert primary is not None

    validators: dict[str, tuple[Path, Callable[[], dict[str, Any]], str]] = {
        "visual": (
            VISUAL_OUTPUT,
            lambda: validate_visual_product(binding, primary, protocol),
            VISUAL_PHYSICAL_GPU,
        ),
        "endpoint": (
            ENDPOINT_OUTPUT,
            lambda: validate_endpoint_product(binding, primary, protocol),
            ENDPOINT_PHYSICAL_GPU,
        ),
    }
    products: dict[str, Any] = {"primary": primary}
    pending: dict[str, tuple[Callable[[], dict[str, Any]], str]] = {}
    for name, (path, validator, gpu) in validators.items():
        exists, observed = _validate_or_absent(path, validator)
        if exists:
            products[name] = observed
        else:
            pending[name] = (validator, gpu)
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as executor:
            futures = {
                executor.submit(_run_command, name, commands[name], gpu): name
                for name, (_validator, gpu) in pending.items()
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
        for name, (validator, _gpu) in pending.items():
            products[name] = validator()
    if set(products) != {"primary", "visual", "endpoint"}:
        raise RuntimeError("three-product family is incomplete")
    _publish_receipt(protocol, binding, binding_manifest, plan, products)
    return validate_receipt(protocol, binding, binding_manifest, plan)


def synthetic_self_test() -> None:
    commands = _expected_commands()
    if set(commands) != {"primary", "visual", "endpoint"}:
        raise AssertionError("three-command family changed")
    flattened = [token.lower() for command in commands.values() for token in command]
    for forbidden in ("--consensus", "--label", "--review", "--threshold", "--score"):
        if forbidden in flattened:
            raise AssertionError(f"supervised option entered command family: {forbidden}")
    if commands["visual"][commands["visual"].index("--checkpoints") + 1] != ",".join(
        map(str, CHECKPOINTS)
    ):
        raise AssertionError("visual checkpoints changed")
    called = False

    def valid() -> dict[str, Any]:
        nonlocal called
        called = True
        return {"valid": True}

    exists, result = _validate_or_absent(
        Path("/synthetic"), valid, _exists=lambda _path: False
    )
    assert exists is False and result is None and called is False
    try:
        def invalid() -> dict[str, Any]:
            raise RuntimeError("synthetic invalid immutable output")

        _validate_or_absent(
            Path("/synthetic"), invalid, _exists=lambda _path: True
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid existing product was accepted")
    if TRAJECTORY_COUNT != 1800 or SEEDS[0] != 250 or SEEDS[-1] != 849:
        raise AssertionError("third-pool cohort changed")
    if VISUAL_PHYSICAL_GPU == ENDPOINT_PHYSICAL_GPU:
        raise AssertionError("parallel extractors share a physical GPU")
    try:
        require_exact_path(Path("/synthetic/not-the-lock"), DEFAULT_SOURCE_LOCK, "test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-frozen path override was accepted")

    temporary = Path(tempfile.mkdtemp(prefix="third-pool-product-selftest-"))
    try:
        sample_index = temporary / "sample_index.csv"
        seed_outputs = [
            {"seed": seed, "identity_sha256": f"{seed:064x}"} for seed in SEEDS
        ]
        endpoint_outputs = []
        with sample_index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_index",
                    "global_seed",
                    "class_slot",
                    "class_id",
                    "trace_root",
                    "trace_identity_sha256",
                    "endpoint_png_path",
                    "endpoint_sha256",
                    "endpoint_pixel_sha256",
                ),
            )
            writer.writeheader()
            for seed in SEEDS:
                trace_root = Path("/synthetic/sealed_pool") / f"third_pool_v1_seed{seed:03d}"
                for slot, class_id in enumerate(CLASSES):
                    index = len(endpoint_outputs)
                    relative = f"images/{slot:02d}_class{class_id:04d}.png"
                    byte_hash = f"{index + 10_000:064x}"
                    pixel_hash = f"{index + 20_000:064x}"
                    endpoint_outputs.append(
                        {
                            "sample_index": index,
                            "global_seed": seed,
                            "class_slot": slot,
                            "class_id": class_id,
                            "relative_path": relative,
                            "bytes": 1,
                            "sha256": byte_hash,
                            "pixel_sha256": pixel_hash,
                        }
                    )
                    writer.writerow(
                        {
                            "sample_index": index,
                            "global_seed": seed,
                            "class_slot": slot,
                            "class_id": class_id,
                            "trace_root": str(trace_root),
                            "trace_identity_sha256": f"{seed:064x}",
                            "endpoint_png_path": str(trace_root / relative),
                            "endpoint_sha256": byte_hash,
                            "endpoint_pixel_sha256": pixel_hash,
                        }
                    )
        hashes, identities = _validate_endpoint_sample_lineage(
            sample_index,
            {
                "pool": {
                    "path": "/synthetic/sealed_pool",
                    "seed_outputs": seed_outputs,
                    "terminal_endpoint_outputs": endpoint_outputs,
                }
            },
        )
        if len(hashes) != TRAJECTORY_COUNT or identities != [
            item["identity_sha256"] for item in seed_outputs
        ]:
            raise AssertionError("synthetic endpoint lineage did not preserve exact axis")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print(
        "synthetic self-test passed: exact 1800 axis, label-free command family, "
        "terminal pixel lineage, primary-before-parallel ordering, distinct GPUs, "
        "fixed paths, and fail-closed reuse"
    )


def dry_run(source_lock: Path) -> dict[str, Any]:
    protocol, manifest = validate_source_lock(source_lock)
    return {
        "status": "DRY_RUN_SOURCE_ASSETS_ONLY_NO_POOL_OR_PRODUCT_ACCESS",
        "product_protocol_identity_sha256": protocol["identity_sha256"],
        "source_manifest_identity_sha256": manifest["identity_sha256"],
        "commands": protocol["execution"]["commands"],
        "pool_path_opened_statted_or_hashed": False,
        "product_paths_opened_statted_or_hashed": False,
        "labels_reviews_scores_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--bind-pool", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--validate-complete", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        validate_source_lock(DEFAULT_SOURCE_LOCK)
        synthetic_self_test()
        return 0
    if args.dry_run:
        print(json.dumps(dry_run(DEFAULT_SOURCE_LOCK), indent=2, sort_keys=True))
        return 0
    if args.bind_pool:
        path = bind_pool(DEFAULT_SOURCE_LOCK, DEFAULT_POOL_BINDING)
        record, manifest = _validate_small_lock(
            path,
            artifact_kind="dit_bad_good_third_pool_label_free_pool_binding_v1",
            record_name="pool_binding.json",
        )
        print(
            json.dumps(
                {
                    "path": str(path),
                    "pool_binding_identity_sha256": record["identity_sha256"],
                    "manifest_identity_sha256": manifest["identity_sha256"],
                    "pool_identity_sha256": record["pool"]["pool_identity_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.launch:
        print(
            json.dumps(
                launch(DEFAULT_SOURCE_LOCK, DEFAULT_POOL_BINDING),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    protocol, _ = validate_source_lock(DEFAULT_SOURCE_LOCK)
    binding, binding_manifest = validate_pool_binding(
        DEFAULT_POOL_BINDING, DEFAULT_SOURCE_LOCK
    )
    plan = build_execution_plan(protocol, binding, binding_manifest)
    print(json.dumps(validate_receipt(protocol, binding, binding_manifest, plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
