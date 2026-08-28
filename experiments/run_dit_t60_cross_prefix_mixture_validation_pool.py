#!/usr/bin/env python3
"""Prospective cross-prefix validation of the t60 fixed path-LR mixture.

Each invocation creates one immutable shard of eight independent class-207
DiT-XL/2 trajectories.  A branch owns one explicitly seeded ``torch.Generator``
and consumes exactly 251 draws from it: one initial latent followed by one
transition innovation at every internal timestep 249..0 (including the draw
that is multiplied by zero at t=0).  Eight shards form the fixed 64-path pool.

The implemented baseline-P sampler is never changed by evidence.  The released
``forward_with_cfg`` contract remains eight target branches -> sixteen
conditional/null model inputs, with CFG applied by upstream code.  Only the
first eight paths evolve; their distribution is the ordinary first-half DiT
ancestral distribution, generated with explicit branch-local RNG streams.

At t60..0 the runner observes one predeclared e-process: the uniform mixture of
34 *complete-path* likelihood ratios.  Its fixed components are global plus
row-major 4x4 latent tiles 00..15, each with path-fixed +theta and -theta.
Every component uses Delta-nu=0.25 and has total suffix conditional-KL cap
K=0.5.  The primary alarm is the mixture E >= 5; no tile, including tile_12,
is a primary statistic.  Alternatives are constructed before each innovation
exists and never feed into a P state update, rejection, ranking, or selection.

Evidence is kept in ``trace_private.npz``.  Public JSON and stdout contain no
evidence values, alarms, or ranks.  The staged bundle is validated by complete
transition and evidence reconstruction, self-hashed, and atomically installed
without replacement.  Procedural sealing is not cryptographic encryption.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    from .adm64_path_evidence import nearest_additive_heat_shift
    from .intervene_dit_imagenet256_suffix import (
        _atomic_install_directory_noreplace,
        _with_upstream_imports,
    )
    from .observe_dit_imagenet256_path_evidence import (
        BATCH_SIZE,
        FULL_BATCH_SIZE,
        fixed_tile_bounds,
        fixed_tile_masks,
        load_schedule,
    )
    from .reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_SHA256,
        CHECKPOINT_FILENAME,
        DIT_REVISION,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_REVISION,
        VAE_MODEL_ID,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        inspect_png,
        sha256_file,
        sha256_json,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from adm64_path_evidence import nearest_additive_heat_shift
    from intervene_dit_imagenet256_suffix import (
        _atomic_install_directory_noreplace,
        _with_upstream_imports,
    )
    from observe_dit_imagenet256_path_evidence import (
        BATCH_SIZE,
        FULL_BATCH_SIZE,
        fixed_tile_bounds,
        fixed_tile_masks,
        load_schedule,
    )
    from reproduce_dit_imagenet256 import (
        CFG_SCALE,
        CHECKPOINT_SHA256,
        CHECKPOINT_FILENAME,
        DIT_REVISION,
        IMAGE_SIZE,
        LATENT_CHANNELS,
        LATENT_SIZE,
        MODEL_NAME,
        NULL_CLASS_ID,
        NUM_CLASSES,
        NUM_SAMPLING_STEPS,
        VAE_REVISION,
        VAE_MODEL_ID,
        VAE_SCALING_FACTOR,
        atomic_json_dump,
        checkpoint_dry_probe,
        dependency_identity,
        ensure_single_process,
        inspect_png,
        sha256_file,
        sha256_json,
        validate_checkpoint,
        validate_repository,
        validate_vae_snapshot,
    )


EXPERIMENT = "dit_imagenet256_t60_cross_prefix_mixture_validation_pool"
SCHEMA_VERSION = 1
TARGET_CLASS_ID = 207
EVIDENCE_START_INTERNAL_TIMESTEP = 60
TOTAL_SHARDS = 8
BRANCHES_PER_SHARD = BATCH_SIZE
TOTAL_POOL_BRANCHES = TOTAL_SHARDS * BRANCHES_PER_SHARD
POOL_SEED = 20_260_827
RNG_NAMESPACE = "eqvae-dit256-t60-cross-prefix-mixture-validation-v1"
BLIND_NAMESPACE = "eqvae-dit256-t60-cross-prefix-blind-id-v1"

DELTA_NU = 0.25
TOTAL_K_PER_COMPONENT = 0.5
ALPHA_E = 0.2
ALARM_LOG_E = math.log(1.0 / ALPHA_E)
GRID_SIZE = 4
LOCAL_COMPONENT_COUNT = GRID_SIZE * GRID_SIZE
BASE_COMPONENT_COUNT = 1 + LOCAL_COMPONENT_COUNT
SIGN_VALUES = (1, -1)
SIGNED_COMPONENT_COUNT = len(SIGN_VALUES) * BASE_COMPONENT_COUNT
PRIMARY_STATISTIC = "uniform_fixed_34_complete_path_likelihood_ratio_mixture"
TRACE_NAME = "trace_private.npz"
PROTOCOL_COPY_NAME = "protocol.json"
RUNNER_DIR = Path(__file__).resolve().parent

EXPECTED_CUDA_DEVICE_NAME = "NVIDIA GeForce RTX 4090"
EXPECTED_CUDA_DEVICE_CAPABILITY = (8, 9)
EXPECTED_CUDNN_VERSION = 91_900
ANCHOR_BUILDER_FILENAME = "build_dit_class207_visual_anchor_pack.py"
ANCHOR_CONFIG_FILENAME = "dit_imagenet256_class207_visual_anchors_v1.json"
DEFAULT_ANCHOR_CONFIG_PATH = RUNNER_DIR / "configs" / ANCHOR_CONFIG_FILENAME
BLIND_PIPELINE_FILENAMES = {
    "blind_pack_builder": "build_dit_t60_cross_prefix_blind_pack.py",
    "consensus_locker": "lock_dit_t60_cross_prefix_consensus.py",
    "aggregate_summarizer": "summarize_dit_t60_cross_prefix_mixture_validation.py",
}
SEED_AUDIT_SCRIPT_FILENAME = "audit_dit_t60_cross_prefix_seed_materialization.py"
SEED_AUDIT_FORMAT = "eqvae_dit_t60_cross_prefix_seed_materialization_audit_v1"
SEED_AUDIT_STATUS = "PASS"
ANCHOR_PACK_COMPLETE_STATUS = "COMPLETE_AND_FROZEN_BEFORE_GPU_EXECUTION"

DEFAULT_PROTOCOL_PATH = (
    RUNNER_DIR / "configs/dit_imagenet256_t60_cross_prefix_mixture_validation_v1.json"
)

TRACE_DTYPES: dict[str, np.dtype[Any]] = {
    "branch_global_index": np.dtype(np.int16),
    "branch_stream_seed": np.dtype(np.int64),
    "generator_state_sha256_before": np.dtype("<U64"),
    "generator_state_sha256_after": np.dtype("<U64"),
    "rng_draw_tensor_raw_sha256": np.dtype("<U64"),
    "rng_draw_internal_timestep": np.dtype(np.int16),
    "full_internal_timestep": np.dtype(np.int16),
    "full_original_timestep": np.dtype(np.int16),
    "full_internal_alpha_bar": np.dtype(np.float64),
    "full_original_timestep_map": np.dtype(np.int64),
    "initial_latent": np.dtype(np.float32),
    "state_before": np.dtype(np.float32),
    "pred_xstart": np.dtype(np.float32),
    "p_mean": np.dtype(np.float32),
    "p_standard_deviation": np.dtype(np.float32),
    "transition_innovation": np.dtype(np.float32),
    "final_latents": np.dtype(np.float32),
    "decoded_images": np.dtype(np.float32),
    "evidence_internal_timestep": np.dtype(np.int16),
    "evidence_full_step_index": np.dtype(np.int16),
    "evidence_current_alpha_bar": np.dtype(np.float64),
    "shifted_internal_timestep": np.dtype(np.int16),
    "shifted_original_timestep": np.dtype(np.int16),
    "shifted_alpha_bar": np.dtype(np.float64),
    "rho": np.dtype(np.float64),
    "effective_nonidentity": np.dtype(np.uint8),
    "per_step_K_cap": np.dtype(np.float64),
    "tile_bounds_yxyx": np.dtype(np.int16),
    "base_component_name": np.dtype("<U16"),
    "signed_component_name": np.dtype("<U32"),
    "signed_component_base_index": np.dtype(np.int16),
    "signed_component_sign": np.dtype(np.int8),
    "component_weight": np.dtype(np.float64),
    "epsilon_current_reconstructed": np.dtype(np.float32),
    "epsilon_shifted": np.dtype(np.float32),
    "theta": np.dtype(np.float64),
    "component_raw_K": np.dtype(np.float64),
    "component_scale": np.dtype(np.float64),
    "component_K": np.dtype(np.float64),
    "component_R": np.dtype(np.float64),
    "component_L": np.dtype(np.float64),
    "component_log_e": np.dtype(np.float64),
    "mixture_path_log_e": np.dtype(np.float64),
    "mixture_alarm_after_transition": np.dtype(np.uint8),
    "mixture_ever_alarm": np.dtype(np.uint8),
    "mixture_first_alarm_step_index": np.dtype(np.int16),
    "mixture_first_alarm_internal_timestep": np.dtype(np.int16),
    "mixture_terminal_log_e": np.dtype(np.float64),
    "mixture_running_max_log_e": np.dtype(np.float64),
}


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON record must be an object: {path}")
    return payload


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = _read_json(path)
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and value not in {character * 64 for character in "0123456789abcdef"}
    )


def _torch_backend_flags() -> dict[str, Any]:
    """Return every float32/determinism switch frozen for formal GPU shards."""

    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _dependency_identity_without_cuda_initialization() -> dict[str, Any]:
    """Mirror dependency_identity while keeping the DRAFT gate CUDA-cold.

    Querying ``torch.backends.cudnn.version()`` initializes CUDA in the pinned
    torch build.  The linked cuDNN version is therefore an explicit frozen
    constant here and is checked against the live value immediately after an
    authorized device is selected.
    """

    packages = {
        "torchvision": "torchvision",
        "timm": "timm",
        "diffusers": "diffusers",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface-hub",
        "pillow": "pillow",
    }
    try:
        versions = {
            key: importlib.metadata.version(distribution)
            for key, distribution in packages.items()
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"missing frozen runtime dependency: {exc.name}") from exc
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        **versions,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": EXPECTED_CUDNN_VERSION,
    }


def _required_cuda_execution_contract() -> dict[str, Any]:
    identity = _dependency_identity_without_cuda_initialization()
    return {
        "required_device_name": EXPECTED_CUDA_DEVICE_NAME,
        "required_compute_capability": list(EXPECTED_CUDA_DEVICE_CAPABILITY),
        "runtime_dependency_identity": identity,
        "runtime_dependency_identity_sha256": sha256_json(identity),
        "torch_backend_flags": _torch_backend_flags(),
    }


def _validate_actual_cuda_hardware(name: Any, capability: Any) -> None:
    if name != EXPECTED_CUDA_DEVICE_NAME:
        raise RuntimeError(
            f"formal shard requires {EXPECTED_CUDA_DEVICE_NAME}, observed {name!r}"
        )
    if list(capability) != list(EXPECTED_CUDA_DEVICE_CAPABILITY):
        raise RuntimeError(
            "formal shard requires CUDA compute capability "
            f"{list(EXPECTED_CUDA_DEVICE_CAPABILITY)}, observed {capability!r}"
        )


def _validate_frozen_cuda_execution_binding(binding: Any) -> None:
    if not isinstance(binding, dict):
        raise RuntimeError("frozen protocol lacks an exact execution binding")
    if binding.get("cuda_execution_contract") != _required_cuda_execution_contract():
        raise RuntimeError(
            "frozen execution binding lacks the exact 4090/runtime/backend contract"
        )


def _validate_bound_script(
    record: Any, *, expected_filename: str, label: str
) -> Path:
    if not isinstance(record, dict) or set(record) != {"filename", "sha256"}:
        raise RuntimeError(f"{label} frozen source binding schema changed")
    if record.get("filename") != expected_filename or not _is_sha256(
        record.get("sha256")
    ):
        raise RuntimeError(f"{label} frozen source binding is missing or a placeholder")
    path = RUNNER_DIR / expected_filename
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"{label} frozen source file differs from its protocol binding")
    return path


def _validate_seed_materialization_audit_binding(protocol: dict[str, Any]) -> None:
    binding = protocol.get("seed_materialization_audit_binding")
    required_binding_keys = {
        "status",
        "audit_root",
        "audit_source",
        "candidate_seed_list_sha256",
        "report_payload_sha256",
        "report_file_sha256",
        "completion_payload_sha256",
        "completion_file_sha256",
        "inventory_identity_sha256",
        "inventory_file_sha256",
        "inventory_records_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != required_binding_keys:
        raise RuntimeError("frozen protocol lacks the exact seed-audit binding schema")
    if binding.get("status") != SEED_AUDIT_STATUS:
        raise RuntimeError("seed-materialization audit is not complete")
    for key in (
        "candidate_seed_list_sha256",
        "report_payload_sha256",
        "report_file_sha256",
        "completion_payload_sha256",
        "completion_file_sha256",
        "inventory_identity_sha256",
        "inventory_file_sha256",
        "inventory_records_sha256",
    ):
        if not _is_sha256(binding.get(key)):
            raise RuntimeError(f"seed-materialization audit has invalid {key}")
    expected_seed_hash = protocol["seed_lineage"][
        "branch_local_trajectory_seed_list_sha256"
    ]
    if binding["candidate_seed_list_sha256"] != expected_seed_hash:
        raise RuntimeError("seed-audit binding refers to a different 64-seed slate")
    source_path = _validate_bound_script(
        binding.get("audit_source"),
        expected_filename=SEED_AUDIT_SCRIPT_FILENAME,
        label="seed-materialization auditor",
    )
    raw_root = binding.get("audit_root")
    if not isinstance(raw_root, str) or not raw_root or not Path(raw_root).is_absolute():
        raise RuntimeError("seed-audit root must be a nonempty absolute path")
    root = Path(raw_root)
    if root.resolve() != root or not root.is_dir() or root.is_symlink():
        raise RuntimeError("seed-audit root is missing, indirect, or not a plain directory")
    expected_paths = {root / name for name in ("report.json", "completion.json", "inventory.json")}
    if (
        any(path.is_symlink() or not path.is_file() for path in expected_paths)
        or {path for path in root.rglob("*") if path.is_file()} != expected_paths
        or any(path.is_dir() for path in root.rglob("*"))
    ):
        raise RuntimeError("seed-audit bundle is not the exact closed three-file set")
    report_path = root / "report.json"
    completion_path = root / "completion.json"
    inventory_path = root / "inventory.json"
    file_bindings = {
        "report_file_sha256": sha256_file(report_path),
        "completion_file_sha256": sha256_file(completion_path),
        "inventory_file_sha256": sha256_file(inventory_path),
    }
    if any(binding[key] != value for key, value in file_bindings.items()):
        raise RuntimeError("seed-audit report/completion/inventory file hash changed")
    report = _read_self_hashed_json(report_path, "payload_sha256")
    completion = _read_self_hashed_json(completion_path, "payload_sha256")
    inventory = _read_self_hashed_json(inventory_path, "inventory_identity_sha256")
    if (
        binding["report_payload_sha256"] != report["payload_sha256"]
        or binding["completion_payload_sha256"] != completion["payload_sha256"]
        or binding["inventory_identity_sha256"]
        != inventory["inventory_identity_sha256"]
        or binding["inventory_records_sha256"]
        != inventory.get("inventory_records_sha256")
    ):
        raise RuntimeError("seed-audit canonical identities changed")
    flattened = [
        value
        for row in protocol["seed_lineage"]["branch_local_trajectory_seeds_by_shard"]
        for value in row
    ]
    audited_binding = report.get("audited_seed_binding", {})
    candidate_slate = report.get("candidate_seed_slate", {})
    finding = report.get("finding", {})
    zero_hit_counts = {"ledger": 0, "numpy": 0, "path": 0, "text": 0, "total": 0}
    if (
        report.get("format") != SEED_AUDIT_FORMAT
        or report.get("status") != SEED_AUDIT_STATUS
        or audited_binding.get("candidate_seed_list_sha256") != expected_seed_hash
        or audited_binding.get("candidate_seed_count") != TOTAL_POOL_BRANCHES
        or audited_binding.get("derivation_exact_match") is not True
        or audited_binding.get("derivation_namespace") != RNG_NAMESPACE
        or candidate_slate.get("values_in_trajectory_index_order") != flattened
        or candidate_slate.get("planned_protocol_occurrence_excluded") is not True
        or finding.get("hit_counts") != zero_hit_counts
        or finding.get("prior_materialization_hit_count") != 0
        or finding.get("unreadable_count") != 0
        or any(
            finding.get(key) != 0
            for key in (
                "candidate_duplicate_count",
                "candidate_known_value_collision_count",
                "candidate_namespace_value_collision_count",
                "candidate_zero_count",
                "numpy_hit_count",
                "path_hit_count",
                "text_hit_count",
            )
        )
    ):
        raise RuntimeError("seed-audit report does not certify the exact unseen seed slate")
    source_binding = report.get("audit_source_binding", {})
    if source_binding != {"path": str(source_path), "sha256": sha256_file(source_path)}:
        raise RuntimeError("seed-audit report was produced by a different source")
    stable_counts = report.get("scope", {}).get("stable_counts")
    stable_flags = report.get("scope", {}).get("stable_flags")
    required_count_keys = {
        "root_count",
        "files_inventoried",
        "file_bytes_inventoried",
        "text_files_scanned",
        "text_bytes_scanned",
        "dit_numpy_containers_inspected",
        "dit_numpy_arrays_header_inspected",
        "dit_numpy_arrays_body_inspected",
    }
    required_flags = {
        "all_classified_text_exact_decimal_scanned": True,
        "all_visited_paths_exact_decimal_scanned": True,
        "candidate_protocol_declaration_content_excluded": True,
        "dit_numpy_seed_capable_integer_string_metadata_bodies_inspected": True,
        "gpu_used": False,
        "path_relevant_dit_npy_npz_headers_inspected": True,
        "prior_run_manifests_results_logs_annotations_and_code_in_scope": True,
        "review_and_export_ledgers_in_scope": True,
        "unexecuted_candidate_runner_source_content_excluded": True,
    }
    if (
        not isinstance(stable_counts, dict)
        or set(stable_counts) != required_count_keys
        or any(type(stable_counts[key]) is not int or stable_counts[key] <= 0 for key in required_count_keys)
        or not isinstance(stable_flags, dict)
        or any(stable_flags.get(key) is not value for key, value in required_flags.items())
    ):
        raise RuntimeError("seed-audit scan scope/completion accounting is incomplete")
    inventory_binding = report.get("inventory_binding", {})
    if inventory_binding != {
        "inventory_file_sha256": binding["inventory_file_sha256"],
        "inventory_identity_sha256": binding["inventory_identity_sha256"],
    }:
        raise RuntimeError("seed-audit report/inventory binding changed")
    if (
        inventory.get("format") != f"{SEED_AUDIT_FORMAT}_inventory"
        or inventory.get("counters", {}).get("files_inventoried")
        != stable_counts["files_inventoried"]
        or inventory.get("counters", {}).get("file_bytes_inventoried")
        != stable_counts["file_bytes_inventoried"]
    ):
        raise RuntimeError("seed-audit inventory does not reconstruct stable scope counts")
    expected_completion = {
        "format": f"{SEED_AUDIT_FORMAT}_completion",
        "status": SEED_AUDIT_STATUS,
        "candidate_seed_list_sha256": expected_seed_hash,
        "hit_counts": zero_hit_counts,
        "prior_materialization_hit_count": 0,
        "unreadable_count": 0,
        "scope_counts": stable_counts,
        "scope_flags": stable_flags,
        "filesystem_scan_read_only": True,
        "gpu_used": False,
        "report_file_sha256": binding["report_file_sha256"],
        "report_payload_sha256": binding["report_payload_sha256"],
        "inventory_file_sha256": binding["inventory_file_sha256"],
        "inventory_identity_sha256": binding["inventory_identity_sha256"],
        "inventory_records_sha256": binding["inventory_records_sha256"],
        "audit_source_sha256": binding["audit_source"]["sha256"],
        "completed_utc": report.get("finished_utc"),
    }
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        raise RuntimeError("seed-audit completion does not repeat the stable audit result")


def _validate_blind_pipeline_binding(protocol: dict[str, Any]) -> None:
    binding = protocol.get("blind_pipeline_binding")
    expected = {
        "blind_pack_builder_filename": BLIND_PIPELINE_FILENAMES[
            "blind_pack_builder"
        ],
        "blind_pack_builder_sha256": sha256_file(
            RUNNER_DIR / BLIND_PIPELINE_FILENAMES["blind_pack_builder"]
        ),
        "consensus_locker_filename": BLIND_PIPELINE_FILENAMES[
            "consensus_locker"
        ],
        "consensus_locker_sha256": sha256_file(
            RUNNER_DIR / BLIND_PIPELINE_FILENAMES["consensus_locker"]
        ),
        "aggregate_unseal_summarizer_filename": BLIND_PIPELINE_FILENAMES[
            "aggregate_summarizer"
        ],
        "aggregate_unseal_summarizer_sha256": sha256_file(
            RUNNER_DIR / BLIND_PIPELINE_FILENAMES["aggregate_summarizer"]
        ),
    }
    if not isinstance(binding, dict) or binding != expected:
        raise RuntimeError("frozen protocol lacks the exact blind-pipeline binding schema")
    for key, filename in BLIND_PIPELINE_FILENAMES.items():
        prefix = {
            "blind_pack_builder": "blind_pack_builder",
            "consensus_locker": "consensus_locker",
            "aggregate_summarizer": "aggregate_unseal_summarizer",
        }[key]
        _validate_bound_script(
            {
                "filename": binding[f"{prefix}_filename"],
                "sha256": binding[f"{prefix}_sha256"],
            },
            expected_filename=filename,
            label=key,
        )


def _validate_blind_mapping_commitment_binding(protocol: dict[str, Any]) -> None:
    if "blind_mapping_commitment_path" in protocol:
        raise RuntimeError(
            "top-level blind_mapping_commitment_path is forbidden; bind it inside "
            "blind_mapping_commitment_binding"
        )
    raw_path = protocol.get("blind_mapping_commitment_binding", {}).get(
        "commitment_path"
    )
    if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
        raise RuntimeError("frozen protocol lacks an absolute blind-mapping commitment path")
    path = Path(raw_path)
    if path.resolve() != path or not path.is_file() or path.is_symlink():
        raise RuntimeError("blind-mapping commitment is missing, indirect, or not a plain file")
    try:
        from . import build_dit_t60_cross_prefix_blind_pack as blind_builder
    except ImportError:  # pragma: no cover - direct CLI execution.
        import build_dit_t60_cross_prefix_blind_pack as blind_builder
    blind_builder.validate_mapping_commitment(path, protocol)


def _validate_external_anchor_binding(protocol: dict[str, Any]) -> None:
    binding = protocol.get("external_visual_anchor_binding", {}).get(
        "metadata_stripped_anchor_pack"
    )
    required_keys = {
        "status",
        "public_pack_root",
        "public_pack_root_identity_sha256",
        "anchor_config_path",
        "anchor_config_identity_sha256",
        "anchor_config_file_sha256",
        "builder",
        "manifest_identity_sha256",
        "manifest_file_sha256",
        "pack_payload_sha256",
        "rubric_identity_sha256",
        "rubric_file_sha256",
        "completion_payload_sha256",
        "completion_file_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != required_keys:
        raise RuntimeError("frozen protocol lacks the complete external-anchor binding")
    if binding.get("status") != ANCHOR_PACK_COMPLETE_STATUS:
        raise RuntimeError("external visual anchor pack is not complete and frozen")
    for key in required_keys - {
        "status",
        "public_pack_root",
        "anchor_config_path",
        "builder",
    }:
        if not _is_sha256(binding.get(key)):
            raise RuntimeError(f"external-anchor binding has invalid {key}")
    _validate_bound_script(
        binding.get("builder"),
        expected_filename=ANCHOR_BUILDER_FILENAME,
        label="external-anchor builder",
    )
    raw_config_path = binding.get("anchor_config_path")
    if not isinstance(raw_config_path, str) or not Path(raw_config_path).is_absolute():
        raise RuntimeError("external-anchor configuration path must be absolute")
    config_path = Path(raw_config_path)
    if config_path.resolve() != DEFAULT_ANCHOR_CONFIG_PATH.resolve():
        raise RuntimeError("external-anchor binding points to the wrong configuration file")
    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError("external-anchor configuration is missing or indirect")
    if sha256_file(config_path) != binding["anchor_config_file_sha256"]:
        raise RuntimeError("external-anchor configuration file SHA-256 changed")
    config = _read_self_hashed_json(config_path, "anchor_config_identity_sha256")
    if (
        config.get("anchor_config_identity_sha256")
        != binding["anchor_config_identity_sha256"]
        or config.get("anchor_status") != "FROZEN_BEFORE_CROSS_PREFIX_POOL"
    ):
        raise RuntimeError("external-anchor configuration identity/status changed")
    raw_root = binding.get("public_pack_root")
    if not isinstance(raw_root, str) or not raw_root or not Path(raw_root).is_absolute():
        raise RuntimeError("external-anchor public pack root must be absolute")
    root = Path(raw_root)
    if root.resolve() != root or not root.is_dir() or root.is_symlink():
        raise RuntimeError("external-anchor public pack root is missing or indirect")
    root_identity = sha256_json(
        {
            "absolute_root": str(root),
            "pack_payload_sha256": binding["pack_payload_sha256"],
        }
    )
    if binding["public_pack_root_identity_sha256"] != root_identity:
        raise RuntimeError("external-anchor root identity differs from the frozen binding")
    try:
        from . import build_dit_class207_visual_anchor_pack as anchor_builder
    except ImportError:  # pragma: no cover - direct CLI execution.
        import build_dit_class207_visual_anchor_pack as anchor_builder
    anchor_builder.validate_anchor_config(
        config, binding["anchor_config_identity_sha256"]
    )
    manifest, completion = anchor_builder.validate_output_bundle(
        root, binding["anchor_config_identity_sha256"]
    )
    observed = {
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "pack_payload_sha256": manifest["pack_payload_sha256"],
        "rubric_identity_sha256": _read_self_hashed_json(
            root / "rubric.json", "rubric_identity_sha256"
        )["rubric_identity_sha256"],
        "rubric_file_sha256": sha256_file(root / "rubric.json"),
        "completion_payload_sha256": completion["payload_sha256"],
        "completion_file_sha256": sha256_file(root / "completion.json"),
    }
    if any(binding.get(key) != value for key, value in observed.items()):
        raise RuntimeError("external-anchor manifest/payload/completion binding changed")


def _require_frozen_pre_sampling_bindings(protocol: dict[str, Any]) -> None:
    _validate_seed_materialization_audit_binding(protocol)
    _validate_blind_pipeline_binding(protocol)
    _validate_blind_mapping_commitment_binding(protocol)
    _validate_external_anchor_binding(protocol)


def _array_raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _copy_npz_array_preserve_shape(value: np.ndarray) -> np.ndarray:
    return np.array(value, copy=True, order="C")


def _generator_state_sha256(generator: torch.Generator) -> str:
    state = generator.get_state().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(state).tobytes(order="C")).hexdigest()


def _batch_row_raw_sha256(value: np.ndarray) -> np.ndarray:
    if value.ndim < 2 or value.shape[0] != BRANCHES_PER_SHARD:
        raise ValueError("RNG draw array must have one row per shard branch")
    return np.asarray([_array_raw_sha256(value[index]) for index in range(value.shape[0])])


def _global_rng_state_sha256(device: torch.device) -> str:
    state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    return hashlib.sha256(state.cpu().numpy().tobytes(order="C")).hexdigest()


def _all_global_rng_states_sha256(device: torch.device) -> dict[str, str]:
    states = {
        "cpu": _global_rng_state_sha256(torch.device("cpu")),
    }
    if device.type == "cuda":
        states["cuda_device"] = _global_rng_state_sha256(device)
    return states


def _logmeanexp(value: np.ndarray, axis: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    answer = maximum + np.log(
        np.mean(np.exp(array - maximum), axis=axis, keepdims=True, dtype=np.float64)
    )
    return np.asarray(np.squeeze(answer, axis=axis), dtype=np.float64)


def _transcript_update(
    digest: Any, label: str, value: np.ndarray
) -> None:
    array = np.ascontiguousarray(value)
    metadata = json.dumps(
        {"label": label, "dtype": array.dtype.str, "shape": list(array.shape)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = array.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _new_P_replay_transcript(
    internal_timestep: np.ndarray, initial_latent: np.ndarray
) -> Any:
    digest = hashlib.sha256()
    digest.update(b"eqvae-cross-prefix-P-replay-transcript-v1\0")
    _transcript_update(digest, "internal_timestep", internal_timestep)
    _transcript_update(digest, "initial_latent", initial_latent)
    return digest


def _P_replay_transcript_step(
    digest: Any,
    step_index: int,
    *,
    state_before: np.ndarray,
    pred_xstart: np.ndarray,
    p_mean: np.ndarray,
    p_standard_deviation: np.ndarray,
    innovation: np.ndarray,
    state_after: np.ndarray,
) -> None:
    for name, value in (
        ("state_before", state_before),
        ("pred_xstart", pred_xstart),
        ("p_mean", p_mean),
        ("p_standard_deviation", p_standard_deviation),
        ("innovation", innovation),
        ("state_after", state_after),
    ):
        _transcript_update(digest, f"step_{step_index:03d}/{name}", value)


def P_replay_transcript_sha256(
    *,
    internal_timestep: np.ndarray,
    initial_latent: np.ndarray,
    state_before: np.ndarray,
    pred_xstart: np.ndarray,
    p_mean: np.ndarray,
    p_standard_deviation: np.ndarray,
    innovation: np.ndarray,
    final_latents: np.ndarray,
) -> str:
    steps = len(internal_timestep)
    digest = _new_P_replay_transcript(internal_timestep, initial_latent)
    for step_index in range(steps):
        following = (
            state_before[:, step_index + 1]
            if step_index + 1 < steps
            else final_latents
        )
        _P_replay_transcript_step(
            digest,
            step_index,
            state_before=state_before[:, step_index],
            pred_xstart=pred_xstart[:, step_index],
            p_mean=p_mean[:, step_index],
            p_standard_deviation=p_standard_deviation[:, step_index],
            innovation=innovation[:, step_index],
            state_after=following,
        )
    return digest.hexdigest()


def numpy_P_replay(
    initial_latent: np.ndarray,
    p_mean: np.ndarray,
    p_standard_deviation: np.ndarray,
    innovation: np.ndarray,
    internal_timestep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        initial_latent.dtype != np.float32
        or p_mean.dtype != np.float32
        or p_standard_deviation.dtype != np.float32
        or innovation.dtype != np.float32
        or p_mean.shape != p_standard_deviation.shape
        or p_mean.shape != innovation.shape
        or p_mean.shape[0] != initial_latent.shape[0]
        or p_mean.shape[2:] != initial_latent.shape[1:]
        or p_mean.shape[1] != len(internal_timestep)
    ):
        raise ValueError("invalid arrays for deterministic numpy P replay")
    states = np.empty_like(p_mean)
    current = initial_latent.copy()
    for step_index, internal_t in enumerate(internal_timestep.tolist()):
        states[:, step_index] = current
        current = np.ascontiguousarray(
            p_mean[:, step_index]
            + np.float32(internal_t > 0)
            * p_standard_deviation[:, step_index]
            * innovation[:, step_index],
            dtype=np.float32,
        )
    return states, current


def require_numpy_P_replay_bitwise_match(
    *,
    reference_state_before: np.ndarray,
    reference_final_latents: np.ndarray,
    initial_latent: np.ndarray,
    p_mean: np.ndarray,
    p_standard_deviation: np.ndarray,
    innovation: np.ndarray,
    internal_timestep: np.ndarray,
) -> None:
    mirror_states, mirror_final = numpy_P_replay(
        initial_latent,
        p_mean,
        p_standard_deviation,
        innovation,
        internal_timestep,
    )
    if not np.array_equal(mirror_states, reference_state_before):
        raise RuntimeError("numpy evidence-disabled P mirror state mismatch")
    if not np.array_equal(mirror_final, reference_final_latents):
        raise RuntimeError("numpy evidence-disabled P mirror final mismatch")


def branch_stream_seed(global_index: int) -> int:
    if global_index not in range(TOTAL_POOL_BRANCHES):
        raise ValueError("global branch index is outside the fixed 64-path pool")
    payload = (
        f"{RNG_NAMESPACE}\0{POOL_SEED}\0{TARGET_CLASS_ID}\0"
        f"{NUM_SAMPLING_STEPS}\0{global_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def blind_id(global_index: int) -> str:
    if global_index not in range(TOTAL_POOL_BRANCHES):
        raise ValueError("global branch index is outside the fixed 64-path pool")
    digest = hashlib.sha256(f"{BLIND_NAMESPACE}\0{global_index}".encode("ascii")).hexdigest()
    return f"cp1_{digest[:12]}"


def shard_global_indices(shard_index: int) -> tuple[int, ...]:
    if shard_index not in range(TOTAL_SHARDS):
        raise ValueError("shard index is outside the fixed pool")
    start = shard_index * BRANCHES_PER_SHARD
    return tuple(range(start, start + BRANCHES_PER_SHARD))


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_self_hashed_json(path, "protocol_identity_sha256")
    expected_top = {
        "schema_version": 1,
        "protocol_name": "dit_imagenet256_t60_cross_prefix_mixture_validation_v1",
    }
    if any(protocol.get(key) != value for key, value in expected_top.items()):
        raise RuntimeError("cross-prefix protocol name/schema changed")
    status = protocol.get("protocol_status")
    allowed_status = {"DRAFT_NOT_AUTHORIZED_FOR_GPU", "FROZEN_BEFORE_GPU_EXECUTION"}
    if status not in allowed_status:
        raise RuntimeError("cross-prefix protocol has an unknown authorization state")
    authorized = protocol.get("authorization_gate", {}).get("gpu_execution_authorized")
    if authorized is not (status == "FROZEN_BEFORE_GPU_EXECUTION"):
        raise RuntimeError("protocol status and GPU authorization gate disagree")

    pool = protocol.get("pool", {})
    pool_expected = {
        "shard_count": TOTAL_SHARDS,
        "class207_trajectories_per_shard": BRANCHES_PER_SHARD,
        "total_class207_trajectories": TOTAL_POOL_BRANCHES,
        "total_branch_local_generators": TOTAL_POOL_BRANCHES,
        "target_class_id": TARGET_CLASS_ID,
        "evidence_start_internal_timestep": EVIDENCE_START_INTERNAL_TIMESTEP,
        "independent_branch_generators_per_shard": BRANCHES_PER_SHARD,
        "per_prefix_suffix_count": 1,
    }
    if any(pool.get(key) != value for key, value in pool_expected.items()):
        raise RuntimeError("protocol pool constants changed")
    cfg = pool.get("cfg_batch_contract", {})
    if (
        cfg.get("model_batch_size") != FULL_BATCH_SIZE
        or cfg.get("null_class_id") != NULL_CLASS_ID
        or cfg.get("full_x_expression")
        != (
            "full_x = cat([x, x], dim=0) for x containing the eight independently "
            "evolving class-207 target states"
        )
        or cfg.get("labels")
        != "[207,207,207,207,207,207,207,207,1000,1000,1000,1000,1000,1000,1000,1000]"
    ):
        raise RuntimeError("protocol CFG 8-to-16 contract changed")

    candidate = protocol.get("evidence_candidate", {})
    construction = candidate.get("component_construction", {})
    if (
        construction.get("component_count") != SIGNED_COMPONENT_COUNT
        or construction.get("delta_nu") != DELTA_NU
        or construction.get("signs") != ["+theta", "-theta"]
        or construction.get("path_component_order")
        != (
            "global +theta, global -theta, then tile_00 +theta, tile_00 -theta, "
            "..., tile_15 +theta, tile_15 -theta"
        )
    ):
        raise RuntimeError("protocol fixed 34-component construction changed")
    mixture = candidate.get("fixed_path_mixture", {})
    if (
        mixture.get("initial_value") != 1.0
        or mixture.get("weights") != "uniform 1/34, fixed before sampling"
        or mixture.get("posthoc_component_max_forbidden") is not True
    ):
        raise RuntimeError("protocol complete-path mixture changed")
    alarm = candidate.get("alarm_boundary", {})
    if (
        alarm.get("alpha_e") != ALPHA_E
        or alarm.get("e_value_threshold") != 1.0 / ALPHA_E
        or not math.isclose(
            float(alarm.get("log_e_threshold", math.nan)),
            ALARM_LOG_E,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("protocol mixture alarm boundary changed")
    information = candidate.get("information_cap", {})
    if (
        information.get("per_component_total_K_cap") != TOTAL_K_PER_COMPONENT
        or information.get("same_rule_for_all_trajectories") is not True
    ):
        raise RuntimeError("protocol per-component K cap changed")
    window = candidate.get("observation_window", {})
    if (
        window.get("start_internal_timestep") != EVIDENCE_START_INTERNAL_TIMESTEP
        or window.get("end_internal_timestep") != 0
    ):
        raise RuntimeError("protocol evidence window changed")
    primary_text = candidate.get("primary_evidence_endpoint")
    if not isinstance(primary_text, str) or "No tile" not in primary_text:
        raise RuntimeError("protocol does not forbid a single-tile primary")

    sampler = protocol.get("sampler_binding_intent", {})
    sampler_expected = {
        "model": MODEL_NAME,
        "diffusion_steps": NUM_SAMPLING_STEPS,
        "image_size": IMAGE_SIZE,
        "cfg_scale": CFG_SCALE,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_revision": DIT_REVISION,
        "vae_model_id": VAE_MODEL_ID,
        "vae_revision": VAE_REVISION,
    }
    if any(sampler.get(key) != value for key, value in sampler_expected.items()):
        raise RuntimeError("protocol sampler/model binding changed")

    lineage = protocol.get("seed_lineage", {})
    derivation = lineage.get("derivation", {})
    derivation_expected = {
        "namespace": RNG_NAMESPACE,
        "pool_seed": POOL_SEED,
        "target_class_id": TARGET_CLASS_ID,
        "num_sampling_steps": NUM_SAMPLING_STEPS,
        "encoding": "ASCII exactly",
        "payload_formula": (
            "namespace + NUL + decimal(pool_seed) + NUL + decimal(target_class_id) + "
            "NUL + decimal(num_sampling_steps) + NUL + decimal(trajectory_index)"
        ),
    }
    if any(derivation.get(key) != value for key, value in derivation_expected.items()):
        raise RuntimeError("protocol branch-seed derivation changed")
    expected_rows = [
        [branch_stream_seed(index) for index in shard_global_indices(shard_index)]
        for shard_index in range(TOTAL_SHARDS)
    ]
    if lineage.get("branch_local_trajectory_seeds_by_shard") != expected_rows:
        raise RuntimeError("protocol's explicit 8x8 branch-seed slate changed")
    flattened = [value for row in expected_rows for value in row]
    expected_seed_hash = sha256_json(flattened)
    if lineage.get("branch_local_trajectory_seed_list_sha256") != expected_seed_hash:
        raise RuntimeError("protocol branch-seed list hash changed")
    if len(flattened) != TOTAL_POOL_BRANCHES or len(set(flattened)) != len(flattened):
        raise RuntimeError("protocol branch-seed slate is incomplete or non-unique")
    if any(value <= 0 or value >= (1 << 63) for value in flattened):
        raise RuntimeError("protocol branch seed lies outside (0,2^63)")

    exclusion = lineage.get("draft_exclusion_snapshot", {})
    records = exclusion.get("existing_DiT_branch_local_namespaces")
    if not isinstance(records, list) or len(records) != 2:
        raise RuntimeError("protocol known branch-local exclusion registry changed")
    old_values: list[int] = []
    for record in records:
        values = record.get("values")
        if (
            not isinstance(values, list)
            or record.get("value_count") != len(values)
            or record.get("values_sha256") != sha256_json(values)
        ):
            raise RuntimeError("protocol branch-local exclusion record is invalid")
        old_values.extend(values)
    if (
        exclusion.get("combined_existing_branch_local_value_count") != len(old_values)
        or exclusion.get("combined_existing_branch_local_values_sorted_sha256")
        != sha256_json(sorted(old_values))
    ):
        raise RuntimeError("protocol combined branch-local exclusion digest changed")
    legacy = exclusion.get("legacy_upstream_demo_seed_integers_conservative_only")
    if legacy != list(range(30)):
        raise RuntimeError("protocol conservative legacy seed exclusion changed")
    if set(flattened).intersection(old_values) or set(flattened).intersection(legacy):
        raise RuntimeError("new branch-seed slate intersects a declared prior seed set")
    canonical_json = protocol.get("hash_conventions", {}).get("canonical_json")
    if canonical_json != {
        "encoding": "UTF-8",
        "ensure_ascii": False,
        "separators": [",", ":"],
        "sort_keys": True,
    }:
        raise RuntimeError("protocol canonical JSON/hash convention changed")
    visual_primary = protocol.get("blind_review", {}).get("primary_visual_endpoint", {})
    if (
        visual_primary.get("name")
        != "overall_obvious_structural_bad_under_frozen_external_anchor_rubric"
        or "other 63 new endpoints" not in visual_primary.get("independent_per_image_rule", "")
    ):
        raise RuntimeError("protocol primary visual endpoint is not externally anchored")
    if status == "FROZEN_BEFORE_GPU_EXECUTION":
        _validate_frozen_cuda_execution_binding(
            protocol.get("frozen_execution_binding")
        )
        _require_frozen_pre_sampling_bindings(protocol)
    return protocol


def _evidence_schedule(alpha: np.ndarray) -> dict[str, np.ndarray | int | float]:
    if alpha.shape != (NUM_SAMPLING_STEPS,):
        raise ValueError("DiT alpha schedule must have 250 internal entries")
    internal = np.arange(EVIDENCE_START_INTERNAL_TIMESTEP, -1, -1, dtype=np.int64)
    shifted = np.zeros_like(internal)
    stochastic = internal > 0
    mapping = nearest_additive_heat_shift(alpha, internal[stochastic], DELTA_NU)
    shifted[stochastic] = mapping.shifted_timestep
    shifted[~stochastic] = 0
    effective = stochastic & (shifted != internal)
    effective_count = int(effective.sum())
    if effective_count <= 0:
        raise RuntimeError("Delta-nu=0.25 has no effective stochastic t60 suffix step")
    per_step_cap = TOTAL_K_PER_COMPONENT / effective_count * (1.0 - 2e-12)
    return {
        "internal_timestep": internal,
        "full_step_index": (NUM_SAMPLING_STEPS - 1 - internal).astype(np.int64),
        "shifted_internal_timestep": shifted,
        "effective_nonidentity": effective.astype(np.uint8),
        "effective_step_count": effective_count,
        "per_step_K_cap": float(per_step_cap),
    }


def reconstruct_current_epsilon(
    state: np.ndarray, pred_xstart: np.ndarray, alpha_bar: float
) -> np.ndarray:
    if state.shape != pred_xstart.shape or state.ndim != 4:
        raise ValueError("state and pred_xstart must match [branch,C,H,W]")
    if not 0.0 < alpha_bar < 1.0:
        raise ValueError("alpha_bar must lie in (0,1)")
    epsilon = (
        state.astype(np.float64, copy=False)
        - math.sqrt(alpha_bar) * pred_xstart.astype(np.float64, copy=False)
    ) / math.sqrt(1.0 - alpha_bar)
    if not np.isfinite(epsilon).all():
        raise ValueError("reconstructed epsilon is non-finite")
    return np.ascontiguousarray(epsilon, dtype=np.float32)


def _base_components() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = fixed_tile_bounds(grid_size=GRID_SIZE, height=LATENT_SIZE, width=LATENT_SIZE)
    local = fixed_tile_masks(bounds, height=LATENT_SIZE, width=LATENT_SIZE)
    global_mask = np.ones((1, 1, LATENT_SIZE, LATENT_SIZE), dtype=np.float64)
    masks = np.ascontiguousarray(np.concatenate([global_mask, local], axis=0))
    names = np.asarray(
        ["global", *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT))],
        dtype="<U16",
    )
    return masks, bounds, names


def _signed_component_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, base_names = _base_components()
    base_index = np.asarray(
        [index for index in range(BASE_COMPONENT_COUNT) for _ in SIGN_VALUES],
        dtype=np.int16,
    )
    signs = np.asarray(
        [sign for _ in range(BASE_COMPONENT_COUNT) for sign in SIGN_VALUES],
        dtype=np.int8,
    )
    names = np.asarray(
        [
            f"{('+' if sign > 0 else '-')}theta/{base_names[index]}"
            for index in range(BASE_COMPONENT_COUNT)
            for sign in SIGN_VALUES
        ],
        dtype="<U32",
    )
    weights = np.full((SIGNED_COMPONENT_COUNT,), 1.0 / SIGNED_COMPONENT_COUNT, dtype=np.float64)
    return base_index, signs, names, weights


def primary_spec() -> dict[str, Any]:
    return {
        "statistic": PRIMARY_STATISTIC,
        "aggregation": "uniform fixed mixture of 34 complete-path likelihood ratios",
        "component_count": SIGNED_COMPONENT_COUNT,
        "single_component_primary": False,
        "alarm_e_value": 1.0 / ALPHA_E,
        "alarm_log_e": ALARM_LOG_E,
    }


def construct_signed_components_before_innovation(
    theta: np.ndarray,
    p_sigma: np.ndarray,
    masks: np.ndarray,
    per_step_K_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct all predictable path-fixed +/- shifts without accepting z."""

    if theta.shape != p_sigma.shape or theta.ndim != 4:
        raise ValueError("theta and p_sigma must match [branch,C,H,W]")
    if theta.dtype != np.float64 or p_sigma.dtype != np.float32:
        raise TypeError("theta must be float64 and P sigma must be float32")
    if masks.shape != (BASE_COMPONENT_COUNT, 1, LATENT_SIZE, LATENT_SIZE):
        raise ValueError("global+tile mask geometry changed")
    if np.any(p_sigma <= 0.0) or not np.isfinite(theta).all():
        raise ValueError("invalid predictable Q input")
    if not math.isfinite(per_step_K_cap) or per_step_K_cap <= 0.0:
        raise ValueError("per-step K cap must be finite and positive")
    unsigned = np.ascontiguousarray(
        p_sigma.astype(np.float64, copy=False)[:, None] * theta[:, None] * masks[None]
    )
    signed = np.ascontiguousarray(
        np.stack([unsigned, -unsigned], axis=2).reshape(
            theta.shape[0],
            SIGNED_COMPONENT_COUNT,
            theta.shape[1],
            theta.shape[2],
            theta.shape[3],
        )
    )
    raw_K = 0.5 * np.sum(np.square(signed), axis=(2, 3, 4), dtype=np.float64)
    scale = np.ones_like(raw_K)
    positive = raw_K > 0.0
    scale[positive] = np.minimum(1.0, np.sqrt(per_step_K_cap / raw_K[positive]))
    whitened = np.ascontiguousarray(signed * scale[:, :, None, None, None])
    K = 0.5 * np.sum(np.square(whitened), axis=(2, 3, 4), dtype=np.float64)
    if np.any(K > per_step_K_cap * (1.0 + 2e-12)):
        raise AssertionError("a component exceeded its per-step K cap")
    return raw_K, scale, K, whitened


def evaluate_components_after_innovation(
    whitened_shift: np.ndarray, innovation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    expected = (
        innovation.shape[0],
        SIGNED_COMPONENT_COUNT,
        innovation.shape[1],
        innovation.shape[2],
        innovation.shape[3],
    )
    if whitened_shift.shape != expected:
        raise ValueError("innovation does not match the preconstructed shifts")
    if whitened_shift.dtype != np.float64 or innovation.dtype != np.float32:
        raise TypeError("LR evaluator requires float64 shifts and float32 innovation")
    reward = np.sum(
        whitened_shift * innovation.astype(np.float64, copy=False)[:, None],
        axis=(2, 3, 4),
        dtype=np.float64,
    )
    K = 0.5 * np.sum(np.square(whitened_shift), axis=(2, 3, 4), dtype=np.float64)
    return np.ascontiguousarray(reward), np.ascontiguousarray(reward - K)


def summarize_mixture(component_L: np.ndarray) -> dict[str, np.ndarray]:
    if component_L.ndim != 3 or component_L.shape[2] != SIGNED_COMPONENT_COUNT:
        raise ValueError("component increments must be [branch,step,34]")
    component_log_e = np.cumsum(component_L, axis=1, dtype=np.float64)
    mixture = _logmeanexp(component_log_e, axis=2)
    alarm = mixture >= ALARM_LOG_E
    ever = alarm.any(axis=1)
    first = np.full((component_L.shape[0],), -1, dtype=np.int16)
    first_internal = np.full_like(first, -1)
    for branch_index in range(component_L.shape[0]):
        if ever[branch_index]:
            first[branch_index] = int(np.flatnonzero(alarm[branch_index])[0])
            first_internal[branch_index] = (
                EVIDENCE_START_INTERNAL_TIMESTEP - first[branch_index]
            )
    return {
        "component_log_e": np.ascontiguousarray(component_log_e, dtype=np.float64),
        "mixture_path_log_e": np.ascontiguousarray(mixture, dtype=np.float64),
        "mixture_alarm_after_transition": np.ascontiguousarray(alarm.astype(np.uint8)),
        "mixture_ever_alarm": np.ascontiguousarray(ever.astype(np.uint8)),
        "mixture_first_alarm_step_index": first,
        "mixture_first_alarm_internal_timestep": first_internal,
        "mixture_terminal_log_e": np.ascontiguousarray(mixture[:, -1], dtype=np.float64),
        "mixture_running_max_log_e": np.ascontiguousarray(
            np.maximum(0.0, mixture.max(axis=1)), dtype=np.float64
        ),
    }


def _draw_branch_tensors(
    generators: list[torch.Generator],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    if len(generators) != BRANCHES_PER_SHARD:
        raise ValueError("one explicit generator per branch is required")
    global_before = _all_global_rng_states_sha256(device)
    before = np.empty((BRANCHES_PER_SHARD,), dtype="<U64")
    after = np.empty_like(before)
    draws: list[torch.Tensor] = []
    for index, generator in enumerate(generators):
        before[index] = _generator_state_sha256(generator)
        draws.append(
            torch.randn(
                (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        after[index] = _generator_state_sha256(generator)
        if before[index] == after[index]:
            raise RuntimeError("branch generator state did not advance")
    if _all_global_rng_states_sha256(device) != global_before:
        raise RuntimeError("explicit branch generators modified global RNG")
    return torch.cat(draws, dim=0), before, after


def _trace_shapes() -> dict[str, tuple[int, ...]]:
    b = BRANCHES_PER_SHARD
    full_steps = NUM_SAMPLING_STEPS
    evidence_steps = EVIDENCE_START_INTERNAL_TIMESTEP + 1
    state = (b, full_steps, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    evidence_state = (b, evidence_steps, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    component = (b, evidence_steps, SIGNED_COMPONENT_COUNT)
    branch_evidence = (b, evidence_steps)
    latent = (b, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    return {
        "branch_global_index": (b,),
        "branch_stream_seed": (b,),
        "generator_state_sha256_before": (b, full_steps + 1),
        "generator_state_sha256_after": (b, full_steps + 1),
        "rng_draw_tensor_raw_sha256": (b, full_steps + 1),
        "rng_draw_internal_timestep": (full_steps + 1,),
        "full_internal_timestep": (full_steps,),
        "full_original_timestep": (full_steps,),
        "full_internal_alpha_bar": (full_steps,),
        "full_original_timestep_map": (full_steps,),
        "initial_latent": latent,
        "state_before": state,
        "pred_xstart": state,
        "p_mean": state,
        "p_standard_deviation": state,
        "transition_innovation": state,
        "final_latents": latent,
        "decoded_images": (b, 3, IMAGE_SIZE, IMAGE_SIZE),
        "evidence_internal_timestep": (evidence_steps,),
        "evidence_full_step_index": (evidence_steps,),
        "evidence_current_alpha_bar": (evidence_steps,),
        "shifted_internal_timestep": (evidence_steps,),
        "shifted_original_timestep": (evidence_steps,),
        "shifted_alpha_bar": (evidence_steps,),
        "rho": (evidence_steps,),
        "effective_nonidentity": (evidence_steps,),
        "per_step_K_cap": (),
        "tile_bounds_yxyx": (LOCAL_COMPONENT_COUNT, 4),
        "base_component_name": (BASE_COMPONENT_COUNT,),
        "signed_component_name": (SIGNED_COMPONENT_COUNT,),
        "signed_component_base_index": (SIGNED_COMPONENT_COUNT,),
        "signed_component_sign": (SIGNED_COMPONENT_COUNT,),
        "component_weight": (SIGNED_COMPONENT_COUNT,),
        "epsilon_current_reconstructed": evidence_state,
        "epsilon_shifted": evidence_state,
        "theta": evidence_state,
        "component_raw_K": component,
        "component_scale": component,
        "component_K": component,
        "component_R": component,
        "component_L": component,
        "component_log_e": component,
        "mixture_path_log_e": branch_evidence,
        "mixture_alarm_after_transition": branch_evidence,
        "mixture_ever_alarm": (b,),
        "mixture_first_alarm_step_index": (b,),
        "mixture_first_alarm_internal_timestep": (b,),
        "mixture_terminal_log_e": (b,),
        "mixture_running_max_log_e": (b,),
    }


def _assert_trace_schema(arrays: dict[str, np.ndarray]) -> None:
    shapes = _trace_shapes()
    if set(arrays) != set(TRACE_DTYPES) or set(arrays) != set(shapes):
        raise RuntimeError("private trace key set changed")
    for key, value in arrays.items():
        if value.shape != shapes[key] or value.dtype != TRACE_DTYPES[key]:
            raise RuntimeError(
                f"private trace contract changed: {key}: {value.shape}/{value.dtype}"
            )
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite private trace value: {key}")


def _atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _trace_record(path: Path, arrays: dict[str, np.ndarray], root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "keys": sorted(arrays),
        "schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(arrays.items())
        },
    }


def _png_record(path: Path, root: Path, size: tuple[int, int]) -> dict[str, Any]:
    record = {"relative_path": path.relative_to(root).as_posix()}
    record.update(inspect_png(path, "RGB", size))
    return record


def _encoded_save_image_png(
    tensor: torch.Tensor,
    *,
    nrow: int,
    padding: int,
) -> bytes:
    """Encode with the exact torchvision save_image pixel/PNG path in memory."""

    from torchvision.utils import make_grid

    grid = make_grid(
        tensor,
        nrow=nrow,
        padding=padding,
        normalize=True,
        value_range=(-1, 1),
    )
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    image = Image.fromarray(ndarr)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _grid_tile_pixels(grid_path: Path, local_index: int) -> np.ndarray:
    if local_index not in range(BRANCHES_PER_SHARD):
        raise ValueError("grid tile index is invalid")
    row, column = divmod(local_index, 4)
    left = 2 + column * (IMAGE_SIZE + 2)
    top = 2 + row * (IMAGE_SIZE + 2)
    with Image.open(grid_path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (1_034, 518):
            raise RuntimeError("unexpected blind-grid geometry")
        return np.ascontiguousarray(
            np.asarray(
                image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)), dtype=np.uint8
            )
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _source_dependencies() -> dict[str, dict[str, str]]:
    names = (
        "adm64_path_evidence.py",
        ANCHOR_BUILDER_FILENAME,
        SEED_AUDIT_SCRIPT_FILENAME,
        BLIND_PIPELINE_FILENAMES["blind_pack_builder"],
        BLIND_PIPELINE_FILENAMES["consensus_locker"],
        BLIND_PIPELINE_FILENAMES["aggregate_summarizer"],
        "intervene_dit_imagenet256_suffix.py",
        "observe_dit_imagenet256_path_evidence.py",
        "reproduce_dit_imagenet256.py",
    )
    return {
        name: {"path": str(RUNNER_DIR / name), "sha256": sha256_file(RUNNER_DIR / name)}
        for name in names
    }


def frozen_execution_binding_candidate(
    *,
    source: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> dict[str, Any]:
    local = _source_dependencies()
    vae_files = [
        {"name": record["name"], "bytes": record["bytes"], "sha256": record["sha256"]}
        for record in vae["files"]
    ]
    return {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "local_dependency_sha256": {
            name: record["sha256"] for name, record in sorted(local.items())
        },
        "dependency_identity": _dependency_identity_without_cuda_initialization(),
        "dependency_identity_sha256": sha256_json(
            _dependency_identity_without_cuda_initialization()
        ),
        "cuda_execution_contract": _required_cuda_execution_contract(),
        "dit_revision": source["revision"],
        "dit_commit_tree": source["commit_tree"],
        "dit_working_tracked_tree_sha256": source["working_tracked_tree_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vae_artifact_list_sha256": sha256_json(vae_files),
        "alpha_bar_raw_sha256": _array_raw_sha256(
            np.ascontiguousarray(alpha, dtype=np.float64)
        ),
        "original_timestep_map_raw_sha256": _array_raw_sha256(
            np.ascontiguousarray(timestep_map, dtype=np.int64)
        ),
        "cfg_scale": CFG_SCALE,
        "target_batch_size": BATCH_SIZE,
        "conditional_null_model_batch_size": FULL_BATCH_SIZE,
    }


def _require_frozen_gpu_authorization(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    source: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> None:
    """Fail before any CUDA/model/sampling action unless the exact file is frozen."""

    # Check cheap authorization fields first.  In particular, a DRAFT call
    # returns here without asking torch about CUDA or touching model inputs.
    if (
        protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION"
        or protocol.get("authorization_gate", {}).get("gpu_execution_authorized") is not True
    ):
        raise RuntimeError("lowest GPU entry rejected a non-frozen cross-prefix protocol")
    if protocol.get("protocol_identity_sha256") != _canonical_self_hash(
        protocol, "protocol_identity_sha256"
    ):
        raise RuntimeError("lowest GPU entry rejected a non-self-hashed protocol object")
    _require_frozen_pre_sampling_bindings(protocol)
    protocol_path = getattr(args, "protocol", None)
    if not isinstance(protocol_path, Path) or not protocol_path.is_file():
        raise RuntimeError("lowest GPU entry requires the exact frozen protocol file")
    disk_protocol = _load_protocol(protocol_path)
    if disk_protocol != protocol:
        raise RuntimeError("passed protocol object differs from the frozen protocol file")
    candidate = frozen_execution_binding_candidate(
        source=source,
        vae=vae,
        alpha=alpha,
        timestep_map=timestep_map,
    )
    if protocol.get("frozen_execution_binding") != candidate:
        raise RuntimeError(
            "lowest GPU entry rejected a runner/source/runtime/schedule binding mismatch"
        )


def canonical_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--shard-index",
        str(args.shard_index),
        "--device-index",
        str(args.device_index),
        "--protocol",
        str(args.protocol),
        "--dit-root",
        str(args.dit_root),
        "--checkpoint",
        str(args.checkpoint),
        "--vae-snapshot",
        str(args.vae_snapshot),
        "--outdir",
        str(args.outdir),
    ]


def build_manifest(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> dict[str, Any]:
    indices = shard_global_indices(args.shard_index)
    all_seeds = np.asarray(
        [branch_stream_seed(index) for index in range(TOTAL_POOL_BRANCHES)], dtype=np.int64
    )
    if len(set(all_seeds.tolist())) != TOTAL_POOL_BRANCHES:
        raise AssertionError("domain-separated full-trajectory seeds collided")
    schedule = _evidence_schedule(alpha)
    runner = Path(__file__).resolve()
    primary = primary_spec()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "PROSPECTIVE_CROSS_PREFIX_OBSERVE_ONLY_VALIDATION_SHARD",
        "scope": {
            "independent_full_trajectories": True,
            "independent_initial_latents": True,
            "independent_prefixes_at_t60": True,
            "same_class_across_pool": True,
            "cross_class_claim_eligible": False,
            "target_class_id": TARGET_CLASS_ID,
            "selection_origin": (
                "mixture family and t60 window were fixed after within-prefix discovery; "
                "all 64 full-trajectory branch streams are fresh"
            ),
        },
        "sampling_distribution": {
            "baseline_P_full_trajectory_unchanged": True,
            "evidence_changes_transition": False,
            "intervention_rejection_rollback_retry_or_guidance": False,
            "automatic_scoring_ranking_or_selection": False,
            "all_generated_branches_retained": True,
            "P_update_expression": "mean + 1[t>0] * sigma * branch_local_innovation",
            "evidence_disabled_full_P_mirror_required": True,
            "mirror_reuses_saved_initial_latents_and_250_innovations": True,
            "mirror_generator_draw_count": 0,
            "mirror_shifted_observer_call_count": 0,
            "mirror_comparison": "bitwise for every P tensor/state, final latent, decode, and PNG",
        },
        "pool": {
            "pool_seed": POOL_SEED,
            "total_shards": TOTAL_SHARDS,
            "branches_per_shard": BRANCHES_PER_SHARD,
            "total_pool_branches": TOTAL_POOL_BRANCHES,
            "this_shard_index": args.shard_index,
            "this_shard_global_branch_indices": list(indices),
            "all_eight_shards_required": True,
        },
        "primary_e_process": {
            **primary,
            "delta_nu": DELTA_NU,
            "observation_window_internal_timesteps": [
                EVIDENCE_START_INTERNAL_TIMESTEP,
                0,
            ],
            "base_components_in_order": [
                "global",
                *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT)),
            ],
            "signed_component_order": _signed_component_metadata()[2].tolist(),
            "ordering_rule": "base-major with +theta then -theta within every base",
            "path_fixed_sign": True,
            "component_weights": "exactly 1/34 fixed before every trajectory",
            "total_suffix_K_cap_per_component": TOTAL_K_PER_COMPONENT,
            "per_effective_step_K_cap": schedule["per_step_K_cap"],
            "posthoc_component_max_used": False,
            "tile_12_or_any_single_tile_is_primary": False,
        },
        "predictability": {
            "Q_built_before_each_observed_innovation_draw": True,
            "Q_constructor_accepts_innovation": False,
            "same_covariance_operational_Q": True,
            "component_log_lr_increment": "<u,z>-0.5*||u||^2",
            "mixture_is_over_complete_path_LRs_not_per_step_sign_switching": True,
            "ideal_heat_marginal_ratio_claimed": False,
        },
        "cfg_contract": {
            "model": MODEL_NAME,
            "cfg_scale": CFG_SCALE,
            "target_first_half_branch_count": BATCH_SIZE,
            "conditional_null_model_batch_count": FULL_BATCH_SIZE,
            "all_first_half_labels": TARGET_CLASS_ID,
            "all_second_half_labels": NULL_CLASS_ID,
            "upstream_forward_with_cfg": True,
            "first_half_duplicated_inside_upstream_model": True,
            "guided_epsilon_channels": [0, 1, 2],
            "fourth_epsilon_channel_retained": True,
            "incoming_second_half_is_computational_carrier_not_sampled_path": True,
        },
        "rng": {
            "namespace": RNG_NAMESPACE,
            "one_explicit_generator_per_branch": True,
            "global_rng_used_for_initial_latents_or_innovations": False,
            "draws_per_branch": 1 + NUM_SAMPLING_STEPS,
            "draw_order": "initial latent, then innovations at internal t=249..0",
            "t0_innovation_is_drawn_then_zero_multiplied": True,
            "per_branch_draw_shape": [1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "all_64_stream_seed_array_raw_sha256": _array_raw_sha256(all_seeds),
            "all_64_stream_seed_list_canonical_json_sha256": sha256_json(
                all_seeds.tolist()
            ),
            "this_shard_streams": [
                {
                    "global_index": index,
                    "blind_id": blind_id(index),
                    "seed": int(all_seeds[index]),
                }
                for index in indices
            ],
        },
        "blind_review": {
            "evidence_trace": TRACE_NAME,
            "procedurally_sealed_until_annotations_are_locked_and_hashed": True,
            "cryptographically_encrypted": False,
            "public_results_contain_evidence_values_alarm_flags_or_ranks": False,
            "sampling_stdout_contains_evidence_values_alarm_flags_or_ranks": False,
            "image_names_depend_on_evidence": False,
            "branch_output_selection": "none",
        },
        "schedule": {
            "num_internal_steps": NUM_SAMPLING_STEPS,
            "full_transition_axis": list(range(NUM_SAMPLING_STEPS - 1, -1, -1)),
            "evidence_transition_axis": list(
                range(EVIDENCE_START_INTERNAL_TIMESTEP, -1, -1)
            ),
            "effective_shifted_stochastic_steps": schedule["effective_step_count"],
            "alpha_bar_raw_sha256": _array_raw_sha256(
                np.ascontiguousarray(alpha, dtype=np.float64)
            ),
            "original_timestep_map_raw_sha256": _array_raw_sha256(
                np.ascontiguousarray(timestep_map, dtype=np.int64)
            ),
        },
        "protocol": {
            "source_path": str(args.protocol),
            "copied_relative_path": PROTOCOL_COPY_NAME,
            "protocol_identity_sha256": protocol["protocol_identity_sha256"],
            "protocol_status": protocol["protocol_status"],
            "source_file_sha256": sha256_file(args.protocol),
            "branch_seed_list_sha256": protocol["seed_lineage"][
                "branch_local_trajectory_seed_list_sha256"
            ],
            "frozen_execution_binding": frozen_execution_binding_candidate(
                source=source,
                vae=vae,
                alpha=alpha,
                timestep_map=timestep_map,
            ),
        },
        "sources": {
            "dit": source,
            "checkpoint": checkpoint,
            "vae": vae,
            "local_dependencies": _source_dependencies(),
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "dependencies": _dependency_identity_without_cuda_initialization(),
        "cuda_execution_contract": _required_cuda_execution_contract(),
        "device_index": args.device_index,
        "canonical_command": canonical_command(args),
        "outputs": {
            "private_trace": TRACE_NAME,
            "blind_image_directory": "blind_images",
            "blind_grid": "blind_grid.png",
            "all_outputs_retained": True,
            "atomic_no_replace": True,
            "no_overwrite": True,
        },
    }
    payload["identity_sha256"] = _canonical_self_hash(payload, "identity_sha256")
    return payload


def run_sampling_shard(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    source: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, dict[str, Any]]:
    _require_frozen_gpu_authorization(
        args,
        protocol=protocol,
        source=source,
        vae=vae,
        alpha=alpha,
        timestep_map=timestep_map,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a real cross-prefix shard")
    if args.device_index >= torch.cuda.device_count():
        raise RuntimeError("--device-index is not visible")
    ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"

    full_internal = np.arange(NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int64)
    evidence_schedule = _evidence_schedule(alpha)
    evidence_internal = np.asarray(evidence_schedule["internal_timestep"], dtype=np.int64)
    evidence_full_index = np.asarray(evidence_schedule["full_step_index"], dtype=np.int64)
    shifted = np.asarray(evidence_schedule["shifted_internal_timestep"], dtype=np.int64)
    effective = np.asarray(
        evidence_schedule["effective_nonidentity"], dtype=np.uint8
    ).astype(bool)
    per_step_cap = float(evidence_schedule["per_step_K_cap"])
    masks, tile_bounds, base_names = _base_components()
    component_base, component_sign, component_names, component_weights = (
        _signed_component_metadata()
    )
    global_indices = shard_global_indices(args.shard_index)
    seeds = [branch_stream_seed(index) for index in global_indices]

    b = BRANCHES_PER_SHARD
    fs = NUM_SAMPLING_STEPS
    es = EVIDENCE_START_INTERNAL_TIMESTEP + 1
    latent_shape = (b, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    full_shape = (b, fs, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    evidence_shape = (b, es, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    component_shape = (b, es, SIGNED_COMPONENT_COUNT)
    state_arrays = {
        key: np.empty(full_shape, dtype=np.float32)
        for key in (
            "state_before",
            "pred_xstart",
            "p_mean",
            "p_standard_deviation",
            "transition_innovation",
        )
    }
    epsilon_current = np.empty(evidence_shape, dtype=np.float32)
    epsilon_shifted = np.empty(evidence_shape, dtype=np.float32)
    theta = np.empty(evidence_shape, dtype=np.float64)
    component_raw_K = np.zeros(component_shape, dtype=np.float64)
    component_scale = np.ones(component_shape, dtype=np.float64)
    component_K = np.zeros(component_shape, dtype=np.float64)
    component_R = np.zeros(component_shape, dtype=np.float64)
    component_L = np.zeros(component_shape, dtype=np.float64)
    generator_before = np.empty((b, fs + 1), dtype="<U64")
    generator_after = np.empty_like(generator_before)
    draw_sha256 = np.empty_like(generator_before)

    def _execute() -> tuple[
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        from diffusion import create_diffusion
        from diffusers.models import AutoencoderKL
        from download import find_model
        from models import DiT_models

        imported = {
            "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
            "download": Path(sys.modules["download"].__file__).resolve(),
            "models": Path(sys.modules["models"].__file__).resolve(),
        }
        expected_imports = {
            "diffusion": (args.dit_root / "diffusion/__init__.py").resolve(),
            "download": (args.dit_root / "download.py").resolve(),
            "models": (args.dit_root / "models.py").resolve(),
        }
        if imported != expected_imports:
            raise RuntimeError(f"upstream import shadowing: {imported} != {expected_imports}")

        torch.cuda.set_device(args.device_index)
        device = torch.device("cuda", args.device_index)
        cuda_device_name = torch.cuda.get_device_name(device)
        cuda_device_capability = list(torch.cuda.get_device_capability(device))
        _validate_actual_cuda_hardware(cuda_device_name, cuda_device_capability)
        runtime_dependency_identity = dependency_identity()
        runtime_backend_flags = _torch_backend_flags()
        if _required_cuda_execution_contract() != {
            "required_device_name": cuda_device_name,
            "required_compute_capability": cuda_device_capability,
            "runtime_dependency_identity": runtime_dependency_identity,
            "runtime_dependency_identity_sha256": sha256_json(
                runtime_dependency_identity
            ),
            "torch_backend_flags": runtime_backend_flags,
        }:
            raise RuntimeError("live CUDA/runtime contract differs from the frozen contract")
        model = DiT_models[MODEL_NAME](input_size=LATENT_SIZE, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(find_model(str(args.checkpoint)))
        model.eval()
        diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
        if not np.array_equal(np.asarray(diffusion.timestep_map), timestep_map):
            raise RuntimeError("runtime DiT timestep map differs from the validated schedule")
        vae = AutoencoderKL.from_pretrained(
            str(args.vae_snapshot), local_files_only=True, use_safetensors=True
        ).to(device)
        vae.eval()

        y = torch.cat(
            [
                torch.full((b,), TARGET_CLASS_ID, dtype=torch.long, device=device),
                torch.full((b,), NULL_CLASS_ID, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        model_kwargs = {"y": y, "cfg_scale": CFG_SCALE}
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        sampling_global_rng = _all_global_rng_states_sha256(device)
        previous_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        shifted_forward_calls = 0
        try:
            first, before_hash, after_hash = _draw_branch_tensors(
                generators, device=device, dtype=torch.float32
            )
            generator_before[:, 0] = before_hash
            generator_after[:, 0] = after_hash
            initial_latent = np.ascontiguousarray(first.cpu().numpy(), dtype=np.float32)
            draw_sha256[:, 0] = _batch_row_raw_sha256(initial_latent)
            if initial_latent.shape != latent_shape:
                raise AssertionError("initial branch latent shape changed")

            for step_index, internal_t in enumerate(full_internal.tolist()):
                full_x = torch.cat([first, first], dim=0)
                t_internal = torch.full(
                    (FULL_BATCH_SIZE,), internal_t, dtype=torch.long, device=device
                )
                rng_before_model = _all_global_rng_states_sha256(device)
                out = diffusion.p_mean_variance(
                    model.forward_with_cfg,
                    full_x,
                    t_internal,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )
                if _all_global_rng_states_sha256(device) != rng_before_model:
                    raise RuntimeError("implemented-P model call consumed global CPU/CUDA RNG")
                mean = out["mean"][:b].contiguous()
                pred = out["pred_xstart"][:b].contiguous()
                sigma = torch.exp(0.5 * out["log_variance"][:b]).contiguous()
                expected_shape = (b, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
                if any(tuple(value.shape) != expected_shape for value in (mean, pred, sigma)):
                    raise RuntimeError("implemented-P transition output shape changed")

                state_np = np.ascontiguousarray(first.cpu().numpy(), dtype=np.float32)
                pred_np = np.ascontiguousarray(pred.cpu().numpy(), dtype=np.float32)
                mean_np = np.ascontiguousarray(mean.cpu().numpy(), dtype=np.float32)
                sigma_np = np.ascontiguousarray(sigma.cpu().numpy(), dtype=np.float32)
                state_arrays["state_before"][:, step_index] = state_np
                state_arrays["pred_xstart"][:, step_index] = pred_np
                state_arrays["p_mean"][:, step_index] = mean_np
                state_arrays["p_standard_deviation"][:, step_index] = sigma_np

                evidence_index = (
                    EVIDENCE_START_INTERNAL_TIMESTEP - internal_t
                    if internal_t <= EVIDENCE_START_INTERNAL_TIMESTEP
                    else None
                )
                whitened: np.ndarray | None = None
                if evidence_index is not None:
                    if int(evidence_internal[evidence_index]) != internal_t:
                        raise AssertionError("evidence/full transition axes diverged")
                    current_eps = reconstruct_current_epsilon(
                        state_np, pred_np, float(alpha[internal_t])
                    )
                    if effective[evidence_index]:
                        shifted_t = int(shifted[evidence_index])
                        rho_value = math.sqrt(float(alpha[shifted_t]) / float(alpha[internal_t]))
                        shifted_first = first * rho_value
                        shifted_full = torch.cat([shifted_first, shifted_first], dim=0)
                        shifted_original_t = torch.full(
                            (FULL_BATCH_SIZE,),
                            int(timestep_map[shifted_t]),
                            dtype=torch.long,
                            device=device,
                        )
                        rng_before_shifted = _all_global_rng_states_sha256(device)
                        shifted_output = model.forward_with_cfg(
                            shifted_full,
                            shifted_original_t,
                            y=y,
                            cfg_scale=CFG_SCALE,
                        )
                        if _all_global_rng_states_sha256(device) != rng_before_shifted:
                            raise RuntimeError("shifted model call consumed global CPU/CUDA RNG")
                        expected_shifted_shape = (
                            FULL_BATCH_SIZE,
                            2 * LATENT_CHANNELS,
                            LATENT_SIZE,
                            LATENT_SIZE,
                        )
                        if tuple(shifted_output.shape) != expected_shifted_shape:
                            raise RuntimeError("shifted DiT output shape changed")
                        shifted_eps = np.ascontiguousarray(
                            shifted_output[:b, :LATENT_CHANNELS].cpu().numpy(),
                            dtype=np.float32,
                        )
                        direction = (
                            -rho_value
                            * shifted_eps.astype(np.float64, copy=False)
                            / math.sqrt(1.0 - float(alpha[shifted_t]))
                            + current_eps.astype(np.float64, copy=False)
                            / math.sqrt(1.0 - float(alpha[internal_t]))
                        )
                        direction = np.ascontiguousarray(direction, dtype=np.float64)
                        raw, scale, K, whitened = construct_signed_components_before_innovation(
                            direction, sigma_np, masks, per_step_cap
                        )
                        component_raw_K[:, evidence_index] = raw
                        component_scale[:, evidence_index] = scale
                        component_K[:, evidence_index] = K
                        shifted_forward_calls += 1
                    else:
                        shifted_eps = current_eps.copy()
                        direction = np.zeros_like(current_eps, dtype=np.float64)
                        whitened = np.zeros(
                            (
                                b,
                                SIGNED_COMPONENT_COUNT,
                                LATENT_CHANNELS,
                                LATENT_SIZE,
                                LATENT_SIZE,
                            ),
                            dtype=np.float64,
                        )
                    epsilon_current[:, evidence_index] = current_eps
                    epsilon_shifted[:, evidence_index] = shifted_eps
                    theta[:, evidence_index] = direction

                    # A direct mutation guard complements the algebraic replay
                    # validator: observer calls may not alter any already-built
                    # P tensor or the current state before z is drawn.
                    for tensor, frozen_value, label in (
                        (first, state_np, "state"),
                        (pred, pred_np, "pred_xstart"),
                        (mean, mean_np, "mean"),
                        (sigma, sigma_np, "standard_deviation"),
                    ):
                        if not np.array_equal(
                            np.ascontiguousarray(tensor.cpu().numpy(), dtype=np.float32),
                            frozen_value,
                        ):
                            raise RuntimeError(
                                f"observation mutated implemented-P {label} at t={internal_t}"
                            )

                # The P innovation is created only after any observed Q shifts.
                innovation, before_hash, after_hash = _draw_branch_tensors(
                    generators, device=device, dtype=first.dtype
                )
                generator_before[:, step_index + 1] = before_hash
                generator_after[:, step_index + 1] = after_hash
                innovation_np = np.ascontiguousarray(innovation.cpu().numpy(), dtype=np.float32)
                draw_sha256[:, step_index + 1] = _batch_row_raw_sha256(innovation_np)
                state_arrays["transition_innovation"][:, step_index] = innovation_np
                if evidence_index is not None:
                    if whitened is None:
                        raise AssertionError("observed Q was not constructed before innovation")
                    reward, increment = evaluate_components_after_innovation(
                        whitened, innovation_np
                    )
                    component_R[:, evidence_index] = reward
                    component_L[:, evidence_index] = increment

                first = (mean + float(internal_t != 0) * sigma * innovation).detach()
                if step_index % 25 == 0 or step_index + 1 == fs:
                    print(
                        f"shard {args.shard_index}: sampled {step_index + 1}/{fs} "
                        "baseline-P transitions",
                        flush=True,
                    )

            if _all_global_rng_states_sha256(device) != sampling_global_rng:
                raise RuntimeError("full-trajectory sampling changed global CPU/CUDA RNG")
            main_final = first.detach()
            final_latents = np.ascontiguousarray(main_final.cpu().numpy(), dtype=np.float32)
            transcript_internal = np.ascontiguousarray(full_internal, dtype=np.int16)
            main_transcript = P_replay_transcript_sha256(
                internal_timestep=transcript_internal,
                initial_latent=initial_latent,
                state_before=state_arrays["state_before"],
                pred_xstart=state_arrays["pred_xstart"],
                p_mean=state_arrays["p_mean"],
                p_standard_deviation=state_arrays["p_standard_deviation"],
                innovation=state_arrays["transition_innovation"],
                final_latents=final_latents,
            )

            # Full evidence-disabled P mirror.  It reuses only the saved initial
            # latent and innovations: no generator call and no shifted forward.
            mirror_global_before = _all_global_rng_states_sha256(device)
            mirror_generator_before = [_generator_state_sha256(value) for value in generators]
            mirror = torch.from_numpy(initial_latent).to(device=device)
            mirror_digest = _new_P_replay_transcript(transcript_internal, initial_latent)
            for step_index, internal_t in enumerate(full_internal.tolist()):
                expected_state = state_arrays["state_before"][:, step_index]
                mirror_state_np = np.ascontiguousarray(mirror.cpu().numpy(), dtype=np.float32)
                if not np.array_equal(mirror_state_np, expected_state):
                    raise RuntimeError(
                        f"evidence-disabled P mirror state differs at t={internal_t}"
                    )
                mirror_full = torch.cat([mirror, mirror], dim=0)
                mirror_t = torch.full(
                    (FULL_BATCH_SIZE,), internal_t, dtype=torch.long, device=device
                )
                rng_before_mirror_model = _all_global_rng_states_sha256(device)
                mirror_out = diffusion.p_mean_variance(
                    model.forward_with_cfg,
                    mirror_full,
                    mirror_t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )
                if _all_global_rng_states_sha256(device) != rng_before_mirror_model:
                    raise RuntimeError(
                        "evidence-disabled P mirror model call consumed global CPU/CUDA RNG"
                    )
                mirror_mean = mirror_out["mean"][:b].contiguous()
                mirror_pred = mirror_out["pred_xstart"][:b].contiguous()
                mirror_sigma = torch.exp(
                    0.5 * mirror_out["log_variance"][:b]
                ).contiguous()
                mirror_pred_np = np.ascontiguousarray(
                    mirror_pred.cpu().numpy(), dtype=np.float32
                )
                mirror_mean_np = np.ascontiguousarray(
                    mirror_mean.cpu().numpy(), dtype=np.float32
                )
                mirror_sigma_np = np.ascontiguousarray(
                    mirror_sigma.cpu().numpy(), dtype=np.float32
                )
                for live, expected, label in (
                    (
                        mirror_pred_np,
                        state_arrays["pred_xstart"][:, step_index],
                        "pred_xstart",
                    ),
                    (mirror_mean_np, state_arrays["p_mean"][:, step_index], "mean"),
                    (
                        mirror_sigma_np,
                        state_arrays["p_standard_deviation"][:, step_index],
                        "standard_deviation",
                    ),
                ):
                    if not np.array_equal(live, expected):
                        raise RuntimeError(
                            f"evidence-disabled P mirror {label} differs bitwise at "
                            f"t={internal_t}"
                        )
                mirror_innovation_np = state_arrays["transition_innovation"][:, step_index]
                mirror_innovation = torch.from_numpy(mirror_innovation_np).to(device=device)
                mirror_following = (
                    mirror_mean
                    + float(internal_t != 0) * mirror_sigma * mirror_innovation
                ).detach()
                mirror_following_np = np.ascontiguousarray(
                    mirror_following.cpu().numpy(), dtype=np.float32
                )
                expected_following = (
                    state_arrays["state_before"][:, step_index + 1]
                    if step_index + 1 < fs
                    else final_latents
                )
                if not np.array_equal(mirror_following_np, expected_following):
                    raise RuntimeError(
                        f"evidence-disabled P mirror next state differs bitwise at "
                        f"t={internal_t}"
                    )
                _P_replay_transcript_step(
                    mirror_digest,
                    step_index,
                    state_before=mirror_state_np,
                    pred_xstart=mirror_pred_np,
                    p_mean=mirror_mean_np,
                    p_standard_deviation=mirror_sigma_np,
                    innovation=mirror_innovation_np,
                    state_after=mirror_following_np,
                )
                mirror = mirror_following
                if step_index % 25 == 0 or step_index + 1 == fs:
                    print(
                        f"shard {args.shard_index}: bitwise P mirror "
                        f"{step_index + 1}/{fs}",
                        flush=True,
                    )
            mirror_transcript = mirror_digest.hexdigest()
            mirror_generator_after = [_generator_state_sha256(value) for value in generators]
            mirror_global_after = _all_global_rng_states_sha256(device)
            if mirror_generator_after != mirror_generator_before:
                raise RuntimeError("evidence-disabled P mirror advanced a branch generator")
            if mirror_global_after != mirror_global_before:
                raise RuntimeError("evidence-disabled P mirror changed global CPU/CUDA RNG")
            if mirror_transcript != main_transcript:
                raise RuntimeError("main and evidence-disabled P transcript hashes differ")
            mirror_final = mirror.detach()
            mirror_final_np = np.ascontiguousarray(
                mirror_final.cpu().numpy(), dtype=np.float32
            )
            if not np.array_equal(mirror_final_np, final_latents):
                raise RuntimeError("main and evidence-disabled P final latents differ bitwise")

            decoded = vae.decode(main_final / VAE_SCALING_FACTOR).sample
            mirror_decoded = vae.decode(mirror_final / VAE_SCALING_FACTOR).sample
            if not torch.equal(decoded, mirror_decoded):
                raise RuntimeError("main and mirror VAE decoded tensors differ bitwise")
            if _all_global_rng_states_sha256(device) != sampling_global_rng:
                raise RuntimeError("P mirror or VAE decode changed global CPU/CUDA RNG")
            torch.cuda.synchronize(device)
            decoded_np = np.ascontiguousarray(decoded.cpu().numpy(), dtype=np.float32)
            mirror_decoded_np = np.ascontiguousarray(
                mirror_decoded.cpu().numpy(), dtype=np.float32
            )
            if not np.array_equal(decoded_np, mirror_decoded_np):
                raise RuntimeError("main and mirror decoded CPU tensors differ bitwise")
            if (
                dependency_identity() != runtime_dependency_identity
                or _torch_backend_flags() != runtime_backend_flags
            ):
                raise RuntimeError("CUDA/runtime contract changed during formal sampling")
            imported_records = {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in imported.items()
            }
            execution = {
                "device": str(device),
                "cuda_device_name": cuda_device_name,
                "cuda_device_capability": cuda_device_capability,
                "runtime_dependency_identity": runtime_dependency_identity,
                "runtime_dependency_identity_sha256": sha256_json(
                    runtime_dependency_identity
                ),
                "torch_backend_flags": runtime_backend_flags,
                "runtime_upstream_imports": imported_records,
                "implemented_P_forward_calls": fs,
                "evidence_disabled_mirror_P_forward_calls": fs,
                "total_P_forward_calls_including_mirror": 2 * fs,
                "shifted_observer_forward_calls": shifted_forward_calls,
                "evidence_disabled_mirror_shifted_forward_calls": 0,
                "branch_count": b,
                "initial_draw_count_per_branch": 1,
                "transition_draw_count_per_branch_including_t0": fs,
                "total_explicit_generator_draw_count_per_branch": fs + 1,
                "global_cuda_rng_sha256_after_model_setup": sampling_global_rng[
                    "cuda_device"
                ],
                "global_cpu_rng_sha256_after_model_setup": sampling_global_rng["cpu"],
                "global_cuda_rng_unchanged_during_sampling_and_decode": True,
                "global_cpu_rng_unchanged_during_sampling_and_decode": True,
                "evidence_disabled_mirror_generator_draw_count": 0,
                "evidence_disabled_mirror_branch_generator_state_sha256": (
                    mirror_generator_after
                ),
                "evidence_disabled_mirror_global_cuda_rng_sha256_before": (
                    mirror_global_before["cuda_device"]
                ),
                "evidence_disabled_mirror_global_cuda_rng_sha256_after": (
                    mirror_global_after["cuda_device"]
                ),
                "evidence_disabled_mirror_global_cpu_rng_sha256_before": (
                    mirror_global_before["cpu"]
                ),
                "evidence_disabled_mirror_global_cpu_rng_sha256_after": (
                    mirror_global_after["cpu"]
                ),
                "evidence_disabled_mirror_bitwise_fields": [
                    "state_before",
                    "pred_xstart",
                    "p_mean",
                    "p_standard_deviation",
                    "transition_innovation",
                    "state_after",
                    "final_latents",
                    "decoded_images",
                ],
                "evidence_disabled_mirror_bitwise_pass": True,
                "main_P_replay_transcript_sha256": main_transcript,
                "mirror_P_replay_transcript_sha256": mirror_transcript,
                "main_final_latents_raw_sha256": _array_raw_sha256(final_latents),
                "mirror_final_latents_raw_sha256": _array_raw_sha256(mirror_final_np),
                "main_decoded_images_raw_sha256": _array_raw_sha256(decoded_np),
                "mirror_decoded_images_raw_sha256": _array_raw_sha256(
                    mirror_decoded_np
                ),
                "CFG_8_to_16_shape_preserved": True,
                "evidence_changed_P_state": False,
                "P_tensors_unchanged_by_observer_calls": True,
            }
            return (
                decoded,
                mirror_decoded,
                initial_latent,
                final_latents,
                decoded_np,
                execution,
            )
        finally:
            torch.set_grad_enabled(previous_grad)

    (
        decoded,
        mirror_decoded,
        initial_latent,
        final_latents,
        decoded_images,
        execution,
    ) = _with_upstream_imports(args.dit_root, _execute)
    summaries = summarize_mixture(component_L)
    arrays: dict[str, np.ndarray] = {
        "branch_global_index": np.asarray(global_indices, dtype=np.int16),
        "branch_stream_seed": np.asarray(seeds, dtype=np.int64),
        "generator_state_sha256_before": generator_before,
        "generator_state_sha256_after": generator_after,
        "rng_draw_tensor_raw_sha256": draw_sha256,
        "rng_draw_internal_timestep": np.asarray(
            [-1, *full_internal.tolist()], dtype=np.int16
        ),
        "full_internal_timestep": np.ascontiguousarray(full_internal, dtype=np.int16),
        "full_original_timestep": np.ascontiguousarray(
            timestep_map[full_internal], dtype=np.int16
        ),
        "full_internal_alpha_bar": np.ascontiguousarray(alpha, dtype=np.float64),
        "full_original_timestep_map": np.ascontiguousarray(timestep_map, dtype=np.int64),
        "initial_latent": initial_latent,
        **state_arrays,
        "final_latents": final_latents,
        "decoded_images": decoded_images,
        "evidence_internal_timestep": np.ascontiguousarray(evidence_internal, dtype=np.int16),
        "evidence_full_step_index": np.ascontiguousarray(evidence_full_index, dtype=np.int16),
        "evidence_current_alpha_bar": np.ascontiguousarray(
            alpha[evidence_internal], dtype=np.float64
        ),
        "shifted_internal_timestep": np.ascontiguousarray(shifted, dtype=np.int16),
        "shifted_original_timestep": np.ascontiguousarray(
            timestep_map[shifted], dtype=np.int16
        ),
        "shifted_alpha_bar": np.ascontiguousarray(alpha[shifted], dtype=np.float64),
        "rho": np.ascontiguousarray(
            np.sqrt(alpha[shifted] / alpha[evidence_internal]), dtype=np.float64
        ),
        "effective_nonidentity": np.ascontiguousarray(effective.astype(np.uint8)),
        "per_step_K_cap": np.asarray(per_step_cap, dtype=np.float64),
        "tile_bounds_yxyx": np.ascontiguousarray(tile_bounds, dtype=np.int16),
        "base_component_name": base_names,
        "signed_component_name": component_names,
        "signed_component_base_index": component_base,
        "signed_component_sign": component_sign,
        "component_weight": component_weights,
        "epsilon_current_reconstructed": epsilon_current,
        "epsilon_shifted": epsilon_shifted,
        "theta": theta,
        "component_raw_K": component_raw_K,
        "component_scale": component_scale,
        "component_K": component_K,
        "component_R": component_R,
        "component_L": component_L,
        **summaries,
    }
    _assert_trace_schema(arrays)
    return arrays, decoded, mirror_decoded, execution


def _save_blind_outputs(
    staging: Path,
    decoded: torch.Tensor,
    mirror_decoded: torch.Tensor,
    global_indices: tuple[int, ...],
    *,
    save_image: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not torch.equal(decoded, mirror_decoded):
        raise RuntimeError("main/mirror decoded tensors diverged before PNG publication")
    decoded_cpu = decoded.detach().cpu()
    mirror_decoded_cpu = mirror_decoded.detach().cpu()
    if not torch.equal(decoded_cpu, mirror_decoded_cpu):
        raise RuntimeError("main/mirror decoded CPU tensors diverged before PNG publication")
    directory = staging / "blind_images"
    directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for local_index, global_index in enumerate(global_indices):
        identifier = blind_id(global_index)
        path = directory / f"{identifier}.png"
        save_image(
            decoded_cpu[local_index],
            path,
            nrow=1,
            padding=0,
            normalize=True,
            value_range=(-1, 1),
        )
        mirror_png = _encoded_save_image_png(
            mirror_decoded_cpu[local_index], nrow=1, padding=0
        )
        if path.read_bytes() != mirror_png:
            raise RuntimeError(f"main/mirror endpoint PNG differs: {identifier}")
        records.append(
            {
                "local_index": local_index,
                "global_index": global_index,
                "blind_id": identifier,
                "stream_seed": branch_stream_seed(global_index),
                "image": _png_record(path, staging, (IMAGE_SIZE, IMAGE_SIZE)),
            }
        )
    grid_path = staging / "blind_grid.png"
    save_image(decoded_cpu, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
    mirror_grid_png = _encoded_save_image_png(mirror_decoded_cpu, nrow=4, padding=2)
    if grid_path.read_bytes() != mirror_grid_png:
        raise RuntimeError("main/mirror blind-grid PNG differs")
    grid_record = _png_record(grid_path, staging, (1_034, 518))
    for local_index, record in enumerate(records):
        tile_sha = _array_raw_sha256(_grid_tile_pixels(grid_path, local_index))
        if tile_sha != record["image"]["pixel_sha256"]:
            raise RuntimeError("blind image differs from its grid tile")
        record["grid_tile_pixel_sha256"] = tile_sha
    png_transcript = sha256_json(
        [record["image"]["sha256"] for record in records]
        + [grid_record["sha256"]]
    )
    verification = {
        "evidence_disabled_mirror_png_byte_identity_pass": True,
        "endpoint_and_grid_png_sha256_transcript": png_transcript,
    }
    return records, grid_record, verification


def _load_trace(path: Path, record: dict[str, Any], root: Path) -> dict[str, np.ndarray]:
    if set(record) != {"relative_path", "bytes", "sha256", "keys", "schema"}:
        raise RuntimeError("private trace public record schema changed or leaks metadata")
    if record.get("relative_path") != path.relative_to(root).as_posix():
        raise RuntimeError("private trace relative path changed")
    if (
        not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError("private trace file identity failed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: _copy_npz_array_preserve_shape(archive[key]) for key in archive.files
        }
    if sorted(arrays) != record.get("keys"):
        raise RuntimeError("private trace key set changed")
    _assert_trace_schema(arrays)
    expected_schema = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(arrays.items())
    }
    if record.get("schema") != expected_schema:
        raise RuntimeError("private trace public shape/dtype schema changed")
    return arrays


def _validate_trace_math(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    shard = int(manifest["pool"]["this_shard_index"])
    expected_indices = np.asarray(shard_global_indices(shard), dtype=np.int16)
    expected_seeds = np.asarray(
        [branch_stream_seed(int(index)) for index in expected_indices], dtype=np.int64
    )
    if not np.array_equal(arrays["branch_global_index"], expected_indices):
        raise RuntimeError("trace branch allocation changed")
    if not np.array_equal(arrays["branch_stream_seed"], expected_seeds):
        raise RuntimeError("trace branch seed derivation changed")
    all_seeds = np.asarray(
        [branch_stream_seed(index) for index in range(TOTAL_POOL_BRANCHES)], dtype=np.int64
    )
    if len(set(all_seeds.tolist())) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("full-pool stream seeds are not unique")
    if manifest["rng"].get("all_64_stream_seed_array_raw_sha256") != _array_raw_sha256(
        all_seeds
    ):
        raise RuntimeError("manifest full-pool stream-seed digest changed")

    full_internal = np.arange(NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int16)
    if not np.array_equal(arrays["full_internal_timestep"], full_internal):
        raise RuntimeError("full transition axis is not exactly 249..0")
    expected_draw_axis = np.asarray([-1, *full_internal.tolist()], dtype=np.int16)
    if not np.array_equal(arrays["rng_draw_internal_timestep"], expected_draw_axis):
        raise RuntimeError("RNG draw axis must be initial latent then t249..0")
    alpha = arrays["full_internal_alpha_bar"]
    timestep_map = arrays["full_original_timestep_map"]
    if (
        np.any(alpha <= 0.0)
        or np.any(alpha >= 1.0)
        or np.any(np.diff(alpha) >= 0.0)
        or timestep_map[0] != 0
        or np.any(np.diff(timestep_map) <= 0)
        or int(timestep_map[-1]) > 999
    ):
        raise RuntimeError("saved full DiT schedule is invalid")
    if not np.array_equal(arrays["full_original_timestep"], timestep_map[full_internal]):
        raise RuntimeError("full original timestep axis does not match schedule")
    if _array_raw_sha256(alpha) != manifest["schedule"]["alpha_bar_raw_sha256"]:
        raise RuntimeError("saved alpha schedule differs from manifest")
    if _array_raw_sha256(timestep_map) != manifest["schedule"][
        "original_timestep_map_raw_sha256"
    ]:
        raise RuntimeError("saved timestep map differs from manifest")

    remapped = _evidence_schedule(alpha)
    evidence_internal = np.asarray(remapped["internal_timestep"], dtype=np.int16)
    evidence_full_index = np.asarray(remapped["full_step_index"], dtype=np.int16)
    shifted = np.asarray(remapped["shifted_internal_timestep"], dtype=np.int16)
    effective = np.asarray(remapped["effective_nonidentity"], dtype=np.uint8).astype(bool)
    if not np.array_equal(arrays["evidence_internal_timestep"], evidence_internal):
        raise RuntimeError("evidence transition axis changed")
    if not np.array_equal(arrays["evidence_full_step_index"], evidence_full_index):
        raise RuntimeError("evidence/full-axis join changed")
    if not np.array_equal(full_internal[evidence_full_index], evidence_internal):
        raise RuntimeError("evidence rows do not map to full t60..0 transitions")
    if not np.array_equal(arrays["evidence_current_alpha_bar"], alpha[evidence_internal]):
        raise RuntimeError("current evidence alpha-bar changed")
    if not np.array_equal(arrays["shifted_internal_timestep"], shifted):
        raise RuntimeError("nearest additive Delta-nu mapping changed")
    if not np.array_equal(arrays["shifted_original_timestep"], timestep_map[shifted]):
        raise RuntimeError("shifted original timestep mapping changed")
    if not np.array_equal(arrays["shifted_alpha_bar"], alpha[shifted]):
        raise RuntimeError("shifted alpha-bar changed")
    if not np.array_equal(arrays["effective_nonidentity"].astype(bool), effective):
        raise RuntimeError("effective shifted-step flags changed")
    expected_rho = np.sqrt(alpha[shifted] / alpha[evidence_internal])
    if not np.allclose(arrays["rho"], expected_rho, rtol=0.0, atol=2e-16):
        raise RuntimeError("scale pullback rho does not reconstruct")
    expected_cap = float(remapped["per_step_K_cap"])
    if arrays["per_step_K_cap"].item() != expected_cap:
        raise RuntimeError("per-effective-step K allocation changed")

    masks, expected_bounds, expected_base_names = _base_components()
    expected_base, expected_sign, expected_names, expected_weights = (
        _signed_component_metadata()
    )
    component_bindings = {
        "tile_bounds_yxyx": expected_bounds,
        "base_component_name": expected_base_names,
        "signed_component_base_index": expected_base,
        "signed_component_sign": expected_sign,
        "signed_component_name": expected_names,
        "component_weight": expected_weights,
    }
    for key, expected in component_bindings.items():
        if not np.array_equal(arrays[key], expected):
            raise RuntimeError(f"fixed 34-component ordering/geometry changed: {key}")
    declared_primary = manifest.get("primary_e_process", {})
    for key, expected in primary_spec().items():
        observed = declared_primary.get(key)
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise RuntimeError(f"primary mixture declaration changed: {key}")
        elif observed != expected:
            raise RuntimeError(f"primary mixture declaration changed: {key}")
    if declared_primary.get("tile_12_or_any_single_tile_is_primary") is not False:
        raise RuntimeError("tile_12 or another single component became primary")
    if "primary_component_index" in declared_primary:
        raise RuntimeError("single-component primary index is forbidden")

    if not np.array_equal(arrays["initial_latent"], arrays["state_before"][:, 0]):
        raise RuntimeError("initial latent is not the first full-trajectory state")
    initial_hashes = [_array_raw_sha256(row) for row in arrays["initial_latent"]]
    if len(set(initial_hashes)) != BRANCHES_PER_SHARD:
        raise RuntimeError("independent branch streams produced duplicate initial latents")
    t60_states = arrays["state_before"][:, int(evidence_full_index[0])]
    if len({_array_raw_sha256(row) for row in t60_states}) != BRANCHES_PER_SHARD:
        raise RuntimeError("cross-prefix shard contains duplicate t60 states")
    if np.any(arrays["p_standard_deviation"] <= 0.0):
        raise RuntimeError("implemented-P sigma must be strictly positive")

    for step_index, internal_t in enumerate(full_internal.tolist()):
        expected_next = (
            arrays["p_mean"][:, step_index]
            + np.float32(1.0 if internal_t > 0 else 0.0)
            * arrays["p_standard_deviation"][:, step_index]
            * arrays["transition_innovation"][:, step_index]
        )
        actual_next = (
            arrays["state_before"][:, step_index + 1]
            if step_index + 1 < NUM_SAMPLING_STEPS
            else arrays["final_latents"]
        )
        if not np.array_equal(expected_next, actual_next):
            error = float(
                np.max(
                    np.abs(
                        expected_next.astype(np.float64) - actual_next.astype(np.float64)
                    ),
                    initial=0.0,
                )
            )
            if error > 2e-6:
                raise RuntimeError(
                    f"baseline-P transition fails reconstruction at t={internal_t}: "
                    f"max_abs={error}"
                )
    if not np.array_equal(arrays["final_latents"], arrays["p_mean"][:, -1]):
        error = float(
            np.max(
                np.abs(
                    arrays["final_latents"].astype(np.float64)
                    - arrays["p_mean"][:, -1].astype(np.float64)
                ),
                initial=0.0,
            )
        )
        if error > 1e-7:
            raise RuntimeError("t=0 innovation was not exactly zero-multiplied")

    for key in (
        "generator_state_sha256_before",
        "generator_state_sha256_after",
        "rng_draw_tensor_raw_sha256",
    ):
        if np.any(np.char.str_len(arrays[key]) != 64):
            raise RuntimeError(f"invalid RNG SHA-256 trace: {key}")
    if np.any(
        arrays["generator_state_sha256_before"]
        == arrays["generator_state_sha256_after"]
    ):
        raise RuntimeError("a branch generator failed to advance on a required draw")
    if not np.array_equal(
        arrays["generator_state_sha256_after"][:, :-1],
        arrays["generator_state_sha256_before"][:, 1:],
    ):
        raise RuntimeError("a branch-local initial+250-draw generator stream is discontinuous")
    for draw_index in range(NUM_SAMPLING_STEPS + 1):
        if (
            len(set(arrays["generator_state_sha256_before"][:, draw_index].tolist()))
            != BRANCHES_PER_SHARD
            or len(set(arrays["generator_state_sha256_after"][:, draw_index].tolist()))
            != BRANCHES_PER_SHARD
        ):
            raise RuntimeError(f"branch generator states collide at draw {draw_index}")
    expected_draw_hashes = np.empty_like(arrays["rng_draw_tensor_raw_sha256"])
    expected_draw_hashes[:, 0] = _batch_row_raw_sha256(arrays["initial_latent"])
    for step_index in range(NUM_SAMPLING_STEPS):
        expected_draw_hashes[:, step_index + 1] = _batch_row_raw_sha256(
            arrays["transition_innovation"][:, step_index]
        )
    if not np.array_equal(arrays["rng_draw_tensor_raw_sha256"], expected_draw_hashes):
        raise RuntimeError("saved RNG draw hashes do not bind the actual tensors")

    es = EVIDENCE_START_INTERNAL_TIMESTEP + 1
    expected_eps = np.empty_like(arrays["epsilon_current_reconstructed"])
    expected_theta = np.zeros_like(arrays["theta"])
    expected_raw = np.zeros_like(arrays["component_raw_K"])
    expected_scale = np.ones_like(arrays["component_scale"])
    expected_K = np.zeros_like(arrays["component_K"])
    expected_R = np.zeros_like(arrays["component_R"])
    expected_L = np.zeros_like(arrays["component_L"])
    for evidence_index in range(es):
        full_index = int(evidence_full_index[evidence_index])
        internal_t = int(evidence_internal[evidence_index])
        current_eps = reconstruct_current_epsilon(
            arrays["state_before"][:, full_index],
            arrays["pred_xstart"][:, full_index],
            float(alpha[internal_t]),
        )
        expected_eps[:, evidence_index] = current_eps
        if effective[evidence_index]:
            shifted_t = int(shifted[evidence_index])
            direction = (
                -float(arrays["rho"][evidence_index])
                * arrays["epsilon_shifted"][:, evidence_index].astype(
                    np.float64, copy=False
                )
                / math.sqrt(1.0 - float(alpha[shifted_t]))
                + current_eps.astype(np.float64, copy=False)
                / math.sqrt(1.0 - float(alpha[internal_t]))
            )
            expected_theta[:, evidence_index] = direction
            raw, scale, K, whitened = construct_signed_components_before_innovation(
                np.ascontiguousarray(direction, dtype=np.float64),
                arrays["p_standard_deviation"][:, full_index],
                masks,
                expected_cap,
            )
            R, L = evaluate_components_after_innovation(
                whitened, arrays["transition_innovation"][:, full_index]
            )
            expected_raw[:, evidence_index] = raw
            expected_scale[:, evidence_index] = scale
            expected_K[:, evidence_index] = K
            expected_R[:, evidence_index] = R
            expected_L[:, evidence_index] = L
        elif not np.array_equal(
            arrays["epsilon_shifted"][:, evidence_index], current_eps
        ):
            raise RuntimeError("inactive shifted epsilon must equal current epsilon")
    eps_error = float(
        np.max(
            np.abs(arrays["epsilon_current_reconstructed"] - expected_eps), initial=0.0
        )
    )
    if eps_error > 1e-6:
        raise RuntimeError(f"current epsilon reconstruction failed: max_abs={eps_error}")
    theta_error = float(np.max(np.abs(arrays["theta"] - expected_theta), initial=0.0))
    if theta_error > 2e-13:
        raise RuntimeError(f"theta reconstruction failed: max_abs={theta_error}")
    for key, expected in (
        ("component_raw_K", expected_raw),
        ("component_scale", expected_scale),
        ("component_K", expected_K),
        ("component_R", expected_R),
        ("component_L", expected_L),
    ):
        if not np.array_equal(arrays[key], expected):
            error = float(np.max(np.abs(arrays[key] - expected), initial=0.0))
            raise RuntimeError(f"evidence reconstruction failed: {key}/max_abs={error}")
    summaries = summarize_mixture(expected_L)
    for key, expected in summaries.items():
        if not np.array_equal(arrays[key], expected):
            error = (
                float(np.max(np.abs(arrays[key] - expected), initial=0.0))
                if arrays[key].dtype.kind == "f"
                else None
            )
            raise RuntimeError(f"fixed path-mixture summary changed: {key}/max_abs={error}")
    if np.any(
        arrays["component_K"].sum(axis=1)
        > TOTAL_K_PER_COMPONENT * (1.0 + 2e-12)
    ):
        raise RuntimeError("a complete path component exceeded total suffix K=0.5")


def validate_output_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("cross-prefix shard bundle must not contain symlinks")
    manifest = _read_self_hashed_json(root / "manifest.json", "identity_sha256")
    fixed_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "PROSPECTIVE_CROSS_PREFIX_OBSERVE_ONLY_VALIDATION_SHARD",
    }
    if any(manifest.get(key) != value for key, value in fixed_manifest.items()):
        raise RuntimeError("cross-prefix manifest identity changed")
    if manifest.get("scope", {}).get("independent_full_trajectories") is not True:
        raise RuntimeError("bundle does not declare independent full trajectories")
    sampling = manifest.get("sampling_distribution", {})
    if (
        sampling.get("baseline_P_full_trajectory_unchanged") is not True
        or sampling.get("evidence_changes_transition") is not False
        or sampling.get("evidence_disabled_full_P_mirror_required") is not True
    ):
        raise RuntimeError("bundle does not declare unchanged baseline-P sampling")
    required_cuda = _required_cuda_execution_contract()
    if (
        manifest.get("dependencies") != required_cuda["runtime_dependency_identity"]
        or manifest.get("cuda_execution_contract") != required_cuda
    ):
        raise RuntimeError("manifest CUDA/runtime contract differs from the frozen runner")
    pool = manifest.get("pool", {})
    pool_expected = {
        "pool_seed": POOL_SEED,
        "total_shards": TOTAL_SHARDS,
        "branches_per_shard": BRANCHES_PER_SHARD,
        "total_pool_branches": TOTAL_POOL_BRANCHES,
        "all_eight_shards_required": True,
    }
    if any(pool.get(key) != value for key, value in pool_expected.items()):
        raise RuntimeError("fixed 8x8 pool contract changed")
    shard = int(pool.get("this_shard_index", -1))
    if pool.get("this_shard_global_branch_indices") != list(shard_global_indices(shard)):
        raise RuntimeError("manifest shard allocation changed")
    runner = Path(__file__).resolve()
    if manifest.get("runner", {}).get("sha256") != sha256_file(runner):
        raise RuntimeError("bundle was produced by a different runner source")
    for name, record in manifest.get("sources", {}).get("local_dependencies", {}).items():
        path = Path(record.get("path", ""))
        if path.name != name or not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"local dependency changed: {name}")
    protocol_path = root / PROTOCOL_COPY_NAME
    protocol = _load_protocol(protocol_path)
    protocol_record = manifest.get("protocol", {})
    if (
        protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION"
        or protocol.get("authorization_gate", {}).get("gpu_execution_authorized") is not True
        or protocol_record.get("protocol_identity_sha256")
        != protocol.get("protocol_identity_sha256")
        or protocol_record.get("protocol_status") != protocol.get("protocol_status")
        or protocol_record.get("source_file_sha256") != sha256_file(protocol_path)
        or protocol_record.get("branch_seed_list_sha256")
        != protocol["seed_lineage"]["branch_local_trajectory_seed_list_sha256"]
        or protocol_record.get("frozen_execution_binding")
        != protocol.get("frozen_execution_binding")
        or protocol_record.get("copied_relative_path") != PROTOCOL_COPY_NAME
    ):
        raise RuntimeError("bundle is not bound to one frozen cross-prefix protocol")
    if manifest.get("rng", {}).get(
        "all_64_stream_seed_list_canonical_json_sha256"
    ) != protocol["seed_lineage"]["branch_local_trajectory_seed_list_sha256"]:
        raise RuntimeError("manifest RNG slate differs from the frozen protocol")

    results = _read_self_hashed_json(root / "results.json", "payload_sha256")
    fixed_results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "protocol_status": "FROZEN_BEFORE_GPU_EXECUTION",
        "cross_prefix_validation": True,
        "independent_full_trajectories": True,
        "baseline_P_sampling_unchanged": True,
        "primary_statistic": PRIMARY_STATISTIC,
        "evidence_values_exposed": False,
        "alarm_flags_exposed": False,
        "branch_ranking_or_selection_performed": False,
    }
    if any(results.get(key) != value for key, value in fixed_results.items()):
        raise RuntimeError("results identity/scope changed")
    expected_result_keys = {
        *fixed_results,
        "shard_index",
        "branch_records",
        "blind_grid",
        "private_trace",
        "execution",
        "wall_seconds_before_publication",
        "platform",
        "payload_sha256",
    }
    if set(results) != expected_result_keys or results.get("shard_index") != shard:
        raise RuntimeError("results JSON schema or shard identity changed")
    execution = results.get("execution", {})
    expected_execution_keys = {
        "device",
        "cuda_device_name",
        "cuda_device_capability",
        "runtime_dependency_identity",
        "runtime_dependency_identity_sha256",
        "torch_backend_flags",
        "runtime_upstream_imports",
        "implemented_P_forward_calls",
        "evidence_disabled_mirror_P_forward_calls",
        "total_P_forward_calls_including_mirror",
        "shifted_observer_forward_calls",
        "evidence_disabled_mirror_shifted_forward_calls",
        "branch_count",
        "initial_draw_count_per_branch",
        "transition_draw_count_per_branch_including_t0",
        "total_explicit_generator_draw_count_per_branch",
        "global_cuda_rng_sha256_after_model_setup",
        "global_cpu_rng_sha256_after_model_setup",
        "global_cuda_rng_unchanged_during_sampling_and_decode",
        "global_cpu_rng_unchanged_during_sampling_and_decode",
        "evidence_disabled_mirror_generator_draw_count",
        "evidence_disabled_mirror_branch_generator_state_sha256",
        "evidence_disabled_mirror_global_cuda_rng_sha256_before",
        "evidence_disabled_mirror_global_cuda_rng_sha256_after",
        "evidence_disabled_mirror_global_cpu_rng_sha256_before",
        "evidence_disabled_mirror_global_cpu_rng_sha256_after",
        "evidence_disabled_mirror_bitwise_fields",
        "evidence_disabled_mirror_bitwise_pass",
        "main_P_replay_transcript_sha256",
        "mirror_P_replay_transcript_sha256",
        "main_final_latents_raw_sha256",
        "mirror_final_latents_raw_sha256",
        "main_decoded_images_raw_sha256",
        "mirror_decoded_images_raw_sha256",
        "evidence_disabled_mirror_png_byte_identity_pass",
        "endpoint_and_grid_png_sha256_transcript",
        "CFG_8_to_16_shape_preserved",
        "evidence_changed_P_state",
        "P_tensors_unchanged_by_observer_calls",
    }
    if (
        not isinstance(execution, dict)
        or set(execution) != expected_execution_keys
        or execution.get("cuda_device_name") != EXPECTED_CUDA_DEVICE_NAME
        or execution.get("cuda_device_capability")
        != list(EXPECTED_CUDA_DEVICE_CAPABILITY)
        or execution.get("runtime_dependency_identity")
        != required_cuda["runtime_dependency_identity"]
        or execution.get("runtime_dependency_identity_sha256")
        != required_cuda["runtime_dependency_identity_sha256"]
        or execution.get("torch_backend_flags")
        != required_cuda["torch_backend_flags"]
        or execution.get("implemented_P_forward_calls") != NUM_SAMPLING_STEPS
        or execution.get("evidence_disabled_mirror_P_forward_calls")
        != NUM_SAMPLING_STEPS
        or execution.get("total_P_forward_calls_including_mirror")
        != 2 * NUM_SAMPLING_STEPS
        or execution.get("shifted_observer_forward_calls")
        != manifest["schedule"]["effective_shifted_stochastic_steps"]
        or execution.get("evidence_disabled_mirror_shifted_forward_calls") != 0
        or execution.get("branch_count") != BRANCHES_PER_SHARD
        or execution.get("initial_draw_count_per_branch") != 1
        or execution.get("transition_draw_count_per_branch_including_t0")
        != NUM_SAMPLING_STEPS
        or execution.get("total_explicit_generator_draw_count_per_branch")
        != NUM_SAMPLING_STEPS + 1
        or not isinstance(execution.get("global_cuda_rng_sha256_after_model_setup"), str)
        or len(execution["global_cuda_rng_sha256_after_model_setup"]) != 64
        or not isinstance(execution.get("global_cpu_rng_sha256_after_model_setup"), str)
        or len(execution["global_cpu_rng_sha256_after_model_setup"]) != 64
        or execution.get("global_cuda_rng_unchanged_during_sampling_and_decode") is not True
        or execution.get("global_cpu_rng_unchanged_during_sampling_and_decode") is not True
        or execution.get("evidence_disabled_mirror_generator_draw_count") != 0
        or execution.get("evidence_disabled_mirror_global_cuda_rng_sha256_before")
        != execution.get("evidence_disabled_mirror_global_cuda_rng_sha256_after")
        or execution.get("evidence_disabled_mirror_global_cpu_rng_sha256_before")
        != execution.get("evidence_disabled_mirror_global_cpu_rng_sha256_after")
        or execution.get("evidence_disabled_mirror_global_cuda_rng_sha256_before")
        != execution.get("global_cuda_rng_sha256_after_model_setup")
        or execution.get("evidence_disabled_mirror_global_cpu_rng_sha256_before")
        != execution.get("global_cpu_rng_sha256_after_model_setup")
        or execution.get("evidence_disabled_mirror_bitwise_fields")
        != [
            "state_before",
            "pred_xstart",
            "p_mean",
            "p_standard_deviation",
            "transition_innovation",
            "state_after",
            "final_latents",
            "decoded_images",
        ]
        or execution.get("evidence_disabled_mirror_bitwise_pass") is not True
        or execution.get("evidence_disabled_mirror_png_byte_identity_pass") is not True
        or execution.get("main_P_replay_transcript_sha256")
        != execution.get("mirror_P_replay_transcript_sha256")
        or execution.get("main_final_latents_raw_sha256")
        != execution.get("mirror_final_latents_raw_sha256")
        or execution.get("main_decoded_images_raw_sha256")
        != execution.get("mirror_decoded_images_raw_sha256")
        or execution.get("CFG_8_to_16_shape_preserved") is not True
        or execution.get("evidence_changed_P_state") is not False
        or execution.get("P_tensors_unchanged_by_observer_calls") is not True
    ):
        raise RuntimeError("execution provenance does not match the frozen sampler")
    mirror_generator_hashes = execution[
        "evidence_disabled_mirror_branch_generator_state_sha256"
    ]
    execution_sha_keys = (
        "global_cuda_rng_sha256_after_model_setup",
        "global_cpu_rng_sha256_after_model_setup",
        "evidence_disabled_mirror_global_cuda_rng_sha256_before",
        "evidence_disabled_mirror_global_cuda_rng_sha256_after",
        "evidence_disabled_mirror_global_cpu_rng_sha256_before",
        "evidence_disabled_mirror_global_cpu_rng_sha256_after",
        "main_P_replay_transcript_sha256",
        "mirror_P_replay_transcript_sha256",
        "main_final_latents_raw_sha256",
        "mirror_final_latents_raw_sha256",
        "main_decoded_images_raw_sha256",
        "mirror_decoded_images_raw_sha256",
        "endpoint_and_grid_png_sha256_transcript",
    )
    if (
        not isinstance(mirror_generator_hashes, list)
        or len(mirror_generator_hashes) != BRANCHES_PER_SHARD
        or any(not isinstance(value, str) or len(value) != 64 for value in mirror_generator_hashes)
        or any(
            not isinstance(execution.get(key), str) or len(execution[key]) != 64
            for key in execution_sha_keys
        )
    ):
        raise RuntimeError("mirror execution SHA-256 provenance is malformed")
    runtime_imports = execution.get("runtime_upstream_imports", {})
    if set(runtime_imports) != {"diffusion", "download", "models"}:
        raise RuntimeError("runtime upstream import set changed")
    dit_root = Path(manifest["sources"]["dit"]["root"]).resolve()
    expected_runtime_paths = {
        "diffusion": (dit_root / "diffusion/__init__.py").resolve(),
        "download": (dit_root / "download.py").resolve(),
        "models": (dit_root / "models.py").resolve(),
    }
    for name, record in runtime_imports.items():
        path = Path(record.get("path", "")).resolve()
        if path != expected_runtime_paths[name]:
            raise RuntimeError(f"unexpected runtime upstream import: {name}")
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"runtime upstream import changed: {name}")

    records = results.get("branch_records")
    indices = shard_global_indices(shard)
    if not isinstance(records, list) or len(records) != BRANCHES_PER_SHARD:
        raise RuntimeError("blind branch record count changed")
    expected_files = {
        (root / "manifest.json").resolve(),
        (root / "results.json").resolve(),
        (root / "completion.json").resolve(),
        protocol_path.resolve(),
        (root / TRACE_NAME).resolve(),
        (root / "blind_grid.png").resolve(),
    }
    grid_record = _png_record(root / "blind_grid.png", root, (1_034, 518))
    if results.get("blind_grid") != grid_record:
        raise RuntimeError("blind grid identity changed")
    for local_index, global_index in enumerate(indices):
        identifier = blind_id(global_index)
        path = root / "blind_images" / f"{identifier}.png"
        expected_files.add(path.resolve())
        image_record = _png_record(path, root, (IMAGE_SIZE, IMAGE_SIZE))
        expected_record = {
            "local_index": local_index,
            "global_index": global_index,
            "blind_id": identifier,
            "stream_seed": branch_stream_seed(global_index),
            "image": image_record,
            "grid_tile_pixel_sha256": image_record["pixel_sha256"],
        }
        if records[local_index] != expected_record:
            raise RuntimeError(f"blind branch record changed: {identifier}")
        if _array_raw_sha256(
            _grid_tile_pixels(root / "blind_grid.png", local_index)
        ) != image_record["pixel_sha256"]:
            raise RuntimeError(f"blind image/grid mismatch: {identifier}")

    arrays = _load_trace(root / TRACE_NAME, results["private_trace"], root)
    reconstructed_binding = frozen_execution_binding_candidate(
        source=manifest["sources"]["dit"],
        vae=manifest["sources"]["vae"],
        alpha=arrays["full_internal_alpha_bar"],
        timestep_map=arrays["full_original_timestep_map"],
    )
    if (
        reconstructed_binding != protocol.get("frozen_execution_binding")
        or reconstructed_binding != protocol_record.get("frozen_execution_binding")
    ):
        raise RuntimeError("bundle does not reconstruct the frozen execution binding")
    expected_P_transcript = P_replay_transcript_sha256(
        internal_timestep=arrays["full_internal_timestep"],
        initial_latent=arrays["initial_latent"],
        state_before=arrays["state_before"],
        pred_xstart=arrays["pred_xstart"],
        p_mean=arrays["p_mean"],
        p_standard_deviation=arrays["p_standard_deviation"],
        innovation=arrays["transition_innovation"],
        final_latents=arrays["final_latents"],
    )
    if (
        execution["main_P_replay_transcript_sha256"] != expected_P_transcript
        or execution["mirror_P_replay_transcript_sha256"] != expected_P_transcript
        or execution["main_final_latents_raw_sha256"]
        != _array_raw_sha256(arrays["final_latents"])
        or execution["mirror_final_latents_raw_sha256"]
        != _array_raw_sha256(arrays["final_latents"])
        or execution["main_decoded_images_raw_sha256"]
        != _array_raw_sha256(arrays["decoded_images"])
        or execution["mirror_decoded_images_raw_sha256"]
        != _array_raw_sha256(arrays["decoded_images"])
        or execution["evidence_disabled_mirror_branch_generator_state_sha256"]
        != arrays["generator_state_sha256_after"][:, -1].tolist()
    ):
        raise RuntimeError("private trace does not reconstruct the live P mirror attestation")
    expected_png_transcript = sha256_json(
        [record["image"]["sha256"] for record in records]
        + [grid_record["sha256"]]
    )
    if execution["endpoint_and_grid_png_sha256_transcript"] != expected_png_transcript:
        raise RuntimeError("published PNG set differs from the mirror attestation")
    decoded_cpu = torch.from_numpy(arrays["decoded_images"])
    for local_index, global_index in enumerate(indices):
        endpoint_path = root / "blind_images" / f"{blind_id(global_index)}.png"
        expected_png = _encoded_save_image_png(
            decoded_cpu[local_index], nrow=1, padding=0
        )
        if endpoint_path.read_bytes() != expected_png:
            raise RuntimeError(
                f"decoded trace does not reconstruct endpoint PNG: {blind_id(global_index)}"
            )
    if (root / "blind_grid.png").read_bytes() != _encoded_save_image_png(
        decoded_cpu, nrow=4, padding=2
    ):
        raise RuntimeError("decoded trace does not reconstruct blind-grid PNG")
    _validate_trace_math(arrays, manifest)
    if not np.array_equal(
        arrays["branch_stream_seed"],
        np.asarray([record["stream_seed"] for record in records], dtype=np.int64),
    ):
        raise RuntimeError("public seed accounting is not bound to private trace")

    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("cross-prefix shard file set changed")
    actual_directories = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_directories != {(root / "blind_images").resolve()}:
        raise RuntimeError("cross-prefix shard directory set changed")

    completion = _read_self_hashed_json(root / "completion.json", "payload_sha256")
    fixed_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
        "results_payload_sha256": results["payload_sha256"],
        "results_file_sha256": sha256_file(root / "results.json"),
        "private_trace_sha256": results["private_trace"]["sha256"],
        "shard_index": shard,
        "branch_count": BRANCHES_PER_SHARD,
        "evidence_values_exposed": False,
    }
    if any(completion.get(key) != value for key, value in fixed_completion.items()):
        raise RuntimeError("completion record links/hashes changed")
    if set(completion) != {
        *fixed_completion,
        "finished_unix",
        "wall_seconds",
        "payload_sha256",
    }:
        raise RuntimeError("completion record schema changed")
    return manifest, results


def validate_output_pool(roots: Iterable[Path]) -> dict[str, Any]:
    paths = tuple(path.resolve() for path in roots)
    if len(paths) != TOTAL_SHARDS or len(set(paths)) != TOTAL_SHARDS:
        raise RuntimeError("the complete pool requires eight distinct shard directories")
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in paths:
        records.append(validate_output_bundle(path))
    by_shard: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for manifest, results in records:
        shard = int(manifest["pool"]["this_shard_index"])
        if shard in by_shard:
            raise RuntimeError(f"duplicate completed shard index: {shard}")
        by_shard[shard] = (manifest, results)
    if set(by_shard) != set(range(TOTAL_SHARDS)):
        raise RuntimeError("completed pool does not contain exactly shard indices 0..7")

    reference = by_shard[0][0]
    invariant_paths = (
        ("protocol", "protocol_identity_sha256"),
        ("protocol", "source_file_sha256"),
        ("protocol", "branch_seed_list_sha256"),
        ("runner", "sha256"),
        ("schedule", "alpha_bar_raw_sha256"),
        ("schedule", "original_timestep_map_raw_sha256"),
        ("schedule", "effective_shifted_stochastic_steps"),
        ("rng", "all_64_stream_seed_array_raw_sha256"),
        ("rng", "all_64_stream_seed_list_canonical_json_sha256"),
    )
    for shard, (manifest, _) in by_shard.items():
        for parent, key in invariant_paths:
            if manifest[parent][key] != reference[parent][key]:
                raise RuntimeError(f"cross-shard binding differs at {parent}.{key}: {shard}")
        for key in ("dit", "checkpoint", "vae", "local_dependencies"):
            if manifest["sources"][key] != reference["sources"][key]:
                raise RuntimeError(f"cross-shard source binding differs: {key}/{shard}")
        if manifest["primary_e_process"] != reference["primary_e_process"]:
            raise RuntimeError(f"cross-shard primary e-process differs: {shard}")
        if manifest["cfg_contract"] != reference["cfg_contract"]:
            raise RuntimeError(f"cross-shard CFG contract differs: {shard}")
        if manifest["cuda_execution_contract"] != reference["cuda_execution_contract"]:
            raise RuntimeError(f"cross-shard frozen CUDA contract differs: {shard}")
        reference_execution = by_shard[0][1]["execution"]
        execution = by_shard[shard][1]["execution"]
        for key in (
            "cuda_device_name",
            "cuda_device_capability",
            "runtime_dependency_identity",
            "runtime_dependency_identity_sha256",
            "torch_backend_flags",
        ):
            if execution[key] != reference_execution[key]:
                raise RuntimeError(
                    f"cross-shard live CUDA/runtime contract differs at {key}: {shard}"
                )

    global_indices: list[int] = []
    seeds: list[int] = []
    identifiers: list[str] = []
    for shard in range(TOTAL_SHARDS):
        _, results = by_shard[shard]
        for record in results["branch_records"]:
            global_indices.append(int(record["global_index"]))
            seeds.append(int(record["stream_seed"]))
            identifiers.append(str(record["blind_id"]))
    if global_indices != list(range(TOTAL_POOL_BRANCHES)):
        raise RuntimeError("complete pool branch order/coverage is not exactly 0..63")
    if seeds != [branch_stream_seed(index) for index in range(TOTAL_POOL_BRANCHES)]:
        raise RuntimeError("complete pool stream-seed slate changed")
    if len(set(seeds)) != TOTAL_POOL_BRANCHES or len(set(identifiers)) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("complete pool contains duplicate seeds or blind IDs")
    return {
        "status": "valid-complete-pool",
        "shard_count": TOTAL_SHARDS,
        "trajectory_count": TOTAL_POOL_BRANCHES,
        "protocol_identity_sha256": reference["protocol"]["protocol_identity_sha256"],
        "runner_sha256": reference["runner"]["sha256"],
        "evidence_values_exposed": False,
    }


def run_real(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> None:
    if (
        protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION"
        or protocol.get("authorization_gate", {}).get("gpu_execution_authorized") is not True
    ):
        raise RuntimeError(
            "real GPU execution is forbidden while the cross-prefix protocol is "
            "DRAFT_NOT_AUTHORIZED_FOR_GPU"
        )
    binding_candidate = frozen_execution_binding_candidate(
        source=source,
        vae=vae,
        alpha=alpha,
        timestep_map=timestep_map,
    )
    if protocol.get("frozen_execution_binding") != binding_candidate:
        raise RuntimeError(
            "frozen protocol execution binding differs from this runner/source/runtime/schedule"
        )
    if args.outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output path: {args.outdir}")
    manifest = build_manifest(
        args,
        protocol=protocol,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
        alpha=alpha,
        timestep_map=timestep_map,
    )
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(
        prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        atomic_json_dump(manifest, staging / "manifest.json")
        # Keep the exact reviewed bytes: reserialization would change the
        # source-file identity even when the canonical object is unchanged.
        protocol_copy = staging / PROTOCOL_COPY_NAME
        shutil.copyfile(args.protocol, protocol_copy)
        arrays, decoded, mirror_decoded, execution = run_sampling_shard(
            args,
            protocol=protocol,
            source=source,
            vae=vae,
            alpha=alpha,
            timestep_map=timestep_map,
        )

        def _save() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
            from torchvision.utils import save_image

            return _save_blind_outputs(
                staging,
                decoded,
                mirror_decoded,
                shard_global_indices(args.shard_index),
                save_image=save_image,
            )

        branch_records, grid_record, mirror_png_verification = _save()
        execution.update(mirror_png_verification)
        trace_path = staging / TRACE_NAME
        _atomic_npz_dump(arrays, trace_path)
        trace_record = _trace_record(trace_path, arrays, staging)
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "protocol_identity_sha256": protocol["protocol_identity_sha256"],
            "protocol_status": protocol["protocol_status"],
            "cross_prefix_validation": True,
            "independent_full_trajectories": True,
            "baseline_P_sampling_unchanged": True,
            "primary_statistic": PRIMARY_STATISTIC,
            "evidence_values_exposed": False,
            "alarm_flags_exposed": False,
            "branch_ranking_or_selection_performed": False,
            "shard_index": args.shard_index,
            "branch_records": branch_records,
            "blind_grid": grid_record,
            "private_trace": trace_record,
            "execution": execution,
            "wall_seconds_before_publication": time.time() - started,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "dependencies": dependency_identity(),
            },
        }
        results["payload_sha256"] = _canonical_self_hash(results, "payload_sha256")
        atomic_json_dump(results, staging / "results.json")
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "protocol_identity_sha256": protocol["protocol_identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": sha256_file(staging / "results.json"),
            "private_trace_sha256": trace_record["sha256"],
            "shard_index": args.shard_index,
            "branch_count": BRANCHES_PER_SHARD,
            "evidence_values_exposed": False,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / "completion.json")
        validate_output_bundle(staging)
        _atomic_install_directory_noreplace(staging, args.outdir)
    validate_output_bundle(args.outdir)
    print(
        json.dumps(
            {
                "status": "complete",
                "outdir": str(args.outdir),
                "shard_index": args.shard_index,
                "blind_ids": [
                    blind_id(index) for index in shard_global_indices(args.shard_index)
                ],
                "full_independent_trajectories": True,
                "primary_statistic": PRIMARY_STATISTIC,
                "evidence_values_exposed": False,
                "review_before_opening": TRACE_NAME,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_self_test(protocol_path: Path) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must begin before CUDA initialization")
    protocol = _load_protocol(protocol_path)
    if protocol["protocol_name"] != "dit_imagenet256_t60_cross_prefix_mixture_validation_v1":
        raise AssertionError("protocol guard failed")
    draft_probe = json.loads(json.dumps(protocol))
    draft_probe["protocol_status"] = "DRAFT_NOT_AUTHORIZED_FOR_GPU"
    draft_probe["authorization_gate"]["gpu_execution_authorized"] = False
    draft_probe["protocol_identity_sha256"] = _canonical_self_hash(
        draft_probe, "protocol_identity_sha256"
    )
    cuda_was_initialized = torch.cuda.is_initialized()
    try:
        run_sampling_shard(
            argparse.Namespace(protocol=protocol_path),
            protocol=draft_probe,
            source={},
            vae={},
            alpha=np.empty((0,), dtype=np.float64),
            timestep_map=np.empty((0,), dtype=np.int64),
        )
    except RuntimeError as exc:
        if "lowest GPU entry rejected" not in str(exc):
            raise AssertionError("direct DRAFT GPU-entry probe failed for the wrong reason") from exc
    else:
        raise AssertionError("direct run_sampling_shard call bypassed the DRAFT gate")
    if torch.cuda.is_initialized() != cuda_was_initialized:
        raise AssertionError("DRAFT direct-entry rejection touched CUDA")
    _validate_actual_cuda_hardware(
        EXPECTED_CUDA_DEVICE_NAME, EXPECTED_CUDA_DEVICE_CAPABILITY
    )
    for name, capability in (
        (None, EXPECTED_CUDA_DEVICE_CAPABILITY),
        ("NVIDIA RTX 6000 Ada Generation", EXPECTED_CUDA_DEVICE_CAPABILITY),
        (EXPECTED_CUDA_DEVICE_NAME, None),
        (EXPECTED_CUDA_DEVICE_NAME, (8, 6)),
    ):
        try:
            _validate_actual_cuda_hardware(name, capability)
        except (RuntimeError, TypeError):
            pass
        else:
            raise AssertionError("missing/wrong formal CUDA hardware was accepted")
    valid_execution_binding = {
        "cuda_execution_contract": _required_cuda_execution_contract()
    }
    _validate_frozen_cuda_execution_binding(valid_execution_binding)
    for bad_binding in (
        {},
        {"cuda_execution_contract": {}},
        {
            "cuda_execution_contract": {
                **_required_cuda_execution_contract(),
                "required_device_name": "NVIDIA RTX 6000 Ada Generation",
            }
        },
        {
            "cuda_execution_contract": {
                **_required_cuda_execution_contract(),
                "torch_backend_flags": {},
            }
        },
    ):
        try:
            _validate_frozen_cuda_execution_binding(bad_binding)
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing/wrong frozen CUDA/runtime contract was accepted")
    try:
        _require_frozen_pre_sampling_bindings({})
    except RuntimeError as exc:
        if "seed-audit" not in str(exc):
            raise AssertionError("pre-sampling binding probe failed unexpectedly") from exc
    else:
        raise AssertionError("missing frozen pre-sampling artifacts were accepted")
    scalar = np.asarray(0.5, dtype=np.float64)
    if _copy_npz_array_preserve_shape(scalar).shape != ():
        raise AssertionError("NPZ scalar copy changed zero-dimensional shape")

    expected_partition = list(range(TOTAL_POOL_BRANCHES))
    actual_partition = sorted(
        index
        for shard_index in range(TOTAL_SHARDS)
        for index in shard_global_indices(shard_index)
    )
    if actual_partition != expected_partition:
        raise AssertionError("eight shards do not partition the 64-path pool")
    if any(len(shard_global_indices(index)) != 8 for index in range(TOTAL_SHARDS)):
        raise AssertionError("a shard does not contain exactly eight branches")
    seeds = [branch_stream_seed(index) for index in range(TOTAL_POOL_BRANCHES)]
    if len(set(seeds)) != TOTAL_POOL_BRANCHES:
        raise AssertionError("cross-prefix branch seed collision")
    ids = [blind_id(index) for index in range(TOTAL_POOL_BRANCHES)]
    if len(set(ids)) != TOTAL_POOL_BRANCHES:
        raise AssertionError("cross-prefix blind-id collision")

    generators = [
        torch.Generator(device="cpu").manual_seed(seed) for seed in seeds[:BRANCHES_PER_SHARD]
    ]
    previous_after: np.ndarray | None = None
    first_draw: torch.Tensor | None = None
    for draw_index in range(NUM_SAMPLING_STEPS + 1):
        draws, before, after = _draw_branch_tensors(
            generators, device=torch.device("cpu"), dtype=torch.float32
        )
        if draw_index == 0:
            first_draw = draws.clone()
        if previous_after is not None and not np.array_equal(previous_after, before):
            raise AssertionError("initial+250 branch-local toy stream is discontinuous")
        previous_after = after
    if first_draw is None or any(
        torch.equal(first_draw[0], first_draw[index])
        for index in range(1, BRANCHES_PER_SHARD)
    ):
        raise AssertionError("distinct full-trajectory seeds produced identical first draws")
    replay = torch.randn(
        (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
        generator=torch.Generator(device="cpu").manual_seed(seeds[0]),
    )
    if not torch.equal(first_draw[0:1], replay):
        raise AssertionError("branch-local initial latent is not reproducible")

    mirror_rng = np.random.default_rng(17)
    toy_initial = np.ascontiguousarray(
        mirror_rng.normal(size=(2, 1, 2, 2)), dtype=np.float32
    )
    toy_mean = np.ascontiguousarray(
        mirror_rng.normal(size=(2, 3, 1, 2, 2)), dtype=np.float32
    )
    toy_sigma = np.ascontiguousarray(
        np.exp(mirror_rng.normal(-2.0, 0.1, size=toy_mean.shape)), dtype=np.float32
    )
    toy_innovation = np.ascontiguousarray(
        mirror_rng.normal(size=toy_mean.shape), dtype=np.float32
    )
    toy_internal = np.asarray([2, 1, 0], dtype=np.int16)
    toy_states, toy_final = numpy_P_replay(
        toy_initial, toy_mean, toy_sigma, toy_innovation, toy_internal
    )
    require_numpy_P_replay_bitwise_match(
        reference_state_before=toy_states,
        reference_final_latents=toy_final,
        initial_latent=toy_initial,
        p_mean=toy_mean,
        p_standard_deviation=toy_sigma,
        innovation=toy_innovation,
        internal_timestep=toy_internal,
    )
    tampered_innovation = toy_innovation.copy()
    tampered_innovation[0, 0, 0, 0, 0] += np.float32(1.0)
    try:
        require_numpy_P_replay_bitwise_match(
            reference_state_before=toy_states,
            reference_final_latents=toy_final,
            initial_latent=toy_initial,
            p_mean=toy_mean,
            p_standard_deviation=toy_sigma,
            innovation=tampered_innovation,
            internal_timestep=toy_internal,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("tampered evidence-disabled P mirror was not rejected")
    from torchvision.utils import save_image

    toy_decoded = torch.from_numpy(
        np.ascontiguousarray(mirror_rng.uniform(-1.0, 1.0, size=(2, 3, 4, 4)), dtype=np.float32)
    )
    with tempfile.TemporaryDirectory(prefix="eqvae-mirror-png-selftest-") as temporary:
        toy_png = Path(temporary) / "toy.png"
        save_image(
            toy_decoded,
            toy_png,
            nrow=2,
            padding=2,
            normalize=True,
            value_range=(-1, 1),
        )
        if toy_png.read_bytes() != _encoded_save_image_png(
            toy_decoded, nrow=2, padding=2
        ):
            raise AssertionError("in-memory mirror PNG encoding differs from save_image")

    masks, bounds, base_names = _base_components()
    base_index, signs, component_names, weights = _signed_component_metadata()
    expected_names = [
        name
        for base in ["global", *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT))]
        for name in (f"+theta/{base}", f"-theta/{base}")
    ]
    if component_names.tolist() != expected_names:
        raise AssertionError("34-component sign/base ordering changed")
    if base_names.tolist() != [
        "global",
        *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT)),
    ]:
        raise AssertionError("base component ordering changed")
    if bounds.shape != (16, 4):
        raise AssertionError("row-major 4x4 tile geometry changed")
    if not np.array_equal(
        base_index,
        np.asarray([index for index in range(17) for _ in range(2)], dtype=np.int16),
    ) or not np.array_equal(
        signs, np.asarray([1, -1] * 17, dtype=np.int8)
    ):
        raise AssertionError("path-fixed sign blocks changed")
    if not np.array_equal(weights, np.full(34, 1.0 / 34.0, dtype=np.float64)):
        raise AssertionError("component weights are not fixed uniform 1/34")

    synthetic_alpha = np.exp(-np.linspace(0.0001, 8.0, NUM_SAMPLING_STEPS))
    schedule = _evidence_schedule(synthetic_alpha)
    if list(np.asarray(schedule["internal_timestep"]).astype(int)) != list(
        range(EVIDENCE_START_INTERNAL_TIMESTEP, -1, -1)
    ):
        raise AssertionError("synthetic evidence axis changed")
    effective_count = int(schedule["effective_step_count"])
    per_step_cap = float(schedule["per_step_K_cap"])
    if effective_count * per_step_cap > TOTAL_K_PER_COMPONENT:
        raise AssertionError("total component K allocation exceeds 0.5")

    rng = np.random.default_rng(23)
    theta = np.ascontiguousarray(
        rng.normal(size=(BRANCHES_PER_SHARD, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)),
        dtype=np.float64,
    )
    sigma = np.ascontiguousarray(
        np.exp(rng.normal(-2.0, 0.1, size=theta.shape)), dtype=np.float32
    )
    raw, scale, K, whitened = construct_signed_components_before_innovation(
        theta, sigma, masks, per_step_cap
    )
    if (
        raw.shape != (BRANCHES_PER_SHARD, SIGNED_COMPONENT_COUNT)
        or np.any(scale > 1.0)
        or np.any(K > per_step_cap * (1.0 + 2e-12))
        or np.any(K * effective_count > TOTAL_K_PER_COMPONENT * (1.0 + 2e-12))
    ):
        raise AssertionError("per-component K cap construction failed")
    if not np.array_equal(
        whitened[:, 0::2],
        -whitened[:, 1::2],
    ):
        raise AssertionError("+/- complete-path component pairing changed")
    innovation = np.ascontiguousarray(rng.normal(size=theta.shape), dtype=np.float32)
    reward, increment = evaluate_components_after_innovation(whitened, innovation)
    if not np.array_equal(increment, reward - K):
        raise AssertionError("Gaussian LR identity L=<u,z>-K failed")

    tile_only = np.zeros((1, 1, SIGNED_COMPONENT_COUNT), dtype=np.float64)
    tile_12_plus_index = 2 * (1 + 12)
    tile_only[0, 0, tile_12_plus_index] = ALARM_LOG_E + 0.1
    tile_summary = summarize_mixture(tile_only)
    if tile_summary["mixture_ever_alarm"].item() != 0:
        raise AssertionError("tile_12 was incorrectly promoted to the primary alarm")
    all_components = np.full(
        (1, 1, SIGNED_COMPONENT_COUNT), ALARM_LOG_E + 0.01, dtype=np.float64
    )
    if summarize_mixture(all_components)["mixture_ever_alarm"].item() != 1:
        raise AssertionError("the uniform 34-path mixture is not the primary alarm")
    spec = primary_spec()
    if spec.get("single_component_primary") is not False or "tile_12" in json.dumps(spec):
        raise AssertionError("primary specification contains a forbidden tile_12 binding")

    toy_u = np.zeros((1, SIGNED_COMPONENT_COUNT, 1, 1, 1), dtype=np.float64)
    toy_u[0, 0, 0, 0, 0] = 0.15
    toy_u[0, 1, 0, 0, 0] = -0.15
    e_values = []
    for _ in range(20_000):
        toy_z = np.ascontiguousarray(rng.normal(size=(1, 1, 1, 1)), dtype=np.float32)
        _, toy_L = evaluate_components_after_innovation(toy_u, toy_z)
        e_values.append(float(np.exp(_logmeanexp(toy_L, axis=1)[0])))
    if abs(float(np.mean(e_values)) - 1.0) > 0.01:
        raise AssertionError("fixed signed path-mixture e-value calibration failed")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: fixed 8x8=64 shard allocation, 64 unique streams, each "
        "toy stream's initial+250 draw chain, blind IDs, exact 34-component order and "
        "uniform path mixture, tile_12-primary prohibition, total K=0.5 cap, Gaussian "
        "LR identity/calibration, bitwise P mirror plus tamper rejection, direct-call "
        "DRAFT GPU-gate enforcement, frozen pre-sampling artifact enforcement, exact "
        "RTX-4090/runtime/backend contract rejection tests, and CPU-only execution"
    )


def dry_run(
    args: argparse.Namespace,
    *,
    protocol: dict[str, Any],
    source: dict[str, Any],
    vae: dict[str, Any],
    alpha: np.ndarray,
    timestep_map: np.ndarray,
) -> None:
    schedule = _evidence_schedule(alpha)
    all_seeds = [branch_stream_seed(index) for index in range(TOTAL_POOL_BRANCHES)]
    indices = shard_global_indices(args.shard_index)
    probe = checkpoint_dry_probe(args.checkpoint)
    payload = {
        "status": "dry-run",
        "experiment": EXPERIMENT,
        "gpu_model_loaded": False,
        "gpu_sampling_started": False,
        "protocol_status": protocol["protocol_status"],
        "protocol_gpu_execution_authorized": protocol["authorization_gate"][
            "gpu_execution_authorized"
        ],
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "frozen_execution_binding_candidate": frozen_execution_binding_candidate(
            source=source,
            vae=vae,
            alpha=alpha,
            timestep_map=timestep_map,
        ),
        "cross_prefix_validation": True,
        "pool": {
            "shards": TOTAL_SHARDS,
            "branches_per_shard": BRANCHES_PER_SHARD,
            "total_branches": TOTAL_POOL_BRANCHES,
            "all_64_stream_seeds_unique": len(set(all_seeds)) == TOTAL_POOL_BRANCHES,
            "this_shard_index": args.shard_index,
            "this_shard_global_indices": list(indices),
            "this_shard_blind_ids": [blind_id(index) for index in indices],
            "this_shard_stream_seeds": [all_seeds[index] for index in indices],
        },
        "baseline_P": {
            "class_id": TARGET_CLASS_ID,
            "independent_initial_latent_per_branch": True,
            "full_transition_axis": [NUM_SAMPLING_STEPS - 1, 0],
            "branch_local_draws": "initial latent plus 250 innovations including t0",
            "evidence_changes_state": False,
            "post_sampling_evidence_disabled_250_step_bitwise_P_mirror_required": True,
            "mirror_generator_draws": 0,
            "mirror_shifted_observer_calls": 0,
        },
        "primary": {
            **primary_spec(),
            "delta_nu": DELTA_NU,
            "window": [EVIDENCE_START_INTERNAL_TIMESTEP, 0],
            "total_K_per_component": TOTAL_K_PER_COMPONENT,
            "effective_stochastic_steps": schedule["effective_step_count"],
            "per_step_K_cap": schedule["per_step_K_cap"],
            "tile_12_or_any_single_tile_is_primary": False,
        },
        "CFG": {
            "first_half_target_branches": BATCH_SIZE,
            "full_conditional_null_model_batch": FULL_BATCH_SIZE,
            "official_8_to_16_shape_preserved": True,
            "cfg_scale": CFG_SCALE,
        },
        "cuda_execution_contract": _required_cuda_execution_contract(),
        "checkpoint_probe": probe,
        "source": source,
        "vae": vae,
        "schedule_hashes": {
            "alpha": _array_raw_sha256(np.ascontiguousarray(alpha, dtype=np.float64)),
            "timestep_map": _array_raw_sha256(
                np.ascontiguousarray(timestep_map, dtype=np.int64)
            ),
        },
        "static_inputs_ready": bool(
            probe["exists"] and probe["size_matches"] and probe["sha256_pinned"]
        ),
        "evidence_scores_computed_or_exposed": False,
        "outdir": str(args.outdir),
        "canonical_command": canonical_command(args),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_dit = data_root / "baselines/DiT"
    default_vae = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / VAE_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, choices=range(TOTAL_SHARDS), default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--dit-root", type=Path, default=default_dit)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vae-snapshot", type=Path, default=default_vae)
    parser.add_argument("--outdir", type=Path, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-bundle", type=Path, default=None)
    modes.add_argument(
        "--validate-pool",
        type=Path,
        nargs=TOTAL_SHARDS,
        metavar="SHARD_DIR",
        default=None,
    )
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.protocol = args.protocol.expanduser().absolute().resolve()
    if not args.protocol.is_file():
        parser.error(f"cross-prefix protocol JSON is missing: {args.protocol}")
    if args.self_test:
        return
    if args.validate_bundle is not None:
        args.validate_bundle = args.validate_bundle.expanduser().absolute().resolve()
        if not args.validate_bundle.is_dir():
            parser.error(f"validation bundle is not a directory: {args.validate_bundle}")
        return
    if args.validate_pool is not None:
        args.validate_pool = [path.expanduser().absolute().resolve() for path in args.validate_pool]
        missing = [path for path in args.validate_pool if not path.is_dir()]
        if missing:
            parser.error(f"validation pool contains missing shard directories: {missing}")
        return
    if args.device_index < 0:
        parser.error("--device-index must be nonnegative")
    data_root = Path(
        os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")
    ).expanduser().absolute().resolve()
    args.dit_root = args.dit_root.expanduser().absolute().resolve()
    args.checkpoint = (
        args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
        if args.checkpoint is None
        else args.checkpoint.expanduser().absolute().resolve()
    )
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    runner_sha = sha256_file(Path(__file__).resolve())[:7]
    default_out = (
        data_root
        / "cross_scale_evidence/dit_imagenet256_t60_cross_prefix_validation"
        / f"class0207_poolv1_shard{args.shard_index:02d}of08_{runner_sha}"
    )
    requested = (
        default_out.expanduser().absolute()
        if args.outdir is None
        else args.outdir.expanduser().absolute()
    )
    if os.path.lexists(requested):
        parser.error(f"no-overwrite target already exists: {requested}")
    args.outdir = requested.resolve()
    protected = (
        args.protocol,
        args.dit_root,
        args.checkpoint,
        args.vae_snapshot,
        RUNNER_DIR.parent,
    )
    for path in protected:
        if _paths_overlap(args.outdir, path):
            parser.error(f"--outdir overlaps protected input/source path: {path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)
    protocol = _load_protocol(args.protocol)
    if args.self_test:
        run_self_test(args.protocol)
        return 0
    if args.validate_bundle is not None:
        validate_output_bundle(args.validate_bundle)
        print(json.dumps({"status": "valid", "bundle": str(args.validate_bundle)}, indent=2))
        return 0
    if args.validate_pool is not None:
        print(json.dumps(validate_output_pool(args.validate_pool), indent=2, sort_keys=True))
        return 0
    if (
        not args.dry_run
        and (
            protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION"
            or protocol.get("authorization_gate", {}).get("gpu_execution_authorized")
            is not True
        )
    ):
        raise RuntimeError(
            "real GPU execution is intentionally locked by the draft cross-prefix "
            "protocol; use --dry-run or --self-test until it is explicitly frozen"
        )
    source = validate_repository(args.dit_root, args.checkpoint)
    vae = validate_vae_snapshot(args.vae_snapshot)
    alpha, timestep_map = load_schedule(args.dit_root)
    if args.dry_run:
        dry_run(
            args,
            protocol=protocol,
            source=source,
            vae=vae,
            alpha=alpha,
            timestep_map=timestep_map,
        )
    else:
        checkpoint = validate_checkpoint(args.checkpoint)
        run_real(
            args,
            protocol=protocol,
            source=source,
            checkpoint=checkpoint,
            vae=vae,
            alpha=alpha,
            timestep_map=timestep_map,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
