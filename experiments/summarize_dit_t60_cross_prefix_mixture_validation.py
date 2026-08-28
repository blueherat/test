#!/usr/bin/env python3
"""One-time aggregate-only readout for the 64-path cross-prefix validation.

The program first validates the reviewer-only blind pack and the closed
two-review/adjudicated consensus lock.  It then creates a fixed one-time claim
sentinel *before* touching the private blind mapping, shard results, or traces.
Only an in-memory annotation-lock capability can enter the private readout.

Primary output is the predeclared binary ``ever E_mix >= 5`` 2x2 table against
``overall_obvious_structural_bad_under_frozen_external_anchor_rubric``, exact
intervals, one-sided Fisher test, and the frozen decision tree.  Secondary
output is limited to tie-aware AUC/rank summaries for the same fixed mixture's
running maximum and terminal value, aggregate shard QA, and descriptive visual
label frequencies.  No per-image label/evidence join, blind ID, runner ID,
seed, trace row, score, rank, or alarm is emitted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import beta, fisher_exact, rankdata

try:
    from .build_dit_t60_cross_prefix_blind_pack import (
        AGGREGATE_SUMMARIZER,
        ANNOTATION_FIELDS,
        BINARY_TAIL_FIELDS,
        CONSENSUS_LOCKER,
        PRIVATE_DIR_NAME,
        PRIVATE_MAPPING_NAME,
        PROTOCOL_SOURCE,
        PUBLIC_DIR_NAME,
        PUBLIC_MANIFEST_NAME,
        RUNNER as BLIND_PACK_BUILDER,
        RUBRIC_NAME,
        MAPPING_COMMITMENT_SCHEMA,
        MAPPING_COMMITMENT_STATUS,
        TERNARY_TAIL_FIELDS,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        _canonical_self_hash,
        _is_sha256,
        _read_self_hashed_json,
        _reject_special_entries,
        _silence_process_output,
        _validate_protocol_for_pipeline,
        validate_output_bundle as validate_blind_bundle,
        validate_public_pack,
    )
    from .lock_dit_t60_cross_prefix_consensus import (
        CONSENSUS_NAME,
        validate_consensus_lock,
    )
    from .run_dit_t60_cross_prefix_mixture_validation_pool import (
        ALARM_LOG_E,
        ALPHA_E,
        BRANCHES_PER_SHARD,
        EXPERIMENT as SHARD_EXPERIMENT,
        PRIMARY_STATISTIC,
        TRACE_NAME,
        blind_id as runner_blind_id,
        shard_global_indices,
        validate_output_bundle as validate_shard_bundle,
        validate_output_pool as validate_shard_pool,
    )
    from .reproduce_dit_imagenet256 import (
        atomic_json_dump,
        load_json,
        sha256_file,
        sha256_json,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from build_dit_t60_cross_prefix_blind_pack import (
        AGGREGATE_SUMMARIZER,
        ANNOTATION_FIELDS,
        BINARY_TAIL_FIELDS,
        CONSENSUS_LOCKER,
        PRIVATE_DIR_NAME,
        PRIVATE_MAPPING_NAME,
        PROTOCOL_SOURCE,
        PUBLIC_DIR_NAME,
        PUBLIC_MANIFEST_NAME,
        RUNNER as BLIND_PACK_BUILDER,
        RUBRIC_NAME,
        MAPPING_COMMITMENT_SCHEMA,
        MAPPING_COMMITMENT_STATUS,
        TERNARY_TAIL_FIELDS,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        _canonical_self_hash,
        _is_sha256,
        _read_self_hashed_json,
        _reject_special_entries,
        _silence_process_output,
        _validate_protocol_for_pipeline,
        validate_output_bundle as validate_blind_bundle,
        validate_public_pack,
    )
    from lock_dit_t60_cross_prefix_consensus import (
        CONSENSUS_NAME,
        validate_consensus_lock,
    )
    from run_dit_t60_cross_prefix_mixture_validation_pool import (
        ALARM_LOG_E,
        ALPHA_E,
        BRANCHES_PER_SHARD,
        EXPERIMENT as SHARD_EXPERIMENT,
        PRIMARY_STATISTIC,
        TRACE_NAME,
        blind_id as runner_blind_id,
        shard_global_indices,
        validate_output_bundle as validate_shard_bundle,
        validate_output_pool as validate_shard_pool,
    )
    from reproduce_dit_imagenet256 import (
        atomic_json_dump,
        load_json,
        sha256_file,
        sha256_json,
    )


EXPERIMENT = "dit_imagenet256_t60_cross_prefix_mixture_validation_closed_summary"
SCHEMA_VERSION = 1
SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
CONFIDENCE_LEVEL = 0.95
RUNNER = Path(__file__).resolve()
SHARD_RUNNER = RUNNER.with_name(
    "run_dit_t60_cross_prefix_mixture_validation_pool.py"
)
UNSEAL_SENTINEL_SUFFIX = "aggregate_unseal_consumed.json"

PRIMARY_BAD = "clear_overall_structural_bad"
PRIMARY_NOT_BAD = "not_clear_overall_structural_bad"
PRIMARY_UNCERTAIN = "uncertain"
_LOCK_NONCE = object()

TRAJECTORY_SCALAR_FEATURES = (
    "log_E_mix_running_max",
    "log_E_mix_terminal",
    "log_E_mix_peak_to_terminal_drawdown",
    "log_E_mix_max_positive_one_step_jump",
    "log_E_mix_sum_positive_variation",
    "global_raw_K_running_max",
    "global_raw_K_max_positive_one_step_jump",
    "mean_tiles_raw_K_running_max",
    "mean_tiles_raw_K_max_positive_one_step_jump",
    "max_tile_concentration_running_max",
    "max_tile_concentration_max_positive_one_step_jump",
)


def frozen_trajectory_shape_panel_definition() -> dict[str, Any]:
    """Exact pre-GPU secondary feature panel; no data-dependent feature creation."""

    return {
        "status": "FROZEN_BEFORE_GPU_HYPOTHESIS_GENERATING_ONLY",
        "scope": (
            "Secondary trajectory-shape diagnostics only. They cannot rescue or alter "
            "the primary absolute-crossing decision and cannot authorize same-pool tuning."
        ),
        "time_axis": (
            "Use post-transition fixed-mixture log E values at internal t=60..0 in "
            "sampling order and prepend initial log E=0 at pseudo-t=61."
        ),
        "tie_and_zero_rule": (
            "Argmax uses the first chronological occurrence (numpy argmax on the stated "
            "sampling-order axis). A maximum-positive-jump time is the destination "
            "internal timestep of the first maximum; it is null iff every increment is <=0."
        ),
        "fixed_log_E_mix_features": {
            "log_E_mix_running_max": "max over the prepended initial 0 and the t60..0 path",
            "log_E_mix_terminal": "post-transition path value at internal t=0",
            "log_E_mix_peak_to_terminal_drawdown": "running_max minus terminal",
            "log_E_mix_max_positive_one_step_jump": (
                "max(0, each consecutive increment of [initial 0, t60..0 path])"
            ),
            "log_E_mix_sum_positive_variation": (
                "sum of max(increment,0) over [initial 0, t60..0 path]"
            ),
            "log_E_mix_argmax_internal_timestep": (
                "pseudo-t=61 for the initial value or internal t=60..0 for the first maximum"
            ),
            "log_E_mix_argmax_positive_jump_internal_timestep": (
                "destination t of the first largest positive increment, null if none"
            ),
        },
        "fixed_raw_K_reduction": {
            "signed_pair_rule": (
                "Require +theta/-theta raw_K equality for every base component, then use "
                "the fixed base order global,tile_00,...,tile_15 exactly once."
            ),
            "global_raw_K_track": "the global base-component raw_K at each t60..0 step",
            "mean_tiles_raw_K_track": "arithmetic mean of the sixteen tile raw_K values at each step",
            "max_tile_concentration_track": (
                "max(tile raw_K)/sum(tile raw_K) at each step; define 0 when the sum is 0"
            ),
            "features_per_track": (
                "running max including a prepended 0, and maximum positive one-step jump "
                "including the initial-to-t60 increment"
            ),
            "specific_tile_or_sign_identity_reported": False,
        },
        "readout": {
            "directional_scalars": (
                "For every fixed scalar, report clear-bad and not-clear-bad aggregate "
                "group summaries plus tie-aware AUC; no p-values or multiplicity claim."
            ),
            "timesteps": (
                "Report only aggregate frequency distributions by the three primary-label "
                "groups; do not compute a timestep significance test."
            ),
            "per_sample_values_emitted": False,
        },
    }


@dataclass(frozen=True)
class _AnnotationLockToken:
    _nonce: object
    consensus: dict[str, Any]
    consensus_identity_sha256: str
    consensus_file_sha256: str
    consensus_lock_manifest_identity_sha256: str


@dataclass(frozen=True)
class _JoinedRecord:
    primary_label: str
    hind_limb_label: str
    tail_identity: str
    tail_scorable: str
    tail_confidence: str
    tail_derived_label: str
    tail_values: tuple[int | None, ...]
    alarm: int
    running_max_log_e: float
    terminal_log_e: float
    shard_index: int
    trajectory_scalar_features: tuple[float, ...]
    log_e_argmax_internal_timestep: int
    log_e_argmax_positive_jump_internal_timestep: int | None


def _require_token(token: _AnnotationLockToken) -> None:
    if not isinstance(token, _AnnotationLockToken) or token._nonce is not _LOCK_NONCE:
        raise RuntimeError("private readout requires a validated annotation-lock token")


def _canonical_file_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _claim_sentinel_path(blind_bundle: Path, bundle_identity: str) -> Path:
    return blind_bundle.parent / (
        f".{blind_bundle.name}.{bundle_identity[:16]}.{UNSEAL_SENTINEL_SUFFIX}"
    )


def _claim_one_time_unseal(
    blind_bundle: Path,
    *,
    bundle_identity: str,
    protocol_identity: str,
    token: _AnnotationLockToken,
) -> tuple[Path, dict[str, Any]]:
    _require_token(token)
    path = _claim_sentinel_path(blind_bundle, bundle_identity)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "CLAIMED_BEFORE_ANY_PRIVATE_MAPPING_OR_EVIDENCE_ACCESS",
        "blind_bundle_identity_sha256": bundle_identity,
        "protocol_identity_sha256": protocol_identity,
        "consensus_annotation_identity_sha256": token.consensus_identity_sha256,
        "summarizer_sha256": sha256_file(RUNNER),
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    payload["claim_identity_sha256"] = _canonical_self_hash(
        payload, "claim_identity_sha256"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise RuntimeError(
            "the fixed one-time aggregate unseal was already claimed for this bundle"
        ) from None
    try:
        data = _canonical_file_json(payload)
        written = os.write(descriptor, data)
        if written != len(data):
            raise RuntimeError("short write while claiming one-time unseal")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.is_symlink() or sha256_file(path) != hashlib.sha256(data).hexdigest():
        raise RuntimeError("one-time unseal claim did not publish exactly")
    return path, payload


def _validate_protocol_before_unseal_claim(protocol: dict[str, Any]) -> None:
    """Validate frozen public declarations without opening private artifacts.

    In particular, this stage must never call ``validate_mapping_commitment``:
    that validator opens the private pre-GPU index-to-blind-ID commitment.  The
    complete protocol validation remains mandatory, but only after the atomic
    one-time unseal claim exists.
    """

    if protocol.get("protocol_identity_sha256") != _canonical_self_hash(
        protocol, "protocol_identity_sha256"
    ):
        raise RuntimeError("cross-prefix protocol self-hash changed")
    if protocol.get("protocol_name") != (
        "dit_imagenet256_t60_cross_prefix_mixture_validation_v1"
    ):
        raise RuntimeError("wrong cross-prefix protocol")
    if (
        protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION"
        or protocol.get("authorization_gate", {}).get("gpu_execution_authorized")
        is not True
    ):
        raise RuntimeError("cross-prefix protocol is not frozen and authorized")

    pipeline_paths = (
        BLIND_PACK_BUILDER,
        CONSENSUS_LOCKER,
        AGGREGATE_SUMMARIZER,
    )
    if any(not path.is_file() or path.is_symlink() for path in pipeline_paths):
        raise RuntimeError("a frozen blind-pipeline source is missing or indirect")
    if AGGREGATE_SUMMARIZER.resolve() != RUNNER:
        raise RuntimeError("aggregate summarizer source identity changed")
    expected_pipeline_binding = {
        "blind_pack_builder_filename": BLIND_PACK_BUILDER.name,
        "blind_pack_builder_sha256": sha256_file(BLIND_PACK_BUILDER),
        "consensus_locker_filename": CONSENSUS_LOCKER.name,
        "consensus_locker_sha256": sha256_file(CONSENSUS_LOCKER),
        "aggregate_unseal_summarizer_filename": RUNNER.name,
        "aggregate_unseal_summarizer_sha256": sha256_file(RUNNER),
    }
    if protocol.get("blind_pipeline_binding") != expected_pipeline_binding:
        raise RuntimeError("protocol blind-pipeline source binding changed")

    # Validate only the commitment declaration here.  Do not resolve, stat, or
    # read commitment_path before the one-time claim is on disk.
    commitment_binding = protocol.get("blind_mapping_commitment_binding")
    commitment_keys = {
        "status",
        "commitment_schema",
        "pool_size",
        "commitment_path",
        "mapping_builder_filename",
        "mapping_builder_sha256",
        "commitment_identity_sha256",
        "commitment_file_sha256",
    }
    raw_commitment_path = (
        commitment_binding.get("commitment_path")
        if isinstance(commitment_binding, dict)
        else None
    )
    if (
        not isinstance(commitment_binding, dict)
        or set(commitment_binding) != commitment_keys
        or commitment_binding.get("status") != MAPPING_COMMITMENT_STATUS
        or commitment_binding.get("commitment_schema")
        != MAPPING_COMMITMENT_SCHEMA
        or commitment_binding.get("pool_size") != TOTAL_POOL_BRANCHES
        or not isinstance(raw_commitment_path, str)
        or not Path(raw_commitment_path).is_absolute()
        or ".." in Path(raw_commitment_path).parts
        or commitment_binding.get("mapping_builder_filename")
        != BLIND_PACK_BUILDER.name
        or commitment_binding.get("mapping_builder_sha256")
        != sha256_file(BLIND_PACK_BUILDER)
        or not _is_sha256(
            commitment_binding.get("commitment_identity_sha256")
        )
        or not _is_sha256(commitment_binding.get("commitment_file_sha256"))
    ):
        raise RuntimeError("protocol blind-mapping commitment declaration changed")

    review = protocol.get("blind_review", {})
    primary = review.get("primary_visual_endpoint", {})
    if (
        primary.get("name")
        != "overall_obvious_structural_bad_under_frozen_external_anchor_rubric"
        or primary.get("role") != "sole primary visual endpoint"
        or primary.get("labels")
        != [
            "clear_overall_structural_bad",
            "not_clear_overall_structural_bad",
            "uncertain",
        ]
        or review.get("annotation_lock", {}).get("unseal_count") != 1
    ):
        raise RuntimeError("frozen blind-review declaration changed")
    pool = protocol.get("pool", {})
    if (
        pool.get("shard_count") != TOTAL_SHARDS
        or pool.get("class207_trajectories_per_shard") != BRANCHES_PER_SHARD
        or pool.get("total_class207_trajectories") != TOTAL_POOL_BRANCHES
    ):
        raise RuntimeError("protocol 8x8 pool changed")
    _validate_statistical_protocol(protocol)


def _validate_blind_bundle_envelope_before_unseal_claim(
    blind_bundle: Path, public_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Validate the non-secret top-level envelope without opening private files."""

    manifest_path = blind_bundle / "bundle_manifest.json"
    completion_path = blind_bundle / "bundle_completion.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or completion_path.is_symlink()
        or not completion_path.is_file()
    ):
        raise RuntimeError("blind-bundle public envelope is missing or indirect")
    manifest = _read_self_hashed_json(manifest_path, "identity_sha256")
    expected_manifest_keys = {
        "schema_version",
        "experiment",
        "role",
        "reviewer_delivery_relative_path",
        "private_seal_relative_path",
        "public_manifest_identity_sha256",
        "public_manifest_file_sha256",
        "public_completion_file_sha256",
        "private_mapping_identity_sha256",
        "private_mapping_file_sha256",
        "private_mapping_commitment_identity_sha256",
        "private_mapping_commitment_file_sha256",
        "private_completion_file_sha256",
        "identity_sha256",
    }
    public_root = blind_bundle / PUBLIC_DIR_NAME
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("experiment")
        != "dit_imagenet256_t60_cross_prefix_blind_review_bundle"
        or manifest.get("role") != "SEALED_PUBLIC_PRIVATE_PAIR"
        or manifest.get("reviewer_delivery_relative_path") != PUBLIC_DIR_NAME
        or manifest.get("private_seal_relative_path") != PRIVATE_DIR_NAME
        or manifest.get("public_manifest_identity_sha256")
        != public_manifest.get("identity_sha256")
        or manifest.get("public_manifest_file_sha256")
        != sha256_file(public_root / PUBLIC_MANIFEST_NAME)
        or manifest.get("public_completion_file_sha256")
        != sha256_file(public_root / "completion.json")
        or any(
            not _is_sha256(manifest.get(key))
            for key in (
                "private_mapping_identity_sha256",
                "private_mapping_file_sha256",
                "private_mapping_commitment_identity_sha256",
                "private_mapping_commitment_file_sha256",
                "private_completion_file_sha256",
            )
        )
    ):
        raise RuntimeError("blind-bundle public envelope manifest changed")
    completion = _read_self_hashed_json(completion_path, "payload_sha256")
    expected_completion = {
        "complete": True,
        "bundle_manifest_identity_sha256": manifest["identity_sha256"],
        "bundle_manifest_file_sha256": sha256_file(manifest_path),
        "public_manifest_identity_sha256": public_manifest["identity_sha256"],
        "private_mapping_identity_sha256": manifest[
            "private_mapping_identity_sha256"
        ],
        "private_mapping_commitment_identity_sha256": manifest[
            "private_mapping_commitment_identity_sha256"
        ],
        "payload_sha256": completion.get("payload_sha256"),
    }
    if completion != expected_completion:
        raise RuntimeError("blind-bundle public envelope completion changed")
    return manifest


def _require_exact_unseal_claim(
    blind_bundle: Path,
    *,
    bundle_identity: str,
    protocol: dict[str, Any],
    token: _AnnotationLockToken,
    claim_path: Path,
    claim_payload: dict[str, Any],
) -> None:
    """Require the exact claim capability before any complete/private validation."""

    _require_token(token)
    if claim_path != _claim_sentinel_path(blind_bundle, bundle_identity):
        raise RuntimeError("private validation received the wrong unseal-claim path")
    expected_keys = {
        "schema_version",
        "status",
        "blind_bundle_identity_sha256",
        "protocol_identity_sha256",
        "consensus_annotation_identity_sha256",
        "summarizer_sha256",
        "claimed_at_utc",
        "claim_identity_sha256",
    }
    if (
        not isinstance(claim_payload, dict)
        or set(claim_payload) != expected_keys
        or claim_payload.get("schema_version") != SCHEMA_VERSION
        or claim_payload.get("status")
        != "CLAIMED_BEFORE_ANY_PRIVATE_MAPPING_OR_EVIDENCE_ACCESS"
        or claim_payload.get("blind_bundle_identity_sha256") != bundle_identity
        or claim_payload.get("protocol_identity_sha256")
        != protocol.get("protocol_identity_sha256")
        or claim_payload.get("consensus_annotation_identity_sha256")
        != token.consensus_identity_sha256
        or claim_payload.get("summarizer_sha256") != sha256_file(RUNNER)
        or claim_payload.get("claim_identity_sha256")
        != _canonical_self_hash(claim_payload, "claim_identity_sha256")
    ):
        raise RuntimeError("private validation received an invalid unseal claim")
    expected_data = _canonical_file_json(claim_payload)
    if (
        claim_path.is_symlink()
        or not claim_path.is_file()
        or sha256_file(claim_path) != hashlib.sha256(expected_data).hexdigest()
        or claim_path.read_bytes() != expected_data
    ):
        raise RuntimeError("private validation requires the exact published unseal claim")


def _validate_private_protocol_after_claim(
    protocol: dict[str, Any],
    *,
    blind_bundle: Path,
    bundle_identity: str,
    token: _AnnotationLockToken,
    claim_path: Path,
    claim_payload: dict[str, Any],
) -> None:
    _require_exact_unseal_claim(
        blind_bundle,
        bundle_identity=bundle_identity,
        protocol=protocol,
        token=token,
        claim_path=claim_path,
        claim_payload=claim_payload,
    )
    # This complete validation opens the private pre-GPU mapping commitment and
    # therefore is deliberately unreachable until the exact claim exists.
    _validate_protocol_for_pipeline(protocol)


def _validate_statistical_protocol(protocol: dict[str, Any]) -> None:
    candidate = protocol.get("evidence_candidate", {})
    alarm = candidate.get("alarm_boundary", {})
    if (
        candidate.get("primary_evidence_endpoint")
        != (
            "The single predeclared primary evidence variable is the binary "
            "ever-crossing indicator of the fixed 34-way complete-path mixture at "
            "E_mix>=5. No tile, sign, component, terminal score, or alternate "
            "threshold is co-primary."
        )
        or alarm.get("alpha_e") != ALPHA_E
        or alarm.get("e_value_threshold") != 5.0
        or not math.isclose(
            float(alarm.get("log_e_threshold", math.nan)),
            ALARM_LOG_E,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("frozen binary fixed-mixture endpoint changed")
    plan = protocol.get("statistical_readout_after_blind_lock", {})
    expected_outcomes = [
        "frozen_threshold_failed_to_pass",
        "event_limited_inconclusive",
        "prospective_screen_survival",
        "prospective_screen_failed_retire_fixed_candidate",
    ]
    expected_conditions = [
        "N_clear_bad >= 5",
        "N_not_clear_bad >= 1",
        "TPR - FPR >= 0.20",
        "FPR <= 0.20",
        "TP >= 3 clear-bad crossings",
        "one-sided Fisher exact p-value <= 0.10 in the direction TPR > FPR",
    ]
    if (
        plan.get("terminal_outcomes_no_posthoc_branching") != expected_outcomes
        or plan.get("prospective_screen_survival", {}).get(
            "necessary_and_joint_conditions"
        )
        != expected_conditions
        or plan.get("analysis_denominators")
        != {
            "FPR": "FP / N_not_clear_bad among consensus non-uncertain rows",
            "TPR": "TP / N_clear_bad among consensus non-uncertain rows",
            "uncertain": "excluded from both denominators and reported separately",
        }
    ):
        raise RuntimeError("frozen statistical decision tree changed")
    if plan.get("secondary_trajectory_shape_panel") != (
        frozen_trajectory_shape_panel_definition()
    ):
        raise RuntimeError("frozen secondary trajectory-shape panel changed")
    prohibitions = plan.get("same_pool_prohibitions")
    if not isinstance(prohibitions, list) or not any(
        "no alternate threshold" in value for value in prohibitions
    ):
        raise RuntimeError("same-pool multiplicity prohibition changed")


def _stage_annotation_lock(
    consensus_lock: Path, blind_bundle: Path
) -> _AnnotationLockToken:
    # This function is deliberately called before private mapping/shard access.
    manifest, _, consensus = validate_consensus_lock(consensus_lock, blind_bundle)
    return _AnnotationLockToken(
        _nonce=_LOCK_NONCE,
        consensus=consensus,
        consensus_identity_sha256=consensus["annotation_identity_sha256"],
        consensus_file_sha256=sha256_file(consensus_lock / CONSENSUS_NAME),
        consensus_lock_manifest_identity_sha256=manifest["identity_sha256"],
    )


def _strict_shards_after_lock(
    token: _AnnotationLockToken, shard_roots: tuple[Path, ...]
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    _require_token(token)
    if len(shard_roots) != TOTAL_SHARDS or len(set(shard_roots)) != TOTAL_SHARDS:
        raise RuntimeError("aggregate unseal requires eight distinct shard directories")
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    try:
        with _silence_process_output():
            for root in shard_roots:
                records.append(validate_shard_bundle(root))
            pool = validate_shard_pool(shard_roots)
    except Exception as exc:
        del exc
        raise RuntimeError("a private shard/pool failed strict post-lock validation") from None
    if pool.get("status") != "valid-complete-pool":
        raise RuntimeError("private runner did not validate the complete pool")
    return tuple(records)


def _positive_path_features(
    track: np.ndarray,
) -> tuple[float, float]:
    if track.ndim != 1 or track.size != 61 or not np.all(np.isfinite(track)):
        raise RuntimeError("frozen trajectory track must be one finite t60..0 path")
    augmented = np.concatenate([np.zeros((1,), dtype=np.float64), track.astype(np.float64)])
    increments = np.diff(augmented)
    positive = np.maximum(increments, 0.0)
    return float(np.max(augmented)), float(np.max(positive))


def _trajectory_shape_features(
    path_log_e: np.ndarray,
    raw_k_signed: np.ndarray,
    internal_timestep: np.ndarray,
) -> tuple[tuple[float, ...], int, int | None]:
    if (
        path_log_e.shape != (61,)
        or raw_k_signed.shape != (61, 34)
        or internal_timestep.shape != (61,)
        or not np.array_equal(internal_timestep, np.arange(60, -1, -1))
        or not np.all(np.isfinite(path_log_e))
        or not np.all(np.isfinite(raw_k_signed))
        or np.any(raw_k_signed < 0.0)
    ):
        raise RuntimeError("frozen trajectory-shape source arrays changed")
    plus = raw_k_signed[:, 0::2]
    minus = raw_k_signed[:, 1::2]
    if plus.shape != (61, 17) or not np.array_equal(plus, minus):
        raise RuntimeError("raw_K +/- base-component pairing changed")
    augmented_log = np.concatenate(
        [np.zeros((1,), dtype=np.float64), path_log_e.astype(np.float64)]
    )
    increments = np.diff(augmented_log)
    positive = np.maximum(increments, 0.0)
    running_max = float(np.max(augmented_log))
    terminal = float(path_log_e[-1])
    drawdown = running_max - terminal
    max_jump = float(np.max(positive))
    sum_positive = float(np.sum(positive, dtype=np.float64))
    augmented_times = np.concatenate(
        [np.asarray([61], dtype=np.int64), internal_timestep.astype(np.int64)]
    )
    argmax_t = int(augmented_times[int(np.argmax(augmented_log))])
    argmax_jump_t = (
        None
        if max_jump <= 0.0
        else int(internal_timestep[int(np.argmax(positive))])
    )

    global_track = plus[:, 0]
    tiles = plus[:, 1:]
    mean_tiles = np.mean(tiles, axis=1, dtype=np.float64)
    tile_sum = np.sum(tiles, axis=1, dtype=np.float64)
    concentration = np.divide(
        np.max(tiles, axis=1),
        tile_sum,
        out=np.zeros_like(tile_sum, dtype=np.float64),
        where=tile_sum > 0.0,
    )
    global_max, global_jump = _positive_path_features(global_track)
    mean_max, mean_jump = _positive_path_features(mean_tiles)
    concentration_max, concentration_jump = _positive_path_features(concentration)
    features = (
        running_max,
        terminal,
        drawdown,
        max_jump,
        sum_positive,
        global_max,
        global_jump,
        mean_max,
        mean_jump,
        concentration_max,
        concentration_jump,
    )
    if len(features) != len(TRAJECTORY_SCALAR_FEATURES) or not all(
        math.isfinite(value) for value in features
    ):
        raise RuntimeError("frozen trajectory-shape feature vector changed")
    return features, argmax_t, argmax_jump_t


def _load_private_join_after_lock(
    token: _AnnotationLockToken,
    blind_bundle: Path,
    shard_roots: tuple[Path, ...],
) -> tuple[list[_JoinedRecord], dict[str, Any], tuple[dict[str, Any], ...]]:
    _require_token(token)
    # Full blind-bundle validation is intentionally delayed until the lock token
    # exists because it opens the private mapping.
    validate_blind_bundle(blind_bundle, require_protocol=True)
    mapping = _read_self_hashed_json(
        blind_bundle / PRIVATE_DIR_NAME / PRIVATE_MAPPING_NAME,
        "identity_sha256",
    )
    validated = _strict_shards_after_lock(token, shard_roots)
    by_shard: dict[int, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for root, (manifest, results) in zip(shard_roots, validated):
        shard_index = manifest.get("pool", {}).get("this_shard_index")
        if type(shard_index) is not int or shard_index in by_shard:
            raise RuntimeError("private shard identity/allocation changed")
        by_shard[shard_index] = (root, manifest, results)
    if set(by_shard) != set(range(TOTAL_SHARDS)):
        raise RuntimeError("private shards do not cover indices 0..7")
    for expected, actual in zip(mapping["shards"], range(TOTAL_SHARDS)):
        root, manifest, results = by_shard[actual]
        observed = {
            "shard_index": actual,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(root / "manifest.json"),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": sha256_file(root / "results.json"),
            "trace_file_sha256": results["private_trace"]["sha256"],
            "completion_file_sha256": sha256_file(root / "completion.json"),
        }
        if expected != observed:
            raise RuntimeError("private shard differs from blind-pack mapping seal")

    actual_branch_records: dict[int, dict[str, Any]] = {}
    for shard_index in range(TOTAL_SHARDS):
        _, _, results = by_shard[shard_index]
        records = results.get("branch_records")
        if not isinstance(records, list) or len(records) != BRANCHES_PER_SHARD:
            raise RuntimeError("private shard branch-record count changed")
        for local_index, record in enumerate(records):
            global_index = shard_index * BRANCHES_PER_SHARD + local_index
            if (
                not isinstance(record, dict)
                or record.get("global_index") != global_index
                or record.get("local_index") != local_index
                or record.get("blind_id") != runner_blind_id(global_index)
                or not isinstance(record.get("image"), dict)
            ):
                raise RuntimeError("private shard branch-record identity changed")
            actual_branch_records[global_index] = record

    evidence_by_global: dict[
        int, tuple[int, float, float, int, tuple[float, ...], int, int | None]
    ] = {}
    shard_bindings: list[dict[str, Any]] = []
    for shard_index in range(TOTAL_SHARDS):
        root, manifest, results = by_shard[shard_index]
        trace_record = results["private_trace"]
        trace_path = root / TRACE_NAME
        if (
            trace_path.is_symlink()
            or not trace_path.is_file()
            or sha256_file(trace_path) != trace_record["sha256"]
        ):
            raise RuntimeError("private trace file binding changed")
        with np.load(trace_path, allow_pickle=False) as archive:
            required = {
                "branch_global_index",
                "mixture_ever_alarm",
                "mixture_terminal_log_e",
                "mixture_running_max_log_e",
                "mixture_path_log_e",
                "component_raw_K",
                "evidence_internal_timestep",
            }
            if not required.issubset(archive.files):
                raise RuntimeError("private trace lacks a frozen mixture readout")
            indices = np.array(archive["branch_global_index"], copy=True)
            alarms = np.array(archive["mixture_ever_alarm"], copy=True)
            terminal = np.array(archive["mixture_terminal_log_e"], copy=True)
            running = np.array(archive["mixture_running_max_log_e"], copy=True)
            paths = np.array(archive["mixture_path_log_e"], copy=True)
            raw_k = np.array(archive["component_raw_K"], copy=True)
            evidence_internal = np.array(
                archive["evidence_internal_timestep"], copy=True
            )
        expected_indices = np.asarray(shard_global_indices(shard_index), dtype=np.int16)
        if (
            indices.shape != (BRANCHES_PER_SHARD,)
            or not np.array_equal(indices, expected_indices)
            or alarms.shape != (BRANCHES_PER_SHARD,)
            or terminal.shape != (BRANCHES_PER_SHARD,)
            or running.shape != (BRANCHES_PER_SHARD,)
            or paths.shape != (BRANCHES_PER_SHARD, 61)
            or raw_k.shape != (BRANCHES_PER_SHARD, 61, 34)
            or evidence_internal.shape != (61,)
            or not np.all(np.isfinite(terminal))
            or not np.all(np.isfinite(running))
            or np.any((alarms != 0) & (alarms != 1))
            or np.any(running < -1e-15)
            or np.any(running + 1e-15 < terminal)
            or not np.array_equal(alarms.astype(bool), running >= ALARM_LOG_E)
        ):
            raise RuntimeError("private fixed-mixture readout failed exact reconstruction")
        for local_index, global_index in enumerate(expected_indices.tolist()):
            features, argmax_t, argmax_jump_t = _trajectory_shape_features(
                paths[local_index], raw_k[local_index], evidence_internal
            )
            if (
                features[0] != float(running[local_index])
                or features[1] != float(terminal[local_index])
            ):
                raise RuntimeError(
                    "trajectory-shape panel does not reconstruct saved mixture summaries"
                )
            evidence_by_global[int(global_index)] = (
                int(alarms[local_index]),
                float(running[local_index]),
                float(terminal[local_index]),
                shard_index,
                features,
                argmax_t,
                argmax_jump_t,
            )
        shard_bindings.append(
            {
                "shard_index": shard_index,
                "manifest_identity_sha256": manifest["identity_sha256"],
                "results_payload_sha256": results["payload_sha256"],
                "trace_file_sha256": trace_record["sha256"],
            }
        )
    if set(evidence_by_global) != set(range(TOTAL_POOL_BRANCHES)):
        raise RuntimeError("private evidence readout does not cover indices 0..63")

    consensus_by_id = {
        row["blind_id"]: row for row in token.consensus["rows"]
    }
    entries = mapping["entries"]
    if set(consensus_by_id) != {
        entry["public_blind_id"] for entry in entries
    }:
        raise RuntimeError("consensus/private mapping ID coverage changed")
    joined: list[_JoinedRecord] = []
    for entry in entries:
        identifier = entry["public_blind_id"]
        global_index = int(entry["global_index"])
        if entry["runner_blind_id"] != runner_blind_id(global_index):
            raise RuntimeError("private runner blind-ID binding changed")
        actual_branch = actual_branch_records[global_index]
        if (
            entry["shard_index"] != global_index // BRANCHES_PER_SHARD
            or entry["local_index"] != global_index % BRANCHES_PER_SHARD
            or entry["source_png_sha256"] != actual_branch["image"]["sha256"]
            or entry["source_pixel_sha256"]
            != actual_branch["image"]["pixel_sha256"]
        ):
            raise RuntimeError("private blind mapping differs from validated shard endpoint")
        row = consensus_by_id[identifier]
        (
            alarm,
            running,
            terminal,
            shard_index,
            features,
            argmax_t,
            argmax_jump_t,
        ) = evidence_by_global[global_index]
        joined.append(
            _JoinedRecord(
                primary_label=row["primary_overall_structural_quality"],
                hind_limb_label=row["secondary_hind_limb_topology"],
                tail_identity=row["tail_identity"],
                tail_scorable=row["tail_scorable"],
                tail_confidence=row["tail_confidence"],
                tail_derived_label=row["tail_derived_label"],
                tail_values=tuple(
                    row[field] for field in TERNARY_TAIL_FIELDS + BINARY_TAIL_FIELDS
                ),
                alarm=alarm,
                running_max_log_e=running,
                terminal_log_e=terminal,
                shard_index=shard_index,
                trajectory_scalar_features=features,
                log_e_argmax_internal_timestep=argmax_t,
                log_e_argmax_positive_jump_internal_timestep=argmax_jump_t,
            )
        )
    if len(joined) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("private label/evidence join does not contain exactly 64 rows")
    return joined, mapping, tuple(shard_bindings)


def _clopper_pearson(successes: int, total: int) -> dict[str, float] | None:
    if total == 0:
        return None
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return {"confidence_level": CONFIDENCE_LEVEL, "lower": lower, "upper": upper}


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "total": total,
        "estimate": None if total == 0 else successes / total,
        "clopper_pearson_exact_interval": _clopper_pearson(successes, total),
    }


def _auc_and_ranks(
    positive_scores: np.ndarray, negative_scores: np.ndarray
) -> dict[str, Any]:
    n_positive = int(positive_scores.size)
    n_negative = int(negative_scores.size)
    if n_positive == 0 or n_negative == 0:
        return {
            "available": False,
            "reason": "both non-uncertain quality classes are required",
            "N_clear_bad": n_positive,
            "N_not_clear_bad": n_negative,
            "auc": None,
            "pairwise_wins_with_ties_counted_as_half": None,
            "pairwise_comparisons": n_positive * n_negative,
            "tie_aware_descending_rank": None,
        }
    combined = np.concatenate([positive_scores, negative_scores])
    ascending = rankdata(combined, method="average")
    rank_sum_positive = float(np.sum(ascending[:n_positive]))
    wins = rank_sum_positive - n_positive * (n_positive + 1) / 2
    auc = wins / (n_positive * n_negative)
    descending = rankdata(-combined, method="average")
    return {
        "available": True,
        "N_clear_bad": n_positive,
        "N_not_clear_bad": n_negative,
        "auc": float(auc),
        "pairwise_wins_with_ties_counted_as_half": float(wins),
        "pairwise_comparisons": n_positive * n_negative,
        "tie_aware_descending_rank": {
            "mean_clear_bad": float(np.mean(descending[:n_positive])),
            "mean_not_clear_bad": float(np.mean(descending[n_positive:])),
            "median_clear_bad": float(np.median(descending[:n_positive])),
            "median_not_clear_bad": float(np.median(descending[n_positive:])),
        },
    }


def _scalar_group_summary(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise RuntimeError("trajectory scalar group must be one finite vector")
    if values.size == 0:
        return {
            "N": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "N": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _timestep_frequency(values: Iterable[int | None]) -> dict[str, int]:
    materialized = list(values)
    allowed = [61, *range(60, -1, -1), None]
    if any(value not in allowed for value in materialized):
        raise RuntimeError("trajectory-shape timestep lies outside the frozen axis")
    counts: dict[str, int] = {}
    for value in allowed:
        count = materialized.count(value)
        if count:
            counts["null" if value is None else str(value)] = count
    return counts


def _counts(values: Iterable[Any], categories: Sequence[Any]) -> dict[str, int]:
    materialized = list(values)
    return {str(category): materialized.count(category) for category in categories}


def _frozen_decision(
    *,
    total_crossings: int,
    n_bad: int,
    n_not_bad: int,
    tp: int,
    tpr: float | None,
    fpr: float | None,
    fisher_p: float | None,
) -> dict[str, Any]:
    difference = None if tpr is None or fpr is None else tpr - fpr
    conditions = {
        "N_clear_bad_at_least_5": n_bad >= 5,
        "N_not_clear_bad_at_least_1": n_not_bad >= 1,
        "TPR_minus_FPR_at_least_0_20": (
            difference is not None and difference >= 0.20
        ),
        "FPR_at_most_0_20": fpr is not None and fpr <= 0.20,
        "TP_at_least_3": tp >= 3,
        "one_sided_Fisher_p_at_most_0_10": (
            fisher_p is not None and fisher_p <= 0.10
        ),
    }
    if total_crossings == 0:
        outcome = "frozen_threshold_failed_to_pass"
    elif n_bad < 5:
        outcome = "event_limited_inconclusive"
    elif all(conditions.values()):
        outcome = "prospective_screen_survival"
    else:
        outcome = "prospective_screen_failed_retire_fixed_candidate"
    return {
        "outcome": outcome,
        "event_limited": n_bad < 5,
        "joint_screen_conditions": conditions,
        "all_joint_screen_conditions_met": all(conditions.values()),
        "interpretation": (
            "Screen survival only permits a larger independently predeclared "
            "confirmation pool; it is not final validation or a quality-improvement claim."
            if outcome == "prospective_screen_survival"
            else "This outcome cannot be rescued by secondary AUC, a component, a shard, or relabeling."
        ),
    }


def summarize_joined(records: Sequence[_JoinedRecord]) -> dict[str, Any]:
    if len(records) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("summary requires exactly 64 joined records")
    labels = [record.primary_label for record in records]
    alarms = np.asarray([record.alarm for record in records], dtype=np.int8)
    bad_mask = np.asarray([label == PRIMARY_BAD for label in labels], dtype=bool)
    not_mask = np.asarray([label == PRIMARY_NOT_BAD for label in labels], dtype=bool)
    uncertain_mask = np.asarray(
        [label == PRIMARY_UNCERTAIN for label in labels], dtype=bool
    )
    if not np.all(bad_mask | not_mask | uncertain_mask):
        raise RuntimeError("joined records contain an unknown primary label")
    n_bad = int(np.sum(bad_mask))
    n_not = int(np.sum(not_mask))
    n_uncertain = int(np.sum(uncertain_mask))
    tp = int(np.sum(alarms[bad_mask]))
    fp = int(np.sum(alarms[not_mask]))
    uncertain_crossings = int(np.sum(alarms[uncertain_mask]))
    fn = n_bad - tp
    tn = n_not - fp
    total_crossings = int(np.sum(alarms))
    tpr = None if n_bad == 0 else tp / n_bad
    fpr = None if n_not == 0 else fp / n_not
    difference = None if tpr is None or fpr is None else tpr - fpr
    if n_bad == 0 or n_not == 0:
        odds_ratio = None
        fisher_p = None
    else:
        fisher = fisher_exact([[tp, fn], [fp, tn]], alternative="greater")
        odds_ratio = float(fisher.statistic)
        if math.isinf(odds_ratio):
            odds_ratio = "infinity"
        fisher_p = float(fisher.pvalue)
    decision = _frozen_decision(
        total_crossings=total_crossings,
        n_bad=n_bad,
        n_not_bad=n_not,
        tp=tp,
        tpr=tpr,
        fpr=fpr,
        fisher_p=fisher_p,
    )

    running = np.asarray([record.running_max_log_e for record in records])
    terminal = np.asarray([record.terminal_log_e for record in records])
    secondary_scores: dict[str, Any] = {}
    for name, values in (
        ("running_max_log_E_mix_including_initial_1", running),
        ("terminal_log_E_mix", terminal),
    ):
        secondary_scores[name] = _auc_and_ranks(values[bad_mask], values[not_mask])

    hind_categories = (
        "clear_failure",
        "not_clear_failure",
        "uncertain_or_not_scorable",
    )
    tail_identity_categories = ("clear", "plausible", "unclear")
    tail_scorable_categories = ("yes", "no")
    tail_confidence_categories = ("high", "medium", "low")
    tail_derived_categories = (
        "natural",
        "odd",
        "malformed",
        "uncertain_or_not_scorable",
    )
    tail_by_field: dict[str, dict[str, int]] = {}
    all_tail_fields = TERNARY_TAIL_FIELDS + BINARY_TAIL_FIELDS
    for field_index, field in enumerate(all_tail_fields):
        categories: Sequence[Any] = (
            (0, 1, 2, None) if field in TERNARY_TAIL_FIELDS else (0, 1, None)
        )
        tail_by_field[field] = _counts(
            (record.tail_values[field_index] for record in records), categories
        )
    shard_crossings = [
        {
            "shard_index": shard,
            "trajectory_count": sum(
                record.shard_index == shard for record in records
            ),
            "crossing_count": sum(
                record.shard_index == shard and record.alarm == 1
                for record in records
            ),
        }
        for shard in range(TOTAL_SHARDS)
    ]
    feature_matrix = np.asarray(
        [record.trajectory_scalar_features for record in records], dtype=np.float64
    )
    if feature_matrix.shape != (
        TOTAL_POOL_BRANCHES,
        len(TRAJECTORY_SCALAR_FEATURES),
    ) or not np.all(np.isfinite(feature_matrix)):
        raise RuntimeError("frozen trajectory-shape feature matrix changed")
    scalar_panel: dict[str, Any] = {}
    for feature_index, feature_name in enumerate(TRAJECTORY_SCALAR_FEATURES):
        values = feature_matrix[:, feature_index]
        scalar_panel[feature_name] = {
            "clear_bad_group_summary": _scalar_group_summary(values[bad_mask]),
            "not_clear_bad_group_summary": _scalar_group_summary(values[not_mask]),
            "tie_aware_AUC_clear_bad_higher": _auc_and_ranks(
                values[bad_mask], values[not_mask]
            ),
            "p_value_computed": False,
        }

    def timestep_groups(attribute: str) -> dict[str, dict[str, int]]:
        return {
            PRIMARY_BAD: _timestep_frequency(
                getattr(record, attribute)
                for record in records
                if record.primary_label == PRIMARY_BAD
            ),
            PRIMARY_NOT_BAD: _timestep_frequency(
                getattr(record, attribute)
                for record in records
                if record.primary_label == PRIMARY_NOT_BAD
            ),
            PRIMARY_UNCERTAIN: _timestep_frequency(
                getattr(record, attribute)
                for record in records
                if record.primary_label == PRIMARY_UNCERTAIN
            ),
        }

    trajectory_shape_panel = {
        "role": (
            "Frozen pre-GPU hypothesis-generating secondary panel only; no feature "
            "can rescue primary, trigger same-pool tuning, or support a multiplicity claim."
        ),
        "definition": frozen_trajectory_shape_panel_definition(),
        "directional_scalar_aggregate_readout": scalar_panel,
        "timestep_group_frequency_only": {
            "log_E_mix_argmax_internal_timestep": timestep_groups(
                "log_e_argmax_internal_timestep"
            ),
            "log_E_mix_argmax_positive_jump_internal_timestep": timestep_groups(
                "log_e_argmax_positive_jump_internal_timestep"
            ),
            "significance_test_performed": False,
        },
        "specific_tile_or_sign_reported": False,
        "per_sample_values_emitted": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "scope": (
            "One frozen 64-trajectory class-207 cross-prefix pool; aggregate-only "
            "readout with consensus-uncertain primary rows excluded from TPR/FPR."
        ),
        "primary": {
            "quality_endpoint": (
                "overall_obvious_structural_bad_under_frozen_external_anchor_rubric"
            ),
            "evidence_endpoint": {
                "statistic": PRIMARY_STATISTIC,
                "event": "ever E_mix >= 5",
                "threshold": 5.0,
                "alpha_e": ALPHA_E,
            },
            "counts_reported_before_interpretation": {
                "N_clear_bad": n_bad,
                "N_not_clear_bad": n_not,
                "N_uncertain": n_uncertain,
                "total_crossings_all_64": total_crossings,
                "clear_bad_crossings_TP": tp,
                "clear_bad_non_crossings_FN": fn,
                "not_clear_bad_crossings_FP": fp,
                "not_clear_bad_non_crossings_TN": tn,
                "uncertain_crossings_reported_not_analyzed": uncertain_crossings,
                "uncertain_non_crossings_reported_not_analyzed": (
                    n_uncertain - uncertain_crossings
                ),
            },
            "TPR": _rate(tp, n_bad),
            "FPR": _rate(fp, n_not),
            "TPR_minus_FPR": difference,
            "total_crossing_rate_all_64": _rate(total_crossings, TOTAL_POOL_BRANCHES),
            "fisher_exact_one_sided_TPR_greater_than_FPR": {
                "table_rows": ["clear_bad", "not_clear_bad"],
                "table_columns": ["crossing", "not_crossing"],
                "odds_ratio": odds_ratio,
                "p_value": fisher_p,
                "alternative": "greater",
                "performed": fisher_p is not None,
            },
            "frozen_decision_tree": decision,
        },
        "secondary_same_fixed_mixture_only": {
            "role": (
                "Secondary ranking diagnostics only; neither can rescue the absolute "
                "crossing gate or select a new threshold/component/subset."
            ),
            **secondary_scores,
        },
        "secondary_frozen_trajectory_shape_panel": trajectory_shape_panel,
        "secondary_visual_descriptive_only": {
            "primary_label_frequency_including_uncertain": _counts(
                labels, (PRIMARY_BAD, PRIMARY_NOT_BAD, PRIMARY_UNCERTAIN)
            ),
            "hind_limb_topology": _counts(
                (record.hind_limb_label for record in records), hind_categories
            ),
            "tail_identity": _counts(
                (record.tail_identity for record in records), tail_identity_categories
            ),
            "tail_scorable": _counts(
                (record.tail_scorable for record in records), tail_scorable_categories
            ),
            "tail_confidence": _counts(
                (record.tail_confidence for record in records), tail_confidence_categories
            ),
            "tail_derived_label": _counts(
                (record.tail_derived_label for record in records),
                tail_derived_categories,
            ),
            "tail_R_T_F_D_P_B_S_separate": tail_by_field,
            "tail_identity_is_not_naturalness": True,
            "no_tail_or_hind_limb_association_test_performed": True,
        },
        "execution_shard_QA_only": {
            "role": "mechanical corruption/implementation QA, not eight replications",
            "crossings_by_shard": shard_crossings,
        },
        "multiplicity_guard": {
            "alternate_threshold_scanned": False,
            "component_or_tile_scanned": False,
            "sign_scanned": False,
            "start_time_scanned": False,
            "quality_endpoint_redefined": False,
            "uncertain_rows_forced_binary": False,
            "secondary_used_to_change_decision": False,
            "per_sample_output_emitted": False,
        },
    }


def _assert_aggregate_only(
    payload: dict[str, Any],
    *,
    public_ids: Iterable[str],
    runner_ids: Iterable[str],
) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    forbidden_fragments = (
        "branch_global_index",
        "branch_stream_seed",
        "public_blind_id",
        "runner_blind_id",
        "source_png",
        "source_pixel",
        "mixture_first_alarm",
        "component_log_e",
        "per_sample",
    )
    lower = serialized.lower()
    # The literal boolean declaration `per_sample_output_emitted=false` is safe;
    # reject identifiers/fields rather than that approved aggregate statement.
    for fragment in forbidden_fragments[:-1]:
        if fragment in lower:
            raise RuntimeError("aggregate summary contains a forbidden per-sample field")
    for identifier in (*tuple(public_ids), *tuple(runner_ids)):
        if identifier in serialized:
            raise RuntimeError("aggregate summary leaks an individual identifier")


def _manifest_for_summary(
    *,
    protocol: dict[str, Any],
    token: _AnnotationLockToken,
    blind_bundle_manifest: dict[str, Any],
    public_manifest: dict[str, Any],
    mapping: dict[str, Any],
    shard_bindings: tuple[dict[str, Any], ...],
    claim_path: Path,
    claim_payload: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "protocol_file_sha256": sha256_file(PROTOCOL_SOURCE),
        "pipeline_sources": {
            "blind_pack_builder_sha256": sha256_file(BLIND_PACK_BUILDER),
            "consensus_locker_sha256": sha256_file(CONSENSUS_LOCKER),
            "aggregate_unseal_summarizer_sha256": sha256_file(RUNNER),
            "sampling_runner_sha256": sha256_file(SHARD_RUNNER),
        },
        "blind_bundle_identity_sha256": blind_bundle_manifest["identity_sha256"],
        "public_blind_pack_manifest_identity_sha256": public_manifest[
            "identity_sha256"
        ],
        "private_mapping_identity_sha256": mapping["identity_sha256"],
        "blind_mapping_commitment_identity_sha256": mapping[
            "blind_mapping_commitment_identity_sha256"
        ],
        "blind_mapping_commitment_file_sha256": mapping[
            "blind_mapping_commitment_file_sha256"
        ],
        "consensus_lock_manifest_identity_sha256": (
            token.consensus_lock_manifest_identity_sha256
        ),
        "consensus_annotation_identity_sha256": token.consensus_identity_sha256,
        "consensus_file_sha256": token.consensus_file_sha256,
        "external_visual_anchor_binding": mapping[
            "external_visual_anchor_binding"
        ],
        "shards": list(shard_bindings),
        "one_time_unseal_claim": {
            "fixed_sibling_filename": claim_path.name,
            "claim_identity_sha256": claim_payload["claim_identity_sha256"],
            "claim_file_sha256": sha256_file(claim_path),
            "claimed_before_private_access": True,
            "unseal_count": 1,
        },
        "summary_payload_sha256": summary["payload_sha256"],
        "summary_file_sha256": None,
        "no_per_sample_output": True,
    }


def _write_summary_bundle(
    outdir: Path,
    *,
    protocol: dict[str, Any],
    token: _AnnotationLockToken,
    blind_bundle_manifest: dict[str, Any],
    public_manifest: dict[str, Any],
    mapping: dict[str, Any],
    shard_bindings: tuple[dict[str, Any], ...],
    claim_path: Path,
    claim_payload: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    if os.path.lexists(outdir):
        raise RuntimeError("refusing to overwrite an existing aggregate summary")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary) / "summary"
        staging.mkdir()
        atomic_json_dump(summary, staging / SUMMARY_NAME)
        manifest = _manifest_for_summary(
            protocol=protocol,
            token=token,
            blind_bundle_manifest=blind_bundle_manifest,
            public_manifest=public_manifest,
            mapping=mapping,
            shard_bindings=shard_bindings,
            claim_path=claim_path,
            claim_payload=claim_payload,
            summary=summary,
        )
        manifest["summary_file_sha256"] = sha256_file(staging / SUMMARY_NAME)
        manifest["identity_sha256"] = _canonical_self_hash(
            manifest, "identity_sha256"
        )
        atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / MANIFEST_NAME),
            "summary_payload_sha256": summary["payload_sha256"],
            "summary_file_sha256": sha256_file(staging / SUMMARY_NAME),
            "consensus_annotation_identity_sha256": token.consensus_identity_sha256,
            "one_time_unseal_claim_identity_sha256": claim_payload[
                "claim_identity_sha256"
            ],
            "aggregate_only": True,
        }
        completion["payload_sha256"] = _canonical_self_hash(
            completion, "payload_sha256"
        )
        atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_summary_bundle(staging)
        # Reuse Linux atomic no-replace through the blind builder.
        try:
            from .build_dit_t60_cross_prefix_blind_pack import (
                _atomic_install_directory_noreplace,
            )
        except ImportError:  # pragma: no cover
            from build_dit_t60_cross_prefix_blind_pack import (
                _atomic_install_directory_noreplace,
            )
        _atomic_install_directory_noreplace(staging, outdir)
    validate_summary_bundle(outdir)


def validate_summary_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _reject_special_entries(root)
    summary = _read_self_hashed_json(root / SUMMARY_NAME, "payload_sha256")
    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "identity_sha256")
    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    expected_summary_keys = {
        "schema_version",
        "experiment",
        "scope",
        "primary",
        "secondary_same_fixed_mixture_only",
        "secondary_frozen_trajectory_shape_panel",
        "secondary_visual_descriptive_only",
        "execution_shard_QA_only",
        "multiplicity_guard",
        "annotation_and_binding",
        "payload_sha256",
    }
    if set(summary) != expected_summary_keys:
        raise RuntimeError("aggregate summary top-level schema changed")
    _assert_aggregate_only(summary, public_ids=(), runner_ids=())

    def reject_row_sized_sequence(value: Any, context: str) -> None:
        if isinstance(value, list):
            if len(value) == TOTAL_POOL_BRANCHES:
                raise RuntimeError(
                    f"aggregate summary contains a row-sized sequence at {context}"
                )
            for index, item in enumerate(value):
                reject_row_sized_sequence(item, f"{context}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                reject_row_sized_sequence(item, f"{context}.{key}")

    reject_row_sized_sequence(summary, "summary")
    expected_manifest_keys = {
        "schema_version",
        "experiment",
        "protocol_identity_sha256",
        "protocol_file_sha256",
        "pipeline_sources",
        "blind_bundle_identity_sha256",
        "public_blind_pack_manifest_identity_sha256",
        "private_mapping_identity_sha256",
        "blind_mapping_commitment_identity_sha256",
        "blind_mapping_commitment_file_sha256",
        "consensus_lock_manifest_identity_sha256",
        "consensus_annotation_identity_sha256",
        "consensus_file_sha256",
        "external_visual_anchor_binding",
        "shards",
        "one_time_unseal_claim",
        "summary_payload_sha256",
        "summary_file_sha256",
        "no_per_sample_output",
        "identity_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise RuntimeError("aggregate manifest schema changed")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("summary_payload_sha256") != summary["payload_sha256"]
        or manifest.get("summary_file_sha256") != sha256_file(root / SUMMARY_NAME)
        or manifest.get("no_per_sample_output") is not True
    ):
        raise RuntimeError("aggregate summary manifest binding changed")
    expected_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
        "summary_payload_sha256": summary["payload_sha256"],
        "summary_file_sha256": sha256_file(root / SUMMARY_NAME),
        "consensus_annotation_identity_sha256": manifest[
            "consensus_annotation_identity_sha256"
        ],
        "one_time_unseal_claim_identity_sha256": manifest[
            "one_time_unseal_claim"
        ]["claim_identity_sha256"],
        "aggregate_only": True,
        "payload_sha256": completion.get("payload_sha256"),
    }
    if completion != expected_completion:
        raise RuntimeError("aggregate summary completion binding changed")
    expected_files = {
        (root / SUMMARY_NAME).resolve(),
        (root / MANIFEST_NAME).resolve(),
        (root / COMPLETION_NAME).resolve(),
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    actual_dirs = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_files != expected_files or actual_dirs:
        raise RuntimeError("aggregate summary is not a closed flat file set")
    return summary, completion


def run_real(
    blind_bundle: Path,
    consensus_lock: Path,
    shard_roots: tuple[Path, ...],
    outdir: Path,
) -> None:
    # Phase 1: only public quality artifacts and complete consensus are read.
    public_manifest, _ = validate_public_pack(blind_bundle / PUBLIC_DIR_NAME)
    token = _stage_annotation_lock(consensus_lock, blind_bundle)
    protocol = _read_self_hashed_json(PROTOCOL_SOURCE, "protocol_identity_sha256")
    _validate_protocol_before_unseal_claim(protocol)
    # The top-level envelope declares private hashes but contains no reversible
    # mapping or evidence values, so its complete public-facing structure can be
    # rejected before consuming the one-time claim.
    blind_bundle_manifest = _validate_blind_bundle_envelope_before_unseal_claim(
        blind_bundle, public_manifest
    )
    # Claim the one permitted readout before opening the private seal or shards.
    claim_path, claim_payload = _claim_one_time_unseal(
        blind_bundle,
        bundle_identity=blind_bundle_manifest["identity_sha256"],
        protocol_identity=protocol["protocol_identity_sha256"],
        token=token,
    )
    # Phase 2: private access is capability-gated and output remains aggregate.
    _validate_private_protocol_after_claim(
        protocol,
        blind_bundle=blind_bundle,
        bundle_identity=blind_bundle_manifest["identity_sha256"],
        token=token,
        claim_path=claim_path,
        claim_payload=claim_payload,
    )
    joined, mapping, shard_bindings = _load_private_join_after_lock(
        token, blind_bundle, shard_roots
    )
    summary = summarize_joined(joined)
    summary["annotation_and_binding"] = {
        "consensus_annotation_identity_sha256": token.consensus_identity_sha256,
        "consensus_lock_manifest_identity_sha256": (
            token.consensus_lock_manifest_identity_sha256
        ),
        "blind_bundle_identity_sha256": blind_bundle_manifest["identity_sha256"],
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "external_visual_anchor_binding": mapping[
            "external_visual_anchor_binding"
        ],
    }
    summary["payload_sha256"] = _canonical_self_hash(summary, "payload_sha256")
    _assert_aggregate_only(
        summary,
        public_ids=(entry["public_blind_id"] for entry in mapping["entries"]),
        runner_ids=(entry["runner_blind_id"] for entry in mapping["entries"]),
    )
    _write_summary_bundle(
        outdir,
        protocol=protocol,
        token=token,
        blind_bundle_manifest=blind_bundle_manifest,
        public_manifest=public_manifest,
        mapping=mapping,
        shard_bindings=shard_bindings,
        claim_path=claim_path,
        claim_payload=claim_payload,
        summary=summary,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "aggregate_only": True,
                "one_time_unseal_consumed": True,
                "summary_payload_sha256": summary["payload_sha256"],
            },
            sort_keys=True,
        )
    )


def _synthetic_records(
    *,
    n_bad: int,
    bad_alarms: int,
    not_bad_alarms: int,
    uncertain: int = 0,
) -> list[_JoinedRecord]:
    if n_bad + uncertain > TOTAL_POOL_BRANCHES:
        raise ValueError("invalid synthetic class counts")
    n_not = TOTAL_POOL_BRANCHES - n_bad - uncertain
    records: list[_JoinedRecord] = []
    labels = [PRIMARY_BAD] * n_bad + [PRIMARY_NOT_BAD] * n_not + [PRIMARY_UNCERTAIN] * uncertain
    bad_seen = 0
    not_seen = 0
    for index, label in enumerate(labels):
        if label == PRIMARY_BAD:
            alarm = int(bad_seen < bad_alarms)
            bad_seen += 1
            score = 2.0 if alarm else 0.3
        elif label == PRIMARY_NOT_BAD:
            alarm = int(not_seen < not_bad_alarms)
            not_seen += 1
            score = 1.8 if alarm else 0.1
        else:
            alarm = 0
            score = 0.2
        records.append(
            _JoinedRecord(
                primary_label=label,
                hind_limb_label="not_clear_failure",
                tail_identity="clear",
                tail_scorable="yes",
                tail_confidence="high",
                tail_derived_label="natural",
                tail_values=(0, 0, 0, 0, 0, 0, 0),
                alarm=alarm,
                running_max_log_e=max(score, 0.0),
                terminal_log_e=score - 0.2,
                shard_index=index // BRANCHES_PER_SHARD,
                trajectory_scalar_features=(
                    max(score, 0.0),
                    score - 0.2,
                    max(score, 0.0) - (score - 0.2),
                    max(score, 0.0),
                    max(score, 0.0),
                    0.4 + 0.01 * index,
                    0.2,
                    0.3 + 0.005 * index,
                    0.1,
                    0.2,
                    0.05,
                ),
                log_e_argmax_internal_timestep=30,
                log_e_argmax_positive_jump_internal_timestep=60,
            )
        )
    return records


def _expect_failure(operation: Callable[[], Any], label: str) -> None:
    try:
        operation()
    except RuntimeError:
        return
    raise AssertionError(f"negative self-test did not fail: {label}")


def run_self_test() -> None:
    synthetic_path = np.arange(61, dtype=np.float64) / 64.0 - 0.5
    synthetic_raw = np.zeros((61, 34), dtype=np.float64)
    for base_index in range(17):
        synthetic_raw[:, 2 * base_index] = (
            np.linspace(0.0, 1.0, 61) * (base_index + 1)
        )
        synthetic_raw[:, 2 * base_index + 1] = synthetic_raw[:, 2 * base_index]
    feature_probe, peak_t, jump_t = _trajectory_shape_features(
        synthetic_path, synthetic_raw, np.arange(60, -1, -1)
    )
    if (
        len(feature_probe) != len(TRAJECTORY_SCALAR_FEATURES)
        or peak_t != 0
        or jump_t != 59
    ):
        raise AssertionError("frozen trajectory-shape construction failed")
    mutated_raw = synthetic_raw.copy()
    mutated_raw[0, 1] += 1e-12
    _expect_failure(
        lambda: _trajectory_shape_features(
            synthetic_path, mutated_raw, np.arange(60, -1, -1)
        ),
        "raw_K sign-pair mutation",
    )
    zero = summarize_joined(
        _synthetic_records(n_bad=6, bad_alarms=0, not_bad_alarms=0)
    )
    if zero["primary"]["frozen_decision_tree"]["outcome"] != (
        "frozen_threshold_failed_to_pass"
    ):
        raise AssertionError("zero-crossing decision branch failed")
    limited = summarize_joined(
        _synthetic_records(n_bad=4, bad_alarms=3, not_bad_alarms=0)
    )
    if limited["primary"]["frozen_decision_tree"]["outcome"] != (
        "event_limited_inconclusive"
    ):
        raise AssertionError("event-limited decision branch failed")
    survived = summarize_joined(
        _synthetic_records(n_bad=10, bad_alarms=5, not_bad_alarms=0)
    )
    if survived["primary"]["frozen_decision_tree"]["outcome"] != (
        "prospective_screen_survival"
    ):
        raise AssertionError("prospective screen-survival branch failed")
    failed = summarize_joined(
        _synthetic_records(n_bad=10, bad_alarms=2, not_bad_alarms=3)
    )
    if failed["primary"]["frozen_decision_tree"]["outcome"] != (
        "prospective_screen_failed_retire_fixed_candidate"
    ):
        raise AssertionError("prospective failure branch failed")
    tied = _auc_and_ranks(np.asarray([1.0, 1.0]), np.asarray([1.0, 0.0]))
    if tied["auc"] != 0.75 or tied[
        "pairwise_wins_with_ties_counted_as_half"
    ] != 3.0:
        raise AssertionError("tie-aware AUC/ranks failed")
    if _clopper_pearson(0, 10)["lower"] != 0.0 or _clopper_pearson(10, 10)[
        "upper"
    ] != 1.0:
        raise AssertionError("exact binomial boundary intervals failed")
    _expect_failure(
        lambda: _require_token(
            _AnnotationLockToken(object(), {}, "a" * 64, "b" * 64, "c" * 64)
        ),
        "forged annotation-lock capability",
    )
    with tempfile.TemporaryDirectory(prefix="cross-prefix-unseal-selftest-") as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        token = _AnnotationLockToken(
            _LOCK_NONCE, {}, "a" * 64, "b" * 64, "c" * 64
        )
        protocol_probe = load_json(PROTOCOL_SOURCE)
        protocol_probe["protocol_status"] = "FROZEN_BEFORE_GPU_EXECUTION"
        protocol_probe["authorization_gate"]["gpu_execution_authorized"] = True
        protocol_probe["blind_pipeline_binding"] = {
            "blind_pack_builder_filename": BLIND_PACK_BUILDER.name,
            "blind_pack_builder_sha256": sha256_file(BLIND_PACK_BUILDER),
            "consensus_locker_filename": CONSENSUS_LOCKER.name,
            "consensus_locker_sha256": sha256_file(CONSENSUS_LOCKER),
            "aggregate_unseal_summarizer_filename": RUNNER.name,
            "aggregate_unseal_summarizer_sha256": sha256_file(RUNNER),
        }
        protocol_probe["blind_mapping_commitment_binding"] = {
            "status": MAPPING_COMMITMENT_STATUS,
            "commitment_schema": MAPPING_COMMITMENT_SCHEMA,
            "pool_size": TOTAL_POOL_BRANCHES,
            "commitment_path": str(root / "private-never-opened.json"),
            "mapping_builder_filename": BLIND_PACK_BUILDER.name,
            "mapping_builder_sha256": sha256_file(BLIND_PACK_BUILDER),
            "commitment_identity_sha256": "d" * 64,
            "commitment_file_sha256": "e" * 64,
        }
        protocol_probe["statistical_readout_after_blind_lock"][
            "secondary_trajectory_shape_panel"
        ] = frozen_trajectory_shape_panel_definition()
        protocol_probe["protocol_identity_sha256"] = _canonical_self_hash(
            protocol_probe, "protocol_identity_sha256"
        )
        private_validation_events: list[str] = []
        original_private_protocol_validator = globals()[
            "_validate_protocol_for_pipeline"
        ]

        def guarded_private_protocol_validator(value: dict[str, Any]) -> None:
            if value is not protocol_probe:
                raise RuntimeError("instrumented private validator received wrong protocol")
            sentinel = _claim_sentinel_path(root, "d" * 64)
            if not sentinel.is_file() or sentinel.is_symlink():
                raise RuntimeError("private protocol validation ran before unseal claim")
            private_validation_events.append("claim_seen_before_private_validation")

        globals()["_validate_protocol_for_pipeline"] = (
            guarded_private_protocol_validator
        )
        try:
            _validate_protocol_before_unseal_claim(protocol_probe)
            if private_validation_events:
                raise AssertionError("pre-claim validation touched a private validator")
            _expect_failure(
                lambda: guarded_private_protocol_validator(protocol_probe),
                "instrumented private validation before claim",
            )
            path, payload = _claim_one_time_unseal(
                root,
                bundle_identity="d" * 64,
                protocol_identity=protocol_probe["protocol_identity_sha256"],
                token=token,
            )
            _validate_private_protocol_after_claim(
                protocol_probe,
                blind_bundle=root,
                bundle_identity="d" * 64,
                token=token,
                claim_path=path,
                claim_payload=payload,
            )
        finally:
            globals()["_validate_protocol_for_pipeline"] = (
                original_private_protocol_validator
            )
        if private_validation_events != ["claim_seen_before_private_validation"]:
            raise AssertionError("private protocol validation ordering was not observed")
        if not path.is_file() or payload["status"] != (
            "CLAIMED_BEFORE_ANY_PRIVATE_MAPPING_OR_EVIDENCE_ACCESS"
        ):
            raise AssertionError("one-time claim creation failed")
        _expect_failure(
            lambda: _claim_one_time_unseal(
                root,
                bundle_identity="d" * 64,
                protocol_identity=protocol_probe["protocol_identity_sha256"],
                token=token,
            ),
            "second aggregate unseal claim",
        )
    leak_payload = {"safe": True, "nested": {"public_blind_id": "xr1_deadbeefdeadbeef"}}
    _expect_failure(
        lambda: _assert_aggregate_only(
            leak_payload,
            public_ids=("xr1_deadbeefdeadbeef",),
            runner_ids=(),
        ),
        "per-sample lineage leakage",
    )
    print(
        "self-test passed: all four frozen decision branches, exact intervals, "
        "tie-aware AUC, fixed log-E/raw-K trajectory-shape construction, annotation "
        "capability, claim-before-private-access ordering, one-time claim, raw-K "
        "mutation failure, and per-sample leakage rejection; CPU/synthetic only"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-bundle", type=Path)
    parser.add_argument("--consensus-lock", type=Path)
    parser.add_argument("--shard-dir", type=Path, action="append", default=[])
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _plain_directory(
    path: Path | None, parser: argparse.ArgumentParser, label: str
) -> Path:
    if path is None:
        parser.error(f"{label} is required")
    value = path.expanduser().absolute()
    if value.is_symlink() or not value.is_dir():
        parser.error(f"{label} must be a plain directory")
    return value.resolve()


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    blind_bundle = _plain_directory(args.blind_bundle, parser, "--blind-bundle")
    consensus_lock = _plain_directory(
        args.consensus_lock, parser, "--consensus-lock"
    )
    if len(args.shard_dir) != TOTAL_SHARDS:
        parser.error("provide exactly eight --shard-dir arguments")
    shard_roots = tuple(
        _plain_directory(path, parser, "--shard-dir") for path in args.shard_dir
    )
    if len(set(shard_roots)) != TOTAL_SHARDS:
        parser.error("the eight shard directories must be distinct")
    if args.outdir is None:
        parser.error("--outdir is required")
    outdir = args.outdir.expanduser().absolute()
    if os.path.lexists(outdir):
        parser.error("--outdir already exists")
    for protected in (blind_bundle, consensus_lock, *shard_roots, RUNNER.parent.parent):
        left = outdir.resolve()
        right = protected.resolve()
        if left == right or left in right.parents or right in left.parents:
            parser.error("--outdir overlaps a protected input/source tree")
    run_real(blind_bundle, consensus_lock, shard_roots, outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
