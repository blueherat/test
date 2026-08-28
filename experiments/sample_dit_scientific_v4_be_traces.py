#!/usr/bin/env python3
"""Sample the prospective scientific-v4 B/E minimum DiT trace.

Each singleton ``(global_seed,class_id)`` stream is pair-keyed.  The baseline
sampler is the frozen 250-step ancestral DDPM and consumes one full 2B
``randn_like`` transition draw at every step, including the masked t=0 draw.
At the nine frozen checkpoints, all current/shifted DiT observations and the
temporary VAE decode finish before that draw and must leave CUDA RNG unchanged.

The immutable NPZ lives under a method-only tree and stores only the seven
B/E-required arrays (the six tensor/schedule objects plus the sampling-step
axis).  The endpoint PNG lives under a disjoint blind-review tree.  No B/E score,
threshold, alert, endpoint metric, label, review, or intervention is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image

try:
    from . import dit_scientific_v4_be_contract as be_contract
    from .dit_scientific_v4_be_contract import (
        CHECKPOINTS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        INTERNAL_TIMESTEPS,
        METHOD_LOCK_ID,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        fixed_no_touch_pair,
        load_json,
        pair_relative_directory,
        publish_artifact,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
        validate_manifest_tree,
        validate_dynamic_contract_payload,
        validate_method_lock,
        validate_scientific_protocol,
        validate_trace_plan,
        without_identity,
        write_json,
        write_json_exclusive,
    )
except ImportError:
    import dit_scientific_v4_be_contract as be_contract  # type: ignore
    from dit_scientific_v4_be_contract import (  # type: ignore
        CHECKPOINTS,
        DEFAULT_DYNAMIC_SOURCE_LOCK,
        INTERNAL_TIMESTEPS,
        METHOD_LOCK_ID,
        canonical_sha256,
        derive_pair_seed,
        exact_pairs,
        fixed_no_touch_pair,
        load_json,
        pair_relative_directory,
        publish_artifact,
        require_directory,
        require_regular,
        sha256_array,
        sha256_file,
        validate_manifest_tree,
        validate_dynamic_contract_payload,
        validate_method_lock,
        validate_scientific_protocol,
        validate_trace_plan,
        without_identity,
        write_json,
        write_json_exclusive,
    )


RUNNER = "sample_dit_scientific_v4_be_traces"
PAIR_SCHEMA = 1
TRACE_NAME = "be_minimum_trace.npz"
ENDPOINT_NAME = "endpoint.png"
TRACE_PAIR_MANIFEST = "trace_manifest.json"
TRACE_PAIR_COMPLETION = "trace_completion.json"
ENDPOINT_PAIR_MANIFEST = "endpoint_manifest.json"
ENDPOINT_PAIR_COMPLETION = "endpoint_completion.json"
TRACE_POOL_MANIFEST = "trace_pool_manifest.json"
TRACE_POOL_COMPLETION = "trace_pool_completion.json"
ENDPOINT_POOL_MANIFEST = "pool_manifest.json"
ENDPOINT_POOL_COMPLETION = "pool_completion.json"
METHOD_TREE = "method_traces"
REVIEW_TREE = "review_endpoints"
TRACE_ARRAYS = (
    "state_before",
    "pred_xstart",
    "p_standard_deviation",
    "transition_innovation",
    "sampling_step",
    "internal_timestep",
    "alpha_bar",
)


def _load_module(path: Path, name: str) -> ModuleType:
    path = require_regular(path, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register the immutable unique name while executing (dataclasses and
    # import machinery consult sys.modules). Never add a live repository path.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _load_frozen_method_v2(source_lock: Path) -> ModuleType:
    """Load v2 with its inherited v1 dependency pinned inside the source lock."""

    v1 = _load_module(
        source_lock / "sources/observe_dit_blur_focused_eprocess_v1.py",
        "_v4_be_frozen_method_core_v1",
    )
    import_name = "observe_dit_blur_focused_eprocess"
    previous = sys.modules.get(import_name)
    sys.modules[import_name] = v1
    try:
        v2 = _load_module(
            source_lock / "sources/observe_dit_blur_focused_eprocess_v2.py",
            "_v4_be_frozen_method_core_v2",
        )
    finally:
        if previous is None:
            sys.modules.pop(import_name, None)
        else:
            sys.modules[import_name] = previous
    if getattr(v2, "v1", None) is not v1 or getattr(v2, "SCHEMA_VERSION", None) != 2:
        raise RuntimeError("method-v2 core escaped its frozen v1 dependency")
    v1_file = Path(getattr(v1, "__file__", "")).resolve()
    v2_file = Path(getattr(v2, "__file__", "")).resolve()
    frozen_sources = (source_lock / "sources").resolve()
    if v1_file.parent != frozen_sources or v2_file.parent != frozen_sources:
        raise RuntimeError("method core loaded from outside the dynamic source lock")
    return v2


def load_source_lock(
    source_lock: Path,
) -> tuple[dict[str, Any], dict[str, Any], ModuleType, ModuleType]:
    source_lock = require_directory(source_lock, "scientific-v4 dynamic source lock")
    manifest, completion = validate_manifest_tree(source_lock)
    contract = load_json(require_regular(source_lock / "dynamic_contract.json", "dynamic contract"))
    validate_dynamic_contract_payload(contract)
    contract_id = contract.get("identity_sha256")
    if (
        not isinstance(contract_id, str)
        or canonical_sha256(without_identity(contract)) != contract_id
        or manifest.get("dynamic_contract_identity_sha256") != contract_id
        or completion.get("dynamic_contract_identity_sha256") != contract_id
    ):
        raise RuntimeError("dynamic source-lock contract identity changed")
    by_name = {row["name"]: row for row in manifest["files"]}
    contract_source = "sources/dit_scientific_v4_be_contract.py"
    running_contract_path = Path(getattr(be_contract, "__file__", "")).resolve()
    if by_name.get(contract_source, {}).get("sha256") != sha256_file(
        require_regular(running_contract_path, "running B/E contract source")
    ):
        raise RuntimeError("running B/E contract differs from frozen source snapshot")
    current = "sources/sample_dit_scientific_v4_be_traces.py"
    if by_name.get(current, {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("running sampler differs from its frozen snapshot")
    strict = _load_module(
        source_lock / "sources/reproduce_dit_imagenet256.py", "_v4_be_frozen_strict"
    )
    method_core = _load_frozen_method_v2(source_lock)
    return contract, manifest, strict, method_core


def _array_contract() -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    state_shape = (len(CHECKPOINTS), 4, 32, 32)
    return {
        "state_before": (state_shape, np.dtype(np.float32)),
        "pred_xstart": (state_shape, np.dtype(np.float32)),
        "p_standard_deviation": (state_shape, np.dtype(np.float32)),
        "transition_innovation": (state_shape, np.dtype(np.float32)),
        "sampling_step": ((len(CHECKPOINTS),), np.dtype(np.int16)),
        "internal_timestep": ((len(CHECKPOINTS),), np.dtype(np.int16)),
        "alpha_bar": ((len(CHECKPOINTS),), np.dtype(np.float64)),
    }


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    contract = _array_contract()
    if tuple(arrays) != TRACE_ARRAYS or set(arrays) != set(contract):
        raise RuntimeError("minimum trace member set/order changed")
    for name, (shape, dtype) in contract.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise RuntimeError(f"invalid minimum trace array {name}: {value.shape}/{value.dtype}")
    if not np.array_equal(arrays["sampling_step"], np.asarray(CHECKPOINTS, dtype=np.int16)):
        raise RuntimeError("sampling checkpoint axis changed")
    if not np.array_equal(
        arrays["internal_timestep"], np.asarray(INTERNAL_TIMESTEPS, dtype=np.int16)
    ):
        raise RuntimeError("internal timestep axis changed")
    if np.any(arrays["p_standard_deviation"] <= 0.0):
        raise RuntimeError("P standard deviation must be strictly positive")
    if np.any(arrays["alpha_bar"] <= 0.0) or np.any(arrays["alpha_bar"] > 1.0):
        raise RuntimeError("alpha_bar must lie in (0,1]")


def array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(arrays[name].shape),
            "dtype": arrays[name].dtype.str,
            "raw_sha256": sha256_array(arrays[name]),
        }
        for name in TRACE_ARRAYS
    }


def inspect_png(path: Path) -> dict[str, Any]:
    path = require_regular(path, "blind-review endpoint PNG")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or tuple(image.size) != (256, 256):
            raise RuntimeError("endpoint must be RGB 256x256")
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    return {
        "name": ENDPOINT_NAME,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_hash,
        "mode": "RGB",
        "size": [256, 256],
    }


def pair_identity(
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": PAIR_SCHEMA,
        "runner": RUNNER,
        "dynamic_contract_identity_sha256": contract["identity_sha256"],
        "scientific_protocol_identity_sha256": plan["protocol_identity_sha256"],
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "trace_plan_identity_sha256": plan["identity_sha256"],
        "pair_key": {"phase": phase, "global_seed": global_seed, "class_id": class_id},
        "derived_torch_seed": derive_pair_seed(global_seed, class_id),
        "trace_arrays": list(TRACE_ARRAYS),
        "observation_only": True,
        "score_threshold_alert_or_intervention_computed": False,
        "endpoint_label_review_or_external_representation_opened": False,
        "sampler_contract": contract["sampler_contract"],
    }


def neutral_sampling_pair_identity(
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
) -> dict[str, Any]:
    """Cross-tree identity safe to expose to the external review firewall."""

    return {
        "schema_version": PAIR_SCHEMA,
        "runner": "sample_dit_scientific_v4_observed_pair",
        "event_protocol_identity_sha256": plan["protocol_identity_sha256"],
        "anchor_plan_identity_sha256": plan["identity_sha256"],
        "pair_key": {"phase": phase, "global_seed": global_seed, "class_id": class_id},
        "derived_torch_seed": derive_pair_seed(global_seed, class_id),
        "sampler_contract": contract["sampler_contract"],
        "source_payload_role": "external_review_endpoints_only",
        "internal_method_payload_opened": False,
    }


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_trace_pair(
    root: Path,
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
    load_array_names: Iterable[str] = TRACE_ARRAYS,
) -> dict[str, Any]:
    relative = pair_relative_directory(phase, global_seed, class_id)
    pair = require_directory(root / relative, "v4 B/E trace pair")
    required = {TRACE_NAME, TRACE_PAIR_MANIFEST, TRACE_PAIR_COMPLETION}
    if {item.name for item in pair.iterdir()} != required:
        raise RuntimeError("pair member set changed")
    # This method-side directory physically contains no endpoint payload/envelope.
    manifest_path = require_regular(pair / TRACE_PAIR_MANIFEST, "trace-pair manifest")
    completion_path = require_regular(pair / TRACE_PAIR_COMPLETION, "trace-pair completion")
    trace_path = require_regular(pair / TRACE_NAME, "minimum trace")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = pair_identity(
        contract=contract,
        plan=plan,
        phase=phase,
        global_seed=global_seed,
        class_id=class_id,
    )
    if (
        set(manifest)
        != {
            "schema_version",
            "status",
            "identity",
            "identity_sha256",
            "trace",
            "execution",
            "neutral_sampling_pair_identity_sha256",
        }
        or manifest.get("schema_version") != PAIR_SCHEMA
        or
        manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("identity_sha256") != canonical_sha256(identity)
        or manifest.get("neutral_sampling_pair_identity_sha256")
        != canonical_sha256(
            neutral_sampling_pair_identity(
                contract=contract,
                plan=plan,
                phase=phase,
                global_seed=global_seed,
                class_id=class_id,
            )
        )
    ):
        raise RuntimeError("trace pair identity changed")
    execution = manifest.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("derived_torch_seed") != derive_pair_seed(global_seed, class_id)
        or execution.get("transition_randn_like_calls") != 250
        or execution.get("transition_draw_shape") != [2, 4, 32, 32]
        or execution.get("terminal_t0_draw_consumed_then_masked") is not True
        or execution.get("preinnovation_observation_enabled") is not True
        or execution.get("preinnovation_observation_count") != len(CHECKPOINTS)
        or execution.get("all_observations_rng_neutral") is not True
    ):
        raise RuntimeError("trace execution/RNG observation contract changed")
    trace_record = manifest.get("trace")
    if not isinstance(trace_record, dict) or trace_record.get("sha256") != sha256_file(trace_path):
        raise RuntimeError("trace archive hash changed")
    requested = tuple(load_array_names)
    if len(set(requested)) != len(requested) or not set(requested) <= set(TRACE_ARRAYS):
        raise RuntimeError("trace array whitelist is duplicated or unknown")
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            if tuple(archive.files) != TRACE_ARRAYS:
                raise RuntimeError("trace NPZ member set/order changed")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in requested}
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot read minimum trace NPZ") from exc
    records = trace_record.get("arrays")
    if not isinstance(records, dict) or set(records) != set(TRACE_ARRAYS):
        raise RuntimeError("trace array-record member set/order changed")
    contract_map = _array_contract()
    for name, value in arrays.items():
        shape, dtype = contract_map[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise RuntimeError(f"invalid whitelisted trace array: {name}")
        observed_record = {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "raw_sha256": sha256_array(value),
        }
        if records.get(name) != observed_record:
            raise RuntimeError(f"whitelisted trace array record changed: {name}")
    if set(requested) == set(TRACE_ARRAYS):
        validate_arrays(arrays)
    expected_completion = {
        "complete": True,
        "identity_sha256": manifest["identity_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "trace_sha256": trace_record["sha256"],
    }
    if completion != expected_completion:
        raise RuntimeError("trace pair completion changed")
    return {
        "phase": phase,
        "global_seed": global_seed,
        "class_id": class_id,
        "relative_directory": relative,
        "identity_sha256": manifest["identity_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "trace_sha256": trace_record["sha256"],
    }


def validate_endpoint_pair(
    root: Path,
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
) -> dict[str, Any]:
    pair = require_directory(
        root / pair_relative_directory(phase, global_seed, class_id),
        "v4 blind-review endpoint pair",
    )
    if {item.name for item in pair.iterdir()} != {
        ENDPOINT_NAME,
        ENDPOINT_PAIR_MANIFEST,
        ENDPOINT_PAIR_COMPLETION,
    }:
        raise RuntimeError("endpoint pair member set changed")
    manifest_path = require_regular(pair / ENDPOINT_PAIR_MANIFEST, "endpoint manifest")
    completion_path = require_regular(pair / ENDPOINT_PAIR_COMPLETION, "endpoint completion")
    endpoint = inspect_png(pair / ENDPOINT_NAME)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = neutral_sampling_pair_identity(
        contract=contract,
        plan=plan,
        phase=phase,
        global_seed=global_seed,
        class_id=class_id,
    )
    if (
        set(manifest)
        != {
            "schema_version",
            "status",
            "identity",
            "identity_sha256",
            "endpoint",
            "internal_method_payload_opened",
        }
        or manifest.get("schema_version") != PAIR_SCHEMA
        or
        manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("identity_sha256") != canonical_sha256(identity)
        or manifest.get("endpoint") != endpoint
        or manifest.get("internal_method_payload_opened") is not False
    ):
        raise RuntimeError("endpoint pair identity/record changed")
    expected = {
        "complete": True,
        "identity_sha256": manifest["identity_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "endpoint_sha256": endpoint["sha256"],
        "endpoint_pixel_sha256": endpoint["pixel_sha256"],
    }
    if completion != expected:
        raise RuntimeError("endpoint pair completion changed")
    return {
        "phase": phase,
        "global_seed": global_seed,
        "class_id": class_id,
        "relative_directory": pair_relative_directory(phase, global_seed, class_id),
        "identity_sha256": manifest["identity_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "endpoint_sha256": endpoint["sha256"],
        "endpoint_pixel_sha256": endpoint["pixel_sha256"],
    }


def load_trace_array_whitelist(
    root: Path,
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
    names: Iterable[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    requested = tuple(names)
    record = validate_trace_pair(
        root,
        contract=contract,
        plan=plan,
        phase=phase,
        global_seed=global_seed,
        class_id=class_id,
        load_array_names=requested,
    )
    path = root / pair_relative_directory(phase, global_seed, class_id) / TRACE_NAME
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in requested}
    return arrays, record


def publish_pair(
    root: Path,
    *,
    decoded: torch.Tensor,
    arrays: Mapping[str, np.ndarray],
    execution: Mapping[str, Any],
    save_image: Any,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    global_seed: int,
    class_id: int,
) -> dict[str, Any]:
    relative = pair_relative_directory(phase, global_seed, class_id)
    trace_root = root / METHOD_TREE
    endpoint_root = root / REVIEW_TREE
    trace_destination = trace_root / relative
    endpoint_destination = endpoint_root / relative
    has_endpoint = phase == "confirmation"
    if os.path.lexists(trace_destination) or (
        has_endpoint and os.path.lexists(endpoint_destination)
    ):
        if not os.path.lexists(trace_destination) or (
            has_endpoint and not os.path.lexists(endpoint_destination)
        ):
            raise RuntimeError("partial cross-tree pair exists; refuse ambiguous recovery")
        trace_record = validate_trace_pair(
            trace_root,
            contract=contract,
            plan=plan,
            phase=phase,
            global_seed=global_seed,
            class_id=class_id,
        )
        if has_endpoint:
            validate_endpoint_pair(
                endpoint_root,
                contract=contract,
                plan=plan,
                phase=phase,
                global_seed=global_seed,
                class_id=class_id,
            )
        return trace_record
    trace_destination.parent.mkdir(parents=True, exist_ok=True)
    if has_endpoint:
        endpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".pair-{global_seed}-{class_id}.tmp-", dir=root))
    trace_staging = staging / METHOD_TREE
    endpoint_staging = staging / REVIEW_TREE
    trace_staging.mkdir()
    if has_endpoint:
        endpoint_staging.mkdir()
    try:
        validate_arrays(arrays)
        _save_npz(trace_staging / TRACE_NAME, arrays)
        if has_endpoint:
            save_image(
                decoded,
                endpoint_staging / ENDPOINT_NAME,
                nrow=1,
                padding=0,
                normalize=True,
                value_range=(-1, 1),
            )
        identity = pair_identity(
            contract=contract,
            plan=plan,
            phase=phase,
            global_seed=global_seed,
            class_id=class_id,
        )
        trace_record = {
            "name": TRACE_NAME,
            "bytes": (trace_staging / TRACE_NAME).stat().st_size,
            "sha256": sha256_file(trace_staging / TRACE_NAME),
            "arrays": array_records(arrays),
        }
        endpoint = inspect_png(endpoint_staging / ENDPOINT_NAME) if has_endpoint else None
        trace_manifest = {
            "schema_version": PAIR_SCHEMA,
            "status": "complete",
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "trace": trace_record,
            "execution": dict(execution),
            "neutral_sampling_pair_identity_sha256": canonical_sha256(
                neutral_sampling_pair_identity(
                    contract=contract,
                    plan=plan,
                    phase=phase,
                    global_seed=global_seed,
                    class_id=class_id,
                )
            ),
        }
        endpoint_identity = neutral_sampling_pair_identity(
            contract=contract,
            plan=plan,
            phase=phase,
            global_seed=global_seed,
            class_id=class_id,
        )
        endpoint_manifest = {
            "schema_version": PAIR_SCHEMA,
            "status": "complete",
            "identity": endpoint_identity,
            "identity_sha256": canonical_sha256(endpoint_identity),
            "endpoint": endpoint,
            "internal_method_payload_opened": False,
        }
        write_json_exclusive(trace_staging / TRACE_PAIR_MANIFEST, trace_manifest)
        write_json_exclusive(
            trace_staging / TRACE_PAIR_COMPLETION,
            {
                "complete": True,
                "identity_sha256": trace_manifest["identity_sha256"],
                "manifest_sha256": sha256_file(trace_staging / TRACE_PAIR_MANIFEST),
                "trace_sha256": trace_record["sha256"],
            },
        )
        if has_endpoint:
            assert endpoint is not None
            write_json_exclusive(endpoint_staging / ENDPOINT_PAIR_MANIFEST, endpoint_manifest)
            write_json_exclusive(
                endpoint_staging / ENDPOINT_PAIR_COMPLETION,
                {
                    "complete": True,
                    "identity_sha256": endpoint_manifest["identity_sha256"],
                    "manifest_sha256": sha256_file(endpoint_staging / ENDPOINT_PAIR_MANIFEST),
                    "endpoint_sha256": endpoint["sha256"],
                    "endpoint_pixel_sha256": endpoint["pixel_sha256"],
                },
            )
        os.replace(trace_staging, trace_destination)
        if has_endpoint:
            os.replace(endpoint_staging, endpoint_destination)
        staging.rmdir()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    trace_result = validate_trace_pair(
        trace_root,
        contract=contract,
        plan=plan,
        phase=phase,
        global_seed=global_seed,
        class_id=class_id,
    )
    if has_endpoint:
        validate_endpoint_pair(
            endpoint_root,
            contract=contract,
            plan=plan,
            phase=phase,
            global_seed=global_seed,
            class_id=class_id,
        )
    return trace_result


def validate_trace_pool(
    root: Path,
    *,
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, int, int], dict[str, Any]]]:
    """Validate only the method-side trace envelope; never open endpoint files."""

    root = require_directory(root, "scientific-v4 B/E trace pool")
    if {item.name for item in root.iterdir()} != {
        "pairs",
        TRACE_POOL_MANIFEST,
        TRACE_POOL_COMPLETION,
    }:
        raise RuntimeError("method-only trace root exact tree changed")
    validate_method_tree_firewall(root)
    return _validate_trace_pool_after_firewall(
        root, contract=contract, protocol=protocol, plan=plan
    )


def _upstream_dit_module_names() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"models", "download", "diffusion"}
        or name.startswith("diffusion.")
    }


def reject_preexisting_upstream_dit_modules() -> None:
    """Fail before import if generic upstream names could shadow frozen DiT."""

    preexisting = _upstream_dit_module_names()
    if preexisting:
        raise RuntimeError(
            "ambiguous pre-imported upstream DiT modules: "
            + repr(sorted(preexisting))
        )


def validate_method_tree_firewall(root: Path) -> None:
    """Reject evaluation/supervision payloads from any method-only tree."""

    root = require_directory(root, "method-only tree")
    forbidden_names = (
        "endpoint",
        "label",
        "review",
        "consensus",
        "inception",
        "dino",
        "fid",
        "clip",
        "embedding",
    )
    for path in root.rglob("*"):
        lowered = path.name.lower()
        if any(token in lowered for token in forbidden_names):
            raise RuntimeError(f"forbidden payload entered method-only trace tree: {path.name}")


def validate_review_tree_firewall(root: Path) -> None:
    """Reject any material internal-method payload from the review-only tree."""

    root = require_directory(root, "review-only tree")
    forbidden_fragments = (
        "trace",
        "method",
        "candidate",
        "score",
        "metric",
        "feature",
        "embedding",
        "inception",
        "dino",
        "fid",
        "clip",
        "calibration",
        "observer",
        "eprocess",
        "b_persistence",
        "e_blur_gated",
    )
    forbidden_suffixes = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".csv")

    def check_json(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in forbidden_fragments) and child not in (
                    False,
                    None,
                    "",
                    [],
                    {},
                ):
                    raise RuntimeError(f"review tree has material forbidden field: {location}:{key}")
                check_json(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_json(child, f"{location}[{index}]")

    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("review-only tree contains a symlink")
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if any(fragment in lowered for fragment in forbidden_fragments):
            raise RuntimeError(f"review-only tree has forbidden member name: {relative}")
        if path.is_dir():
            continue
        if path.suffix.lower() in forbidden_suffixes:
            raise RuntimeError(f"review-only tree has forbidden payload type: {relative}")
        if path.suffix.lower() == ".json":
            check_json(load_json(path), relative)


def _validate_trace_pool_after_firewall(
    root: Path,
    *,
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, int, int], dict[str, Any]]]:
    """Implementation tail split out so the firewall is independently testable."""

    manifest_path = require_regular(root / TRACE_POOL_MANIFEST, "trace-pool manifest")
    completion_path = require_regular(root / TRACE_POOL_COMPLETION, "trace-pool completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity_sha256")
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifact_kind") != "SCIENTIFIC_V4_B_E_MINIMUM_TRACE_POOL"
        or canonical_sha256(without_identity(manifest)) != identity
        or manifest.get("dynamic_contract_identity_sha256") != contract["identity_sha256"]
        or manifest.get("scientific_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID
        or manifest.get("trace_plan_identity_sha256") != plan["identity_sha256"]
        or manifest.get("ordered_pair_axis_sha256")
        != canonical_sha256(
            [
                {"phase": phase, "global_seed": seed, "class_id": class_id}
                for phase, seed, class_id in exact_pairs(plan)
            ]
        )
        or manifest.get("labels_reviews_endpoint_metrics_or_external_representations_opened")
        is not False
        or manifest.get("scores_thresholds_alerts_or_interventions_computed") is not False
        or completion
        != {
            "complete": True,
            "manifest_identity_sha256": identity,
            "manifest_file_sha256": sha256_file(manifest_path),
            "pair_count": len(exact_pairs(plan)),
        }
    ):
        raise RuntimeError("trace-pool envelope changed")
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or len(rows) != len(exact_pairs(plan)):
        raise RuntimeError("trace-pool pair inventory changed")
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    for expected, row in zip(exact_pairs(plan), rows, strict=True):
        if not isinstance(row, dict):
            raise RuntimeError("trace-pool pair row is malformed")
        key = (row.get("phase"), row.get("global_seed"), row.get("class_id"))
        if key != expected or key in result:
            raise RuntimeError("trace-pool pair axis/order changed")
        # Trace pair validation never opens an endpoint payload/envelope.
        replay = validate_trace_pair(
            root,
            contract=contract,
            plan=plan,
            phase=expected[0],
            global_seed=expected[1],
            class_id=expected[2],
            load_array_names=(),
        )
        if replay != row:
            raise RuntimeError("trace-pool pair inventory no longer replays")
        result[expected] = replay
    return manifest, result


def load_models(
    strict: ModuleType, contract: Mapping[str, Any], args: argparse.Namespace
) -> tuple[Any, Any, Any, Any]:
    reject_preexisting_upstream_dit_modules()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v4 DiT sampling")
    dit_root = require_directory(args.dit_root, "DiT repository")
    checkpoint = require_regular(args.checkpoint, "DiT checkpoint")
    vae_snapshot = require_directory(args.vae_snapshot, "VAE snapshot")
    assets = contract["assets"]
    if (
        strict.validate_repository(dit_root, checkpoint) != assets["dit_repository"]
        or strict.validate_checkpoint(checkpoint) != assets["checkpoint"]
        or strict.validate_vae_snapshot(vae_snapshot) != assets["vae_snapshot"]
    ):
        raise RuntimeError("runtime DiT/VAE assets differ from source lock")
    strict.ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    try:
        os.chdir(dit_root)
        sys.path.insert(0, str(dit_root))
        from diffusion import create_diffusion
        from diffusers.models import AutoencoderKL
        from download import find_model
        from models import DiT_models
        from torchvision.utils import save_image

        imported = {
            "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
            "download": Path(sys.modules["download"].__file__).resolve(),
            "models": Path(sys.modules["models"].__file__).resolve(),
        }
        expected = {
            "diffusion": (dit_root / "diffusion/__init__.py").resolve(),
            "download": (dit_root / "download.py").resolve(),
            "models": (dit_root / "models.py").resolve(),
        }
        if imported != expected:
            raise RuntimeError(
                f"upstream DiT import shadowing detected: {imported} != {expected}"
            )
        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
        ).to("cuda")
        model.load_state_dict(find_model(str(checkpoint)))
        model.eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        vae = AutoencoderKL.from_pretrained(
            str(vae_snapshot), local_files_only=True, use_safetensors=True
        ).to(device="cuda", dtype=torch.float32).eval()
        return model, diffusion, vae, save_image
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        for name in sorted(_upstream_dit_module_names(), reverse=True):
            sys.modules.pop(name, None)


def _run_preinnovation_observation(
    *,
    model: Any,
    vae: Any,
    diffusion: Any,
    strict: ModuleType,
    method_core: ModuleType,
    state: torch.Tensor,
    pred_xstart: torch.Tensor,
    internal_t: int,
    checkpoint_index: int,
    model_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Run required observations without returning anything sampler can consume."""

    before = strict.cuda_rng_state_sha256()
    alpha = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    timestep_map = np.asarray(diffusion.timestep_map, dtype=np.int64)
    alpha_current = float(alpha[internal_t])
    shifted_evaluations = 0
    with torch.inference_mode():
        # A temporary draft decode is needed for B/mask semantics.  It is never
        # saved here and its tensor is deliberately not returned to the sampler.
        _ = vae.decode(pred_xstart[:1] / strict.VAE_SCALING_FACTOR).sample
        for shifted_row in method_core.SHIFTED_INTERNAL_TIMESTEPS:
            shifted_internal = int(shifted_row[checkpoint_index])
            if shifted_internal == internal_t:
                continue
            rho = math.sqrt(float(alpha[shifted_internal]) / alpha_current)
            shifted_t = torch.full(
                (state.shape[0],),
                int(timestep_map[shifted_internal]),
                dtype=torch.long,
                device=state.device,
            )
            _ = model.forward_with_cfg(
                state * rho,
                shifted_t,
                y=model_kwargs["y"],
                cfg_scale=model_kwargs["cfg_scale"],
            )
            shifted_evaluations += 1
    after = strict.cuda_rng_state_sha256()
    if before != after:
        raise RuntimeError("pre-innovation shifted DiT/VAE observation consumed CUDA RNG")
    return {
        "cuda_rng_sha256_before_and_after": before,
        "temporary_vae_decodes": 1,
        "shifted_dit_forwards": shifted_evaluations,
        "finished_before_transition_innovation": True,
    }


def sample_one(
    *,
    model: Any,
    diffusion: Any,
    vae: Any,
    strict: ModuleType,
    method_core: ModuleType,
    global_seed: int,
    class_id: int,
    observe_hooks: bool = True,
) -> tuple[torch.Tensor, dict[str, np.ndarray], dict[str, Any]]:
    pair_seed = derive_pair_seed(global_seed, class_id)
    torch.manual_seed(pair_seed)
    device = torch.device("cuda")
    after_seed = strict.cuda_rng_state_sha256()
    initial = torch.randn(1, 4, 32, 32, device=device)
    after_initial = strict.cuda_rng_state_sha256()
    state = torch.cat((initial, initial), dim=0)
    y = torch.tensor([class_id, strict.NULL_CLASS_ID], device=device)
    kwargs = {"y": y, "cfg_scale": strict.CFG_SCALE}
    checkpoint_lookup = {step: index for index, step in enumerate(CHECKPOINTS)}
    captured: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "state_before",
            "pred_xstart",
            "p_standard_deviation",
            "transition_innovation",
        )
    }
    alpha_schedule = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    observation_records: list[dict[str, Any]] = []
    for sampling_step, internal_t in enumerate(range(249, -1, -1)):
        checkpoint_index = checkpoint_lookup.get(sampling_step)
        rng_before_prediction = strict.cuda_rng_state_sha256() if checkpoint_index is not None else None
        t = torch.full((2,), internal_t, device=device, dtype=torch.long)
        out = diffusion.p_mean_variance(
            model.forward_with_cfg,
            state,
            t,
            clip_denoised=False,
            model_kwargs=kwargs,
        )
        if checkpoint_index is not None:
            captured["state_before"].append(
                np.ascontiguousarray(state[0].detach().cpu().numpy(), dtype=np.float32)
            )
            captured["pred_xstart"].append(
                np.ascontiguousarray(out["pred_xstart"][0].detach().cpu().numpy(), dtype=np.float32)
            )
            captured["p_standard_deviation"].append(
                np.ascontiguousarray(
                    torch.exp(0.5 * out["log_variance"])[0].detach().cpu().numpy(),
                    dtype=np.float32,
                )
            )
            if observe_hooks:
                record = _run_preinnovation_observation(
                    model=model,
                    vae=vae,
                    diffusion=diffusion,
                    strict=strict,
                    method_core=method_core,
                    state=state,
                    pred_xstart=out["pred_xstart"],
                    internal_t=internal_t,
                    checkpoint_index=checkpoint_index,
                    model_kwargs=kwargs,
                )
                if record["cuda_rng_sha256_before_and_after"] != rng_before_prediction:
                    raise RuntimeError("current prediction or observation changed CUDA RNG")
                observation_records.append(record)
        # This is the sole transition draw.  It is always full 2B and is also
        # consumed at t=0 before the nonzero mask nulls its effect.
        innovation_2b = torch.randn_like(state)
        if checkpoint_index is not None:
            captured["transition_innovation"].append(
                np.ascontiguousarray(innovation_2b[0].detach().cpu().numpy(), dtype=np.float32)
            )
        nonzero = (t != 0).float().view(-1, 1, 1, 1)
        state = (
            out["mean"]
            + nonzero * torch.exp(0.5 * out["log_variance"]) * innovation_2b
        ).detach()
    after_diffusion = strict.cuda_rng_state_sha256()
    kept, discarded = state.chunk(2, dim=0)
    decoded = vae.decode(kept / strict.VAE_SCALING_FACTOR).sample
    torch.cuda.synchronize()
    arrays: dict[str, np.ndarray] = {
        "state_before": np.ascontiguousarray(np.stack(captured["state_before"]), dtype=np.float32),
        "pred_xstart": np.ascontiguousarray(np.stack(captured["pred_xstart"]), dtype=np.float32),
        "p_standard_deviation": np.ascontiguousarray(
            np.stack(captured["p_standard_deviation"]), dtype=np.float32
        ),
        "transition_innovation": np.ascontiguousarray(
            np.stack(captured["transition_innovation"]), dtype=np.float32
        ),
        "sampling_step": np.asarray(CHECKPOINTS, dtype=np.int16),
        "internal_timestep": np.asarray(INTERNAL_TIMESTEPS, dtype=np.int16),
        "alpha_bar": np.asarray(
            [alpha_schedule[value] for value in INTERNAL_TIMESTEPS], dtype=np.float64
        ),
    }
    validate_arrays(arrays)
    execution = {
        "derived_torch_seed": pair_seed,
        "rng_state_sha256": {
            "after_pair_seed_reset": after_seed,
            "after_initial_noise": after_initial,
            "after_250_full_2B_transition_draws": after_diffusion,
        },
        "tensor_sha256": {
            "initial_noise_b1": strict.tensor_sha256(initial),
            "final_latent_kept_b1": strict.tensor_sha256(kept),
            "final_latent_discarded_b1": strict.tensor_sha256(discarded),
            "decoded_sample_b1": strict.tensor_sha256(decoded),
        },
        "transition_randn_like_calls": 250,
        "transition_draw_shape": [2, 4, 32, 32],
        "terminal_t0_draw_consumed_then_masked": True,
        "preinnovation_observation_enabled": observe_hooks,
        "preinnovation_observation_count": len(observation_records),
        "all_observations_rng_neutral": bool(
            observe_hooks
            and len(observation_records) == len(CHECKPOINTS)
            and all(row["finished_before_transition_innovation"] for row in observation_records)
        ),
    }
    return decoded, arrays, execution


def run_sample(args: argparse.Namespace) -> None:
    contract, _, strict, method_core = load_source_lock(args.source_lock)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready")
    validate_method_lock(Path(contract["method_lock"]["path"]))
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    key = (args.phase, args.global_seed, args.class_id)
    if key not in set(exact_pairs(plan)):
        raise RuntimeError(f"pair is outside immutable v4 axis: {key}")
    output_root = args.output_root.expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    for phase in ("calibration", "confirmation"):
        (output_root / METHOD_TREE / "pairs" / phase).mkdir(parents=True, exist_ok=True)
    (output_root / REVIEW_TREE / "pairs" / "confirmation").mkdir(
        parents=True, exist_ok=True
    )
    if any(
        os.path.lexists(output_root / tree / name)
        for tree, name in (
            (METHOD_TREE, TRACE_POOL_MANIFEST),
            (METHOD_TREE, TRACE_POOL_COMPLETION),
            (REVIEW_TREE, ENDPOINT_POOL_MANIFEST),
            (REVIEW_TREE, ENDPOINT_POOL_COMPLETION),
        )
    ):
        raise RuntimeError("trace pool is already finalized")
    trace_destination = output_root / METHOD_TREE / pair_relative_directory(*key)
    endpoint_destination = output_root / REVIEW_TREE / pair_relative_directory(*key)
    has_endpoint = args.phase == "confirmation"
    if os.path.lexists(trace_destination) or (
        has_endpoint and os.path.lexists(endpoint_destination)
    ):
        if not os.path.lexists(trace_destination) or (
            has_endpoint and not os.path.lexists(endpoint_destination)
        ):
            raise RuntimeError("partial cross-tree pair exists")
        trace_record = validate_trace_pair(
            output_root / METHOD_TREE,
            contract=contract,
            plan=plan,
            phase=args.phase,
            global_seed=args.global_seed,
            class_id=args.class_id,
        )
        if has_endpoint:
            validate_endpoint_pair(
                output_root / REVIEW_TREE,
                contract=contract,
                plan=plan,
                phase=args.phase,
                global_seed=args.global_seed,
                class_id=args.class_id,
            )
        print(json.dumps(trace_record, sort_keys=True))
        return
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    previous_grad = torch.is_grad_enabled()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    try:
        model, diffusion, vae, save_image = load_models(strict, contract, args)
        started = time.time()
        decoded, arrays, execution = sample_one(
            model=model,
            diffusion=diffusion,
            vae=vae,
            strict=strict,
            method_core=method_core,
            global_seed=args.global_seed,
            class_id=args.class_id,
        )
        record = publish_pair(
            output_root,
            decoded=decoded,
            arrays=arrays,
            execution=execution,
            save_image=save_image,
            contract=contract,
            plan=plan,
            phase=args.phase,
            global_seed=args.global_seed,
            class_id=args.class_id,
        )
        print(json.dumps({"elapsed_seconds": time.time() - started, "record": record}, sort_keys=True))
    finally:
        torch.set_grad_enabled(previous_grad)
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def finalize_pool(args: argparse.Namespace) -> None:
    contract, _, _, _ = load_source_lock(args.source_lock)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready")
    validate_method_lock(Path(contract["method_lock"]["path"]))
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    root = require_directory(args.output_root, "trace pool")
    trace_root = require_directory(root / METHOD_TREE, "method-only trace tree")
    endpoint_root = require_directory(root / REVIEW_TREE, "review-only endpoint tree")
    if any(
        os.path.lexists(base / name)
        for base, name in (
            (trace_root, TRACE_POOL_MANIFEST),
            (trace_root, TRACE_POOL_COMPLETION),
            (endpoint_root, ENDPOINT_POOL_MANIFEST),
            (endpoint_root, ENDPOINT_POOL_COMPLETION),
        )
    ):
        raise RuntimeError("refusing to overwrite pool finalization")
    trace_records = [
        validate_trace_pair(
            trace_root,
            contract=contract,
            plan=plan,
            phase=phase,
            global_seed=seed,
            class_id=class_id,
        )
        for phase, seed, class_id in exact_pairs(plan)
    ]
    endpoint_records = [
        validate_endpoint_pair(
            endpoint_root,
            contract=contract,
            plan=plan,
            phase=phase,
            global_seed=seed,
            class_id=class_id,
        )
        for phase, seed, class_id in exact_pairs(plan, phases=("confirmation",))
    ]
    trace_manifest = {
        "schema_version": 1,
        "status": "complete",
        "artifact_kind": "SCIENTIFIC_V4_B_E_MINIMUM_TRACE_POOL",
        "dynamic_contract_identity_sha256": contract["identity_sha256"],
        "scientific_protocol_identity_sha256": protocol["identity_sha256"],
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "trace_plan_identity_sha256": plan["identity_sha256"],
        "ordered_pair_axis_sha256": canonical_sha256(
            [
                {"phase": phase, "global_seed": seed, "class_id": class_id}
                for phase, seed, class_id in exact_pairs(plan)
            ]
        ),
        "pair_count": len(trace_records),
        "pairs": trace_records,
        "labels_reviews_endpoint_metrics_or_external_representations_opened": False,
        "scores_thresholds_alerts_or_interventions_computed": False,
    }
    trace_manifest["identity_sha256"] = canonical_sha256(trace_manifest)
    endpoint_manifest = {
        "schema_version": 1,
        "status": "complete",
        "sampling_protocol_identity_sha256": contract[
            "endpoint_sampling_source"
        ]["sampling_protocol_identity_sha256"],
        "event_protocol_identity_sha256": protocol["identity_sha256"],
        "anchor_plan_identity_sha256": plan["identity_sha256"],
        "execution_plan_sha256": sha256_file(args.trace_plan),
        "class_count": len(plan["selected_classes"]),
        "global_seed_count": len(CONFIRMATION_SEEDS),
        "endpoint_count": len(endpoint_records),
        "pair_outputs": endpoint_records,
        "runner_logs": [],
        "endpoint_only": True,
        "trace_saved": False,
        "labels_reviews_metrics_features_embeddings_or_scores_read": False,
    }
    endpoint_manifest["identity_sha256"] = canonical_sha256(endpoint_manifest)
    write_json_exclusive(trace_root / TRACE_POOL_MANIFEST, trace_manifest)
    write_json_exclusive(
        trace_root / TRACE_POOL_COMPLETION,
        {
            "complete": True,
            "manifest_identity_sha256": trace_manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(trace_root / TRACE_POOL_MANIFEST),
            "pair_count": len(trace_records),
        },
    )
    write_json_exclusive(endpoint_root / ENDPOINT_POOL_MANIFEST, endpoint_manifest)
    write_json_exclusive(
        endpoint_root / ENDPOINT_POOL_COMPLETION,
        {
            "complete": True,
            "pool_identity_sha256": endpoint_manifest["identity_sha256"],
            "pool_manifest_sha256": sha256_file(endpoint_root / ENDPOINT_POOL_MANIFEST),
            "execution_plan_sha256": sha256_file(args.trace_plan),
            "endpoint_count": len(endpoint_records),
        },
    )
    validate_review_tree_firewall(endpoint_root)
    print(
        json.dumps(
            {
                "trace_pool_identity_sha256": trace_manifest["identity_sha256"],
                "endpoint_pool_identity_sha256": endpoint_manifest["identity_sha256"],
            },
            sort_keys=True,
        )
    )


def audit_no_touch(args: argparse.Namespace) -> None:
    """GPU replay proving observation hooks leave the baseline trajectory intact."""

    contract, source_manifest, strict, method_core = load_source_lock(args.source_lock)
    if contract.get("execution_ready") is not True:
        raise RuntimeError("dynamic source lock is not execution-ready")
    validate_method_lock(Path(contract["method_lock"]["path"]))
    _, protocol = validate_scientific_protocol(Path(contract["scientific_protocol"]["path"]))
    plan = validate_trace_plan(args.trace_plan, protocol)
    trace_pool_manifest, _ = validate_trace_pool(
        require_directory(
            args.output_root.expanduser().absolute() / METHOD_TREE,
            "method-only trace pool for no-touch audit",
        ),
        contract=contract,
        protocol=protocol,
        plan=plan,
    )
    key = (args.phase, args.global_seed, args.class_id)
    if key != fixed_no_touch_pair(plan):
        raise RuntimeError(
            "no-touch audit must use the sole pre-registered pair "
            f"{fixed_no_touch_pair(plan)!r}"
        )
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    previous_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        model, diffusion, vae, _ = load_models(strict, contract, args)
        base_decoded, base_arrays, base_execution = sample_one(
            model=model,
            diffusion=diffusion,
            vae=vae,
            strict=strict,
            method_core=method_core,
            global_seed=args.global_seed,
            class_id=args.class_id,
            observe_hooks=False,
        )
        observed_decoded, observed_arrays, observed_execution = sample_one(
            model=model,
            diffusion=diffusion,
            vae=vae,
            strict=strict,
            method_core=method_core,
            global_seed=args.global_seed,
            class_id=args.class_id,
            observe_hooks=True,
        )
        if any(
            not np.array_equal(base_arrays[name], observed_arrays[name])
            for name in TRACE_ARRAYS
        ):
            raise RuntimeError("observation hook perturbed a saved trace array")
        if strict.tensor_sha256(base_decoded) != strict.tensor_sha256(observed_decoded):
            raise RuntimeError("observation hook perturbed endpoint pixels/tensor")
        for key_name in (
            "after_pair_seed_reset",
            "after_initial_noise",
            "after_250_full_2B_transition_draws",
        ):
            if (
                base_execution["rng_state_sha256"][key_name]
                != observed_execution["rng_state_sha256"][key_name]
            ):
                raise RuntimeError(f"observation hook perturbed RNG state: {key_name}")
        baseline_array_hashes = {
            name: sha256_array(base_arrays[name]) for name in TRACE_ARRAYS
        }
        observed_array_hashes = {
            name: sha256_array(observed_arrays[name]) for name in TRACE_ARRAYS
        }
        receipt = {
            "schema_version": 1,
            "status": "PASS_OBSERVATION_NO_TOUCH",
            "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "dynamic_contract_identity_sha256": contract["identity_sha256"],
            "scientific_protocol_identity_sha256": protocol["identity_sha256"],
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "trace_plan_identity_sha256": plan["identity_sha256"],
            "trace_plan_file_sha256": sha256_file(require_regular(args.trace_plan, "trace plan")),
            "trace_pool_identity_sha256": trace_pool_manifest["identity_sha256"],
            "trace_pool_ordered_pair_axis_sha256": trace_pool_manifest[
                "ordered_pair_axis_sha256"
            ],
            "confirmation_ordered_pair_axis_sha256": canonical_sha256(
                [
                    {"phase": phase, "global_seed": seed, "class_id": class_id}
                    for phase, seed, class_id in exact_pairs(
                        plan, phases=("confirmation",)
                    )
                ]
            ),
            "asset_identities": contract["assets"],
            "asset_identities_sha256": canonical_sha256(contract["assets"]),
            "pair": {
                "phase": args.phase,
                "global_seed": args.global_seed,
                "class_id": args.class_id,
            },
            "derived_torch_seed": derive_pair_seed(args.global_seed, args.class_id),
            "baseline_trace_array_sha256": baseline_array_hashes,
            "observed_trace_array_sha256": observed_array_hashes,
            "baseline_rng_boundary_sha256": base_execution["rng_state_sha256"],
            "observed_rng_boundary_sha256": observed_execution["rng_state_sha256"],
            "all_trace_arrays_bitwise_equal": True,
            "endpoint_tensor_sha256_equal": True,
            "rng_boundaries_equal": True,
            "baseline_endpoint_tensor_sha256": strict.tensor_sha256(base_decoded),
            "observed_endpoint_tensor_sha256": strict.tensor_sha256(observed_decoded),
            "labels_reviews_external_representations_opened": False,
        }
        receipt["identity_sha256"] = canonical_sha256(receipt)
        if args.receipt is not None:
            publish_artifact(
                args.receipt,
                artifact_kind="SCIENTIFIC_V4_OBSERVATION_NO_TOUCH_AUDIT",
                payloads={
                    "no_touch_receipt.json": json.dumps(
                        receipt, indent=2, sort_keys=True
                    )
                    + "\n"
                },
                manifest_fields={
                    "dynamic_source_lock_manifest_identity_sha256": source_manifest[
                        "identity_sha256"
                    ],
                    "dynamic_contract_identity_sha256": contract["identity_sha256"],
                    "scientific_protocol_identity_sha256": protocol["identity_sha256"],
                    "method_lock_identity_sha256": METHOD_LOCK_ID,
                    "trace_plan_identity_sha256": plan["identity_sha256"],
                    "trace_pool_identity_sha256": receipt[
                        "trace_pool_identity_sha256"
                    ],
                    "confirmation_ordered_pair_axis_sha256": receipt[
                        "confirmation_ordered_pair_axis_sha256"
                    ],
                    "pair": receipt["pair"],
                    "receipt_identity_sha256": receipt["identity_sha256"],
                },
            )
        print(json.dumps(receipt, sort_keys=True))
    finally:
        torch.set_grad_enabled(previous_grad)
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-lock", type=Path, default=DEFAULT_DYNAMIC_SOURCE_LOCK)
    common.add_argument("--trace-plan", type=Path, required=True)
    common.add_argument("--output-root", type=Path, required=True)
    sample = sub.add_parser("sample", parents=[common])
    sample.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    sample.add_argument("--global-seed", type=int, required=True)
    sample.add_argument("--class-id", type=int, required=True)
    sample.add_argument("--dit-root", type=Path, required=True)
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--vae-snapshot", type=Path, required=True)
    final = sub.add_parser("finalize", parents=[common])
    audit = sub.add_parser("audit-no-touch", parents=[common])
    audit.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    audit.add_argument("--global-seed", type=int, required=True)
    audit.add_argument("--class-id", type=int, required=True)
    audit.add_argument("--dit-root", type=Path, required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--vae-snapshot", type=Path, required=True)
    audit.add_argument("--receipt", type=Path)
    sample.set_defaults(func=run_sample)
    final.set_defaults(func=finalize_pool)
    audit.set_defaults(func=audit_no_touch)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
