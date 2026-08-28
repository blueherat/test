#!/usr/bin/env python3
"""Extract auditable bad/good discovery metrics from completed custom DiT traces.

This program is intentionally label-free by default.  It reads one or more
completed outputs of ``trace_dit_imagenet256_custom_batch.py``, verifies their
manifests, completion receipts, source snapshots, archive/array hashes, path
replay, CFG semantics, and decoded endpoint pixels, and then emits a broad but
fixed inventory of trajectory and endpoint features.

An optional *locked consensus* file can be joined only after label-free feature
extraction.  In that mode the program reports discovery-only univariate AUCs;
it never reads reviewer drafts, fits a classifier, chooses a threshold, or
authorizes an intervention.  Existing output directories are never changed or
overwritten.

Time convention
---------------
Sampling step ``k=0..249`` corresponds exactly to internal diffusion timestep
``t=249..0``.  Predictable tracks are known after the model evaluation and
before the transition draw at their indicated step.  Online-causal innovation
tracks are known only after the corresponding transition (or after the next
prediction for a temporal difference).  Although the recorder retains the
raw random draw at ``t=0`` for RNG fidelity, the upstream nonzero mask discards
it; every innovation and transition-density feature therefore has 249 rows,
for ``k=0..248`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUNNER_NAME = "trace_dit_imagenet256_custom_batch"
TRACE_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 1
STEPS = 250
CHANNELS = 4
LATENT_SIZE = 32
IMAGE_SIZE = 256
EPS = 1e-12
MAD_FACTOR = 1.4826
MAD_FLOOR = 1e-6
PHASES = (
    ("q0", 0, 50, "sampling k=0..49; internal t=249..200"),
    ("q1", 50, 100, "sampling k=50..99; internal t=199..150"),
    ("q2", 100, 150, "sampling k=100..149; internal t=149..100"),
    ("q3", 150, 200, "sampling k=150..199; internal t=99..50"),
    ("q4", 200, 250, "sampling k=200..249; internal t=49..0"),
)
STEP_ARRAY_NAMES = (
    "state_before",
    "pred_xstart",
    "p_mean",
    "p_standard_deviation",
    "transition_innovation",
    "conditional_epsilon_raw",
    "unconditional_epsilon_raw",
    "conditional_variance_values_raw",
    "unconditional_variance_values_raw",
)
TRACE_ARRAY_NAMES = (
    *STEP_ARRAY_NAMES,
    "final_latents",
    "decoded_images",
    "internal_timestep",
    "alpha_bar",
)
SOURCE_SNAPSHOT_BINDINGS = {
    "runner_source.py": "runner_source",
    "custom_baseline_helper.py": "custom_baseline_helper",
    "strict_reproduction_helper.py": "strict_reproduction_helper",
}
IDENTIFIER_COLUMNS = (
    "sample_index",
    "run_index",
    "global_seed",
    "class_slot",
    "class_id",
    "trace_dir",
    "endpoint_png_path",
    "label",
    "raw_consensus_label",
)


PROTOCOL: dict[str, Any] = {
    "schema_version": ANALYSIS_SCHEMA_VERSION,
    "status": "DISCOVERY_ONLY_LABEL_FREE_EXTRACTION_OPTIONAL_LOCKED_LABEL_JOIN",
    "sampling_axis": {
        "sampling_step": "k=0..249",
        "internal_timestep": "t=249-k",
        "predictable": "known after current model call and before its transition draw",
        "online_causal": (
            "known after the realized transition, or after the next model call for "
            "an adjacent-prediction difference"
        ),
        "retrospective": "requires a whole-path reduction or decoded endpoint",
        "masked_t0_innovation": (
            "retained by the trace for RNG fidelity but excluded from all innovation, "
            "transition-NLL, and innovation-alignment metrics"
        ),
    },
    "phases": [
        {"name": name, "start_inclusive": start, "stop_exclusive": stop, "meaning": text}
        for name, start, stop, text in PHASES
    ],
    "feature_families": [
        "predicted_clean",
        "state_control",
        "reconstructed_epsilon_control",
        "reverse_variance_head",
        "operational_reverse_kernel",
        "realized_innovation",
        "raw_conditional_unconditional_epsilon_gap",
        "raw_conditional_unconditional_variance_gap",
        "conditional_unconditional_predicted_clean_disagreement",
        "innovation_cfg_gap_alignment",
        "endpoint_structure",
        "fixed_label_free_reference_score",
    ],
    "scalar_reductions": {
        "fixed_phase": [
            "mean",
            "standard_deviation",
            "max_positive_jump",
            "max_negative_jump",
            "centered_cusum_range",
        ],
        "whole_track_retrospective": [
            "mean",
            "maximum",
            "minimum",
            "terminal",
            "total_variation",
            "max_positive_jump",
            "max_negative_jump",
            "max_drawup",
            "max_drawdown",
            "centered_cusum_range",
            "max_abs_second_difference",
            "difference_sign_flip_rate",
        ],
        "centered_cusum_definition": (
            "range of cumulative (value-median)/(1.4826*MAD+1e-6), divided by sqrt(n); "
            "the 1e-6 term is an additive numerical stabilizer"
        ),
    },
    "fixed_simple_score": {
        "name": "fixed_two_phase_predicted_clean_score_label_free_reference",
        "formula": (
            "(mean_{k=50..99} z_all(pred_xstart normalized Dirichlet mean) + "
            "mean_{k=100..149} z_all(pred_xstart log spatial variance mean))/sqrt(2)"
        ),
        "reference": (
            "per-step median and max(1.4826*MAD, 1e-6) over every extracted sample; "
            "labels are never used"
        ),
        "warning": (
            "the reference is transductive in this discovery output and must be frozen "
            "before prospective use"
        ),
    },
    "label_policy": {
        "default": "no labels",
        "allowed": "one completed locked consensus artifact only",
        "forbidden": "reviewer draft/sealed files and threshold fitting",
        "analysis": "univariate discovery-only AUC/AP, with no p-values",
    },
}


@dataclass(frozen=True)
class TrackSpec:
    name: str
    family: str
    availability: str
    observation_offset_steps: int
    formula: str
    length: int
    uses_realized_innovation: bool = False
    deployment_note: str = "none"


@dataclass(frozen=True)
class TraceRecord:
    root: Path
    global_seed: int
    classes: tuple[int, ...]
    identity_sha256: str
    manifest_sha256: str
    completion_sha256: str
    trace_sha256: str
    source_snapshot_sha256: dict[str, str]
    scientific_fingerprint_sha256: str
    cfg_scale: float
    cfg_epsilon_channels: int
    reconstruction_max_abs_error: float
    variance_reconstruction_max_logstd_error: float


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_regular(path: Path, description: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} is missing, non-regular, or a symlink: {path}")


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or relative in {"", "."}:
        raise RuntimeError(f"unsafe manifest relative path: {relative!r}")
    resolved = (root / candidate).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError(f"manifest path escapes trace root: {relative!r}")
    return resolved


def _expected_shapes(batch: int) -> dict[str, tuple[int, ...]]:
    step = (batch, STEPS, CHANNELS, LATENT_SIZE, LATENT_SIZE)
    return {
        **{name: step for name in STEP_ARRAY_NAMES},
        "final_latents": (batch, CHANNELS, LATENT_SIZE, LATENT_SIZE),
        "decoded_images": (batch, 3, IMAGE_SIZE, IMAGE_SIZE),
        "internal_timestep": (STEPS,),
        "alpha_bar": (STEPS,),
    }


def _expected_dtypes() -> dict[str, np.dtype[Any]]:
    return {
        **{
            name: np.dtype(np.float32)
            for name in (*STEP_ARRAY_NAMES, "final_latents", "decoded_images")
        },
        "internal_timestep": np.dtype(np.int16),
        "alpha_bar": np.dtype(np.float64),
    }


def _inspect_png(path: Path, mode: str, size: tuple[int, int]) -> dict[str, Any]:
    _require_regular(path, "PNG")
    with Image.open(path) as image:
        image.load()
        observed_mode = image.mode
        observed_size = tuple(image.size)
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if observed_mode != mode or observed_size != size:
        raise RuntimeError(
            f"PNG properties changed for {path}: {observed_mode}/{observed_size}"
        )
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_hash,
        "mode": observed_mode,
        "size": list(observed_size),
    }


def _scientific_fingerprint(identity: Mapping[str, Any]) -> str:
    protocol = dict(identity.get("protocol", {}))
    protocol.pop("global_torch_seed", None)
    payload = {
        "runner": identity.get("runner"),
        "schema": identity.get("schema"),
        "runner_source_sha256": identity.get("runner_source", {}).get("sha256"),
        "custom_baseline_helper_sha256": identity.get("custom_baseline_helper", {}).get(
            "sha256"
        ),
        "strict_reproduction_helper_sha256": identity.get(
            "strict_reproduction_helper", {}
        ).get("sha256"),
        "protocol_without_seed": protocol,
        "source": identity.get("source"),
        "checkpoint": identity.get("checkpoint"),
        "vae_snapshot": identity.get("vae_snapshot"),
        "dependencies": identity.get("dependencies"),
    }
    return canonical_sha256(payload)


def _validate_output_records(
    root: Path, manifest: Mapping[str, Any], classes: Sequence[int]
) -> dict[str, Mapping[str, Any]]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs or not all(
        isinstance(item, dict) for item in outputs
    ):
        raise RuntimeError(f"malformed trace output records: {root}")
    if manifest.get("outputs_sha256") != canonical_sha256(outputs):
        raise RuntimeError(f"trace outputs aggregate hash changed: {root}")
    by_relative: dict[str, Mapping[str, Any]] = {}
    for record in outputs:
        relative = record.get("relative_path")
        if not isinstance(relative, str) or relative in by_relative:
            raise RuntimeError(f"duplicate or invalid trace output path: {root}")
        path = _safe_relative(root, relative)
        _require_regular(path, "trace payload")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"trace payload size/hash changed: {path}")
        if relative.endswith(".png"):
            expected_mode = record.get("mode")
            expected_size = tuple(record.get("size", []))
            observed = _inspect_png(path, expected_mode, expected_size)  # type: ignore[arg-type]
            if any(observed.get(key) != record.get(key) for key in observed):
                raise RuntimeError(f"trace PNG record changed: {path}")
        by_relative[relative] = record

    expected = {
        "sample.png",
        "trace.npz",
        *SOURCE_SNAPSHOT_BINDINGS,
        *{
            f"images/{slot:02d}_class{class_id:04d}.png"
            for slot, class_id in enumerate(classes)
        },
    }
    if set(by_relative) != expected:
        raise RuntimeError(
            f"trace output member set differs from final source-snapshot schema: "
            f"{set(by_relative)} != {expected}"
        )
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"trace output contains a symlink: {path}")
        if path.is_dir():
            observed_dirs.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise RuntimeError(f"trace output contains a non-regular entry: {path}")
    if observed_dirs != {"images"} or observed_files != {
        *expected,
        "manifest.json",
        "completion.json",
    }:
        raise RuntimeError(f"trace output layout changed or contains unbound payloads: {root}")
    return by_relative


def load_validated_trace(root: Path) -> tuple[TraceRecord, dict[str, np.ndarray]]:
    root = root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"trace root is missing, non-directory, or a symlink: {root}")
    root = root.resolve()
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    trace_path = root / "trace.npz"
    for path, description in (
        (manifest_path, "trace manifest"),
        (completion_path, "trace completion receipt"),
        (trace_path, "trace archive"),
    ):
        _require_regular(path, description)
    for snapshot_name in SOURCE_SNAPSHOT_BINDINGS:
        _require_regular(root / snapshot_name, f"trace source snapshot {snapshot_name}")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity")
    if (
        manifest.get("schema") != TRACE_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(identity, dict)
        or identity.get("runner") != RUNNER_NAME
        or identity.get("schema") != TRACE_SCHEMA_VERSION
        or identity.get("observation_only") is not True
        or identity.get("selection") is not None
        or identity.get("intervention") is not None
    ):
        raise RuntimeError(f"trace is not a completed observation-only custom run: {root}")
    identity_hash = canonical_sha256(identity)
    if manifest.get("identity_sha256") != identity_hash:
        raise RuntimeError(f"trace identity hash changed: {root}")
    protocol = identity.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError(f"trace protocol is missing: {root}")
    classes_raw = protocol.get("class_ids_ordered")
    seed = protocol.get("global_torch_seed")
    if (
        not isinstance(classes_raw, list)
        or not classes_raw
        or any(type(value) is not int or not 0 <= value < 1000 for value in classes_raw)
        or len(set(classes_raw)) != len(classes_raw)
        or type(seed) is not int
    ):
        raise RuntimeError(f"invalid ordered classes or seed in trace protocol: {root}")
    classes = tuple(classes_raw)
    if (
        protocol.get("sampling_steps") != STEPS
        or protocol.get("trace_axis_order") != "[B, sampling_step, C, H, W]"
        or protocol.get("internal_timestep_order") != "249..0"
        or protocol.get("batch_size_before_duplication") != len(classes)
        or protocol.get("recorded_slice") != "first B"
        or protocol.get("full_2B_randn_like_each_transition_including_t0") is not True
        or protocol.get("raw_cfg_components_observed_from_same_model_forward") is not True
        or protocol.get("raw_cfg_component_order")
        != "first B conditional, second B unconditional"
        or protocol.get("raw_epsilon_channels") != CHANNELS
        or protocol.get("raw_learned_range_channels") != CHANNELS
    ):
        raise RuntimeError(f"trace protocol no longer has the required semantics: {root}")
    cfg_scale = protocol.get("cfg_scale")
    cfg_channels = protocol.get("cfg_epsilon_channels")
    if not isinstance(cfg_scale, (int, float)) or not np.isfinite(cfg_scale) or cfg_channels != 3:
        raise RuntimeError(f"unsupported CFG scale/channel contract: {root}")

    by_relative = _validate_output_records(root, manifest, classes)
    outputs_hash = canonical_sha256(manifest["outputs"])
    expected_completion = {
        "schema": TRACE_SCHEMA_VERSION,
        "identity_sha256": identity_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "outputs_sha256": outputs_hash,
        "output_count": len(manifest["outputs"]),
    }
    if completion != expected_completion:
        raise RuntimeError(f"trace completion receipt is invalid: {root}")
    snapshot_hashes: dict[str, str] = {}
    for snapshot_name, identity_key in SOURCE_SNAPSHOT_BINDINGS.items():
        source_identity = identity.get(identity_key)
        if not isinstance(source_identity, dict):
            raise RuntimeError(f"trace identity lacks source binding {identity_key}: {root}")
        snapshot_hash = sha256_file(root / snapshot_name)
        if (
            snapshot_hash != source_identity.get("sha256")
            or snapshot_hash != by_relative[snapshot_name].get("sha256")
        ):
            raise RuntimeError(
                f"trace source snapshot differs from identity: {root}/{snapshot_name}"
            )
        snapshot_hashes[snapshot_name] = snapshot_hash

    trace_record = by_relative["trace.npz"]
    recorded_arrays = trace_record.get("arrays")
    identity_arrays = identity.get("trace_arrays")
    if (
        not isinstance(recorded_arrays, dict)
        or not isinstance(identity_arrays, dict)
        or set(recorded_arrays) != set(TRACE_ARRAY_NAMES)
        or set(identity_arrays) != set(TRACE_ARRAY_NAMES)
        or manifest.get("trace_array_records") != recorded_arrays
    ):
        raise RuntimeError(f"trace array schema bindings are malformed: {root}")
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            if set(archive.files) != set(TRACE_ARRAY_NAMES):
                raise RuntimeError(f"trace archive member set changed: {root}")
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in TRACE_ARRAY_NAMES
            }
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load trace archive: {trace_path}") from exc
    shapes = _expected_shapes(len(classes))
    dtypes = _expected_dtypes()
    for name in TRACE_ARRAY_NAMES:
        value = arrays[name]
        record = recorded_arrays[name]
        expected_identity = {"shape": list(shapes[name]), "dtype": dtypes[name].str}
        if identity_arrays[name] != expected_identity:
            raise RuntimeError(f"trace identity schema changed for {name}: {root}")
        if (
            value.shape != shapes[name]
            or value.dtype != dtypes[name]
            or not np.isfinite(value).all()
            or record.get("shape") != list(value.shape)
            or record.get("dtype") != value.dtype.str
            or record.get("raw_sha256") != sha256_array(value)
        ):
            raise RuntimeError(f"trace array shape/dtype/hash/finite audit failed: {name}")

    internal = arrays["internal_timestep"]
    alpha = arrays["alpha_bar"]
    if not np.array_equal(internal, np.arange(249, -1, -1, dtype=np.int16)):
        raise RuntimeError(f"sampling/internal timestep join changed: {root}")
    if (
        np.any(alpha <= 0.0)
        or np.any(alpha > 1.0)
        or np.any(np.diff(alpha) <= 0.0)
        or np.any(arrays["p_standard_deviation"] <= 0.0)
    ):
        raise RuntimeError(f"alpha or reverse-kernel standard deviation is invalid: {root}")

    state = arrays["state_before"]
    mean = arrays["p_mean"]
    pstd = arrays["p_standard_deviation"]
    innovation = arrays["transition_innovation"]
    for step in range(STEPS - 1):
        replayed = mean[:, step] + pstd[:, step] * innovation[:, step]
        if not np.array_equal(replayed, state[:, step + 1]):
            error = float(np.max(np.abs(replayed.astype(np.float64) - state[:, step + 1])))
            raise RuntimeError(f"trace transition replay differs at k={step}; max_abs={error}")
    if (
        not np.array_equal(mean[:, -1], arrays["final_latents"])
        or not np.array_equal(arrays["pred_xstart"][:, -1], arrays["final_latents"])
    ):
        raise RuntimeError(f"deterministic t=0 endpoint identity changed: {root}")

    cond = arrays["conditional_epsilon_raw"]
    uncond = arrays["unconditional_epsilon_raw"]
    guided = cond.copy()
    guided[:, :, :3] = uncond[:, :, :3] + np.float32(cfg_scale) * (
        cond[:, :, :3] - uncond[:, :, :3]
    )
    reconstruction_max = 0.0
    for start in range(0, STEPS, 25):
        stop = min(start + 25, STEPS)
        a = alpha[start:stop][None, :, None, None, None]
        reconstructed = (
            state[:, start:stop].astype(np.float64)
            - np.sqrt(1.0 - a) * guided[:, start:stop].astype(np.float64)
        ) / np.sqrt(a)
        reconstruction_max = max(
            reconstruction_max,
            float(
                np.max(
                    np.abs(
                        reconstructed
                        - arrays["pred_xstart"][:, start:stop].astype(np.float64)
                    )
                )
            ),
        )
    if reconstruction_max > 1e-4:
        raise RuntimeError(
            f"raw CFG branches do not reconstruct pred_xstart: max_abs={reconstruction_max}"
        )

    # DiT uses the learned-range variance parameterization without clipping the
    # raw network head.  Reconstruct the exact conditional reverse standard
    # deviation from the respaced alpha-bar schedule to bind branch order and
    # semantics, including raw values outside [-1, 1].
    alpha_ascending = alpha[::-1]
    alpha_previous = np.concatenate(([1.0], alpha_ascending[:-1]))
    beta = 1.0 - alpha_ascending / alpha_previous
    posterior_variance = beta * (1.0 - alpha_previous) / (1.0 - alpha_ascending)
    posterior_variance[0] = posterior_variance[1]
    if (
        np.any(beta <= 0.0)
        or np.any(beta >= 1.0)
        or np.any(posterior_variance <= 0.0)
        or not np.isfinite(beta).all()
        or not np.isfinite(posterior_variance).all()
    ):
        raise RuntimeError(f"cannot reconstruct learned-range schedule: {root}")
    minimum_log_variance = np.log(posterior_variance)[::-1]
    maximum_log_variance = np.log(beta)[::-1]
    conditional_variance_raw = arrays["conditional_variance_values_raw"]
    variance_reconstruction_max = 0.0
    for start in range(0, STEPS, 25):
        stop = min(start + 25, STEPS)
        fraction = (
            conditional_variance_raw[:, start:stop].astype(np.float64) + 1.0
        ) / 2.0
        reconstructed_logstd = 0.5 * (
            fraction * maximum_log_variance[None, start:stop, None, None, None]
            + (1.0 - fraction)
            * minimum_log_variance[None, start:stop, None, None, None]
        )
        variance_reconstruction_max = max(
            variance_reconstruction_max,
            float(
                np.max(
                    np.abs(
                        reconstructed_logstd
                        - np.log(
                            arrays["p_standard_deviation"][:, start:stop].astype(
                                np.float64
                            )
                        )
                    )
                )
            ),
        )
    if variance_reconstruction_max > 1e-5:
        raise RuntimeError(
            "conditional learned-range head does not reconstruct the operational "
            f"reverse standard deviation: max_logstd_abs={variance_reconstruction_max}"
        )

    decoded = arrays["decoded_images"]
    for slot, class_id in enumerate(classes):
        relative = f"images/{slot:02d}_class{class_id:04d}.png"
        with Image.open(root / relative) as image:
            image.load()
            png = np.asarray(image.convert("RGB"), dtype=np.uint8)
        from_trace = np.floor(
            (np.clip(decoded[slot], -1.0, 1.0) + 1.0) * 127.5 + 0.5
        ).astype(np.uint8).transpose(1, 2, 0)
        if not np.array_equal(from_trace, png):
            raise RuntimeError(f"decoded trace and endpoint PNG differ: {root}/{relative}")

    record = TraceRecord(
        root=root,
        global_seed=seed,
        classes=classes,
        identity_sha256=identity_hash,
        manifest_sha256=sha256_file(manifest_path),
        completion_sha256=sha256_file(completion_path),
        trace_sha256=sha256_file(trace_path),
        source_snapshot_sha256=snapshot_hashes,
        scientific_fingerprint_sha256=_scientific_fingerprint(identity),
        cfg_scale=float(cfg_scale),
        cfg_epsilon_channels=int(cfg_channels),
        reconstruction_max_abs_error=reconstruction_max,
        variance_reconstruction_max_logstd_error=variance_reconstruction_max,
    )
    return record, arrays


def _spatial_statistics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel log variance, log normalized Dirichlet energy, raw energy."""

    if values.ndim != 5 or values.shape[2:] != (CHANNELS, LATENT_SIZE, LATENT_SIZE):
        raise ValueError(f"expected [B,T,4,32,32], got {values.shape}")
    work = values.astype(np.float64, copy=False)
    centered = work - np.mean(work, axis=(-2, -1), keepdims=True)
    variance = np.mean(centered * centered, axis=(-2, -1))
    vertical = np.diff(work, axis=-2)
    horizontal = np.diff(work, axis=-1)
    energy = np.mean(vertical * vertical, axis=(-2, -1)) + np.mean(
        horizontal * horizontal, axis=(-2, -1)
    )
    return (
        np.log(variance + EPS),
        np.log(energy / (variance + EPS) + EPS),
        energy,
    )


def _rms(values: np.ndarray, channels: slice | Sequence[int]) -> np.ndarray:
    selected = values[:, :, channels]
    return np.sqrt(np.mean(selected.astype(np.float64) ** 2, axis=(2, 3, 4)))


def _cosine(
    left: np.ndarray, right: np.ndarray, channels: slice | Sequence[int]
) -> np.ndarray:
    a = left[:, :, channels].astype(np.float64).reshape(left.shape[0], left.shape[1], -1)
    b = right[:, :, channels].astype(np.float64).reshape(right.shape[0], right.shape[1], -1)
    return np.sum(a * b, axis=2) / (
        np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) + EPS
    )


def _tile_concentration(
    values: np.ndarray, channels: slice | Sequence[int], grid: int = 4
) -> np.ndarray:
    selected = values[:, :, channels].astype(np.float64)
    energy = selected * selected
    height, width = energy.shape[-2:]
    if height % grid or width % grid:
        raise ValueError("latent spatial dimensions are not divisible by tile grid")
    tile_h, tile_w = height // grid, width // grid
    tiles = []
    for row in range(grid):
        for column in range(grid):
            tile = energy[
                ...,
                row * tile_h : (row + 1) * tile_h,
                column * tile_w : (column + 1) * tile_w,
            ]
            tiles.append(np.mean(tile, axis=(2, 3, 4)))
    stacked = np.stack(tiles, axis=2)
    return np.max(stacked, axis=2) / (np.sum(stacked, axis=2) + EPS)


def _centered_cusum_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    standardized = (values - median) / (MAD_FACTOR * mad + MAD_FLOOR)
    cumulative = np.concatenate(([0.0], np.cumsum(standardized)))
    return float((np.max(cumulative) - np.min(cumulative)) / math.sqrt(len(values)))


def _max_drawup_drawdown(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    running_min = np.minimum.accumulate(values)
    running_max = np.maximum.accumulate(values)
    return float(np.max(values - running_min)), float(np.max(running_max - values))


def _trajectory_reductions(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("trajectory reduction expects a finite vector of length >= 2")
    difference = np.diff(values)
    drawup, drawdown = _max_drawup_drawdown(values)
    second = np.diff(values, n=2)
    nonzero = difference[np.abs(difference) > EPS]
    sign_flip = (
        float(np.mean(nonzero[:-1] * nonzero[1:] < 0.0)) if len(nonzero) >= 2 else 0.0
    )
    return {
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "minimum": float(np.min(values)),
        "terminal": float(values[-1]),
        "total_variation": float(np.sum(np.abs(difference))),
        "max_positive_jump": float(max(0.0, np.max(difference))),
        "max_negative_jump": float(max(0.0, np.max(-difference))),
        "max_drawup": drawup,
        "max_drawdown": drawdown,
        "centered_cusum_range": _centered_cusum_range(values),
        "max_abs_second_difference": float(
            np.max(np.abs(second)) if len(second) else 0.0
        ),
        "difference_sign_flip_rate": sign_flip,
    }


def _phase_reductions(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    difference = np.diff(values)
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "max_positive_jump": float(max(0.0, np.max(difference))) if len(difference) else 0.0,
        "max_negative_jump": float(max(0.0, np.max(-difference))) if len(difference) else 0.0,
        "centered_cusum_range": _centered_cusum_range(values),
    }


def _timing(
    spec: TrackSpec, *, start: int | None = None, stop: int | None = None
) -> dict[str, Any]:
    if start is None or stop is None:
        latest = spec.length - 1 + spec.observation_offset_steps
        if not 0 <= latest <= 249:
            raise ValueError(f"invalid full-track latest step for {spec.name}: {latest}")
        return {
            "availability": "retrospective",
            "latest_required_sampling_step": latest,
            "latest_required_internal_timestep": 249 - latest,
            "observation_timing": "whole_path_reduction",
            "preterminal_actionable": latest < 249,
        }
    bounded_stop = min(stop, spec.length)
    if start >= bounded_stop:
        raise ValueError(f"phase has no rows for track {spec.name}")
    latest = bounded_stop - 1 + spec.observation_offset_steps
    if not 0 <= latest <= 249:
        raise ValueError(f"invalid latest step for {spec.name}: {latest}")
    return {
        "availability": spec.availability,
        "latest_required_sampling_step": latest,
        "latest_required_internal_timestep": 249 - latest,
        "observation_timing": (
            "before_transition_at_latest_step"
            if spec.observation_offset_steps == 0
            else "after_transition_or_next_prediction_at_latest_step"
        ),
        "preterminal_actionable": latest < 249,
    }


def _catalog_entry(
    name: str,
    spec: TrackSpec,
    reduction: str,
    formula: str,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "feature": name,
        "track": spec.name,
        "family": spec.family,
        "reduction": reduction,
        "track_formula": spec.formula,
        "feature_formula": formula,
        "track_length": spec.length,
        "availability": timing["availability"],
        "latest_required_sampling_step": timing["latest_required_sampling_step"],
        "latest_required_internal_timestep": timing[
            "latest_required_internal_timestep"
        ],
        "observation_timing": timing["observation_timing"],
        "preterminal_actionable": timing["preterminal_actionable"],
        "uses_realized_innovation": spec.uses_realized_innovation,
        "deployment_note": spec.deployment_note,
    }


def reduce_tracks(
    tracks: Mapping[str, np.ndarray], specs: Mapping[str, TrackSpec]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if set(tracks) != set(specs):
        raise RuntimeError("track/spec name sets differ")
    sample_count = next(iter(tracks.values())).shape[0]
    rows: list[dict[str, float]] = [dict() for _ in range(sample_count)]
    catalog: dict[str, dict[str, Any]] = {}
    for track_name in sorted(tracks):
        values = np.asarray(tracks[track_name], dtype=np.float64)
        spec = specs[track_name]
        if values.shape != (sample_count, spec.length) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid extracted track: {track_name} {values.shape}")
        for sample_index in range(sample_count):
            reductions = _trajectory_reductions(values[sample_index])
            for reduction, scalar in reductions.items():
                feature = f"{track_name}__full_{reduction}"
                rows[sample_index][feature] = scalar
                if feature not in catalog:
                    catalog[feature] = _catalog_entry(
                        feature,
                        spec,
                        f"full_{reduction}",
                        f"{reduction} over all {spec.length} rows of {track_name}",
                        _timing(spec),
                    )
        for phase, start, stop, _ in PHASES:
            bounded_stop = min(stop, spec.length)
            if start >= bounded_stop:
                continue
            timing = _timing(spec, start=start, stop=stop)
            for sample_index in range(sample_count):
                reductions = _phase_reductions(values[sample_index, start:bounded_stop])
                for reduction, scalar in reductions.items():
                    feature = f"{track_name}__{phase}_{reduction}"
                    rows[sample_index][feature] = scalar
                    if feature not in catalog:
                        catalog[feature] = _catalog_entry(
                            feature,
                            spec,
                            f"{phase}_{reduction}",
                            (
                                f"{reduction} over {track_name}[{start}:{bounded_stop}] "
                                f"under fixed phase {phase}"
                            ),
                            timing,
                        )
    return pd.DataFrame(rows), catalog


def _add_track(
    tracks: dict[str, np.ndarray],
    specs: dict[str, TrackSpec],
    values: np.ndarray,
    *,
    name: str,
    family: str,
    availability: str,
    observation_offset_steps: int,
    formula: str,
    uses_realized_innovation: bool = False,
    deployment_note: str = "none",
) -> None:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] not in {249, 250} or not np.isfinite(value).all():
        raise RuntimeError(f"track {name} is invalid: {value.shape}")
    spec = TrackSpec(
        name=name,
        family=family,
        availability=availability,
        observation_offset_steps=observation_offset_steps,
        formula=formula,
        length=value.shape[1],
        uses_realized_innovation=uses_realized_innovation,
        deployment_note=deployment_note,
    )
    if name in tracks or name in specs:
        raise RuntimeError(f"duplicate track name: {name}")
    tracks[name] = value
    specs[name] = spec


def _endpoint_features(rgb: np.ndarray) -> dict[str, float]:
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("endpoint must be uint8 RGB 256x256")
    image = rgb.astype(np.float64) / 255.0
    gray = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    sobel_x = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    sobel_y = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    gradient = np.hypot(sobel_x, sobel_y)
    laplacian = ndimage.laplace(gray, mode="reflect")
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - np.mean(gray)))) ** 2
    yy, xx = np.indices(gray.shape)
    radius = np.sqrt(
        (yy - (gray.shape[0] - 1) / 2.0) ** 2
        + (xx - (gray.shape[1] - 1) / 2.0) ** 2
    ) / (min(gray.shape) / 2.0)
    local_laplacian: list[float] = []
    local_sobel: list[float] = []
    for row in range(4):
        for column in range(4):
            region = (
                slice(row * 64, (row + 1) * 64),
                slice(column * 64, (column + 1) * 64),
            )
            local_laplacian.append(float(np.var(laplacian[region])))
            local_sobel.append(float(np.mean(gradient[region])))
    local_lap = np.asarray(local_laplacian)
    local_edge = np.asarray(local_sobel)
    return {
        "endpoint_gray_standard_deviation": float(np.std(gray)),
        "endpoint_sobel_mean": float(np.mean(gradient)),
        "endpoint_sobel_q90": float(np.quantile(gradient, 0.90)),
        "endpoint_sobel_q99": float(np.quantile(gradient, 0.99)),
        "endpoint_edge_density_gt_0p05": float(np.mean(gradient > 0.05)),
        "endpoint_edge_density_gt_0p10": float(np.mean(gradient > 0.10)),
        "endpoint_laplacian_variance": float(np.var(laplacian)),
        "endpoint_laplacian_absolute_mean": float(np.mean(np.abs(laplacian))),
        "endpoint_fft_power_radius_gt_0p4": float(
            np.sum(spectrum[radius > 0.4]) / (np.sum(spectrum) + EPS)
        ),
        "endpoint_fft_power_radius_gt_0p5": float(
            np.sum(spectrum[radius > 0.5]) / (np.sum(spectrum) + EPS)
        ),
        "endpoint_local_laplacian_minimum": float(np.min(local_lap)),
        "endpoint_local_laplacian_q25": float(np.quantile(local_lap, 0.25)),
        "endpoint_local_laplacian_cv": float(
            np.std(local_lap) / (np.mean(local_lap) + EPS)
        ),
        "endpoint_local_sobel_minimum": float(np.min(local_edge)),
        "endpoint_local_sobel_q25": float(np.quantile(local_edge, 0.25)),
        "endpoint_local_sobel_cv": float(
            np.std(local_edge) / (np.mean(local_edge) + EPS)
        ),
        "endpoint_gaussian_residual_sigma_0p5_rms": float(
            np.sqrt(
                np.mean(
                    (gray - ndimage.gaussian_filter(gray, sigma=0.5, mode="reflect")) ** 2
                )
            )
        ),
        "endpoint_gaussian_residual_sigma_1p0_rms": float(
            np.sqrt(
                np.mean(
                    (gray - ndimage.gaussian_filter(gray, sigma=1.0, mode="reflect")) ** 2
                )
            )
        ),
    }


def extract_trace_tracks(
    record: TraceRecord, arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, TrackSpec], list[dict[str, Any]]]:
    tracks: dict[str, np.ndarray] = {}
    specs: dict[str, TrackSpec] = {}
    alpha = arrays["alpha_bar"].astype(np.float64)
    pred = arrays["pred_xstart"]
    state = arrays["state_before"]
    cond_eps = arrays["conditional_epsilon_raw"]
    uncond_eps = arrays["unconditional_epsilon_raw"]
    cond_var = arrays["conditional_variance_values_raw"]
    uncond_var = arrays["unconditional_variance_values_raw"]

    pred_logvar, pred_logdir, pred_energy = _spatial_statistics(pred)
    for channel in range(CHANNELS):
        _add_track(
            tracks,
            specs,
            pred_logvar[:, :, channel],
            name=f"pred_xstart_log_spatial_variance_c{channel}",
            family="predicted_clean",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"log(Var_hw(pred_xstart channel {channel}) + 1e-12)",
        )
        _add_track(
            tracks,
            specs,
            pred_logdir[:, :, channel],
            name=f"pred_xstart_log_normalized_dirichlet_c{channel}",
            family="predicted_clean",
            availability="predictable",
            observation_offset_steps=0,
            formula=(
                f"log((mean vertical-difference^2 + mean horizontal-difference^2) / "
                f"(Var_hw + 1e-12) + 1e-12), pred_xstart channel {channel}"
            ),
        )
        _add_track(
            tracks,
            specs,
            alpha[None, :] * pred_energy[:, :, channel],
            name=f"pred_xstart_alpha_compensated_gradient_energy_c{channel}",
            family="predicted_clean",
            availability="predictable",
            observation_offset_steps=0,
            formula=(
                f"alpha_bar[k] * (mean vertical-difference^2 + mean "
                f"horizontal-difference^2), pred_xstart channel {channel}"
            ),
        )
    _add_track(
        tracks,
        specs,
        np.mean(pred_logvar, axis=2),
        name="pred_xstart_log_spatial_variance_mean_channels",
        family="predicted_clean",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean over four channels of log(Var_hw(pred_xstart_c)+1e-12)",
    )
    _add_track(
        tracks,
        specs,
        np.mean(pred_logdir, axis=2),
        name="pred_xstart_log_normalized_dirichlet_mean_channels",
        family="predicted_clean",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean over four channels of amplitude-normalized log Dirichlet energy",
    )
    _add_track(
        tracks,
        specs,
        alpha[None, :] * np.mean(pred_energy, axis=2),
        name="pred_xstart_alpha_compensated_gradient_energy_mean_channels",
        family="predicted_clean",
        availability="predictable",
        observation_offset_steps=0,
        formula="alpha_bar[k] times mean four-channel pred_xstart spatial-gradient energy",
    )

    pred64 = pred.astype(np.float64)
    delta = np.diff(pred64, axis=1)
    delta_energy = np.mean(delta * delta, axis=(-2, -1))
    pair_variance = np.var(pred64[:, :-1], axis=(-2, -1)) + np.var(
        pred64[:, 1:], axis=(-2, -1)
    )
    temporal = delta_energy / (pair_variance + EPS)
    for channel in range(CHANNELS):
        _add_track(
            tracks,
            specs,
            temporal[:, :, channel],
            name=f"pred_xstart_temporal_instability_c{channel}",
            family="predicted_clean",
            availability="online_causal",
            observation_offset_steps=1,
            formula=(
                f"mean_hw((pred[k+1]-pred[k])^2) / "
                f"(Var_hw(pred[k])+Var_hw(pred[k+1])+1e-12), channel {channel}; "
                "row j requires prediction j+1"
            ),
        )
    _add_track(
        tracks,
        specs,
        np.mean(temporal, axis=2),
        name="pred_xstart_temporal_instability_mean_channels",
        family="predicted_clean",
        availability="online_causal",
        observation_offset_steps=1,
        formula="mean over channels of normalized adjacent-prediction squared change",
    )
    del pred64, delta

    state_logvar, state_logdir, _ = _spatial_statistics(state)
    _add_track(
        tracks,
        specs,
        np.mean(state_logvar, axis=2),
        name="state_log_spatial_variance_mean_channels",
        family="state_control",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean_c log(Var_hw(x_k,c)+1e-12)",
    )
    _add_track(
        tracks,
        specs,
        np.mean(state_logdir, axis=2),
        name="state_log_normalized_dirichlet_mean_channels",
        family="state_control",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean_c amplitude-normalized log Dirichlet energy of state x_k",
    )

    a = alpha[None, :, None, None, None]
    reconstructed_eps = (
        state.astype(np.float64) - np.sqrt(a) * pred.astype(np.float64)
    ) / np.sqrt(1.0 - a)
    eps_logvar, eps_logdir, _ = _spatial_statistics(reconstructed_eps)
    _add_track(
        tracks,
        specs,
        np.mean(eps_logvar, axis=2),
        name="reconstructed_epsilon_log_spatial_variance_mean_channels",
        family="reconstructed_epsilon_control",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean_c log(Var_hw((x_k-sqrt(alpha)*pred_x0)/sqrt(1-alpha))+1e-12)",
    )
    _add_track(
        tracks,
        specs,
        np.mean(eps_logdir, axis=2),
        name="reconstructed_epsilon_log_normalized_dirichlet_mean_channels",
        family="reconstructed_epsilon_control",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean_c amplitude-normalized log Dirichlet energy of reconstructed epsilon",
    )

    logstd = np.log(arrays["p_standard_deviation"].astype(np.float64))
    logstd_logvar, logstd_logdir, _ = _spatial_statistics(logstd)
    _add_track(
        tracks,
        specs,
        np.mean(logstd, axis=(2, 3, 4)),
        name="variance_head_logstd_mean_all_dimensions",
        family="reverse_variance_head",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean_{c,h,w} log(p_standard_deviation[k,c,h,w]) including t=0 head",
    )
    _add_track(
        tracks,
        specs,
        np.std(logstd, axis=(2, 3, 4)),
        name="variance_head_logstd_standard_deviation_all_dimensions",
        family="reverse_variance_head",
        availability="predictable",
        observation_offset_steps=0,
        formula="standard deviation_{c,h,w} of reverse-kernel log standard deviation",
    )
    _add_track(
        tracks,
        specs,
        np.mean(logstd_logdir, axis=2),
        name="variance_head_logstd_log_normalized_dirichlet_mean_channels",
        family="reverse_variance_head",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean_c normalized spatial Dirichlet morphology of log reverse std",
    )
    _add_track(
        tracks,
        specs,
        np.mean(logstd_logvar, axis=2),
        name="variance_head_logstd_log_spatial_variance_mean_channels",
        family="reverse_variance_head",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean_c log spatial variance of reverse-kernel log standard deviation",
    )
    _add_track(
        tracks,
        specs,
        np.mean(logstd[:, :-1], axis=(2, 3, 4)),
        name="operational_logstd_mean_all_dimensions",
        family="operational_reverse_kernel",
        availability="predictable",
        observation_offset_steps=0,
        formula="mean reverse-kernel log standard deviation for stochastic k=0..248 only",
        deployment_note="t=0 variance head excluded because its draw is masked",
    )

    innovation = arrays["transition_innovation"][:, :-1].astype(np.float64)
    effective_logstd = logstd[:, :-1]
    groups: dict[str, slice | Sequence[int]] = {
        "guided3": slice(0, 3),
        "channel4": [3],
        "all4": slice(0, 4),
    }
    for group, channels in groups.items():
        z2 = np.mean(innovation[:, :, channels] ** 2, axis=(2, 3, 4))
        mean_logstd = np.mean(effective_logstd[:, :, channels], axis=(2, 3, 4))
        _add_track(
            tracks,
            specs,
            z2,
            name=f"innovation_z2_mean_{group}",
            family="realized_innovation",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"mean squared whitened Gaussian innovation over {group}, k=0..248",
            uses_realized_innovation=True,
            deployment_note="raw saved t=0 draw excluded",
        )
        _add_track(
            tracks,
            specs,
            np.sqrt(z2),
            name=f"innovation_rms_norm_{group}",
            family="realized_innovation",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"sqrt(mean squared whitened Gaussian innovation over {group})",
            uses_realized_innovation=True,
            deployment_note="raw saved t=0 draw excluded",
        )
        _add_track(
            tracks,
            specs,
            0.5 * z2 + mean_logstd + 0.5 * math.log(2.0 * math.pi),
            name=f"transition_nll_per_dimension_{group}",
            family="operational_reverse_kernel",
            availability="online_causal",
            observation_offset_steps=1,
            formula=(
                f"mean over {group} of 0.5*z^2 + log(sigma) + 0.5*log(2*pi), "
                "the implemented Gaussian transition NLL per dimension"
            ),
            uses_realized_innovation=True,
            deployment_note="raw saved t=0 draw excluded",
        )
        _add_track(
            tracks,
            specs,
            _tile_concentration(innovation, channels),
            name=f"innovation_energy_tile4x4_concentration_{group}",
            family="realized_innovation",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"maximum / sum of sixteen 8x8 tile mean innovation energies, {group}",
            uses_realized_innovation=True,
            deployment_note="raw saved t=0 draw excluded",
        )

    eps_gap = cond_eps.astype(np.float64) - uncond_eps.astype(np.float64)
    var_gap = cond_var.astype(np.float64) - uncond_var.astype(np.float64)
    factor = np.sqrt((1.0 - alpha) / alpha)[None, :, None, None, None]
    pred_cond = state.astype(np.float64) / np.sqrt(a) - factor * cond_eps.astype(np.float64)
    pred_uncond = state.astype(np.float64) / np.sqrt(a) - factor * uncond_eps.astype(np.float64)
    pred_gap = pred_cond - pred_uncond
    for group, channels in groups.items():
        _add_track(
            tracks,
            specs,
            _rms(eps_gap, channels),
            name=f"cfg_epsilon_gap_rms_{group}",
            family="raw_conditional_unconditional_epsilon_gap",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"RMS(conditional raw epsilon - unconditional raw epsilon), {group}",
        )
        _add_track(
            tracks,
            specs,
            _cosine(cond_eps, uncond_eps, channels),
            name=f"cfg_epsilon_cond_uncond_cosine_{group}",
            family="raw_conditional_unconditional_epsilon_gap",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"cosine angle between raw conditional and unconditional epsilon, {group}",
        )
        _add_track(
            tracks,
            specs,
            _tile_concentration(eps_gap, channels),
            name=f"cfg_epsilon_gap_tile4x4_concentration_{group}",
            family="raw_conditional_unconditional_epsilon_gap",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"maximum/sum of sixteen tile energies of conditional-unconditional epsilon gap, {group}",
        )

        _add_track(
            tracks,
            specs,
            _rms(var_gap, channels),
            name=f"cfg_variance_raw_gap_rms_{group}",
            family="raw_conditional_unconditional_variance_gap",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"RMS(raw conditional learned-range value - unconditional value), {group}",
        )
        _add_track(
            tracks,
            specs,
            _cosine(cond_var, uncond_var, channels),
            name=f"cfg_variance_raw_cond_uncond_cosine_{group}",
            family="raw_conditional_unconditional_variance_gap",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"cosine angle of conditional/unconditional raw learned-range heads, {group}",
        )
        _add_track(
            tracks,
            specs,
            _tile_concentration(var_gap, channels),
            name=f"cfg_variance_raw_gap_tile4x4_concentration_{group}",
            family="raw_conditional_unconditional_variance_gap",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"tile concentration of raw learned-range conditional-unconditional gap, {group}",
        )

        pred_gap_rms = _rms(pred_gap, channels)
        pred_branch_scale = np.sqrt(
            0.5
            * (
                np.mean(pred_cond[:, :, channels] ** 2, axis=(2, 3, 4))
                + np.mean(pred_uncond[:, :, channels] ** 2, axis=(2, 3, 4))
            )
        )
        _add_track(
            tracks,
            specs,
            pred_gap_rms,
            name=f"pred_xstart_cond_uncond_disagreement_rms_{group}",
            family="conditional_unconditional_predicted_clean_disagreement",
            availability="predictable",
            observation_offset_steps=0,
            formula=(
                f"RMS of branch pred_x0 disagreement = sqrt((1-alpha)/alpha) "
                f"times epsilon-gap RMS, {group}"
            ),
        )
        _add_track(
            tracks,
            specs,
            pred_gap_rms / (pred_branch_scale + EPS),
            name=f"pred_xstart_cond_uncond_disagreement_relative_rms_{group}",
            family="conditional_unconditional_predicted_clean_disagreement",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"branch pred_x0 disagreement RMS divided by joint branch RMS, {group}",
        )
        _add_track(
            tracks,
            specs,
            _cosine(pred_cond, pred_uncond, channels),
            name=f"pred_xstart_cond_uncond_cosine_{group}",
            family="conditional_unconditional_predicted_clean_disagreement",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"cosine angle between conditional and unconditional predicted x0, {group}",
        )
        _add_track(
            tracks,
            specs,
            _tile_concentration(pred_gap, channels),
            name=f"pred_xstart_cond_uncond_gap_tile4x4_concentration_{group}",
            family="conditional_unconditional_predicted_clean_disagreement",
            availability="predictable",
            observation_offset_steps=0,
            formula=f"tile concentration of conditional-unconditional predicted-x0 gap, {group}",
        )

        effective_gap = eps_gap[:, :-1, channels]
        effective_innovation = innovation[:, :, channels]
        gap_flat = effective_gap.reshape(effective_gap.shape[0], effective_gap.shape[1], -1)
        innovation_flat = effective_innovation.reshape(
            effective_innovation.shape[0], effective_innovation.shape[1], -1
        )
        inner = np.sum(gap_flat * innovation_flat, axis=2)
        _add_track(
            tracks,
            specs,
            inner
            / (
                np.linalg.norm(gap_flat, axis=2)
                * np.linalg.norm(innovation_flat, axis=2)
                + EPS
            ),
            name=f"innovation_cfg_epsilon_gap_cosine_{group}",
            family="innovation_cfg_gap_alignment",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"cosine between realized innovation and current CFG epsilon gap, {group}",
            uses_realized_innovation=True,
            deployment_note="raw saved t=0 draw excluded",
        )
        _add_track(
            tracks,
            specs,
            inner / (np.linalg.norm(gap_flat, axis=2) + EPS),
            name=f"innovation_projection_on_cfg_epsilon_gap_unit_{group}",
            family="innovation_cfg_gap_alignment",
            availability="online_causal",
            observation_offset_steps=1,
            formula=f"dot(realized innovation, epsilon gap) / ||epsilon gap||, {group}",
            uses_realized_innovation=True,
            deployment_note="raw saved t=0 draw excluded",
        )

    endpoint_rows: list[dict[str, Any]] = []
    decoded = arrays["decoded_images"]
    for slot, class_id in enumerate(record.classes):
        rgb = np.floor(
            (np.clip(decoded[slot], -1.0, 1.0) + 1.0) * 127.5 + 0.5
        ).astype(np.uint8).transpose(1, 2, 0)
        endpoint = _endpoint_features(rgb)
        endpoint_rows.append(
            {
                "global_seed": record.global_seed,
                "class_slot": slot,
                "class_id": class_id,
                "endpoint_png_path": str(
                    record.root / f"images/{slot:02d}_class{class_id:04d}.png"
                ),
                **endpoint,
            }
        )
    return tracks, specs, endpoint_rows


def _combine_trace_tracks(
    per_trace: Sequence[tuple[dict[str, np.ndarray], dict[str, TrackSpec]]]
) -> tuple[dict[str, np.ndarray], dict[str, TrackSpec]]:
    if not per_trace:
        raise RuntimeError("no extracted traces")
    names = set(per_trace[0][0])
    specs = per_trace[0][1]
    for tracks, observed_specs in per_trace:
        if set(tracks) != names or observed_specs != specs:
            raise RuntimeError("trace feature schemas differ")
    return {
        name: np.concatenate([tracks[name] for tracks, _ in per_trace], axis=0)
        for name in sorted(names)
    }, specs


def _robust_all_sample_z(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.maximum(MAD_FACTOR * mad, MAD_FLOOR)
    return (values - median) / scale, median, scale


def add_fixed_label_free_reference_features(
    frame: pd.DataFrame,
    tracks: Mapping[str, np.ndarray],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    rough_name = "pred_xstart_log_normalized_dirichlet_mean_channels"
    amplitude_name = "pred_xstart_log_spatial_variance_mean_channels"
    instability_name = "pred_xstart_temporal_instability_mean_channels"
    rough_z, rough_median, rough_scale = _robust_all_sample_z(tracks[rough_name])
    amplitude_z, amplitude_median, amplitude_scale = _robust_all_sample_z(
        tracks[amplitude_name]
    )
    instability_z, instability_median, instability_scale = _robust_all_sample_z(
        tracks[instability_name]
    )
    q1_rough = np.mean(rough_z[:, 50:100], axis=1)
    q2_amplitude = np.mean(amplitude_z[:, 100:150], axis=1)
    q1_instability = np.mean(instability_z[:, 50:100], axis=1)
    score = (q1_rough + q2_amplitude) / math.sqrt(2.0)
    definitions = (
        (
            "label_free_reference_q1_pred_roughness_z_mean",
            q1_rough,
            "predictable",
            99,
            150,
            "mean k=50..99 all-sample robust z of predicted-clean normalized Dirichlet",
        ),
        (
            "label_free_reference_q2_pred_amplitude_z_mean",
            q2_amplitude,
            "predictable",
            149,
            100,
            "mean k=100..149 all-sample robust z of predicted-clean log spatial variance",
        ),
        (
            "label_free_reference_q1_pred_temporal_instability_z_mean",
            q1_instability,
            "online_causal",
            100,
            149,
            "mean j=50..99 robust z of adjacent predicted-clean instability; requires pred[100]",
        ),
        (
            "fixed_two_phase_predicted_clean_score_label_free_reference",
            score,
            "predictable",
            149,
            100,
            "(q1 predicted-clean roughness robust-z mean + q2 amplitude robust-z mean)/sqrt(2)",
        ),
    )
    for name, values, availability, latest_step, internal_t, formula in definitions:
        frame[name] = values
        catalog[name] = {
            "feature": name,
            "track": "fixed_label_free_reference_composite",
            "family": "fixed_label_free_reference_score",
            "reduction": "fixed_formula",
            "track_formula": formula,
            "feature_formula": formula,
            "track_length": None,
            "availability": availability,
            "latest_required_sampling_step": latest_step,
            "latest_required_internal_timestep": internal_t,
            "observation_timing": (
                "before_transition_at_latest_step"
                if availability == "predictable"
                else "after_transition_or_next_prediction_at_latest_step"
            ),
            "preterminal_actionable": latest_step < 249,
            "uses_realized_innovation": False,
            "deployment_note": (
                "reference uses all samples without labels in this discovery output; "
                "freeze median/scale arrays before prospective deployment"
            ),
        }
    return {
        "pred_roughness_median": rough_median,
        "pred_roughness_scale": rough_scale,
        "pred_amplitude_median": amplitude_median,
        "pred_amplitude_scale": amplitude_scale,
        "pred_temporal_instability_median": instability_median,
        "pred_temporal_instability_scale": instability_scale,
    }


def _contains_value(payload: Any, expected: str) -> bool:
    if payload == expected:
        return True
    if isinstance(payload, dict):
        return any(_contains_value(value, expected) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_value(value, expected) for value in payload)
    return False


def _infer_seed_class(row: Mapping[str, Any], row_key: str | None) -> tuple[int, int]:
    seed = next(
        (row.get(key) for key in ("global_seed", "seed", "sampling_seed") if key in row),
        None,
    )
    class_id = next(
        (row.get(key) for key in ("class_id", "imagenet_class_id", "class") if key in row),
        None,
    )
    searchable = " ".join(
        str(value)
        for value in (
            row_key,
            row.get("sample_id"),
            row.get("blind_id"),
            row.get("filename"),
            row.get("image"),
            row.get("path"),
        )
        if value is not None
    )
    if type(seed) is not int:
        match = re.search(r"seed[_-]?(\d+)", searchable, flags=re.IGNORECASE)
        if match:
            seed = int(match.group(1))
    if type(class_id) is not int:
        match = re.search(r"class[_-]?0*(\d+)", searchable, flags=re.IGNORECASE)
        if match:
            class_id = int(match.group(1))
    if type(seed) is not int or type(class_id) is not int:
        raise RuntimeError(
            "locked consensus row cannot be joined: explicit/inferable seed and class_id required"
        )
    return seed, class_id


def _canonical_label(raw: Any) -> str:
    if type(raw) in {int, float} and not isinstance(raw, bool):
        if float(raw) == 0.0:
            return "clean_good"
        if float(raw) == 1.0:
            return "mild_or_disputed"
        if float(raw) in {2.0, 3.0}:
            return "clear_bad"
    text = str(raw).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if compact in {
        "clear_overall_structural_bad",
        "clear_structural_bad",
        "clear_bad",
        "bad",
        "catastrophic",
        "score_2",
        "score_3",
        "2",
        "3",
    }:
        return "clear_bad"
    if compact in {"clean_good", "good", "normal", "pass", "score_0", "0"}:
        return "clean_good"
    if compact in {
        "not_clear_overall_structural_bad",
        "not_clear_structural_bad",
        "not_clear_bad",
    }:
        return "not_clear_bad"
    if compact in {
        "u",
        "uncertain",
        "uncertain_or_not_scorable",
        "excluded",
        "exclude",
        "mild",
        "mild_or_excluded",
        "mild_or_disputed",
        "downgrade_mild_or_typical",
        "score_1",
        "1",
    }:
        return "mild_or_disputed"
    raise RuntimeError(f"unsupported locked consensus label: {raw!r}")


def load_locked_consensus(path: Path) -> tuple[dict[tuple[int, int], tuple[str, str]], dict[str, Any]]:
    requested = path.expanduser().absolute()
    if any("draft" in part.lower() for part in requested.parts):
        raise RuntimeError("reviewer drafts are forbidden; supply a locked consensus artifact")
    _require_regular(requested, "locked consensus")
    path = requested.resolve()
    manifest_path = path.parent / "manifest.json"
    completion_path = path.parent / "completion.json"
    _require_regular(manifest_path, "consensus manifest")
    _require_regular(completion_path, "consensus completion receipt")
    consensus = load_json(path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    status = str(manifest.get("status", "")).upper()
    consensus_hash = sha256_file(path)
    manifest_hash = sha256_file(manifest_path)
    if (
        "LOCKED" not in status
        or completion.get("complete") is not True
        or not _contains_value(manifest, consensus_hash)
        or not _contains_value(completion, consensus_hash)
        or not _contains_value(completion, manifest_hash)
        or (
            manifest.get("identity_sha256") is not None
            and not _contains_value(completion, str(manifest.get("identity_sha256")))
        )
    ):
        raise RuntimeError("consensus lock/manifest/completion hash lineage is invalid")
    container = consensus.get("rows", consensus.get("annotations"))
    keyed_rows: list[tuple[str | None, Mapping[str, Any]]] = []
    if isinstance(container, list):
        if not all(isinstance(row, dict) for row in container):
            raise RuntimeError("locked consensus rows are malformed")
        keyed_rows = [(None, row) for row in container]
    elif isinstance(container, dict):
        if not all(isinstance(row, dict) for row in container.values()):
            raise RuntimeError("locked consensus annotation mapping is malformed")
        keyed_rows = [(str(key), row) for key, row in container.items()]
    else:
        raise RuntimeError("locked consensus must contain rows or annotations")
    mapping: dict[tuple[int, int], tuple[str, str]] = {}
    label_fields = (
        "consensus_label",
        "primary_label",
        "primary_overall_structural_quality",
        "final_label",
        "consensus_score",
        "quality_score",
        "score",
        "label",
    )
    for row_key, row in keyed_rows:
        seed, class_id = _infer_seed_class(row, row_key)
        candidates = [row.get(field) for field in label_fields if row.get(field) is not None]
        if not candidates:
            raise RuntimeError("locked consensus row has no supported consensus label field")
        raw = candidates[0]
        label = _canonical_label(raw)
        key = (seed, class_id)
        if key in mapping:
            raise RuntimeError(f"duplicate locked consensus sample key: {key}")
        mapping[key] = (label, str(raw))
    return mapping, {
        "path": str(path),
        "sha256": consensus_hash,
        "manifest_sha256": manifest_hash,
        "completion_sha256": sha256_file(completion_path),
        "manifest_status": manifest.get("status"),
        "row_count": len(mapping),
    }


def join_labels(
    frame: pd.DataFrame,
    mapping: Mapping[tuple[int, int], tuple[str, str]],
) -> None:
    trace_keys = set(zip(frame["global_seed"].astype(int), frame["class_id"].astype(int)))
    outside = set(mapping) - trace_keys
    if outside:
        raise RuntimeError(
            f"locked consensus contains {len(outside)} rows outside supplied traces; "
            f"first={sorted(outside)[:3]}"
        )
    labels: list[str] = []
    raw_labels: list[str] = []
    for seed, class_id in zip(frame["global_seed"], frame["class_id"]):
        value = mapping.get((int(seed), int(class_id)))
        labels.append(value[0] if value is not None else "unlabeled")
        raw_labels.append(value[1] if value is not None else "")
    frame["label"] = labels
    frame["raw_consensus_label"] = raw_labels


def evaluate_univariate(
    frame: pd.DataFrame,
    catalog: pd.DataFrame,
    negative_policy: str,
) -> pd.DataFrame:
    negatives = {"clean_good"}
    if negative_policy == "include_not_clear_bad":
        negatives.add("not_clear_bad")
    include = frame["label"].isin({"clear_bad", *negatives})
    selected = frame.loc[include].copy()
    target = (selected["label"] == "clear_bad").astype(int).to_numpy()
    if np.sum(target) == 0 or np.sum(target == 0) == 0:
        raise RuntimeError(
            "locked labels do not contain both clear_bad and allowed negative examples"
        )
    rows = []
    for feature in catalog["feature"]:
        values = pd.to_numeric(selected[feature], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or len(np.unique(values)) < 2:
            continue
        raw_auc = float(roc_auc_score(target, values))
        direction = 1.0 if raw_auc >= 0.5 else -1.0
        oriented = direction * values
        per_class_raw = []
        per_class_oriented = []
        for _, group in selected.assign(_target=target, _value=values).groupby("class_id"):
            group_target = group["_target"].to_numpy(int)
            if len(np.unique(group_target)) == 2:
                group_value = group["_value"].to_numpy(float)
                per_class_raw.append(float(roc_auc_score(group_target, group_value)))
                per_class_oriented.append(
                    float(roc_auc_score(group_target, direction * group_value))
                )
        meta = catalog.loc[catalog["feature"] == feature].iloc[0].to_dict()
        rows.append(
            {
                **meta,
                "negative_policy": negative_policy,
                "N_clear_bad": int(np.sum(target)),
                "N_negative": int(np.sum(target == 0)),
                "N_informative_classes": len(per_class_raw),
                "raw_auc_bad_high": raw_auc,
                "exploratory_orientation": (
                    "higher_is_bad" if direction > 0 else "lower_is_bad"
                ),
                "oriented_auc": float(roc_auc_score(target, oriented)),
                "oriented_average_precision": float(
                    average_precision_score(target, oriented)
                ),
                "within_class_raw_auc_macro": (
                    float(np.mean(per_class_raw)) if per_class_raw else np.nan
                ),
                "within_class_oriented_auc_macro": (
                    float(np.mean(per_class_oriented)) if per_class_oriented else np.nan
                ),
                "clear_bad_mean": float(np.mean(values[target == 1])),
                "negative_mean": float(np.mean(values[target == 0])),
                "p_value_computed": False,
                "confirmatory_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oriented_auc", "oriented_average_precision", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("integer list must be nonempty and unique")
    return values


def discover_trace_dirs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.trace_dir or []:
        paths.append(raw.expanduser().absolute())
    for raw_root in args.trace_root or []:
        root = raw_root.expanduser().absolute()
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"trace discovery root is invalid: {root}")
        matches = [
            path
            for path in root.glob(args.trace_glob)
            if path.is_dir() and not path.is_symlink()
        ]
        paths.extend(sorted(matches, key=lambda path: path.name))
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        value = path.resolve()
        if value in seen:
            raise RuntimeError(f"duplicate trace directory: {value}")
        seen.add(value)
        resolved.append(value)
    if not resolved:
        raise RuntimeError("no trace directories selected")
    return resolved


def publish(args: argparse.Namespace) -> Path:
    trace_dirs = discover_trace_dirs(args)
    # Validate and reduce one 300+ MiB archive at a time.  Keeping all raw
    # arrays resident until the final run is unnecessary and would scale the
    # peak memory linearly with the number of seeds.
    extracted_runs: list[
        tuple[
            TraceRecord,
            dict[str, np.ndarray],
            dict[str, TrackSpec],
            list[dict[str, Any]],
        ]
    ] = []
    for path in trace_dirs:
        record, arrays = load_validated_trace(path)
        run_tracks, run_specs, endpoints = extract_trace_tracks(record, arrays)
        extracted_runs.append((record, run_tracks, run_specs, endpoints))
        del arrays
    records = [item[0] for item in extracted_runs]
    seeds = [record.global_seed for record in records]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"duplicate global seeds across trace runs: {seeds}")
    extracted_runs.sort(key=lambda item: item[0].global_seed)
    records = [item[0] for item in extracted_runs]
    if args.expected_seeds is not None and tuple(record.global_seed for record in records) != tuple(
        sorted(args.expected_seeds)
    ):
        raise RuntimeError("observed seeds differ from --expected-seeds")
    class_orders = {record.classes for record in records}
    fingerprints = {record.scientific_fingerprint_sha256 for record in records}
    if len(class_orders) != 1 or len(fingerprints) != 1:
        raise RuntimeError("trace class order or scientific sampler identity differs across runs")
    classes = records[0].classes
    if args.expected_classes is not None and classes != args.expected_classes:
        raise RuntimeError(
            f"ordered classes differ: observed={classes}, expected={args.expected_classes}"
        )

    per_trace_tracks = []
    endpoint_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for run_index, (record, run_tracks, run_specs, endpoints) in enumerate(
        extracted_runs
    ):
        per_trace_tracks.append((run_tracks, run_specs))
        endpoints.sort(key=lambda row: int(row["class_slot"]))
        for endpoint in endpoints:
            sample_rows.append(
                {
                    "sample_index": len(sample_rows),
                    "run_index": run_index,
                    "global_seed": record.global_seed,
                    "class_slot": endpoint["class_slot"],
                    "class_id": endpoint["class_id"],
                    "trace_dir": str(record.root),
                    "endpoint_png_path": endpoint["endpoint_png_path"],
                    "label": "unlabeled",
                    "raw_consensus_label": "",
                }
            )
        endpoint_rows.extend(endpoints)

    tracks, specs = _combine_trace_tracks(per_trace_tracks)
    scalar_frame, catalog_map = reduce_tracks(tracks, specs)
    identity_frame = pd.DataFrame(sample_rows)
    endpoint_frame = pd.DataFrame(endpoint_rows)
    if not np.array_equal(
        identity_frame[["global_seed", "class_slot", "class_id"]].to_numpy(),
        endpoint_frame[["global_seed", "class_slot", "class_id"]].to_numpy(),
    ):
        raise RuntimeError("endpoint/sample row ordering changed")
    frame = pd.concat(
        [
            identity_frame.reset_index(drop=True),
            endpoint_frame.drop(
                columns=["global_seed", "class_slot", "class_id", "endpoint_png_path"]
            ).reset_index(drop=True),
            scalar_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    for feature in endpoint_frame.columns:
        if feature in {"global_seed", "class_slot", "class_id", "endpoint_png_path"}:
            continue
        catalog_map[feature] = {
            "feature": feature,
            "track": "decoded_endpoint_rgb",
            "family": "endpoint_structure",
            "reduction": "endpoint",
            "track_formula": "decoded endpoint pixel statistic",
            "feature_formula": feature.replace("endpoint_", "").replace("_", " "),
            "track_length": None,
            "availability": "retrospective",
            "latest_required_sampling_step": 249,
            "latest_required_internal_timestep": 0,
            "observation_timing": "decoded_endpoint",
            "preterminal_actionable": False,
            "uses_realized_innovation": False,
            "deployment_note": "endpoint mining/control only; not an early-warning feature",
        }
    reference_arrays = add_fixed_label_free_reference_features(frame, tracks, catalog_map)
    catalog = pd.DataFrame([catalog_map[name] for name in sorted(catalog_map)])

    consensus_inventory: dict[str, Any] | None = None
    if args.consensus is not None:
        label_mapping, consensus_inventory = load_locked_consensus(args.consensus)
        join_labels(frame, label_mapping)

    output = args.output_dir.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite any existing output path: {output}")
    output = output.resolve()
    input_roots = [record.root for record in records]
    if any(root == output or root in output.parents or output in root.parents for root in input_roots):
        raise RuntimeError("analysis output must not overlap a trace input")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        shutil.copyfile(Path(__file__).resolve(), staging / "analysis_source.py")
        atomic_json_dump(PROTOCOL, staging / "protocol_snapshot.json")
        frame.to_csv(staging / "sample_features.csv", index=False)
        catalog.to_csv(staging / "feature_catalog.csv", index=False)
        np.savez_compressed(
            staging / "time_series.npz",
            sample_index=frame["sample_index"].to_numpy(np.int32),
            global_seed=frame["global_seed"].to_numpy(np.int64),
            class_slot=frame["class_slot"].to_numpy(np.int16),
            class_id=frame["class_id"].to_numpy(np.int16),
            sampling_step_250=np.arange(250, dtype=np.int16),
            internal_timestep_250=np.arange(249, -1, -1, dtype=np.int16),
            transition_sampling_step_249=np.arange(249, dtype=np.int16),
            transition_internal_timestep_from_249=np.arange(249, 0, -1, dtype=np.int16),
            **tracks,
        )
        np.savez_compressed(staging / "label_free_reference_stats.npz", **reference_arrays)
        results: pd.DataFrame | None = None
        if args.consensus is not None:
            results = evaluate_univariate(frame, catalog, args.negative_label_policy)
            results.to_csv(staging / "univariate_discovery_results.csv", index=False)

        source_inventory = {
            "analysis_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "protocol_canonical_sha256": canonical_sha256(PROTOCOL),
            "ordered_classes": list(classes),
            "ordered_seeds": [record.global_seed for record in records],
            "scientific_fingerprint_sha256": records[0].scientific_fingerprint_sha256,
            "trace_runs": [
                {
                    **asdict(record),
                    "root": str(record.root),
                    "classes": list(record.classes),
                }
                for record in records
            ],
            "locked_consensus": consensus_inventory,
        }
        atomic_json_dump(source_inventory, staging / "source_inventory.json")
        label_counts = frame["label"].value_counts().sort_index().to_dict()
        family_counts = catalog["family"].value_counts().sort_index().to_dict()
        summary: dict[str, Any] = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "status": "DISCOVERY_ONLY_NOT_AN_INTERVENTION_TRIGGER",
            "sample_count": len(frame),
            "run_count": len(records),
            "ordered_classes": list(classes),
            "ordered_seeds": [record.global_seed for record in records],
            "track_count": len(tracks),
            "scalar_feature_count": len(catalog),
            "feature_family_counts": {str(k): int(v) for k, v in family_counts.items()},
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "labels_joined": args.consensus is not None,
            "univariate_result_count": 0 if results is None else len(results),
            "negative_label_policy": args.negative_label_policy,
            "timing_audit": {
                "sampling_step_0": "internal timestep 249",
                "sampling_step_249": "internal timestep 0",
                "innovation_rows": 249,
                "masked_t0_draw_excluded": True,
                "maximum_cfg_pred_reconstruction_error": max(
                    record.reconstruction_max_abs_error for record in records
                ),
                "maximum_conditional_variance_logstd_reconstruction_error": max(
                    record.variance_reconstruction_max_logstd_error
                    for record in records
                ),
            },
            "multiplicity_warning": (
                "This is a broad discovery screen over many correlated metrics and fixed "
                "time reductions. Any large same-pool AUC is hypothesis-generating only."
            ),
            "label_warning": (
                "Only an externally locked consensus is accepted. clear_bad is never "
                "compared with mild/uncertain; not_clear_bad is excluded by default."
            ),
            "combination_policy": (
                "No trained combination or threshold is fit. The only composite is the "
                "fixed two-phase predicted-clean score recorded in the protocol snapshot."
            ),
        }
        atomic_json_dump(summary, staging / "summary.json")

        files = []
        for path in sorted(staging.iterdir()):
            if path.name in {"manifest.json", "completion.json"}:
                continue
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        manifest = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "experiment": "dit_bad_good_custom_trace_metric_discovery",
            "status": "complete",
            "analysis_source_sha256": sha256_file(staging / "analysis_source.py"),
            "protocol_snapshot_sha256": sha256_file(staging / "protocol_snapshot.json"),
            "source_inventory_sha256": sha256_file(staging / "source_inventory.json"),
            "trace_identity_sha256_ordered": [record.identity_sha256 for record in records],
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        atomic_json_dump(manifest, staging / "manifest.json")
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "summary_file_sha256": sha256_file(staging / "summary.json"),
        }
        completion["payload_sha256"] = canonical_sha256(completion)
        atomic_json_dump(completion, staging / "completion.json")
        # Validate while still recoverable.  Atomic rename publishes only an
        # already closed, hash-bound, semantically checked directory.
        validate_analysis_output(staging)
        staging.rename(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_analysis_output(output: Path) -> None:
    manifest_path = output / "manifest.json"
    completion_path = output / "completion.json"
    _require_regular(manifest_path, "analysis manifest")
    _require_regular(completion_path, "analysis completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = dict(manifest)
    observed_identity = identity.pop("identity_sha256", None)
    if observed_identity != canonical_sha256(identity):
        raise RuntimeError("analysis manifest identity hash is invalid")
    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or not all(isinstance(record, dict) for record in files)
        or len({record.get("name") for record in files}) != len(files)
    ):
        raise RuntimeError("analysis manifest files are malformed")
    summary = load_json(output / "summary.json")
    required_payloads = {
        "analysis_source.py",
        "feature_catalog.csv",
        "label_free_reference_stats.npz",
        "protocol_snapshot.json",
        "sample_features.csv",
        "source_inventory.json",
        "summary.json",
        "time_series.npz",
    }
    if summary.get("labels_joined") is True:
        required_payloads.add("univariate_discovery_results.csv")
    listed_names = {record.get("name") for record in files}
    observed_names = {path.name for path in output.iterdir()}
    if listed_names != required_payloads or observed_names != {
        *required_payloads,
        "manifest.json",
        "completion.json",
    }:
        raise RuntimeError("analysis output member set is not closed or does not match mode")
    if any(not path.is_file() or path.is_symlink() for path in output.iterdir()):
        raise RuntimeError("analysis output contains a non-regular or indirect member")
    for record in files:
        path = _safe_relative(output, str(record.get("name")))
        _require_regular(path, "analysis payload")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"analysis payload hash changed: {path}")
    expected_completion = dict(completion)
    payload_hash = expected_completion.pop("payload_sha256", None)
    if payload_hash != canonical_sha256(expected_completion):
        raise RuntimeError("analysis completion payload hash is invalid")
    if (
        completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != observed_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("summary_file_sha256") != sha256_file(output / "summary.json")
    ):
        raise RuntimeError("analysis completion receipt is invalid")

    if load_json(output / "protocol_snapshot.json") != PROTOCOL:
        raise RuntimeError("analysis protocol snapshot differs from the executing protocol")
    if (
        manifest.get("analysis_source_sha256")
        != sha256_file(output / "analysis_source.py")
        or manifest.get("protocol_snapshot_sha256")
        != sha256_file(output / "protocol_snapshot.json")
        or manifest.get("source_inventory_sha256")
        != sha256_file(output / "source_inventory.json")
    ):
        raise RuntimeError("analysis manifest top-level source bindings are invalid")

    frame = pd.read_csv(output / "sample_features.csv")
    catalog = pd.read_csv(output / "feature_catalog.csv")
    expected_catalog_columns = {
        "feature",
        "track",
        "family",
        "reduction",
        "track_formula",
        "feature_formula",
        "track_length",
        "availability",
        "latest_required_sampling_step",
        "latest_required_internal_timestep",
        "observation_timing",
        "preterminal_actionable",
        "uses_realized_innovation",
        "deployment_note",
    }
    if (
        set(catalog.columns) != expected_catalog_columns
        or catalog["feature"].duplicated().any()
        or frame.columns.duplicated().any()
        or set(frame.columns) != {*IDENTIFIER_COLUMNS, *catalog["feature"].tolist()}
        or len(frame) != summary.get("sample_count")
        or len(catalog) != summary.get("scalar_feature_count")
        or not np.array_equal(frame["sample_index"].to_numpy(), np.arange(len(frame)))
        or frame[["global_seed", "class_id"]].duplicated().any()
    ):
        raise RuntimeError("analysis CSV schema, counts, or sample identities are invalid")
    scalar = frame[catalog["feature"].tolist()].to_numpy(dtype=np.float64)
    if not np.isfinite(scalar).all():
        raise RuntimeError("analysis scalar feature matrix contains non-finite values")
    latest = catalog["latest_required_sampling_step"].to_numpy(dtype=np.int64)
    internal = catalog["latest_required_internal_timestep"].to_numpy(dtype=np.int64)
    actionable = catalog["preterminal_actionable"].astype(bool).to_numpy()
    if (
        np.any(latest < 0)
        or np.any(latest > 249)
        or not np.array_equal(latest + internal, np.full(len(catalog), 249))
        or not np.array_equal(actionable, latest < 249)
    ):
        raise RuntimeError("analysis feature timing metadata is internally inconsistent")

    track_lengths: dict[str, int] = {}
    for track, group in catalog.loc[catalog["track_length"].notna()].groupby("track"):
        lengths = group["track_length"].astype(int).unique()
        if len(lengths) != 1 or lengths[0] not in {249, 250}:
            raise RuntimeError(f"analysis catalog has inconsistent track length: {track}")
        track_lengths[str(track)] = int(lengths[0])
    fixed_series = {
        "sample_index": frame["sample_index"].to_numpy(np.int32),
        "global_seed": frame["global_seed"].to_numpy(np.int64),
        "class_slot": frame["class_slot"].to_numpy(np.int16),
        "class_id": frame["class_id"].to_numpy(np.int16),
        "sampling_step_250": np.arange(250, dtype=np.int16),
        "internal_timestep_250": np.arange(249, -1, -1, dtype=np.int16),
        "transition_sampling_step_249": np.arange(249, dtype=np.int16),
        "transition_internal_timestep_from_249": np.arange(249, 0, -1, dtype=np.int16),
    }
    try:
        with np.load(output / "time_series.npz", allow_pickle=False) as archive:
            if set(archive.files) != {*fixed_series, *track_lengths}:
                raise RuntimeError("analysis time-series member set is invalid")
            for name, expected in fixed_series.items():
                if not np.array_equal(archive[name], expected):
                    raise RuntimeError(f"analysis time-series axis changed: {name}")
            for name, length in track_lengths.items():
                values = archive[name]
                if values.shape != (len(frame), length) or not np.isfinite(values).all():
                    raise RuntimeError(f"analysis time-series track is invalid: {name}")
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot validate analysis time-series archive") from exc
    if len(track_lengths) != summary.get("track_count"):
        raise RuntimeError("analysis track count differs from catalog")

    reference_shapes = {
        "pred_roughness_median": (250,),
        "pred_roughness_scale": (250,),
        "pred_amplitude_median": (250,),
        "pred_amplitude_scale": (250,),
        "pred_temporal_instability_median": (249,),
        "pred_temporal_instability_scale": (249,),
    }
    try:
        with np.load(output / "label_free_reference_stats.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(reference_shapes):
                raise RuntimeError("label-free reference member set is invalid")
            for name, shape in reference_shapes.items():
                values = archive[name]
                if values.shape != shape or not np.isfinite(values).all():
                    raise RuntimeError(f"label-free reference array is invalid: {name}")
                if name.endswith("_scale") and np.any(values < MAD_FLOOR):
                    raise RuntimeError(f"label-free reference scale violates hard floor: {name}")
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot validate label-free reference archive") from exc

    inventory = load_json(output / "source_inventory.json")
    labels_joined = summary.get("labels_joined")
    if type(labels_joined) is not bool or (
        (inventory.get("locked_consensus") is not None) != labels_joined
    ):
        raise RuntimeError("analysis label-mode source lineage is inconsistent")
    label_counts = frame["label"].value_counts().sort_index().to_dict()
    if {str(key): int(value) for key, value in label_counts.items()} != summary.get(
        "label_counts"
    ):
        raise RuntimeError("analysis label counts differ from summary")
    if not labels_joined and (
        set(frame["label"]) != {"unlabeled"}
        or frame["raw_consensus_label"].fillna("").astype(str).str.len().any()
    ):
        raise RuntimeError("label-free analysis unexpectedly contains joined labels")
    if labels_joined:
        results = pd.read_csv(output / "univariate_discovery_results.csv")
        if len(results) != summary.get("univariate_result_count"):
            raise RuntimeError("univariate discovery result count differs from summary")


def self_test() -> None:
    rng = np.random.default_rng(37)
    array = rng.normal(size=(3, 8, 4, 8, 8)).astype(np.float32)
    # Exercise helpers at the real spatial shape by tiling the deterministic fixture.
    array32 = np.tile(array, (1, 1, 1, 4, 4))
    logvar, logdir, energy = _spatial_statistics(array32)
    scaled_logvar, scaled_logdir, scaled_energy = _spatial_statistics(7.3 * array32)
    assert logvar.shape == (3, 8, 4)
    assert np.allclose(logdir, scaled_logdir, atol=2e-7)
    assert np.allclose(scaled_logvar - logvar, 2.0 * math.log(7.3), atol=2e-6)
    assert np.allclose(scaled_energy, energy * 7.3**2, rtol=2e-6, atol=1e-8)
    concentration = _tile_concentration(array32, slice(0, 4))
    assert concentration.shape == (3, 8)
    assert np.all((concentration > 0.0) & (concentration <= 1.0))

    predictable = TrackSpec("p", "test", "predictable", 0, "x", 250)
    online = TrackSpec("o", "test", "online_causal", 1, "x", 249, True)
    operational = TrackSpec("v", "test", "predictable", 0, "x", 249)
    assert _timing(predictable, start=100, stop=150) == {
        "availability": "predictable",
        "latest_required_sampling_step": 149,
        "latest_required_internal_timestep": 100,
        "observation_timing": "before_transition_at_latest_step",
        "preterminal_actionable": True,
    }
    assert _timing(online, start=200, stop=250)["latest_required_sampling_step"] == 249
    assert _timing(online, start=200, stop=250)["preterminal_actionable"] is False
    assert _timing(operational, start=200, stop=250)["latest_required_sampling_step"] == 248
    assert _timing(operational, start=200, stop=250)[
        "latest_required_internal_timestep"
    ] == 1
    assert _timing(operational)["latest_required_sampling_step"] == 248
    assert _timing(operational)["preterminal_actionable"] is True

    robust_fixture = np.asarray([[0.0, 0.0], [0.0, 1e-8], [0.0, -1e-8]])
    _, _, robust_scale = _robust_all_sample_z(robust_fixture)
    assert np.array_equal(robust_scale, np.asarray([MAD_FLOOR, MAD_FLOOR]))

    cond = rng.normal(size=(2, 3, 4, 4, 4)).astype(np.float32)
    uncond = rng.normal(size=cond.shape).astype(np.float32)
    guided = cond.copy()
    guided[:, :, :3] = uncond[:, :, :3] + 4.0 * (
        cond[:, :, :3] - uncond[:, :, :3]
    )
    assert np.array_equal(guided[:, :, 3], cond[:, :, 3])
    assert not np.array_equal(guided[:, :, :3], cond[:, :, :3])

    innovations = rng.normal(size=(2, 250, 4, 4, 4))
    effective = innovations[:, :-1].copy()
    assert effective.shape[1] == 249
    innovations[:, -1] = 1e30
    assert np.array_equal(effective, innovations[:, :-1])
    assert not np.any(effective == 1e30)

    ramp = np.linspace(-2.0, 3.0, 20)
    reductions = _trajectory_reductions(ramp)
    assert reductions["max_drawup"] == 5.0
    assert reductions["max_drawdown"] == 0.0
    assert reductions["centered_cusum_range"] > 0.0
    rgb = np.full((256, 256, 3), 127, dtype=np.uint8)
    assert all(np.isfinite(value) for value in _endpoint_features(rgb).values())
    try:
        load_locked_consensus(Path("review_A_draft.json"))
    except RuntimeError as exc:
        assert "draft" in str(exc).lower()
    else:
        raise AssertionError("draft consensus path was not rejected")
    assert _canonical_label(0) == "clean_good"
    assert _canonical_label(2) == "clear_bad"
    assert _canonical_label("mild_or_disputed") == "mild_or_disputed"
    assert _canonical_label("not_clear_overall_structural_bad") == "not_clear_bad"
    print(
        "self-test passed: scale-normalized spatial metrics, timing/actionability, "
        "three-channel CFG semantics, masked-t0 exclusion, trajectory reductions, "
        "endpoint controls, and draft-label rejection"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        action="append",
        help="completed custom trace directory; repeat for multiple runs",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        action="append",
        help="parent whose immediate children matching --trace-glob are traces",
    )
    parser.add_argument("--trace-glob", default="targeted_scan_v1_seed*")
    parser.add_argument("--expected-classes", type=_parse_csv_ints)
    parser.add_argument("--expected-seeds", type=_parse_csv_ints)
    parser.add_argument(
        "--consensus",
        type=Path,
        help="optional completed locked consensus; drafts are rejected",
    )
    parser.add_argument(
        "--negative-label-policy",
        choices=("clean_good_only", "include_not_clear_bad"),
        default="clean_good_only",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if not args.trace_dir and not args.trace_root:
        parser.error("at least one --trace-dir or --trace-root is required")
    output = publish(args)
    summary = load_json(output / "summary.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": summary["status"],
                "samples": summary["sample_count"],
                "tracks": summary["track_count"],
                "scalar_features": summary["scalar_feature_count"],
                "labels_joined": summary["labels_joined"],
                "univariate_results": summary["univariate_result_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
