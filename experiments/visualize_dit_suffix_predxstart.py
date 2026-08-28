#!/usr/bin/env python3
"""Strictly visualize ``pred_xstart`` trajectories from one DiT suffix bundle.

This tool is deliberately posthoc visualization only.  It performs no image
quality scoring, ranking, branch selection, alarm construction, resampling, or
intervention.  The source bundle's historical oracle label is not adopted as a
quality judgment here.

Before decoding anything, the program revalidates the completed suffix bundle
through ``intervene_dit_imagenet256_suffix.validate_bundle``.  That includes
the suffix manifest/results/completion self-hashes, the frozen observer and
baseline, all five branch transition trails, trace file and per-array hashes,
shapes, dtypes, finite values, and the complete reverse timestep axis.  The
selected ``target_pred_xstart`` latents are then decoded with the exact pinned
``sd-vae-ft-mse`` snapshot through
``visualize_dit_imagenet256_trace.decode_latents``.

Every individual frame has an unresampled native 256x256 decoded region plus a
visible label strip.  The contact sheet is a fixed matrix: rows are attempts
0..4 and columns are the requested internal timesteps.  Outputs are staged,
self-hashed, strictly revalidated, atomically published, and never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

try:  # Package and direct CLI imports.
    from . import intervene_dit_imagenet256_suffix as suffix
    from . import reproduce_dit_imagenet256 as strict
    from . import visualize_dit_imagenet256_trace as tracevis
except ImportError:  # pragma: no cover - direct CLI invocation.
    import intervene_dit_imagenet256_suffix as suffix
    import reproduce_dit_imagenet256 as strict
    import visualize_dit_imagenet256_trace as tracevis


RUNNER_NAME = "visualize_dit_suffix_predxstart"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
CONTACT_SHEET_NAME = "contact_sheet.png"
DEFAULT_TIMESTEPS_BY_ROLLBACK: dict[int, tuple[int, ...]] = {
    60: (60, 50, 40, 30, 20, 10, 0),
    120: (120, 100, 80, 60, 40, 20, 0),
    180: (180, 150, 120, 90, 60, 30, 0),
}
FRAME_LABEL_HEIGHT = 40
SHEET_CELL_LABEL_HEIGHT = 22
SHEET_HEADER_HEIGHT = 48
SHEET_LEFT_GUTTER = 128
OUTER_MARGIN = 8
CELL_GAP = 8
BACKGROUND = (25, 25, 25)
TEXT = (245, 245, 245)
MUTED_TEXT = (190, 190, 190)
ACCENT_TEXT = (255, 210, 80)


@dataclass(frozen=True)
class BranchTrace:
    branch_id: str
    attempt_index: int
    role: str
    trace_record: dict[str, Any]
    timesteps: np.ndarray
    selected_latents: np.ndarray
    selected_rows: tuple[int, ...]
    selected_hashes: tuple[str, ...]


@dataclass(frozen=True)
class BundleContext:
    root: Path
    manifest: dict[str, Any]
    results: dict[str, Any]
    completion: dict[str, Any]
    observed: suffix.ObservedInput
    source: dict[str, Any]
    checkpoint: dict[str, Any]
    vae: dict[str, Any]
    rollback_internal_timestep: int
    target_batch_index: int
    target_class_id: int
    timesteps: tuple[int, ...]
    branches: tuple[BranchTrace, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_raw_sha256(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return strict.sha256_json(stripped)


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = strict.load_json(path)
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def _absolute_manifest_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"suffix manifest lacks {context}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"suffix manifest {context} is not absolute: {path}")
    return path.resolve()


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _require_file_hash(path: Path, expected: Any, context: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{context} is missing or not a regular file: {path}")
    if not isinstance(expected, str) or strict.sha256_file(path) != expected:
        raise RuntimeError(f"{context} file hash changed: {path}")


def _validate_recorded_provenance(
    manifest: dict[str, Any],
    *,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    vae: dict[str, Any],
    observed: suffix.ObservedInput,
) -> None:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("suffix manifest lacks source provenance")
    expected_sources = {"dit": source, "checkpoint": checkpoint, "vae": vae}
    mismatches = {
        key: (sources.get(key), value)
        for key, value in expected_sources.items()
        if sources.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"suffix source/checkpoint/VAE provenance changed: {mismatches}")

    observer_record = manifest.get("frozen_observe_bundle")
    baseline_record = manifest.get("frozen_baseline")
    if not isinstance(observer_record, dict) or not isinstance(baseline_record, dict):
        raise RuntimeError("suffix manifest lacks frozen observer/baseline provenance")
    expected_observer = {
        "root": str(observed.root),
        "manifest_identity_sha256": observed.identity_sha256,
        "manifest_file_sha256": strict.sha256_file(observed.root / "manifest.json"),
        "results_payload_sha256": observed.results["payload_sha256"],
        "results_file_sha256": strict.sha256_file(observed.root / "results.json"),
        "completion_file_sha256": strict.sha256_file(observed.root / "completion.json"),
        "trace_relative_path": suffix.OBSERVER_TRACE_NAME,
        "trace_sha256": observed.results["trace"]["sha256"],
        "strict_completion_and_trace_math_validated": True,
    }
    expected_baseline = {
        "root": str(observed.baseline.root),
        "manifest_identity_sha256": observed.baseline.identity_sha256,
        "manifest_file_sha256": strict.sha256_file(observed.baseline.root / "manifest.json"),
        "completion_file_sha256": strict.sha256_file(observed.baseline.root / "completion.json"),
        "outputs_sha256": observed.baseline.manifest["outputs_sha256"],
    }
    if observer_record != expected_observer:
        raise RuntimeError("suffix manifest no longer matches its frozen observer bundle")
    if baseline_record != expected_baseline:
        raise RuntimeError("suffix manifest no longer matches its frozen baseline bundle")


def _validate_requested_timesteps(
    requested: Sequence[int] | None, rollback_internal_timestep: int
) -> tuple[int, ...]:
    if requested is None:
        try:
            values = DEFAULT_TIMESTEPS_BY_ROLLBACK[rollback_internal_timestep]
        except KeyError as exc:
            raise ValueError(
                "no default frame set is defined for rollback "
                f"{rollback_internal_timestep}; pass --timesteps explicitly"
            ) from exc
    else:
        values = tuple(requested)
    if not values:
        raise ValueError("at least one internal timestep is required")
    if any(type(value) is not int for value in values):
        raise ValueError("internal timesteps must be integers")
    if len(set(values)) != len(values):
        raise ValueError("internal timesteps must be unique")
    if any(not 0 <= value <= rollback_internal_timestep for value in values):
        raise ValueError(
            f"requested timesteps must lie in [0,{rollback_internal_timestep}]"
        )
    return tuple(values)


def validate_suffix_bundle(
    requested_root: Path, requested_timesteps: Sequence[int] | None
) -> BundleContext:
    if requested_root.is_symlink():
        raise RuntimeError(f"suffix bundle must not be a symlink: {requested_root}")
    root = requested_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"suffix bundle is not a directory: {root}")

    # Preliminary self-hash reads only locate the immutable inputs.  Acceptance
    # still requires the suffix runner's complete validator below.
    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "identity_sha256")
    results_preliminary = _read_self_hashed_json(root / "results.json", "payload_sha256")
    completion_preliminary = _read_self_hashed_json(
        root / COMPLETION_NAME, "payload_sha256"
    )
    if completion_preliminary.get("complete") is not True:
        raise RuntimeError("suffix bundle is not marked complete")
    seed = manifest.get("seed")
    target = manifest.get("target")
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise RuntimeError("suffix manifest has an invalid seed")
    if not isinstance(target, dict):
        raise RuntimeError("suffix manifest lacks a target")
    target_batch_index = target.get("batch_index")
    target_class_id = target.get("class_id")
    if type(target_batch_index) is not int or not 0 <= target_batch_index < suffix.BATCH_SIZE:
        raise RuntimeError("suffix manifest has an invalid target batch index")
    if target_class_id != int(strict.CLASS_IDS[target_batch_index]):
        raise RuntimeError("suffix target class does not match the frozen official batch")
    rollback = results_preliminary.get("rollback_internal_timestep")
    if type(rollback) is not int or rollback not in suffix.FROZEN_ROLLBACK_INTERNAL_TIMESTEPS:
        raise RuntimeError("suffix results have an invalid rollback timestep")
    if manifest.get("frozen_screen_protocol", {}).get(
        "this_invocation_rollback_internal_timestep"
    ) != rollback:
        raise RuntimeError("suffix manifest/results rollback timestep mismatch")
    timesteps = _validate_requested_timesteps(requested_timesteps, rollback)

    source_record = manifest.get("sources", {}).get("dit")
    checkpoint_record = manifest.get("sources", {}).get("checkpoint")
    vae_record = manifest.get("sources", {}).get("vae")
    if not all(isinstance(item, dict) for item in (source_record, checkpoint_record, vae_record)):
        raise RuntimeError("suffix manifest source/checkpoint/VAE records are incomplete")
    dit_root = _absolute_manifest_path(source_record.get("root"), "DiT source root")
    checkpoint_path = _absolute_manifest_path(
        checkpoint_record.get("path"), "checkpoint path"
    )
    vae_snapshot = _absolute_manifest_path(vae_record.get("snapshot"), "VAE snapshot")
    observer_root = _absolute_manifest_path(
        manifest.get("frozen_observe_bundle", {}).get("root"), "observer root"
    )
    baseline_root = _absolute_manifest_path(
        manifest.get("frozen_baseline", {}).get("root"), "baseline root"
    )

    source = suffix.validate_repository(dit_root, checkpoint_path)
    checkpoint = suffix.validate_checkpoint(checkpoint_path)
    vae = suffix.validate_vae_snapshot(vae_snapshot)
    loader_args = argparse.Namespace(
        observe_dir=observer_root,
        baseline_dir=baseline_root,
        seed=seed,
        dit_root=dit_root,
        checkpoint=checkpoint_path,
        vae_snapshot=vae_snapshot,
    )
    observed = suffix.load_observed_input(
        loader_args, source=source, checkpoint=checkpoint, vae=vae
    )
    _validate_recorded_provenance(
        manifest,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
        observed=observed,
    )
    results = suffix.validate_bundle(
        root,
        manifest=manifest,
        observed=observed,
        rollback_internal_timestep=rollback,
        target_batch_index=target_batch_index,
        require_completion=True,
    )
    if results != results_preliminary:
        raise RuntimeError("suffix results changed during validation")
    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    if completion != completion_preliminary:
        raise RuntimeError("suffix completion changed during validation")

    expected_axis = np.arange(rollback, -1, -1, dtype=np.int16)
    expected_shape = (
        rollback + 1,
        strict.LATENT_CHANNELS,
        strict.LATENT_SIZE,
        strict.LATENT_SIZE,
    )
    branch_traces: list[BranchTrace] = []
    branches = results.get("branches")
    if not isinstance(branches, list) or len(branches) != suffix.TOTAL_BRANCH_COUNT:
        raise RuntimeError("suffix result branch count changed")
    for attempt_index, record in enumerate(branches):
        branch = suffix.branch_id(attempt_index)
        if record.get("branch_id") != branch or record.get("attempt_index") != attempt_index:
            raise RuntimeError(f"suffix branch identity changed: expected {branch}")
        trace_path = root / str(record.get("trace", {}).get("relative_path", ""))
        arrays = suffix._load_npz_exact(  # noqa: SLF001 - canonical strict loader.
            trace_path, record.get("trace", {}), root
        )
        if arrays["target_pred_xstart"].shape != expected_shape:
            raise RuntimeError(f"target_pred_xstart shape changed: {branch}")
        if arrays["target_pred_xstart"].dtype != np.dtype(np.float32):
            raise RuntimeError(f"target_pred_xstart dtype changed: {branch}")
        if not np.isfinite(arrays["target_pred_xstart"]).all():
            raise RuntimeError(f"target_pred_xstart contains non-finite data: {branch}")
        axis = arrays["transition_internal_timestep"]
        if axis.dtype != np.dtype(np.int16) or not np.array_equal(axis, expected_axis):
            raise RuntimeError(f"suffix trace timestep axis changed: {branch}")
        row_by_t = {int(value): row for row, value in enumerate(axis.tolist())}
        rows = tuple(row_by_t[value] for value in timesteps)
        selected = np.ascontiguousarray(
            np.stack([arrays["target_pred_xstart"][row] for row in rows]),
            dtype=np.float32,
        )
        if selected.shape != (len(timesteps), *expected_shape[1:]):
            raise RuntimeError(f"selected latent shape changed: {branch}")
        if not np.array_equal(
            arrays["target_pred_xstart"][-1],
            arrays["final_first_half"][target_batch_index],
        ):
            raise RuntimeError(f"t=0 pred_xstart is not the final target latent: {branch}")
        branch_traces.append(
            BranchTrace(
                branch_id=branch,
                attempt_index=attempt_index,
                role=str(record["role"]),
                trace_record=record["trace"],
                timesteps=axis,
                selected_latents=selected,
                selected_rows=rows,
                selected_hashes=tuple(_array_raw_sha256(item) for item in selected),
            )
        )
    return BundleContext(
        root=root,
        manifest=manifest,
        results=results,
        completion=completion,
        observed=observed,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
        rollback_internal_timestep=rollback,
        target_batch_index=target_batch_index,
        target_class_id=int(target_class_id),
        timesteps=timesteps,
        branches=tuple(branch_traces),
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "diffusers": _package_version("diffusers"),
        "safetensors": _package_version("safetensors"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_build": torch.version.cuda,
    }


def frame_relative_path(branch_id: str, timestep: int) -> str:
    return f"frames/{branch_id}/internal_t{timestep:03d}_pred_xstart.png"


def expected_output_paths(context: BundleContext) -> tuple[str, ...]:
    paths = [
        frame_relative_path(branch.branch_id, timestep)
        for branch in context.branches
        for timestep in context.timesteps
    ]
    paths.append(CONTACT_SHEET_NAME)
    return tuple(paths)


def _device_identity(device_name: str, *, query_runtime: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "requested": device_name,
        "runtime_queried": query_runtime,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if not query_runtime:
        record["reason_not_queried"] = "dry-run does not load the VAE or query CUDA"
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        record.update(
            {
                "torch_device": f"cuda:{index}",
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    elif device_name == "cpu":
        record.update(
            {"torch_device": "cpu", "machine": platform.machine(), "processor": platform.processor()}
        )
    else:
        raise ValueError("decoder device must be cpu or cuda")
    return record


def build_identity(
    context: BundleContext,
    *,
    device_name: str,
    device_runtime: dict[str, Any],
    decode_batch_size: int,
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    suffix_runner = Path(suffix.__file__).resolve()
    decoder_runner = Path(tracevis.__file__).resolve()
    frames = [
        {
            "attempt_index": branch.attempt_index,
            "branch_id": branch.branch_id,
            "role": branch.role,
            "internal_timestep": timestep,
            "trace_row": row,
            "latent_raw_sha256": latent_hash,
            "relative_path": frame_relative_path(branch.branch_id, timestep),
        }
        for branch in context.branches
        for timestep, row, latent_hash in zip(
            context.timesteps,
            branch.selected_rows,
            branch.selected_hashes,
            strict=True,
        )
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "runner_source": {"path": str(runner), "sha256": strict.sha256_file(runner)},
        "role": "POSTHOC_SUFFIX_PRED_XSTART_VISUALIZATION_ONLY",
        "visualization_only": True,
        "posthoc": True,
        "automatic_image_quality_scoring": False,
        "manual_or_automatic_ranking": False,
        "branch_or_timestep_selection": False,
        "alarm_or_threshold_construction": False,
        "sampling_resampling_guidance_or_intervention": False,
        "source_oracle_label_adopted_as_quality_judgment": False,
        "source_suffix_bundle": {
            "root": str(context.root),
            "manifest_identity_sha256": context.manifest["identity_sha256"],
            "manifest_file_sha256": strict.sha256_file(context.root / MANIFEST_NAME),
            "results_payload_sha256": context.results["payload_sha256"],
            "results_file_sha256": strict.sha256_file(context.root / "results.json"),
            "completion_payload_sha256": context.completion["payload_sha256"],
            "completion_file_sha256": strict.sha256_file(context.root / COMPLETION_NAME),
            "strict_suffix_bundle_validation": True,
            "suffix_validator_source": {
                "path": str(suffix_runner),
                "sha256": strict.sha256_file(suffix_runner),
            },
        },
        "target": {
            "batch_index": context.target_batch_index,
            "class_id": context.target_class_id,
            "rollback_internal_timestep": context.rollback_internal_timestep,
            "attempts_in_row_order": [branch.branch_id for branch in context.branches],
            "internal_timesteps_in_column_order": list(context.timesteps),
        },
        "traces": [
            {
                "branch_id": branch.branch_id,
                "attempt_index": branch.attempt_index,
                "role": branch.role,
                "relative_path": branch.trace_record["relative_path"],
                "file_sha256": branch.trace_record["sha256"],
                "target_pred_xstart_array": branch.trace_record["arrays"][
                    "target_pred_xstart"
                ],
                "transition_internal_timestep_array": branch.trace_record["arrays"][
                    "transition_internal_timestep"
                ],
                "complete_trace_file_and_all_arrays_validated": True,
            }
            for branch in context.branches
        ],
        "frames": frames,
        "decoder": {
            "model_id": strict.VAE_MODEL_ID,
            "kind": strict.VAE_KIND,
            "revision": strict.VAE_REVISION,
            "snapshot": context.vae["snapshot"],
            "snapshot_files": context.vae["files"],
            "strict_snapshot_validation": True,
            "latent_divisor": strict.VAE_SCALING_FACTOR,
            "utility_source": {
                "path": str(decoder_runner),
                "sha256": strict.sha256_file(decoder_runner),
                "function": "decode_latents",
            },
            "device": device_name,
            "device_runtime": device_runtime,
            "decode_batch_size": decode_batch_size,
            "dtype": "float32",
            "eval_mode": True,
            "inference_mode": True,
            "offline_only": True,
            "png_mapping_utility": "visualize_dit_imagenet256_trace.decoded_to_pil",
        },
        "layout": {
            "individual_frame_size": [strict.IMAGE_SIZE, strict.IMAGE_SIZE + FRAME_LABEL_HEIGHT],
            "native_decoded_region_size": [strict.IMAGE_SIZE, strict.IMAGE_SIZE],
            "native_decoded_region_resampling": "none",
            "individual_visible_labels": True,
            "contact_sheet_rows": suffix.TOTAL_BRANCH_COUNT,
            "contact_sheet_columns": len(context.timesteps),
            "contact_sheet_row_semantics": "attempts in attempts_in_row_order",
            "contact_sheet_column_semantics": "internal timesteps in column order",
            "contact_sheet_resampling": "none",
            "contact_sheet_visible_visualization_only_banner": True,
        },
        "dependencies": dependency_identity(),
        "expected_outputs": list(expected_output_paths(context)),
        "no_overwrite": True,
    }
    payload["identity_sha256"] = _canonical_self_hash(payload, "identity_sha256")
    return payload


def _labelled_frame(
    native: Image.Image, *, branch_id: str, role: str, timestep: int
) -> Image.Image:
    if native.mode != "RGB" or native.size != (strict.IMAGE_SIZE, strict.IMAGE_SIZE):
        raise ValueError("native decoded frame must be 256x256 RGB")
    canvas = Image.new(
        "RGB", (strict.IMAGE_SIZE, strict.IMAGE_SIZE + FRAME_LABEL_HEIGHT), BACKGROUND
    )
    canvas.paste(native, (0, 0))
    draw = ImageDraw.Draw(canvas)
    role_label = "REPLAY" if role == "exact_replay_control" else "FRESH SUFFIX"
    draw.text((4, strict.IMAGE_SIZE + 3), f"{branch_id} | {role_label}", fill=TEXT)
    draw.text(
        (4, strict.IMAGE_SIZE + 19),
        f"pred_xstart | internal t={timestep} | POSTHOC VIS ONLY",
        fill=ACCENT_TEXT,
    )
    return canvas


def contact_sheet_size(columns: int, rows: int) -> tuple[int, int]:
    if columns < 1 or rows < 1:
        raise ValueError("contact sheet must have positive row/column counts")
    width = (
        2 * OUTER_MARGIN
        + SHEET_LEFT_GUTTER
        + columns * strict.IMAGE_SIZE
        + (columns - 1) * CELL_GAP
    )
    height = (
        2 * OUTER_MARGIN
        + SHEET_HEADER_HEIGHT
        + rows * (strict.IMAGE_SIZE + SHEET_CELL_LABEL_HEIGHT)
        + (rows - 1) * CELL_GAP
    )
    return width, height


def render_contact_sheet(
    native_images: Sequence[Sequence[Image.Image]], context: BundleContext
) -> Image.Image:
    rows = len(context.branches)
    columns = len(context.timesteps)
    if len(native_images) != rows or any(len(row) != columns for row in native_images):
        raise ValueError("contact-sheet image matrix shape mismatch")
    sheet = Image.new("RGB", contact_sheet_size(columns, rows), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (OUTER_MARGIN, OUTER_MARGIN),
        "POSTHOC PRED_XSTART VISUALIZATION ONLY | NO SCORING / RANKING / SELECTION",
        fill=ACCENT_TEXT,
    )
    draw.text(
        (OUTER_MARGIN, OUTER_MARGIN + 17),
        f"class {context.target_class_id:04d} | rollback R={context.rollback_internal_timestep}",
        fill=MUTED_TEXT,
    )
    image_origin_x = OUTER_MARGIN + SHEET_LEFT_GUTTER
    image_origin_y = OUTER_MARGIN + SHEET_HEADER_HEIGHT
    for column, timestep in enumerate(context.timesteps):
        x = image_origin_x + column * (strict.IMAGE_SIZE + CELL_GAP)
        draw.text((x + 3, OUTER_MARGIN + 31), f"internal t={timestep}", fill=TEXT)
    for row, (branch, images) in enumerate(zip(context.branches, native_images, strict=True)):
        y = image_origin_y + row * (
            strict.IMAGE_SIZE + SHEET_CELL_LABEL_HEIGHT + CELL_GAP
        )
        role_label = "replay" if branch.role == "exact_replay_control" else "fresh suffix"
        draw.text((OUTER_MARGIN + 3, y + 5), branch.branch_id, fill=TEXT)
        draw.text((OUTER_MARGIN + 3, y + 21), role_label, fill=MUTED_TEXT)
        for column, (native, timestep) in enumerate(
            zip(images, context.timesteps, strict=True)
        ):
            if native.mode != "RGB" or native.size != (
                strict.IMAGE_SIZE,
                strict.IMAGE_SIZE,
            ):
                raise ValueError("contact sheet source must be native 256x256 RGB")
            x = image_origin_x + column * (strict.IMAGE_SIZE + CELL_GAP)
            sheet.paste(native, (x, y))
            draw.text(
                (x + 3, y + strict.IMAGE_SIZE + 3),
                f"{branch.branch_id} | t={timestep} | pred_xstart",
                fill=TEXT,
            )
    return sheet


def _frame_metadata(identity: dict[str, Any], frame: dict[str, Any]) -> dict[str, str]:
    return {
        "runner": RUNNER_NAME,
        "identity_sha256": identity["identity_sha256"],
        "representation": "target_pred_xstart",
        "branch_id": frame["branch_id"],
        "attempt_index": str(frame["attempt_index"]),
        "role": frame["role"],
        "internal_timestep": str(frame["internal_timestep"]),
        "trace_row": str(frame["trace_row"]),
        "source_trace_file_sha256": next(
            item["file_sha256"]
            for item in identity["traces"]
            if item["branch_id"] == frame["branch_id"]
        ),
        "latent_raw_sha256": frame["latent_raw_sha256"],
        "native_decoded_region": "0,0,256,256",
        "native_decoded_region_resampling": "none",
        "visualization_only": "true",
        "posthoc": "true",
        "quality_scoring": "false",
        "ranking_or_selection": "false",
    }


def _sheet_metadata(identity: dict[str, Any]) -> dict[str, str]:
    target = identity["target"]
    return {
        "runner": RUNNER_NAME,
        "identity_sha256": identity["identity_sha256"],
        "representation": "target_pred_xstart",
        "display": "attempt rows by internal-timestep columns",
        "attempts_in_row_order": ",".join(target["attempts_in_row_order"]),
        "internal_timesteps_in_column_order": ",".join(
            str(item) for item in target["internal_timesteps_in_column_order"]
        ),
        "contact_sheet_resampling": "none",
        "visualization_only": "true",
        "posthoc": "true",
        "quality_scoring": "false",
        "ranking_or_selection": "false",
    }


def _save_png(image: Image.Image, path: Path, metadata_fields: dict[str, str]) -> None:
    if image.mode != "RGB":
        raise ValueError("visualization PNG must be RGB")
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to overwrite PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    for key, value in metadata_fields.items():
        metadata.add_text(key, value)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, path)


def _expected_png_metadata(identity: dict[str, Any], relative_path: str) -> dict[str, str]:
    if relative_path == CONTACT_SHEET_NAME:
        return _sheet_metadata(identity)
    for frame in identity["frames"]:
        if frame["relative_path"] == relative_path:
            return _frame_metadata(identity, frame)
    raise RuntimeError(f"unexpected visualization output: {relative_path}")


def inspect_png_outputs(root: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    target = identity["target"]
    sheet_size = contact_sheet_size(
        len(target["internal_timesteps_in_column_order"]),
        len(target["attempts_in_row_order"]),
    )
    for relative in sorted(identity["expected_outputs"]):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/non-regular visualization PNG: {path}")
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            size = image.size
            pixels = image.tobytes()
            metadata = dict(image.info)
        expected_size = (
            sheet_size
            if relative == CONTACT_SHEET_NAME
            else (strict.IMAGE_SIZE, strict.IMAGE_SIZE + FRAME_LABEL_HEIGHT)
        )
        if mode != "RGB" or size != expected_size:
            raise RuntimeError(f"visualization PNG geometry changed: {path}: {mode}/{size}")
        if metadata != _expected_png_metadata(identity, relative):
            raise RuntimeError(f"visualization PNG metadata changed: {path}")
        records.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": strict.sha256_file(path),
                "pixel_sha256": _sha256_bytes(pixels),
                "mode": mode,
                "size": list(size),
            }
        )
    return records


def _validate_closed_tree(root: Path, identity: dict[str, Any]) -> None:
    expected_files = {
        (root / relative).absolute() for relative in identity["expected_outputs"]
    } | {(root / MANIFEST_NAME).absolute(), (root / COMPLETION_NAME).absolute()}
    expected_dirs = {(root / "frames").absolute()}
    expected_dirs.update(
        (root / "frames" / branch_id).absolute()
        for branch_id in identity["target"]["attempts_in_row_order"]
    )
    actual_files: set[Path] = set()
    actual_dirs: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"visualization output contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.absolute())
        elif path.is_dir():
            actual_dirs.add(path.absolute())
        else:
            raise RuntimeError(f"visualization output contains a special entry: {path}")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise RuntimeError(
            "visualization output tree is not closed; "
            f"missing_files={sorted(expected_files-actual_files)[:2]}, "
            f"extra_files={sorted(actual_files-expected_files)[:2]}, "
            f"missing_dirs={sorted(expected_dirs-actual_dirs)[:2]}, "
            f"extra_dirs={sorted(actual_dirs-expected_dirs)[:2]}"
        )


def validate_completed_output(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"visualization output is not a plain directory: {root}")
    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "payload_sha256")
    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError("visualization manifest is not complete")
    if manifest.get("identity") != identity:
        raise RuntimeError("visualization identity changed")
    runner = Path(__file__).resolve()
    if identity.get("runner_source") != {
        "path": str(runner),
        "sha256": strict.sha256_file(runner),
    }:
        raise RuntimeError("visualization was produced by a different runner source")
    records = inspect_png_outputs(root, identity)
    outputs_sha256 = strict.sha256_json(records)
    if manifest.get("outputs") != records or manifest.get("outputs_sha256") != outputs_sha256:
        raise RuntimeError("visualization output hashes changed")
    _validate_closed_tree(root, identity)
    fixed_completion = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "identity_sha256": identity["identity_sha256"],
        "manifest_payload_sha256": manifest["payload_sha256"],
        "manifest_file_sha256": strict.sha256_file(root / MANIFEST_NAME),
        "outputs_sha256": outputs_sha256,
        "output_count": len(records),
    }
    mismatches = {
        key: (completion.get(key), value)
        for key, value in fixed_completion.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"visualization completion links/hashes changed: {mismatches}")
    return manifest


def _endpoint_decode_audit(
    context: BundleContext, native_images: Sequence[Sequence[Image.Image]]
) -> list[dict[str, Any]]:
    if 0 not in context.timesteps:
        return [
            {
                "branch_id": branch.branch_id,
                "t0_requested": False,
                "pixel_comparison_performed": False,
                "pixel_equality_required": False,
            }
            for branch in context.branches
        ]
    ordinal = context.timesteps.index(0)
    audits: list[dict[str, Any]] = []
    for branch, images in zip(context.branches, native_images, strict=True):
        visualized = np.ascontiguousarray(np.asarray(images[ordinal], dtype=np.uint8))
        source_path = context.root / "branches" / branch.branch_id / "target.png"
        with Image.open(source_path) as source_image:
            source_image.load()
            if source_image.mode != "RGB" or source_image.size != (
                strict.IMAGE_SIZE,
                strict.IMAGE_SIZE,
            ):
                raise RuntimeError(f"source target PNG geometry changed: {source_path}")
            source_pixels = np.ascontiguousarray(np.asarray(source_image, dtype=np.uint8))
        difference = np.abs(visualized.astype(np.int16) - source_pixels.astype(np.int16))
        audits.append(
            {
                "branch_id": branch.branch_id,
                "t0_requested": True,
                "pixel_comparison_performed": True,
                "pixel_equality_required": False,
                "source_target_relative_path": f"branches/{branch.branch_id}/target.png",
                "source_target_png_sha256": strict.sha256_file(source_path),
                "pixel_equal": bool(np.array_equal(visualized, source_pixels)),
                "differing_pixels": int(np.count_nonzero(np.any(difference != 0, axis=2))),
                "differing_channel_values": int(np.count_nonzero(difference)),
                "maximum_absolute_channel_difference": int(difference.max(initial=0)),
                "interpretation": (
                    "Decoder/backend consistency audit only; never an image-quality score. "
                    "The validated t=0 latent is exact even if quantization-boundary pixels differ."
                ),
            }
        )
    return audits


def run_real(
    args: argparse.Namespace, context: BundleContext, identity: dict[str, Any]
) -> None:
    outdir = args.outdir
    if os.path.lexists(outdir):
        raise RuntimeError(f"refusing to overwrite output path: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    all_latents = np.ascontiguousarray(
        np.concatenate([branch.selected_latents for branch in context.branches], axis=0),
        dtype=np.float32,
    )
    decoded = tracevis.decode_latents(
        all_latents,
        vae_snapshot=Path(context.vae["snapshot"]),
        device_name=args.device,
        batch_size=args.decode_batch_size,
    )
    expected_count = len(context.branches) * len(context.timesteps)
    if decoded.shape[0] != expected_count:
        raise RuntimeError("decoded frame count changed")
    native_matrix: list[list[Image.Image]] = []
    offset = 0
    for _branch in context.branches:
        row = [
            tracevis.decoded_to_pil(decoded[offset + column])
            for column in range(len(context.timesteps))
        ]
        native_matrix.append(row)
        offset += len(context.timesteps)
    if offset != expected_count:
        raise AssertionError("decoded frame matrix split failed")

    frame_by_key = {
        (frame["branch_id"], frame["internal_timestep"]): frame
        for frame in identity["frames"]
    }
    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary)
        for branch, images in zip(context.branches, native_matrix, strict=True):
            for timestep, native in zip(context.timesteps, images, strict=True):
                frame = frame_by_key[(branch.branch_id, timestep)]
                labelled = _labelled_frame(
                    native,
                    branch_id=branch.branch_id,
                    role=branch.role,
                    timestep=timestep,
                )
                _save_png(
                    labelled,
                    staging / frame["relative_path"],
                    _frame_metadata(identity, frame),
                )
        sheet = render_contact_sheet(native_matrix, context)
        _save_png(sheet, staging / CONTACT_SHEET_NAME, _sheet_metadata(identity))

        output_records = inspect_png_outputs(staging, identity)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "outputs": output_records,
            "outputs_sha256": strict.sha256_json(output_records),
            "endpoint_decode_audit": _endpoint_decode_audit(context, native_matrix),
        }
        manifest["payload_sha256"] = _canonical_self_hash(manifest, "payload_sha256")
        strict.atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "identity_sha256": identity["identity_sha256"],
            "manifest_payload_sha256": manifest["payload_sha256"],
            "manifest_file_sha256": strict.sha256_file(staging / MANIFEST_NAME),
            "outputs_sha256": manifest["outputs_sha256"],
            "output_count": len(output_records),
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        strict.atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_completed_output(staging, identity)
        tracevis._publish_directory_noreplace(  # noqa: SLF001 - strict atomic helper.
            staging, outdir
        )
    validate_completed_output(outdir, identity)
    print(
        json.dumps(
            {
                "complete": True,
                "outdir": str(outdir),
                "contact_sheet": str(outdir / CONTACT_SHEET_NAME),
                "rollback_internal_timestep": context.rollback_internal_timestep,
                "attempt_rows": [branch.branch_id for branch in context.branches],
                "internal_timestep_columns": list(context.timesteps),
                "frame_count": expected_count,
                "visualization_only": True,
                "posthoc": True,
                "quality_scoring": False,
                "ranking_or_selection": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def dry_run(args: argparse.Namespace, context: BundleContext, identity: dict[str, Any]) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("dry-run unexpectedly initialized CUDA")
    print(
        json.dumps(
            {
                "status": "dry-run",
                "suffix_bundle": str(context.root),
                "strict_suffix_bundle_and_all_branch_traces_validated": True,
                "rollback_internal_timestep": context.rollback_internal_timestep,
                "attempt_rows": [branch.branch_id for branch in context.branches],
                "internal_timestep_columns": list(context.timesteps),
                "expected_outputs": identity["expected_outputs"],
                "outdir": str(args.outdir),
                "gpu_or_vae_model_loaded": False,
                "visualization_only": True,
                "posthoc": True,
                "quality_scoring": False,
                "ranking_or_selection": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_self_test() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must start without CUDA initialization")
    for rollback, expected in DEFAULT_TIMESTEPS_BY_ROLLBACK.items():
        if _validate_requested_timesteps(None, rollback) != expected:
            raise AssertionError("rollback-specific default timestep selection failed")
    decoded = np.zeros((3, strict.IMAGE_SIZE, strict.IMAGE_SIZE), dtype=np.float32)
    decoded[0], decoded[1], decoded[2] = -1.0, 0.0, 1.0
    native = tracevis.decoded_to_pil(decoded)
    labelled = _labelled_frame(
        native,
        branch_id="attempt_000",
        role="exact_replay_control",
        timestep=60,
    )
    if labelled.size != (strict.IMAGE_SIZE, strict.IMAGE_SIZE + FRAME_LABEL_HEIGHT):
        raise AssertionError("labelled native-frame geometry changed")
    if not np.array_equal(np.asarray(labelled)[: strict.IMAGE_SIZE], np.asarray(native)):
        raise AssertionError("label strip changed the native decoded image region")

    synthetic_context = argparse.Namespace(
        target_class_id=int(strict.CLASS_IDS[0]),
        rollback_internal_timestep=60,
        timesteps=(60, 0),
        branches=(
            argparse.Namespace(
                branch_id="attempt_000", attempt_index=0, role="exact_replay_control"
            ),
            argparse.Namespace(
                branch_id="attempt_001", attempt_index=1, role="fresh_target_suffix_attempt"
            ),
        ),
    )
    sheet = render_contact_sheet([[native, native], [native, native]], synthetic_context)
    if sheet.size != contact_sheet_size(2, 2):
        raise AssertionError("attempt-row/timestep-column sheet geometry changed")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: rollback defaults, native decoded-region preservation, "
        "attempt-row/timestep-column sheet layout, CPU-only"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix-dir", type=Path, default=None)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Internal timesteps in column order. Defaults by rollback: "
            "R60=60,50,40,30,20,10,0; R120=120,100,80,60,40,20,0; "
            "R180=180,150,120,90,60,30,0."
        ),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--decode-batch-size", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def _normalize_output_path(
    requested: Path | None, context: BundleContext
) -> Path:
    if requested is None:
        data_root = Path(
            os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")
        ).expanduser().resolve()
        requested = (
            data_root
            / "cross_scale_evidence/dit_imagenet256_suffix_predxstart_visualizations"
            / f"{context.root.name}_predxstart_v1"
        )
    requested = requested.expanduser().absolute()
    if os.path.lexists(requested) and requested.is_symlink():
        raise ValueError(f"output path must not be a symlink: {requested}")
    outdir = requested.resolve()
    protected = {
        "suffix bundle": context.root,
        "observer bundle": context.observed.root,
        "frozen baseline": context.observed.baseline.root,
        "DiT source": Path(context.source["root"]),
        "checkpoint": Path(context.checkpoint["path"]),
        "VAE snapshot": Path(context.vae["snapshot"]),
        "research repository": Path(__file__).resolve().parent.parent,
    }
    overlaps = [label for label, path in protected.items() if _paths_overlap(outdir, path)]
    if overlaps:
        raise ValueError(
            "output path overlaps protected input/source path(s): " + ", ".join(overlaps)
        )
    return outdir


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.suffix_dir is None:
        parser.error("--suffix-dir is required unless --self-test is used")
    if args.decode_batch_size < 1:
        parser.error("--decode-batch-size must be positive")
    requested = args.suffix_dir.expanduser().absolute()
    try:
        context = validate_suffix_bundle(requested, args.timesteps)
        args.outdir = _normalize_output_path(args.outdir, context)
        if os.path.lexists(args.outdir):
            raise ValueError(f"no-overwrite target already exists: {args.outdir}")
        identity = build_identity(
            context,
            device_name=args.device,
            device_runtime=_device_identity(args.device, query_runtime=not args.dry_run),
            decode_batch_size=args.decode_batch_size,
        )
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        dry_run(args, context, identity)
    else:
        run_real(args, context, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
