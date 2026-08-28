#!/usr/bin/env python3
"""Visualize one validated DiT suffix cross-scale replay bundle, read-only.

This program consumes the raw ``cross_scale_replay.npz`` artifact directly;
it never reads an offline summary and never modifies either the raw replay or
its summarizer.  Before using an array it calls the raw runner's fail-closed
``validate_output_bundle`` routine, which checks the closed file set, JSON and
array hashes, schemas, schedules, reconstructed score directions, Gaussian
likelihood-ratio decomposition, cumulative evidence, and completion links.

For a bundle with the frozen four fresh attempts and three Delta-nu scales it
exports these fixed, label-independent views:

* a 4-attempt x 3-scale panel of 4x4 local-tile +theta running-maximum
  log-e heatmaps, with all 16 row-major tiles and one shared color scale;
* three trajectory panels for the raw runner's uniform fixed mixture over all
  3 x (global + 16 local tiles) = 51 +theta components, the corresponding
  fixed 50/50 sign mixture, and a uniform fixed-start change-point/sign
  mixture;
* the same three trajectories for the explicitly separate 3 x 16 = 48 local
  tiles only (never presented as the raw runner's saved 51-component mix);
* for rollback t=60 only, component trajectories for fixed ``tile_12``.  That
  tile was noticed after inspecting the discovery example, so the figure is
  prominently marked posthoc and is ineligible as a confirmatory detector.
* for rollback t=60 only, the four strictly hash-validated endpoint PNGs with
  an identical nominal latent-to-image 4x4 grid.  Tiles 0..15 are labelled and
  only tile_12 receives one consistent outline.  The figure explicitly notes
  that convolutional VAE receptive fields make the box nominal, not hard
  pixel support.

Every trajectory row indexed by internal timestep ``t`` is the evidence after
the transition ``t -> t-1``.  Optional v2 quality labels may add parenthetical
posthoc text only.  They cannot alter calculations, included attempts/scales,
ordering, color, line style, limits, or filenames.  Outputs are staged,
self-hashed, validated as a closed tree, atomically installed without replace,
and never overwrite an existing path.
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
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib import patheffects
from PIL import Image

try:  # Package and direct CLI imports.
    from .intervene_dit_imagenet256_suffix import (
        _atomic_install_directory_noreplace,
    )
    from .replay_dit_suffix_cross_scale_diagnostics import (
        COMPONENT_COUNT,
        EXPERIMENT as RAW_EXPERIMENT,
        GRID_SIZE,
        LOCAL_COMPONENT_COUNT,
        TRACE_NAME as RAW_TRACE_NAME,
        validate_output_bundle as validate_raw_bundle,
    )
    from .reproduce_dit_imagenet256 import (
        atomic_json_dump,
        load_json,
        sha256_file,
        sha256_json,
    )
    from . import visualize_dit_suffix_predxstart as suffix_endpoint_validator
except ImportError:  # pragma: no cover - direct CLI invocation.
    from intervene_dit_imagenet256_suffix import (
        _atomic_install_directory_noreplace,
    )
    from replay_dit_suffix_cross_scale_diagnostics import (
        COMPONENT_COUNT,
        EXPERIMENT as RAW_EXPERIMENT,
        GRID_SIZE,
        LOCAL_COMPONENT_COUNT,
        TRACE_NAME as RAW_TRACE_NAME,
        validate_output_bundle as validate_raw_bundle,
    )
    from reproduce_dit_imagenet256 import (
        atomic_json_dump,
        load_json,
        sha256_file,
        sha256_json,
    )
    import visualize_dit_suffix_predxstart as suffix_endpoint_validator


EXPERIMENT = "dit_imagenet256_suffix_cross_scale_replay_visualization"
SCHEMA_VERSION = 1
EXPECTED_ATTEMPTS = (1, 2, 3, 4)
EXPECTED_SCALE_COUNT = 3
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"
COMPLETION_NAME = "completion.json"
HEATMAP_NAME = "local_tile_plus_running_max_heatmaps.png"
RAW_MIXTURE_NAME = "scale_global_plus_tile_mixture_trajectories.png"
LOCAL_MIXTURE_NAME = "scale_local_tile_only_mixture_trajectories.png"
T60_TILE12_NAME = "t60_tile12_posthoc_component_trajectories.png"
T60_ENDPOINT_GRID_NAME = "t60_endpoints_posthoc_tile12_nominal_grid.png"

INK = "#263238"
MUTED = "#66727A"
GRID = "#D7DEE2"
ZERO = "#6F7780"
ATTEMPT_COLORS = ("#3B6FB6", "#C99724", "#D76A2F", "#758C3A")
ATTEMPT_LINESTYLES = ("-", "--", "-.", ":")
ATTEMPT_MARKERS = ("o", "s", "^", "D")
SCALE_COLORS = ("#3B6FB6", "#C99724", "#D76A2F")
SCALE_LINESTYLES = ("-", "--", "-.")
SCALE_MARKERS = ("o", "s", "^")


@dataclass(frozen=True)
class ReplayContext:
    root: Path
    manifest: dict[str, Any]
    results: dict[str, Any]
    arrays: dict[str, np.ndarray]
    rollback: int
    attempts: tuple[int, ...]
    delta_nu: tuple[float, ...]
    internal_timestep: np.ndarray


@dataclass(frozen=True)
class EvidencePaths:
    plus_component: np.ndarray
    minus_component: np.ndarray
    sign_component: np.ndarray
    change_point_sign_component: np.ndarray
    raw_plus: np.ndarray
    raw_sign: np.ndarray
    raw_change_point_sign: np.ndarray
    local_plus: np.ndarray
    local_sign: np.ndarray
    local_change_point_sign: np.ndarray
    local_tile_running_max: np.ndarray


@dataclass(frozen=True)
class EndpointImage:
    attempt_index: int
    path: Path
    record: dict[str, Any]
    pixels: np.ndarray


@dataclass(frozen=True)
class EndpointContext:
    suffix_root: Path
    suffix_manifest_identity_sha256: str
    suffix_manifest_file_sha256: str
    suffix_results_payload_sha256: str
    suffix_results_file_sha256: str
    suffix_completion_file_sha256: str
    validator_source: dict[str, str]
    images: tuple[EndpointImage, ...]
    tile_bounds_latent_yxyx: tuple[tuple[int, int, int, int], ...]
    tile_bounds_nominal_image_yxyx: tuple[tuple[int, int, int, int], ...]


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = load_json(path)
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def _array_raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _logsumexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("logsumexp input contains non-finite values")
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    axes = (axis,) if isinstance(axis, int) else axis
    for item in sorted((value % values.ndim for value in axes), reverse=True):
        result = np.squeeze(result, axis=item)
    return np.asarray(result, dtype=np.float64)


def _logmeanexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """Match the raw replay runner's saved-mixture arithmetic exactly."""

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("logmeanexp input contains non-finite values")
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.mean(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    axes = (axis,) if isinstance(axis, int) else axis
    for item in sorted((value % values.ndim for value in axes), reverse=True):
        result = np.squeeze(result, axis=item)
    return np.asarray(result, dtype=np.float64)


def fixed_sign_log_mixture(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    """Fixed 50/50 mixture of the +theta and -theta e-processes."""

    if plus.shape != minus.shape:
        raise ValueError("plus/minus paths must have the same shape")
    return _logmeanexp(np.stack([plus, minus], axis=0), axis=0)


def uniform_change_point_log_mixture(
    increments: np.ndarray, *, start_count: int
) -> np.ndarray:
    """Uniform fixed-prior change-point log mixture with time on the last axis.

    Starts that have not launched yet contribute E=1.  The terminal t=0 row
    is bookkeeping and does not add another candidate start.
    """

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1 or not np.isfinite(values).all():
        raise ValueError("increments must have a finite non-empty time axis")
    if not 1 <= start_count <= values.shape[-1]:
        raise ValueError("start_count must lie within the time axis")
    prefix = np.cumsum(values, axis=-1, dtype=np.float64)
    output = np.empty_like(prefix)
    leading = values.shape[:-1]
    for time_index in range(values.shape[-1]):
        launched = min(time_index + 1, start_count)
        starts = np.arange(launched, dtype=np.int64)
        before = np.zeros(leading + (launched,), dtype=np.float64)
        if launched > 1:
            before[..., 1:] = prefix[..., starts[1:] - 1]
        active = prefix[..., time_index, None] - before
        future_count = start_count - launched
        if future_count:
            terms = np.concatenate(
                [active, np.zeros(leading + (future_count,), dtype=np.float64)],
                axis=-1,
            )
        else:
            terms = active
        output[..., time_index] = _logsumexp(terms, axis=-1) - math.log(start_count)
    return np.ascontiguousarray(output)


def _load_trace_after_validation(
    root: Path, results: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Reopen the trace after validation and repeat its identity checks."""

    record = results.get("trace", {})
    path = root / RAW_TRACE_NAME
    if (
        record.get("relative_path") != RAW_TRACE_NAME
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError("raw replay trace identity changed after validation")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if sorted(arrays) != record.get("keys"):
        raise RuntimeError("raw replay trace key set changed after validation")
    for key, value in arrays.items():
        expected = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_bytes_sha256": _array_raw_sha256(value),
        }
        if record.get("arrays", {}).get(key) != expected:
            raise RuntimeError(f"raw replay array identity changed after validation: {key}")
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise RuntimeError(f"raw replay array contains non-finite values: {key}")
    return arrays


def validate_and_load_raw_bundle(root: Path) -> ReplayContext:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"raw replay bundle is not a plain directory: {root}")
    manifest, results = validate_raw_bundle(root)
    if manifest.get("experiment") != RAW_EXPERIMENT:
        raise RuntimeError("unexpected raw replay experiment")
    arrays = _load_trace_after_validation(root, results)
    attempts = tuple(int(value) for value in arrays["attempt_index"].tolist())
    delta_nu = tuple(float(value) for value in arrays["delta_nu"].tolist())
    internal = arrays["internal_timestep"]
    rollback = int(internal[0])
    expected_components = np.asarray(
        ["global", *(f"tile_{index:02d}" for index in range(LOCAL_COMPONENT_COUNT))],
        dtype="<U16",
    )
    if attempts != EXPECTED_ATTEMPTS:
        raise RuntimeError(f"visual layout requires frozen attempts {EXPECTED_ATTEMPTS}: {attempts}")
    if len(delta_nu) != EXPECTED_SCALE_COUNT or len(set(delta_nu)) != len(delta_nu):
        raise RuntimeError("visual layout requires exactly three distinct Delta-nu scales")
    if COMPONENT_COUNT != 17 or LOCAL_COMPONENT_COUNT != 16 or GRID_SIZE != 4:
        raise RuntimeError("raw global + row-major 4x4 tile component contract changed")
    if not np.array_equal(arrays["component_name"], expected_components):
        raise RuntimeError("raw replay component names/order changed")
    if not np.array_equal(internal, np.arange(rollback, -1, -1, dtype=np.int16)):
        raise RuntimeError("raw replay time axis is not a complete descending suffix")
    if manifest.get("evidence_indexing") != (
        "trace row with internal timestep t stores the increment for transition t->t-1; "
        "component_log_e at that row is post-transition evidence"
    ):
        raise RuntimeError("raw replay evidence indexing semantics changed")
    return ReplayContext(
        root=root,
        manifest=manifest,
        results=results,
        arrays=arrays,
        rollback=rollback,
        attempts=attempts,
        delta_nu=delta_nu,
        internal_timestep=internal,
    )


def validate_and_load_t60_endpoints(context: ReplayContext) -> EndpointContext | None:
    """Strictly validate and load the four endpoint PNGs bound by raw replay.

    The original suffix bundle validator checks the entire closed suffix
    artifact, all endpoint file/pixel hashes, target/full-grid equality,
    traces, and transitions.  This function then repeats the four selected
    endpoint identities after reopening their PNGs.
    """

    if context.rollback != 60:
        return None
    raw_record = context.manifest.get("input_suffix_bundle")
    if not isinstance(raw_record, dict):
        raise RuntimeError("raw replay lacks its bound suffix bundle")
    raw_root_value = raw_record.get("root")
    if not isinstance(raw_root_value, str) or not Path(raw_root_value).is_absolute():
        raise RuntimeError("raw replay suffix root is missing or not absolute")
    suffix_root = Path(raw_root_value).resolve()
    suffix_context = suffix_endpoint_validator.validate_suffix_bundle(
        suffix_root, requested_timesteps=(0,)
    )
    if (
        suffix_context.rollback_internal_timestep != context.rollback
        or suffix_context.target_class_id
        != context.manifest.get("cfg", {}).get("conditional_class_id")
    ):
        raise RuntimeError("strictly validated suffix target does not match raw replay")
    expected_binding = {
        "root": suffix_root,
        "manifest_identity_sha256": suffix_context.manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(suffix_root / "manifest.json"),
        "results_file_sha256": sha256_file(suffix_root / "results.json"),
        "completion_file_sha256": sha256_file(suffix_root / "completion.json"),
    }
    for key, expected in expected_binding.items():
        observed = (
            Path(str(raw_record.get(key))).resolve()
            if key == "root"
            else raw_record.get(key)
        )
        if observed != expected:
            raise RuntimeError(f"raw replay/suffix endpoint binding changed: {key}")
    results = suffix_context.results
    results_payload = results.get("payload_sha256")
    if not isinstance(results_payload, str):
        raise RuntimeError("validated suffix results lack a payload identity")

    branch_records = results.get("branches")
    if not isinstance(branch_records, list):
        raise RuntimeError("validated suffix results lack branch records")
    by_attempt = {int(item["attempt_index"]): item for item in branch_records}
    if len(by_attempt) != len(branch_records):
        raise RuntimeError("validated suffix branch attempts are duplicated")
    images: list[EndpointImage] = []
    for attempt in context.attempts:
        record = by_attempt.get(attempt)
        if record is None or record.get("branch_id") != f"attempt_{attempt:03d}":
            raise RuntimeError(f"validated suffix lacks endpoint attempt {attempt}")
        image_record = record.get("target_image")
        if not isinstance(image_record, dict):
            raise RuntimeError(f"validated suffix lacks target image record: attempt {attempt}")
        expected_relative = f"branches/attempt_{attempt:03d}/target.png"
        path = suffix_root / expected_relative
        if (
            image_record.get("relative_path") != expected_relative
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != image_record.get("bytes")
            or sha256_file(path) != image_record.get("sha256")
        ):
            raise RuntimeError(f"endpoint PNG file identity failed: attempt {attempt}")
        with Image.open(path) as image:
            image.load()
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (256, 256)
                or image_record.get("mode") != "RGB"
                or image_record.get("size") != [256, 256]
            ):
                raise RuntimeError(f"endpoint PNG geometry changed: attempt {attempt}")
            pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        if hashlib.sha256(pixels.tobytes(order="C")).hexdigest() != image_record.get(
            "pixel_sha256"
        ):
            raise RuntimeError(f"endpoint PNG pixel hash failed: attempt {attempt}")
        images.append(
            EndpointImage(
                attempt_index=attempt,
                path=path,
                record=dict(image_record),
                pixels=pixels,
            )
        )

    latent_bounds_array = context.arrays["tile_bounds_yxyx"].astype(np.int64)
    latent_height, latent_width = context.arrays["saved_state_before"].shape[-2:]
    if (
        latent_bounds_array.shape != (LOCAL_COMPONENT_COUNT, 4)
        or 256 % latent_height
        or 256 % latent_width
    ):
        raise RuntimeError("latent-to-image nominal grid cannot be mapped exactly")
    scale_y, scale_x = 256 // latent_height, 256 // latent_width
    latent_bounds = tuple(tuple(int(value) for value in row) for row in latent_bounds_array)
    nominal_image_bounds = tuple(
        (
            int(y0 * scale_y),
            int(x0 * scale_x),
            int(y1 * scale_y),
            int(x1 * scale_x),
        )
        for y0, x0, y1, x1 in latent_bounds
    )
    if nominal_image_bounds[12] != (192, 0, 256, 64):
        raise RuntimeError("fixed row-major tile_12 nominal image mapping changed")
    validator_runner = Path(suffix_endpoint_validator.__file__).resolve()
    return EndpointContext(
        suffix_root=suffix_root,
        suffix_manifest_identity_sha256=suffix_context.manifest["identity_sha256"],
        suffix_manifest_file_sha256=sha256_file(suffix_root / "manifest.json"),
        suffix_results_payload_sha256=results_payload,
        suffix_results_file_sha256=sha256_file(suffix_root / "results.json"),
        suffix_completion_file_sha256=sha256_file(suffix_root / "completion.json"),
        validator_source={
            "path": str(validator_runner),
            "sha256": sha256_file(validator_runner),
        },
        images=tuple(images),
        tile_bounds_latent_yxyx=latent_bounds,
        tile_bounds_nominal_image_yxyx=nominal_image_bounds,
    )


def reconstruct_evidence(context: ReplayContext) -> EvidencePaths:
    arrays = context.arrays
    reward = arrays["R_component"].astype(np.float64, copy=False)
    cost = arrays["K_component"].astype(np.float64, copy=False)
    plus_increment = reward - cost
    minus_increment = -reward - cost
    if not np.array_equal(plus_increment, arrays["L_component"]):
        raise RuntimeError("+theta Gaussian LR is not exactly R-K")
    plus = np.cumsum(plus_increment, axis=2, dtype=np.float64)
    minus = np.cumsum(minus_increment, axis=2, dtype=np.float64)
    if not np.array_equal(plus, arrays["component_log_e"]):
        raise RuntimeError("+theta component paths do not exactly reconstruct")
    sign = fixed_sign_log_mixture(plus, minus)

    start_count = int(np.count_nonzero(context.internal_timestep > 0))
    cp_plus_last = uniform_change_point_log_mixture(
        np.moveaxis(plus_increment, 2, -1), start_count=start_count
    )
    cp_minus_last = uniform_change_point_log_mixture(
        np.moveaxis(minus_increment, 2, -1), start_count=start_count
    )
    cp_sign = np.moveaxis(fixed_sign_log_mixture(cp_plus_last, cp_minus_last), -1, 2)

    raw_plus = _logmeanexp(plus, axis=(0, 3))
    if not np.array_equal(raw_plus, arrays["all_component_mixture_log_e"]):
        raise RuntimeError("raw saved 51-component +theta mixture does not reconstruct")
    raw_sign = _logmeanexp(sign, axis=(0, 3))
    raw_cp_sign = _logmeanexp(cp_sign, axis=(0, 3))
    local_plus = _logmeanexp(plus[..., 1:], axis=(0, 3))
    local_sign = _logmeanexp(sign[..., 1:], axis=(0, 3))
    local_cp_sign = _logmeanexp(cp_sign[..., 1:], axis=(0, 3))
    tile_running_max = np.maximum(0.0, np.max(plus[..., 1:], axis=2))

    expected_path_shape = (len(context.attempts), len(context.internal_timestep))
    for name, value in {
        "raw_plus": raw_plus,
        "raw_sign": raw_sign,
        "raw_change_point_sign": raw_cp_sign,
        "local_plus": local_plus,
        "local_sign": local_sign,
        "local_change_point_sign": local_cp_sign,
    }.items():
        if value.shape != expected_path_shape or not np.isfinite(value).all():
            raise RuntimeError(f"invalid reconstructed mixture trajectory: {name}")
    expected_heatmap_shape = (
        len(context.delta_nu),
        len(context.attempts),
        LOCAL_COMPONENT_COUNT,
    )
    if tile_running_max.shape != expected_heatmap_shape:
        raise RuntimeError("local-tile heatmap tensor shape changed")
    return EvidencePaths(
        plus_component=plus,
        minus_component=minus,
        sign_component=sign,
        change_point_sign_component=cp_sign,
        raw_plus=raw_plus,
        raw_sign=raw_sign,
        raw_change_point_sign=raw_cp_sign,
        local_plus=local_plus,
        local_sign=local_sign,
        local_change_point_sign=local_cp_sign,
        local_tile_running_max=tile_running_max,
    )


def load_optional_posthoc_labels(
    path: Path | None, context: ReplayContext
) -> tuple[dict[int, str | None], dict[str, Any] | None]:
    empty = {attempt: None for attempt in context.attempts}
    if path is None:
        return empty, None
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"label file is missing or not regular: {path}")
    payload = load_json(path)
    if (
        payload.get("schema_version") != 2
        or payload.get("status")
        != "posthoc_discovery_labels_not_for_confirmatory_claims"
    ):
        raise RuntimeError("optional labels are not the expected discovery-only v2 schema")
    label_input = payload.get("input", {})
    if label_input.get("class_id") != context.manifest.get("cfg", {}).get(
        "conditional_class_id"
    ):
        raise RuntimeError("optional label class does not match the raw replay")
    labels = dict(empty)
    seen: set[int] = set()
    for item in payload.get("branch_reviews", []):
        if item.get("internal_timestep") != context.rollback:
            continue
        attempt = item.get("attempt")
        if attempt not in context.attempts or attempt in seen:
            if attempt in seen:
                raise RuntimeError(f"duplicate optional label for attempt {attempt}")
            continue
        value = item.get("binary_discovery_label")
        if value not in {None, "good", "bad"}:
            raise RuntimeError(f"invalid optional binary discovery label: {value}")
        labels[int(attempt)] = value
        seen.add(int(attempt))
    if seen != set(context.attempts):
        raise RuntimeError("optional labels do not cover all four raw attempts")
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": 2,
        "status": payload["status"],
        "annotation_only": True,
        "affects_metrics_selection_order_color_limits_or_filenames": False,
        "displayed_binary_labels_by_attempt": {
            f"attempt_{attempt:03d}": labels[attempt] for attempt in context.attempts
        },
    }
    return labels, record


def _attempt_text(attempt: int, labels: dict[int, str | None]) -> str:
    text = f"attempt {attempt:03d}"
    value = labels[attempt]
    return text if value is None else f"{text} (posthoc {value})"


def _apply_axes_style(axis: plt.Axes) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=INK, labelsize=9)


def _add_research_blossom(figure: plt.Figure) -> None:
    """Small, fixed top-right research-chart mark shared by every figure."""

    center_x, center_y, radius, offset = 0.979, 0.981, 0.0042, 0.0050
    for dx, dy, color in (
        (-offset, 0.0, ATTEMPT_COLORS[0]),
        (offset, 0.0, ATTEMPT_COLORS[0]),
        (0.0, -offset, ATTEMPT_COLORS[1]),
        (0.0, offset, ATTEMPT_COLORS[1]),
    ):
        figure.add_artist(
            Circle(
                (center_x + dx, center_y + dy),
                radius,
                transform=figure.transFigure,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
                zorder=20,
            )
        )


def _save_figure(figure: plt.Figure, path: Path, identity_sha256: str) -> None:
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to overwrite figure: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=160,
        facecolor="white",
        metadata={
            "Title": path.stem,
            "Author": EXPERIMENT,
            "Subject": "posthoc discovery visualization; no quality selection",
            "Keywords": f"identity_sha256={identity_sha256}",
        },
    )
    plt.close(figure)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def render_heatmaps(
    context: ReplayContext,
    paths: EvidencePaths,
    labels: dict[int, str | None],
    output: Path,
    identity_sha256: str,
) -> None:
    values = paths.local_tile_running_max
    vmax = float(values.max(initial=0.0))
    display_vmax = vmax if vmax > 0.0 else 1.0
    figure, axes = plt.subplots(4, 3, figsize=(13.2, 14.5), squeeze=False)
    image = None
    for attempt_row, attempt in enumerate(context.attempts):
        for scale_column, delta_nu in enumerate(context.delta_nu):
            axis = axes[attempt_row, scale_column]
            matrix = values[scale_column, attempt_row].reshape(GRID_SIZE, GRID_SIZE)
            image = axis.imshow(
                matrix,
                cmap="cividis",
                vmin=0.0,
                vmax=display_vmax,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_title(
                f"{_attempt_text(attempt, labels)} | Δν={delta_nu:g}",
                fontsize=10,
                color=INK,
                pad=7,
            )
            axis.set_xticks(range(GRID_SIZE), labels=range(GRID_SIZE))
            axis.set_yticks(range(GRID_SIZE), labels=range(GRID_SIZE))
            axis.set_xlabel("latent tile column", fontsize=8, color=MUTED)
            axis.set_ylabel("latent tile row", fontsize=8, color=MUTED)
            axis.tick_params(colors=INK, labelsize=8, length=0)
            for row in range(GRID_SIZE):
                for column in range(GRID_SIZE):
                    value = float(matrix[row, column])
                    normalized = value / display_vmax
                    color = "#FFFFFF" if normalized < 0.38 or normalized > 0.80 else INK
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        color=color,
                    )
    if image is None:
        raise AssertionError("heatmap grid was not rendered")
    # Reserve a dedicated colorbar axis so it can never encroach on the fixed
    # third panel column, regardless of Matplotlib layout heuristics.
    colorbar_axis = figure.add_axes([0.915, 0.18, 0.014, 0.64])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(
        "max(0, running max +θ component log e)", color=INK, fontsize=9
    )
    colorbar.ax.tick_params(colors=INK, labelsize=8)
    figure.suptitle(
        "Local-tile +θ running-maximum log-e evidence",
        x=0.06,
        y=0.991,
        ha="left",
        fontsize=16,
        color=INK,
        fontweight="semibold",
    )
    figure.text(
        0.06,
        0.969,
        (
            f"Rollback t={context.rollback}; fixed 4×4 row-major latent tiles; one color scale "
            "across all 12 panels; initial log e=0 included."
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    figure.text(
        0.06,
        0.951,
        "Optional good/bad text is posthoc annotation only; it did not affect any visual encoding.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    _add_research_blossom(figure)
    figure.subplots_adjust(left=0.06, right=0.875, top=0.925, bottom=0.045, hspace=0.36, wspace=0.27)
    _save_figure(figure, output, identity_sha256)


def _render_mixture_figure(
    context: ReplayContext,
    labels: dict[int, str | None],
    trajectories: Sequence[np.ndarray],
    panel_titles: Sequence[str],
    *,
    title: str,
    subtitle: str,
    output: Path,
    identity_sha256: str,
) -> None:
    if len(trajectories) != 3 or len(panel_titles) != 3:
        raise ValueError("mixture figure requires exactly three trajectory panels")
    figure, axes = plt.subplots(3, 1, figsize=(12.6, 10.8), sharex=True)
    x = context.internal_timestep.astype(np.int64)
    mark_every = max(1, len(x) // 10)
    for panel_index, (axis, values, panel_title) in enumerate(
        zip(axes, trajectories, panel_titles, strict=True)
    ):
        _apply_axes_style(axis)
        axis.axhline(0.0, color=ZERO, linewidth=0.9, alpha=0.85, zorder=1)
        for branch_index, attempt in enumerate(context.attempts):
            axis.plot(
                x,
                values[branch_index],
                color=ATTEMPT_COLORS[branch_index],
                linestyle=ATTEMPT_LINESTYLES[branch_index],
                marker=ATTEMPT_MARKERS[branch_index],
                markevery=mark_every,
                markersize=3.5,
                linewidth=1.65,
                label=_attempt_text(attempt, labels),
                zorder=3,
            )
        axis.set_title(panel_title, loc="left", fontsize=11, color=INK, pad=7)
        axis.set_ylabel("log e", color=INK, fontsize=9.5)
        axis.set_xlim(context.rollback, 0)
        if panel_index == 0:
            axis.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.38),
                ncol=4,
                frameon=False,
                fontsize=8.5,
                handlelength=2.8,
            )
    axes[-1].set_xlabel(
        "source internal timestep t (each row is post-transition t→t−1; generation proceeds left→right)",
        color=INK,
        fontsize=9.5,
    )
    figure.suptitle(
        title,
        x=0.065,
        y=0.991,
        ha="left",
        fontsize=16,
        color=INK,
        fontweight="semibold",
    )
    figure.text(0.065, 0.966, subtitle, ha="left", va="top", fontsize=9.3, color=MUTED)
    figure.text(
        0.065,
        0.947,
        "Fixed priors only; no posthoc maximum over scale, tile, sign, or start time.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    _add_research_blossom(figure)
    figure.subplots_adjust(left=0.085, right=0.975, top=0.865, bottom=0.08, hspace=0.31)
    _save_figure(figure, output, identity_sha256)


def render_raw_mixture_curves(
    context: ReplayContext,
    paths: EvidencePaths,
    labels: dict[int, str | None],
    output: Path,
    identity_sha256: str,
) -> None:
    _render_mixture_figure(
        context,
        labels,
        (
            paths.raw_plus,
            paths.raw_sign,
            paths.raw_change_point_sign,
        ),
        (
            "+θ fixed scale/global+tile mixture",
            "Fixed 50/50 ±θ sign mixture",
            "Fixed-start change-point × sign mixture",
        ),
        title="Scale/global+tile fixed-mixture evidence trajectories",
        subtitle=(
            f"Rollback t={context.rollback}; uniform over {len(context.delta_nu)} scales × "
            f"(1 global + {LOCAL_COMPONENT_COUNT} local tiles) = "
            f"{len(context.delta_nu) * COMPONENT_COUNT} components; optional labels are annotation only."
        ),
        output=output,
        identity_sha256=identity_sha256,
    )


def render_local_mixture_curves(
    context: ReplayContext,
    paths: EvidencePaths,
    labels: dict[int, str | None],
    output: Path,
    identity_sha256: str,
) -> None:
    _render_mixture_figure(
        context,
        labels,
        (
            paths.local_plus,
            paths.local_sign,
            paths.local_change_point_sign,
        ),
        (
            "+θ fixed scale/local-tile-only mixture",
            "Fixed 50/50 ±θ sign mixture (local tiles only)",
            "Fixed-start change-point × sign mixture (local tiles only)",
        ),
        title="Scale/local-tile-only fixed-mixture evidence trajectories",
        subtitle=(
            f"Separate diagnostic excluding global: uniform over {len(context.delta_nu)} scales × "
            f"{LOCAL_COMPONENT_COUNT} local tiles = "
            f"{len(context.delta_nu) * LOCAL_COMPONENT_COUNT} components; not the raw saved 51-component mix."
        ),
        output=output,
        identity_sha256=identity_sha256,
    )


def render_t60_tile12_curves(
    context: ReplayContext,
    paths: EvidencePaths,
    labels: dict[int, str | None],
    output: Path,
    identity_sha256: str,
) -> None:
    if context.rollback != 60:
        raise ValueError("tile_12 posthoc figure is only defined for rollback t=60")
    names = context.arrays["component_name"].astype(str).tolist()
    if names.count("tile_12") != 1:
        raise RuntimeError("fixed tile_12 component is missing or duplicated")
    component_index = names.index("tile_12")
    values = paths.plus_component[..., component_index]  # [scale, attempt, time]
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 9.5), sharex=True, sharey=True)
    x = context.internal_timestep.astype(np.int64)
    mark_every = max(1, len(x) // 10)
    for attempt_row, (axis, attempt) in enumerate(
        zip(axes.ravel(), context.attempts, strict=True)
    ):
        _apply_axes_style(axis)
        axis.axhline(0.0, color=ZERO, linewidth=0.9, alpha=0.85, zorder=1)
        for scale_index, delta_nu in enumerate(context.delta_nu):
            axis.plot(
                x,
                values[scale_index, attempt_row],
                color=SCALE_COLORS[scale_index],
                linestyle=SCALE_LINESTYLES[scale_index],
                marker=SCALE_MARKERS[scale_index],
                markevery=mark_every,
                markersize=3.5,
                linewidth=1.65,
                label=f"Δν={delta_nu:g}",
            )
        axis.set_title(_attempt_text(attempt, labels), loc="left", fontsize=10.5, color=INK)
        axis.set_xlim(context.rollback, 0)
        axis.set_xlabel("source internal timestep t", fontsize=9, color=INK)
        axis.set_ylabel("tile_12 +θ component log e", fontsize=9, color=INK)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3)
    figure.suptitle(
        "POSTHOC DISCOVERY VIEW: fixed tile_12 +θ component trajectories",
        x=0.065,
        y=0.991,
        ha="left",
        fontsize=15.5,
        color=INK,
        fontweight="semibold",
    )
    figure.text(
        0.065,
        0.963,
        (
            "tile_12 was examined after viewing this discovery example; this panel cannot be used "
            "as a confirmatory detector or an unbiased localization result."
        ),
        ha="left",
        va="top",
        fontsize=9.3,
        color="#9A4D23",
    )
    figure.text(
        0.065,
        0.941,
        "Each row at t is evidence after transition t→t−1; generation proceeds left→right (60 to 0).",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    _add_research_blossom(figure)
    figure.subplots_adjust(left=0.085, right=0.975, top=0.885, bottom=0.08, hspace=0.27, wspace=0.17)
    _save_figure(figure, output, identity_sha256)


def render_t60_endpoint_nominal_grid(
    endpoint_context: EndpointContext,
    output: Path,
    identity_sha256: str,
) -> None:
    """Overlay the same nominal latent-to-image 4x4 grid on four endpoints."""

    if tuple(image.attempt_index for image in endpoint_context.images) != EXPECTED_ATTEMPTS:
        raise RuntimeError("endpoint figure attempt order changed")
    if len(endpoint_context.tile_bounds_nominal_image_yxyx) != LOCAL_COMPONENT_COUNT:
        raise RuntimeError("endpoint figure tile-bound count changed")
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 5.5), squeeze=False)
    axes_row = axes[0]
    grid_boundaries = (-0.5, 63.5, 127.5, 191.5, 255.5)
    for axis, endpoint in zip(axes_row, endpoint_context.images, strict=True):
        axis.imshow(endpoint.pixels, interpolation="nearest", origin="upper")
        axis.set_xlim(-0.5, 255.5)
        axis.set_ylim(255.5, -0.5)
        axis.set_aspect("equal")
        axis.set_title(
            f"attempt {endpoint.attempt_index:03d}",
            fontsize=11,
            color=INK,
            pad=8,
        )
        # Every panel receives exactly the same neutral grid and labels.
        for boundary in grid_boundaries:
            axis.axvline(boundary, color="#FFFFFF", linewidth=1.0, alpha=0.88)
            axis.axhline(boundary, color="#FFFFFF", linewidth=1.0, alpha=0.88)
        for tile_index, (y0, x0, y1, x1) in enumerate(
            endpoint_context.tile_bounds_nominal_image_yxyx
        ):
            axis.text(
                x0 + 5,
                y0 + 5,
                str(tile_index),
                ha="left",
                va="top",
                fontsize=9.5,
                color="#FFFFFF",
                path_effects=[patheffects.withStroke(linewidth=2.2, foreground="#111111")],
            )
        y0, x0, y1, x1 = endpoint_context.tile_bounds_nominal_image_yxyx[12]
        axis.add_patch(
            Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=ATTEMPT_COLORS[1],
                linewidth=4.0,
                joinstyle="miter",
                clip_on=False,
                zorder=10,
            )
        )
        axis.set_axis_off()

    figure.suptitle(
        "POSTHOC DISCOVERY MAP: tile_12 nominal latent-to-image alignment",
        x=0.055,
        y=0.988,
        ha="left",
        fontsize=15.5,
        color=INK,
        fontweight="semibold",
    )
    figure.text(
        0.055,
        0.946,
        (
            "All endpoints use the same row-major 4×4 grid (tiles 0–15); the gold outline marks "
            "tile_12 only. No quality label affects layout or styling."
        ),
        ha="left",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    figure.text(
        0.055,
        0.915,
        (
            "tile_12 was selected after this discovery example and cannot be used for confirmation. "
            "Its nominal box is image y=192:256, x=0:64; VAE receptive fields can spread influence beyond it."
        ),
        ha="left",
        va="top",
        fontsize=9.2,
        color="#9A4D23",
    )
    _add_research_blossom(figure)
    figure.subplots_adjust(left=0.045, right=0.985, top=0.82, bottom=0.055, wspace=0.10)
    _save_figure(figure, output, identity_sha256)


def _file_record(path: Path, root: Path, *, inspect_image: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if inspect_image:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.width < 1 or image.height < 1:
                raise RuntimeError(f"invalid exported PNG: {path}")
            record.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "size": [image.width, image.height],
                    "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
            )
    return record


def _series_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "final_log_e": float(values[-1]),
        "running_max_log_e_including_initial_zero": float(
            max(0.0, np.max(values, initial=-np.inf))
        ),
        "running_min_log_e_including_initial_zero": float(
            min(0.0, np.min(values, initial=np.inf))
        ),
    }


def build_numeric_results(
    context: ReplayContext,
    paths: EvidencePaths,
    endpoint_context: EndpointContext | None,
    identity_sha256: str,
) -> dict[str, Any]:
    per_attempt = []
    for branch_index, attempt in enumerate(context.attempts):
        per_attempt.append(
            {
                "attempt_index": attempt,
                "scale_global_plus_tile_51_component": {
                    "plus_theta": _series_summary(paths.raw_plus[branch_index]),
                    "sign_mixture": _series_summary(paths.raw_sign[branch_index]),
                    "change_point_sign_mixture": _series_summary(
                        paths.raw_change_point_sign[branch_index]
                    ),
                },
                "scale_local_tile_only_48_component": {
                    "plus_theta": _series_summary(paths.local_plus[branch_index]),
                    "sign_mixture": _series_summary(paths.local_sign[branch_index]),
                    "change_point_sign_mixture": _series_summary(
                        paths.local_change_point_sign[branch_index]
                    ),
                },
                "local_tile_plus_running_max_log_e_by_delta_nu": {
                    f"{delta_nu:g}": paths.local_tile_running_max[
                        scale_index, branch_index
                    ].tolist()
                    for scale_index, delta_nu in enumerate(context.delta_nu)
                },
            }
        )
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "identity_sha256": identity_sha256,
        "status": "POSTHOC_DISCOVERY_VISUALIZATION_ONLY",
        "quality_claim_eligible": False,
        "threshold_or_branch_selection_performed": False,
        "evidence_indexing": (
            "row indexed by internal timestep t is post-transition evidence for t->t-1"
        ),
        "raw_saved_plus_mixture_exactly_reconstructed": True,
        "raw_mixture_component_count": len(context.delta_nu) * COMPONENT_COUNT,
        "local_only_mixture_component_count": len(context.delta_nu)
        * LOCAL_COMPONENT_COUNT,
        "per_attempt": per_attempt,
    }
    if context.rollback == 60:
        if endpoint_context is None:
            raise RuntimeError("t60 numeric results require validated endpoint images")
        component_index = context.arrays["component_name"].astype(str).tolist().index(
            "tile_12"
        )
        output["t60_tile12_posthoc_only"] = {
            "selection_status": (
                "chosen after viewing the discovery example; not confirmatory"
            ),
            "plus_theta_component_paths": [
                {
                    "attempt_index": attempt,
                    "by_delta_nu": {
                        f"{delta_nu:g}": _series_summary(
                            paths.plus_component[scale_index, branch_index, :, component_index]
                        )
                        for scale_index, delta_nu in enumerate(context.delta_nu)
                    },
                }
                for branch_index, attempt in enumerate(context.attempts)
            ],
        }
        output["t60_endpoint_nominal_grid"] = {
            "status": "POSTHOC_DISCOVERY_SPATIAL_MAPPING_ONLY_NOT_CONFIRMATORY",
            "endpoint_file_and_pixel_hashes_strictly_validated": True,
            "attempt_order": list(context.attempts),
            "tile_order": list(range(LOCAL_COMPONENT_COUNT)),
            "tile_bounds_latent_yxyx": [
                list(item) for item in endpoint_context.tile_bounds_latent_yxyx
            ],
            "tile_bounds_nominal_image_yxyx": [
                list(item)
                for item in endpoint_context.tile_bounds_nominal_image_yxyx
            ],
            "tile_12_nominal_image_yxyx": list(
                endpoint_context.tile_bounds_nominal_image_yxyx[12]
            ),
            "mapping_caveat": (
                "nominal spatial alignment only; convolutional VAE decoder receptive fields can spread influence"
            ),
            "endpoint_records": [
                {
                    "attempt_index": image.attempt_index,
                    "path": str(image.path),
                    "file_sha256": image.record["sha256"],
                    "pixel_sha256": image.record["pixel_sha256"],
                    "mode": image.record["mode"],
                    "size": image.record["size"],
                }
                for image in endpoint_context.images
            ],
        }
    output["payload_sha256"] = _canonical_self_hash(output, "payload_sha256")
    return output


def dependency_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": importlib.metadata.version("matplotlib"),
        "pillow": importlib.metadata.version("pillow"),
    }


def build_identity(
    args: argparse.Namespace,
    context: ReplayContext,
    endpoint_context: EndpointContext | None,
    label_record: dict[str, Any] | None,
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    raw_runner = Path(__file__).with_name("replay_dit_suffix_cross_scale_diagnostics.py").resolve()
    trace_record = context.results["trace"]
    outputs = [HEATMAP_NAME, RAW_MIXTURE_NAME, LOCAL_MIXTURE_NAME]
    if context.rollback == 60:
        if endpoint_context is None:
            raise RuntimeError("t60 visualization identity requires validated endpoints")
        outputs.extend([T60_TILE12_NAME, T60_ENDPOINT_GRID_NAME])
    endpoint_record = None
    if endpoint_context is not None:
        endpoint_record = {
            "suffix_root": str(endpoint_context.suffix_root),
            "suffix_manifest_identity_sha256": (
                endpoint_context.suffix_manifest_identity_sha256
            ),
            "suffix_manifest_file_sha256": endpoint_context.suffix_manifest_file_sha256,
            "suffix_results_payload_sha256": (
                endpoint_context.suffix_results_payload_sha256
            ),
            "suffix_results_file_sha256": endpoint_context.suffix_results_file_sha256,
            "suffix_completion_file_sha256": (
                endpoint_context.suffix_completion_file_sha256
            ),
            "strict_validator_source": endpoint_context.validator_source,
            "endpoint_images": [
                {
                    "attempt_index": image.attempt_index,
                    "path": str(image.path),
                    "record": image.record,
                }
                for image in endpoint_context.images
            ],
            "tile_bounds_latent_yxyx": [
                list(item) for item in endpoint_context.tile_bounds_latent_yxyx
            ],
            "tile_bounds_nominal_image_yxyx": [
                list(item)
                for item in endpoint_context.tile_bounds_nominal_image_yxyx
            ],
            "endpoint_hashes_validated_by_original_suffix_validator_and_rechecked": True,
        }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "READ_ONLY_POSTHOC_DISCOVERY_VISUALIZATION",
        "raw_replay_bundle": {
            "root": str(context.root),
            "experiment": RAW_EXPERIMENT,
            "manifest_identity_sha256": context.manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(context.root / "manifest.json"),
            "results_payload_sha256": context.results["payload_sha256"],
            "results_file_sha256": sha256_file(context.root / "results.json"),
            "completion_file_sha256": sha256_file(context.root / "completion.json"),
            "trace_relative_path": RAW_TRACE_NAME,
            "trace_sha256": trace_record["sha256"],
            "strict_original_validator_passed_before_array_use": True,
            "raw_or_summarizer_modified": False,
        },
        "runner_source": {"path": str(runner), "sha256": sha256_file(runner)},
        "raw_validator_source": {
            "path": str(raw_runner),
            "sha256": sha256_file(raw_runner),
        },
        "validated_endpoint_source": endpoint_record,
        "target": {
            "rollback_internal_timestep": context.rollback,
            "attempts_in_fixed_raw_order": list(context.attempts),
            "delta_nu_in_fixed_raw_order": list(context.delta_nu),
            "component_names_in_fixed_raw_order": context.arrays["component_name"]
            .astype(str)
            .tolist(),
        },
        "labels": label_record,
        "labels_affect_calculation_selection_order_color_limits_or_filenames": False,
        "chart_contract": {
            "analytical_question": (
                "Where and when does predeclared cross-scale evidence accumulate across the four suffix attempts?"
            ),
            "takeaway_policy": "neutral descriptive discovery view; no quality-direction claim",
            "heatmap": (
                "4 attempt rows x 3 Delta-nu columns; every panel is a fixed 4x4 local-tile matrix; "
                "one sequential cividis color scale over all panels"
            ),
            "primary_curves": (
                "three panels; four attempts in raw order; uniform fixed mixture over 3 scales x "
                "(global + 16 tiles)"
            ),
            "secondary_curves": (
                "same three panels for a separately named 3-scale x 16-local-tile-only fixed mixture"
            ),
            "attempt_palette_and_styles_fixed_before_labels": {
                f"attempt_{attempt:03d}": {
                    "color": ATTEMPT_COLORS[index],
                    "linestyle": ATTEMPT_LINESTYLES[index],
                    "marker": ATTEMPT_MARKERS[index],
                }
                for index, attempt in enumerate(context.attempts)
            },
            "time_axis": (
                "source internal timestep t; descending t displayed left-to-right; row t is post-transition t->t-1"
            ),
            "initial_evidence": "log e=0 is included when computing running maxima",
            "fixed_sign_prior": {"plus_theta": 0.5, "minus_theta": 0.5},
            "minus_increment": "-R-K, not the negative of (R-K)",
            "fixed_change_point_prior": (
                "uniform over every stochastic suffix transition; future starts contribute E=1"
            ),
            "tile_12_t60": (
                "hardcoded posthoc discovery view selected after image inspection; never confirmatory"
            ),
            "endpoint_nominal_grid_t60": (
                "attempts 1..4 in raw order; identical row-major 4x4 neutral grid and labels; "
                "only tile_12 receives one consistent gold outline; nominal image box y=192:256,x=0:64; "
                "VAE receptive-field spillover explicitly disclosed"
            ),
        },
        "expected_figures": outputs,
        "dependencies": dependency_identity(),
        "outdir": str(args.outdir),
        "no_overwrite": True,
        "atomic_no_replace_publication": True,
    }
    payload["identity_sha256"] = _canonical_self_hash(payload, "identity_sha256")
    return payload


def _validate_raw_identity_still_bound(identity: dict[str, Any]) -> None:
    record = identity["raw_replay_bundle"]
    root = Path(record["root"])
    expected = {
        "manifest.json": record["manifest_file_sha256"],
        "results.json": record["results_file_sha256"],
        "completion.json": record["completion_file_sha256"],
        RAW_TRACE_NAME: record["trace_sha256"],
    }
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError(f"bound raw replay input changed: {path}")


def _validate_endpoint_identity_still_bound(identity: dict[str, Any]) -> None:
    record = identity.get("validated_endpoint_source")
    if record is None:
        return
    root = Path(record["suffix_root"])
    expected_files = {
        "manifest.json": record["suffix_manifest_file_sha256"],
        "results.json": record["suffix_results_file_sha256"],
        "completion.json": record["suffix_completion_file_sha256"],
    }
    for relative, digest in expected_files.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError(f"bound suffix endpoint source changed: {path}")
    validator = record.get("strict_validator_source", {})
    validator_path = Path(str(validator.get("path", "")))
    if (
        not validator_path.is_file()
        or validator_path.is_symlink()
        or sha256_file(validator_path) != validator.get("sha256")
    ):
        raise RuntimeError("strict endpoint validator source changed")
    for image_record in record.get("endpoint_images", []):
        path = Path(image_record["path"])
        expected = image_record["record"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected.get("bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise RuntimeError(f"bound endpoint PNG changed: {path}")
        with Image.open(path) as image:
            image.load()
            pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            if (
                image.format != "PNG"
                or image.mode != expected.get("mode")
                or list(image.size) != expected.get("size")
                or hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
                != expected.get("pixel_sha256")
            ):
                raise RuntimeError(f"bound endpoint PNG pixel identity changed: {path}")


def validate_completed_output(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"visualization output is not a plain directory: {root}")
    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "payload_sha256")
    results = _read_self_hashed_json(root / RESULTS_NAME, "payload_sha256")
    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity
    ):
        raise RuntimeError("visualization manifest identity/status changed")
    runner = Path(__file__).resolve()
    if identity.get("runner_source") != {
        "path": str(runner),
        "sha256": sha256_file(runner),
    }:
        raise RuntimeError("visualization was produced by a different runner source")
    _validate_raw_identity_still_bound(identity)
    _validate_endpoint_identity_still_bound(identity)
    if (
        results.get("experiment") != EXPERIMENT
        or results.get("identity_sha256") != identity["identity_sha256"]
        or results.get("quality_claim_eligible") is not False
    ):
        raise RuntimeError("visualization numeric results scope/identity changed")
    expected_figures = identity["expected_figures"]
    actual_records = [
        _file_record(root / relative, root, inspect_image=True)
        for relative in expected_figures
    ]
    if manifest.get("figures") != actual_records:
        raise RuntimeError("visualization figure file/pixel hashes changed")
    result_record = _file_record(root / RESULTS_NAME, root, inspect_image=False)
    if manifest.get("results") != result_record:
        raise RuntimeError("visualization result-file hash changed")
    outputs_sha256 = sha256_json([result_record, *actual_records])
    if manifest.get("outputs_sha256") != outputs_sha256:
        raise RuntimeError("visualization aggregate output hash changed")
    expected_files = {
        (root / relative).resolve()
        for relative in [
            *expected_figures,
            RESULTS_NAME,
            MANIFEST_NAME,
            COMPLETION_NAME,
        ]
    }
    actual_files: set[Path] = set()
    actual_dirs: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"visualization output contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.resolve())
        elif path.is_dir():
            actual_dirs.add(path.resolve())
        else:
            raise RuntimeError(f"visualization output contains a special entry: {path}")
    if actual_files != expected_files or actual_dirs:
        raise RuntimeError("visualization output tree is not closed")
    fixed_completion = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "identity_sha256": identity["identity_sha256"],
        "results_payload_sha256": results["payload_sha256"],
        "results_file_sha256": sha256_file(root / RESULTS_NAME),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
        "outputs_sha256": outputs_sha256,
        "figure_count": len(expected_figures),
    }
    mismatches = {
        key: (completion.get(key), value)
        for key, value in fixed_completion.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"visualization completion links/hashes changed: {mismatches}")
    return manifest


def run_real(
    args: argparse.Namespace,
    context: ReplayContext,
    paths: EvidencePaths,
    endpoint_context: EndpointContext | None,
    labels: dict[int, str | None],
    identity: dict[str, Any],
) -> None:
    outdir = args.outdir
    if os.path.lexists(outdir):
        raise RuntimeError(f"refusing to overwrite output path: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        identity_sha256 = identity["identity_sha256"]
        render_heatmaps(
            context, paths, labels, staging / HEATMAP_NAME, identity_sha256
        )
        render_raw_mixture_curves(
            context, paths, labels, staging / RAW_MIXTURE_NAME, identity_sha256
        )
        render_local_mixture_curves(
            context, paths, labels, staging / LOCAL_MIXTURE_NAME, identity_sha256
        )
        if context.rollback == 60:
            if endpoint_context is None:
                raise RuntimeError("t60 rendering requires validated endpoints")
            render_t60_tile12_curves(
                context, paths, labels, staging / T60_TILE12_NAME, identity_sha256
            )
            render_t60_endpoint_nominal_grid(
                endpoint_context,
                staging / T60_ENDPOINT_GRID_NAME,
                identity_sha256,
            )

        numeric_results = build_numeric_results(
            context, paths, endpoint_context, identity_sha256
        )
        atomic_json_dump(numeric_results, staging / RESULTS_NAME)
        figure_records = [
            _file_record(staging / relative, staging, inspect_image=True)
            for relative in identity["expected_figures"]
        ]
        result_record = _file_record(
            staging / RESULTS_NAME, staging, inspect_image=False
        )
        outputs_sha256 = sha256_json([result_record, *figure_records])
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "identity": identity,
            "results": result_record,
            "figures": figure_records,
            "outputs_sha256": outputs_sha256,
        }
        manifest["payload_sha256"] = _canonical_self_hash(manifest, "payload_sha256")
        atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "identity_sha256": identity_sha256,
            "results_payload_sha256": numeric_results["payload_sha256"],
            "results_file_sha256": sha256_file(staging / RESULTS_NAME),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "manifest_file_sha256": sha256_file(staging / MANIFEST_NAME),
            "outputs_sha256": outputs_sha256,
            "figure_count": len(figure_records),
        }
        completion["payload_sha256"] = _canonical_self_hash(
            completion, "payload_sha256"
        )
        atomic_json_dump(completion, staging / COMPLETION_NAME)
        # Fail closed on both the output and the immutable raw source before
        # atomic publication.
        validate_completed_output(staging, identity)
        manifest_again, results_again = validate_raw_bundle(context.root)
        if manifest_again != context.manifest or results_again != context.results:
            raise RuntimeError("raw replay changed while figures were being rendered")
        if endpoint_context is not None:
            endpoint_again = validate_and_load_t60_endpoints(context)
            if endpoint_again is None:
                raise RuntimeError("strict endpoint revalidation unexpectedly disappeared")
            if [image.record for image in endpoint_again.images] != [
                image.record for image in endpoint_context.images
            ]:
                raise RuntimeError("endpoint identities changed while figures were rendered")
        if os.path.lexists(outdir):
            raise RuntimeError("output appeared during staging; refusing overwrite")
        _atomic_install_directory_noreplace(staging, outdir)
    validate_completed_output(outdir, identity)
    print(
        json.dumps(
            {
                "complete": True,
                "outdir": str(outdir),
                "identity_sha256": identity["identity_sha256"],
                "rollback_internal_timestep": context.rollback,
                "attempts": list(context.attempts),
                "delta_nu": list(context.delta_nu),
                "figures": identity["expected_figures"],
                "strict_raw_validation_passed": True,
                "raw_or_summarizer_modified": False,
                "quality_claim_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_self_test() -> None:
    increments = np.asarray([[0.2, -0.1, 0.3]], dtype=np.float64)
    observed = uniform_change_point_log_mixture(increments, start_count=3)[0]
    expected = np.asarray(
        [
            math.log((math.exp(0.2) + 2.0) / 3.0),
            math.log((math.exp(0.1) + math.exp(-0.1) + 1.0) / 3.0),
            math.log((math.exp(0.4) + math.exp(0.2) + math.exp(0.3)) / 3.0),
        ]
    )
    if not np.allclose(observed, expected, rtol=0.0, atol=2e-16):
        raise AssertionError("fixed-start change-point mixture is wrong")
    reward = np.asarray([0.5, -0.25], dtype=np.float64)
    cost = np.asarray([0.125, 0.0625], dtype=np.float64)
    plus_increment = reward - cost
    minus_increment = -reward - cost
    if np.array_equal(minus_increment, -plus_increment):
        raise AssertionError("negative-theta LR incorrectly used -L_plus")
    sign = fixed_sign_log_mixture(
        np.cumsum(plus_increment), np.cumsum(minus_increment)
    )
    manual = np.log(
        (
            np.exp(np.cumsum(plus_increment))
            + np.exp(np.cumsum(minus_increment))
        )
        / 2.0
    )
    if not np.allclose(sign, manual, rtol=0.0, atol=2e-16):
        raise AssertionError("fixed sign mixture is wrong")
    synthetic = np.asarray([-0.2, 0.1, 0.8, 0.3])
    if float(max(0.0, synthetic.max())) != 0.8:
        raise AssertionError("running maximum must include initial log e=0")
    print("self-test passed: sign/change-point math and running-max convention")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional v2 posthoc discovery labels; annotations only.",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        return
    if args.bundle_dir is None:
        parser.error("--bundle-dir is required")
    if args.outdir is None:
        parser.error("--outdir is required")
    args.bundle_dir = args.bundle_dir.expanduser().absolute().resolve()
    args.outdir = args.outdir.expanduser().absolute()
    if args.labels is not None:
        args.labels = args.labels.expanduser().absolute().resolve()
    if _paths_overlap(args.bundle_dir, args.outdir):
        parser.error("--outdir must not overlap the read-only raw replay bundle")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    normalize_args(args, parser)
    if args.self_test:
        run_self_test()
        return
    context = validate_and_load_raw_bundle(args.bundle_dir)
    # Evidence construction is deliberately completed before optional labels
    # are loaded, making their annotation-only role explicit in control flow.
    paths = reconstruct_evidence(context)
    endpoint_context = validate_and_load_t60_endpoints(context)
    labels, label_record = load_optional_posthoc_labels(args.labels, context)
    identity = build_identity(args, context, endpoint_context, label_record)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "raw_bundle": str(context.root),
                    "rollback_internal_timestep": context.rollback,
                    "attempts": list(context.attempts),
                    "delta_nu": list(context.delta_nu),
                    "strict_raw_validation_passed": True,
                    "all_evidence_reconstructed": True,
                    "endpoint_file_and_pixel_hashes_strictly_validated": (
                        endpoint_context is not None
                    ),
                    "labels_loaded_after_evidence_construction": True,
                    "expected_figures": identity["expected_figures"],
                    "outdir": str(args.outdir),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_real(args, context, paths, endpoint_context, labels, identity)


if __name__ == "__main__":
    main()
