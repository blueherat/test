#!/usr/bin/env python3
"""Strict CPU-only summary of completed DiT path-evidence observers.

Every input must be a completed ``observe_dit_imagenet256_path_evidence.py``
bundle.  This program calls that observer's own fail-closed bundle validator
before reading a trace for analysis.  It then reports, for every seed/image,
the final and running-maximum fixed-tile-mixture log evidence, the internal
timestep of the observed path maximum, and the most supported fixed tile.

The frozen exploratory class-207/seed-2 candidate is compared only with the
five same-class anchors declared before path evidence was inspected.  Ranks
are descriptive ranks inside those six samples, not a formal statistical
test.  No image-quality model or automatic bad-case score is used here.

The output contains self-hashed JSON, summary and long-form curve CSV files,
and two readable PNG figures.  Inputs are read-only, the output is staged and
atomically published, and an existing output path is never overwritten.  No
model, checkpoint, VAE, CUDA tensor, or GPU process is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA_VERSION = 1
EXPERIMENT = "dit_imagenet256_path_evidence_multirun_summary"
EXPECTED_INPUT_EXPERIMENT = "dit_imagenet256_local_cross_scale_path_evidence_observe_only"
EXPECTED_PRESELECTION_SHA256 = (
    "9505f26ba7b9264dd51bc7513db385250d2bd375100a5479a4f58e2b4c833207"
)
EXPECTED_CLASS_IDS = (207, 360, 387, 974, 88, 979, 417, 279)
EXPECTED_CANDIDATE_SEED = 2
EXPECTED_CANDIDATE_BATCH_INDEX = 0
EXPECTED_CANDIDATE_CLASS_ID = 207
EXPECTED_CONTROL_SEEDS = (1, 3, 4, 7, 9)
EXPECTED_STEPS = 250
EXPECTED_TILES = 16
LATENT_TO_PIXEL_SCALE = 8

SUMMARY_CSV_NAME = "sample_summary.csv"
CURVE_CSV_NAME = "sample_mixture_curves.csv"
ALL_CURVES_PNG_NAME = "sample_mixture_curves.png"
CLASS207_PNG_NAME = "class0207_seed2_vs_controls.png"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"
COMPLETION_NAME = "completion.json"

CANDIDATE_COLOR = "#E17C05"
CONTROL_COLORS = {
    1: "#4C78A8",
    3: "#72B7B2",
    4: "#54A24B",
    7: "#B279A2",
    9: "#9D755D",
}
OTHER_COLOR = "#A0A0A0"


@dataclass(frozen=True)
class FrozenComparison:
    path: Path
    sha256: str
    payload: dict[str, Any]
    candidate_seed: int
    candidate_batch_index: int
    candidate_class_id: int
    control_seeds: tuple[int, ...]

    @property
    def required_seeds(self) -> tuple[int, ...]:
        return tuple(sorted((self.candidate_seed, *self.control_seeds)))


@dataclass(frozen=True)
class ValidatedRun:
    root: Path
    seed: int
    manifest: dict[str, Any]
    results: dict[str, Any]
    arrays: dict[str, np.ndarray]
    baseline_root: Path
    baseline_identity_sha256: str


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    copied = dict(payload)
    copied.pop(key, None)
    return sha256_json(copied)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected one JSON object: {path}")
    return payload


def read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = read_json(path)
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def atomic_json_dump(payload: Any, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite JSON: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv_dump(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite CSV: {path}")
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows do not share one stable field order")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_figure_dump(figure: plt.Figure, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite PNG: {path}")
    temporary = path.with_name(path.name + ".tmp")
    figure.savefig(
        temporary,
        format="png",
        dpi=180,
        facecolor="white",
        metadata={"Software": Path(__file__).name},
    )
    plt.close(figure)
    with temporary.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"matplotlib did not produce a PNG: {temporary}")
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stable_logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values64).all():
        raise ValueError("log-mean-exp input contains a non-finite value")
    maximum = np.max(values64, axis=axis, keepdims=True)
    centered = np.exp(values64 - maximum)
    result = maximum + np.log(
        np.mean(centered, axis=axis, keepdims=True, dtype=np.float64)
    )
    return np.squeeze(result, axis=axis)


def load_frozen_comparison(path: Path) -> FrozenComparison:
    path = path.expanduser().absolute().resolve()
    digest = sha256_file(path)
    if digest != EXPECTED_PRESELECTION_SHA256:
        raise RuntimeError(
            "DiT candidate preselection is not the frozen v1 file: "
            f"{digest} != {EXPECTED_PRESELECTION_SHA256}"
        )
    payload = read_json(path)
    expected_top = {
        "schema_version": 1,
        "status": "exploratory_single_reviewer_preselection_not_a_formal_bad_label",
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected_top.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen preselection scope changed: {mismatches}")
    constraints = payload.get("selection_constraints", {})
    if constraints.get("path_evidence_seen") is not False:
        raise RuntimeError("candidate must have been frozen before path evidence")
    if constraints.get("formal_endpoint") is not False or constraints.get("reviewers") != 1:
        raise RuntimeError("candidate must remain a single-reviewer exploratory endpoint")
    candidate = payload.get("candidate", {})
    identity = (
        candidate.get("global_seed"),
        candidate.get("batch_index"),
        candidate.get("class_id"),
    )
    expected_identity = (
        EXPECTED_CANDIDATE_SEED,
        EXPECTED_CANDIDATE_BATCH_INDEX,
        EXPECTED_CANDIDATE_CLASS_ID,
    )
    if identity != expected_identity:
        raise RuntimeError(
            f"frozen candidate identity changed: {identity} != {expected_identity}"
        )
    anchors = payload.get("provisional_same_class_typical_anchors", [])
    controls = tuple(sorted(int(anchor["global_seed"]) for anchor in anchors))
    if controls != EXPECTED_CONTROL_SEEDS or len(anchors) != len(controls):
        raise RuntimeError(
            f"frozen same-class control seeds changed: {controls} != {EXPECTED_CONTROL_SEEDS}"
        )
    for anchor in anchors:
        expected_path = (
            f"official_demo_seed{int(anchor['global_seed'])}/images/"
            "00_class0207.png"
        )
        if anchor.get("relative_path") != expected_path:
            raise RuntimeError(f"frozen anchor path changed: {anchor}")
    return FrozenComparison(
        path=path,
        sha256=digest,
        payload=payload,
        candidate_seed=EXPECTED_CANDIDATE_SEED,
        candidate_batch_index=EXPECTED_CANDIDATE_BATCH_INDEX,
        candidate_class_id=EXPECTED_CANDIDATE_CLASS_ID,
        control_seeds=controls,
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def validate_input_runs(
    run_dirs: Sequence[Path], *, dit_root: Path, checkpoint: Path
) -> tuple[list[ValidatedRun], dict[str, Any]]:
    """Call the observer's full validator, then return its exact trace arrays."""

    # Lazy import keeps --self-test independent of torch and all model code.
    try:
        from . import observe_dit_imagenet256_path_evidence as observer
    except ImportError:  # pragma: no cover - direct CLI execution.
        import observe_dit_imagenet256_path_evidence as observer

    source = observer.validate_repository(dit_root, checkpoint)
    alpha_bar, timestep_map = observer.load_schedule(dit_root)
    spec = observer.build_evidence_spec(alpha_bar, timestep_map)
    validated: list[ValidatedRun] = []
    seeds_seen: set[int] = set()
    for raw_root in run_dirs:
        root = raw_root.expanduser().absolute().resolve()
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"observer bundle is not a regular directory: {root}")
        probe = read_json(root / MANIFEST_NAME)
        if probe.get("experiment") != EXPECTED_INPUT_EXPERIMENT:
            raise RuntimeError(f"wrong observer experiment at {root}")
        seed = int(probe.get("seed"))
        if seed in seeds_seen:
            raise RuntimeError(f"duplicate observer seed: {seed}")
        seeds_seen.add(seed)
        baseline_record = probe.get("frozen_baseline", {})
        baseline_root_value = baseline_record.get("root")
        if not isinstance(baseline_root_value, str):
            raise RuntimeError(f"observer manifest lacks its frozen baseline root: {root}")
        baseline_root = Path(baseline_root_value).expanduser().absolute().resolve()
        baseline = observer.validate_baseline_run(baseline_root, seed=seed)

        # Required strict validation: this reconstructs trace math, all hashes,
        # the 250-transition state/RNG chain, PNG equality, and completion links.
        results = observer.validate_output_bundle(root, baseline=baseline, spec=spec)
        manifest = observer._read_self_hashed_json(  # noqa: SLF001 - same-repo validator API.
            root / MANIFEST_NAME, "identity_sha256"
        )
        arrays = observer._load_trace_exact(  # noqa: SLF001 - avoid a post-validation TOCTOU gap.
            root / observer.TRACE_NAME, results["trace"], root
        )
        if tuple(manifest.get("class_ids_in_official_batch_order", ())) != EXPECTED_CLASS_IDS:
            raise RuntimeError(f"official batch class order changed at {root}")
        validated.append(
            ValidatedRun(
                root=root,
                seed=seed,
                manifest=manifest,
                results=results,
                arrays=arrays,
                baseline_root=baseline.root,
                baseline_identity_sha256=baseline.identity_sha256,
            )
        )
    return sorted(validated, key=lambda item: item.seed), source


def comparison_role(
    seed: int, batch_index: int, class_id: int, frozen: FrozenComparison
) -> str:
    if class_id != frozen.candidate_class_id or batch_index != frozen.candidate_batch_index:
        return "other_official_batch_class_not_quality_labeled"
    if seed == frozen.candidate_seed:
        return "frozen_exploratory_candidate"
    if seed in frozen.control_seeds:
        return "frozen_provisional_same_class_anchor"
    return "class207_context_not_in_frozen_comparison"


def tile_record(bounds: np.ndarray, tile_index: int) -> dict[str, Any]:
    latent = [int(value) for value in bounds[tile_index].tolist()]
    pixels = [int(value) * LATENT_TO_PIXEL_SCALE for value in latent]
    return {
        "tile_index": int(tile_index),
        "tile_row": int(tile_index // 4),
        "tile_column": int(tile_index % 4),
        "bounds_latent_yxyx_half_open": latent,
        "bounds_decoded_pixel_yxyx_half_open": pixels,
    }


def derive_run_records(
    run: ValidatedRun | Any, frozen: FrozenComparison
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    arrays = run.arrays
    internal_timestep = np.asarray(arrays["internal_timestep"], dtype=np.int64)
    sample_mix = np.asarray(arrays["sample_mixture_log_e"], dtype=np.float64)
    components = np.asarray(arrays["component_log_e"], dtype=np.float64)
    bounds = np.asarray(arrays["tile_bounds_yxyx"], dtype=np.int64)
    expected_shape = (EXPECTED_STEPS, len(EXPECTED_CLASS_IDS))
    if sample_mix.shape != expected_shape:
        raise RuntimeError(f"sample mixture shape changed for seed {run.seed}: {sample_mix.shape}")
    if components.shape != (*expected_shape, EXPECTED_TILES):
        raise RuntimeError(f"component evidence shape changed for seed {run.seed}")
    if internal_timestep.tolist() != list(range(249, -1, -1)):
        raise RuntimeError(f"internal timestep order changed for seed {run.seed}")
    if bounds.shape != (EXPECTED_TILES, 4):
        raise RuntimeError(f"tile bounds shape changed for seed {run.seed}")
    reconstructed = stable_logmeanexp(components, axis=2)
    if not np.allclose(reconstructed, sample_mix, rtol=0.0, atol=2e-13):
        raise RuntimeError(f"sample mixture does not reconstruct from components for seed {run.seed}")

    summaries: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    curve_json: list[dict[str, Any]] = []
    image_paths = [
        f"images/{index:02d}_class{class_id:04d}.png"
        for index, class_id in enumerate(EXPECTED_CLASS_IDS)
    ]
    for batch_index, (class_id, image_path) in enumerate(
        zip(EXPECTED_CLASS_IDS, image_paths)
    ):
        curve = sample_mix[:, batch_index]
        component_curve = components[:, batch_index, :]
        post_index = int(np.argmax(curve))
        post_max = float(curve[post_index])
        running_curve = np.maximum.accumulate(
            np.concatenate((np.asarray([0.0]), curve))
        )[1:]
        if post_max > 0.0:
            anytime_max = post_max
            anytime_index: int | None = post_index
            anytime_timestep: int | None = int(internal_timestep[post_index])
            anytime_source = "observed_transition"
        else:
            anytime_max = 0.0
            anytime_index = None
            anytime_timestep = None
            anytime_source = "initial_E0_equals_1"

        top_indices = np.argmax(component_curve, axis=1).astype(np.int64)
        top_values = component_curve[np.arange(EXPECTED_STEPS), top_indices]
        posterior = np.exp(
            top_values - (curve + math.log(EXPECTED_TILES))
        )
        if np.any(posterior < 1.0 / EXPECTED_TILES - 2e-15) or np.any(
            posterior > 1.0 + 2e-15
        ):
            raise RuntimeError(f"invalid tile posterior for seed/class {run.seed}/{class_id}")
        final_tile = tile_record(bounds, int(top_indices[-1]))
        post_tile = tile_record(bounds, int(top_indices[post_index]))
        role = comparison_role(run.seed, batch_index, class_id, frozen)
        summary = {
            "seed": int(run.seed),
            "batch_index": int(batch_index),
            "class_id": int(class_id),
            "image_relative_path": image_path,
            "observer_bundle": str(run.root),
            "frozen_comparison_role": role,
            "automatic_image_quality_score": None,
            "final_sample_mixture_log_e": float(curve[-1]),
            "running_max_sample_mixture_log_e_from_E0": float(anytime_max),
            "running_max_source": anytime_source,
            "running_max_reverse_step_index": anytime_index,
            "running_max_internal_timestep": anytime_timestep,
            "post_transition_max_sample_mixture_log_e": post_max,
            "post_transition_max_reverse_step_index": post_index,
            "post_transition_max_internal_timestep": int(internal_timestep[post_index]),
            "top_tile_at_post_transition_max": post_tile,
            "top_tile_component_log_e_at_post_transition_max": float(
                top_values[post_index]
            ),
            "top_tile_posterior_probability_at_post_transition_max": float(
                posterior[post_index]
            ),
            "top_tile_at_final_transition": final_tile,
            "top_tile_component_log_e_at_final_transition": float(top_values[-1]),
            "top_tile_posterior_probability_at_final_transition": float(posterior[-1]),
        }
        summaries.append(summary)

        update_flags = np.empty(EXPECTED_STEPS, dtype=bool)
        previous = 0.0
        for row_index in range(EXPECTED_STEPS):
            update_flags[row_index] = bool(curve[row_index] > previous)
            previous = max(previous, float(curve[row_index]))
        for row_index in range(EXPECTED_STEPS):
            per_step_tile = tile_record(bounds, int(top_indices[row_index]))
            curve_rows.append(
                {
                    "seed": int(run.seed),
                    "batch_index": int(batch_index),
                    "class_id": int(class_id),
                    "image_relative_path": image_path,
                    "frozen_comparison_role": role,
                    "reverse_step_index": int(row_index),
                    "internal_timestep": int(internal_timestep[row_index]),
                    "sample_mixture_log_e": float(curve[row_index]),
                    "running_max_sample_mixture_log_e_from_E0": float(
                        running_curve[row_index]
                    ),
                    "is_strict_running_max_update": bool(update_flags[row_index]),
                    "top_tile_index": per_step_tile["tile_index"],
                    "top_tile_row": per_step_tile["tile_row"],
                    "top_tile_column": per_step_tile["tile_column"],
                    "top_tile_bounds_latent_yxyx": json.dumps(
                        per_step_tile["bounds_latent_yxyx_half_open"],
                        separators=(",", ":"),
                    ),
                    "top_tile_component_log_e": float(top_values[row_index]),
                    "top_tile_posterior_probability": float(posterior[row_index]),
                }
            )
        curve_json.append(
            {
                "seed": int(run.seed),
                "batch_index": int(batch_index),
                "class_id": int(class_id),
                "image_relative_path": image_path,
                "frozen_comparison_role": role,
                "reverse_step_index": list(range(EXPECTED_STEPS)),
                "internal_timestep": internal_timestep.astype(int).tolist(),
                "sample_mixture_log_e": curve.tolist(),
                "running_max_sample_mixture_log_e_from_E0": running_curve.tolist(),
                "top_tile_index": top_indices.astype(int).tolist(),
                "top_tile_component_log_e": top_values.tolist(),
                "top_tile_posterior_probability": posterior.tolist(),
            }
        )
    return summaries, curve_rows, curve_json


def descending_rank(
    records: Sequence[dict[str, Any]], *, candidate_seed: int, metric: str
) -> dict[str, Any]:
    candidate = next(record for record in records if int(record["seed"]) == candidate_seed)
    candidate_value = float(candidate[metric])
    values = [(int(record["seed"]), float(record[metric])) for record in records]
    greater = sum(value > candidate_value for _, value in values)
    ties = sum(value == candidate_value for _, value in values)
    controls = [
        {"seed": seed, "value": value}
        for seed, value in values
        if seed != candidate_seed
    ]
    control_values = [item["value"] for item in controls]
    return {
        "metric": metric,
        "direction": "larger_log_e_means_more_operational_Q_over_P_evidence",
        "candidate_seed": candidate_seed,
        "candidate_value": candidate_value,
        "controls": controls,
        "candidate_descending_competition_rank_among_six": 1 + greater,
        "comparison_sample_count": len(values),
        "exact_tie_count_including_candidate": ties,
        "candidate_minus_control_mean": candidate_value
        - float(np.mean(control_values, dtype=np.float64)),
        "candidate_greater_than_every_control": bool(
            candidate_value > max(control_values)
        ),
    }


def build_class207_comparison(
    summaries: Sequence[dict[str, Any]], frozen: FrozenComparison
) -> dict[str, Any]:
    required = set(frozen.required_seeds)
    records = [
        record
        for record in summaries
        if int(record["class_id"]) == frozen.candidate_class_id
        and int(record["batch_index"]) == frozen.candidate_batch_index
        and int(record["seed"]) in required
    ]
    observed = {int(record["seed"]) for record in records}
    if observed != required or len(records) != len(required):
        raise RuntimeError(
            "class207 seed2 comparison requires exactly the frozen candidate and five "
            f"anchors; observed={sorted(observed)}, required={sorted(required)}"
        )
    records = sorted(records, key=lambda item: int(item["seed"]))
    metrics = (
        "running_max_sample_mixture_log_e_from_E0",
        "post_transition_max_sample_mixture_log_e",
        "final_sample_mixture_log_e",
    )
    return {
        "status": "descriptive_exploratory_rank_not_a_formal_statistical_test",
        "quality_labels": "frozen_single_reviewer_candidate_and_provisional_typical_anchors_only",
        "candidate": {
            "seed": frozen.candidate_seed,
            "batch_index": frozen.candidate_batch_index,
            "class_id": frozen.candidate_class_id,
        },
        "control_seeds": list(frozen.control_seeds),
        "ranks": [
            descending_rank(records, candidate_seed=frozen.candidate_seed, metric=metric)
            for metric in metrics
        ],
    }


def summary_csv_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in summaries:
        post_tile = item["top_tile_at_post_transition_max"]
        final_tile = item["top_tile_at_final_transition"]
        rows.append(
            {
                "seed": item["seed"],
                "batch_index": item["batch_index"],
                "class_id": item["class_id"],
                "image_relative_path": item["image_relative_path"],
                "frozen_comparison_role": item["frozen_comparison_role"],
                "automatic_image_quality_score": "",
                "final_sample_mixture_log_e": item["final_sample_mixture_log_e"],
                "running_max_sample_mixture_log_e_from_E0": item[
                    "running_max_sample_mixture_log_e_from_E0"
                ],
                "running_max_source": item["running_max_source"],
                "running_max_reverse_step_index": ""
                if item["running_max_reverse_step_index"] is None
                else item["running_max_reverse_step_index"],
                "running_max_internal_timestep": ""
                if item["running_max_internal_timestep"] is None
                else item["running_max_internal_timestep"],
                "post_transition_max_sample_mixture_log_e": item[
                    "post_transition_max_sample_mixture_log_e"
                ],
                "post_transition_max_reverse_step_index": item[
                    "post_transition_max_reverse_step_index"
                ],
                "post_transition_max_internal_timestep": item[
                    "post_transition_max_internal_timestep"
                ],
                "top_tile_index_at_post_transition_max": post_tile["tile_index"],
                "top_tile_row_at_post_transition_max": post_tile["tile_row"],
                "top_tile_column_at_post_transition_max": post_tile["tile_column"],
                "top_tile_bounds_latent_yxyx_at_post_transition_max": json.dumps(
                    post_tile["bounds_latent_yxyx_half_open"], separators=(",", ":")
                ),
                "top_tile_posterior_at_post_transition_max": item[
                    "top_tile_posterior_probability_at_post_transition_max"
                ],
                "top_tile_index_at_final_transition": final_tile["tile_index"],
                "top_tile_bounds_latent_yxyx_at_final_transition": json.dumps(
                    final_tile["bounds_latent_yxyx_half_open"], separators=(",", ":")
                ),
                "top_tile_posterior_at_final_transition": item[
                    "top_tile_posterior_probability_at_final_transition"
                ],
            }
        )
    return rows


def _curve_lookup(curves: Sequence[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    result = {}
    for curve in curves:
        key = (int(curve["seed"]), int(curve["class_id"]))
        if key in result:
            raise RuntimeError(f"duplicate curve key: {key}")
        result[key] = curve
    return result


def render_all_curves(
    curves: Sequence[dict[str, Any]], seeds: Sequence[int], output: Path
) -> None:
    lookup = _curve_lookup(curves)
    figure, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axes_flat = axes.ravel()
    palette = plt.get_cmap("tab10")
    seed_colors = {seed: palette(index % 10) for index, seed in enumerate(seeds)}
    for axis, class_id in zip(axes_flat, EXPECTED_CLASS_IDS):
        for seed in seeds:
            curve = lookup[(seed, class_id)]
            x = np.asarray(curve["internal_timestep"], dtype=np.int64)
            y = np.asarray(curve["sample_mixture_log_e"], dtype=np.float64)
            if class_id == EXPECTED_CANDIDATE_CLASS_ID and seed == EXPECTED_CANDIDATE_SEED:
                color, width, alpha, zorder = CANDIDATE_COLOR, 2.7, 1.0, 5
                label = "seed 2 frozen candidate"
            else:
                color, width, alpha, zorder = seed_colors[seed], 1.25, 0.78, 2
                label = f"seed {seed}"
            axis.plot(x, y, color=color, linewidth=width, alpha=alpha, label=label, zorder=zorder)
        axis.axhline(0.0, color="#303030", linewidth=0.8, linestyle="--")
        axis.set_title(f"class {class_id}", loc="left", fontsize=11)
        axis.set_xlim(249, 0)
        axis.set_ylabel("sample-mixture log E")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes_flat[-2].set_xlabel("internal timestep (noisiest to cleanest)")
    axes_flat[-1].set_xlabel("internal timestep (noisiest to cleanest)")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(6, len(seeds)), frameon=False)
    figure.suptitle(
        "DiT-XL/2 fixed-tile sample-mixture operational path evidence",
        fontsize=14,
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "Observe-only curves; the orange label was frozen before evidence inspection and is not an automatic quality score.",
        ha="center",
        fontsize=9,
        color="#404040",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.965))
    atomic_figure_dump(figure, output)


def render_class207_comparison(
    curves: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    comparison: dict[str, Any],
    frozen: FrozenComparison,
    output: Path,
) -> None:
    lookup = _curve_lookup(curves)
    by_seed = {
        int(item["seed"]): item
        for item in summaries
        if int(item["class_id"]) == frozen.candidate_class_id
        and int(item["seed"]) in set(frozen.required_seeds)
    }
    figure, axis = plt.subplots(1, 1, figsize=(11, 6.4))
    for seed in frozen.required_seeds:
        curve = lookup[(seed, frozen.candidate_class_id)]
        x = np.asarray(curve["internal_timestep"], dtype=np.int64)
        y = np.asarray(curve["sample_mixture_log_e"], dtype=np.float64)
        summary = by_seed[seed]
        peak_index = int(summary["post_transition_max_reverse_step_index"])
        if seed == frozen.candidate_seed:
            color, width, zorder = CANDIDATE_COLOR, 3.0, 5
            label = "seed 2 frozen candidate"
        else:
            color, width, zorder = CONTROL_COLORS[seed], 1.8, 3
            label = f"seed {seed} frozen anchor"
        axis.plot(x, y, color=color, linewidth=width, label=label, zorder=zorder)
        axis.scatter(
            [x[peak_index]], [y[peak_index]], s=34, color=color, edgecolor="white", zorder=zorder + 1
        )
    rank = next(
        item
        for item in comparison["ranks"]
        if item["metric"] == "post_transition_max_sample_mixture_log_e"
    )
    axis.axhline(0.0, color="#303030", linewidth=0.9, linestyle="--", label="initial log E = 0")
    axis.set_xlim(249, 0)
    axis.set_xlabel("internal timestep (noisiest to cleanest)")
    axis.set_ylabel("sample-mixture log E")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
    axis.legend(loc="best", frameon=False, ncol=2)
    axis.set_title(
        "Class 207: seed 2 versus five frozen same-class anchors\n"
        f"seed 2 descriptive rank by observed path maximum: "
        f"{rank['candidate_descending_competition_rank_among_six']}/6",
        loc="left",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.012,
        "Larger means more evidence for the implemented cross-scale Q over P; this is exploratory, not a quality classifier or formal endpoint.",
        ha="center",
        fontsize=9,
        color="#404040",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    atomic_figure_dump(figure, output)


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def input_record(run: ValidatedRun) -> dict[str, Any]:
    trace = run.root / run.results["trace"]["relative_path"]
    return {
        "seed": run.seed,
        "root": str(run.root),
        "manifest_identity_sha256": run.manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(run.root / MANIFEST_NAME),
        "results_payload_sha256": run.results["payload_sha256"],
        "results_file_sha256": sha256_file(run.root / RESULTS_NAME),
        "completion_file_sha256": sha256_file(run.root / COMPLETION_NAME),
        "trace_file_sha256": sha256_file(trace),
        "frozen_baseline_root": str(run.baseline_root),
        "frozen_baseline_identity_sha256": run.baseline_identity_sha256,
        "strict_observer_validator_passed": True,
    }


def validate_summary_bundle(root: Path) -> dict[str, Any]:
    manifest = read_self_hashed_json(root / MANIFEST_NAME, "identity_sha256")
    results = read_self_hashed_json(root / RESULTS_NAME, "payload_sha256")
    completion = read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    runner = Path(__file__).resolve()
    fixed_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "cpu_only": True,
        "observe_only_inputs": True,
        "automatic_image_quality_scoring": False,
        "formal_statistical_test": False,
    }
    if any(manifest.get(key) != value for key, value in fixed_manifest.items()):
        raise RuntimeError("summary manifest scope changed")
    if manifest.get("runner", {}).get("sha256") != sha256_file(runner):
        raise RuntimeError("summary was produced by a different summarizer source")
    if results.get("manifest_identity_sha256") != manifest["identity_sha256"]:
        raise RuntimeError("summary results are not bound to the manifest")
    artifact_paths = (
        root / SUMMARY_CSV_NAME,
        root / CURVE_CSV_NAME,
        root / ALL_CURVES_PNG_NAME,
        root / CLASS207_PNG_NAME,
    )
    artifacts = [file_record(path, root) for path in artifact_paths]
    if results.get("artifacts") != artifacts or results.get("artifacts_sha256") != sha256_json(artifacts):
        raise RuntimeError("summary artifact identity changed")
    with (root / SUMMARY_CSV_NAME).open(newline="", encoding="utf-8") as handle:
        summary_count = sum(1 for _ in csv.DictReader(handle))
    with (root / CURVE_CSV_NAME).open(newline="", encoding="utf-8") as handle:
        curve_count = sum(1 for _ in csv.DictReader(handle))
    if summary_count != results.get("sample_summary_count"):
        raise RuntimeError("summary CSV row count changed")
    if curve_count != results.get("curve_point_count"):
        raise RuntimeError("curve CSV row count changed")
    expected_files = {
        (root / MANIFEST_NAME).resolve(),
        (root / RESULTS_NAME).resolve(),
        (root / COMPLETION_NAME).resolve(),
        *(path.resolve() for path in artifact_paths),
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            "summary file set changed; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    expected_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
        "results_payload_sha256": results["payload_sha256"],
        "results_file_sha256": sha256_file(root / RESULTS_NAME),
        "artifacts_sha256": results["artifacts_sha256"],
    }
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        raise RuntimeError("summary completion links are invalid")
    return results


def run_summary(
    *,
    runs: Sequence[ValidatedRun],
    frozen: FrozenComparison,
    source: dict[str, Any],
    dit_root: Path,
    outdir: Path,
) -> None:
    if outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output path: {outdir}")
    summaries: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for run in runs:
        run_summaries, run_curve_rows, run_curves = derive_run_records(run, frozen)
        summaries.extend(run_summaries)
        curve_rows.extend(run_curve_rows)
        curves.extend(run_curves)
    summaries.sort(key=lambda item: (int(item["seed"]), int(item["batch_index"])))
    curve_rows.sort(
        key=lambda item: (
            int(item["seed"]),
            int(item["batch_index"]),
            int(item["reverse_step_index"]),
        )
    )
    curves.sort(key=lambda item: (int(item["seed"]), int(item["batch_index"])))
    comparison = build_class207_comparison(summaries, frozen)

    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{outdir.name}.staging-", dir=outdir.parent)
    ).resolve()
    published = False
    try:
        runner = Path(__file__).resolve()
        observer_runner = runner.with_name("observe_dit_imagenet256_path_evidence.py")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "cpu_only": True,
            "gpu_model_loaded": False,
            "observe_only_inputs": True,
            "sampling_distribution_changed": False,
            "automatic_image_quality_scoring": False,
            "formal_statistical_test": False,
            "interpretation": (
                "descriptive operational Q/P evidence aggregation; larger log E is not by "
                "itself an image-quality judgment"
            ),
            "runner": {"path": str(runner), "sha256": sha256_file(runner)},
            "observer_runner": {
                "path": str(observer_runner),
                "sha256": sha256_file(observer_runner),
            },
            "strict_input_validation": {
                "validator": "observe_dit_imagenet256_path_evidence.validate_output_bundle",
                "trace_loader": "observe_dit_imagenet256_path_evidence._load_trace_exact",
                "passed_for_every_input": True,
            },
            "dit_source": source,
            "dit_root": str(dit_root),
            "frozen_preselection": {
                "path": str(frozen.path),
                "sha256": frozen.sha256,
                "status": frozen.payload["status"],
                "candidate_seed": frozen.candidate_seed,
                "candidate_class_id": frozen.candidate_class_id,
                "control_seeds": list(frozen.control_seeds),
            },
            "input_runs": [input_record(run) for run in runs],
            "seed_order": [run.seed for run in runs],
            "class_order": list(EXPECTED_CLASS_IDS),
            "running_max_definition": (
                "max over initial log E0=0 and all post-transition sample-mixture log E; "
                "timestep is null when E0 alone supplies the maximum"
            ),
            "top_tile_definition": (
                "fixed tile with greatest cumulative component log E at the specified "
                "post-transition row; ties use the lowest row-major tile index"
            ),
            "output_contract": {
                "summary_csv": SUMMARY_CSV_NAME,
                "curve_csv": CURVE_CSV_NAME,
                "all_curves_png": ALL_CURVES_PNG_NAME,
                "class207_comparison_png": CLASS207_PNG_NAME,
                "no_overwrite": True,
                "atomic_directory_publish": True,
            },
        }
        manifest["identity_sha256"] = canonical_self_hash(manifest, "identity_sha256")
        atomic_json_dump(manifest, staging / MANIFEST_NAME)
        atomic_csv_dump(summary_csv_rows(summaries), staging / SUMMARY_CSV_NAME)
        atomic_csv_dump(curve_rows, staging / CURVE_CSV_NAME)
        seeds = [run.seed for run in runs]
        render_all_curves(curves, seeds, staging / ALL_CURVES_PNG_NAME)
        render_class207_comparison(
            curves, summaries, comparison, frozen, staging / CLASS207_PNG_NAME
        )
        artifact_paths = (
            staging / SUMMARY_CSV_NAME,
            staging / CURVE_CSV_NAME,
            staging / ALL_CURVES_PNG_NAME,
            staging / CLASS207_PNG_NAME,
        )
        artifacts = [file_record(path, staging) for path in artifact_paths]
        results: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "automatic_image_quality_scoring": False,
            "formal_statistical_test": False,
            "sample_summary_count": len(summaries),
            "curve_count": len(curves),
            "curve_point_count": len(curve_rows),
            "sample_summaries": summaries,
            "sample_mixture_curves": curves,
            "class207_seed2_vs_frozen_controls": comparison,
            "artifacts": artifacts,
            "artifacts_sha256": sha256_json(artifacts),
        }
        results["payload_sha256"] = canonical_self_hash(results, "payload_sha256")
        atomic_json_dump(results, staging / RESULTS_NAME)
        completion: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / MANIFEST_NAME),
            "results_payload_sha256": results["payload_sha256"],
            "results_file_sha256": sha256_file(staging / RESULTS_NAME),
            "artifacts_sha256": results["artifacts_sha256"],
        }
        completion["payload_sha256"] = canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_summary_bundle(staging)
        if outdir.exists():
            raise RuntimeError(f"no-overwrite target appeared during staging: {outdir}")
        os.replace(staging, outdir)
        published = True
        validated_results = validate_summary_bundle(outdir)
        rank = next(
            item
            for item in validated_results["class207_seed2_vs_frozen_controls"]["ranks"]
            if item["metric"] == "post_transition_max_sample_mixture_log_e"
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "outdir": str(outdir),
                    "seeds": [run.seed for run in runs],
                    "samples": len(summaries),
                    "curve_points": len(curve_rows),
                    "class207_seed2_rank_by_post_transition_max": rank[
                        "candidate_descending_competition_rank_among_six"
                    ],
                    "rank_denominator": 6,
                    "results_payload_sha256": validated_results["payload_sha256"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _synthetic_run(seed: int, amplitude: float) -> Any:
    component = np.zeros(
        (EXPECTED_STEPS, len(EXPECTED_CLASS_IDS), EXPECTED_TILES), dtype=np.float64
    )
    decay = -0.002 * (np.arange(EXPECTED_STEPS, dtype=np.float64) + 1.0)
    component[:] = decay[:, None, None]
    component[3, 0, 5] += amplitude
    sample_mix = stable_logmeanexp(component, axis=2)
    bounds = []
    for row in range(4):
        for column in range(4):
            bounds.append((row * 8, column * 8, (row + 1) * 8, (column + 1) * 8))
    arrays = {
        "internal_timestep": np.arange(249, -1, -1, dtype=np.int16),
        "sample_mixture_log_e": sample_mix,
        "component_log_e": component,
        "tile_bounds_yxyx": np.asarray(bounds, dtype=np.int16),
    }
    return type(
        "SyntheticRun",
        (),
        {"seed": seed, "root": Path(f"/synthetic/seed{seed}"), "arrays": arrays},
    )()


def run_self_test() -> None:
    payload = {
        "status": "exploratory_single_reviewer_preselection_not_a_formal_bad_label",
    }
    frozen = FrozenComparison(
        path=Path("/synthetic/preselection.json"),
        sha256="0" * 64,
        payload=payload,
        candidate_seed=EXPECTED_CANDIDATE_SEED,
        candidate_batch_index=EXPECTED_CANDIDATE_BATCH_INDEX,
        candidate_class_id=EXPECTED_CANDIDATE_CLASS_ID,
        control_seeds=EXPECTED_CONTROL_SEEDS,
    )
    amplitudes = {1: 1.0, 2: 3.0, 3: 0.8, 4: 1.2, 7: 0.6, 9: 1.1}
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for seed in frozen.required_seeds:
        run_summaries, curve_rows, run_curves = derive_run_records(
            _synthetic_run(seed, amplitudes[seed]), frozen
        )
        if len(run_summaries) != 8 or len(curve_rows) != 8 * EXPECTED_STEPS:
            raise AssertionError("synthetic per-image/per-step row count failed")
        summaries.extend(run_summaries)
        curves.extend(run_curves)
    candidate = next(
        item
        for item in summaries
        if item["seed"] == 2 and item["class_id"] == EXPECTED_CANDIDATE_CLASS_ID
    )
    if candidate["post_transition_max_internal_timestep"] != 246:
        raise AssertionError("path-maximum timestep failed")
    if candidate["top_tile_at_post_transition_max"]["tile_index"] != 5:
        raise AssertionError("top fixed-tile localization failed")
    negative = next(
        item for item in summaries if item["seed"] == 2 and item["class_id"] == 360
    )
    if (
        negative["running_max_sample_mixture_log_e_from_E0"] != 0.0
        or negative["running_max_internal_timestep"] is not None
        or negative["running_max_source"] != "initial_E0_equals_1"
    ):
        raise AssertionError("initial-E0 running maximum semantics failed")
    comparison = build_class207_comparison(summaries, frozen)
    post_rank = next(
        item
        for item in comparison["ranks"]
        if item["metric"] == "post_transition_max_sample_mixture_log_e"
    )
    if post_rank["candidate_descending_competition_rank_among_six"] != 1:
        raise AssertionError("candidate descriptive rank failed")
    with tempfile.TemporaryDirectory(prefix="dit-evidence-summary-self-test-") as temporary:
        root = Path(temporary)
        atomic_csv_dump(summary_csv_rows(summaries), root / SUMMARY_CSV_NAME)
        render_class207_comparison(
            curves,
            summaries,
            comparison,
            frozen,
            root / CLASS207_PNG_NAME,
        )
        if sha256_file(root / SUMMARY_CSV_NAME) == sha256_file(root / CLASS207_PNG_NAME):
            raise AssertionError("synthetic artifact identity test failed")
    if "torch" in sys.modules:
        raise AssertionError("CPU-only self-test unexpectedly imported torch")
    print(
        "self-test passed: E0-aware maximum, observed maximum timestep, fixed-tile "
        "localization, per-step curves, frozen six-sample ranks, CSV/PNG, CPU-only"
    )


def build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parent.parent
    data_root = Path(
        os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")
    ).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="*",
        help="Completed observer bundles; must include seed 2 and frozen seeds 1,3,4,7,9.",
    )
    parser.add_argument("--dit-root", type=Path, default=data_root / "baselines/DiT")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--preselection",
        type=Path,
        default=repository
        / "experiments/annotations/dit_imagenet256_relative_bad_preselection_v1.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=data_root
        / "cross_scale_evidence/dit_imagenet256_path_evidence_analysis"
        / "class207_seed2_vs_frozen_controls",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        if args.run_dirs:
            parser.error("--self-test does not accept observer bundles")
        return
    if not args.run_dirs:
        parser.error("provide completed observer bundle directories")
    args.dit_root = args.dit_root.expanduser().absolute().resolve()
    args.checkpoint = (
        args.dit_root / "pretrained_models/DiT-XL-2-256x256.pt"
        if args.checkpoint is None
        else args.checkpoint.expanduser().absolute().resolve()
    )
    args.preselection = args.preselection.expanduser().absolute().resolve()
    args.outdir = args.outdir.expanduser().absolute().resolve()
    args.run_dirs = [path.expanduser().absolute().resolve() for path in args.run_dirs]
    if len(set(args.run_dirs)) != len(args.run_dirs):
        parser.error("duplicate observer bundle path")
    if args.outdir.exists():
        parser.error(f"no-overwrite target already exists: {args.outdir}")
    protected = [
        *args.run_dirs,
        args.dit_root,
        args.preselection,
        Path(__file__).resolve().parent.parent,
    ]
    if any(_paths_overlap(args.outdir, path) for path in protected):
        parser.error("--outdir overlaps an input, source tree, annotation, or repository")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)
    if args.self_test:
        run_self_test()
        return 0
    frozen = load_frozen_comparison(args.preselection)
    runs, source = validate_input_runs(
        args.run_dirs, dit_root=args.dit_root, checkpoint=args.checkpoint
    )
    observed_seeds = {run.seed for run in runs}
    required_seeds = set(frozen.required_seeds)
    if not required_seeds.issubset(observed_seeds):
        parser.error(
            "inputs must include the frozen class207 comparison seeds; "
            f"missing={sorted(required_seeds - observed_seeds)}"
        )
    run_summary(
        runs=runs,
        frozen=frozen,
        source=source,
        dit_root=args.dit_root,
        outdir=args.outdir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
