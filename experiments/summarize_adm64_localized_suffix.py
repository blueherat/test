#!/usr/bin/env python3
"""Read-only pixel-difference summary for completed ADM64 localized suffix runs.

The summarizer never loads ADM weights and never starts CUDA.  For every input
directory it reuses the localized-suffix runner's CPU validator, including the
completed manifest/results self-hashes, exact branch set, PNG provenance, NPZ
hashes, original replay identity, and hard-mask innovation construction.

It then reports descriptive RGB differences relative to ``original_replay``:
MAE, RMSE, changed-spatial-pixel fraction (any RGB channel differs), and PSNR
for the full image, inside the frozen rectangle, and outside it.  Values remain
in the native uint8 [0,255] scale.  These are difference magnitudes, not image-
quality scores.  No branch is ranked, accepted, rejected, or selected.

Two contact sheets contain every output in the fixed order
``original_replay, localized_attempt_000..N, full_fresh_control``.  One uses
nearest-neighbour enlargement and one uses smooth Lanczos enlargement.  The
CSV and grids are hashed by a self-hashed JSON summary.  The analysis directory
is staged atomically and never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

try:  # Package and direct CLI imports.
    from .intervene_adm64_localized_suffix import (
        EXPERIMENT as LOCALIZED_EXPERIMENT,
        InputTrace,
        Rectangle,
        _read_self_hashed_json,
        branch_ids,
        branch_image_path,
        load_baseline_pair,
        load_input_trace,
        validate_bundle,
    )
    from .observe_adm64_cross_scale_evidence import (
        BaselineReference,
        _canonical_payload_sha,
        decoded_pixels,
        original_schedule_and_timestep_map,
    )
    from .reproduce_adm64_guided import atomic_json_dump, sha256_file
except ImportError:  # pragma: no cover.
    from intervene_adm64_localized_suffix import (
        EXPERIMENT as LOCALIZED_EXPERIMENT,
        InputTrace,
        Rectangle,
        _read_self_hashed_json,
        branch_ids,
        branch_image_path,
        load_baseline_pair,
        load_input_trace,
        validate_bundle,
    )
    from observe_adm64_cross_scale_evidence import (
        BaselineReference,
        _canonical_payload_sha,
        decoded_pixels,
        original_schedule_and_timestep_map,
    )
    from reproduce_adm64_guided import atomic_json_dump, sha256_file


EXPERIMENT = "adm64_localized_suffix_pixel_difference_summary"
SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "run_index",
    "run_dir",
    "manifest_identity_sha256",
    "results_payload_sha256",
    "class_id",
    "seed",
    "rollback_internal_timestep",
    "rectangle_xyxy_half_open",
    "branch_order",
    "branch_id",
    "role",
    "attempt_index",
    "stream_seed",
    "region",
    "spatial_pixel_count",
    "rgb_value_count",
    "rgb_mae_0_255",
    "rgb_rmse_0_255",
    "changed_pixel_fraction_any_rgb",
    "psnr_db",
    "psnr_is_infinite",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def region_metrics(
    original: np.ndarray,
    candidate: np.ndarray,
    spatial_mask: np.ndarray,
) -> dict[str, Any]:
    if original.shape != candidate.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("metrics require matching HxWx3 RGB arrays")
    if original.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise TypeError("metrics require native uint8 images")
    if spatial_mask.shape != original.shape[:2] or spatial_mask.dtype != np.bool_:
        raise ValueError("spatial mask must be a boolean HxW array")
    count = int(spatial_mask.sum(dtype=np.int64))
    if count == 0:
        return {
            "spatial_pixel_count": 0,
            "rgb_value_count": 0,
            "rgb_mae_0_255": None,
            "rgb_rmse_0_255": None,
            "changed_pixel_fraction_any_rgb": None,
            "psnr_db": None,
            "psnr_is_infinite": False,
            "metric_defined": False,
        }
    left = original[spatial_mask].astype(np.float64)
    right = candidate[spatial_mask].astype(np.float64)
    difference = right - left
    absolute = np.abs(difference)
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    rmse = math.sqrt(mse)
    changed = np.any(difference != 0, axis=1)
    infinite = rmse == 0.0
    return {
        "spatial_pixel_count": count,
        "rgb_value_count": count * 3,
        "rgb_mae_0_255": float(np.mean(absolute, dtype=np.float64)),
        "rgb_rmse_0_255": rmse,
        "changed_pixel_fraction_any_rgb": float(changed.mean(dtype=np.float64)),
        "psnr_db": None if infinite else float(20.0 * math.log10(255.0 / rmse)),
        "psnr_is_infinite": infinite,
        "metric_defined": True,
    }


def compute_branch_metrics(
    original: np.ndarray,
    candidate: np.ndarray,
    rectangle: Rectangle,
) -> dict[str, dict[str, Any]]:
    height, width = original.shape[:2]
    x0, y0, x1, y1 = rectangle
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("rectangle lies outside decoded image")
    inside = np.zeros((height, width), dtype=np.bool_)
    inside[y0:y1, x0:x1] = True
    return {
        "full": region_metrics(original, candidate, np.ones_like(inside)),
        "inside_mask": region_metrics(original, candidate, inside),
        "outside_mask": region_metrics(original, candidate, ~inside),
    }


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def build_grid(
    runs: Sequence[dict[str, Any]],
    *,
    scale: int,
    smooth: bool,
    output_path: Path,
) -> None:
    if not runs:
        raise ValueError("cannot build an empty result grid")
    maximum_columns = max(len(run["visuals"]) for run in runs)
    tile = 64 * scale
    title_height = 34
    label_height = 28
    row_height = title_height + label_height + tile
    canvas = Image.new("RGB", (maximum_columns * tile, len(runs) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = _font(18), _font(15)
    resampling = Image.Resampling.LANCZOS if smooth else Image.Resampling.NEAREST
    for row, run in enumerate(runs):
        top = row * row_height
        draw.rectangle((0, top, canvas.width, top + title_height), fill=(32, 32, 32))
        draw.text((8, top + 6), run["label"], fill="white", font=title_font)
        for column, (branch_id, pixels) in enumerate(run["visuals"]):
            left = column * tile
            label_top = top + title_height
            draw.rectangle(
                (left, label_top, left + tile, label_top + label_height),
                fill=(225, 225, 225),
            )
            draw.text((left + 5, label_top + 5), branch_id, fill="black", font=label_font)
            enlarged = Image.fromarray(pixels, mode="RGB").resize(
                (tile, tile), resample=resampling
            )
            canvas.paste(enlarged, (left, label_top + label_height))
    metadata = PngInfo()
    metadata.add_text("experiment", EXPERIMENT)
    metadata.add_text("rendering", "smooth_lanczos" if smooth else "nearest_neighbor")
    metadata.add_text("run_count", str(len(runs)))
    metadata.add_text("branch_order", "original_replay,localized_attempts,full_fresh_control")
    canvas.save(output_path, format="PNG", pnginfo=metadata)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in CSV_COLUMNS})


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def validate_and_measure_run(
    run_dir: Path,
    run_index: int,
    *,
    schedule_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    baseline_cache: dict[tuple[str, int, int], BaselineReference],
    trace_cache: dict[tuple[str, int, int], InputTrace],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_self_hashed_json(run_dir / "manifest.json", "identity_sha256")
    if manifest.get("experiment") != LOCALIZED_EXPERIMENT:
        raise RuntimeError(f"not a localized suffix run: {run_dir}")
    if manifest.get("method_claim_eligible") is not False or manifest.get(
        "statistical_scope", {}
    ).get("conditional_Ville_retry_bound_applicable") is not False:
        raise RuntimeError("input run lost its oracle/no-bound scope labels")
    runner_record = manifest.get("runner", {})
    current_runner = Path(__file__).with_name("intervene_adm64_localized_suffix.py").resolve()
    if runner_record.get("sha256") != sha256_file(current_runner):
        raise RuntimeError("input run localized-suffix runner hash differs from current source")
    pair_record = manifest.get("pair", {})
    pair = (int(pair_record.get("class_id", -1)), int(pair_record.get("seed", -1)))
    oracle = manifest.get("oracle_inputs", {})
    rollback_t = int(oracle.get("rollback_internal_timestep", -1))
    rectangle_values = oracle.get("rectangle_xyxy_half_open")
    if not isinstance(rectangle_values, list) or len(rectangle_values) != 4:
        raise RuntimeError("input run has no valid rectangle")
    rectangle: Rectangle = tuple(int(value) for value in rectangle_values)  # type: ignore[assignment]
    attempt_count = int(oracle.get("attempt_count", -1))
    expected_branches = branch_ids(attempt_count)
    if manifest.get("outputs", {}).get("branch_ids") != list(expected_branches):
        raise RuntimeError("manifest branch set/order is invalid")
    checkpoint_records = manifest.get("checkpoints", {})
    model_sha = checkpoint_records.get("diffusion", {}).get("sha256")
    classifier_sha = checkpoint_records.get("classifier", {}).get("sha256")
    if not isinstance(model_sha, str) or not isinstance(classifier_sha, str):
        raise RuntimeError("input run lacks checkpoint hashes")

    guided_root = Path(manifest.get("sources", {}).get("guided_diffusion_root", ""))
    schedule_key = str(guided_root.resolve())
    if schedule_key not in schedule_cache:
        schedule_cache[schedule_key] = original_schedule_and_timestep_map(guided_root)
    original_alpha_bar, timestep_map = schedule_cache[schedule_key]
    baseline_root = Path(manifest.get("frozen_baseline", {}).get("root", ""))
    baseline_key = (str(baseline_root.resolve()), pair[0], pair[1])
    if baseline_key not in baseline_cache:
        baseline_cache[baseline_key] = load_baseline_pair(
            baseline_root,
            pair,
            model_sha256=model_sha,
            classifier_sha256=classifier_sha,
        )
    baseline = baseline_cache[baseline_key]
    if baseline.manifest_identity_sha256 != manifest.get("frozen_baseline", {}).get(
        "manifest_identity_sha256"
    ):
        raise RuntimeError("validated baseline identity differs from run manifest")

    input_root = Path(manifest.get("input_all_step_local_trace", {}).get("root", ""))
    trace_key = (str(input_root.resolve()), pair[0], pair[1])
    if trace_key not in trace_cache:
        trace_cache[trace_key] = load_input_trace(
            input_root,
            pair,
            baseline,
            model_sha256=model_sha,
            classifier_sha256=classifier_sha,
            original_alpha_bar=original_alpha_bar,
            timestep_map=timestep_map,
        )
    input_trace = trace_cache[trace_key]
    if input_trace.trace_sha256 != manifest.get("input_all_step_local_trace", {}).get(
        "trace_sha256"
    ):
        raise RuntimeError("validated all-step trace identity differs from run manifest")

    # This is the complete runner-side CPU validator.  It loads branch NPZs,
    # but never loads a neural-network checkpoint or initializes CUDA.
    results = validate_bundle(
        run_dir,
        manifest,
        input_trace,
        baseline,
        pair,
        rectangle,
        attempt_count=attempt_count,
        rollback_internal_timestep=rollback_t,
        require_completion=True,
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("read-only summarization unexpectedly initialized CUDA")

    original = decoded_pixels(branch_image_path(run_dir, "original_replay"))
    visuals: list[tuple[str, np.ndarray]] = []
    branch_summaries: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    result_records = {record["branch_id"]: record for record in results["branches"]}
    for branch_order, branch_id in enumerate(expected_branches):
        pixels = decoded_pixels(branch_image_path(run_dir, branch_id))
        visuals.append((branch_id, pixels))
        metrics = compute_branch_metrics(original, pixels, rectangle)
        source_record = result_records[branch_id]
        branch_summary = {
            "branch_order": branch_order,
            "branch_id": branch_id,
            "role": source_record["role"],
            "attempt_index": source_record["attempt_index"],
            "stream_seed": source_record["stream_seed"],
            "pixel_sha256": source_record["image"]["pixel_sha256"],
            "metrics_relative_to_original_replay": metrics,
        }
        branch_summaries.append(branch_summary)
        for region in ("full", "inside_mask", "outside_mask"):
            region_values = metrics[region]
            csv_rows.append(
                {
                    "run_index": run_index,
                    "run_dir": str(run_dir),
                    "manifest_identity_sha256": manifest["identity_sha256"],
                    "results_payload_sha256": results["payload_sha256"],
                    "class_id": pair[0],
                    "seed": pair[1],
                    "rollback_internal_timestep": rollback_t,
                    "rectangle_xyxy_half_open": ",".join(str(v) for v in rectangle),
                    "branch_order": branch_order,
                    "branch_id": branch_id,
                    "role": source_record["role"],
                    "attempt_index": source_record["attempt_index"],
                    "stream_seed": source_record["stream_seed"],
                    "region": region,
                    **{key: value for key, value in region_values.items() if key in CSV_COLUMNS},
                }
            )
    run_summary = {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "results_payload_sha256": results["payload_sha256"],
        "completion_file_sha256": sha256_file(run_dir / "completion.json"),
        "class_id": pair[0],
        "seed": pair[1],
        "rollback_internal_timestep": rollback_t,
        "rectangle_xyxy_half_open": list(rectangle),
        "attempt_count": attempt_count,
        "branch_order": list(expected_branches),
        "validated_with_runner_cpu_validator": True,
        "branches": branch_summaries,
    }
    visual_record = {
        "label": (
            f"run {run_index}: class {pair[0]} seed {pair[1]} | rollback t={rollback_t} | "
            f"rect={rectangle}"
        ),
        "visuals": visuals,
    }
    return run_summary, csv_rows, visual_record


def validate_analysis_output(root: Path) -> dict[str, Any]:
    summary = _read_self_hashed_json(root / "summary.json", "payload_sha256")
    expected_files = {
        (root / "summary.json").resolve(),
        (root / "branch_metrics.csv").resolve(),
        (root / "grid_nearest.png").resolve(),
        (root / "grid_smooth.png").resolve(),
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("analysis output file set is incomplete or unexpected")
    for name, record in summary.get("artifacts", {}).items():
        path = root / record.get("relative_path", "")
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"analysis artifact missing/wrong size: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"analysis artifact hash failed: {name}")
    with (root / "branch_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise RuntimeError("analysis CSV schema changed")
        rows = list(reader)
    if len(rows) != summary.get("csv_row_count"):
        raise RuntimeError("analysis CSV row count differs from summary")
    for name in ("grid_nearest.png", "grid_smooth.png"):
        with Image.open(root / name) as image:
            if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
                raise RuntimeError(f"invalid grid image: {name}")
            image.verify()
    return summary


def run_summary(run_dirs: Sequence[Path], output_dir: Path, scale: int) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("summarizer must run in a fresh CPU-only process")
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite analysis directory: {output_dir}")
    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError("duplicate input run directory")
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        if _paths_overlap(output_dir, run_dir):
            raise ValueError("analysis output may not overlap an input run directory")

    schedule_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    baseline_cache: dict[tuple[str, int, int], BaselineReference] = {}
    trace_cache: dict[tuple[str, int, int], InputTrace] = {}
    run_summaries: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    start = time.monotonic()
    for index, run_dir in enumerate(run_dirs):
        run_summary, run_rows, visual = validate_and_measure_run(
            run_dir,
            index,
            schedule_cache=schedule_cache,
            baseline_cache=baseline_cache,
            trace_cache=trace_cache,
        )
        run_summaries.append(run_summary)
        csv_rows.extend(run_rows)
        visuals.append(visual)
        print(f"validated and measured {index + 1}/{len(run_dirs)}: {run_dir.name}", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("summarizer initialized CUDA during validation")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        csv_path = staging / "branch_metrics.csv"
        nearest_path = staging / "grid_nearest.png"
        smooth_path = staging / "grid_smooth.png"
        write_csv(csv_rows, csv_path)
        build_grid(visuals, scale=scale, smooth=False, output_path=nearest_path)
        build_grid(visuals, scale=scale, smooth=True, output_path=smooth_path)
        artifacts = {
            "branch_metrics_csv": {
                "relative_path": csv_path.name,
                "bytes": csv_path.stat().st_size,
                "sha256": sha256_file(csv_path),
            },
            "grid_nearest": {
                "relative_path": nearest_path.name,
                "bytes": nearest_path.stat().st_size,
                "sha256": sha256_file(nearest_path),
                "resampling": "nearest_neighbor",
            },
            "grid_smooth": {
                "relative_path": smooth_path.name,
                "bytes": smooth_path.stat().st_size,
                "sha256": sha256_file(smooth_path),
                "resampling": "lanczos",
            },
        }
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "read_only": True,
            "gpu_model_loaded": False,
            "cuda_initialized": False,
            "automatic_quality_scoring": False,
            "branch_ranking_or_selection": False,
            "metric_interpretation": "descriptive RGB difference from original replay; not quality",
            "metric_definitions": {
                "value_scale": "native uint8 RGB [0,255]",
                "rgb_mae_0_255": "mean absolute difference over selected pixels and 3 channels",
                "rgb_rmse_0_255": "root mean squared difference over selected pixels and 3 channels",
                "changed_pixel_fraction_any_rgb": (
                    "fraction of selected spatial pixels where at least one RGB channel differs"
                ),
                "psnr_db": "20*log10(255/RMSE); null with psnr_is_infinite=true when RMSE=0",
                "regions": "full image, frozen rectangle, and its spatial complement",
            },
            "input_run_count": len(run_dirs),
            "csv_row_count": len(csv_rows),
            "grid_scale": scale,
            "grid_branch_order": "original_replay, localized attempts by index, full_fresh_control",
            "runs": run_summaries,
            "artifacts": artifacts,
            "wall_seconds": time.monotonic() - start,
        }
        summary["payload_sha256"] = _canonical_payload_sha(summary, "payload_sha256")
        atomic_json_dump(summary, staging / "summary.json")
        validate_analysis_output(staging)
        if output_dir.exists():
            raise RuntimeError("analysis target appeared during staging; refusing overwrite")
        os.replace(staging, output_dir)
    final = validate_analysis_output(output_dir)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def run_self_test() -> None:
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    candidate = original.copy()
    candidate[0, 0] = np.asarray([3, 4, 0], dtype=np.uint8)
    candidate[3, 3] = np.asarray([0, 0, 6], dtype=np.uint8)
    metrics = compute_branch_metrics(original, candidate, (0, 0, 2, 2))
    if metrics["full"]["spatial_pixel_count"] != 16 or not math.isclose(
        metrics["full"]["changed_pixel_fraction_any_rgb"], 2 / 16
    ):
        raise AssertionError("full-image changed-pixel metric failed")
    if metrics["inside_mask"]["spatial_pixel_count"] != 4 or not math.isclose(
        metrics["inside_mask"]["changed_pixel_fraction_any_rgb"], 1 / 4
    ):
        raise AssertionError("inside-mask metric failed")
    if metrics["outside_mask"]["spatial_pixel_count"] != 12 or not math.isclose(
        metrics["outside_mask"]["changed_pixel_fraction_any_rgb"], 1 / 12
    ):
        raise AssertionError("outside-mask metric failed")
    zero = compute_branch_metrics(original, original, (0, 0, 2, 2))
    if any(
        zero[region]["psnr_db"] is not None
        or zero[region]["psnr_is_infinite"] is not True
        for region in zero
    ):
        raise AssertionError("zero-difference infinite PSNR encoding failed")
    expected_mse = (3**2 + 4**2 + 6**2) / (16 * 3)
    if not math.isclose(metrics["full"]["rgb_rmse_0_255"], math.sqrt(expected_mse)):
        raise AssertionError("RGB RMSE formula failed")

    image_a = np.zeros((64, 64, 3), dtype=np.uint8)
    image_b = np.full((64, 64, 3), 127, dtype=np.uint8)
    visuals = [
        {
            "label": "toy run t=3",
            "visuals": [
                ("original_replay", image_a),
                ("localized_attempt_000", image_b),
                ("full_fresh_control", image_a),
            ],
        }
    ]
    with tempfile.TemporaryDirectory(prefix="adm64-localized-summary-self-test-") as temporary:
        root = Path(temporary)
        nearest = root / "nearest.png"
        smooth = root / "smooth.png"
        build_grid(visuals, scale=2, smooth=False, output_path=nearest)
        build_grid(visuals, scale=2, smooth=True, output_path=smooth)
        with Image.open(nearest) as image:
            if image.width != 3 * 128 or image.height != 34 + 28 + 128:
                raise AssertionError("toy grid dimensions failed")
        row = {key: "" for key in CSV_COLUMNS}
        row.update(
            {
                "run_index": 0,
                "branch_id": "original_replay",
                "region": "full",
                "psnr_is_infinite": True,
            }
        )
        csv_path = root / "metrics.csv"
        write_csv([row], csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS or len(list(reader)) != 1:
                raise AssertionError("toy CSV schema failed")
        payload = {
            "csv_sha256": sha256_file(csv_path),
            "nearest_sha256": sha256_file(nearest),
            "smooth_sha256": sha256_file(smooth),
        }
        payload["payload_sha256"] = _canonical_payload_sha(payload, "payload_sha256")
        if payload["payload_sha256"] != _canonical_payload_sha(payload, "payload_sha256"):
            raise AssertionError("toy self-hashed summary failed")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU summarizer self-test initialized CUDA")
    print(
        "self-test passed: full/inside/outside RGB metrics, infinite-PSNR encoding, "
        "fixed-order nearest/smooth grids, CSV schema, hashes, and CPU-only execution"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="completed localized suffix directory; repeat for multiple runs",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grid-scale", type=int, choices=(2, 4, 6, 8), default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.run_dir:
        raise ValueError("at least one --run-dir is required")
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    run_dirs = [path.resolve() for path in args.run_dir]
    output_dir = args.output_dir.resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "read_only": True,
                    "gpu_model_loaded": False,
                    "automatic_quality_scoring": False,
                    "branch_ranking_or_selection": False,
                    "run_dirs": [str(path) for path in run_dirs],
                    "run_count": len(run_dirs),
                    "output_dir": str(output_dir),
                    "grid_scale": args.grid_scale,
                    "outputs": [
                        "summary.json",
                        "branch_metrics.csv",
                        "grid_nearest.png",
                        "grid_smooth.png",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_summary(run_dirs, output_dir, args.grid_scale)


if __name__ == "__main__":
    main()
