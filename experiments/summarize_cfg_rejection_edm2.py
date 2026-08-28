#!/usr/bin/env python3
"""Aggregate CFG-Rejection reproduction signals and render symmetric tails.

The grids are diagnostics, not bad-sample labels.  Every available class is
rendered with both the lowest and highest values of every registered metric so
that qualitative inspection cannot silently show only a favorable tail.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr


METRICS = (
    "official_notebook_metric_tau5",
    "denoiser_asd_tau5",
    "denoiser_asd_full",
    "score_asd_tau5",
    "score_asd_full",
)


def parse_class_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--grid-classes must be comma-separated integers") from exc
    if not result or len(result) != len(set(result)) or any(item < 0 or item >= 1_000 for item in result):
        raise argparse.ArgumentTypeError("--grid-classes must contain unique ImageNet IDs in [0, 999]")
    return result


@dataclass(frozen=True)
class Record:
    class_id: int
    seed: int
    image_path: Path
    signal_path: Path
    metrics: dict[str, float]


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_class_names(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value[1]) for key, value in raw.items()}


def load_records(run_dir: Path) -> list[Record]:
    records: list[Record] = []
    for signal_path in sorted((run_dir / "signals").glob("class_*/*.npz")):
        with np.load(signal_path, allow_pickle=False) as payload:
            class_id = int(payload["class_id"])
            seed = int(payload["seed"])
            metrics = {name: float(payload[name]) for name in METRICS}
        image_path = run_dir / "images" / f"class_{class_id:04d}" / f"{seed:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"signal has no matching image: {signal_path}")
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"non-finite metric in {signal_path}: {metrics}")
        records.append(Record(class_id, seed, image_path, signal_path, metrics))
    if not records:
        raise FileNotFoundError(f"no signal NPZ files found under {run_dir / 'signals'}")
    return records


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required run metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def pair_from_path(path: Path) -> tuple[int, int]:
    class_match = re.fullmatch(r"class_(\d{4})", path.parent.name)
    seed_match = re.fullmatch(r"(\d{6,})", path.stem)
    if class_match is None or seed_match is None:
        raise RuntimeError(f"unexpected sample path layout: {path}")
    return int(class_match.group(1)), int(seed_match.group(1))


def validate_run(
    run_dir: Path,
    records: list[Record],
    manifest: dict[str, Any],
    completion: dict[str, Any] | None,
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    class_ids = tuple(int(value) for value in manifest.get("class_ids", []))
    seeds = tuple(int(value) for value in manifest.get("seeds", []))
    if not class_ids or not seeds or len(set(class_ids)) != len(class_ids) or len(set(seeds)) != len(seeds):
        raise RuntimeError("manifest must contain non-empty, unique class_ids and seeds")
    expected = {(class_id, seed) for class_id in class_ids for seed in seeds}
    manifest_count = int(manifest.get("sample_count", -1))
    if manifest_count != len(expected):
        raise RuntimeError(f"manifest sample_count mismatch: {manifest_count} != {len(expected)}")

    record_pairs = [(record.class_id, record.seed) for record in records]
    if len(record_pairs) != len(set(record_pairs)):
        raise RuntimeError("duplicate (class_id, seed) records were found")
    actual = set(record_pairs)
    extra = actual - expected
    missing = expected - actual
    if extra:
        raise RuntimeError(f"run contains {len(extra)} unexpected signal pairs; first={sorted(extra)[:3]}")
    if missing and not allow_partial:
        raise RuntimeError(f"run is incomplete: {len(missing)} expected pairs are missing; first={sorted(missing)[:3]}")

    image_paths = sorted((run_dir / "images").glob("class_*/*.png"))
    image_pairs = [pair_from_path(path) for path in image_paths]
    if len(image_pairs) != len(set(image_pairs)):
        raise RuntimeError("duplicate (class_id, seed) images were found")
    if set(image_pairs) != actual:
        raise RuntimeError("the image and signal pair sets do not match exactly")

    if completion is None:
        if not allow_partial:
            raise RuntimeError("completion.json is required unless --allow-partial is explicit")
    else:
        if int(completion.get("total_expected", -1)) != len(expected):
            raise RuntimeError("completion total_expected does not match manifest")
        completed = int(completion.get("generated_this_run", -1)) + int(
            completion.get("already_complete", -1)
        )
        if completed != len(actual):
            raise RuntimeError(f"completion count mismatch: {completed} != {len(actual)}")
        if missing and not allow_partial:
            raise RuntimeError("completion metadata exists but the run is still incomplete")

    return {
        "expected_count": len(expected),
        "observed_count": len(actual),
        "missing_count": len(missing),
        "complete": not missing,
    }


def write_csv(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class_id", "seed", "image_path", "signal_path", *METRICS),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "class_id": record.class_id,
                    "seed": record.seed,
                    "image_path": str(record.image_path),
                    "signal_path": str(record.signal_path),
                    **record.metrics,
                }
            )
    temporary.replace(path)


def summarize(records: list[Record], validation: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    by_class: dict[int, list[Record]] = {}
    for record in records:
        by_class.setdefault(record.class_id, []).append(record)

    matrix = np.asarray([[record.metrics[name] for name in METRICS] for record in records])
    correlations = np.asarray(spearmanr(matrix, axis=0).statistic)
    summary: dict[str, Any] = {
        "sample_count": len(records),
        "class_count": len(by_class),
        "protocol": manifest.get("protocol"),
        "run_validation": validation,
        "metrics": list(METRICS),
        "dependence_note": (
            "The same seed reuses the same initial noise across classes. Pooled correlations are descriptive; "
            "inference must stratify by class and cluster or hierarchically bootstrap by seed/class."
        ),
        "global_spearman": {
            row_name: {column_name: float(correlations[row, column]) for column, column_name in enumerate(METRICS)}
            for row, row_name in enumerate(METRICS)
        },
        "per_class": {},
    }
    for class_id, class_records in sorted(by_class.items()):
        class_summary: dict[str, Any] = {"sample_count": len(class_records), "metrics": {}}
        for metric in METRICS:
            values = np.asarray([record.metrics[metric] for record in class_records], dtype=np.float64)
            class_summary["metrics"][metric] = {
                "min": float(values.min()),
                "q10": float(np.quantile(values, 0.1)),
                "median": float(np.median(values)),
                "q90": float(np.quantile(values, 0.9)),
                "max": float(values.max()),
            }
        if len(class_records) >= 3:
            class_matrix = np.asarray(
                [[record.metrics[name] for name in METRICS] for record in class_records], dtype=np.float64
            )
            class_correlations = np.asarray(spearmanr(class_matrix, axis=0).statistic)
            class_summary["spearman"] = {
                row_name: {
                    column_name: float(class_correlations[row, column])
                    for column, column_name in enumerate(METRICS)
                }
                for row, row_name in enumerate(METRICS)
            }
        summary["per_class"][str(class_id)] = class_summary
    return summary


def labeled_thumbnail(record: Record, metric: str, side: int, rank_label: str) -> Image.Image:
    label_height = 42
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((side, side), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (side, side + label_height), "white")
    tile.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    draw = ImageDraw.Draw(tile)
    draw.text((4, side + 3), f"{rank_label} seed={record.seed}", fill="black")
    draw.text((4, side + 20), f"{record.metrics[metric]:.5g}", fill="black")
    return tile


def render_tail_grid(
    class_records: list[Record],
    metric: str,
    output: Path,
    *,
    class_name: str,
    tail_count: int,
    side: int,
) -> None:
    ordered = sorted(class_records, key=lambda record: record.metrics[metric])
    count = min(tail_count, max(1, len(ordered) // 2))
    low = ordered[:count]
    high = ordered[-count:][::-1]
    label_height = 42
    heading_height = 38
    grid = Image.new("RGB", (count * side, heading_height + 2 * (side + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((4, 3), f"class {class_records[0].class_id}: {class_name} | {metric}", fill="black")
    draw.text((4, 20), "top row: LOW tail | bottom row: HIGH tail", fill="black")
    for column, record in enumerate(low):
        grid.paste(labeled_thumbnail(record, metric, side, f"low#{column + 1}"), (column * side, heading_height))
    for column, record in enumerate(high):
        grid.paste(
            labeled_thumbnail(record, metric, side, f"high#{column + 1}"),
            (column * side, heading_height + side + label_height),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    grid.save(temporary, format="PNG")
    temporary.replace(output)


def main() -> None:
    data_root = Path("/home/zhoushunyu/data/eqvae")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tail-count", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--no-grids", action="store_true")
    parser.add_argument(
        "--grid-classes",
        type=parse_class_ids,
        default=None,
        help="Render grids only for these comma-separated class IDs; summaries still use the full run.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Analyze an explicitly incomplete run; never use this output as a final result.",
    )
    parser.add_argument(
        "--class-index-json",
        type=Path,
        default=(
            data_root
            / "baselines"
            / "fkc-diffusion"
            / "applications"
            / "images"
            / "edm2"
            / "imagenet_class_index.json"
        ),
    )
    args = parser.parse_args()
    if args.tail_count < 1 or args.thumb_size < 32:
        parser.error("--tail-count must be positive and --thumb-size must be at least 32")
    output_dir = args.output_dir or args.run_dir / "analysis"
    manifest = load_json_object(args.run_dir / "manifest.json")
    completion_path = args.run_dir / "completion.json"
    completion = load_json_object(completion_path) if completion_path.is_file() else None
    records = load_records(args.run_dir)
    validation = validate_run(
        args.run_dir,
        records,
        manifest,
        completion,
        allow_partial=args.allow_partial,
    )
    write_csv(records, output_dir / "signals.csv")
    summary = summarize(records, validation, manifest)
    atomic_json_dump(summary, output_dir / "summary.json")

    if not args.no_grids:
        class_names = load_class_names(args.class_index_json)
        by_class: dict[int, list[Record]] = {}
        for record in records:
            by_class.setdefault(record.class_id, []).append(record)
        if args.grid_classes is not None:
            missing_grid_classes = sorted(set(args.grid_classes) - set(by_class))
            if missing_grid_classes:
                raise ValueError(f"--grid-classes are absent from the run: {missing_grid_classes}")
            by_class = {class_id: by_class[class_id] for class_id in args.grid_classes}
        for metric in METRICS:
            for class_id, class_records in sorted(by_class.items()):
                render_tail_grid(
                    class_records,
                    metric,
                    output_dir / "grids" / metric / f"class_{class_id:04d}.png",
                    class_name=class_names.get(class_id, "unknown"),
                    tail_count=args.tail_count,
                    side=args.thumb_size,
                )

    print(json.dumps({"records": len(records), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
