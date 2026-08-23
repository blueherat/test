#!/usr/bin/env python3
"""Summarize the three held-out AdvFD Frechet representation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FDR3_MODELS = (
    ("convnext", 56.87),
    ("dinov2_cls", 14.19),
    ("clip_cls", 5.60),
)


def parse_condition_csv(value: str) -> tuple[str, Path]:
    try:
        condition, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected CONDITION=CSV_PATH") from exc
    if not condition or not path:
        raise argparse.ArgumentTypeError("expected non-empty CONDITION=CSV_PATH")
    return condition, Path(path)


def summarize_csv(condition: str, path: Path) -> dict[str, float | int | str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_model: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row.get("model", "")
        if model in dict(FDR3_MODELS):
            if model in by_model:
                raise ValueError(f"duplicate {model!r} row in {path}")
            by_model[model] = row

    missing = [model for model, _ in FDR3_MODELS if model not in by_model]
    if missing:
        raise ValueError(f"missing FDr3 rows in {path}: {', '.join(missing)}")

    sample_counts = {int(by_model[model]["n"]) for model, _ in FDR3_MODELS}
    if len(sample_counts) != 1:
        raise ValueError(f"inconsistent sample counts in {path}: {sample_counts}")

    summary: dict[str, float | int | str] = {
        "condition": condition,
        "source_csv": str(path.resolve()),
        "num_images": sample_counts.pop(),
    }
    fdr_values = []
    for model, validation_fd in FDR3_MODELS:
        raw_fd = float(by_model[model]["fd"])
        reported_fdr = by_model[model].get("fdr", "")
        fdr = raw_fd / validation_fd
        if reported_fdr and abs(float(reported_fdr) - fdr) > 1e-5:
            raise ValueError(
                f"official FDr mismatch for {model}: {reported_fdr} vs {fdr}"
            )
        prefix = model.removesuffix("_cls")
        summary[f"{prefix}_fd"] = raw_fd
        summary[f"{prefix}_fdr"] = fdr
        fdr_values.append(fdr)
    summary["fdr3"] = sum(fdr_values) / len(fdr_values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition-csv",
        action="append",
        required=True,
        type=parse_condition_csv,
        metavar="CONDITION=CSV_PATH",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    summaries = [summarize_csv(condition, path) for condition, path in args.condition_csv]
    counts = {int(row["num_images"]) for row in summaries}
    if len(counts) != 1:
        raise ValueError(f"conditions use different sample counts: {counts}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0])
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    by_condition = {str(row["condition"]): row for row in summaries}
    comparisons = {}
    if {"static", "advfd"}.issubset(by_condition):
        static = by_condition["static"]
        advfd = by_condition["advfd"]
        for key in (
            "convnext_fd",
            "convnext_fdr",
            "dinov2_fd",
            "dinov2_fdr",
            "clip_fd",
            "clip_fdr",
            "fdr3",
        ):
            comparisons[f"advfd_minus_static_{key}"] = float(advfd[key]) - float(
                static[key]
            )

    payload = {
        "definition": "mean(raw FD / ImageNet validation FD) over ConvNeXt, DINOv2-L, and CLIP-L",
        "normalization_fds": dict(FDR3_MODELS),
        "conditions": summaries,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for row in summaries:
        print(
            f"{row['condition']}: FDr3={float(row['fdr3']):.6f} "
            f"(ConvNeXt={float(row['convnext_fd']):.6f}, "
            f"DINOv2={float(row['dinov2_fd']):.6f}, "
            f"CLIP={float(row['clip_fd']):.6f})"
        )
    if comparisons:
        print(
            "AdvFD - static FDr3: "
            f"{comparisons['advfd_minus_static_fdr3']:+.6f}"
        )


if __name__ == "__main__":
    main()
