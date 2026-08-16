#!/usr/bin/env python3
"""Package the frozen-v800 hidden-state extrapolation experiment for Git."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_hidden_state_depth8_ema"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "docs/data/imagenet100_sit_hidden_state_extrapolation"
)
EXPECTED_BASELINE_SAMPLE_SHA256 = (
    "9d1caa72fe8ad1776c978131ea42d44ecaf839ce1da4cdb1374bb2febcd44a3c"
)
EXPECTED_SOURCE_SHA256 = (
    "b7f7d7318ee4b480fe591bc451c2aceb09efaf4111e57fd9dbefcd1bcfd88caa"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_fid_sweep(rows: list[dict[str, Any]], output_path: Path) -> None:
    baseline = next(float(row["fid"]) for row in rows if row["condition"] == "final")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for column, space in enumerate(("hidden", "output")):
        selected = sorted(
            (
                row
                for row in rows
                if row["mode"] == "extrapolation"
                and row["extrapolation_space"] == space
            ),
            key=lambda row: float(row["gamma"]),
        )
        gammas = [float(row["gamma"]) for row in selected]
        fids = [float(row["fid"]) for row in selected]
        scores = [float(row["inception_score"]) for row in selected]
        axes[0, column].plot(gammas, fids, marker="o", linewidth=1.8)
        axes[0, column].axhline(
            baseline,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"v800 baseline {baseline:.2f}",
        )
        axes[0, column].set_xscale("log")
        axes[0, column].set_title(f"{space}-space extrapolation")
        axes[0, column].set_ylabel("FID-1K (lower is better)")
        axes[0, column].legend(frameon=False)
        axes[1, column].plot(gammas, scores, marker="o", linewidth=1.8)
        axes[1, column].set_xscale("log")
        axes[1, column].set_xlabel("gamma (log scale)")
        axes[1, column].set_ylabel("Inception Score (higher is better)")
        for axis in (axes[0, column], axes[1, column]):
            axis.grid(alpha=0.25)
    fig.suptitle("Frozen v800: depth-8 to final hidden-state extrapolation")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_gap_audit(rows: list[dict[str, Any]], output_path: Path) -> None:
    times = [float(row["time"]) for row in rows]
    cosines = [float(row["raw_trained_gap_cosine_mean"]) for row in rows]
    ratios = [float(row["trained_gap_over_raw_gap_rms"]) for row in rows]
    hidden_cosines = [float(row["hidden_cosine_mean"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    axes[0].plot(times, cosines, marker="o", label="raw vs trained gap")
    axes[0].plot(times, hidden_cosines, marker="s", label="h8 vs h12")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("mean cosine")
    axes[0].legend(frameon=False)
    axes[1].plot(times, ratios, marker="o", color="#c44e52")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("trained-gap RMS / raw-gap RMS")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Raw source readout and trained auxiliary head are different fields")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_preview_comparison(input_root: Path, output_path: Path) -> None:
    conditions = (
        ("final", "v800 baseline"),
        ("hidden_gamma_0p005", "hidden gamma=0.005"),
        ("hidden_gamma_0p1", "hidden gamma=0.1"),
        ("hidden_gamma_0p4", "hidden gamma=0.4"),
        ("internal_depth8", "source FinalLayer(h8)"),
    )
    panels: list[tuple[str, Image.Image]] = []
    for condition, title in conditions:
        path = input_root / condition / "preview_rank_00.png"
        if not path.is_file():
            raise FileNotFoundError(f"missing preview: {path}")
        image = Image.open(path).convert("RGB")
        target_width = 780
        target_height = round(image.height * target_width / image.width)
        panels.append((title, image.resize((target_width, target_height))))
    title_height = 34
    gap = 8
    width = panels[0][1].width
    height = sum(image.height + title_height for _, image in panels) + gap * (
        len(panels) - 1
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    cursor = 0
    for title, image in panels:
        draw.text((10, cursor + 8), title, fill="black")
        cursor += title_height
        canvas.paste(image, (0, cursor))
        cursor += image.height + gap
    canvas.save(output_path, optimize=True)


def main(args: argparse.Namespace) -> None:
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    fid_summary = read_json(input_root / "hidden_state_extrapolation_fid1k.json")
    gap_summary = read_json(input_root / "hidden_state_gap_audit.json")
    if fid_summary.get("protocol") != "imagenet100_sit_hidden_state_extrapolation_fid1k_v1":
        raise ValueError("unexpected FID summary protocol")
    if gap_summary.get("protocol") != "imagenet100_sit_hidden_state_gap_audit_v1":
        raise ValueError("unexpected gap-audit protocol")
    rows = fid_summary["rows"]
    gap_rows = gap_summary["rows"]
    if len(rows) != 23 or len({row["condition"] for row in rows}) != 23:
        raise ValueError("expected 23 unique paired FID conditions")
    if len(
        {(row["noise_fingerprint"], row["label_fingerprint"]) for row in rows}
    ) != 1:
        raise ValueError("FID conditions are not paired")
    baseline = next(row for row in rows if row["condition"] == "final")
    if baseline["sample_sha256"] != EXPECTED_BASELINE_SAMPLE_SHA256:
        raise ValueError("baseline sample SHA256 does not match prior v800")
    checkpoint = fid_summary["checkpoint"]
    if checkpoint["checkpoint_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("source v800 checkpoint SHA256 changed")

    hidden_rows = [
        row
        for row in rows
        if row["mode"] == "extrapolation"
        and row["extrapolation_space"] == "hidden"
    ]
    output_rows = [
        row
        for row in rows
        if row["mode"] == "extrapolation"
        and row["extrapolation_space"] == "output"
    ]
    best_hidden = min(hidden_rows, key=lambda row: float(row["fid"]))
    best_output = min(output_rows, key=lambda row: float(row["fid"]))
    internal = next(row for row in rows if row["condition"] == "internal_depth8")
    sample_files = list(input_root.glob("*/samples_unguided_n1000.npz"))
    sample_bytes = sum(path.stat().st_size for path in sample_files)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "fid1k.csv", rows)
    write_csv(output_root / "hidden_state_gap_audit.csv", gap_rows)
    plot_fid_sweep(rows, output_root / "fid1k_sweep.png")
    plot_gap_audit(gap_rows, output_root / "hidden_state_gap_audit.png")
    make_preview_comparison(input_root, output_root / "preview_comparison.png")
    portable = {
        "protocol": "imagenet100_sit_hidden_state_extrapolation_portable_summary_v1",
        "experiment": {
            "model": checkpoint["model_name"],
            "dataset": "ImageNet-100 cached SD-VAE latents",
            "source_step": checkpoint["step"],
            "source_weights": checkpoint["weights"],
            "source_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "internal_depth": 8,
            "source_block_count": 12,
            "trainable_parameters": 0,
            "hidden_formula": "FinalLayer(h12 + gamma * (h12 - h8), c)",
            "output_control_formula": (
                "FinalLayer(h12,c) + gamma * "
                "(FinalLayer(h12,c) - FinalLayer(h8,c))"
            ),
            "sampler": "Dopri5, FP32/TF32, CFG=1",
            "metric": "paired ADM FID/sFID/IS on 1000 samples",
        },
        "audit": {
            "comparison_is_paired": True,
            "condition_count": len(rows),
            "shared_noise_and_labels": True,
            "gamma_zero_matches_prior_v800_byte_for_byte": True,
            "baseline_sample_sha256": baseline["sample_sha256"],
            "sampling_peak_memory_mib": max(
                int(row["sampling_peak_memory_mib"]) for row in rows
            ),
            "fid_peak_memory_mib": max(int(row["fid_peak_memory_mib"]) for row in rows),
        },
        "key_results": {
            "baseline": baseline,
            "internal_depth8": internal,
            "best_hidden_extrapolation": best_hidden,
            "best_output_extrapolation": best_output,
            "all_positive_hidden_gammas_worse_than_baseline": all(
                float(row["fid"]) > float(baseline["fid"]) for row in hidden_rows
            ),
            "all_positive_output_gammas_worse_than_baseline": all(
                float(row["fid"]) > float(baseline["fid"]) for row in output_rows
            ),
        },
        "gap_audit": {
            "protocol": gap_summary["protocol"],
            "samples": gap_summary["samples"],
            "seed": gap_summary["seed"],
            "definitions": gap_summary["definitions"],
            "rows": gap_rows,
        },
        "rows": rows,
        "excluded_local_artifacts": {
            "sample_npz_count": len(sample_files),
            "sample_npz_bytes": sample_bytes,
            "reason": "generated samples and checkpoints remain on the local data disk",
        },
    }
    write_json(output_root / "summary.json", portable)
    print(json.dumps(portable["key_results"], indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
