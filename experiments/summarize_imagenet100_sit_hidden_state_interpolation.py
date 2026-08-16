#!/usr/bin/env python3
"""Package the frozen-v800 hidden-state interpolation experiment for Git."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_hidden_state_depth8_interpolation_ema"
)
DEFAULT_FINE_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_hidden_state_depth8_interpolation_fine_paired_ema"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "docs/data/imagenet100_sit_hidden_state_extrapolation"
)
EXPECTED_PROTOCOL = "imagenet100_sit_hidden_state_extrapolation_fid1k_v1"
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: list[dict[str, Any]], output_path: Path) -> None:
    baseline = next(float(row["fid"]) for row in rows if row["condition"] == "final")
    treatments = [row for row in rows if row["mode"] == "interpolation"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for space, color in (("hidden", "#4c72b0"), ("output", "#c44e52")):
        selected = sorted(
            (row for row in treatments if row["extrapolation_space"] == space),
            key=lambda row: float(row["alpha"]),
        )
        alpha = [float(row["alpha"]) for row in selected]
        fid = [float(row["fid"]) for row in selected]
        sfid = [float(row["sfid"]) for row in selected]
        score = [float(row["inception_score"]) for row in selected]
        local = [index for index, value in enumerate(alpha) if value <= 0.1]
        axes[0, 0].plot(alpha, fid, marker="o", label=space, color=color)
        axes[0, 1].plot(
            [alpha[index] for index in local],
            [fid[index] for index in local],
            marker="o",
            label=space,
            color=color,
        )
        axes[1, 0].plot(
            [alpha[index] for index in local],
            [sfid[index] for index in local],
            marker="o",
            label=space,
            color=color,
        )
        axes[1, 1].plot(
            [alpha[index] for index in local],
            [score[index] for index in local],
            marker="o",
            label=space,
            color=color,
        )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    for axis in (axes[0, 0], axes[0, 1]):
        axis.axhline(baseline, color="black", linestyle="--", linewidth=1)
        axis.set_ylabel("FID-1K (lower is better)")
    axes[0, 0].set_title("Full interpolation sweep")
    axes[0, 0].set_xlabel("alpha")
    axes[0, 1].set_title("Local region")
    axes[0, 1].set_xlabel("alpha (<= 0.1)")
    axes[1, 0].set_xlabel("alpha (<= 0.1)")
    axes[1, 0].set_ylabel("sFID-1K (lower is better)")
    axes[1, 1].set_xlabel("alpha (<= 0.1)")
    axes[1, 1].set_ylabel("Inception Score (higher is better)")
    fig.suptitle("Frozen v800: depth-8 to final hidden-state interpolation")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_preview(main_root: Path, fine_root: Path, output_path: Path) -> None:
    conditions = (
        (main_root / "final", "v800 baseline"),
        (fine_root / "hidden_alpha_0p0175", "hidden alpha=0.0175"),
        (main_root / "hidden_alpha_0p05", "hidden alpha=0.05"),
        (main_root / "hidden_alpha_0p1", "hidden alpha=0.1"),
        (main_root / "internal_depth8", "source FinalLayer(h8)"),
    )
    panels: list[tuple[str, Image.Image]] = []
    for directory, title in conditions:
        path = directory / "preview_rank_00.png"
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
    main_root = args.main_root.expanduser().resolve()
    fine_root = args.fine_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    main_summary = read_json(main_root / "hidden_state_extrapolation_fid1k.json")
    fine_summary = read_json(fine_root / "hidden_state_extrapolation_fid1k.json")
    for summary in (main_summary, fine_summary):
        if summary.get("protocol") != EXPECTED_PROTOCOL:
            raise ValueError("unexpected interpolation summary protocol")
        if summary["checkpoint"]["checkpoint_sha256"] != EXPECTED_SOURCE_SHA256:
            raise ValueError("source v800 checkpoint SHA256 changed")
    main_rows = main_summary["rows"]
    fine_rows = fine_summary["rows"]
    if len(main_rows) != 21 or len(fine_rows) != 5:
        raise ValueError("expected 21 main rows and 5 completed fine-sweep rows")
    rows = [*main_rows, *fine_rows]
    if len({row["condition"] for row in rows}) != len(rows):
        raise ValueError("interpolation conditions are not unique")
    fingerprints = {
        (row["noise_fingerprint"], row["label_fingerprint"]) for row in rows
    }
    if len(fingerprints) != 1:
        raise ValueError("interpolation conditions are not paired")
    baseline = next(row for row in rows if row["condition"] == "final")
    if baseline["sample_sha256"] != EXPECTED_BASELINE_SAMPLE_SHA256:
        raise ValueError("baseline sample SHA256 does not match prior v800")
    hidden_rows = [
        row
        for row in rows
        if row["mode"] == "interpolation"
        and row["extrapolation_space"] == "hidden"
    ]
    output_rows = [
        row
        for row in rows
        if row["mode"] == "interpolation"
        and row["extrapolation_space"] == "output"
    ]
    best_hidden = min(hidden_rows, key=lambda row: float(row["fid"]))
    best_output = min(output_rows, key=lambda row: float(row["fid"]))
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "interpolation_fid1k.csv", rows)
    plot_sweep(rows, output_root / "interpolation_sweep.png")
    make_preview(main_root, fine_root, output_root / "interpolation_preview.png")
    portable = {
        "protocol": "imagenet100_sit_hidden_state_interpolation_portable_v1",
        "experiment": {
            "source_checkpoint": main_summary["checkpoint"],
            "internal_depth": 8,
            "source_block_count": 12,
            "trainable_parameters": 0,
            "hidden_formula": "FinalLayer((1-alpha)*h12 + alpha*h8, c)",
            "output_formula": (
                "(1-alpha)*FinalLayer(h12,c) + alpha*FinalLayer(h8,c)"
            ),
            "sampler": "Dopri5, FP32/TF32, CFG=1, two GPUs",
            "metric": "paired ADM FID/sFID/IS on 1000 samples",
        },
        "audit": {
            "condition_count": len(rows),
            "comparison_is_paired": True,
            "shared_noise_and_labels": True,
            "baseline_matches_prior_v800_byte_for_byte": True,
            "baseline_sample_sha256": baseline["sample_sha256"],
            "sampling_peak_memory_mib": max(
                int(row["sampling_peak_memory_mib"]) for row in rows
            ),
            "fid_peak_memory_mib": max(
                int(row["fid_peak_memory_mib"]) for row in rows
            ),
        },
        "key_results": {
            "baseline": baseline,
            "best_hidden_interpolation": best_hidden,
            "best_hidden_fid_delta": float(best_hidden["fid"])
            - float(baseline["fid"]),
            "best_output_interpolation": best_output,
            "best_output_fid_delta": float(best_output["fid"])
            - float(baseline["fid"]),
            "fid1k_signal_is_small_and_not_5k_validated": True,
        },
        "rows": rows,
        "excluded_local_artifacts": {
            "reason": "generated samples, logs, and checkpoint stay on the data disk"
        },
    }
    write_json(output_root / "interpolation_summary.json", portable)
    print(json.dumps(portable["key_results"], indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--fine-root", type=Path, default=DEFAULT_FINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
