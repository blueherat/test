#!/usr/bin/env python3
"""Package the frozen-v800 clean-head experiment into portable Git artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-clean-head_seed0"
)
DEFAULT_FID_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_frozen_clean_head_step50000"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "docs/data/imagenet100_sit_frozen_v_clean_head_50k"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_validation_rows(metrics_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    final_time_bins: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "ema_validation" not in payload:
            continue
        step = int(payload["step"])
        for branch_key, branch in (("raw", payload["raw_validation"]), ("ema", payload["ema_validation"])):
            validation_rows.append(
                {
                    "step": step,
                    "branch": branch_key,
                    "clean_mse": float(branch["clean_mse"]),
                    "clean_derived_velocity_mse": float(
                        branch["clean_derived_velocity_mse"]
                    ),
                    "frozen_velocity_mse": float(branch["frozen_velocity_mse"]),
                }
            )
        final_time_bins = []
        for branch_key, branch in (("raw", payload["raw_validation"]), ("ema", payload["ema_validation"])):
            for time_bin in branch["time_bins"]:
                final_time_bins.append(
                    {
                        "step": step,
                        "branch": branch_key,
                        "t_min": float(time_bin["t_min"]),
                        "t_max": float(time_bin["t_max"]),
                        "count": int(time_bin["count"]),
                        "clean_mse": float(time_bin["clean_mse"]),
                        "clean_derived_velocity_mse": float(
                            time_bin["clean_derived_velocity_mse"]
                        ),
                        "frozen_velocity_mse": float(
                            time_bin["frozen_velocity_mse"]
                        ),
                    }
                )
    if not validation_rows:
        raise ValueError(f"no validation records found in {metrics_path}")
    return validation_rows, final_time_bins


def compact_fid_rows(fid_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in fid_summary["rows"]:
        rows.append(
            {
                "condition": source["condition"],
                "mode": source["mode"],
                "gamma": float(source["gamma"]),
                "num_samples": int(source["num_samples"]),
                "fid": float(source["fid"]),
                "sfid": float(source["sfid"]),
                "inception_score": float(source["inception_score"]),
                "total_nfe": int(source["total_nfe"]),
                "sampling_peak_memory_mib": int(source["sampling_peak_memory_mib"]),
                "fid_peak_memory_mib": int(source["fid_peak_memory_mib"]),
                "noise_fingerprint": source["noise_fingerprint"],
                "label_fingerprint": source["label_fingerprint"],
                "sample_sha256": source["sample_sha256"],
            }
        )
    return rows


def validate_inputs(
    run_config: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    fid_summary: dict[str, Any],
    final_checkpoint: Path,
) -> None:
    config = run_config["config"]
    fairness = run_config["fairness"]
    checkpoint = fid_summary["checkpoint"]
    if run_config["protocol"] != "imagenet100_sit_frozen_v_clean_linear_probe_v1":
        raise ValueError("unexpected training protocol")
    if fid_summary["protocol"] != "imagenet100_sit_frozen_v_clean_head_fid1k_v1":
        raise ValueError("unexpected FID protocol")
    if config["source_checkpoint_sha256"] != checkpoint["source_checkpoint_sha256"]:
        raise ValueError("training and sampling source checkpoint hashes differ")
    if sha256_file(final_checkpoint) != checkpoint["checkpoint_sha256"]:
        raise ValueError("final clean-head checkpoint hash mismatch")
    if not fairness["optimizer_contains_only_clean_head"]:
        raise ValueError("optimizer was not restricted to the clean head")
    if fairness["source_velocity_parameters_updated"]:
        raise ValueError("source velocity parameters were updated")
    frozen_losses = {float(row["frozen_velocity_mse"]) for row in validation_rows}
    if len(frozen_losses) != 1:
        raise ValueError("frozen velocity validation metric changed during training")
    if not fid_summary["comparison_is_paired"]:
        raise ValueError("FID comparison is not marked paired")
    noise = {row["noise_fingerprint"] for row in fid_summary["rows"]}
    labels = {row["label_fingerprint"] for row in fid_summary["rows"]}
    if len(noise) != 1 or len(labels) != 1:
        raise ValueError("FID conditions do not share noise and labels")


def plot_training(rows: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for branch, color in (("raw", "#4378bf"), ("ema", "#d66a32")):
        subset = [row for row in rows if row["branch"] == branch]
        axes[0].plot(
            [row["step"] for row in subset],
            [row["clean_mse"] for row in subset],
            marker="o",
            linewidth=2,
            label=branch.upper(),
            color=color,
        )
        axes[1].plot(
            [row["step"] for row in subset],
            [row["clean_derived_velocity_mse"] for row in subset],
            marker="o",
            linewidth=2,
            label=f"Clean-derived v ({branch.upper()})",
            color=color,
        )
    frozen = float(rows[0]["frozen_velocity_mse"])
    axes[1].axhline(
        frozen,
        color="#25855a",
        linestyle="--",
        linewidth=2,
        label=f"Frozen native v ({frozen:.3f})",
    )
    axes[0].set(
        xlabel="Clean-head training step",
        ylabel="Validation clean MSE",
        title="Clean prediction",
    )
    axes[1].set(
        xlabel="Clean-head training step",
        ylabel="Validation velocity MSE",
        title="Converted velocity quality",
        yscale="log",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Frozen v800 backbone: train only a 6,160-parameter clean head")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_fid(rows: list[dict[str, Any]], output_path: Path) -> None:
    baseline = next(row for row in rows if row["mode"] == "velocity")
    clean = next(row for row in rows if row["mode"] == "clean")
    sweep = [baseline, *sorted(
        (row for row in rows if row["mode"] == "extrapolation"),
        key=lambda row: float(row["gamma"]),
    )]
    gamma = [float(row["gamma"]) for row in sweep]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    specifications = (
        ("fid", "ADM FID-1K (lower is better)", float(clean["fid"])),
        ("inception_score", "Inception Score (higher is better)", float(clean["inception_score"])),
        ("total_nfe", "Total NFE across two ranks", float(clean["total_nfe"])),
    )
    for axis, (key, ylabel, clean_value) in zip(axes, specifications, strict=True):
        axis.plot(
            gamma,
            [float(row[key]) for row in sweep],
            "o-",
            color="#2d6a9f",
            linewidth=2,
            label="v + gamma (v - v_x)",
        )
        axis.axhline(
            clean_value,
            color="#c84d4d",
            linestyle="--",
            linewidth=1.8,
            label="Clean head only",
        )
        axis.set(xlabel="Extrapolation gamma", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Frozen v800 + trained clean head: paired FID-1K sweep")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_preview_montage(fid_root: Path, output_path: Path) -> None:
    conditions = (
        ("velocity", "Native v800"),
        ("clean", "Clean head"),
        ("extrap_gamma_0p1", "gamma = 0.1"),
        ("extrap_gamma_1", "gamma = 1.0"),
    )
    width, height, title_height = 480, 240, 34
    canvas = Image.new("RGB", (width * len(conditions), height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (condition, label) in enumerate(conditions):
        source = Image.open(fid_root / condition / "preview_rank_00.png").convert("RGB")
        source.thumbnail((width, height), Image.Resampling.LANCZOS)
        left = index * width + (width - source.width) // 2
        top = title_height + (height - source.height) // 2
        canvas.paste(source, (left, top))
        draw.text((index * width + 12, 10), label, fill="black")
    canvas.save(output_path, optimize=True)


def main(args: argparse.Namespace) -> None:
    train_root = args.train_root.expanduser().resolve()
    fid_root = args.fid_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = read_json(train_root / "run_config.json")
    fid_summary = read_json(fid_root / "frozen_v_clean_head_fid1k.json")
    validation_rows, final_time_bins = load_validation_rows(
        train_root / "train_metrics.jsonl"
    )
    final_checkpoint = train_root / "checkpoints/step_00050000.pt"
    validate_inputs(run_config, validation_rows, fid_summary, final_checkpoint)
    fid_rows = compact_fid_rows(fid_summary)

    write_csv(output_root / "training_validation.csv", validation_rows)
    write_csv(output_root / "final_time_bins.csv", final_time_bins)
    write_csv(output_root / "fid1k.csv", fid_rows)
    plot_training(validation_rows, output_root / "training_validation.png")
    plot_fid(fid_rows, output_root / "fid1k_sweep.png")
    build_preview_montage(fid_root, output_root / "preview_comparison.png")

    final_raw = next(
        row for row in reversed(validation_rows) if row["branch"] == "raw"
    )
    final_ema = next(
        row for row in reversed(validation_rows) if row["branch"] == "ema"
    )
    baseline = next(row for row in fid_rows if row["mode"] == "velocity")
    clean = next(row for row in fid_rows if row["mode"] == "clean")
    extrapolation = [row for row in fid_rows if row["mode"] == "extrapolation"]
    summary = {
        "protocol": "imagenet100_sit_frozen_v_clean_head_portable_summary_v1",
        "experiment": {
            "model": run_config["config"]["model_name"],
            "dataset": "ImageNet-100 cached SD-VAE latents",
            "source_step": run_config["config"]["source_step"],
            "source_weights": run_config["config"]["source_state_key"],
            "source_checkpoint_sha256": run_config["config"][
                "source_checkpoint_sha256"
            ],
            "head_step": run_config["config"]["max_steps"],
            "head_checkpoint_sha256": fid_summary["checkpoint"][
                "checkpoint_sha256"
            ],
            "total_parameter_count": run_config["parameter_count"],
            "trainable_parameter_count": run_config["trainable_parameter_count"],
            "global_batch_size": run_config["config"]["global_batch_size"],
            "learning_rate": run_config["config"]["learning_rate"],
            "ema_decay": run_config["config"]["ema_decay"],
            "clean_velocity_denominator_floor": run_config["config"][
                "clean_velocity_denominator_floor"
            ],
        },
        "audit": {
            **run_config["fairness"],
            "final_checkpoint_hash_verified": True,
            "frozen_velocity_validation_metric_constant": True,
            "fid_comparison_paired": True,
            "shared_noise_fingerprint": fid_rows[0]["noise_fingerprint"],
            "shared_label_fingerprint": fid_rows[0]["label_fingerprint"],
        },
        "final_validation": {"raw": final_raw, "ema": final_ema},
        "fid1k": {
            "num_samples": baseline["num_samples"],
            "global_seed": 0,
            "baseline": baseline,
            "clean_head_only": clean,
            "best_positive_extrapolation": min(
                extrapolation, key=lambda row: float(row["fid"])
            ),
            "all_positive_extrapolations_worse_than_baseline": all(
                float(row["fid"]) > float(baseline["fid"])
                for row in extrapolation
            ),
            "rows": fid_rows,
        },
        "source_artifact_bytes_excluded_from_git": {
            "generated_npz_count": len(list(fid_root.glob("*/samples_*.npz"))),
            "generated_npz_bytes": sum(
                path.stat().st_size for path in fid_root.glob("*/samples_*.npz")
            ),
        },
    }
    atomic_json_dump(summary, output_root / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--fid-root", type=Path, default=DEFAULT_FID_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
