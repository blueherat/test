from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


STEP_PATTERN = re.compile(r"evaluation_step(?P<step>\d+)_paired5k\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_training_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["step"]): row for row in csv.DictReader(handle)}


def paired_fid(payload: dict[str, Any]) -> tuple[float, str]:
    if "fid_inception_2048_float64" in payload:
        return float(payload["fid_inception_2048_float64"]), "cpu_float64"
    return float(payload["fid_inception_2048_small_sample"]), "gpu_float32"


def evaluation_rows(root: Path) -> list[dict[str, Any]]:
    base_path = root / "base" / "evaluation_paired5k.json"
    base_payload = read_json(base_path)
    base_fid, base_fid_numeric = paired_fid(base_payload)
    rows: list[dict[str, Any]] = [
        {
            "variant": "base",
            "step": 0,
            "fid_5k": base_fid,
            "fid_numeric": base_fid_numeric,
            "delta_vs_base": 0.0,
            "static_fd_train": "",
            "adaptive_fd_train": "",
            "adaptive_weight_train": "",
            "critic_fd_train": "",
            "adaptive_heldout_fd": "",
            "balanced_eval_labels": base_payload.get("balanced_eval_labels"),
            "per_sample_eval_noise": base_payload.get("per_sample_eval_noise"),
            "quantize_eval_images": base_payload.get("quantize_eval_images"),
        }
    ]
    for variant in ("static", "real"):
        training = read_training_rows(root / variant / "train_metrics.csv")
        for path in sorted((root / variant).glob("evaluation_step*_paired5k.json")):
            match = STEP_PATTERN.match(path.name)
            if match is None:
                continue
            step = int(match.group("step"))
            payload = read_json(path)
            metrics = training.get(step, {})
            fid, fid_numeric = paired_fid(payload)
            rows.append(
                {
                    "variant": variant,
                    "step": step,
                    "fid_5k": fid,
                    "fid_numeric": fid_numeric,
                    "delta_vs_base": fid - base_fid,
                    "static_fd_train": metrics.get("static_fd", ""),
                    "adaptive_fd_train": metrics.get("adaptive_fd", ""),
                    "adaptive_weight_train": metrics.get("adaptive_weight", ""),
                    "critic_fd_train": metrics.get("critic_fd", ""),
                    "adaptive_heldout_fd": payload.get("adaptive_heldout_fd", ""),
                    "balanced_eval_labels": payload.get("balanced_eval_labels"),
                    "per_sample_eval_noise": payload.get("per_sample_eval_noise"),
                    "quantize_eval_images": payload.get("quantize_eval_images"),
                }
            )
    return sorted(rows, key=lambda row: (str(row["variant"]), int(row["step"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    base_fid = next(float(row["fid_5k"]) for row in rows if row["variant"] == "base")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.axhline(base_fid, color="black", linestyle="--", linewidth=1.2, label="base")
    for variant, color in (("static", "#1f77b4"), ("real", "#d62728")):
        selected = [row for row in rows if row["variant"] == variant]
        axis.plot(
            [int(row["step"]) for row in selected],
            [float(row["fid_5k"]) for row in selected],
            marker="o",
            color=color,
            label=variant,
        )
    axis.set_xlabel("Generator step")
    axis.set_ylabel("FID-5K")
    axis.set_title("pMF-B paper-only prefix: paired official-style evaluation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    rows = evaluation_rows(root)
    if len(rows) < 2:
        raise RuntimeError("No paired checkpoint evaluations found")
    csv_path = root / "paired_fid5k.csv"
    json_path = root / "paired_fid5k.json"
    plot_path = root / "paired_fid5k.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_plot(plot_path, rows)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"saved {csv_path}\n{json_path}\n{plot_path}", flush=True)


if __name__ == "__main__":
    main()
