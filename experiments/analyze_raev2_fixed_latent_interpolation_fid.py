"""Compute matched-sample FID for fixed-latent interpolation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.run_raev2_decoded_distribution_audit import (
    feature_statistics,
    fid_between_statistics,
    load_reference_statistics,
)
from experiments.run_raev2_fixed_latent_interpolation_audit import (
    _merge_line_features,
)
from experiments.run_raev2_scale_response_study import scale_key


PROTOCOL = "raev2_fixed_latent_interpolation_fid_v1"


def compute_fid_comparison(
    *,
    input_dir: Path,
    interpolation_dir: Path,
    reference_path: Path,
) -> pd.DataFrame:
    source_manifest = json.loads(
        (input_dir / "manifest.json").read_text(encoding="utf-8")
    )
    audit_manifest = json.loads(
        (interpolation_dir / "summary.json").read_text(encoding="utf-8")
    )
    samples = int(audit_manifest["samples"])
    world_size = int(source_manifest["world_size"])
    if samples > int(source_manifest["samples"]):
        raise RuntimeError("interpolation sample count exceeds the source artifact")
    actual_metrics = pd.read_csv(input_dir / "scale_response_metrics.csv")
    actual_metrics = actual_metrics[actual_metrics.space.eq("decoded_inception")]
    reference = load_reference_statistics(reference_path, "2048")
    rows = []
    for scale in [float(value) for value in audit_manifest["scales"]]:
        condition = scale_key(scale)
        line = _merge_line_features(
            interpolation_dir,
            "line_inception",
            condition,
            samples=samples,
            world_size=world_size,
        ).astype(np.float32, copy=False)
        line_stats = feature_statistics(line)
        line_fid = fid_between_statistics(line_stats, reference)
        actual = actual_metrics[np.isclose(actual_metrics.scale, scale)]
        if len(actual) != 1:
            raise RuntimeError(f"missing actual FID at scale {scale}")
        actual_fid = float(actual.iloc[0].fid_to_official)
        rows.append(
            {
                "scale": scale,
                "sample_count": samples,
                "actual_scale_trajectory_fid": actual_fid,
                "fixed_endpoint_chord_fid": line_fid,
                "fixed_minus_actual_fid": line_fid - actual_fid,
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.plot(
        frame.scale,
        frame.actual_scale_trajectory_fid,
        marker="o",
        linewidth=2,
        label="actual recursive scale trajectory",
    )
    axis.plot(
        frame.scale,
        frame.fixed_endpoint_chord_fid,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="fixed z1-to-z1.78 endpoint chord",
    )
    axis.set_xlabel("Internal-guidance scale")
    axis.set_ylabel("5k FID to official ImageNet reference (lower is better)")
    axis.set_title("Actual scale trajectory vs fixed-latent chord")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--interpolation-dir", type=Path, required=True)
    parser.add_argument(
        "--fid-reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    interpolation_dir = args.interpolation_dir.expanduser().resolve()
    frame = compute_fid_comparison(
        input_dir=input_dir,
        interpolation_dir=interpolation_dir,
        reference_path=args.fid_reference.expanduser().resolve(),
    )
    frame.to_csv(interpolation_dir / "fid_comparison.csv", index=False)
    plot_comparison(frame, interpolation_dir / "fid_comparison.png")
    summary = json.loads(
        (interpolation_dir / "summary.json").read_text(encoding="utf-8")
    )
    anchor = float(summary["anchor_scale"])
    anchor_row = frame[np.isclose(frame.scale, anchor)]
    payload = {
        "protocol": PROTOCOL,
        "input_artifact": str(input_dir),
        "interpolation_artifact": str(interpolation_dir),
        "samples": int(frame.sample_count.iloc[0]),
        "fid_reference": str(args.fid_reference.expanduser().resolve()),
        "anchor_fid_recompute_absolute_difference": float(
            anchor_row.fixed_minus_actual_fid.abs().iloc[0]
        ),
        "measurement_guardrail": (
            "Both paths use the same sample count and official feature reference. "
            "A 5k FID is still a finite-sample diagnostic, not the paper's 50k FID."
        ),
    }
    (interpolation_dir / "fid_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
