#!/usr/bin/env python3
"""Summarize paired v400/x400/v270 common-unique mechanism artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from train_imagenet100_sit_flow import atomic_json_dump


BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_ROOT = BASE / "fid5k_step400k_floor_audit_seed0/common_unique_x400_v270"


def default_conditions(root: Path) -> dict[str, tuple[str, str, Path]]:
    audit = BASE / "fid5k_step400k_floor_audit_seed0"
    return {
        "v400": (
            "baseline",
            "v400",
            BASE / "fid5k_static_pair_v_to_jit_x_step400000_seed0/static_s0",
        ),
        "x_orthogonal": (
            "x400",
            "orthogonal",
            audit / "orthogonal_pair/orthogonal_pair_sm1",
        ),
        "x_common_on_v": (
            "x400",
            "common",
            root / "x_common_on_v/x_common_on_v_s1",
        ),
        "x_unique_to_v": (
            "x400",
            "unique",
            root / "x_unique_to_v/x_unique_to_v_s1",
        ),
        "v_orthogonal": (
            "v270",
            "orthogonal",
            audit
            / "v270_direction_decomposition/orthogonal_pair/orthogonal_pair_sm1",
        ),
        "v_common_on_x": (
            "v270",
            "common",
            root / "v_common_on_x/v_common_on_x_s1",
        ),
        "v_unique_to_x": (
            "v270",
            "unique",
            root / "v_unique_to_x/v_unique_to_x_s1",
        ),
    }


def load_condition(
    name: str,
    family: str,
    component: str,
    directory: Path,
) -> tuple[dict[str, object], str, str]:
    result_path = directory / "fid5k_adm_results.json"
    manifest_path = directory / "sampling_manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete condition {name}: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["requested_samples"]) != 5_000:
        raise ValueError(f"{name} does not contain 5,000 requested samples")
    noise = ":".join(manifest["rank_noise_sha256"])
    labels = ":".join(manifest["rank_label_sha256"])
    row: dict[str, object] = {
        "condition": name,
        "family": family,
        "component": component,
        "formula": manifest["formula"],
        "num_samples": int(manifest["requested_samples"]),
        "total_nfe": int(manifest.get("total_nfe", 0)),
        "total_model_forwards": int(manifest.get("total_model_forwards", 0)),
        "fid": float(result["fid"]),
        "sfid": float(result["sfid"]),
        "inception_score": float(result["inception_score"]),
        "directory": str(directory.resolve()),
    }
    return row, noise, labels


def summarize(root: Path) -> pd.DataFrame:
    rows = []
    noise_fingerprints = set()
    label_fingerprints = set()
    for name, (family, component, directory) in default_conditions(root).items():
        row, noise, labels = load_condition(name, family, component, directory)
        rows.append(row)
        noise_fingerprints.add(noise)
        label_fingerprints.add(labels)
    if len(noise_fingerprints) != 1 or len(label_fingerprints) != 1:
        raise RuntimeError("conditions did not use identical initial noise and labels")
    table = pd.DataFrame(rows)
    baseline = float(table.loc[table.condition == "v400", "fid"].iloc[0])
    table["fid_gain_vs_v400"] = baseline - table.fid
    return table


def plot(table: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    labels = ["baseline", "orthogonal", "common", "unique"]
    colors = {
        "baseline": "#777777",
        "orthogonal": "#333333",
        "common": "#2f6fa3",
        "unique": "#c44e38",
    }
    baseline = table.loc[table.condition == "v400"].iloc[0]
    for axis, family, title in zip(
        axes,
        ("x400", "v270"),
        ("x400-derived direction", "v270-derived direction"),
    ):
        family_rows = table.loc[table.family == family].set_index("component")
        values = [float(baseline.fid)] + [
            float(family_rows.loc[component, "fid"])
            for component in ("orthogonal", "common", "unique")
        ]
        bars = axis.bar(
            labels,
            values,
            color=[colors[label] for label in labels],
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_title(title)
        axis.set_ylabel("ADM FID-5K (lower is better)")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Reciprocal common/unique guidance components")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    table = summarize(root)
    csv_path = root / "common_unique_fid5k.csv"
    figure_path = root / "common_unique_fid5k.png"
    table.to_csv(csv_path, index=False)
    plot(table, figure_path)
    payload = {
        "protocol": "imagenet100_sit_common_unique_summary_v1",
        "comparison_is_paired": True,
        "pairing_verified_by_noise_and_label_sha256": True,
        "rows": table.to_dict(orient="records"),
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    atomic_json_dump(payload, root / "common_unique_summary.json")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
