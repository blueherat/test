#!/usr/bin/env python3
"""Summarize paired common/unique SiT guidance artifacts."""

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


def default_conditions(
    root: Path,
    *,
    audit: Path | None = None,
    pair: Path | None = None,
    anchor_label: str = "v400",
    x_label: str = "x400",
    v_label: str = "v270",
) -> dict[str, tuple[str, str, Path]]:
    audit = audit or BASE / "fid5k_step400k_floor_audit_seed0"
    pair = pair or BASE / "fid5k_static_pair_v_to_jit_x_step400000_seed0"
    return {
        anchor_label: (
            "baseline",
            anchor_label,
            pair / "static_s0",
        ),
        "x_orthogonal": (
            x_label,
            "orthogonal",
            audit / "orthogonal_pair/orthogonal_pair_sm1",
        ),
        "x_common_on_v": (
            x_label,
            "common",
            root / "x_common_on_v/x_common_on_v_s1",
        ),
        "x_unique_to_v": (
            x_label,
            "unique",
            root / "x_unique_to_v/x_unique_to_v_s1",
        ),
        "v_orthogonal": (
            v_label,
            "orthogonal",
            audit
            / f"{v_label}_direction_decomposition/orthogonal_pair/orthogonal_pair_sm1",
        ),
        "v_common_on_x": (
            v_label,
            "common",
            root / "v_common_on_x/v_common_on_x_s1",
        ),
        "v_unique_to_x": (
            v_label,
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


def summarize(
    root: Path,
    *,
    audit: Path | None = None,
    pair: Path | None = None,
    anchor_label: str = "v400",
    x_label: str = "x400",
    v_label: str = "v270",
) -> pd.DataFrame:
    rows = []
    noise_fingerprints = set()
    label_fingerprints = set()
    conditions = default_conditions(
        root,
        audit=audit,
        pair=pair,
        anchor_label=anchor_label,
        x_label=x_label,
        v_label=v_label,
    )
    for name, (family, component, directory) in conditions.items():
        row, noise, labels = load_condition(name, family, component, directory)
        rows.append(row)
        noise_fingerprints.add(noise)
        label_fingerprints.add(labels)
    if len(noise_fingerprints) != 1 or len(label_fingerprints) != 1:
        raise RuntimeError("conditions did not use identical initial noise and labels")
    table = pd.DataFrame(rows)
    baseline = float(table.loc[table.condition == anchor_label, "fid"].iloc[0])
    table["fid_gain_vs_anchor"] = baseline - table.fid
    if anchor_label == "v400":
        table["fid_gain_vs_v400"] = table["fid_gain_vs_anchor"]
    return table


def plot(
    table: pd.DataFrame,
    output: Path,
    *,
    anchor_label: str = "v400",
    x_label: str = "x400",
    v_label: str = "v270",
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    labels = ["baseline", "orthogonal", "common", "unique"]
    colors = {
        "baseline": "#777777",
        "orthogonal": "#333333",
        "common": "#2f6fa3",
        "unique": "#c44e38",
    }
    baseline = table.loc[table.condition == anchor_label].iloc[0]
    for axis, family, title in zip(
        axes,
        (x_label, v_label),
        (f"{x_label}-derived direction", f"{v_label}-derived direction"),
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
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--pair-root", type=Path)
    parser.add_argument("--anchor-label", default="v400")
    parser.add_argument("--x-label", default="x400")
    parser.add_argument("--v-label", default="v270")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_root = args.audit_root.expanduser().resolve() if args.audit_root else None
    pair_root = args.pair_root.expanduser().resolve() if args.pair_root else None
    table = summarize(
        root,
        audit=audit_root,
        pair=pair_root,
        anchor_label=args.anchor_label,
        x_label=args.x_label,
        v_label=args.v_label,
    )
    csv_path = root / "common_unique_fid5k.csv"
    figure_path = root / "common_unique_fid5k.png"
    table.to_csv(csv_path, index=False)
    plot(
        table,
        figure_path,
        anchor_label=args.anchor_label,
        x_label=args.x_label,
        v_label=args.v_label,
    )
    payload = {
        "protocol": "imagenet100_sit_common_unique_summary_v1",
        "comparison_is_paired": True,
        "pairing_verified_by_noise_and_label_sha256": True,
        "anchor_label": args.anchor_label,
        "x_label": args.x_label,
        "v_label": args.v_label,
        "rows": table.to_dict(orient="records"),
        "csv": str(csv_path),
        "figure": str(figure_path),
    }
    atomic_json_dump(payload, root / "common_unique_summary.json")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
