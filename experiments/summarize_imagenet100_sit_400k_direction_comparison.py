#!/usr/bin/env python3
"""Summarize x-target versus same-target guidance decompositions."""

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
AUDIT = BASE / "fid5k_step400k_floor_audit_seed0"
PAIR = BASE / "fid5k_static_pair_v_to_jit_x_step400000_seed0"
OUTPUT = AUDIT / "direction_comparison_x400_v270"


def conditions(
    audit: Path,
    pair: Path,
    *,
    anchor_label: str,
    x_label: str,
    v_label: str,
) -> tuple[tuple[str, str, Path], ...]:
    return (
        ("baseline", "baseline", pair / "static_s0"),
        (x_label, "full", pair / "static_sm1"),
        (x_label, "parallel", audit / "parallel_pair/parallel_pair_sm1"),
        (x_label, "orthogonal", audit / "orthogonal_pair/orthogonal_pair_sm1"),
        (v_label, "full", audit / f"same_target_{v_label}/static_sm1"),
        (
            v_label,
            "parallel",
            audit / f"{v_label}_direction_decomposition/parallel_pair/parallel_pair_sm1",
        ),
        (
            v_label,
            "orthogonal",
            audit / f"{v_label}_direction_decomposition/orthogonal_pair/orthogonal_pair_sm1",
        ),
    )


def load_condition(family: str, component: str, directory: Path) -> dict[str, object]:
    metric_path = directory / "fid5k_adm_results.json"
    manifest_path = directory / "sampling_manifest.json"
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_nfe = manifest.get("total_nfe")
    rank_sampling_stats = manifest.get("rank_sampling_stats", [])
    if total_nfe is None and rank_sampling_stats:
        total_nfe = sum(
            int(row.get("nfe", 0)) for row in rank_sampling_stats
        )
    return {
        "family": family,
        "component": component,
        "fid": float(metric["fid"]),
        "sfid": float(metric["sfid"]),
        "inception_score": float(metric["inception_score"]),
        "num_samples": int(manifest["requested_samples"]),
        "noise_fingerprint": ":".join(manifest["rank_noise_sha256"]),
        "label_fingerprint": ":".join(manifest["rank_label_sha256"]),
        "total_nfe": int(total_nfe) if total_nfe is not None else None,
        "metric_path": str(metric_path),
        "manifest_path": str(manifest_path),
    }


def plot_summary(
    fid: pd.DataFrame,
    geometry: pd.DataFrame,
    output: Path,
    *,
    anchor_label: str,
    x_label: str,
    v_label: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    baseline = float(fid.loc[fid.family == "baseline", "fid"].iloc[0])
    bars = fid[fid.family != "baseline"].copy()
    bars["gain"] = baseline - bars.fid
    components = ("full", "parallel", "orthogonal")
    colors = {x_label: "#2864a5", v_label: "#c44e38"}
    positions = list(range(len(components)))
    width = 0.36
    for offset, family in ((-width / 2, x_label), (width / 2, v_label)):
        selected = bars.set_index(["family", "component"])
        values = [float(selected.loc[(family, component), "gain"]) for component in components]
        rectangles = axes[0].bar(
            [position + offset for position in positions],
            values,
            width=width,
            color=colors[family],
            label=family,
        )
        axes[0].bar_label(rectangles, fmt="%.2f", fontsize=8, padding=2)
    axes[0].set_xticks(positions, components)
    axes[0].set(
        title=f"Closed-loop FID gain over {anchor_label}",
        ylabel="FID improvement (higher is better)",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend()

    for context, frame in geometry.groupby("context", sort=False):
        axes[1].plot(
            frame.time,
            frame.orthogonal_cosine_mean,
            "o-",
            label=context,
        )
    axes[1].set(
        title="Alignment of the two orthogonal directions",
        xlabel="flow time t",
        ylabel="cosine",
        ylim=(0, 0.4),
    )
    axes[1].legend()

    rollout_context = (
        "anchor_rollout"
        if "anchor_rollout" in set(geometry.context)
        else f"{anchor_label}_rollout"
    )
    rollout = geometry[geometry.context == rollout_context]
    axes[2].plot(
        rollout.time,
        rollout.x_orthogonal_rms_mean,
        "o-",
        color=colors["x400"],
        label=f"{anchor_label} - {x_label}",
    )
    axes[2].plot(
        rollout.time,
        rollout.v270_orthogonal_rms_mean,
        "s--",
        color=colors["v270"],
        label=f"{anchor_label} - {v_label}",
    )
    axes[2].set(
        title=f"Orthogonal magnitude on {anchor_label} rollout",
        xlabel="flow time t",
        ylabel="per-sample RMS",
    )
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"x-target versus same-target weak-model guidance at {anchor_label}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT)
    parser.add_argument("--pair-root", type=Path, default=PAIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--anchor-label", default="v400")
    parser.add_argument("--x-label", default="x400")
    parser.add_argument("--v-label", default="v270")
    args = parser.parse_args()
    audit = args.audit_root.expanduser().resolve()
    pair = args.pair_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fid = pd.DataFrame(
        [
            load_condition(family, component, directory)
            for family, component, directory in conditions(
                audit,
                pair,
                anchor_label=args.anchor_label,
                x_label=args.x_label,
                v_label=args.v_label,
            )
        ]
    )
    if fid.num_samples.nunique() != 1 or int(fid.num_samples.iloc[0]) != 5000:
        raise RuntimeError("all direction comparisons must use exactly 5K samples")
    if fid.noise_fingerprint.nunique() != 1 or fid.label_fingerprint.nunique() != 1:
        raise RuntimeError("direction comparisons are not paired by noise and labels")
    baseline = float(fid.loc[fid.family == "baseline", "fid"].iloc[0])
    fid["fid_gain_vs_anchor"] = baseline - fid.fid
    if args.anchor_label == "v400":
        fid["fid_gain_vs_v400"] = fid["fid_gain_vs_anchor"]
    geometry_path = audit / f"direction_geometry_{args.x_label}_{args.v_label}/direction_geometry_by_time.csv"
    geometry = pd.read_csv(geometry_path)
    raw_geometry_path = (
        audit / f"direction_geometry_{args.x_label}_{args.v_label}/direction_geometry_per_sample.csv"
    )
    raw_geometry = pd.read_csv(raw_geometry_path)
    fid.to_csv(output / "direction_component_fid5k.csv", index=False)
    plot_summary(
        fid,
        geometry,
        output / "direction_comparison.png",
        anchor_label=args.anchor_label,
        x_label=args.x_label,
        v_label=args.v_label,
    )
    overall = json.loads(
        (audit / f"direction_geometry_{args.x_label}_{args.v_label}/direction_geometry_summary.json").read_text(
            encoding="utf-8"
        )
    )["overall"]
    parallel_comparison = []
    for context, frame in raw_geometry.groupby("context", sort=False):
        left = frame.x_parallel_coefficient
        right = frame.v270_parallel_coefficient
        parallel_comparison.append(
            {
                "context": str(context),
                "coefficient_correlation": float(left.corr(right)),
                "same_sign_fraction": float(((left * right) > 0).mean()),
                "x_parallel_rms_mean": float(frame.x_parallel_rms.mean()),
                "v270_parallel_rms_mean": float(frame.v270_parallel_rms.mean()),
            }
        )
    payload = {
        "protocol": "imagenet100_sit_direction_comparison_v2",
        "anchor_label": args.anchor_label,
        "x_label": args.x_label,
        "v_label": args.v_label,
        "comparison_is_paired": True,
        "pairing_verified_by_noise_and_label_sha256": True,
        "baseline_fid": baseline,
        "fid_rows": fid.to_dict(orient="records"),
        "direction_geometry_overall": overall,
        "parallel_component_comparison": parallel_comparison,
        "geometry_by_time_csv": str(geometry_path),
        "geometry_per_sample_csv": str(raw_geometry_path),
        "fid_csv": str(output / "direction_component_fid5k.csv"),
        "figure": str(output / "direction_comparison.png"),
        "caveat": (
            "FID gains from full, parallel, and orthogonal closed-loop rollouts are not "
            "additive; each intervention induces its own state trajectory."
        ),
    }
    atomic_json_dump(payload, output / "direction_comparison_summary.json")
    print(
        fid[["family", "component", "fid", "fid_gain_vs_anchor", "sfid", "inception_score"]]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
