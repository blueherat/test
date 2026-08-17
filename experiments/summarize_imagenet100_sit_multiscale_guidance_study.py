#!/usr/bin/env python3
"""Build a compact, auditable report bundle for the multiscale guidance study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_ORDER = ("early", "mid", "late")
BAND_ORDER = ("low", "mid", "high")


DISPLAY_NAMES = {
    "screen_baseline_adaptive": "Baseline (Dopri5)",
    "screen_full_depth8_v": "Full depth-8 gap",
    "screen_full_external_v500": "Full v800-v500 gap",
    "screen_full_depth12_x": "Full depth-12 x gap",
    "screen_depth_schedule_native_coarse_to_fine": "Depth 4->8->10",
    "screen_depth_schedule_native_fine_to_coarse": "Depth 10->8->4",
    "screen_depth_schedule_rms_coarse_to_fine": "Depth 4->8->10 (RMS)",
    "screen_depth_schedule_rms_fine_to_coarse": "Depth 10->8->4 (RMS)",
    "screen_spectral_router_native": "Spectral router",
    "screen_spectral_anti_router_native": "Anti-router",
    "screen_spectral_router_rms": "Spectral router (RMS)",
    "screen_spectral_anti_router_rms": "Anti-router (RMS)",
    "screen_euler_baseline": "Euler baseline",
    "screen_euler_depth8_gamma0p4": "Euler + depth-8 gap",
    "screen_spectral_delay_native_gamma0p8": "Spectral delay, gamma=0.8",
    "screen_spectral_delay_rms_gamma0p4": "Spectral delay, RMS, gamma=0.4",
    "screen_raw_compute_rms_fine_to_coarse": "Raw-depth proxy (best RMS)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_series(series: pd.Series) -> bool:
    return bool(series.map(lambda value: math.isfinite(float(value))).all())


def validate_study(root: Path, metrics: pd.DataFrame) -> dict[str, object]:
    required_columns = {
        "name",
        "group",
        "hypothesis_id",
        "kind",
        "num_samples",
        "fid",
        "sfid",
        "inception_score",
        "noise_sha256",
        "label_sha256",
        "total_nfe",
        "elapsed_seconds",
    }
    missing = sorted(required_columns - set(metrics.columns))
    if missing:
        raise ValueError(f"condition table is missing columns: {missing}")
    if not metrics["name"].is_unique:
        raise ValueError("condition names are not unique")
    for column in ("fid", "sfid", "inception_score", "total_nfe", "elapsed_seconds"):
        if not finite_series(metrics[column]):
            raise ValueError(f"column {column!r} contains non-finite values")

    fingerprints = metrics[["noise_sha256", "label_sha256"]].drop_duplicates()
    if len(fingerprints) != 1:
        raise ValueError("screen conditions do not share one paired noise/label fingerprint")
    sample_counts = sorted(int(value) for value in metrics["num_samples"].unique())
    if sample_counts != [1000]:
        raise ValueError(f"unexpected sample counts: {sample_counts}")

    result_paths = sorted((root / "evaluations").glob("*/condition_result.json"))
    if len(result_paths) != len(metrics):
        raise ValueError(
            f"found {len(result_paths)} condition results for {len(metrics)} table rows"
        )
    pipeline = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
    stage_status = Counter(stage.get("status") for stage in pipeline["stages"].values())
    if stage_status != {"complete": len(pipeline["stages"])}:
        raise ValueError(f"pipeline has unfinished stages: {dict(stage_status)}")

    manifests = [
        json.loads(path.read_text(encoding="utf-8"))["sampling_manifest"]
        for path in result_paths
    ]
    max_allocated = max(int(manifest["max_memory_allocated_bytes"]) for manifest in manifests)
    max_reserved = max(int(manifest["max_memory_reserved_bytes"]) for manifest in manifests)
    baseline_manifest = json.loads(
        (
            root
            / "evaluations"
            / "screen_baseline_adaptive"
            / "condition_result.json"
        ).read_text(encoding="utf-8")
    )["sampling_manifest"]
    label_histogram = [int(value) for value in baseline_manifest["label_histogram"]]

    fingerprint = fingerprints.iloc[0]
    return {
        "format": "eqvae_imagenet100_sit_multiscale_portable_audit_v1",
        "study_root": str(root),
        "condition_count": int(len(metrics)),
        "unique_condition_count": int(metrics["name"].nunique()),
        "condition_result_count": int(len(result_paths)),
        "sample_counts": sample_counts,
        "paired_noise_sha256": str(fingerprint["noise_sha256"]),
        "paired_label_sha256": str(fingerprint["label_sha256"]),
        "pipeline_stage_count": int(len(pipeline["stages"])),
        "pipeline_stage_status": dict(stage_status),
        "pipeline_finished_at": pipeline.get("finished_at"),
        "retained_sample_npz_count": len(list(root.rglob("*.npz"))),
        "label_class_count": len(label_histogram),
        "label_count_min": min(label_histogram),
        "label_count_max": max(label_histogram),
        "max_sampling_memory_allocated_gib": max_allocated / 2**30,
        "max_sampling_memory_reserved_gib": max_reserved / 2**30,
    }


def add_baseline_deltas(metrics: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    baseline_rows = metrics.loc[metrics["name"].eq("screen_baseline_adaptive")]
    if len(baseline_rows) != 1:
        raise ValueError("exactly one adaptive baseline is required")
    baseline = float(baseline_rows.iloc[0]["fid"])
    euler_rows = metrics.loc[metrics["name"].eq("screen_euler_baseline")]
    if len(euler_rows) != 1:
        raise ValueError("exactly one fixed-Euler baseline is required")
    euler_baseline = float(euler_rows.iloc[0]["fid"])
    result = metrics.copy()
    result["fid_gain_vs_adaptive_baseline"] = baseline - result["fid"]
    fixed_euler = result["integrator"].eq("fixed_euler")
    result["matched_baseline_name"] = np.where(
        fixed_euler,
        "screen_euler_baseline",
        "screen_baseline_adaptive",
    )
    result["matched_baseline_fid"] = np.where(fixed_euler, euler_baseline, baseline)
    result["fid_gain_vs_integrator_baseline"] = (
        result["matched_baseline_fid"] - result["fid"]
    )
    return result, baseline, euler_baseline


def write_tables(root: Path, output: Path, metrics: pd.DataFrame) -> dict[str, Path]:
    tables: dict[str, Path] = {}

    named = metrics.loc[
        metrics["hypothesis_id"].isin(
            {
                "baseline",
                "successful_failed_control",
                "idea1_time_varying_depth",
                "idea2_spectral_routing",
                "idea3_spectral_delay",
                "idea4_unresolved_computation",
            }
        )
    ].copy()
    named.insert(1, "display_name", named["name"].map(DISPLAY_NAMES).fillna(named["name"]))
    named = named.sort_values("fid")
    tables["named_methods"] = output / "named_method_comparison.csv"
    named.to_csv(tables["named_methods"], index=False)

    causal_map = metrics.loc[metrics["kind"].eq("band_time")].copy()
    causal_map = causal_map.sort_values(["provider", "amplitude", "interval", "band"])
    tables["causal_map"] = output / "causal_map_fid1k.csv"
    causal_map.to_csv(tables["causal_map"], index=False)

    causal_order = metrics.loc[metrics["kind"].eq("ordered_bands")].copy()
    causal_order = causal_order.sort_values(["provider", "amplitude", "order"])
    tables["causal_order"] = output / "causal_order_fid1k.csv"
    causal_order.to_csv(tables["causal_order"], index=False)

    router_rows: list[dict[str, object]] = []
    for condition in (
        "screen_spectral_router_native",
        "screen_spectral_router_rms",
        "screen_spectral_anti_router_native",
        "screen_spectral_anti_router_rms",
    ):
        path = root / "evaluations" / condition / "condition_result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = payload["sampling_manifest"]["router_counts"]
        total = sum(int(value) for value in counts.values())
        for depth, count in sorted(counts.items(), key=lambda item: int(item[0])):
            router_rows.append(
                {
                    "condition": condition,
                    "depth": int(depth),
                    "selection_count": int(count),
                    "selection_fraction": int(count) / total,
                    "selection_total": total,
                }
            )
    router = pd.DataFrame(router_rows)
    tables["router"] = output / "spectral_router_selection_counts.csv"
    router.to_csv(tables["router"], index=False)

    atlas = pd.read_csv(root / "atlas" / "latent_spectrum_atlas.csv")
    atlas_compact = (
        atlas.groupby(
            ["provider", "evidence_label", "known_best_fid1k_gain"],
            dropna=False,
        )
        .agg(
            rms_mean=("rms", "mean"),
            rms_max=("rms", "max"),
            centroid_mean=("centroid", "mean"),
            low_fraction_mean=("low_fraction", "mean"),
            mid_fraction_mean=("mid_fraction", "mean"),
            high_fraction_mean=("high_fraction", "mean"),
        )
        .reset_index()
    )
    tables["atlas_compact"] = output / "latent_spectrum_atlas_compact.csv"
    atlas_compact.to_csv(tables["atlas_compact"], index=False)
    return tables


def plot_overview(metrics: pd.DataFrame, output: Path) -> None:
    names = [
        "screen_baseline_adaptive",
        "screen_full_depth8_v",
        "screen_full_external_v500",
        "screen_depth_schedule_native_coarse_to_fine",
        "screen_depth_schedule_native_fine_to_coarse",
        "screen_spectral_router_rms",
        "screen_spectral_anti_router_rms",
        "screen_euler_depth8_gamma0p4",
        "screen_spectral_delay_native_gamma0p8",
        "screen_raw_compute_rms_fine_to_coarse",
    ]
    selected = metrics.set_index("name").loc[names].copy().sort_values("fid", ascending=False)
    labels = [DISPLAY_NAMES[name] for name in selected.index]
    gains = selected["fid_gain_vs_integrator_baseline"].to_numpy()
    colors = ["#C44E52" if gain < 0 else "#4C72B0" for gain in gains]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.barh(labels, gains, color=colors)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlim(float(gains.min()) - 4.0, float(gains.max()) + 2.5)
    ax.set_xlabel("FID-1K gain versus paired integrator baseline (higher is better)")
    ax.set_title("Multiscale guidance study: paired screening results")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, gains):
        offset = 0.35 if value >= 0 else -0.35
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output / "method_overview_fid1k.png", dpi=180)
    plt.close(fig)


def plot_causal_maps(metrics: pd.DataFrame, baseline: float, output: Path) -> None:
    providers = ("depth8_v", "external_v500", "depth12_x")
    amplitudes = ("native", "equal_action")
    fig = plt.figure(figsize=(10.8, 12.0))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        left=0.08,
        right=0.94,
        bottom=0.06,
        top=0.94,
        hspace=0.34,
        wspace=0.18,
    )
    axes = np.empty((3, 2), dtype=object)
    for row_index in range(3):
        for col_index in range(2):
            axes[row_index, col_index] = fig.add_subplot(grid[row_index, col_index])
    colorbar_axis = fig.add_subplot(grid[:, 2])
    values: dict[tuple[str, str], np.ndarray] = {}
    for provider in providers:
        for amplitude in amplitudes:
            rows = metrics.loc[
                metrics["kind"].eq("band_time")
                & metrics["provider"].eq(provider)
                & metrics["amplitude"].eq(amplitude)
            ]
            pivot = rows.pivot(index="interval", columns="band", values="fid").reindex(
                index=TIME_ORDER, columns=BAND_ORDER
            )
            values[(provider, amplitude)] = baseline - pivot.to_numpy()
    limit = max(float(np.abs(value).max()) for value in values.values())
    for row_index, provider in enumerate(providers):
        for col_index, amplitude in enumerate(amplitudes):
            ax = axes[row_index, col_index]
            value = values[(provider, amplitude)]
            image = ax.imshow(value, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            for i in range(3):
                for j in range(3):
                    text_color = "white" if abs(value[i, j]) > 0.55 * limit else "#222222"
                    ax.text(
                        j,
                        i,
                        f"{value[i, j]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                    )
            ax.set_title(f"{provider} / {amplitude}")
            ax.set_xticks(range(3), BAND_ORDER)
            ax.set_yticks(range(3), TIME_ORDER)
    fig.colorbar(image, cax=colorbar_axis, label="FID-1K gain")
    fig.suptitle("Finite closed-loop interventions by time and latent FFT band", y=0.985)
    fig.savefig(output / "causal_map_fid1k.png", dpi=180)
    plt.close(fig)


def plot_depth_results(metrics: pd.DataFrame, output: Path) -> None:
    names = [
        "screen_static_depth4",
        "screen_static_depth6",
        "screen_static_depth8",
        "screen_static_depth10",
        "screen_static_depth12",
        "screen_depth_schedule_native_coarse_to_fine",
        "screen_depth_schedule_native_fine_to_coarse",
        "screen_depth_schedule_rms_coarse_to_fine",
        "screen_depth_schedule_rms_fine_to_coarse",
        "screen_spectral_router_native",
        "screen_spectral_anti_router_native",
    ]
    labels = {
        **DISPLAY_NAMES,
        "screen_static_depth4": "Static depth 4",
        "screen_static_depth6": "Static depth 6",
        "screen_static_depth8": "Static depth 8",
        "screen_static_depth10": "Static depth 10",
        "screen_static_depth12": "Static depth 12",
    }
    selected = metrics.set_index("name").loc[names].copy()
    x = np.arange(len(selected))
    colors = ["#55A868"] * 5 + ["#4C72B0", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#DD8452"]
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    bars = ax.bar(x, selected["fid"], color=colors)
    ax.axhline(
        float(metrics.loc[metrics["name"].eq("screen_baseline_adaptive"), "fid"].iloc[0]),
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="Adaptive baseline",
    )
    ax.set_ylabel("FID-1K (lower is better)")
    ax.set_title("Static depth, time schedule, and spectral routing")
    ax.set_xticks(x, [labels[name] for name in selected.index], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    for bar, value in zip(bars, selected["fid"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "depth_schedule_and_router_fid1k.png", dpi=180)
    plt.close(fig)


def plot_atlas(root: Path, output: Path) -> None:
    atlas = pd.read_csv(root / "atlas" / "latent_spectrum_atlas.csv")
    providers = ("depth4_v", "depth6_v", "depth8_v", "depth10_v", "depth12_v")
    colors = {
        "depth4_v": "#4C72B0",
        "depth6_v": "#55A868",
        "depth8_v": "#C44E52",
        "depth10_v": "#8172B2",
        "depth12_v": "#DD8452",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for provider in providers:
        rows = atlas.loc[atlas["provider"].eq(provider)].sort_values("time")
        axes[0].plot(rows["time"], rows["rms"], label=provider, color=colors[provider])
        axes[1].plot(rows["time"], rows["centroid"], label=provider, color=colors[provider])
    axes[0].set_title("Gap RMS across sampling time")
    axes[0].set_ylabel("RMS")
    axes[1].set_title("Latent spatial-frequency centroid")
    axes[1].set_ylabel("cycles per latent pixel")
    for ax in axes:
        ax.set_xlabel("t (noise to data)")
        ax.grid(alpha=0.2)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "depth_gap_latent_spectrum_atlas.png", dpi=180)
    plt.close(fig)


def plot_paired_previews(root: Path, output: Path) -> None:
    names = (
        "screen_baseline_adaptive",
        "screen_depth_schedule_native_coarse_to_fine",
        "screen_depth_schedule_native_fine_to_coarse",
        "screen_spectral_router_rms",
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.5))
    for ax, name in zip(axes.ravel(), names):
        image = plt.imread(root / "evaluations" / name / "preview.png")
        ax.imshow(image)
        ax.set_title(DISPLAY_NAMES[name])
        ax.axis("off")
    fig.suptitle("Paired qualitative screen: identical noise and labels", y=0.985)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.025, top=0.93, hspace=0.18, wspace=0.02)
    fig.savefig(output / "paired_preview_montage.png", dpi=150)
    plt.close(fig)


def copy_primary_data(root: Path, output: Path) -> dict[str, str]:
    sources = {
        "condition_metrics.csv": root / "summary" / "condition_metrics.csv",
        "study_summary.json": root / "summary" / "study_summary.json",
        "study_protocol.json": root / "study_protocol.json",
        "atlas_summary.json": root / "atlas" / "atlas_summary.json",
        "latent_spectrum_atlas.csv": root / "atlas" / "latent_spectrum_atlas.csv",
        "band_delay_fit.csv": root / "atlas" / "band_delay_fit.csv",
    }
    digests: dict[str, str] = {}
    for name, source in sources.items():
        destination = output / name
        if source.suffix == ".csv":
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            shutil.copy2(source, destination)
        digests[name] = sha256_file(destination)
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.study_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(root / "summary" / "condition_metrics.csv")
    audit = validate_study(root, metrics)
    metrics, baseline, euler_baseline = add_baseline_deltas(metrics)

    digests = copy_primary_data(root, output)
    tables = write_tables(root, output, metrics)
    plot_overview(metrics, output)
    plot_causal_maps(metrics, baseline, output)
    plot_depth_results(metrics, output)
    plot_atlas(root, output)
    plot_paired_previews(root, output)

    audit["adaptive_baseline_fid1k"] = baseline
    audit["fixed_euler_baseline_fid1k"] = euler_baseline
    best = metrics.sort_values("fid").iloc[0]
    audit["best_screen_condition"] = {
        "name": str(best["name"]),
        "fid1k": float(best["fid"]),
        "fid1k_gain": float(best["fid_gain_vs_adaptive_baseline"]),
    }
    audit["copied_file_sha256"] = digests
    audit["derived_tables"] = {key: path.name for key, path in tables.items()}
    (output / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
