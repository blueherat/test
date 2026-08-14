#!/usr/bin/env python3
"""Summarize 800K nominal-to-off-trajectory guidance geometry runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

try:
    from experiments.train_imagenet100_sit_flow import atomic_json_dump
except ModuleNotFoundError:
    from train_imagenet100_sit_flow import atomic_json_dump


BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_ROOT = BASE / "nominal_guidance_transfer_800k_v1"
FAMILIES = ("x800", "v500")
TRAJECTORIES = ("frozen", "replay", "closed")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).square().mean(dim=1).sqrt()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    denominator = left.norm(dim=1) * right.norm(dim=1)
    return (left * right).sum(dim=1) / denominator.clamp_min(torch.finfo(left.dtype).tiny)


def _run_dirs(root: Path, families: tuple[str, ...], seeds: tuple[int, ...]):
    for family in families:
        for seed in seeds:
            yield family, seed, root / "geometry" / f"{family}_seed{seed}"


def load_runs(
    root: Path,
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    by_time_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []
    endpoint_rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    pairing: dict[int, tuple[str, str]] = {}
    for family, seed, directory in _run_dirs(root, families, seeds):
        manifest_path = directory / "manifest.json"
        time_path = directory / "nominal_transfer_by_time.csv"
        segment_path = directory / "segment_transfer_by_time.csv"
        endpoint_path = directory / "endpoint_latents.pt"
        missing = [
            path for path in (manifest_path, time_path, segment_path, endpoint_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"incomplete {family} seed {seed}: {missing}")
        manifest = _load_json(manifest_path)
        if manifest.get("format") != "eqvae_imagenet100_sit_nominal_transfer_geometry_v1":
            raise ValueError(f"unexpected manifest format: {manifest_path}")
        if int(manifest["seed"]) != seed:
            raise ValueError(f"seed mismatch in {manifest_path}")
        signature = (str(manifest["noise_sha256"]), str(manifest["label_sha256"]))
        if seed in pairing and pairing[seed] != signature:
            raise ValueError(f"families are not paired for seed {seed}")
        pairing[seed] = signature
        manifests.append({"family": family, "directory": str(directory), **manifest})

        time_frame = pd.read_csv(time_path)
        time_frame.insert(0, "seed", seed)
        time_frame.insert(0, "family", family)
        by_time_frames.append(time_frame)
        segment_frame = pd.read_csv(segment_path)
        segment_frame.insert(0, "seed", seed)
        segment_frame.insert(0, "family", family)
        segment_frames.append(segment_frame)

        endpoints = torch.load(endpoint_path, map_location="cpu", weights_only=True)
        baseline = endpoints["baseline"].float()
        for trajectory in TRAJECTORIES:
            shift = endpoints[trajectory].float() - baseline
            endpoint_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "trajectory": trajectory,
                    "samples": len(shift),
                    "shift_rms_mean": float(_rms(shift).mean()),
                    "shift_rms_std": float(_rms(shift).std()),
                }
            )
        frozen_shift = endpoints["frozen"].float() - baseline
        for trajectory in ("replay", "closed"):
            shift = endpoints[trajectory].float() - baseline
            endpoint_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "trajectory": f"{trajectory}_vs_frozen_shift",
                    "samples": len(shift),
                    "shift_rms_mean": float(_rms(shift - frozen_shift).mean()),
                    "shift_rms_std": float(_rms(shift - frozen_shift).std()),
                    "shift_cosine_frozen_mean": float(_cosine(shift, frozen_shift).mean()),
                }
            )

    by_time = pd.concat(by_time_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    endpoints = pd.DataFrame(endpoint_rows)
    return by_time, segments, endpoints, manifests


def plot_frozen_transfer(by_time: pd.DataFrame, output: Path) -> None:
    frozen = by_time.loc[by_time.relation == "frozen"].copy()
    metrics = (
        ("cosine_mean", "cos(g baseline, g frozen)", (0.0, 1.02)),
        ("coefficient_mean", "Projection coefficient", None),
        ("orthogonal_energy_fraction_mean", "Orthogonal energy fraction", (0.0, 1.02)),
        ("change_over_nominal_rms_mean", "Gap change / nominal RMS", None),
        ("anchor_change_over_nominal_rms_mean", "Strong-field feedback / nominal RMS", None),
        ("effective_secant_gain_mean", "Gap secant gain", None),
    )
    colors = {"x800": "#2f6fa3", "v500": "#c44e38"}
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for axis, (metric, title, limits) in zip(axes.flat, metrics, strict=True):
        for family in sorted(frozen.family.unique()):
            family_frame = frozen.loc[frozen.family == family]
            for _, seed_frame in family_frame.groupby("seed"):
                axis.plot(
                    seed_frame.time,
                    seed_frame[metric],
                    color=colors.get(family),
                    alpha=0.25,
                    linewidth=1.2,
                )
            summary = family_frame.groupby("time", as_index=False)[metric].mean()
            axis.plot(
                summary.time,
                summary[metric],
                marker="o",
                linewidth=2.2,
                color=colors.get(family),
                label=family,
            )
        axis.set_title(title)
        axis.grid(alpha=0.2)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.set_xlabel("t (noise to data)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Why nominal-path frozen guidance transfers off trajectory", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_relation_comparison(
    by_time: pd.DataFrame,
    output: Path,
    *,
    families: tuple[str, ...],
) -> None:
    colors = {"frozen": "#2f6fa3", "replay": "#d18f00", "closed": "#3a9142"}
    figure, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for axis, family in zip(axes, families, strict=True):
        family_frame = by_time.loc[by_time.family == family]
        for relation in TRAJECTORIES:
            relation_frame = family_frame.loc[family_frame.relation == relation]
            summary = relation_frame.groupby("time", as_index=False).agg(
                cosine=("cosine_mean", "mean"),
                orthogonal=("orthogonal_energy_fraction_mean", "mean"),
            )
            axis.plot(
                summary.time,
                summary.cosine,
                marker="o",
                color=colors[relation],
                label=f"{relation}: cosine",
            )
            axis.plot(
                summary.time,
                1.0 - summary.orthogonal,
                linestyle="--",
                color=colors[relation],
                alpha=0.75,
                label=f"{relation}: parallel energy",
            )
        axis.set_title(family)
        axis.set_xlabel("t (noise to data)")
        axis.grid(alpha=0.2)
        axis.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Direction agreement")
    axes[-1].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Nominal gap agreement on frozen, replay, and closed trajectories")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_segment_transfer(
    segments: pd.DataFrame,
    output: Path,
    *,
    families: tuple[str, ...],
) -> None:
    frame = segments.copy()
    frame["alpha"] = frame.relation.str.removeprefix("segment_alpha_").astype(float)
    available_times = sorted(frame.time.unique())
    requested = (0.2, 0.6, 0.9, 0.95, 0.99)
    selected_times = [
        min(available_times, key=lambda value: abs(value - requested_time))
        for requested_time in requested
    ]
    selected_times = list(dict.fromkeys(selected_times))
    figure, axes = plt.subplots(
        2,
        len(families),
        figsize=(7 * len(families), 9),
        sharex=True,
    )
    if len(families) == 1:
        axes = axes.reshape(2, 1)
    cmap = plt.get_cmap("viridis")
    for family_index, family in enumerate(families):
        family_frame = frame.loc[frame.family == family]
        for time_index, time_value in enumerate(selected_times):
            time_frame = family_frame.loc[family_frame.time == time_value]
            summary = time_frame.groupby("alpha", as_index=False).agg(
                cosine=("cosine_mean", "mean"),
                coefficient=("coefficient_mean", "mean"),
            )
            color = cmap(time_index / max(1, len(selected_times) - 1))
            axes[0, family_index].plot(
                summary.alpha,
                summary.cosine,
                marker="o",
                color=color,
                label=f"t={time_value:.2f}",
            )
            axes[1, family_index].plot(
                summary.alpha,
                summary.coefficient,
                marker="o",
                color=color,
            )
        axes[0, family_index].set_title(family)
        axes[0, family_index].set_ylabel("cos(g baseline, g segment)")
        axes[1, family_index].set_ylabel("Projection coefficient")
        axes[1, family_index].set_xlabel("Segment alpha: baseline to frozen state")
        for axis in axes[:, family_index]:
            axis.grid(alpha=0.2)
    axes[0, -1].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Guidance stability along baseline-to-frozen displacement")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seeds", default="0,1")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    families = tuple(item for item in args.families.split(",") if item)
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    by_time, segments, endpoints, manifests = load_runs(
        root,
        families=families,
        seeds=seeds,
    )
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    compact_metrics = [
        "cosine_mean",
        "coefficient_mean",
        "orthogonal_energy_fraction_mean",
        "change_over_nominal_rms_mean",
        "anchor_change_over_nominal_rms_mean",
        "state_shift_rms_mean",
        "effective_secant_gain_mean",
    ]
    compact = (
        by_time.groupby(["family", "relation", "time"], as_index=False)[compact_metrics]
        .agg(["mean", "std"])
    )
    compact.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in compact.columns
    ]
    by_time.to_csv(summary_dir / "nominal_transfer_all_runs_by_time.csv", index=False)
    compact.to_csv(summary_dir / "nominal_transfer_compact.csv", index=False)
    segments.to_csv(summary_dir / "segment_transfer_all_runs_by_time.csv", index=False)
    endpoints.to_csv(summary_dir / "endpoint_latent_geometry.csv", index=False)
    plot_frozen_transfer(by_time, summary_dir / "frozen_transfer_geometry.png")
    plot_relation_comparison(
        by_time,
        summary_dir / "trajectory_relation_geometry.png",
        families=families,
    )
    plot_segment_transfer(
        segments,
        summary_dir / "segment_transfer_geometry.png",
        families=families,
    )
    payload = {
        "protocol": "imagenet100_sit_nominal_transfer_summary_v1",
        "question": "why nominal-path frozen guidance retains closed-guidance benefit",
        "families": list(families),
        "seeds": list(seeds),
        "pairing_verified_with_noise_and_label_sha256": True,
        "runs": manifests,
        "outputs": {
            "by_time": str(summary_dir / "nominal_transfer_all_runs_by_time.csv"),
            "compact": str(summary_dir / "nominal_transfer_compact.csv"),
            "segments": str(summary_dir / "segment_transfer_all_runs_by_time.csv"),
            "endpoints": str(summary_dir / "endpoint_latent_geometry.csv"),
            "frozen_plot": str(summary_dir / "frozen_transfer_geometry.png"),
            "relation_plot": str(summary_dir / "trajectory_relation_geometry.png"),
            "segment_plot": str(summary_dir / "segment_transfer_geometry.png"),
        },
    }
    atomic_json_dump(payload, summary_dir / "summary_manifest.json")
    print(endpoints.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
