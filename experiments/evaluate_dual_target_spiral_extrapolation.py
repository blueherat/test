#!/usr/bin/env python3
"""Sweep static prediction-target mixing on the formal continuous-spiral toy.

The sweep reuses trained dual-output heads and defines

    v_s = v_epsilon + s * (v_x - v_epsilon).

Thus ``s=0`` is the epsilon branch, ``s=1`` is the x branch, and ``s>1``
extrapolates beyond x away from epsilon.  Every scale uses the same initial
noise, reference samples, model checkpoint, sampler, and metric randomness.
Heads can come from ``D0_xeps`` or from ``D4_safe`` with its learned gate
deliberately ignored. No model is trained by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_dual_target_closed_loop_spiral_toy as spiral
import run_dual_target_closed_loop_toy as core


DEFAULT_SOURCE = Path.home() / "data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1"
DEFAULT_OUTPUT = (
    Path.home()
    / "data/eqvae/experiments/dual_target_closed_loop_spiral_extrapolation_v1"
)


def scale_tag(scale: float) -> str:
    text = f"{float(scale):.6g}".replace("-", "m").replace(".", "p")
    return f"scale_{text}"


def endpoint_override(
    velocity: torch.Tensor,
    *,
    velocity_x: torch.Tensor,
    velocity_epsilon: torch.Tensor,
    time_value: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    near_noise = time_value[:, None] <= denominator_floor
    near_data = time_value[:, None] >= 1.0 - denominator_floor
    velocity = torch.where(near_noise, velocity_x, velocity)
    return torch.where(near_data, velocity_epsilon, velocity)


@torch.no_grad()
def scaled_dual_velocity(
    *,
    suite: core.ModelSuite,
    state: torch.Tensor,
    time_value: torch.Tensor,
    scale_value: torch.Tensor | float,
    denominator_floor: float,
    endpoint_mode: str = "raw",
    head_source: str = "d0",
) -> torch.Tensor:
    model_ids = {"d0": "D0_xeps", "d4": "D4_safe"}
    if head_source not in model_ids:
        raise ValueError(f"unknown head source: {head_source}")
    output = suite.models[model_ids[head_source]](state, time_value)
    velocity_x, velocity_epsilon = core.endpoint_velocities(
        state=state,
        time_value=time_value,
        clean_prediction=output["x"],
        epsilon_prediction=output["eps"],
        denominator_floor=denominator_floor,
    )
    scale = torch.as_tensor(scale_value, device=state.device, dtype=state.dtype)
    if scale.ndim == 0:
        scale = scale.expand(len(state))
    if scale.shape != (len(state),):
        raise ValueError("scale_value must be scalar or have shape [B]")
    velocity = velocity_epsilon + scale[:, None] * (velocity_x - velocity_epsilon)
    # Preserve the two anchor branches exactly instead of relying on a
    # cancellation such as eps + (x - eps), which can round differently.
    velocity = torch.where(scale[:, None] == 0.0, velocity_epsilon, velocity)
    velocity = torch.where(scale[:, None] == 1.0, velocity_x, velocity)
    if endpoint_mode == "raw":
        return velocity
    if endpoint_mode == "override":
        return endpoint_override(
            velocity,
            velocity_x=velocity_x,
            velocity_epsilon=velocity_epsilon,
            time_value=time_value,
            denominator_floor=denominator_floor,
        )
    raise ValueError(f"unknown endpoint mode: {endpoint_mode}")


@torch.no_grad()
def sample_scale_sweep_heun(
    *,
    suite: core.ModelSuite,
    initial_noise: torch.Tensor,
    scales: Sequence[float],
    steps: int,
    denominator_floor: float,
    endpoint_mode: str = "raw",
    head_source: str = "d0",
) -> dict[float, torch.Tensor]:
    """Integrate all scales in one batched Heun rollout."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    unique_scales = [float(value) for value in scales]
    if len(set(unique_scales)) != len(unique_scales):
        raise ValueError("scales must be unique")
    if not unique_scales:
        raise ValueError("at least one scale is required")

    sample_count, ambient_dim = initial_noise.shape
    step_size = 1.0 / steps

    def integrate(scale_group: Sequence[float]) -> torch.Tensor:
        scale_count = len(scale_group)
        state = initial_noise.unsqueeze(0).expand(scale_count, -1, -1).clone()
        scale_rows = torch.tensor(
            scale_group, device=initial_noise.device, dtype=initial_noise.dtype
        )[:, None].expand(-1, sample_count)
        flat_scale = scale_rows.reshape(-1)

        def field(value: torch.Tensor, time_point: float) -> torch.Tensor:
            flat = value.reshape(scale_count * sample_count, ambient_dim)
            time_value = torch.full(
                (len(flat),), time_point, device=flat.device, dtype=flat.dtype
            )
            velocity = scaled_dual_velocity(
                suite=suite,
                state=flat,
                time_value=time_value,
                scale_value=flat_scale,
                denominator_floor=denominator_floor,
                endpoint_mode=endpoint_mode,
                head_source=head_source,
            )
            return velocity.reshape_as(value)

        for step in range(steps):
            t0 = float(step) / steps
            t1 = float(step + 1) / steps
            velocity0 = field(state, t0)
            proposal = state + step_size * velocity0
            velocity1 = field(proposal, t1)
            state = state + 0.5 * step_size * (velocity0 + velocity1)
            if not torch.isfinite(state).all():
                finite = torch.isfinite(state).flatten(1).all(dim=1)
                failed = [
                    scale_group[index]
                    for index, valid in enumerate(finite.tolist())
                    if not valid
                ]
                raise FloatingPointError(
                    f"non-finite rollout at step {step + 1}/{steps}, scales={failed}"
                )
        return state

    # Evaluate s=0 and s=1 in their own [B, D] forward passes. This makes them
    # numerically identical to the pre-existing branch samplers; the remaining
    # scales are still fused into one large batch.
    anchor_scales = [value for value in unique_scales if value in (0.0, 1.0)]
    other_scales = [value for value in unique_scales if value not in anchor_scales]
    outputs: dict[float, torch.Tensor] = {}
    if other_scales:
        other_state = integrate(other_scales)
        outputs.update(
            {
                scale: other_state[index].detach().cpu()
                for index, scale in enumerate(other_scales)
            }
        )
    for scale in anchor_scales:
        outputs[scale] = integrate([scale])[0].detach().cpu()
    return {scale: outputs[scale] for scale in unique_scales}


def build_setting(
    *, source_dir: Path, ambient_dim: int, seed: int, device: torch.device
) -> tuple[dict, spiral.ContinuousSpiralDistribution, core.ModelSuite]:
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    if int(config["ambient_dim"]) != ambient_dim or int(config["seed"]) != seed:
        raise ValueError(f"source config does not match {source_dir}")
    distribution = spiral.ContinuousSpiralDistribution(
        ambient_dim,
        data_jitter=float(config["data_jitter"]),
        quadrature_points=int(config["quadrature_points"]),
        locator_points=int(config["locator_points"]),
        frequency_scale=float(config["frequency_scale"]),
        embedding_seed=core.stable_seed(seed, ambient_dim, 71),
        device=device,
        scale_mode=str(config["scale_mode"]),
        curvature=float(config["curvature"]),
        bayes_batch_chunk=int(config["bayes_batch_chunk"]),
    )
    model_ids = ("D0_xeps", "D4_safe")
    suite = core.build_model_suite(
        ambient_dim=ambient_dim,
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
        time_dim=int(config["time_dim"]),
        mode_dim=int(config["mode_dim"]),
        model_ids=model_ids,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        seed=core.stable_seed(seed, ambient_dim, 113),
        device=device,
    )
    checkpoint = torch.load(
        source_dir / "checkpoint.pt", map_location=device, weights_only=False
    )
    for model_id in model_ids:
        suite.models[model_id].load_state_dict(checkpoint["models"][model_id])
        suite.models[model_id].eval()
    return config, distribution, suite


def plot_setting_scatter(
    path: Path,
    *,
    generated: dict[float, torch.Tensor],
    dynamic_endpoint: torch.Tensor,
    dynamic_label: str,
    reference_intrinsic: torch.Tensor,
    distribution: spiral.ContinuousSpiralDistribution,
    plot_points: int,
) -> None:
    preferred = (
        0.0,
        0.75,
        0.9,
        0.975,
        1.0,
        1.01,
        1.025,
        1.05,
        1.075,
        1.1,
        1.2,
        1.5,
        1.78,
        2.3,
    )
    selected = [value for value in preferred if value in generated]
    if not selected:
        selected = list(generated)[:8]
    panels: list[tuple[str, np.ndarray]] = [
        ("Reference", reference_intrinsic[:plot_points].numpy())
    ]
    for scale in selected:
        intrinsic = distribution.decode_intrinsic(
            generated[scale][:plot_points].to(distribution.device)
        ).cpu().numpy()
        panels.append((f"s={scale:g} (gamma={scale - 1:g})", intrinsic))
    dynamic_intrinsic = distribution.decode_intrinsic(
        dynamic_endpoint[:plot_points].to(distribution.device)
    ).cpu().numpy()
    panels.append((dynamic_label, dynamic_intrinsic))

    reference = reference_intrinsic.numpy()
    low = np.quantile(reference, 0.002, axis=0)
    high = np.quantile(reference, 0.998, axis=0)
    margin = 0.12 * np.maximum(high - low, 1e-6)
    xlim = (float(low[0] - margin[0]), float(high[0] + margin[0]))
    ylim = (float(low[1] - margin[1]), float(high[1] + margin[1]))

    columns = 5
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(22, 4.4 * rows), squeeze=False)
    for axis, (title, points) in zip(axes.flat, panels):
        axis.scatter(points[:, 0], points[:, 1], s=3, alpha=0.35)
        axis.set_title(title)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    figure.suptitle("Prediction-target extrapolation on the continuous spiral")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate_results(output_root: Path, source_root: Path) -> None:
    paths = sorted(output_root.glob("seed*/D*_H*/endpoint_scale_sweep.csv"))
    if not paths:
        raise RuntimeError(f"no setting results found under {output_root}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    frame.to_csv(aggregate / "scale_sweep_all.csv", index=False)

    metrics = (
        "swd_fullD",
        "swd_2d",
        "ridge_distance_mean",
        "ridge_width_ratio",
        "arc_hist_tv",
        "off_subspace_rms",
    )
    summary = (
        frame.groupby(["ambient_dim", "scale", "gamma"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in metrics
                for stat in ("mean", "std")
            },
        )
        .sort_values(["ambient_dim", "scale"])
    )
    summary.to_csv(aggregate / "scale_sweep_seed_summary.csv", index=False)

    baseline_paths = sorted(source_root.glob("seed*/D*_H*/cross_gate_endpoint_metrics.csv"))
    baseline_frames: list[pd.DataFrame] = []
    for baseline_path in baseline_paths:
        baseline_frame = pd.read_csv(baseline_path)
        source_config = json.loads(
            (baseline_path.parent / "config.json").read_text(encoding="utf-8")
        )
        baseline_frame["ambient_dim"] = int(source_config["ambient_dim"])
        baseline_frame["seed"] = int(source_config["seed"])
        baseline_frames.append(baseline_frame)
    baseline = pd.concat(baseline_frames, ignore_index=True)
    baseline = baseline[
        baseline["condition"].isin(("D3_oracle_bayes_gate", "D4_gate_on_D0"))
    ]
    own_dynamic_frames: list[pd.DataFrame] = []
    for endpoint_path in sorted(source_root.glob("seed*/D*_H*/endpoint_metrics.csv")):
        endpoint_frame = pd.read_csv(endpoint_path)
        endpoint_frame = endpoint_frame[
            endpoint_frame["condition"] == "D4_safe_velocity_gate"
        ].copy()
        source_config = json.loads(
            (endpoint_path.parent / "config.json").read_text(encoding="utf-8")
        )
        endpoint_frame["ambient_dim"] = int(source_config["ambient_dim"])
        endpoint_frame["seed"] = int(source_config["seed"])
        endpoint_frame["condition"] = "D4_own_dynamic"
        own_dynamic_frames.append(endpoint_frame)
    if own_dynamic_frames:
        baseline = pd.concat([baseline, *own_dynamic_frames], ignore_index=True)
    baseline_summary = baseline.groupby(
        ["ambient_dim", "condition"], as_index=False
    ).agg(
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in metrics
            for stat in ("mean", "std")
        }
    )
    baseline_summary.to_csv(aggregate / "dynamic_gate_baselines.csv", index=False)

    plot_scale_curves(
        aggregate / "scale_sweep_curves.png",
        summary=summary,
        baseline_summary=baseline_summary,
        metrics=metrics,
        endpoint_mode=str(frame["endpoint_mode"].iloc[0]),
        head_source=str(frame["head_source"].iloc[0]),
    )
    payload: dict[str, object] = {
        "definition": "v_s = v_epsilon + s * (v_x - v_epsilon); gamma = s - 1",
        "endpoint_mode": str(frame["endpoint_mode"].iloc[0]),
        "head_source": str(frame["head_source"].iloc[0]),
        "settings": len(paths),
        "seeds": sorted(int(value) for value in frame["seed"].unique()),
        "dimensions": sorted(int(value) for value in frame["ambient_dim"].unique()),
        "scales": sorted(float(value) for value in frame["scale"].unique()),
        "best_extrapolation": {},
    }
    for dimension in sorted(frame["ambient_dim"].unique()):
        subset = summary[(summary["ambient_dim"] == dimension) & (summary["scale"] >= 1)]
        base = subset[np.isclose(subset["scale"], 1.0)].iloc[0]
        best_full = subset.loc[subset["swd_fullD_mean"].idxmin()]
        best_intrinsic = subset.loc[subset["swd_2d_mean"].idxmin()]
        payload["best_extrapolation"][str(int(dimension))] = {
            "baseline_scale_1_fullD_swd": float(base["swd_fullD_mean"]),
            "best_fullD_scale": float(best_full["scale"]),
            "best_fullD_swd": float(best_full["swd_fullD_mean"]),
            "best_fullD_relative_change": float(
                best_full["swd_fullD_mean"] / base["swd_fullD_mean"] - 1.0
            ),
            "best_intrinsic_scale": float(best_intrinsic["scale"]),
            "best_intrinsic_swd": float(best_intrinsic["swd_2d_mean"]),
            "best_intrinsic_relative_change": float(
                best_intrinsic["swd_2d_mean"] / base["swd_2d_mean"] - 1.0
            ),
        }
    core.save_json(aggregate / "summary.json", payload)


def plot_scale_curves(
    path: Path,
    *,
    summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    metrics: Sequence[str],
    endpoint_mode: str,
    head_source: str,
) -> None:
    labels = {
        "swd_fullD": "Full-D SWD",
        "swd_2d": "Intrinsic SWD",
        "ridge_distance_mean": "Distance to spiral ridge",
        "ridge_width_ratio": "Ridge width / reference",
        "arc_hist_tv": "Arc coverage TV",
        "off_subspace_rms": "Off-subspace RMS",
    }
    dimensions = sorted(int(value) for value in summary["ambient_dim"].unique())
    figure, axes = plt.subplots(
        len(dimensions), len(metrics), figsize=(5.2 * len(metrics), 4.8 * len(dimensions)),
        squeeze=False,
    )
    baseline_styles = {
        "D3_oracle_bayes_gate": ("black", ":", "D3 pointwise oracle"),
        "D4_gate_on_D0": ("tab:purple", "--", "D4 gate on D0 heads"),
        "D4_own_dynamic": ("tab:green", "-.", "D4 own dynamic gate"),
    }
    for row_index, dimension in enumerate(dimensions):
        values = summary[summary["ambient_dim"] == dimension]
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            axis.errorbar(
                values["scale"],
                values[f"{metric}_mean"],
                yerr=values[f"{metric}_std"],
                marker="o",
                capsize=3,
                color="tab:blue",
                label="static target scale",
            )
            axis.axvline(1.0, color="tab:red", linewidth=1.2, label="s=1: x branch")
            axis.axvspan(1.0, float(values["scale"].max()), color="tab:red", alpha=0.05)
            for condition, (color, style, label) in baseline_styles.items():
                row = baseline_summary[
                    (baseline_summary["ambient_dim"] == dimension)
                    & (baseline_summary["condition"] == condition)
                ]
                if len(row):
                    axis.axhline(
                        float(row[f"{metric}_mean"].iloc[0]),
                        color=color,
                        linestyle=style,
                        linewidth=1.4,
                        label=label,
                    )
            axis.set_title(f"D={dimension}: {labels[metric]}")
            axis.set_xlabel("scale s (s > 1 is extrapolation)")
            axis.grid(alpha=0.25)
            if metric in {
                "swd_fullD",
                "swd_2d",
                "ridge_distance_mean",
                "ridge_width_ratio",
                "off_subspace_rms",
            }:
                axis.set_yscale("log")
            if row_index == 0 and column_index == 0:
                axis.legend(fontsize=8)
    figure.suptitle(
        "Static prediction-target extrapolation sweep "
        f"(heads: {head_source}; endpoint mode: {endpoint_mode}; mean +/- seed std)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_setting(
    *,
    args: argparse.Namespace,
    ambient_dim: int,
    seed: int,
    device: torch.device,
) -> Path:
    source_dir = args.source_root / f"seed{seed}" / f"D{ambient_dim}_H128"
    output_dir = args.output_root / f"seed{seed}" / f"D{ambient_dim}_H128"
    output_dir.mkdir(parents=True, exist_ok=True)
    config, distribution, suite = build_setting(
        source_dir=source_dir, ambient_dim=ambient_dim, seed=seed, device=device
    )
    denominator_floor = float(config["denominator_floor"])
    sample_count = int(args.sample_count or config["sample_count"])
    sample_steps = int(args.sample_steps or config["sample_steps"])
    reference_count = int(args.reference_count or config["reference_count"])

    generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(seed, ambient_dim, 149)
    )
    initial_noise = torch.randn(
        sample_count, ambient_dim, device=device, generator=generator
    )
    started = time.monotonic()
    generated = sample_scale_sweep_heun(
        suite=suite,
        initial_noise=initial_noise,
        scales=args.scales,
        steps=sample_steps,
        denominator_floor=denominator_floor,
        endpoint_mode=args.endpoint_mode,
        head_source=args.head_source,
    )
    scale_one = next(value for value in args.scales if math.isclose(value, 1.0))
    anchor_max_abs: dict[str, float] = {}
    anchor_conditions: dict[float, str] = {}
    if args.head_source == "d0" and args.endpoint_mode == "raw":
        anchor_conditions = {0.0: "D0_eps_shared", 1.0: "D0_x_shared"}
    elif args.head_source == "d4" and args.endpoint_mode == "override":
        anchor_conditions = {0.0: "D4_eps_own", 1.0: "D4_x_own"}
    for anchor_scale, anchor_condition in anchor_conditions.items():
        matching = next(
            (value for value in args.scales if math.isclose(value, anchor_scale)), None
        )
        if matching is None:
            continue
        baseline_endpoint, _ = core.sample_heun(
            anchor_condition,
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=sample_steps,
            denominator_floor=denominator_floor,
            snapshot_times=(),
        )
        max_abs = float(
            (generated[matching].to(device) - baseline_endpoint).abs().max()
        )
        anchor_max_abs[str(anchor_scale)] = max_abs
        if max_abs > 2e-5:
            raise AssertionError(
                f"s={anchor_scale:g} regression mismatch for {anchor_condition}: {max_abs}"
            )

    dynamic_condition = (
        "D4_gate_on_D0" if args.head_source == "d0" else "D4_safe_velocity_gate"
    )
    dynamic_endpoint, _ = core.sample_heun(
        dynamic_condition,
        suite=suite,
        distribution=distribution,
        initial_noise=initial_noise,
        steps=sample_steps,
        denominator_floor=denominator_floor,
        snapshot_times=(),
    )
    reference_generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(seed, ambient_dim, 151)
    )
    reference, reference_intrinsic, _ = distribution.sample(
        reference_count, generator=reference_generator
    )
    resample_generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(seed, ambient_dim, 153)
    )
    reference_resample, _, _ = distribution.sample(
        sample_count, generator=resample_generator
    )

    metric_generated: dict[str, torch.Tensor] = {
        scale_tag(scale_one): generated[scale_one],
        **{
            scale_tag(scale): value
            for scale, value in generated.items()
            if not math.isclose(scale, scale_one)
        },
        dynamic_condition: dynamic_endpoint.detach().cpu(),
        "Reference_resample": reference_resample.detach().cpu(),
    }
    rows = spiral.endpoint_metrics_spiral(
        generated=metric_generated,
        reference=reference.detach().cpu(),
        reference_intrinsic=reference_intrinsic.detach().cpu(),
        distribution=distribution,
        seed=core.stable_seed(seed, ambient_dim, 157),
        swd_projections=int(config["swd_projections"]),
        swd_max_points=int(config["swd_max_points"]),
        full_swd_projections=int(config["full_swd_projections"]),
        full_swd_max_points=int(config["full_swd_max_points"]),
        mmd_max_points=int(config["mmd_max_points"]),
        coverage_bins=int(config["coverage_bins"]),
        conditional_ridge_bins=int(config["conditional_ridge_bins"]),
        conditional_ridge_min_count=int(config["conditional_ridge_min_count"]),
    )
    row_by_condition = {row["condition"]: row for row in rows}
    sweep_rows: list[dict] = []
    for scale in args.scales:
        row = dict(row_by_condition[scale_tag(scale)])
        row.update(
            {
                "seed": seed,
                "ambient_dim": ambient_dim,
                "scale": float(scale),
                "gamma": float(scale - 1.0),
                "regime": (
                    "extrapolation"
                    if scale > 1.0
                    else "x_branch"
                    if math.isclose(scale, 1.0)
                    else "interpolation"
                ),
                "endpoint_mode": args.endpoint_mode,
                "head_source": args.head_source,
            }
        )
        sweep_rows.append(row)
    core.save_csv(output_dir / "endpoint_scale_sweep.csv", sweep_rows)

    source_metrics_name = (
        "cross_gate_endpoint_metrics.csv"
        if args.head_source == "d0"
        else "endpoint_metrics.csv"
    )
    source_dynamic = pd.read_csv(source_dir / source_metrics_name)
    source_dynamic = source_dynamic[
        source_dynamic["condition"] == dynamic_condition
    ].iloc[0]
    evaluated_dynamic = row_by_condition[dynamic_condition]
    dynamic_full_swd_abs_diff = abs(
        float(source_dynamic["swd_fullD"])
        - float(evaluated_dynamic["swd_fullD"])
    )
    formal_protocol = (
        sample_count == int(config["sample_count"])
        and reference_count == int(config["reference_count"])
        and sample_steps == int(config["sample_steps"])
    )
    if formal_protocol and dynamic_full_swd_abs_diff > 1e-6:
        raise AssertionError(
            f"dynamic baseline metric regression mismatch: {dynamic_full_swd_abs_diff}"
        )

    plot_setting_scatter(
        output_dir / "extrapolation_scatter.png",
        generated=generated,
        dynamic_endpoint=dynamic_endpoint.detach().cpu(),
        dynamic_label=dynamic_condition,
        reference_intrinsic=reference_intrinsic.detach().cpu(),
        distribution=distribution,
        plot_points=args.plot_points,
    )
    core.save_json(
        output_dir / "run_manifest.json",
        {
            "source": str(source_dir),
            "seed": seed,
            "ambient_dim": ambient_dim,
            "definition": "v_s = v_epsilon + s * (v_x - v_epsilon)",
            "endpoint_mode": args.endpoint_mode,
            "head_source": args.head_source,
            "scales": list(args.scales),
            "sample_count": sample_count,
            "reference_count": reference_count,
            "sample_steps": sample_steps,
            "anchor_endpoint_max_abs_diff": anchor_max_abs,
            "dynamic_condition": dynamic_condition,
            "dynamic_full_swd_abs_diff_from_formal_source": dynamic_full_swd_abs_diff,
            "formal_protocol": formal_protocol,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        json.dumps(
            {
                "event": "setting_complete",
                "seed": seed,
                "ambient_dim": ambient_dim,
                "elapsed_seconds": time.monotonic() - started,
                "anchor_max_abs": anchor_max_abs,
            }
        ),
        flush=True,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        type=core.parse_int_list,
        default=core.parse_int_list("20260831,20260901,20260902"),
    )
    parser.add_argument("--dims", type=core.parse_int_list, default=core.parse_int_list("2,512"))
    parser.add_argument(
        "--scales",
        type=core.parse_float_list,
        default=core.parse_float_list("0,0.5,0.75,1,1.05,1.1,1.2,1.5,1.78,2,2.3"),
    )
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--reference-count", type=int)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--plot-points", type=int, default=2500)
    parser.add_argument("--endpoint-mode", choices=("raw", "override"), default="raw")
    parser.add_argument("--head-source", choices=("d0", "d4"), default="d0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.scales = [float(value) for value in args.scales]
    if len(set(args.scales)) != len(args.scales):
        parser.error("--scales must contain unique values")
    if not any(math.isclose(value, 1.0) for value in args.scales):
        parser.error("--scales must include 1.0 for the regression check")
    if any(value < 0 for value in args.scales):
        parser.error("negative scales are outside this first sweep")
    return args


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    core.save_json(
        args.output_root / "sweep_config.json",
        {
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
            "seeds": args.seeds,
            "dims": args.dims,
            "scales": args.scales,
            "endpoint_mode": args.endpoint_mode,
            "head_source": args.head_source,
            "device": str(device),
        },
    )
    for seed in args.seeds:
        for ambient_dim in args.dims:
            run_setting(args=args, ambient_dim=ambient_dim, seed=seed, device=device)
    aggregate_results(args.output_root, args.source_root)
    print(json.dumps({"event": "complete", "output_root": str(args.output_root)}))


if __name__ == "__main__":
    main()
