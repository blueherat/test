#!/usr/bin/env python3
"""Dual-target closed-loop diagnostic on the v4/v10 continuous spiral.

The model-side protocol follows SiT:

    x_t = (1 - t) * epsilon + t * x,  t: 0 -> 1
    v   = x - epsilon

The data and endpoint metrics follow the reviewed prediction-target v4/v10
toys: a continuous two-turn spiral with intrinsic jitter, a unit-RMS linear
embedding into R^D, fixed 2-D/full-D SWD, and ridge/coverage diagnostics.
Curvature is deliberately fixed to zero in this first experiment so that the
Bayes conditional vector field is available in closed form up to a dense,
deterministic quadrature of the continuous spiral coordinate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_dual_target_closed_loop_toy as core
import run_prediction_target_extrapolation_toy_v10_final as v10

v4 = v10.v4


class ContinuousSpiralDistribution:
    """v4/v10 spiral with an efficient exact Bayes field for curvature zero."""

    def __init__(
        self,
        ambient_dim: int,
        *,
        data_jitter: float,
        quadrature_points: int,
        locator_points: int,
        frequency_scale: float,
        embedding_seed: int,
        device: torch.device,
        scale_mode: str = "unit_rms",
        curvature: float = 0.0,
        bayes_batch_chunk: int = 4096,
    ) -> None:
        if ambient_dim < 2:
            raise ValueError("ambient_dim must be at least 2")
        if data_jitter <= 0:
            raise ValueError("data_jitter must be positive")
        if quadrature_points < 64:
            raise ValueError("quadrature_points must be at least 64")
        if locator_points < 128:
            raise ValueError("locator_points must be at least 128")
        if curvature != 0.0:
            raise ValueError(
                "the exact closed-loop Bayes oracle currently requires curvature=0"
            )
        if bayes_batch_chunk <= 0:
            raise ValueError("bayes_batch_chunk must be positive")

        self.ambient_dim = int(ambient_dim)
        self.data_jitter = float(data_jitter)
        # v4 scales the complete intrinsic point after adding jitter.
        self.intrinsic_jitter_std = 1.6 * self.data_jitter
        self.device = device
        self.curvature = float(curvature)
        self.bayes_batch_chunk = int(bayes_batch_chunk)
        self.embedding = v4.CurvedEmbedding(
            ambient_dim,
            curvature=curvature,
            frequency_scale=frequency_scale,
            seed=embedding_seed,
            device=device,
            scale_mode=scale_mode,
        )
        self.locator = v10.SpiralLocator(locator_points, device)

        # Midpoint quadrature matches s ~ Uniform[0, 1] without endpoint bias.
        self.quadrature_s = (
            torch.arange(quadrature_points, device=device, dtype=torch.float32) + 0.5
        ) / quadrature_points
        self.quadrature_u = v10.spiral_center(self.quadrature_s)
        self.basis = self.embedding.Q[:, :2]
        self.scale = float(self.embedding.global_scale)

    def sample(
        self, n: int, *, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        intrinsic = v4.sample_spiral_2d(
            n,
            device=self.device,
            jitter=self.data_jitter,
            generator=generator,
        )
        return self.embedding.embed(intrinsic), intrinsic, None

    def decode_intrinsic(self, value: torch.Tensor) -> torch.Tensor:
        return self.embedding.decode_intrinsic(value)

    def off_subspace_rms(self, value: torch.Tensor) -> torch.Tensor:
        return self.embedding.manifold_consistency_rms(value)

    def bayes_clean(self, state: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        """Return E[x | x_t] under the continuous jittered spiral prior."""
        if time_value.shape != (len(state),):
            raise ValueError("time_value must have shape [B]")
        projected = state @ self.basis
        outputs: list[torch.Tensor] = []
        jitter_variance = self.intrinsic_jitter_std**2
        for start in range(0, len(state), self.bayes_batch_chunk):
            stop = min(start + self.bayes_batch_chunk, len(state))
            observed = projected[start:stop]
            t = time_value[start:stop].float()[:, None]
            gain = t * self.scale
            total_variance = (
                gain.square() * jitter_variance + (1.0 - t).square()
            ).clamp_min(1e-12)
            residual = (
                observed[:, None, :]
                - gain[:, None, :] * self.quadrature_u[None, :, :]
            )
            logits = -0.5 * residual.square().sum(dim=2) / total_variance
            weights = torch.softmax(logits, dim=1)
            kalman_gain = jitter_variance * gain / total_variance
            posterior_u = (
                self.quadrature_u[None, :, :]
                + kalman_gain[:, None, :] * residual
            )
            mean_u = (weights[:, :, None] * posterior_u).sum(dim=1)
            outputs.append(self.embedding.embed(mean_u))
        return torch.cat(outputs, dim=0)

    def bayes_velocity(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        *,
        denominator_floor: float,
    ) -> torch.Tensor:
        effective_time = time_value.float().clamp(max=1.0 - denominator_floor)
        clean = self.bayes_clean(state, effective_time)
        return (clean - state) / (1.0 - effective_time)[:, None]

    def intrinsic_nll(self, intrinsic: torch.Tensor) -> torch.Tensor:
        variance = self.intrinsic_jitter_std**2
        normalizer = math.log(2.0 * math.pi * variance) + math.log(
            len(self.quadrature_u)
        )
        rows: list[torch.Tensor] = []
        for start in range(0, len(intrinsic), self.bayes_batch_chunk):
            value = intrinsic[start : start + self.bayes_batch_chunk]
            residual = value[:, None, :] - self.quadrature_u[None, :, :]
            component_log_prob = -0.5 * residual.square().sum(dim=2) / variance
            rows.append(-(torch.logsumexp(component_log_prob, dim=1) - normalizer))
        return torch.cat(rows)


def endpoint_metrics_spiral(
    *,
    generated: dict[str, torch.Tensor],
    reference: torch.Tensor,
    reference_intrinsic: torch.Tensor,
    distribution: ContinuousSpiralDistribution,
    seed: int,
    swd_projections: int,
    swd_max_points: int,
    full_swd_projections: int,
    full_swd_max_points: int,
    mmd_max_points: int,
    coverage_bins: int,
    conditional_ridge_bins: int,
    conditional_ridge_min_count: int,
) -> list[dict]:
    reference_np = reference.cpu().numpy()
    reference_u = reference_intrinsic.cpu().numpy()
    sample_count = len(next(iter(generated.values())))
    context = v10.build_distribution_metric_context(
        sample_count=sample_count,
        reference_count=len(reference_np),
        D=distribution.ambient_dim,
        swd_projections=swd_projections,
        swd_max_points=swd_max_points,
        full_swd_projections=full_swd_projections,
        full_swd_max_points=full_swd_max_points,
        mmd_max_points=mmd_max_points,
        seed=seed,
    )
    reference_geometry = v10.build_reference_endpoint_geometry(
        reference_u,
        locator=distribution.locator,
        bins=coverage_bins,
        device=distribution.device,
    )
    full_theta, full_reference = v10.prepare_swd_reference_device(
        reference_np,
        theta=context.theta_full,
        idx_ref=context.idx_ref_full,
        device=distribution.device,
    )

    baseline_name = "D0_x_shared" if "D0_x_shared" in generated else next(iter(generated))
    baseline_u = distribution.decode_intrinsic(
        generated[baseline_name].to(distribution.device)
    ).cpu().numpy()
    if mmd_max_points > 1:
        context.mmd_sigma2 = v10.rbf_bandwidth_fixed(
            baseline_u,
            reference_u,
            idx_a=context.idx_sample_mmd,
            idx_b=context.idx_ref_mmd,
            bandwidth_subset=context.bandwidth_subset,
        )

    rows: list[dict] = []
    for condition, value_cpu in generated.items():
        ambient = value_cpu.numpy()
        features = v10.endpoint_features(
            ambient,
            emb=distribution.embedding,
            locator=distribution.locator,
            device=distribution.device,
        )
        ridge = v10.ridge_metrics_from_features(
            features,
            reference_geometry,
            bins=coverage_bins,
            conditional_bins=conditional_ridge_bins,
            conditional_min_count=conditional_ridge_min_count,
        )
        swd_2d = v10.swd_fixed(
            features.intrinsic,
            reference_u,
            theta=context.theta_2d,
            idx_a=context.idx_sample_2d,
            idx_b=context.idx_ref_2d,
        )
        swd_full = v10.swd_fixed_against_cached_reference_device(
            ambient,
            theta_t=full_theta,
            idx_candidate=context.idx_sample_full,
            ref_sorted=full_reference,
            device=distribution.device,
        )
        if context.mmd_sigma2 is None:
            mmd = float("nan")
        else:
            mmd = v10.mmd_rbf_fixed(
                features.intrinsic,
                reference_u,
                idx_a=context.idx_sample_mmd,
                idx_b=context.idx_ref_mmd,
                sigma2=context.mmd_sigma2,
            )
        value = value_cpu.to(distribution.device)
        intrinsic = distribution.decode_intrinsic(value)
        rows.append(
            {
                "condition": condition,
                "swd_2d": swd_2d,
                "swd_fullD": swd_full,
                "mmd_2d": mmd,
                "intrinsic_nll": float(distribution.intrinsic_nll(intrinsic).mean()),
                # Aliases keep the common cross-seed summarizer reusable.
                "intrinsic_swd": swd_2d,
                "ambient_swd": swd_full,
                "intrinsic_mmd": mmd,
                "off_subspace_rms": float(distribution.off_subspace_rms(value).mean()),
                **ridge,
            }
        )
    return rows


@torch.no_grad()
def rollout_diagnostics_spiral(
    *,
    snapshots: dict[str, dict[float, torch.Tensor]],
    suite: core.ModelSuite,
    distribution: ContinuousSpiralDistribution,
    denominator_floor: float,
    seed: int,
    swd_projections: int,
) -> list[dict]:
    rows: list[dict] = []
    true_cache: dict[tuple[float, int], torch.Tensor] = {}
    theta = v10.fixed_theta_nd(swd_projections, 2, core.stable_seed(seed, 991))
    for condition, condition_snapshots in snapshots.items():
        for time_point, state_cpu in sorted(condition_snapshots.items()):
            state = state_cpu.to(distribution.device)
            time_value = torch.full((len(state),), time_point, device=state.device)
            prediction, gate = core.condition_field(
                condition,
                suite=suite,
                distribution=distribution,
                state=state,
                time_value=time_value,
                denominator_floor=denominator_floor,
            )
            bayes = distribution.bayes_velocity(
                state, time_value, denominator_floor=denominator_floor
            )
            cache_key = (time_point, len(state))
            if cache_key not in true_cache:
                generator = torch.Generator(device=distribution.device.type).manual_seed(
                    core.stable_seed(seed, int(round(time_point * 10000)), len(state))
                )
                clean, _, _ = distribution.sample(len(state), generator=generator)
                epsilon = torch.randn(clean.shape, device=clean.device, generator=generator)
                true_cache[cache_key] = (
                    (1.0 - time_point) * epsilon + time_point * clean
                ).detach()
            true_state = true_cache[cache_key]
            state_u = distribution.decode_intrinsic(state).cpu().numpy()
            true_u = distribution.decode_intrinsic(true_state).cpu().numpy()
            n = min(len(state_u), len(true_u))
            indices = np.arange(n)
            row = {
                "condition": condition,
                "time": time_point,
                "rollout_bayes_velocity_mse": float(
                    core.row_mse(prediction - bayes).mean()
                ),
                "state_swd_2d": v10.swd_fixed(
                    state_u,
                    true_u,
                    theta=theta,
                    idx_a=indices,
                    idx_b=indices,
                ),
                "state_intrinsic_swd": v10.swd_fixed(
                    state_u,
                    true_u,
                    theta=theta,
                    idx_a=indices,
                    idx_b=indices,
                ),
                "state_off_subspace_rms": float(
                    distribution.off_subspace_rms(state).mean()
                ),
                "true_off_subspace_rms": float(
                    distribution.off_subspace_rms(true_state).mean()
                ),
            }
            if gate is not None:
                row.update(
                    {
                        "gate_mean": float(gate.mean()),
                        "gate_std": float(gate.std(unbiased=False)),
                    }
                )
            rows.append(row)
    return rows


def plot_endpoint_scatter(
    path: Path,
    *,
    generated: dict[str, torch.Tensor],
    reference_intrinsic: torch.Tensor,
    distribution: ContinuousSpiralDistribution,
    limit: int,
) -> None:
    preferred = [
        "Reference_resample",
        "Bayes_exact",
        "B0_v_ind",
        "B1_x_ind",
        "B2_eps_ind",
        "D0_x_shared",
        "D0_eps_shared",
        "D0_fixed_x_eps",
        "D0_safe_schedule",
        "D1_scaled_gate",
        "D2_velocity_gate",
        "D3_oracle_bayes_gate",
        "D4_safe_velocity_gate",
        "S0_xv_switch",
        "S1_xv_consistency_switch",
    ]
    names = [name for name in preferred if name in generated]
    columns = 4
    rows = math.ceil((len(names) + 1) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(18, 4.5 * rows), squeeze=False)
    panels: list[tuple[str, np.ndarray]] = [
        ("Reference", reference_intrinsic[:limit].cpu().numpy())
    ]
    for name in names:
        intrinsic = distribution.decode_intrinsic(
            generated[name].to(distribution.device)
        )
        panels.append((name, intrinsic[:limit].cpu().numpy()))
    for axis, (name, points) in zip(axes.flat, panels):
        axis.scatter(points[:, 0], points[:, 1], s=4, alpha=0.35, rasterized=True)
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.set_xlim(-1.8, 1.8)
        axis.set_ylim(-1.8, 1.8)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_mechanism(
    path: Path,
    *,
    endpoint_rows: Sequence[dict],
    teacher_rows: Sequence[dict],
    rollout_rows: Sequence[dict],
) -> None:
    selected = {
        "B0_v_ind",
        "B1_x_ind",
        "D1_scaled_gate",
        "D2_velocity_gate",
        "D3_oracle_bayes_gate",
        "D4_safe_velocity_gate",
        "S1_xv_consistency_switch",
    }
    figure, axes = plt.subplots(1, 3, figsize=(21, 6))
    endpoint_sorted = sorted(endpoint_rows, key=lambda row: row["swd_fullD"])
    axes[0].barh(
        [row["condition"] for row in endpoint_sorted],
        [row["swd_fullD"] for row in endpoint_sorted],
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("v10 full-D SWD (lower is better)")
    axes[0].invert_yaxis()
    for condition in sorted(selected):
        teacher = sorted(
            (row for row in teacher_rows if row["condition"] == condition),
            key=lambda row: row["time"],
        )
        rollout = sorted(
            (row for row in rollout_rows if row["condition"] == condition),
            key=lambda row: row["time"],
        )
        if teacher:
            axes[1].plot(
                [row["time"] for row in teacher],
                [row["bayes_velocity_mse"] for row in teacher],
                marker="o",
                label=condition,
            )
        if rollout:
            axes[2].plot(
                [row["time"] for row in rollout],
                [row["rollout_bayes_velocity_mse"] for row in rollout],
                marker="o",
                label=condition,
            )
    axes[1].set_title("Teacher states")
    axes[2].set_title("Rollout states")
    for axis in axes[1:]:
        axis.set_yscale("log")
        axis.set_xlabel("t")
        axis.set_ylabel("MSE to quadrature Bayes field")
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_setting(args: argparse.Namespace, ambient_dim: int, seed: int) -> Path:
    output_dir = args.output_root / f"seed{seed}" / f"D{ambient_dim}_H{args.hidden_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and not args.overwrite and not args.evaluation_only:
        print(f"skip complete setting: {output_dir}", flush=True)
        return output_dir

    core.set_seed(seed)
    device = torch.device(args.device)
    distribution = ContinuousSpiralDistribution(
        ambient_dim,
        data_jitter=args.data_jitter,
        quadrature_points=args.quadrature_points,
        locator_points=args.locator_points,
        frequency_scale=args.frequency_scale,
        embedding_seed=core.stable_seed(seed, ambient_dim, 71),
        device=device,
        scale_mode=args.scale_mode,
        curvature=args.curvature,
        bayes_batch_chunk=args.bayes_batch_chunk,
    )
    suite = core.build_model_suite(
        ambient_dim=ambient_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_dim=args.time_dim,
        mode_dim=args.mode_dim,
        model_ids=args.model_ids,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=core.stable_seed(seed, ambient_dim, 113),
        device=device,
    )
    if args.evaluation_only:
        checkpoint = torch.load(
            output_dir / "checkpoint.pt", map_location=device, weights_only=False
        )
        for model_id, model in suite.models.items():
            model.load_state_dict(checkpoint["models"][model_id])
    else:
        config = {
            **vars(args),
            "output_root": str(args.output_root),
            "ambient_dim": ambient_dim,
            "seed": seed,
            "device_resolved": str(device),
            "time_convention": "state=(1-t)*epsilon+t*clean; velocity=clean-epsilon",
            "data_protocol": "v4/v10 continuous spiral; unit-RMS curvature-0 embedding",
        }
        core.save_json(output_dir / "config.json", config)
        history = core.train_models(
            suite=suite,
            distribution=distribution,
            steps=args.train_steps,
            batch_size=args.batch_size,
            t_min=args.train_t_min,
            t_max=args.train_t_max,
            denominator_floor=args.denominator_floor,
            consistency_weight=args.consistency_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            seed=core.stable_seed(seed, ambient_dim, 127),
            checkpoint_path=output_dir / "checkpoint.pt",
        )
        core.save_csv(output_dir / "train_history.csv", history)
    for model in suite.models.values():
        model.eval()
    core.save_json(
        output_dir / "evaluation_config.json",
        {
            "ambient_dim": ambient_dim,
            "seed": seed,
            "data_protocol": "v4/v10 continuous spiral; unit-RMS curvature-0 embedding",
            "input_data_jitter_before_v4_scale": args.data_jitter,
            "intrinsic_jitter_std_after_v4_scale": distribution.intrinsic_jitter_std,
            "bayes_oracle": "continuous-s quadrature with the scaled intrinsic jitter",
            "quadrature_points": args.quadrature_points,
            "denominator_floor": args.denominator_floor,
            "sample_steps": args.sample_steps,
            "sample_count": args.sample_count,
            "reference_count": args.reference_count,
        },
    )

    conditions = core.available_conditions(args.model_ids)
    teacher_rows = core.teacher_diagnostics(
        suite=suite,
        distribution=distribution,
        conditions=conditions,
        times=args.diagnostic_times,
        samples=args.teacher_samples,
        denominator_floor=args.denominator_floor,
        seed=core.stable_seed(seed, ambient_dim, 131),
    )
    core.save_csv(output_dir / "teacher_metrics.csv", teacher_rows)
    core.save_csv(
        output_dir / "gradient_audit.csv",
        core.gradient_audit(
            suite=suite,
            distribution=distribution,
            samples=args.gradient_samples,
            seed=core.stable_seed(seed, ambient_dim, 137),
        ),
    )
    core.save_csv(
        output_dir / "branch_pair_metrics.csv",
        core.branch_pair_diagnostics(
            suite=suite,
            distribution=distribution,
            times=args.diagnostic_times,
            samples=args.teacher_samples,
            denominator_floor=args.denominator_floor,
            seed=core.stable_seed(seed, ambient_dim, 139),
        ),
    )

    cross_conditions = core.available_cross_gate_conditions(args.model_ids)
    cross_teacher_rows: list[dict] = []
    if cross_conditions:
        cross_teacher_rows = core.teacher_diagnostics(
            suite=suite,
            distribution=distribution,
            conditions=cross_conditions,
            times=args.diagnostic_times,
            samples=args.teacher_samples,
            denominator_floor=args.denominator_floor,
            seed=core.stable_seed(seed, ambient_dim, 131),
        )
        cross_teacher_rows = [
            row
            for row in cross_teacher_rows
            if row["condition"] != "D3_oracle_pair_teacher_only"
        ]

    generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(seed, ambient_dim, 149)
    )
    initial_noise = torch.randn(
        args.sample_count, ambient_dim, device=device, generator=generator
    )
    generated: dict[str, torch.Tensor] = {}
    snapshots: dict[str, dict[float, torch.Tensor]] = {}
    for condition in conditions:
        print(f"sampling {condition}", flush=True)
        endpoint, condition_snapshots = core.sample_heun(
            condition,
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=args.sample_steps,
            denominator_floor=args.denominator_floor,
            snapshot_times=args.diagnostic_times,
        )
        generated[condition] = endpoint.detach().cpu()
        snapshots[condition] = condition_snapshots

    cross_generated: dict[str, torch.Tensor] = {}
    cross_snapshots: dict[str, dict[float, torch.Tensor]] = {}
    for condition in cross_conditions:
        print(f"sampling cross-control {condition}", flush=True)
        endpoint, condition_snapshots = core.sample_heun(
            condition,
            suite=suite,
            distribution=distribution,
            initial_noise=initial_noise,
            steps=args.sample_steps,
            denominator_floor=args.denominator_floor,
            snapshot_times=args.diagnostic_times,
        )
        cross_generated[condition] = endpoint.detach().cpu()
        cross_snapshots[condition] = condition_snapshots

    reference_generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(seed, ambient_dim, 151)
    )
    reference, reference_u, _ = distribution.sample(
        args.reference_count, generator=reference_generator
    )
    resample_generator = torch.Generator(device=device.type).manual_seed(
        core.stable_seed(seed, ambient_dim, 153)
    )
    reference_resample, _, _ = distribution.sample(
        args.sample_count, generator=resample_generator
    )
    generated["Reference_resample"] = reference_resample.detach().cpu()
    reference_cpu = reference.detach().cpu()
    reference_u_cpu = reference_u.detach().cpu()

    metric_kwargs = {
        "reference": reference_cpu,
        "reference_intrinsic": reference_u_cpu,
        "distribution": distribution,
        "seed": core.stable_seed(seed, ambient_dim, 157),
        "swd_projections": args.swd_projections,
        "swd_max_points": args.swd_max_points,
        "full_swd_projections": args.full_swd_projections,
        "full_swd_max_points": args.full_swd_max_points,
        "mmd_max_points": args.mmd_max_points,
        "coverage_bins": args.coverage_bins,
        "conditional_ridge_bins": args.conditional_ridge_bins,
        "conditional_ridge_min_count": args.conditional_ridge_min_count,
    }
    endpoint_rows = endpoint_metrics_spiral(generated=generated, **metric_kwargs)
    core.save_csv(output_dir / "endpoint_metrics.csv", endpoint_rows)

    rollout_rows = rollout_diagnostics_spiral(
        snapshots=snapshots,
        suite=suite,
        distribution=distribution,
        denominator_floor=args.denominator_floor,
        seed=core.stable_seed(seed, ambient_dim, 163),
        swd_projections=min(args.swd_projections, 128),
    )
    core.save_csv(output_dir / "rollout_metrics.csv", rollout_rows)

    if cross_conditions:
        cross_metric_generated = {
            "D0_x_shared": generated["D0_x_shared"],
            **cross_generated,
        }
        cross_rows = endpoint_metrics_spiral(
            generated=cross_metric_generated,
            **metric_kwargs,
        )
        cross_rows = [row for row in cross_rows if row["condition"] != "D0_x_shared"]
        common = {"D0_x_shared", "D0_eps_shared", "D3_oracle_bayes_gate"}
        core.save_csv(
            output_dir / "cross_gate_endpoint_metrics.csv",
            [row for row in endpoint_rows if row["condition"] in common] + cross_rows,
        )
        core.save_csv(
            output_dir / "cross_gate_teacher_metrics.csv",
            [row for row in teacher_rows if row["condition"] in common]
            + cross_teacher_rows,
        )
        cross_rollout_rows = rollout_diagnostics_spiral(
            snapshots=cross_snapshots,
            suite=suite,
            distribution=distribution,
            denominator_floor=args.denominator_floor,
            seed=core.stable_seed(seed, ambient_dim, 163),
            swd_projections=min(args.swd_projections, 128),
        )
        core.save_csv(
            output_dir / "cross_gate_rollout_metrics.csv",
            [row for row in rollout_rows if row["condition"] in common]
            + cross_rollout_rows,
        )

    plot_endpoint_scatter(
        output_dir / "endpoint_scatter.png",
        generated=generated,
        reference_intrinsic=reference_u_cpu,
        distribution=distribution,
        limit=args.plot_points,
    )
    plot_mechanism(
        output_dir / "mechanism_summary.png",
        endpoint_rows=endpoint_rows,
        teacher_rows=teacher_rows,
        rollout_rows=rollout_rows,
    )
    core.save_json(
        complete_path,
        {
            "conditions": list(generated),
            "best_swd_2d": min(endpoint_rows, key=lambda row: row["swd_2d"]),
            "best_swd_fullD": min(endpoint_rows, key=lambda row: row["swd_fullD"]),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--dims", type=core.parse_int_list, default=core.parse_int_list("2,512"))
    parser.add_argument(
        "--seeds",
        type=core.parse_int_list,
        default=core.parse_int_list("20260831,20260901,20260902"),
    )
    parser.add_argument("--model-ids", type=core.parse_str_list, default=list(core.MODEL_IDS))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--mode-dim", type=int, default=8)
    parser.add_argument("--data-jitter", type=float, default=0.015)
    parser.add_argument("--curvature", type=float, default=0.0)
    parser.add_argument("--frequency-scale", type=float, default=6.0)
    parser.add_argument("--scale-mode", choices=("constant_norm", "unit_rms"), default="unit_rms")
    parser.add_argument("--quadrature-points", type=int, default=1024)
    parser.add_argument("--locator-points", type=int, default=4096)
    parser.add_argument("--bayes-batch-chunk", type=int, default=4096)
    parser.add_argument("--train-steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--train-t-min", type=float, default=0.001)
    parser.add_argument("--train-t-max", type=float, default=0.999)
    parser.add_argument("--denominator-floor", type=float, default=1e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--diagnostic-times",
        type=core.parse_float_list,
        default=core.parse_float_list("0.01,0.03,0.1,0.3,0.5,0.7,0.9,0.97,0.99"),
    )
    parser.add_argument("--teacher-samples", type=int, default=4096)
    parser.add_argument("--gradient-samples", type=int, default=2048)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--reference-count", type=int, default=8192)
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--swd-max-points", type=int, default=4096)
    parser.add_argument("--full-swd-projections", type=int, default=64)
    parser.add_argument("--full-swd-max-points", type=int, default=4096)
    parser.add_argument("--mmd-max-points", type=int, default=2048)
    parser.add_argument("--coverage-bins", type=int, default=32)
    parser.add_argument("--conditional-ridge-bins", type=int, default=16)
    parser.add_argument("--conditional-ridge-min-count", type=int, default=24)
    parser.add_argument("--plot-points", type=int, default=3000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.curvature != 0.0:
        parser.error("this exact-oracle experiment currently requires --curvature 0")
    return args


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for ambient_dim in args.dims:
            run_setting(args, ambient_dim, seed)


if __name__ == "__main__":
    main()
