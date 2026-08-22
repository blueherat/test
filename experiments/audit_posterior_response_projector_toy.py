#!/usr/bin/env python3
"""Audit a transferable, matrix-free projector on saved curved-toy models.

The projector is not fitted to the known toy embedding.  It is defined by the
local response of a frozen clean estimator and estimated with model forwards:

    P_theta a = (1-t) J_m(z,t) a.

The known manifold tangent is used only after estimation to measure alignment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.posterior_response_projector import (
    posterior_response_action,
    posterior_response_basis,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    DenoiseMLP,
    parse_float_list,
    parse_int_list,
    sample_spiral_2d,
    stable_seed,
    velocity_from_output,
)
from experiments.run_prediction_target_rank_symmetry_toy import save_csv


@dataclass
class EstimatorSource:
    name: str
    kind: str
    model: DenoiseMLP

    def clean(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        output = self.model(state, time)
        if self.kind in {"oracle_x", "x"}:
            return output
        if self.kind == "v":
            return state - time[:, None] * output
        raise ValueError(self.kind)


def row_project(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "bdi,bi->bd", basis, torch.einsum("bdi,bd->bi", basis, value)
    )


def summarize(values: list[torch.Tensor], prefix: str) -> dict[str, float]:
    tensor = torch.cat([value.detach().float().cpu().flatten() for value in values])
    return {
        f"{prefix}_mean": float(tensor.mean()),
        f"{prefix}_median": float(tensor.median()),
        f"{prefix}_q10": float(torch.quantile(tensor, 0.1)),
        f"{prefix}_q90": float(torch.quantile(tensor, 0.9)),
    }


def load_sources(
    *,
    args: argparse.Namespace,
    hidden: int,
    device: torch.device,
) -> tuple[list[EstimatorSource], DenoiseMLP, DenoiseMLP]:
    base = (
        args.experiment_root
        / f"seed{args.seed}"
        / f"D{args.D}"
        / f"curv{str(args.curvature).replace('.', 'p')}"
        / f"scale_{args.scale_mode}"
        / "loss_v"
    )
    oracle = DenoiseMLP(
        args.D, args.oracle_hidden, args.oracle_depth, args.time_dim
    ).to(device)
    oracle.load_state_dict(
        torch.load(base / "oracle_x.pt", map_location=device, weights_only=True)
    )
    checkpoint = torch.load(
        base / f"H{hidden}" / "models.pt", map_location=device, weights_only=True
    )
    x_model = DenoiseMLP(args.D, hidden, args.depth, args.time_dim).to(device)
    v_model = DenoiseMLP(args.D, hidden, args.depth, args.time_dim).to(device)
    x_model.load_state_dict(checkpoint["x"])
    v_model.load_state_dict(checkpoint["v"])
    for model in (oracle, x_model, v_model):
        model.eval()
        model.requires_grad_(False)
    return (
        [
            EstimatorSource("oracle_x", "oracle_x", oracle),
            EstimatorSource(f"x_H{hidden}", "x", x_model),
            EstimatorSource(f"v_to_x_H{hidden}", "v", v_model),
        ],
        x_model,
        v_model,
    )


@torch.no_grad()
def run_source_time(
    *,
    source: EstimatorSource,
    x_model: DenoiseMLP,
    v_model: DenoiseMLP,
    embedding: CurvedEmbedding,
    hidden: int,
    time_value: float,
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> dict:
    accumulators: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "tangent_capture",
            "principal_cosine_min",
            "response_gap",
            "gap_action_cosine",
            "gap_action_relative_error",
            "gap_action_normal_fraction",
            "gap_action_norm_ratio",
            "idempotence_relative_error",
            "symmetry_relative_error",
        )
    }
    for start in range(0, args.samples, args.batch_size):
        n = min(args.batch_size, args.samples - start)
        intrinsic = sample_spiral_2d(
            n,
            device=device,
            jitter=args.data_jitter,
            generator=generator,
        )
        clean = embedding.embed(intrinsic)
        epsilon = torch.randn(clean.shape, device=device, generator=generator)
        time = torch.full((n,), float(time_value), device=device)
        alpha = 1.0 - time
        state = alpha[:, None] * clean + time[:, None] * epsilon
        true_basis = embedding.tangent_basis(intrinsic)

        estimated_basis, singular_values = posterior_response_basis(
            source.clean,
            state=state,
            time=time,
            alpha=alpha,
            probes=args.probes,
            rank=args.intrinsic_dim,
            relative_step=args.relative_step,
            generator=generator,
        )
        overlap = torch.einsum("bdi,bdj->bij", true_basis, estimated_basis)
        principal_cosines = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
        accumulators["tangent_capture"].append(principal_cosines.square().mean(dim=1))
        accumulators["principal_cosine_min"].append(principal_cosines.min(dim=1).values)
        accumulators["response_gap"].append(
            singular_values[:, args.intrinsic_dim - 1]
            / singular_values[:, args.intrinsic_dim].clamp_min(1e-12)
        )

        x_output = x_model(state, time)
        v_output = v_model(state, time)
        x_velocity = velocity_from_output(
            x_output, state, time, "x", args.conversion_clip
        )
        gap = v_output - x_velocity
        true_gap_action = row_project(gap, true_basis)
        estimated_gap_action = posterior_response_action(
            source.clean,
            state=state,
            time=time,
            direction=gap,
            alpha=alpha,
            relative_step=args.relative_step,
        )
        estimated_twice = posterior_response_action(
            source.clean,
            state=state,
            time=time,
            direction=estimated_gap_action,
            alpha=alpha,
            relative_step=args.relative_step,
        )
        estimated_gap_tangent = row_project(estimated_gap_action, true_basis)
        estimated_gap_normal = estimated_gap_action - estimated_gap_tangent
        accumulators["gap_action_cosine"].append(
            F.cosine_similarity(
                estimated_gap_action.double(), true_gap_action.double(), dim=1, eps=1e-12
            ).float()
        )
        accumulators["gap_action_relative_error"].append(
            (estimated_gap_action - true_gap_action).norm(dim=1)
            / true_gap_action.norm(dim=1).clamp_min(1e-8)
        )
        accumulators["gap_action_normal_fraction"].append(
            estimated_gap_normal.square().sum(dim=1)
            / estimated_gap_action.square().sum(dim=1).clamp_min(1e-12)
        )
        accumulators["gap_action_norm_ratio"].append(
            estimated_gap_action.norm(dim=1)
            / true_gap_action.norm(dim=1).clamp_min(1e-8)
        )
        accumulators["idempotence_relative_error"].append(
            (estimated_twice - estimated_gap_action).norm(dim=1)
            / estimated_gap_action.norm(dim=1).clamp_min(1e-8)
        )

        probe_a = torch.randn(gap.shape, device=device, generator=generator)
        probe_b = torch.randn(gap.shape, device=device, generator=generator)
        action_a = posterior_response_action(
            source.clean,
            state=state,
            time=time,
            direction=probe_a,
            alpha=alpha,
            relative_step=args.relative_step,
        )
        action_b = posterior_response_action(
            source.clean,
            state=state,
            time=time,
            direction=probe_b,
            alpha=alpha,
            relative_step=args.relative_step,
        )
        bilinear_ab = (probe_a.double() * action_b.double()).sum(dim=1)
        bilinear_ba = (action_a.double() * probe_b.double()).sum(dim=1)
        symmetry_scale = 0.5 * (
            probe_a.double().norm(dim=1) * action_b.double().norm(dim=1)
            + probe_b.double().norm(dim=1) * action_a.double().norm(dim=1)
        ).clamp_min(1e-8)
        accumulators["symmetry_relative_error"].append(
            ((bilinear_ab - bilinear_ba).abs() / symmetry_scale).float()
        )

    row = {
        "seed": args.seed,
        "D": args.D,
        "curvature": args.curvature,
        "hidden": hidden,
        "time": time_value,
        "source": source.name,
        "samples": args.samples,
        "probes": args.probes,
        "relative_step": args.relative_step,
    }
    for key, values in accumulators.items():
        row.update(summarize(values, key))
    return row


def plot_metric(rows: list[dict], path: Path, metric: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for label in sorted({row["source"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["source"] == label), key=lambda row: row["time"]
        )
        axis.plot(
            [row["time"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            label=label,
        )
    axis.set_xlabel("noise interpolation time t (0=data, 1=noise)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment-root", type=Path, default=Path("/home/zhoushunyu/data/eqvae/experiments/prediction_target_toy_v4_main"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--D", type=int, default=512)
    parser.add_argument("--curvature", type=float, default=0.5)
    parser.add_argument("--hiddens", type=parse_int_list, default=parse_int_list("128,512,1024"))
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--oracle-hidden", type=int, default=2048)
    parser.add_argument("--oracle-depth", type=int, default=6)
    parser.add_argument("--intrinsic-dim", type=int, default=2)
    parser.add_argument("--times", type=parse_float_list, default=parse_float_list("0.02,0.05,0.1,0.2,0.5,0.8"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--relative-step", type=float, default=0.01)
    parser.add_argument("--conversion-clip", type=float, default=0.02)
    parser.add_argument("--frequency-scale", type=float, default=6.0)
    parser.add_argument("--scale-mode", choices=("constant_norm", "unit_rms"), default="unit_rms")
    parser.add_argument("--data-jitter", type=float, default=0.015)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.experiment_root = args.experiment_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    embedding = CurvedEmbedding(
        args.D,
        curvature=args.curvature,
        frequency_scale=args.frequency_scale,
        seed=stable_seed(args.seed, args.D, int(args.curvature * 10000), 41),
        device=device,
        scale_mode=args.scale_mode,
    )
    rows: list[dict] = []
    for hidden in args.hiddens:
        sources, x_model, v_model = load_sources(
            args=args, hidden=hidden, device=device
        )
        for time_index, time_value in enumerate(args.times):
            for source in sources:
                generator = torch.Generator(device=device.type).manual_seed(
                    stable_seed(args.seed, time_index, 8009)
                )
                row = run_source_time(
                    source=source,
                    x_model=x_model,
                    v_model=v_model,
                    embedding=embedding,
                    hidden=hidden,
                    time_value=float(time_value),
                    args=args,
                    generator=generator,
                    device=device,
                )
                rows.append(row)
                print(
                    f"[audit H={hidden} t={time_value:g} {source.name}] "
                    f"capture={row['tangent_capture_mean']:.4f} "
                    f"gap_cos={row['gap_action_cosine_mean']:.4f} "
                    f"idem={row['idempotence_relative_error_median']:.4f}",
                    flush=True,
                )
        del sources, x_model, v_model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    save_csv(args.output_root / "posterior_response_metrics.csv", rows)
    plot_metric(
        rows,
        args.output_root / "tangent_capture.png",
        "tangent_capture_mean",
        "mean captured true-tangent energy",
    )
    plot_metric(
        rows,
        args.output_root / "gap_action_cosine.png",
        "gap_action_cosine_mean",
        "cos(P_theta gap, P_true gap)",
    )
    plot_metric(
        rows,
        args.output_root / "idempotence_error.png",
        "idempotence_relative_error_median",
        "median ||P(Pa)-Pa|| / ||Pa||",
    )
    manifest = {
        "definition": "matrix-free P_theta a=(1-t) J_clean(z,t) a",
        "truth_usage": "known tangent used only for post-hoc metrics",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[done] {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
