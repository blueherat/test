#!/usr/bin/env python3
"""Closed-loop audit of local x/v selection on saved curved-toy models.

The audit deliberately does not train a selector.  It asks whether local
geometry is useful even when supplied by either a large clean estimator or
its posterior-response Jacobian.  Only if this upper-bound experiment works
is an amortized selector worth training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.posterior_response_projector import posterior_response_action
from experiments.run_prediction_target_extrapolation_toy_v4 import (
    CurvedEmbedding,
    DenoiseMLP,
    parse_int_list,
    sample_spiral_2d,
    stable_seed,
)
from experiments.run_prediction_target_rank_symmetry_toy import (
    evaluate_generation,
    plot_generation,
    save_csv,
)


CONDITIONS = (
    "native_x",
    "native_v",
    "large_x_oracle",
    "tangent_from_x",
    "tangent_from_large_x",
    "response_from_x",
    "response_from_large_x",
)


@dataclass(frozen=True)
class SavedModels:
    x: DenoiseMLP
    v: DenoiseMLP
    oracle: DenoiseMLP


def clean_from_v(
    state: torch.Tensor,
    time: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    return state - time[:, None] * velocity


def tangent_selected_clean(
    *,
    x_clean: torch.Tensor,
    v_clean: torch.Tensor,
    geometry_clean: torch.Tensor,
    embedding: CurvedEmbedding,
) -> torch.Tensor:
    """Use v in the estimated tangent and x in its orthogonal complement."""
    if not (x_clean.shape == v_clean.shape == geometry_clean.shape):
        raise ValueError("all clean estimates must have identical shapes")
    intrinsic = embedding.decode_intrinsic(geometry_clean)
    tangent, _normal = embedding.split_tangent_normal(
        v_clean - x_clean, intrinsic
    )
    return x_clean + tangent


def response_selected_clean(
    *,
    x_clean: torch.Tensor,
    v_clean: torch.Tensor,
    clean_estimator,
    state: torch.Tensor,
    time: torch.Tensor,
    relative_step: float,
) -> torch.Tensor:
    """Use the soft response alpha J_m to select the x/v disagreement."""
    alpha = 1.0 - time
    action = posterior_response_action(
        clean_estimator,
        state=state,
        time=time,
        direction=v_clean - x_clean,
        alpha=alpha,
        relative_step=relative_step,
    )
    return x_clean + action


def load_models(
    *,
    experiment_root: Path,
    seed: int,
    D: int,
    curvature: float,
    scale_mode: str,
    hidden: int,
    depth: int,
    time_dim: int,
    oracle_hidden: int,
    oracle_depth: int,
    device: torch.device,
) -> SavedModels:
    base = (
        experiment_root
        / f"seed{seed}"
        / f"D{D}"
        / f"curv{str(curvature).replace('.', 'p')}"
        / f"scale_{scale_mode}"
        / "loss_v"
    )
    checkpoint = torch.load(
        base / f"H{hidden}" / "models.pt",
        map_location=device,
        weights_only=True,
    )
    x_model = DenoiseMLP(D, hidden, depth, time_dim).to(device)
    v_model = DenoiseMLP(D, hidden, depth, time_dim).to(device)
    oracle = DenoiseMLP(D, oracle_hidden, oracle_depth, time_dim).to(device)
    x_model.load_state_dict(checkpoint["x"])
    v_model.load_state_dict(checkpoint["v"])
    oracle.load_state_dict(
        torch.load(base / "oracle_x.pt", map_location=device, weights_only=True)
    )
    for model in (x_model, v_model, oracle):
        model.eval().requires_grad_(False)
    return SavedModels(x=x_model, v=v_model, oracle=oracle)


@torch.inference_mode()
def condition_clean(
    *,
    condition: str,
    models: SavedModels,
    embedding: CurvedEmbedding,
    state: torch.Tensor,
    time: torch.Tensor,
    relative_step: float,
) -> torch.Tensor:
    x_clean = models.x(state, time)
    if condition == "native_x":
        return x_clean
    if condition == "large_x_oracle":
        return models.oracle(state, time)

    v_clean = clean_from_v(state, time, models.v(state, time))
    if condition == "native_v":
        return v_clean
    if condition == "tangent_from_x":
        return tangent_selected_clean(
            x_clean=x_clean,
            v_clean=v_clean,
            geometry_clean=x_clean,
            embedding=embedding,
        )

    if condition in {"tangent_from_large_x", "response_from_large_x"}:
        oracle_clean = models.oracle(state, time)
        if condition == "tangent_from_large_x":
            return tangent_selected_clean(
                x_clean=x_clean,
                v_clean=v_clean,
                geometry_clean=oracle_clean,
                embedding=embedding,
            )

        def oracle_estimator(
            perturbed_state: torch.Tensor, perturbed_time: torch.Tensor
        ) -> torch.Tensor:
            return models.oracle(perturbed_state, perturbed_time)

        return response_selected_clean(
            x_clean=x_clean,
            v_clean=v_clean,
            clean_estimator=oracle_estimator,
            state=state,
            time=time,
            relative_step=relative_step,
        )

    if condition == "response_from_x":

        def x_estimator(
            perturbed_state: torch.Tensor, perturbed_time: torch.Tensor
        ) -> torch.Tensor:
            return models.x(perturbed_state, perturbed_time)

        return response_selected_clean(
            x_clean=x_clean,
            v_clean=v_clean,
            clean_estimator=x_estimator,
            state=state,
            time=time,
            relative_step=relative_step,
        )
    raise ValueError(f"unknown condition: {condition}")


@torch.inference_mode()
def sample_conditions(
    *,
    models: SavedModels,
    embedding: CurvedEmbedding,
    sample_count: int,
    batch_size: int,
    sample_steps: int,
    t_max: float,
    t_min: float,
    relative_step: float,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    grid = torch.linspace(t_max, t_min, sample_steps + 1, device=device)
    for start in range(0, sample_count, batch_size):
        count = min(batch_size, sample_count - start)
        generator = torch.Generator(device=device.type).manual_seed(seed + start)
        initial = float(t_max) * torch.randn(
            count, embedding.D, generator=generator, device=device
        )
        states = {name: initial.clone() for name in CONDITIONS}
        for index in range(sample_steps):
            time_now, time_next = grid[index], grid[index + 1]
            time = time_now.expand(count)
            for name in CONDITIONS:
                clean = condition_clean(
                    condition=name,
                    models=models,
                    embedding=embedding,
                    state=states[name],
                    time=time,
                    relative_step=relative_step,
                )
                velocity = (states[name] - clean) / time_now
                states[name] = states[name] + (time_next - time_now) * velocity

        final_time = grid[-1].expand(count)
        for name in CONDITIONS:
            clean = condition_clean(
                condition=name,
                models=models,
                embedding=embedding,
                state=states[name],
                time=final_time,
                relative_step=relative_step,
            )
            collected[name].append(clean.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def run_seed(args: argparse.Namespace, seed: int, hidden: int, device: torch.device) -> list[dict]:
    embedding = CurvedEmbedding(
        args.D,
        curvature=args.curvature,
        frequency_scale=args.frequency_scale,
        seed=stable_seed(seed, args.D, int(args.curvature * 10_000), 41),
        device=device,
        scale_mode=args.scale_mode,
    )
    models = load_models(
        experiment_root=args.experiment_root,
        seed=seed,
        D=args.D,
        curvature=args.curvature,
        scale_mode=args.scale_mode,
        hidden=hidden,
        depth=args.depth,
        time_dim=args.time_dim,
        oracle_hidden=args.oracle_hidden,
        oracle_depth=args.oracle_depth,
        device=device,
    )
    samples = sample_conditions(
        models=models,
        embedding=embedding,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        t_max=args.t_max,
        t_min=args.t_min,
        relative_step=args.relative_step,
        seed=stable_seed(seed, args.D, hidden, 5101),
        device=device,
    )
    generator = torch.Generator(device=device.type).manual_seed(
        stable_seed(seed, args.D, hidden, 5102)
    )
    reference_intrinsic = sample_spiral_2d(
        max(2 * args.sample_count, 8192),
        device=device,
        jitter=args.data_jitter,
        generator=generator,
    ).cpu().numpy()
    rows = evaluate_generation(
        samples=samples,
        reference_intrinsic=reference_intrinsic,
        embedding=embedding,
        output_rank=hidden,
        seed=stable_seed(seed, args.D, hidden, 5103),
        device=device,
        metric_max_points=args.metric_max_points,
        projections=args.swd_projections,
        rank_dependent_randomness=False,
    )
    for row in rows:
        row["hidden"] = hidden
        row["relative_step"] = args.relative_step
        ambient = samples[str(row["condition"])]
        row["nonfinite_sample_fraction"] = float(
            1.0 - np.isfinite(ambient).all(axis=1).mean()
        )
    output_dir = args.output_root / f"seed{seed}" / f"H{hidden}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "generation_metrics.csv", rows)
    plot_generation(
        output_dir / "generation_scatter.png",
        samples,
        reference_intrinsic,
        embedding,
        args.plot_points,
    )
    np.savez_compressed(
        output_dir / "intrinsic_samples.npz",
        **{
            name: embedding.decode_intrinsic(torch.from_numpy(value).to(device))
            .cpu()
            .numpy()
            for name, value in samples.items()
        },
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/experiments/prediction_target_toy_v4_main"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260807,20260808,20260809"))
    parser.add_argument("--hiddens", type=parse_int_list, default=parse_int_list("128"))
    parser.add_argument("--D", type=int, default=512)
    parser.add_argument("--curvature", type=float, default=0.5)
    parser.add_argument("--scale-mode", choices=("constant_norm", "unit_rms"), default="unit_rms")
    parser.add_argument("--frequency-scale", type=float, default=6.0)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--oracle-hidden", type=int, default=2048)
    parser.add_argument("--oracle-depth", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--t-max", type=float, default=0.98)
    parser.add_argument("--t-min", type=float, default=0.02)
    parser.add_argument("--relative-step", type=float, default=0.01)
    parser.add_argument("--data-jitter", type=float, default=0.015)
    parser.add_argument("--metric-max-points", type=int, default=4096)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--plot-points", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.experiment_root = args.experiment_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not (0.0 < args.t_min < args.t_max < 1.0):
        raise ValueError("require 0 < t_min < t_max < 1")
    if args.relative_step <= 0:
        raise ValueError("relative_step must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "definition": "closed-loop local tangent and posterior-response x/v selection",
        "conditions": list(CONDITIONS),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    device = torch.device(args.device)
    all_rows: list[dict] = []
    for seed in args.seeds:
        for hidden in args.hiddens:
            rows = run_seed(args, seed, hidden, device)
            all_rows.extend(rows)
            compact = " ".join(
                f"{row['condition']}={float(row['swd_2d']):.4f}"
                for row in rows
            )
            print(f"[seed={seed} H={hidden}] {compact}", flush=True)
    save_csv(args.output_root / "generation_metrics.csv", all_rows)
    print(f"[done] {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
