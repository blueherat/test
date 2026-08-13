#!/usr/bin/env python3
"""Finite-strength exact gauge control on an analytic two-dimensional flow.

The target is a circular Gaussian mixture and the source is a standard
Gaussian.  Their linear interpolation path has an analytic score and Bayes
velocity.  We compare two equal-norm controls:

* ``gauge``: a 90-degree rotation of the exact score.  It satisfies
  ``div(p_t u_t) = 0`` exactly and therefore preserves every path marginal for
  any finite constant scale.
* ``active``: the exact score itself.  It has the same pointwise norm but
  generally changes the path density.

This is a calibration experiment for vector-field geometry, not a proposed
image-generation method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev, vmap


def ring_means(count: int, radius: float, *, dtype=torch.float64) -> torch.Tensor:
    if count < 2 or radius <= 0:
        raise ValueError("count must be at least two and radius must be positive")
    angles = torch.arange(count, dtype=dtype) * (2.0 * math.pi / count)
    return radius * torch.stack((angles.cos(), angles.sin()), dim=1)


def mixture_path_score_velocity(
    state: torch.Tensor,
    time_value: float | torch.Tensor,
    means: torch.Tensor,
    *,
    data_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact score and conditional velocity for the linear GMM path."""

    if state.ndim != 2 or state.shape[1] != 2 or means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("state and means must have shapes [N,2] and [K,2]")
    if data_std <= 0:
        raise ValueError("data_std must be positive")
    time_tensor = torch.as_tensor(time_value, device=state.device, dtype=state.dtype)
    if time_tensor.numel() != 1 or not 0.0 <= float(time_tensor) <= 1.0:
        raise ValueError("time_value must lie in [0,1]")
    means = means.to(device=state.device, dtype=state.dtype)
    variance = (1.0 - time_tensor).square() + time_tensor.square() * data_std**2
    residual = state[:, None, :] - time_tensor * means[None, :, :]
    logits = -0.5 * residual.square().sum(dim=-1) / variance
    posterior = logits.softmax(dim=1)
    component_score = -residual / variance
    score = (posterior[..., None] * component_score).sum(dim=1)

    conditional_x = means[None, :, :] + (
        time_tensor * data_std**2 / variance
    ) * residual
    conditional_noise = ((1.0 - time_tensor) / variance) * residual
    component_velocity = conditional_x - conditional_noise
    velocity = (posterior[..., None] * component_velocity).sum(dim=1)
    return score, velocity


def rotate_90(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((-value[..., 1], value[..., 0]), dim=-1)


def controlled_velocity(
    state: torch.Tensor,
    time_value: float | torch.Tensor,
    means: torch.Tensor,
    *,
    data_std: float,
    kind: str,
    gamma: float,
) -> torch.Tensor:
    score, velocity = mixture_path_score_velocity(
        state,
        time_value,
        means,
        data_std=data_std,
    )
    if kind == "baseline" or gamma == 0.0:
        return velocity
    if kind == "gauge":
        control = rotate_90(score)
    elif kind == "active":
        control = score
    else:
        raise ValueError(f"unknown control kind: {kind}")
    return velocity + float(gamma) * control


def rk4_integrate(
    initial: torch.Tensor,
    field,
    *,
    steps: int,
    collect_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    state = initial.clone()
    traces = [state[collect_indices].detach().cpu()] if collect_indices is not None else None
    step = 1.0 / steps
    for index in range(steps):
        time_value = index * step
        k1 = field(state, time_value)
        k2 = field(state + 0.5 * step * k1, time_value + 0.5 * step)
        k3 = field(state + 0.5 * step * k2, time_value + 0.5 * step)
        k4 = field(state + step * k3, time_value + step)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if traces is not None and ((index + 1) % max(1, steps // 100) == 0):
            traces.append(state[collect_indices].detach().cpu())
    trace_tensor = torch.stack(traces) if traces is not None else None
    return state, trace_tensor


def sample_target(
    count: int,
    means: torch.Tensor,
    data_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    labels = torch.randint(len(means), (count,), generator=generator)
    noise = torch.randn(count, 2, generator=generator, dtype=means.dtype)
    return means[labels] + float(data_std) * noise


def sliced_wasserstein_2d(
    left: torch.Tensor,
    right: torch.Tensor,
    directions: torch.Tensor,
) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError("left and right must have equal [N,2] shapes")
    left_projection = (left @ directions.T).sort(dim=0).values
    right_projection = (right @ directions.T).sort(dim=0).values
    return float((left_projection - right_projection).square().mean().sqrt())


def target_negative_log_likelihood(
    samples: torch.Tensor,
    means: torch.Tensor,
    data_std: float,
) -> float:
    variance = float(data_std) ** 2
    residual = samples[:, None, :] - means[None, :, :]
    log_component = (
        -0.5 * residual.square().sum(dim=-1) / variance
        - math.log(2.0 * math.pi * variance)
        - math.log(len(means))
    )
    return float(-torch.logsumexp(log_component, dim=1).mean())


def mode_metrics(samples: torch.Tensor, means: torch.Tensor) -> tuple[float, float]:
    assignment = torch.cdist(samples, means).argmin(dim=1)
    histogram = torch.bincount(assignment, minlength=len(means)).double()
    probability = histogram / histogram.sum()
    uniform = torch.full_like(probability, 1.0 / len(means))
    total_variation = 0.5 * (probability - uniform).abs().sum()
    entropy = -(probability.clamp_min(torch.finfo(probability.dtype).tiny).log() * probability).sum()
    return float(total_variation), float(entropy / math.log(len(means)))


def exact_density_action(
    states: torch.Tensor,
    time_value: float,
    means: torch.Tensor,
    data_std: float,
    kind: str,
) -> torch.Tensor:
    """Compute ``div(u) + u dot score`` with the exact 2-D Jacobian."""

    def single_control(single_state: torch.Tensor) -> torch.Tensor:
        score, _ = mixture_path_score_velocity(
            single_state.unsqueeze(0),
            time_value,
            means,
            data_std=data_std,
        )
        return rotate_90(score[0]) if kind == "gauge" else score[0]

    score, _ = mixture_path_score_velocity(
        states,
        time_value,
        means,
        data_std=data_std,
    )
    controls = vmap(single_control)(states)
    jacobians = vmap(jacrev(single_control))(states)
    divergence = jacobians.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return divergence + (controls * score).sum(dim=-1)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    reference: torch.Tensor,
    endpoints: dict[str, torch.Tensor],
    traces: dict[str, torch.Tensor],
    output: Path,
) -> None:
    conditions = ["baseline", "gauge_g1", "gauge_g2", "active_g0.5", "active_g1"]
    figure, axes = plt.subplots(2, len(conditions) + 1, figsize=(20, 7.4))
    plot_values = [("reference", reference), *[(key, endpoints[key]) for key in conditions]]
    for axis, (name, values) in zip(axes[0], plot_values, strict=True):
        array = values[:5000].numpy()
        axis.scatter(array[:, 0], array[:, 1], s=2, alpha=0.25, rasterized=True)
        axis.set_title(name)
        axis.set_xlim(-4.2, 4.2)
        axis.set_ylim(-4.2, 4.2)
        axis.set_aspect("equal")
    axes[1, 0].axis("off")
    for axis, key in zip(axes[1, 1:], conditions, strict=True):
        trace = traces[key].numpy()
        for sample_index in range(trace.shape[1]):
            axis.plot(trace[:, sample_index, 0], trace[:, sample_index, 1], linewidth=0.65, alpha=0.6)
        axis.set_title(f"{key} trajectories")
        axis.set_xlim(-4.2, 4.2)
        axis.set_ylim(-4.2, 4.2)
        axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    if args.samples <= 0 or args.steps <= 0 or args.modes < 2:
        raise ValueError("samples/steps must be positive and modes at least two")
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    means_cpu = ring_means(args.modes, args.radius, dtype=dtype)
    means = means_cpu.to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial = torch.randn(args.samples, 2, generator=generator, dtype=dtype).to(device)
    reference = sample_target(
        args.samples,
        means_cpu,
        args.data_std,
        torch.Generator(device="cpu").manual_seed(args.seed + 1),
    )
    reference_replication = sample_target(
        args.samples,
        means_cpu,
        args.data_std,
        torch.Generator(device="cpu").manual_seed(args.seed + 3),
    )
    angles = torch.linspace(0.0, math.pi, args.projections + 1, dtype=dtype)[:-1]
    directions = torch.stack((angles.cos(), angles.sin()), dim=1)
    trace_indices = torch.arange(min(args.trace_samples, args.samples), device=device)
    specifications = [
        ("baseline", "baseline", 0.0),
        *[(f"gauge_g{gamma:g}", "gauge", gamma) for gamma in args.gammas],
        *[(f"active_g{gamma:g}", "active", gamma) for gamma in args.gammas],
    ]
    endpoints: dict[str, torch.Tensor] = {}
    traces: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for name, kind, gamma in specifications:
        endpoint, trace = rk4_integrate(
            initial,
            lambda state, time_value, k=kind, g=gamma: controlled_velocity(
                state,
                time_value,
                means,
                data_std=args.data_std,
                kind=k,
                gamma=g,
            ),
            steps=args.steps,
            collect_indices=trace_indices,
        )
        endpoint_cpu = endpoint.detach().cpu()
        endpoints[name] = endpoint_cpu
        assert trace is not None
        traces[name] = trace
        mode_tv, mode_entropy = mode_metrics(endpoint_cpu, means_cpu)
        baseline_endpoint = endpoints.get("baseline", endpoint_cpu)
        rows.append(
            {
                "condition": name,
                "kind": kind,
                "gamma": gamma,
                "swd_to_reference": sliced_wasserstein_2d(
                    endpoint_cpu, reference, directions
                ),
                "target_nll": target_negative_log_likelihood(
                    endpoint_cpu, means_cpu, args.data_std
                ),
                "mode_total_variation": mode_tv,
                "normalized_mode_entropy": mode_entropy,
                "paired_rms_from_baseline": float(
                    (endpoint_cpu - baseline_endpoint).square().mean().sqrt()
                ),
                "mean_radius": float(endpoint_cpu.square().sum(dim=1).sqrt().mean()),
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    source_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    action_rows: list[dict[str, object]] = []
    for time_value in (0.1, 0.5, 0.9):
        clean = sample_target(args.action_samples, means_cpu, args.data_std, source_generator)
        noise = torch.randn(args.action_samples, 2, generator=source_generator, dtype=dtype)
        states = ((1.0 - time_value) * noise + time_value * clean).to(device)
        for kind in ("gauge", "active"):
            action = exact_density_action(
                states,
                time_value,
                means,
                args.data_std,
                kind,
            )
            action_rows.append(
                {
                    "time": time_value,
                    "kind": kind,
                    "action_rms": float(action.square().mean().sqrt().cpu()),
                    "action_abs_max": float(action.abs().max().cpu()),
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "distribution_metrics.csv")
    _write_csv(action_rows, output_dir / "exact_density_action.csv")
    np.savez(
        output_dir / "endpoints.npz",
        reference=reference.numpy(),
        **{key: value.numpy() for key, value in endpoints.items()},
    )
    _plot(reference, endpoints, traces, output_dir / "finite_gauge_toy.png")
    summary = {
        "format": "eqvae_finite_guidance_exact_gauge_toy_v1",
        "statement": (
            "gauge control R(score) obeys div(pu)=0 exactly; active control score "
            "has identical pointwise norm but changes density"
        ),
        "samples": args.samples,
        "steps": args.steps,
        "gammas": args.gammas,
        "seed": args.seed,
        "dtype": args.dtype,
        "modes": args.modes,
        "radius": args.radius,
        "data_std": args.data_std,
        "finite_sample_calibration": {
            "reference_replication_swd": sliced_wasserstein_2d(
                reference, reference_replication, directions
            ),
            "reference_target_nll": target_negative_log_likelihood(
                reference, means_cpu, args.data_std
            ),
            "replication_target_nll": target_negative_log_likelihood(
                reference_replication, means_cpu, args.data_std
            ),
        },
        "distribution_metrics": rows,
        "density_action": action_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _parse_floats(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",") if item.strip()]
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated values")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/experiments/finite_guidance_exact_gauge_toy"
        ),
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gammas", type=_parse_floats, default=_parse_floats("0.5,1,2"))
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--data-std", type=float, default=0.18)
    parser.add_argument("--projections", type=int, default=256)
    parser.add_argument("--trace-samples", type=int, default=24)
    parser.add_argument("--action-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
