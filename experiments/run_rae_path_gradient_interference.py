"""No-training gradient-interference audit for RAE path subspaces."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_layerwise_path import plan_layerwise_path, spatial_center  # noqa: E402
from experiments.train_rae_layerwise_path import (  # noqa: E402
    active_path_mode,
    configure_determinism,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


BASELINE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
CANDIDATE_ROOT = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_path_gradient_interference"
CROSSOVER_ROOT = Path.home() / "data/eqvae/experiments/rae_path_crossover_train_v2"
CONDITIONS = (
    ("static", BASELINE_ROOT / "seed3407_static_rank16_s0_to_10000"),
    ("annealed", BASELINE_ROOT / "seed3407_annealed_rank16_s0_to_10000"),
    ("floor020_p2", CANDIDATE_ROOT / "seed3407_floor020_p2_rank16_s0_to_2000"),
)
CROSSOVER_CONDITIONS = tuple(
    (
        condition,
        CROSSOVER_ROOT / f"seed3407_{condition}_rank16_s2000_to_5000",
    )
    for condition in (
        "floor_to_floor",
        "floor_to_static",
        "static_to_static",
        "static_to_floor",
    )
)
CHECKPOINTS = (2000, 5000)
TIMES = (0.97, 0.85, 0.70, 0.50, 0.30, 0.10)


def basis_projection(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project spatial residuals onto a channel basis."""

    _, residual = spatial_center(value.float())
    basis = basis.to(value)
    rows = residual.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
    projected = (rows @ basis) @ basis.transpose(0, 1)
    return projected.reshape(
        value.shape[0], value.shape[2], value.shape[3], value.shape[1]
    ).permute(0, 3, 1, 2).contiguous()


def component_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = prediction.float() - target.float()
    basis_error = basis_projection(error, basis)
    semantic_error = error - basis_error
    return semantic_error.square().mean(), basis_error.square().mean()


def gradient_pair_metrics(
    semantic: torch.Tensor,
    basis: torch.Tensor,
    *,
    eps: float = 1e-20,
) -> dict[str, float]:
    semantic = semantic.float().flatten()
    basis = basis.float().flatten()
    semantic_norm = torch.linalg.vector_norm(semantic).clamp_min(eps)
    basis_norm = torch.linalg.vector_norm(basis).clamp_min(eps)
    cross = torch.dot(semantic, basis)
    semantic_self = semantic_norm.square()
    basis_self = basis_norm.square()
    return {
        "semantic_gradient_norm": float(semantic_norm),
        "basis_gradient_norm": float(basis_norm),
        "basis_over_semantic_norm": float(basis_norm / semantic_norm),
        "semantic_basis_cosine": float(cross / (semantic_norm * basis_norm)),
        "semantic_descent_ratio": float((semantic_self + cross) / semantic_self),
        "basis_descent_ratio": float((basis_self + cross) / basis_self),
        "cross_over_semantic_self": float(cross / semantic_self),
        "cross_over_basis_self": float(cross / basis_self),
    }


def cross_split_metrics(
    semantic_calibration: torch.Tensor,
    basis_calibration: torch.Tensor,
    semantic_test: torch.Tensor,
    basis_test: torch.Tensor,
    *,
    eps: float = 1e-20,
) -> dict[str, float]:
    def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.dot(left, right) / (
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        ).clamp_min(eps)

    semantic_calibration = semantic_calibration.float().flatten()
    basis_calibration = basis_calibration.float().flatten()
    semantic_test = semantic_test.float().flatten()
    basis_test = basis_test.float().flatten()
    semantic_baseline = torch.dot(semantic_test, semantic_calibration)
    semantic_with_basis = torch.dot(
        semantic_test, semantic_calibration + basis_calibration
    )
    return {
        "semantic_direction_stability": float(
            cosine(semantic_calibration, semantic_test)
        ),
        "basis_direction_stability": float(cosine(basis_calibration, basis_test)),
        "cross_split_semantic_basis_cosine": float(
            cosine(semantic_test, basis_calibration)
        ),
        "cross_split_semantic_descent_ratio": float(
            semantic_with_basis / semantic_baseline.abs().clamp_min(eps)
        ),
        "cross_split_semantic_baseline_dot": float(semantic_baseline),
    }


def _selected_parameter_groups(
    model: torch.nn.Module,
) -> tuple[tuple[torch.nn.Parameter, ...], dict[str, list[int]], int]:
    block_indices = []
    for name, _ in model.named_parameters():
        match = re.match(r"blocks\.(\d+)\.", name)
        if match:
            block_indices.append(int(match.group(1)))
    if not block_indices:
        raise RuntimeError("stage-2 model has no named transformer blocks")
    last_block = max(block_indices)
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith(f"blocks.{last_block}.") or name in (
            "final_layer.linear.weight",
            "final_layer.linear.bias",
        ):
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    names = [name for name, _ in selected]
    groups = {
        "last_block": [
            index
            for index, name in enumerate(names)
            if name.startswith(f"blocks.{last_block}.")
        ],
        "output_head": [
            index
            for index, name in enumerate(names)
            if name.startswith("final_layer.linear.")
        ],
        "all_selected": list(range(len(names))),
    }
    if any(not indices for indices in groups.values()):
        raise RuntimeError("failed to select gradient parameter groups")
    return tuple(parameter for _, parameter in selected), groups, last_block


def _flatten_gradients(
    gradients: Sequence[torch.Tensor], indices: Sequence[int]
) -> torch.Tensor:
    return torch.cat([gradients[index].reshape(-1) for index in indices])


def _load_basis(manifest: dict[str, object]) -> torch.Tensor:
    payload = torch.load(
        Path(str(manifest["subspace_path"])), map_location="cpu", weights_only=False
    )
    rank = int(manifest["subspace_rank"])
    entry = payload["subspaces"].get(rank, payload["subspaces"].get(str(rank)))
    return entry["basis"].float().contiguous()


def _load_online_model(
    branch: Path, step: int, device: torch.device
) -> torch.nn.Module:
    config = OmegaConf.load(branch / "config.yaml")
    model = instantiate_from_config(config.stage_2).to(device=device, dtype=torch.float32)
    payload = torch.load(
        branch / f"checkpoints/step-{int(step):07d}.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(payload["step"]) != int(step):
        raise ValueError("checkpoint step mismatch")
    model.load_state_dict(payload["model"], strict=True)
    del payload
    return model.requires_grad_(False).eval()


def _path_kwargs(
    manifest: dict[str, object],
    *,
    step: int | None = None,
    mode_override: str | None = None,
) -> dict[str, object]:
    if mode_override is not None:
        mode = mode_override
    elif step is not None:
        mode = active_path_mode(
            step,
            str(manifest["path_mode"]),
            manifest.get("path_switch_step"),
            manifest.get("path_mode_after_switch"),
        )
    else:
        mode = str(manifest["path_mode"])
    return {
        "mode": mode,
        "power": float(manifest["path_power"]),
        "family": str(manifest.get("path_family", "power")),
        "floor": float(manifest.get("path_floor", 0.0)),
        "alpha": float(manifest.get("path_alpha", 1.0)),
        "detail_scale": float(manifest["detail_scale"]),
    }


def _zero_accumulators(
    parameters: Sequence[torch.Tensor], device: torch.device
) -> dict[str, list[torch.Tensor]]:
    return {
        component: [
            torch.zeros_like(parameter, device=device) for parameter in parameters
        ]
        for component in ("semantic", "basis")
    }


def _add_gradients(
    accumulator: list[torch.Tensor],
    gradients: Sequence[torch.Tensor],
    weight: float,
) -> None:
    for target, gradient in zip(accumulator, gradients):
        target.add_(gradient.detach(), alpha=float(weight))


def audit_checkpoint(
    condition: str,
    branch: Path,
    step: int,
    clean: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
    basis: torch.Tensor,
    manifest: dict[str, object],
    *,
    batch_size: int,
    times: Sequence[float],
    device: torch.device,
    path_mode_override: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    model = _load_online_model(branch, step, device)
    parameters, groups, last_block = _selected_parameter_groups(model)
    split_point = len(clean) // 2
    batch_rows = []
    aggregate_rows = []
    for time_value in times:
        accumulators = {
            "calibration": _zero_accumulators(parameters, device),
            "test": _zero_accumulators(parameters, device),
        }
        losses_by_split = {
            "calibration": {"semantic": 0.0, "basis": 0.0},
            "test": {"semantic": 0.0, "basis": 0.0},
        }
        counts = {"calibration": 0, "test": 0}
        for batch_index, start in enumerate(range(0, len(clean), batch_size)):
            end = min(start + batch_size, len(clean))
            if start < split_point < end:
                raise ValueError("batch boundary must not cross the calibration split")
            split = "calibration" if end <= split_point else "test"
            data = clean[start:end].to(device)
            batch_noise = noise[start:end].to(device)
            batch_labels = labels[start:end].to(device)
            time = torch.full(
                (len(data),), float(time_value), device=device, dtype=torch.float32
            )
            plan = plan_layerwise_path(
                data,
                batch_noise,
                time,
                basis,
                **_path_kwargs(
                    manifest,
                    step=step,
                    mode_override=path_mode_override,
                ),
            )
            prediction = model(plan.state, time, y=batch_labels)
            semantic_loss, basis_loss = component_losses(prediction, plan.target, basis)
            semantic_gradients = torch.autograd.grad(
                semantic_loss, parameters, retain_graph=True, create_graph=False
            )
            basis_gradients = torch.autograd.grad(
                basis_loss, parameters, retain_graph=False, create_graph=False
            )
            weight = len(data) / float(split_point)
            _add_gradients(
                accumulators[split]["semantic"], semantic_gradients, weight
            )
            _add_gradients(accumulators[split]["basis"], basis_gradients, weight)
            losses_by_split[split]["semantic"] += float(semantic_loss.detach()) * weight
            losses_by_split[split]["basis"] += float(basis_loss.detach()) * weight
            counts[split] += len(data)
            for group_name, indices in groups.items():
                semantic_vector = _flatten_gradients(semantic_gradients, indices)
                basis_vector = _flatten_gradients(basis_gradients, indices)
                batch_rows.append(
                    {
                        "condition": condition,
                        "checkpoint_step": int(step),
                        "time": float(time_value),
                        "split": split,
                        "batch_index": int(batch_index),
                        "parameter_group": group_name,
                        "batch_size": int(len(data)),
                        "semantic_loss": float(semantic_loss.detach()),
                        "basis_loss": float(basis_loss.detach()),
                        **gradient_pair_metrics(semantic_vector, basis_vector),
                    }
                )
            del prediction, semantic_gradients, basis_gradients
        if counts != {"calibration": split_point, "test": len(clean) - split_point}:
            raise RuntimeError("gradient audit split counts are incomplete")
        for group_name, indices in groups.items():
            vectors = {}
            for split in ("calibration", "test"):
                for component in ("semantic", "basis"):
                    vectors[(split, component)] = _flatten_gradients(
                        accumulators[split][component], indices
                    )
            semantic_all = 0.5 * (
                vectors[("calibration", "semantic")]
                + vectors[("test", "semantic")]
            )
            basis_all = 0.5 * (
                vectors[("calibration", "basis")]
                + vectors[("test", "basis")]
            )
            aggregate_rows.append(
                {
                    "condition": condition,
                    "checkpoint_step": int(step),
                    "time": float(time_value),
                    "parameter_group": group_name,
                    "sample_count": int(len(clean)),
                    "semantic_loss": 0.5
                    * (
                        losses_by_split["calibration"]["semantic"]
                        + losses_by_split["test"]["semantic"]
                    ),
                    "basis_loss": 0.5
                    * (
                        losses_by_split["calibration"]["basis"]
                        + losses_by_split["test"]["basis"]
                    ),
                    **gradient_pair_metrics(semantic_all, basis_all),
                    **cross_split_metrics(
                        vectors[("calibration", "semantic")],
                        vectors[("calibration", "basis")],
                        vectors[("test", "semantic")],
                        vectors[("test", "basis")],
                    ),
                }
            )
        del accumulators
        gc.collect()
        torch.cuda.empty_cache()
    metadata = {
        "last_block_index": int(last_block),
        "selected_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_group_counts": {
            group_name: int(sum(parameters[index].numel() for index in indices))
            for group_name, indices in groups.items()
        },
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(batch_rows), pd.DataFrame(aggregate_rows), metadata


def evaluate_predictions(table: pd.DataFrame) -> dict[str, object]:
    main_groups = ("last_block", "output_head")
    high = table[
        table.parameter_group.isin(main_groups) & table.time.isin((0.97, 0.85))
    ]
    pressure = high.pivot_table(
        index=["checkpoint_step", "time", "parameter_group"],
        columns="condition",
        values="basis_over_semantic_norm",
    )
    pressure_5k = pressure.loc[5000]
    p1_hits = int((pressure_5k.floor020_p2 / pressure_5k.annealed >= 2.0).sum())

    step5k = table[
        (table.checkpoint_step == 5000) & table.parameter_group.isin(main_groups)
    ]
    descent = step5k.pivot_table(
        index=["time", "parameter_group"],
        columns="condition",
        values="semantic_descent_ratio",
    )
    descent_gap = descent["static"] - descent["floor020_p2"]
    per_group_hits = {
        group: int((descent_gap.xs(group, level="parameter_group") >= 0.05).sum())
        for group in main_groups
    }
    p2 = all(value >= 4 for value in per_group_hits.values())

    gap_by_step = {}
    for step in CHECKPOINTS:
        subset = table[
            (table.checkpoint_step == step) & table.parameter_group.isin(main_groups)
        ].pivot_table(
            index=["time", "parameter_group"],
            columns="condition",
            values="semantic_descent_ratio",
        )
        gap_by_step[str(step)] = float(
            (subset["static"] - subset["floor020_p2"]).median()
        )
    p3 = gap_by_step["5000"] > gap_by_step["2000"]

    signs = table[table.parameter_group.isin(main_groups)].copy()
    signs["same_sign"] = (
        signs.semantic_basis_cosine.mul(
            signs.cross_split_semantic_basis_cosine
        )
        >= 0.0
    )
    sign_rate = float(signs.same_sign.mean())
    p4 = sign_rate >= 0.75
    predictions = {
        "p1_floor_high_noise_pressure": p1_hits >= 3,
        "p2_static_less_semantic_interference": bool(p2),
        "p3_interference_gap_grows_by_5k": bool(p3),
        "p4_cross_split_sign_stability": bool(p4),
    }
    return {
        "pass": bool(all(predictions.values())),
        "predictions": predictions,
        "details": {
            "p1_hits_of_4": p1_hits,
            "p2_hits_of_6_by_group": per_group_hits,
            "static_minus_floor_median_gap_by_step": gap_by_step,
            "aggregate_cross_split_sign_agreement": sign_rate,
        },
    }


def evaluate_crossover_gradients(table: pd.DataFrame) -> dict[str, object]:
    row = table[
        (table.checkpoint_step == 5000)
        & (table.parameter_group == "last_block")
        & (table.time == 0.1)
    ].set_index("condition")
    expected = {condition for condition, _ in CROSSOVER_CONDITIONS}
    if set(row.index) != expected:
        raise ValueError("crossover gradient table is incomplete")
    metric = "semantic_descent_ratio"
    values = {condition: float(row.loc[condition, metric]) for condition in row.index}
    effects = {
        "switch_floor_to_static": values["floor_to_static"] - values["floor_to_floor"],
        "switch_static_to_floor": values["static_to_floor"] - values["static_to_static"],
    }
    predictions = {
        "floor_to_static_recovers_ge_0p05": effects["switch_floor_to_static"] >= 0.05,
        "static_to_floor_degrades_ge_0p05": effects["switch_static_to_floor"] <= -0.05,
    }
    return {
        "pass_late_path_gradient_prediction": bool(all(predictions.values())),
        "predictions": {key: bool(value) for key, value in predictions.items()},
        "t0p1_last_block_values": values,
        "switch_effects": effects,
    }


def _plot(table: pd.DataFrame, output: Path) -> None:
    metrics = (
        ("basis_over_semantic_norm", "Basis / semantic gradient norm", True),
        ("semantic_basis_cosine", "Semantic-basis gradient cosine", False),
        ("semantic_descent_ratio", "Semantic descent ratio", False),
    )
    colors = {"static": "#4C78A8", "annealed": "#E45756", "floor020_p2": "#54A24B"}
    figure, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    for row, group in enumerate(("last_block", "output_head")):
        subset = table[
            (table.checkpoint_step == 5000) & (table.parameter_group == group)
        ]
        for axis, (metric, title, log_scale) in zip(axes[row], metrics):
            for condition in colors:
                values = subset[subset.condition == condition].sort_values("time")
                axis.plot(
                    values.time,
                    values[metric],
                    marker="o",
                    linewidth=2.2,
                    color=colors[condition],
                    label=condition,
                )
            if log_scale:
                axis.set_yscale("log")
            if metric == "semantic_descent_ratio":
                axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2)
            if metric == "semantic_basis_cosine":
                axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2)
            axis.set_title(f"{group}: {title}")
            axis.set_xlabel("t (higher is noisier)")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
    figure.savefig(output / "step5000_gradient_interference.png", dpi=180)
    plt.close(figure)


def _plot_crossover(table: pd.DataFrame, output: Path) -> None:
    colors = {
        "floor_to_floor": "#E45756",
        "floor_to_static": "#72B7B2",
        "static_to_static": "#4C78A8",
        "static_to_floor": "#F2CF5B",
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for axis, group in zip(axes, ("last_block", "output_head")):
        subset = table[
            (table.checkpoint_step == 5000) & (table.parameter_group == group)
        ]
        for condition, color in colors.items():
            values = subset[subset.condition == condition].sort_values("time")
            axis.plot(
                values.time,
                values.semantic_descent_ratio,
                marker="o",
                linewidth=2.2,
                label=condition,
                color=color,
            )
        axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2)
        axis.set_title(f"{group}: semantic descent ratio")
        axis.set_xlabel("t (higher is noisier)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(output / "crossover_gradient_interference.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="online_2k5k_seed3407_holdout32")
    parser.add_argument("--cache-start", dest="start", type=int, default=100128)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument("--steps", type=int, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--times", type=float, nargs="+", default=list(TIMES))
    parser.add_argument("--study", choices=("original", "crossover"), default="original")
    args = parser.parse_args()
    if args.count % (2 * args.batch_size) != 0:
        raise ValueError("count must be divisible by twice the batch size")
    conditions = CONDITIONS if args.study == "original" else CROSSOVER_CONDITIONS
    if "RANK" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError(f"launch with torchrun on exactly {len(conditions)} GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != len(conditions):
        raise ValueError(f"expected exactly {len(conditions)} processes")
    configure_determinism(args.seed)
    condition, branch = conditions[rank]
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    dataset = CachedRAELatentDataset(
        Path(str(manifest["latent_cache"])),
        start=args.start,
        stop=args.start + args.count,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    clean = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    basis = _load_basis(manifest).to(device)
    output = args.output.expanduser().resolve() / args.run_name
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    batch_frames = []
    aggregate_frames = []
    metadata = {}
    for step in args.steps:
        batch, aggregate, step_metadata = audit_checkpoint(
            condition,
            branch,
            step,
            clean,
            labels,
            noise,
            basis,
            manifest,
            batch_size=args.batch_size,
            times=args.times,
            device=device,
        )
        batch_frames.append(batch)
        aggregate_frames.append(aggregate)
        metadata[str(step)] = step_metadata
        print(f"rank{rank} {condition} step{step} complete", flush=True)
    pd.concat(batch_frames, ignore_index=True).to_csv(
        output / f"batch_rank{rank:02d}.csv", index=False
    )
    pd.concat(aggregate_frames, ignore_index=True).to_csv(
        output / f"aggregate_rank{rank:02d}.csv", index=False
    )
    (output / f"metadata_rank{rank:02d}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    dist.barrier()
    if rank == 0:
        batches = pd.concat(
            [
                pd.read_csv(output / f"batch_rank{index:02d}.csv")
                for index in range(len(conditions))
            ],
            ignore_index=True,
        )
        aggregate = pd.concat(
            [
                pd.read_csv(output / f"aggregate_rank{index:02d}.csv")
                for index in range(len(conditions))
            ],
            ignore_index=True,
        )
        batches.to_csv(output / "batch_metrics.csv", index=False)
        aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
        decision = (
            evaluate_predictions(aggregate)
            if args.study == "original"
            else evaluate_crossover_gradients(aggregate)
        )
        (output / "decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        if args.study == "original":
            _plot(aggregate, output)
        else:
            _plot_crossover(aggregate, output)
        print(json.dumps(decision, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
