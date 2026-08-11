"""Audit endpoint conditioning of the ImageNet-100 dual-output SiT.

The audit is read-only: it loads a frozen checkpoint, constructs a fixed
class-balanced validation bank, and evaluates teacher-forced bridge states on
a dense endpoint-aware time grid.  It intentionally reuses the exact velocity
conversion used by the sampler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

try:
    from experiments.imagenet100_sit_dual_output import (
        dual_output_velocities,
    )
    from experiments.train_imagenet100_sit_dual_output import (
        PROTOCOL,
        create_dual_output_sit,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        NpyMomentsDataset,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
        sha256_file,
    )
except ImportError:
    from imagenet100_sit_dual_output import dual_output_velocities
    from train_imagenet100_sit_dual_output import PROTOCOL, create_dual_output_sit
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        NpyMomentsDataset,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
        sha256_file,
    )


DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_dual-output_seed0/checkpoints/step_00450000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "dual_endpoint_audit_step450000_seed0"
)
DEFAULT_TIMES = (
    0.0,
    0.0005,
    0.001,
    0.0011,
    0.0015,
    0.003,
    0.01,
    0.03,
    0.1,
    0.2,
    0.4,
    0.5,
    0.6,
    0.8,
    0.9,
    0.97,
    0.99,
    0.997,
    0.9985,
    0.9989,
    0.999,
    0.9995,
    1.0,
)


def parse_times(value: str) -> tuple[float, ...]:
    times = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not times:
        raise argparse.ArgumentTypeError("at least one time value is required")
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in times):
        raise argparse.ArgumentTypeError("all time values must be finite and in [0, 1]")
    if any(right <= left for left, right in zip(times[:-1], times[1:])):
        raise argparse.ArgumentTypeError("time values must be strictly increasing")
    return times


def stratified_indices(
    labels: np.ndarray,
    sample_count: int,
    seed: int,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Choose a deterministic, nearly equal number of samples per class."""

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if sample_count <= 0 or sample_count > len(labels):
        raise ValueError("sample_count must be in [1, len(labels)]")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= num_classes:
        raise ValueError("labels contain an out-of-range class")

    base_count, remainder = divmod(sample_count, num_classes)
    generator = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for class_index in range(num_classes):
        class_indices = np.flatnonzero(labels == class_index)
        requested = base_count + int(class_index < remainder)
        if len(class_indices) < requested:
            raise ValueError(
                f"class {class_index} has {len(class_indices)} samples; "
                f"{requested} are required"
            )
        selected.append(generator.permutation(class_indices)[:requested])
    result = np.concatenate(selected)
    result = result[generator.permutation(len(result))]
    if len(np.unique(result)) != sample_count:
        raise AssertionError("stratified sampling produced duplicate indices")
    return result.astype(np.int64, copy=False)


def scalar_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    quantiles = torch.quantile(
        values,
        torch.tensor([0.05, 0.5, 0.95, 0.99], dtype=values.dtype),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "q05": float(quantiles[0].item()),
        "q50": float(quantiles[1].item()),
        "q95": float(quantiles[2].item()),
        "q99": float(quantiles[3].item()),
        "max": float(values.max().item()),
    }


@dataclass
class RmsAccumulator:
    sum_squares: float = 0.0
    element_count: int = 0
    sample_rms: list[torch.Tensor] = field(default_factory=list)

    def update(self, value: torch.Tensor) -> None:
        value = value.detach().float()
        if value.ndim < 2:
            raise ValueError("RMS tensors must include batch and feature dimensions")
        self.sum_squares += float(value.double().square().sum().item())
        self.element_count += int(value.numel())
        self.sample_rms.append(value.square().flatten(1).mean(1).sqrt().cpu())

    def summary(self) -> dict[str, float | int]:
        if self.element_count == 0 or not self.sample_rms:
            raise ValueError("cannot summarize an empty RMS accumulator")
        per_sample = torch.cat(self.sample_rms)
        quantiles = torch.quantile(
            per_sample,
            torch.tensor([0.5, 0.95, 0.99], dtype=per_sample.dtype),
        )
        return {
            "element_count": self.element_count,
            "rms": math.sqrt(self.sum_squares / self.element_count),
            "sample_rms_mean": float(per_sample.mean().item()),
            "sample_rms_q50": float(quantiles[0].item()),
            "sample_rms_q95": float(quantiles[1].item()),
            "sample_rms_q99": float(quantiles[2].item()),
            "sample_rms_max": float(per_sample.max().item()),
        }


def endpoint_tensors(
    output: torch.Tensor,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    time_value: torch.Tensor,
    *,
    gate_activation: str,
    denominator_floor: float,
) -> dict[str, torch.Tensor]:
    """Return sampler-exact velocities and their endpoint-sensitive errors."""

    state, target_velocity = linear_flow_state_target(clean, epsilon, time_value)
    paths = dual_output_velocities(
        output,
        state=state,
        time_value=time_value,
        gate_activation=gate_activation,  # type: ignore[arg-type]
        denominator_floor=denominator_floor,
    )
    time_image = time_value[:, None, None, None].float()
    gate = paths["gate"].float()
    clean_error = paths["clean"] - clean.float()
    epsilon_error = epsilon.float() - paths["epsilon_prediction"]
    x_denominator = (1.0 - time_image).clamp_min(denominator_floor)
    epsilon_denominator = time_image.clamp_min(denominator_floor)
    x_error_contribution = gate * clean_error / x_denominator
    epsilon_error_contribution = (1.0 - gate) * epsilon_error / epsilon_denominator

    result = {
        "gate": gate,
        "clean_native_error": clean_error,
        "epsilon_native_error": epsilon_error,
        "x_velocity": paths["x"],
        "epsilon_velocity": paths["epsilon"],
        "dynamic_velocity": paths["dynamic"],
        "x_velocity_error": paths["x"] - target_velocity,
        "epsilon_velocity_error": paths["epsilon"] - target_velocity,
        "dynamic_velocity_error": paths["dynamic"] - target_velocity,
        "x_error_contribution": x_error_contribution,
        "epsilon_error_contribution": epsilon_error_contribution,
        "pre_switch_dynamic_error": x_error_contribution
        + epsilon_error_contribution,
    }
    return result


RMS_METRICS = (
    "clean_native_error",
    "epsilon_native_error",
    "x_velocity",
    "epsilon_velocity",
    "dynamic_velocity",
    "x_velocity_error",
    "epsilon_velocity_error",
    "dynamic_velocity_error",
    "x_error_contribution",
    "epsilon_error_contribution",
    "pre_switch_dynamic_error",
)


def flatten_summary(prefix: str, summary: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def evaluate_time(
    model: torch.nn.Module,
    clean_bank: torch.Tensor,
    epsilon_bank: torch.Tensor,
    labels: torch.Tensor,
    *,
    time: float,
    batch_size: int,
    device: torch.device,
    gate_activation: str,
    denominator_floor: float,
    precision: str,
) -> dict[str, float | int | bool | None]:
    accumulators = {name: RmsAccumulator() for name in RMS_METRICS}
    gate_values: list[torch.Tensor] = []
    gate_sample_means: list[torch.Tensor] = []
    gate_within_sample_stds: list[torch.Tensor] = []
    contribution_dot = 0.0
    x_contribution_squares = 0.0
    epsilon_contribution_squares = 0.0

    for start in range(0, len(clean_bank), batch_size):
        stop = min(start + batch_size, len(clean_bank))
        clean = clean_bank[start:stop].to(device, non_blocking=True)
        epsilon = epsilon_bank[start:stop].to(device, non_blocking=True)
        batch_labels = labels[start:stop].to(device, non_blocking=True)
        times = torch.full((len(clean),), float(time), device=device)
        state, _ = linear_flow_state_target(clean, epsilon, times)
        if precision == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(state, times, batch_labels)
        else:
            output = model(state, times, batch_labels)
        tensors = endpoint_tensors(
            output,
            clean,
            epsilon,
            times,
            gate_activation=gate_activation,
            denominator_floor=denominator_floor,
        )
        gate = tensors["gate"].flatten(1)
        gate_values.append(gate.cpu())
        gate_sample_means.append(gate.mean(1).cpu())
        gate_within_sample_stds.append(gate.std(1, unbiased=False).cpu())
        for name, accumulator in accumulators.items():
            accumulator.update(tensors[name])
        x_contribution = tensors["x_error_contribution"].detach().double()
        epsilon_contribution = tensors["epsilon_error_contribution"].detach().double()
        contribution_dot += float((x_contribution * epsilon_contribution).sum().item())
        x_contribution_squares += float(x_contribution.square().sum().item())
        epsilon_contribution_squares += float(
            epsilon_contribution.square().sum().item()
        )

    gate = torch.cat(gate_values).reshape(-1)
    sample_means = torch.cat(gate_sample_means)
    within_sample_stds = torch.cat(gate_within_sample_stds)
    between_sample_variance = float(sample_means.var(unbiased=False).item())
    within_sample_variance = float(within_sample_stds.square().mean().item())
    total_gate_variance = between_sample_variance + within_sample_variance
    row: dict[str, float | int | bool | None] = {
        "time": float(time),
        "sample_count": int(len(clean_bank)),
        "gate_objective_weight": float((time * (1.0 - time)) ** 2),
        "dynamic_endpoint_override": bool(
            time <= denominator_floor or time >= 1.0 - denominator_floor
        ),
        **flatten_summary("gate", scalar_summary(gate)),
        "gate_sample_mean_std": math.sqrt(between_sample_variance),
        "gate_within_sample_std_mean": float(within_sample_stds.mean().item()),
        "gate_within_sample_std_q95": float(
            torch.quantile(within_sample_stds, 0.95).item()
        ),
        "gate_between_sample_variance": between_sample_variance,
        "gate_within_sample_variance": within_sample_variance,
        "gate_within_variance_fraction": (
            within_sample_variance / total_gate_variance
            if total_gate_variance > 0.0
            else 0.0
        ),
        "weighted_branch_error_cosine": (
            contribution_dot
            / math.sqrt(x_contribution_squares * epsilon_contribution_squares)
            if x_contribution_squares > 0.0 and epsilon_contribution_squares > 0.0
            else None
        ),
    }

    if time < 1.0:
        row.update(
            flatten_summary("gate_over_one_minus_t", scalar_summary(gate / (1.0 - time)))
        )
    else:
        for key in scalar_summary(gate).keys():
            row[f"gate_over_one_minus_t_{key}"] = None
    if time > 0.0:
        row.update(
            flatten_summary("one_minus_gate_over_t", scalar_summary((1.0 - gate) / time))
        )
    else:
        for key in scalar_summary(gate).keys():
            row[f"one_minus_gate_over_t_{key}"] = None

    rms_summaries = {
        name: accumulator.summary() for name, accumulator in accumulators.items()
    }
    for name, summary in rms_summaries.items():
        row.update(flatten_summary(name, summary))
    best_branch_rms = min(
        float(rms_summaries["x_velocity_error"]["rms"]),
        float(rms_summaries["epsilon_velocity_error"]["rms"]),
    )
    row["dynamic_error_rms_over_best_branch"] = (
        float(rms_summaries["dynamic_velocity_error"]["rms"]) / best_branch_rms
    )
    return row


def load_validation_bank(
    cache_dir: Path,
    *,
    sample_count: int,
    selection_seed: int,
    latent_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    dataset = NpyMomentsDataset(cache_dir, "validation")
    labels_array = np.load(dataset.labels_path, mmap_mode="r", allow_pickle=False)
    indices = stratified_indices(labels_array, sample_count, selection_seed)
    moments_array = np.load(dataset.moments_path, mmap_mode="r", allow_pickle=False)
    moments = torch.from_numpy(np.asarray(moments_array[indices]).copy())
    labels = torch.from_numpy(np.asarray(labels_array[indices]).astype(np.int64, copy=True))
    generator = torch.Generator(device="cpu").manual_seed(int(latent_seed))
    posterior_noise = torch.randn((sample_count, *LATENT_SHAPE), generator=generator)
    epsilon = torch.randn((sample_count, *LATENT_SHAPE), generator=generator)
    clean = sample_sdvae_posterior(moments, posterior_noise)
    return clean.contiguous(), epsilon.contiguous(), labels.contiguous(), indices


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    times = np.asarray([float(row["time"]) for row in rows])
    gate_mean = np.asarray([float(row["gate_mean"]) for row in rows])
    gate_q05 = np.asarray([float(row["gate_q05"]) for row in rows])
    gate_q95 = np.asarray([float(row["gate_q95"]) for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(times, gate_mean, marker="o", label="mean gate r")
    axes[0, 0].fill_between(times, gate_q05, gate_q95, alpha=0.25, label="q05-q95")
    axes[0, 0].set(xlabel="t (noise to data)", ylabel="gate", title="Gate schedule")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].legend()

    valid_x = times < 1.0
    valid_epsilon = times > 0.0
    axes[0, 1].plot(
        times[valid_x],
        [float(rows[index]["gate_over_one_minus_t_q95"]) for index in np.flatnonzero(valid_x)],
        marker="o",
        label="q95 r/(1-t)",
    )
    axes[0, 1].plot(
        times[valid_epsilon],
        [
            float(rows[index]["one_minus_gate_over_t_q95"])
            for index in np.flatnonzero(valid_epsilon)
        ],
        marker="o",
        label="q95 (1-r)/t",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(
        xlabel="t", ylabel="coefficient", title="Endpoint amplification"
    )
    axes[0, 1].legend()

    for key, label in (
        ("x_velocity_error_rms", "x-derived"),
        ("epsilon_velocity_error_rms", "epsilon-derived"),
        ("dynamic_velocity_error_rms", "dynamic"),
    ):
        axes[1, 0].plot(
            times, [float(row[key]) for row in rows], marker="o", label=label
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        xlabel="t", ylabel="teacher-forced RMS", title="Velocity error"
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        times,
        [float(row["weighted_branch_error_cosine"]) for row in rows],
        marker="o",
        label="weighted error cosine",
    )
    axes[1, 1].plot(
        times,
        [float(row["dynamic_error_rms_over_best_branch"]) for row in rows],
        marker="o",
        label="dynamic RMS / best branch",
    )
    axes[1, 1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set(
        xlabel="t",
        ylabel="dimensionless",
        title="Complementarity and dynamic gain",
        ylim=(-0.03, 1.12),
    )
    axes[1, 1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real endpoint audit")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
    config = checkpoint["config"]
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("checkpoint and audit use different official SiT sources")
    model = create_dual_output_sit(
        sit_module,
        model_name=config["model_name"],
        cfg_dropout=float(config["cfg_dropout"]),
    )
    state_key = "ema" if args.weights == "ema" else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval().requires_grad_(False)

    clean, epsilon, labels, indices = load_validation_bank(
        args.cache_dir.expanduser().resolve(),
        sample_count=args.sample_count,
        selection_seed=args.selection_seed,
        latent_seed=args.latent_seed,
    )
    rows = [
        evaluate_time(
            model,
            clean,
            epsilon,
            labels,
            time=time,
            batch_size=args.batch_size,
            device=device,
            gate_activation=config["gate_activation"],
            denominator_floor=float(config["denominator_floor"]),
            precision=args.precision,
        )
        for time in args.times
    ]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "endpoint_metrics.csv"
    json_path = output_dir / "endpoint_audit.json"
    plot_path = output_dir / "endpoint_audit.png"
    write_csv(rows, csv_path)
    save_plot(rows, plot_path)
    payload: dict[str, object] = {
        "protocol": "imagenet100_sit_dual_endpoint_audit_v1",
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_protocol": checkpoint["protocol"],
        "weights": args.weights,
        "model_name": config["model_name"],
        "official_sit": source_metadata,
        "data_manifest_sha256": checkpoint.get("data_manifest_sha256"),
        "validation_selection": {
            "split": "validation",
            "sample_count": args.sample_count,
            "class_count": NUM_CLASSES,
            "selection": "deterministic class-balanced subset",
            "selection_seed": args.selection_seed,
            "latent_seed": args.latent_seed,
            "selected_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        },
        "evaluation": {
            "times": list(args.times),
            "batch_size": args.batch_size,
            "precision": args.precision,
            "allow_tf32": bool(args.allow_tf32),
            "gate_activation": config["gate_activation"],
            "denominator_floor": float(config["denominator_floor"]),
            "guidance": False,
            "teacher_forced_bridge": "x_t=(1-t)*epsilon+t*clean",
        },
        "metric_definitions": {
            "rms": "sqrt(sum(error^2)/number_of_tensor_elements)",
            "sample_rms_qXX": "quantile across per-sample tensor RMS values",
            "gate_over_one_minus_t": "raw r/(1-t), undefined at t=1",
            "one_minus_gate_over_t": "raw (1-r)/t, undefined at t=0",
            "dynamic_endpoint_override": (
                "sampler explicitly selects the finite branch when "
                "t<=floor or t>=1-floor"
            ),
        },
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "csv": str(csv_path),
                "json": str(json_path),
                "plot": str(plot_path),
            }
        ),
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--times", type=parse_times, default=DEFAULT_TIMES)
    parser.add_argument("--sample-count", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument("--latent-seed", type=int, default=20260812)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_count <= 0 or args.batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
