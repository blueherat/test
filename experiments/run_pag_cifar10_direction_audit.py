"""Held-out direction audit for PAG on a public CIFAR-10 DDPM.

This is a low-cost non-RAE mechanism experiment.  For each self-attention
layer, the official Diffusers PAG processor produces an unmodified prediction
``full`` and an identity-attention prediction ``base`` in one paired forward.
We then ask whether ``full - base`` points toward the remaining supervised
epsilon-prediction error.

The CIFAR-10 test set is split into calibration and evaluation subsets.  Layer
and scale selection uses only calibration rows; selected gains are reported on
the disjoint evaluation rows.  No model parameter is trained or modified.
"""

from __future__ import annotations

import argparse
import json
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.models.attention_processor import Attention, PAGIdentitySelfAttnProcessor2_0

from experiments.internal_guidance_direction import direction_metrics, scale_sweep_metrics


DEFAULT_MODEL = Path.home() / "data" / "eqvae" / "models" / "google-ddpm-cifar10-32"
DEFAULT_DATASET = Path("/data/shared/cifar-10-batches-py")
DEFAULT_OUTPUT = Path.home() / "data" / "eqvae" / "pag_cifar10_direction_audit"


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("expected finite comma-separated floats")
    return values


def configure_fp32(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_protocol(
    *,
    samples: int,
    calibration_samples: int,
    batch_size: int,
    timesteps: tuple[int, ...],
    scales: tuple[float, ...],
    train_timesteps: int,
) -> None:
    if samples <= 1 or batch_size <= 0:
        raise ValueError("samples must exceed one and batch_size must be positive")
    if not 0 < calibration_samples < samples:
        raise ValueError("calibration_samples must leave non-empty calibration and evaluation splits")
    if len(set(timesteps)) != len(timesteps) or any(
        value < 0 or value >= train_timesteps for value in timesteps
    ):
        raise ValueError("timesteps must be unique and lie inside the training schedule")
    if len(set(scales)) != len(scales) or any(value < 0 for value in scales):
        raise ValueError("scales must be unique and non-negative")
    if 1.0 not in scales:
        raise ValueError("scales must include 1.0 as the unmodified full baseline")


def load_cifar10_test(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = root.expanduser().resolve() / "test_batch"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    images = np.asarray(payload[b"data"], dtype=np.uint8)
    labels = np.asarray(payload[b"labels"], dtype=np.int64)
    if images.shape != (10_000, 3_072) or labels.shape != (10_000,):
        raise ValueError(f"unexpected CIFAR-10 test shapes: {images.shape}, {labels.shape}")
    tensor = torch.from_numpy(images.copy()).reshape(-1, 3, 32, 32).float()
    return tensor.div_(127.5).sub_(1.0), torch.from_numpy(labels.copy())


def attention_modules(model: torch.nn.Module) -> dict[str, Attention]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, Attention) and not module.is_cross_attention
    }
    if not modules:
        raise ValueError("model contains no compatible self-attention modules")
    return modules


@contextmanager
def pag_identity_layer(module: Attention) -> Iterator[None]:
    original = module.processor
    module.set_processor(PAGIdentitySelfAttnProcessor2_0())
    try:
        yield
    finally:
        module.set_processor(original)


def _model_sample(output: object) -> torch.Tensor:
    sample = getattr(output, "sample", output)
    if not isinstance(sample, torch.Tensor):
        raise TypeError("model output must be a tensor or expose a tensor .sample")
    return sample


@torch.no_grad()
def pag_dual_prediction(
    model: torch.nn.Module,
    module: Attention,
    state: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    paired_state = state.repeat(2, 1, 1, 1)
    paired_timestep = timestep.repeat(2)
    with pag_identity_layer(module):
        paired = _model_sample(model(paired_state, paired_timestep))
    if paired.shape[0] != 2 * state.shape[0]:
        raise ValueError("PAG paired forward returned an unexpected batch size")
    return paired.chunk(2, dim=0)


@torch.no_grad()
def verify_full_branch(
    model: torch.nn.Module,
    module: Attention,
    state: torch.Tensor,
    timestep: torch.Tensor,
    *,
    tolerance: float = 2e-5,
) -> float:
    reference = _model_sample(model(state, timestep))
    paired_full, _ = pag_dual_prediction(model, module, state, timestep)
    maximum = float((reference.float() - paired_full.float()).abs().max().cpu())
    if maximum > tolerance:
        raise RuntimeError(
            f"paired PAG full branch changed the original model by {maximum:.3e}, "
            f"above tolerance {tolerance:.3e}"
        )
    return maximum


def append_metric_rows(
    rows: list[dict[str, object]],
    metrics: dict[str, torch.Tensor],
    *,
    split: str,
    dataset_indices: list[int],
    labels: torch.Tensor,
    layer: str,
    timestep: int,
) -> None:
    for offset, dataset_index in enumerate(dataset_indices):
        row: dict[str, object] = {
            "split": split,
            "dataset_index": int(dataset_index),
            "label": int(labels[offset]),
            "layer": layer,
            "timestep": int(timestep),
        }
        for name, values in metrics.items():
            value = values[offset].detach().cpu()
            row[name] = bool(value) if value.dtype == torch.bool else float(value)
        rows.append(row)


@torch.no_grad()
def run_direction_atlas(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    modules: dict[str, Attention],
    clean_all: torch.Tensor,
    noise_all: torch.Tensor,
    labels_all: torch.Tensor,
    indices: list[int],
    *,
    calibration_samples: int,
    timesteps: tuple[int, ...],
    scales: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    split_ranges = (
        ("calibration", 0, calibration_samples),
        ("evaluation", calibration_samples, len(clean_all)),
    )
    for split, split_start, split_stop in split_ranges:
        for start in range(split_start, split_stop, batch_size):
            stop = min(start + batch_size, split_stop)
            clean = clean_all[start:stop].to(device)
            noise = noise_all[start:stop].to(device)
            labels = labels_all[start:stop]
            batch_indices = indices[start:stop]
            for timestep_value in timesteps:
                timestep = torch.full(
                    (len(clean),), int(timestep_value), device=device, dtype=torch.long
                )
                state = scheduler.add_noise(clean, noise, timestep)
                for layer, module in modules.items():
                    full, base = pag_dual_prediction(model, module, state, timestep)
                    metrics = direction_metrics(full, base, noise)
                    append_metric_rows(
                        metric_rows,
                        metrics,
                        split=split,
                        dataset_indices=batch_indices,
                        labels=labels,
                        layer=layer,
                        timestep=timestep_value,
                    )
                    sweep = scale_sweep_metrics(full, base, noise, scales)
                    for scale_index, scale in enumerate(scales):
                        for offset, dataset_index in enumerate(batch_indices):
                            sweep_rows.append(
                                {
                                    "split": split,
                                    "dataset_index": int(dataset_index),
                                    "label": int(labels[offset]),
                                    "layer": layer,
                                    "timestep": int(timestep_value),
                                    "scale_s": float(scale),
                                    "pag_gamma": float(scale) - 1.0,
                                    "epsilon_mse": float(
                                        sweep["mse"][scale_index, offset].cpu()
                                    ),
                                    "gain_over_full": float(
                                        sweep["gain_over_full"][scale_index, offset].cpu()
                                    ),
                                }
                            )
    return pd.DataFrame(metric_rows), pd.DataFrame(sweep_rows)


def calibration_policy(
    sweep: pd.DataFrame,
    *,
    family: str = "any_nonbaseline",
) -> pd.DataFrame:
    if family == "any_nonbaseline":
        scale_mask = sweep["scale_s"] != 1.0
    elif family == "positive_extrapolation":
        scale_mask = sweep["scale_s"] > 1.0
    else:
        raise ValueError(f"unknown policy family: {family}")
    calibration = sweep[(sweep["split"] == "calibration") & scale_mask]
    grouped = (
        calibration.groupby(["timestep", "layer", "scale_s", "pag_gamma"], as_index=False)
        .agg(calibration_gain_mean=("gain_over_full", "mean"))
        .sort_values(
            ["timestep", "calibration_gain_mean", "layer", "scale_s"],
            ascending=[True, False, True, True],
        )
    )
    selected = grouped.groupby("timestep", as_index=False).first()
    selected.insert(0, "policy_family", family)
    evaluation = sweep[sweep["split"] == "evaluation"].merge(
        selected[["timestep", "layer", "scale_s"]],
        on=["timestep", "layer", "scale_s"],
        how="inner",
    )
    heldout = (
        evaluation.groupby(["timestep", "layer", "scale_s", "pag_gamma"], as_index=False)
        .agg(
            evaluation_gain_mean=("gain_over_full", "mean"),
            evaluation_gain_median=("gain_over_full", "median"),
            evaluation_positive_fraction=("gain_over_full", lambda value: float((value > 0).mean())),
            evaluation_samples=("dataset_index", "size"),
        )
    )
    return selected.merge(
        heldout, on=["timestep", "layer", "scale_s", "pag_gamma"], how="left"
    )


def summary_tables(
    metrics: pd.DataFrame,
    sweep: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direction = (
        metrics.groupby(["split", "timestep", "layer"], as_index=False)
        .agg(
            alignment_cosine_mean=("alignment_cosine", "mean"),
            positive_alignment_fraction=("positive_alignment", "mean"),
            scale_star_median=("scale_star", "median"),
            oracle_relative_gain_mean=("oracle_relative_gain", "mean"),
            direction_rms_mean=("direction_rms", "mean"),
        )
    )
    scale = (
        sweep.groupby(["split", "timestep", "layer", "scale_s", "pag_gamma"], as_index=False)
        .agg(
            gain_mean=("gain_over_full", "mean"),
            gain_median=("gain_over_full", "median"),
            positive_gain_fraction=("gain_over_full", lambda value: float((value > 0).mean())),
        )
    )
    policy = pd.concat(
        [
            calibration_policy(sweep, family="any_nonbaseline"),
            calibration_policy(sweep, family="positive_extrapolation"),
        ],
        ignore_index=True,
    )
    return direction, scale, policy


def plot_atlas(direction: pd.DataFrame, policy: pd.DataFrame, output_path: Path) -> None:
    evaluation = direction[direction["split"] == "evaluation"]
    layers = list(dict.fromkeys(evaluation["layer"]))
    timesteps = sorted(evaluation["timestep"].unique())
    cosine = evaluation.pivot(index="layer", columns="timestep", values="alignment_cosine_mean").reindex(
        index=layers, columns=timesteps
    )
    positive = evaluation.pivot(
        index="layer", columns="timestep", values="positive_alignment_fraction"
    ).reindex(index=layers, columns=timesteps)

    figure, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    for axis, frame, title, cmap, low, high in (
        (axes[0], cosine, "Evaluation direction cosine", "RdBu_r", -1.0, 1.0),
        (axes[1], positive, "Evaluation positive-alignment fraction", "viridis", 0.0, 1.0),
    ):
        image = axis.imshow(frame.to_numpy(), aspect="auto", cmap=cmap, vmin=low, vmax=high)
        axis.set_xticks(range(len(timesteps)), [str(value) for value in timesteps])
        axis.set_yticks(range(len(layers)), layers, fontsize=8)
        axis.set(title=title, xlabel="DDPM timestep", ylabel="perturbed attention layer")
        figure.colorbar(image, ax=axis)

    extrapolation = policy[policy["policy_family"] == "positive_extrapolation"]
    axes[2].plot(
        extrapolation["timestep"],
        extrapolation["calibration_gain_mean"],
        marker="o",
        label="calibration",
        color="#2563EB",
    )
    axes[2].plot(
        extrapolation["timestep"],
        extrapolation["evaluation_gain_mean"],
        marker="o",
        label="held-out evaluation",
        color="#DC2626",
    )
    axes[2].axhline(0.0, color="#111827", linewidth=1)
    axes[2].set(
        title="Calibrated layer/scale generalization",
        xlabel="DDPM timestep",
        ylabel="epsilon-MSE relative gain over full",
    )
    axes[2].legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timesteps", type=parse_int_list, default=(100, 300, 500, 700, 900))
    parser.add_argument(
        "--scales", type=parse_float_list, default=(0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0)
    )
    parser.add_argument("--layers", default="all", help="all or comma-separated exact module names")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fp32(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_root = args.model.expanduser().resolve()
    model = UNet2DModel.from_pretrained(
        model_root, local_files_only=True, use_safetensors=False
    )
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    scheduler = DDPMScheduler.from_pretrained(model_root, local_files_only=True)
    validate_protocol(
        samples=args.samples,
        calibration_samples=args.calibration_samples,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        scales=args.scales,
        train_timesteps=int(scheduler.config.num_train_timesteps),
    )

    available = attention_modules(model)
    if args.layers == "all":
        selected_modules = available
    else:
        layer_names = tuple(value.strip() for value in args.layers.split(",") if value.strip())
        missing = sorted(set(layer_names) - set(available))
        if missing:
            raise ValueError(f"unknown attention layers: {missing}; available: {sorted(available)}")
        selected_modules = {name: available[name] for name in layer_names}

    all_images, all_labels = load_cifar10_test(args.dataset)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    indices = torch.randperm(len(all_images), generator=generator)[: args.samples].tolist()
    clean = all_images[indices]
    labels = all_labels[indices]
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)

    verification_timestep = torch.tensor([args.timesteps[len(args.timesteps) // 2]], device=device)
    verification_state = scheduler.add_noise(
        clean[:1].to(device), noise[:1].to(device), verification_timestep
    )
    verification = {
        layer: verify_full_branch(
            model, module, verification_state, verification_timestep
        )
        for layer, module in selected_modules.items()
    }

    metrics, sweep = run_direction_atlas(
        model,
        scheduler,
        selected_modules,
        clean,
        noise,
        labels,
        indices,
        calibration_samples=args.calibration_samples,
        timesteps=args.timesteps,
        scales=args.scales,
        batch_size=args.batch_size,
        device=device,
    )
    direction_summary, scale_summary, policy = summary_tables(metrics, sweep)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "direction_rows.csv", index=False)
    sweep.to_csv(output_dir / "scale_rows.csv", index=False)
    direction_summary.to_csv(output_dir / "direction_summary.csv", index=False)
    scale_summary.to_csv(output_dir / "scale_summary.csv", index=False)
    policy.to_csv(output_dir / "calibrated_policy_heldout.csv", index=False)
    plot_atlas(direction_summary, policy, output_dir / "direction_atlas.png")

    metadata = {
        "experiment": "public_ddpm_cifar10_pag_direction_audit_v1",
        "training": False,
        "model": str(model_root),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "prediction_type": str(scheduler.config.prediction_type),
        "dataset": str(args.dataset.expanduser().resolve()),
        "dataset_indices": indices,
        "calibration_samples": int(args.calibration_samples),
        "evaluation_samples": int(args.samples - args.calibration_samples),
        "layers": list(selected_modules),
        "timesteps": list(args.timesteps),
        "scale_definition": "guided = base + s * (full - base); standard PAG gamma = s - 1",
        "policy_families": ["any_nonbaseline", "positive_extrapolation"],
        "scales_s": list(args.scales),
        "seed": int(args.seed),
        "precision": "fp32",
        "tf32": False,
        "paired_full_max_abs_difference": verification,
        "claim_boundary": "held-out epsilon-direction mechanism audit, not FID or sample-quality evidence",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(policy.to_string(index=False))
    print(f"saved audit to {output_dir}")


if __name__ == "__main__":
    main()
