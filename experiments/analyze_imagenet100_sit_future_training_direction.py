#!/usr/bin/env python3
"""Compare SiT guidance directions with a continued-training field update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_static_pair import (
        decompose_relative_to_anchor,
        output_to_field_velocity,
        project_onto_direction,
    )
    from experiments.sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NpyMomentsDataset,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
        sample_sdvae_posterior,
    )
except ModuleNotFoundError:
    from imagenet100_sit_static_pair import (
        decompose_relative_to_anchor,
        output_to_field_velocity,
        project_onto_direction,
    )
    from sample_imagenet100_sit_static_pair_fid import (
        _load_field_model,
        validate_pair_compatibility,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NpyMomentsDataset,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
        sample_sdvae_posterior,
    )


BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_V400 = BASE / "runs/sit-s-2_seed0/checkpoints/step_00400000.pt"
DEFAULT_V800 = BASE / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_V270 = BASE / "runs/sit-s-2_seed0/checkpoints/step_00270000.pt"
DEFAULT_X400 = (
    BASE
    / "runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_CACHE = BASE / "imagenet100_cmc_sdvae"
DEFAULT_OUTPUT = (
    BASE
    / "fid5k_step400k_floor_audit_seed0/future_direction_v800_x400_v270"
)
DEFAULT_TIMES = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99)


def _sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.double().flatten(1).square().mean(1).sqrt()


def _sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.double().flatten(1)
    right_flat = right.double().flatten(1)
    numerator = (left_flat * right_flat).sum(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)


def _projection_coefficient(
    value: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    value_flat = value.double().flatten(1)
    direction_flat = direction.double().flatten(1)
    return (value_flat * direction_flat).sum(1) / direction_flat.square().sum(
        1
    ).clamp_min(torch.finfo(torch.float64).tiny)


def future_alignment_metrics(
    anchor: torch.Tensor,
    training_reference: torch.Tensor,
    x_other: torch.Tensor,
    v_other: torch.Tensor,
    *,
    update_direction: str = "reference_minus_anchor",
    x_prefix: str = "x400",
    v_prefix: str = "v270",
) -> dict[str, torch.Tensor]:
    """Measure local alignment with a training update on identical states."""

    if not (
        anchor.shape == training_reference.shape == x_other.shape == v_other.shape
    ):
        raise ValueError("all compared fields must have identical shapes")
    if update_direction == "reference_minus_anchor":
        future_direction = training_reference - anchor
    elif update_direction == "anchor_minus_reference":
        future_direction = anchor - training_reference
    else:
        raise ValueError(f"unsupported update_direction: {update_direction}")
    x_guidance = anchor - x_other
    v_guidance = anchor - v_other
    _, future_orthogonal = decompose_relative_to_anchor(anchor, future_direction)
    _, x_orthogonal = decompose_relative_to_anchor(anchor, x_guidance)
    _, v_orthogonal = decompose_relative_to_anchor(anchor, v_guidance)
    future_rms = _sample_rms(future_direction)
    future_orthogonal_rms = _sample_rms(future_orthogonal)
    tiny = torch.finfo(torch.float64).tiny
    metrics: dict[str, torch.Tensor] = {
        "future_rms": future_rms,
        "future_orthogonal_rms": future_orthogonal_rms,
        "future_orthogonal_energy_fraction": future_orthogonal_rms.square()
        / future_rms.square().clamp_min(tiny),
    }
    for name, guidance, orthogonal in (
        (x_prefix, x_guidance, x_orthogonal),
        (v_prefix, v_guidance, v_orthogonal),
    ):
        full_cosine = _sample_cosine(guidance, future_direction)
        orthogonal_cosine = _sample_cosine(orthogonal, future_orthogonal)
        full_projection = project_onto_direction(guidance, future_direction)
        orthogonal_projection = project_onto_direction(
            orthogonal,
            future_orthogonal,
        )
        guidance_rms = _sample_rms(guidance)
        orthogonal_rms = _sample_rms(orthogonal)
        metrics.update(
            {
                f"{name}_full_cosine": full_cosine,
                f"{name}_full_overlap_cos2": full_cosine.square(),
                f"{name}_full_positive_alignment": (full_cosine > 0).double(),
                f"{name}_full_projection_coefficient": _projection_coefficient(
                    guidance,
                    future_direction,
                ),
                f"{name}_full_projected_rms": _sample_rms(full_projection),
                f"{name}_full_residual_rms": _sample_rms(
                    guidance - full_projection
                ),
                f"{name}_full_rms": guidance_rms,
                f"{name}_over_future_rms": guidance_rms
                / future_rms.clamp_min(tiny),
                f"{name}_orthogonal_cosine": orthogonal_cosine,
                f"{name}_orthogonal_overlap_cos2": orthogonal_cosine.square(),
                f"{name}_orthogonal_positive_alignment": (
                    orthogonal_cosine > 0
                ).double(),
                f"{name}_orthogonal_projection_coefficient": (
                    _projection_coefficient(orthogonal, future_orthogonal)
                ),
                f"{name}_orthogonal_projected_rms": _sample_rms(
                    orthogonal_projection
                ),
                f"{name}_orthogonal_residual_rms": _sample_rms(
                    orthogonal - orthogonal_projection
                ),
                f"{name}_orthogonal_rms": orthogonal_rms,
                f"{name}_orthogonal_over_future_rms": orthogonal_rms
                / future_orthogonal_rms.clamp_min(tiny),
            }
        )
    return metrics


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in raw.columns
        if column not in {"context", "time", "sample_id"}
    ]
    rows: list[dict[str, float | int | str]] = []
    for (context, time_value), frame in raw.groupby(["context", "time"], sort=True):
        row: dict[str, float | int | str] = {
            "context": str(context),
            "time": float(time_value),
            "samples": int(len(frame)),
        }
        for metric in metric_columns:
            values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q10"] = float(np.quantile(values, 0.1))
            row[f"{metric}_q90"] = float(np.quantile(values, 0.9))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["context", "time"]).reset_index(drop=True)


def plot_summary(
    summary: pd.DataFrame,
    output: Path,
    *,
    x_prefix: str,
    v_prefix: str,
    x_label: str,
    v_label: str,
    direction_label: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    colors = {x_prefix: "#2864a5", v_prefix: "#c44e38"}
    display = {x_prefix: x_label, v_prefix: v_label}
    styles = {"teacher": "-", "anchor_rollout": "--"}
    for context, frame in summary.groupby("context", sort=False):
        for candidate in (x_prefix, v_prefix):
            label = f"{display[candidate]}, {context}"
            axes[0, 0].plot(
                frame.time,
                frame[f"{candidate}_full_cosine_mean"],
                marker="o",
                linestyle=styles.get(str(context), "-"),
                color=colors[candidate],
                label=label,
            )
            axes[0, 1].plot(
                frame.time,
                frame[f"{candidate}_orthogonal_cosine_mean"],
                marker="o",
                linestyle=styles.get(str(context), "-"),
                color=colors[candidate],
                label=label,
            )
            axes[1, 0].plot(
                frame.time,
                frame[f"{candidate}_full_projection_coefficient_median"],
                marker="o",
                linestyle=styles.get(str(context), "-"),
                color=colors[candidate],
                label=label,
            )
            axes[1, 1].plot(
                frame.time,
                frame[f"{candidate}_over_future_rms_mean"],
                marker="o",
                linestyle=styles.get(str(context), "-"),
                color=colors[candidate],
                label=label,
            )
    axes[0, 0].set(title=f"Full alignment with {direction_label}", ylabel="mean cosine")
    axes[0, 1].set(title="Anchor-orthogonal alignment", ylabel="mean cosine")
    axes[1, 0].set(title="Projection onto future direction", ylabel="median coefficient")
    axes[1, 1].set(title="Guidance/future magnitude", ylabel="mean RMS ratio")
    for axis in axes.flat:
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.grid(alpha=0.2)
        axis.set_xlabel("flow time t")
        axis.legend(fontsize=8)
    figure.suptitle("Local alignment with continued v-model training", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", "--v400", dest="anchor", type=Path, default=DEFAULT_V400)
    parser.add_argument(
        "--training-reference",
        "--v800",
        dest="training_reference",
        type=Path,
        default=DEFAULT_V800,
    )
    parser.add_argument("--x-other", "--x400", dest="x_other", type=Path, default=DEFAULT_X400)
    parser.add_argument("--v-other", "--v270", dest="v_other", type=Path, default=DEFAULT_V270)
    parser.add_argument(
        "--update-direction",
        choices=("reference_minus_anchor", "anchor_minus_reference"),
        default="reference_minus_anchor",
    )
    parser.add_argument("--anchor-label", default="v400")
    parser.add_argument("--training-reference-label", default="v800")
    parser.add_argument("--x-label", default="x400")
    parser.add_argument("--v-label", default="v270")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--times", nargs="+", type=float, default=list(DEFAULT_TIMES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0 or args.samples % args.batch_size:
        raise ValueError("samples must be positive and divisible by batch size")
    times = tuple(sorted(float(value) for value in args.times))
    if len(set(times)) != len(times) or any(value <= 0 or value >= 1 for value in times):
        raise ValueError("times must be unique and strictly inside (0, 1)")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(), verify_source=True
    )
    models = []
    semantics = []
    metadata = []
    checkpoints = []
    for path in (args.anchor, args.training_reference, args.x_other, args.v_other):
        model, field_semantics, field_metadata, checkpoint = _load_field_model(
            checkpoint_path=path.expanduser().resolve(),
            requested_field="auto",
            weights="ema",
            sit_module=sit_module,
            source_metadata=source_metadata,
            device=device,
        )
        assert model is not None
        models.append(model)
        semantics.append(field_semantics)
        metadata.append(field_metadata)
        checkpoints.append(checkpoint)
    validate_pair_compatibility(
        checkpoints[0], checkpoints[1], metadata[0], metadata[1], allow_step_mismatch=True
    )
    validate_pair_compatibility(checkpoints[0], checkpoints[2], metadata[0], metadata[2])
    validate_pair_compatibility(
        checkpoints[0], checkpoints[3], metadata[0], metadata[3], allow_step_mismatch=True
    )
    del checkpoints

    dataset = NpyMomentsDataset(args.cache_dir.expanduser().resolve(), "validation")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.samples, replace=False)
    moments_array = np.load(dataset.moments_path, mmap_mode="r", allow_pickle=False)
    labels_array = np.load(dataset.labels_path, mmap_mode="r", allow_pickle=False)
    moments = torch.from_numpy(np.asarray(moments_array[indices]).copy())
    labels_all = torch.from_numpy(np.asarray(labels_array[indices]).copy()).long()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    posterior_noise = torch.randn(args.samples, *LATENT_SHAPE, generator=generator)
    path_noise = torch.randn(args.samples, *LATENT_SHAPE, generator=generator)
    data = sample_sdvae_posterior(
        moments,
        posterior_noise,
        scaling_factor=SD_VAE_SCALING_FACTOR,
    )

    def evaluate(
        model_index: int,
        state: torch.Tensor,
        time_value: float,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        time_batch = torch.full(
            (len(state),), time_value, device=device, dtype=torch.float32
        )
        output = models[model_index](state.float(), time_batch, labels)
        return output_to_field_velocity(
            output,
            state=state,
            time_value=time_batch,
            semantics=semantics[model_index],
        )

    raw_rows: list[dict[str, float | int | str]] = []
    integration_times = torch.tensor((0.0, *times), device=device, dtype=torch.float32)
    for start in range(0, args.samples, args.batch_size):
        stop = start + args.batch_size
        labels = labels_all[start:stop].to(device)
        noise = path_noise[start:stop].to(device)
        clean = data[start:stop].to(device)

        def baseline_velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return evaluate(0, state, float(time_value.item()), labels)

        rollout = odeint(
            baseline_velocity,
            noise,
            integration_times,
            method="dopri5",
            atol=1e-6,
            rtol=1e-3,
        )[1:]
        for time_index, time_value in enumerate(times):
            teacher = (1.0 - time_value) * noise + time_value * clean
            for context, state in (
                ("teacher", teacher),
                ("anchor_rollout", rollout[time_index]),
            ):
                fields = [
                    evaluate(model_index, state, time_value, labels)
                    for model_index in range(4)
                ]
                metrics = future_alignment_metrics(
                    *fields,
                    update_direction=args.update_direction,
                    x_prefix="x",
                    v_prefix="v",
                )
                cpu_metrics = {
                    key: value.cpu().numpy() for key, value in metrics.items()
                }
                for local_index in range(len(state)):
                    row: dict[str, float | int | str] = {
                        "context": context,
                        "time": time_value,
                        "sample_id": int(indices[start + local_index]),
                    }
                    row.update(
                        {
                            key: float(value[local_index])
                            for key, value in cpu_metrics.items()
                        }
                    )
                    raw_rows.append(row)
        print(json.dumps({"processed": stop, "samples": args.samples}), flush=True)

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    raw_path = output_dir / "future_alignment_per_sample.csv"
    summary_path = output_dir / "future_alignment_by_time.csv"
    figure_path = output_dir / "future_alignment.png"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    if args.update_direction == "reference_minus_anchor":
        direction_label = f"{args.training_reference_label}-{args.anchor_label}"
    else:
        direction_label = f"{args.anchor_label}-{args.training_reference_label}"
    plot_summary(
        summary,
        figure_path,
        x_prefix="x",
        v_prefix="v",
        x_label=args.x_label,
        v_label=args.v_label,
        direction_label=direction_label,
    )
    overall_rows: list[dict[str, float | int | str]] = []
    for context, frame in raw.groupby("context", sort=True):
        row: dict[str, float | int | str] = {
            "context": str(context),
            "observations": int(len(frame)),
        }
        for candidate in ("x", "v"):
            for metric in (
                "full_cosine",
                "full_overlap_cos2",
                "full_positive_alignment",
                "full_projection_coefficient",
                "orthogonal_cosine",
                "orthogonal_overlap_cos2",
                "orthogonal_positive_alignment",
                "orthogonal_projection_coefficient",
            ):
                values = frame[f"{candidate}_{metric}"].to_numpy(dtype=np.float64)
                row[f"{candidate}_{metric}_mean"] = float(np.mean(values))
                row[f"{candidate}_{metric}_median"] = float(np.median(values))
        overall_rows.append(row)
    payload = {
        "protocol": "imagenet100_sit_training_update_alignment_v2",
        "definition": {
            "training_update_direction": direction_label,
            "x_guidance": f"{args.anchor_label} - {args.x_label}",
            "same_target_guidance": f"{args.anchor_label} - {args.v_label}",
            "comparison_states": ["teacher linear bridge", f"unguided {args.anchor_label} rollout"],
            "projection_scope": "one scalar per sample/state/time over C,H,W",
        },
        "samples": args.samples,
        "seed": args.seed,
        "times": list(times),
        "checkpoints": metadata,
        "overall": overall_rows,
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "figure": str(figure_path),
    }
    atomic_json_dump(payload, output_dir / "future_alignment_summary.json")
    print(json.dumps(payload["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
