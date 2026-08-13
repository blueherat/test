#!/usr/bin/env python3
"""Compare x-target and same-target guidance around a shared SiT anchor."""

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
DEFAULT_V270 = BASE / "runs/sit-s-2_seed0/checkpoints/step_00270000.pt"
DEFAULT_X400 = (
    BASE
    / "runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00400000.pt"
)
DEFAULT_CACHE = BASE / "imagenet100_cmc_sdvae"
DEFAULT_OUTPUT = BASE / "fid5k_step400k_floor_audit_seed0/direction_geometry_x400_v270"
DEFAULT_TIMES = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99)


def _sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.double().flatten(1).square().mean(1).sqrt()


def _sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.double().flatten(1)
    right_flat = right.double().flatten(1)
    numerator = (left_flat * right_flat).sum(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)


def direction_metrics(
    anchor: torch.Tensor,
    x_other: torch.Tensor,
    v_other: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-sample geometry for two extrapolation increments.

    The increments use the actual negative-scale orientation:
    ``g_x = v400 - x400`` and ``g_v = v400 - v270``.
    """

    if anchor.shape != x_other.shape or anchor.shape != v_other.shape:
        raise ValueError("all compared fields must have identical shapes")
    x_full = anchor - x_other
    v_full = anchor - v_other
    x_parallel, x_orthogonal = decompose_relative_to_anchor(anchor, x_full)
    v_parallel, v_orthogonal = decompose_relative_to_anchor(anchor, v_full)
    x_full_rms = _sample_rms(x_full)
    v_full_rms = _sample_rms(v_full)
    x_orthogonal_rms = _sample_rms(x_orthogonal)
    v_orthogonal_rms = _sample_rms(v_orthogonal)
    dims = tuple(range(1, anchor.ndim))
    anchor_energy = anchor.double().square().sum(dim=dims).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    x_coefficient = (x_full.double() * anchor.double()).sum(dim=dims) / anchor_energy
    v_coefficient = (v_full.double() * anchor.double()).sum(dim=dims) / anchor_energy
    return {
        "full_cosine": _sample_cosine(x_full, v_full),
        "orthogonal_cosine": _sample_cosine(x_orthogonal, v_orthogonal),
        "orthogonal_overlap_cos2": _sample_cosine(
            x_orthogonal, v_orthogonal
        ).square(),
        "x_full_rms": x_full_rms,
        "v270_full_rms": v_full_rms,
        "x_parallel_rms": _sample_rms(x_parallel),
        "v270_parallel_rms": _sample_rms(v_parallel),
        "x_orthogonal_rms": x_orthogonal_rms,
        "v270_orthogonal_rms": v_orthogonal_rms,
        "v270_over_x_orthogonal_rms": v_orthogonal_rms
        / x_orthogonal_rms.clamp_min(torch.finfo(torch.float64).tiny),
        "x_orthogonal_energy_fraction": x_orthogonal_rms.square()
        / x_full_rms.square().clamp_min(torch.finfo(torch.float64).tiny),
        "v270_orthogonal_energy_fraction": v_orthogonal_rms.square()
        / v_full_rms.square().clamp_min(torch.finfo(torch.float64).tiny),
        "x_parallel_coefficient": x_coefficient,
        "v270_parallel_coefficient": v_coefficient,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in raw.columns
        if column not in {"context", "time", "sample_id"}
    ]
    rows: list[dict[str, float | str]] = []
    for (context, time_value), frame in raw.groupby(["context", "time"], sort=True):
        row: dict[str, float | str] = {
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
        row["parallel_coefficient_correlation"] = float(
            frame[["x_parallel_coefficient", "v270_parallel_coefficient"]]
            .corr()
            .iloc[0, 1]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["context", "time"]).reset_index(drop=True)


def plot_summary(
    summary: pd.DataFrame,
    output: Path,
    *,
    anchor_label: str = "v400",
    x_label: str = "x400",
    v_label: str = "v270",
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    colors = {"teacher": "#2864a5", "anchor_rollout": "#c44e38"}
    for context, frame in summary.groupby("context", sort=False):
        color = colors.get(str(context), None)
        axes[0, 0].plot(
            frame.time,
            frame.orthogonal_cosine_mean,
            "o-",
            color=color,
            label=context,
        )
        axes[0, 0].fill_between(
            frame.time,
            frame.orthogonal_cosine_q10,
            frame.orthogonal_cosine_q90,
            color=color,
            alpha=0.12,
        )
        axes[0, 1].plot(
            frame.time,
            frame.full_cosine_mean,
            "o-",
            color=color,
            label=context,
        )
        axes[1, 0].plot(
            frame.time,
            frame.x_orthogonal_rms_mean,
            "o-",
            color=color,
            label=f"{x_label} ({context})",
        )
        axes[1, 0].plot(
            frame.time,
            frame.v270_orthogonal_rms_mean,
            "s--",
            color=color,
            label=f"{v_label} ({context})",
        )
        axes[1, 1].plot(
            frame.time,
            frame.x_orthogonal_energy_fraction_mean,
            "o-",
            color=color,
            label=f"{x_label} ({context})",
        )
        axes[1, 1].plot(
            frame.time,
            frame.v270_orthogonal_energy_fraction_mean,
            "s--",
            color=color,
            label=f"{v_label} ({context})",
        )
    axes[0, 0].set(title="Orthogonal-direction alignment", ylabel="cosine")
    axes[0, 1].set(title="Full guidance-direction alignment", ylabel="cosine")
    axes[1, 0].set(title="Orthogonal component magnitude", ylabel="per-sample RMS")
    axes[1, 1].set(title="Orthogonal fraction of gap energy", ylabel="energy fraction")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("flow time t")
        axis.legend(fontsize=8)
    figure.suptitle(
        f"{anchor_label}-{x_label} versus {anchor_label}-{v_label} direction geometry",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", "--v400", dest="anchor", type=Path, default=DEFAULT_V400)
    parser.add_argument("--x-other", "--x400", dest="x_other", type=Path, default=DEFAULT_X400)
    parser.add_argument("--v-other", "--v270", dest="v_other", type=Path, default=DEFAULT_V270)
    parser.add_argument("--anchor-label", default="v400")
    parser.add_argument("--x-label", default="x400")
    parser.add_argument("--v-label", default="v270")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=256)
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
    times = tuple(float(value) for value in args.times)
    if len(set(times)) != len(times) or any(value <= 0 or value >= 1 for value in times):
        raise ValueError("times must be unique and strictly inside (0, 1)")
    times = tuple(sorted(times))
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
    for path in (args.anchor, args.x_other, args.v_other):
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
        checkpoints[0], checkpoints[1], metadata[0], metadata[1]
    )
    validate_pair_compatibility(
        checkpoints[0], checkpoints[2], metadata[0], metadata[2], allow_step_mismatch=True
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

    def evaluate(model_index: int, state: torch.Tensor, time_value: float, labels: torch.Tensor):
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
                anchor = evaluate(0, state, time_value, labels)
                x_other = evaluate(1, state, time_value, labels)
                v_other = evaluate(2, state, time_value, labels)
                metrics = direction_metrics(anchor, x_other, v_other)
                cpu_metrics = {key: value.cpu().numpy() for key, value in metrics.items()}
                for local_index in range(len(state)):
                    row: dict[str, float | int | str] = {
                        "context": context,
                        "time": time_value,
                        "sample_id": int(indices[start + local_index]),
                    }
                    row.update(
                        {key: float(value[local_index]) for key, value in cpu_metrics.items()}
                    )
                    raw_rows.append(row)
        print(json.dumps({"processed": stop, "samples": args.samples}), flush=True)

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    raw.to_csv(output_dir / "direction_geometry_per_sample.csv", index=False)
    summary.to_csv(output_dir / "direction_geometry_by_time.csv", index=False)
    plot_summary(
        summary,
        output_dir / "direction_geometry.png",
        anchor_label=args.anchor_label,
        x_label=args.x_label,
        v_label=args.v_label,
    )
    overall = (
        raw.groupby("context")
        .agg(
            samples=("sample_id", "count"),
            full_cosine_mean=("full_cosine", "mean"),
            orthogonal_cosine_mean=("orthogonal_cosine", "mean"),
            orthogonal_cosine_median=("orthogonal_cosine", "median"),
            orthogonal_overlap_cos2_mean=("orthogonal_overlap_cos2", "mean"),
            x_orthogonal_energy_fraction_mean=("x_orthogonal_energy_fraction", "mean"),
            v270_orthogonal_energy_fraction_mean=(
                "v270_orthogonal_energy_fraction",
                "mean",
            ),
            v270_over_x_orthogonal_rms_mean=(
                "v270_over_x_orthogonal_rms",
                "mean",
            ),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    payload = {
        "protocol": "imagenet100_sit_direction_geometry_x400_v270_v1",
        "definition": {
            "anchor": args.anchor_label,
            "x_guidance": f"{args.anchor_label} - {args.x_label}",
            "same_target_guidance": f"{args.anchor_label} - {args.v_label}",
            "decomposition": (
                f"one scalar projection onto {args.anchor_label} per "
                "sample/state/time over C,H,W"
            ),
            "comparison_states": [
                "teacher linear bridge",
                f"unguided {args.anchor_label} rollout",
            ],
        },
        "samples": args.samples,
        "seed": args.seed,
        "times": list(times),
        "checkpoints": metadata,
        "overall": overall,
        "raw_csv": str(output_dir / "direction_geometry_per_sample.csv"),
        "summary_csv": str(output_dir / "direction_geometry_by_time.csv"),
        "figure": str(output_dir / "direction_geometry.png"),
    }
    atomic_json_dump(payload, output_dir / "direction_geometry_summary.json")
    print(json.dumps(payload["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
