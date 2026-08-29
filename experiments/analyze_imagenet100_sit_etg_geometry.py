#!/usr/bin/env python3
"""Measure whether calibrated ETG differs non-trivially from the v-head gap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_error_triangulated_guidance import (
        TARGETS,
        full_and_internal_predictions,
        fuse_predictions,
        load_etg_model,
        predictions_to_velocity,
    )
    from experiments.sample_imagenet100_sit_etg_fid import (
        DEFAULT_CALIBRATION,
        calibration_checkpoints,
        load_calibration,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from imagenet100_sit_error_triangulated_guidance import (
        TARGETS,
        full_and_internal_predictions,
        fuse_predictions,
        load_etg_model,
        predictions_to_velocity,
    )
    from sample_imagenet100_sit_etg_fid import (
        DEFAULT_CALIBRATION,
        calibration_checkpoints,
        load_calibration,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
    )


DEFAULT_OUTPUT = DEFAULT_CALIBRATION.parent / "geometry_audit.json"


def strong_velocity(model, labels: torch.Tensor):
    def velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        output = model(state, time_value.expand(len(state)), labels)
        return output[:, : LATENT_SHAPE[0]].float()

    return velocity


def empty_sums(time_count: int) -> dict[str, torch.Tensor]:
    names = (
        "dot",
        "reference_sq",
        "candidate_sq",
        "residual_sq",
        "sample_cosine",
        "sample_angle_degrees",
        "sample_beta",
        "sample_count",
    )
    return {name: torch.zeros(time_count, dtype=torch.float64) for name in names}


def update_geometry(
    sums: dict[str, torch.Tensor],
    *,
    time_index: int,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> None:
    reference_flat = reference.float().flatten(1)
    candidate_flat = candidate.float().flatten(1)
    dot = (reference_flat * candidate_flat).sum(dim=1)
    reference_sq = reference_flat.square().sum(dim=1)
    candidate_sq = candidate_flat.square().sum(dim=1)
    beta = dot / reference_sq.clamp_min(1e-12)
    residual = candidate_flat - beta[:, None] * reference_flat
    cosine = dot / (reference_sq.sqrt() * candidate_sq.sqrt()).clamp_min(1e-12)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    sums["dot"][time_index] += dot.double().sum().cpu()
    sums["reference_sq"][time_index] += reference_sq.double().sum().cpu()
    sums["candidate_sq"][time_index] += candidate_sq.double().sum().cpu()
    sums["residual_sq"][time_index] += residual.square().sum().double().cpu()
    sums["sample_cosine"][time_index] += cosine.double().sum().cpu()
    sums["sample_angle_degrees"][time_index] += angle.double().sum().cpu()
    sums["sample_beta"][time_index] += beta.double().sum().cpu()
    sums["sample_count"][time_index] += len(reference)


def summarize_geometry(sums: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    count = sums["sample_count"].clamp_min(1.0)
    global_cosine = sums["dot"] / (
        sums["reference_sq"].sqrt() * sums["candidate_sq"].sqrt()
    ).clamp_min(1e-30)
    return {
        "global_cosine": global_cosine,
        "mean_sample_cosine": sums["sample_cosine"] / count,
        "mean_angle_degrees": sums["sample_angle_degrees"] / count,
        "mean_projection_beta": sums["sample_beta"] / count,
        "candidate_to_reference_rms_ratio": (
            sums["candidate_sq"] / sums["reference_sq"].clamp_min(1e-30)
        ).sqrt(),
        "nonparallel_energy_fraction": sums["residual_sq"]
        / sums["candidate_sq"].clamp_min(1e-30),
        "sample_count": sums["sample_count"],
    }


def overall_geometry(sums: dict[str, torch.Tensor]) -> dict[str, float]:
    total = {name: value.sum() for name, value in sums.items()}
    count = total["sample_count"].clamp_min(1.0)
    cosine = total["dot"] / (
        total["reference_sq"].sqrt() * total["candidate_sq"].sqrt()
    ).clamp_min(1e-30)
    return {
        "global_cosine": float(cosine.item()),
        "mean_sample_cosine": float((total["sample_cosine"] / count).item()),
        "mean_angle_degrees": float((total["sample_angle_degrees"] / count).item()),
        "mean_projection_beta": float((total["sample_beta"] / count).item()),
        "candidate_to_reference_rms_ratio": float(
            (total["candidate_sq"] / total["reference_sq"].clamp_min(1e-30)).sqrt().item()
        ),
        "nonparallel_energy_fraction": float(
            (total["residual_sq"] / total["candidate_sq"].clamp_min(1e-30)).item()
        ),
        "sample_count_times": int(total["sample_count"].item()),
    }


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    calibration_path = args.calibration.expanduser().resolve()
    calibration = load_calibration(calibration_path)
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(), verify_source=True
    )
    model, heads, model_metadata = load_etg_model(
        checkpoint_paths=calibration_checkpoints(calibration),
        head_weights=str(calibration["model"]["head_weights"]),
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if model_metadata != calibration["model"]:
        raise ValueError("model differs from ETG calibration")
    centers = torch.tensor(calibration["config"]["time_bin_centers"], device=device)
    points = torch.cat((torch.zeros(1, device=device), centers))
    scalar_weights = torch.tensor(
        calibration["rollout_calibration"]["scalar"]["weights"],
        dtype=torch.float32,
        device=device,
    )
    channel_weights = torch.tensor(
        calibration["rollout_calibration"]["channel"]["weights"],
        dtype=torch.float32,
        device=device,
    )
    source = model_metadata["source"]
    internal_depth = int(source["internal_depth"])
    denominator_floor = float(source["denominator_floor"])
    names = (
        "etg_scalar_vs_velocity_gap",
        "etg_channel_vs_velocity_gap",
        "private_velocity_vs_common_gap",
        "private_clean_vs_common_gap",
        "private_epsilon_vs_common_gap",
    )
    accumulators = {name: empty_sums(len(centers)) for name in names}
    generator = torch.Generator(device=device).manual_seed(args.seed)
    produced = 0
    while produced < args.num_samples:
        current = min(args.batch_size, args.num_samples - produced)
        noise = torch.randn(current, *LATENT_SHAPE, generator=generator, device=device)
        labels = torch.randint(
            0, NUM_CLASSES, (current,), generator=generator, device=device
        )
        trajectory = odeint(
            strong_velocity(model, labels),
            noise,
            points,
            method="dopri5",
            atol=args.atol,
            rtol=args.rtol,
        )[1:]
        for time_index, (time_value, state) in enumerate(zip(centers, trajectory, strict=True)):
            times = time_value.expand(current)
            full, native = full_and_internal_predictions(
                model,
                heads,
                state,
                times,
                labels,
                internal_depth=internal_depth,
            )
            weak = predictions_to_velocity(
                native,
                state=state,
                time_value=times,
                denominator_floor=denominator_floor,
            )
            common_scalar = fuse_predictions(weak, scalar_weights[time_index, :, 0])
            common_channel = fuse_predictions(weak, channel_weights[time_index])
            velocity_gap = full.float() - weak["velocity"].float()
            scalar_gap = full.float() - common_scalar
            channel_gap = full.float() - common_channel
            update_geometry(
                accumulators["etg_scalar_vs_velocity_gap"],
                time_index=time_index,
                reference=velocity_gap,
                candidate=scalar_gap,
            )
            update_geometry(
                accumulators["etg_channel_vs_velocity_gap"],
                time_index=time_index,
                reference=velocity_gap,
                candidate=channel_gap,
            )
            for target in TARGETS:
                update_geometry(
                    accumulators[f"private_{target}_vs_common_gap"],
                    time_index=time_index,
                    reference=channel_gap,
                    candidate=weak[target].float() - common_channel,
                )
        produced += current
        print(json.dumps({"event": "progress", "samples": produced}), flush=True)

    by_time = {name: summarize_geometry(values) for name, values in accumulators.items()}
    overall = {name: overall_geometry(values) for name, values in accumulators.items()}
    output = args.output.expanduser().resolve()
    payload = {
        "format": "eqvae_imagenet100_sit_etg_geometry_v1",
        "calibration": str(calibration_path),
        "sample_count": args.num_samples,
        "seed": args.seed,
        "time_bin_centers": centers.tolist(),
        "overall": overall,
        "by_time": by_time,
    }
    atomic_json_dump(to_jsonable(payload), output)
    csv_path = output.with_suffix(".csv")
    rows = []
    for name, metrics in by_time.items():
        for index, time_value in enumerate(centers.tolist()):
            rows.append(
                {
                    "comparison": name,
                    "time": time_value,
                    **{key: value[index].item() for key, value in metrics.items()},
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "complete", "output": str(output), "overall": overall}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260832)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
