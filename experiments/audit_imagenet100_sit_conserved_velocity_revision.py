#!/usr/bin/env python3
"""Audit cross-time weak references against the conserved FM velocity target.

For the linear path ``Z_t=(1-t)E+tX``, every prediction time estimates the
same conditional particle velocity ``U=X-E``.  This script uses held-out
ImageNet-100 latents to distinguish a genuine posterior improvement from a
structured counterfactual weak reference.  It is a teacher-path diagnostic;
its MSE values are not generation-quality metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.internal_guidance_path_extrapolation import (
    project_to_forward_ray,
    split_internal_guidance,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (
    load_repo_modules,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import parse_float_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae"
        ),
    )
    parser.add_argument(
        "--official-sit-repo",
        type=Path,
        default=Path("/home/zhoushunyu/data/research_repos/SiT"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--times",
        type=parse_float_list,
        default=parse_float_list("0.05,0.1,0.2,0.3,0.4,0.47"),
    )
    parser.add_argument("--horizon", type=float, default=0.03125)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_validation_latents(
    cache_dir: Path,
    *,
    samples: int,
    seed: int,
    scaling_factor: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    moments = np.load(cache_dir / "validation_moments.npy", mmap_mode="r")
    labels = np.load(cache_dir / "validation_labels.npy", mmap_mode="r")
    if samples > len(moments):
        raise ValueError("samples exceeds the held-out latent bank")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(moments), samples, replace=False)
    selected = torch.from_numpy(np.asarray(moments[indices]).copy()).to(device)
    selected_labels = torch.from_numpy(
        np.asarray(labels[indices]).copy()
    ).long().to(device)
    mean, std = selected.chunk(2, dim=1)
    generator = torch.Generator(device=device.type).manual_seed(seed + 1)
    posterior_noise = torch.randn(
        mean.shape, generator=generator, device=device, dtype=mean.dtype
    )
    clean = (mean + std * posterior_noise) * scaling_factor
    flow_noise = torch.randn(
        clean.shape, generator=generator, device=device, dtype=clean.dtype
    )
    return clean, flow_noise, selected_labels


def per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().flatten(1).mean(1)


def per_sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        left.double().flatten(1), right.double().flatten(1), dim=1
    ).float()


def summarize(values: list[torch.Tensor]) -> dict[str, float]:
    merged = torch.cat(values).float().cpu()
    return {
        "mean": float(merged.mean()),
        "median": float(merged.median()),
        "q10": float(torch.quantile(merged, 0.1)),
        "q90": float(torch.quantile(merged, 0.9)),
    }


def gamma_at(time_value: float) -> float:
    if time_value < 0.25:
        return 0.6
    if time_value < 0.5:
        return 0.7
    return 0.0


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.official_sit_repo = args.official_sit_repo.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if not 0.0 < args.horizon < 0.5:
        raise ValueError("horizon must lie in (0, 0.5)")
    if any(not 0.0 < value < 0.5 for value in args.times):
        raise ValueError("audit times must lie in (0, 0.5)")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    modules = load_repo_modules(REPO_ROOT)
    paths = {
        "strong": args.data_root
        / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt",
        "depth4": args.data_root
        / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))
    sit_module, source_metadata = modules["load_official_sit_module"](
        args.official_sit_repo,
        verify_source=True,
    )
    strong, semantics, strong_metadata = modules["load_sit_field_model"](
        checkpoint_path=paths["strong"],
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("the source model must use native velocity prediction")
    head = modules["load_internal_head_for_source"](
        checkpoint_path=paths["depth4"],
        name="depth4_v",
        head_weights="ema",
        model=strong,
        sit_module=sit_module,
        source_checkpoint_path=paths["strong"],
        source_metadata=source_metadata,
        device=device,
    )
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_internal_head_only,
    )

    clean, flow_noise, labels = load_validation_latents(
        args.cache_dir,
        samples=args.samples,
        seed=args.seed,
        scaling_factor=float(modules["SD_VAE_SCALING_FACTOR"]),
        device=device,
    )
    target = clean - flow_noise
    rows: list[dict[str, float | int | str]] = []
    query_names = (
        "current",
        "time_only",
        "weak_characteristic",
        "strong_characteristic",
        "projected_refinement",
    )
    for time_value in args.times:
        horizon = min(args.horizon, 0.5 - float(time_value))
        reference_time_value = float(time_value) + horizon
        gamma = gamma_at(float(time_value))
        metrics: dict[str, list[torch.Tensor]] = {}
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            batch_clean = clean[start:stop]
            batch_noise = flow_noise[start:stop]
            batch_labels = labels[start:stop]
            batch_target = target[start:stop]
            time = torch.full(
                (len(batch_clean),), float(time_value), device=device
            )
            time_image = time.reshape(-1, 1, 1, 1)
            state = (1.0 - time_image) * batch_noise + time_image * batch_clean
            full, trained, _ = modules["evaluate_source_with_heads"](
                strong,
                state,
                time,
                batch_labels,
                heads={"depth4_v": head},
            )
            weak = trained["depth4_v"]
            weak_base, calibration = split_internal_guidance(
                full,
                weak,
                gamma=gamma,
            )
            guided = weak_base + calibration
            projected = project_to_forward_ray(calibration, guided)
            query_states = {
                "current": state,
                "time_only": state,
                "weak_characteristic": state + horizon * weak,
                "strong_characteristic": state + horizon * full,
                "projected_refinement": state + horizon * projected.parallel,
            }
            query_velocities = {"current": weak}
            reference_time = torch.full(
                (len(batch_clean),), reference_time_value, device=device
            )
            for name in query_names[1:]:
                query_velocities[name] = evaluate_internal_head_only(
                    strong,
                    query_states[name],
                    reference_time,
                    batch_labels,
                    spec=head,
                )
            weak_error = batch_target - weak
            strong_error = batch_target - full
            depth_gap = full - weak
            ordinary = full + gamma * depth_gap
            metrics.setdefault("strong_mse", []).append(
                per_sample_mse(full, batch_target)
            )
            metrics.setdefault("ordinary_ig_mse", []).append(
                per_sample_mse(ordinary, batch_target)
            )
            metrics.setdefault("projected_alpha", []).append(
                projected.coefficient
            )
            for name, reference in query_velocities.items():
                revision = weak - reference
                revised = ordinary + (1.0 + gamma) * revision
                metrics.setdefault(f"{name}_mse", []).append(
                    per_sample_mse(reference, batch_target)
                )
                metrics.setdefault(f"{name}_distance_to_strong", []).append(
                    per_sample_mse(reference, full)
                )
                metrics.setdefault(f"{name}_guided_mse", []).append(
                    per_sample_mse(revised, batch_target)
                )
                metrics.setdefault(
                    f"{name}_revision_weak_error_cosine", []
                ).append(per_sample_cosine(revision, weak_error))
                metrics.setdefault(
                    f"{name}_revision_strong_error_cosine", []
                ).append(per_sample_cosine(revision, strong_error))
                metrics.setdefault(f"{name}_revision_depth_gap_cosine", []).append(
                    per_sample_cosine(revision, depth_gap)
                )

        row: dict[str, float | int | str] = {
            "time": float(time_value),
            "reference_time": reference_time_value,
            "horizon": horizon,
            "gamma": gamma,
            "samples": args.samples,
        }
        for name, values in metrics.items():
            for statistic, value in summarize(values).items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)
        print(
            f"[t={time_value:g}] "
            f"S={row['strong_mse_mean']:.4f} "
            f"W={row['current_mse_mean']:.4f} "
            f"Wt+h={row['time_only_mse_mean']:.4f} "
            f"Wproj={row['projected_refinement_mse_mean']:.4f} "
            f"IG={row['ordinary_ig_mse_mean']:.4f} "
            f"PFR={row['projected_refinement_guided_mse_mean']:.4f}",
            flush=True,
        )

    csv_path = args.output_root / "conserved_velocity_revision.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "format": "eqvae_conserved_velocity_revision_audit_v1",
        "interpretation": (
            "teacher-path MSE audit only; not an unconditional generation metric"
        ),
        "target": "U=X-E shared by every time on the linear conditional path",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "strong": strong_metadata,
        "head": {
            "checkpoint": head.checkpoint,
            "checkpoint_sha256": head.checkpoint_sha256,
            "depth": head.depth,
            "prediction_target": head.prediction_target,
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
