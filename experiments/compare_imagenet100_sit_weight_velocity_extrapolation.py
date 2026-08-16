#!/usr/bin/env python3
"""Compare weight- and velocity-space extrapolation at identical rollout states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torchdiffeq import odeint

try:
    from experiments.build_imagenet100_sit_weight_extrapolation_checkpoints import (
        DEFAULT_STRONG,
        DEFAULT_WEAK,
        parse_scales,
    )
    from experiments.imagenet100_sit_weight_extrapolation import (
        extrapolate_state_dict,
        validate_weight_extrapolation_pair,
        velocity_extrapolation,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )
except ModuleNotFoundError:
    from build_imagenet100_sit_weight_extrapolation_checkpoints import (
        DEFAULT_STRONG,
        DEFAULT_WEAK,
        parse_scales,
    )
    from imagenet100_sit_weight_extrapolation import (
        extrapolate_state_dict,
        validate_weight_extrapolation_pair,
        velocity_extrapolation,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )


DEFAULT_OUTPUT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "weight_extrapolation_v800_v500_v1/local_velocity_comparison"
)
DEFAULT_TIMES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0)


def create_model(sit_module, checkpoint: dict, device: torch.device) -> torch.nn.Module:
    config = checkpoint["config"]
    model = sit_module.SiT_models[config["model_name"]](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config["cfg_dropout"]),
    )
    model.to(device).eval().requires_grad_(False)
    return model


def paired_metrics(
    weight_velocity: torch.Tensor,
    direct_velocity: torch.Tensor,
    strong_velocity: torch.Tensor,
) -> dict[str, float]:
    weight = weight_velocity.float().flatten(1)
    direct = direct_velocity.float().flatten(1)
    strong = strong_velocity.float().flatten(1)
    tiny = torch.finfo(weight.dtype).tiny

    def norm(value: torch.Tensor) -> torch.Tensor:
        return value.square().sum(1).sqrt()

    def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * right).sum(1) / (norm(left) * norm(right)).clamp_min(tiny)

    direct_delta = direct - strong
    weight_delta = weight - strong
    return {
        "velocity_cosine": float(cosine(weight, direct).mean()),
        "velocity_relative_l2": float(
            (norm(weight - direct) / norm(direct).clamp_min(tiny)).mean()
        ),
        "delta_cosine": float(cosine(weight_delta, direct_delta).mean()),
        "delta_norm_ratio": float(
            (norm(weight_delta) / norm(direct_delta).clamp_min(tiny)).mean()
        ),
        "delta_relative_l2": float(
            (norm(weight_delta - direct_delta) / norm(direct_delta).clamp_min(tiny)).mean()
        ),
        "weight_velocity_rms": float(weight.square().mean().sqrt()),
        "direct_velocity_rms": float(direct.square().mean().sqrt()),
    }


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    strong_path = args.strong_checkpoint.expanduser().resolve()
    weak_path = args.weak_checkpoint.expanduser().resolve()
    strong_checkpoint = torch.load(
        strong_path, map_location="cpu", weights_only=False, mmap=True
    )
    weak_checkpoint = torch.load(
        weak_path, map_location="cpu", weights_only=False, mmap=True
    )
    validate_weight_extrapolation_pair(
        strong_checkpoint,
        weak_checkpoint,
        weights="ema",
    )
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    if strong_checkpoint.get("official_sit") != source_metadata:
        raise ValueError("checkpoint and local SiT source differ")

    strong_model = create_model(sit_module, strong_checkpoint, device)
    weak_model = create_model(sit_module, weak_checkpoint, device)
    weight_model = create_model(sit_module, strong_checkpoint, device)
    strong_model.load_state_dict(strong_checkpoint["ema"], strict=True)
    weak_model.load_state_dict(weak_checkpoint["ema"], strict=True)

    labels = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=device)
    noise = torch.randn(args.batch_size, *LATENT_SHAPE, device=device)
    times = torch.tensor(args.times, device=device, dtype=torch.float32)

    def strong_velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return strong_model(
            state,
            time_value.expand(len(state)),
            labels,
        ).float()

    trajectory = odeint(
        strong_velocity,
        noise.float(),
        times,
        method="dopri5",
        atol=args.atol,
        rtol=args.rtol,
    )
    strong_outputs = []
    weak_outputs = []
    for time_value, state in zip(times, trajectory, strict=True):
        expanded = time_value.expand(args.batch_size)
        strong_outputs.append(strong_model(state, expanded, labels).float())
        weak_outputs.append(weak_model(state, expanded, labels).float())

    rows: list[dict[str, float]] = []
    for scale in args.scales:
        state_dict = extrapolate_state_dict(
            strong_checkpoint["ema"],
            weak_checkpoint["ema"],
            scale=scale,
        )
        weight_model.load_state_dict(state_dict, strict=True)
        del state_dict
        for time_value, state, strong_output, weak_output in zip(
            times,
            trajectory,
            strong_outputs,
            weak_outputs,
            strict=True,
        ):
            weight_output = weight_model(
                state,
                time_value.expand(args.batch_size),
                labels,
            ).float()
            direct_output = velocity_extrapolation(
                strong_output,
                weak_output,
                scale=scale,
            )
            rows.append(
                {
                    "gamma": float(scale),
                    "time": float(time_value),
                    **paired_metrics(weight_output, direct_output, strong_output),
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "weight_vs_velocity_same_state.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "imagenet100_sit_weight_vs_velocity_same_state_v1",
        "formula_weight": "theta800 + gamma * (theta800 - theta500)",
        "formula_velocity": "v800(z,t) + gamma * (v800(z,t) - v500(z,t))",
        "state_source": "v800 Dopri5 rollout with paired labels/noise",
        "strong_checkpoint": str(strong_path),
        "strong_checkpoint_sha256": sha256_file(strong_path),
        "weak_checkpoint": str(weak_path),
        "weak_checkpoint_sha256": sha256_file(weak_path),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "times": list(args.times),
        "scales": list(args.scales),
        "rows": rows,
        "csv": str(csv_path),
    }
    atomic_json_dump(summary, output_dir / "weight_vs_velocity_same_state.json")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--weak-checkpoint", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scales",
        type=parse_scales,
        default=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    )
    parser.add_argument(
        "--times",
        type=parse_scales,
        default=DEFAULT_TIMES,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
