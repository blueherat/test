#!/usr/bin/env python3
"""Build lightweight checkpoints extrapolated along one training trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from experiments.imagenet100_sit_weight_extrapolation import (
        extrapolate_state_dict,
        format_scale,
        validate_weight_extrapolation_pair,
    )
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
except ModuleNotFoundError:
    from imagenet100_sit_weight_extrapolation import (
        extrapolate_state_dict,
        format_scale,
        validate_weight_extrapolation_pair,
    )
    from train_imagenet100_sit_flow import atomic_json_dump, sha256_file


DEFAULT_BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_STRONG = DEFAULT_BASE / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_WEAK = DEFAULT_BASE / "runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
DEFAULT_OUTPUT = DEFAULT_BASE / "weight_extrapolation_v800_v500_v1/checkpoints"
DEFAULT_SCALES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not scales or len(set(scales)) != len(scales):
        raise argparse.ArgumentTypeError("scales must be non-empty and unique")
    if any(not torch.isfinite(torch.tensor(scale)) for scale in scales):
        raise argparse.ArgumentTypeError("scales must be finite")
    return scales


def main(args: argparse.Namespace) -> None:
    strong_path = args.strong_checkpoint.expanduser().resolve()
    weak_path = args.weak_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strong_sha = sha256_file(strong_path)
    weak_sha = sha256_file(weak_path)
    strong = torch.load(
        strong_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    weak = torch.load(
        weak_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_weight_extrapolation_pair(strong, weak, weights=args.weights)

    records: list[dict[str, object]] = []
    for scale in args.scales:
        tag = format_scale(scale)
        output_path = (
            output_dir
            / f"v800_plus_g{tag}_v800_minus_v500_{args.weights}.pt"
        )
        expected_metadata = {
            "formula": "theta_strong + gamma * (theta_strong - theta_weak)",
            "gamma": float(scale),
            "weights": args.weights,
            "strong_checkpoint": str(strong_path),
            "strong_checkpoint_sha256": strong_sha,
            "strong_step": int(strong["step"]),
            "weak_checkpoint": str(weak_path),
            "weak_checkpoint_sha256": weak_sha,
            "weak_step": int(weak["step"]),
        }
        reuse = False
        if output_path.is_file():
            existing = torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            reuse = existing.get("weight_extrapolation") == expected_metadata
            del existing
        if not reuse:
            extrapolated = extrapolate_state_dict(
                strong[args.weights],
                weak[args.weights],
                scale=scale,
            )
            payload = {
                "protocol": strong["protocol"],
                "step": int(strong["step"]),
                "config": dict(strong["config"]),
                "data_manifest_sha256": strong.get("data_manifest_sha256"),
                "official_sit": strong.get("official_sit"),
                args.weights: extrapolated,
                "weight_extrapolation": expected_metadata,
            }
            temporary = output_path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, output_path)
            del extrapolated, payload
        record = {
            **expected_metadata,
            "checkpoint": str(output_path),
            "checkpoint_sha256": sha256_file(output_path),
            "checkpoint_size_bytes": output_path.stat().st_size,
            "reused": reuse,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    summary = {
        "protocol": "imagenet100_sit_weight_extrapolation_checkpoints_v1",
        "records": records,
    }
    atomic_json_dump(summary, output_dir / "weight_extrapolation_checkpoints.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--weak-checkpoint", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--scales",
        type=parse_scales,
        default=DEFAULT_SCALES,
        help="Comma-separated gamma values.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
