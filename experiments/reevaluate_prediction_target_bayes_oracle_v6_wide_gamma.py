#!/usr/bin/env python3
"""Re-evaluate saved v6 checkpoints with a wider extrapolation sweep."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from experiments.run_prediction_target_bayes_oracle_v5 import (
    TangentGaussianMixture,
    build_same_init_models,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    save_json,
    stable_seed,
)
from experiments.run_prediction_target_bayes_oracle_v6_trajectory import (
    evaluate_milestone,
)


def load_run_args(
    *,
    manifest_path: Path,
    gammas: list[float],
    geometry_gammas: list[float],
    device: str,
    resume: bool,
    save_samples: bool,
) -> Namespace:
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    values.update(
        {
            "gammas": gammas,
            "geometry_gammas": geometry_gammas,
            "device": device,
            "resume": resume,
            "save_samples": save_samples,
        }
    )
    return Namespace(**values)


def load_models_at_step(
    *,
    checkpoint: Path,
    architecture: str,
    hidden: int,
    args: Namespace,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], list[dict[str, float]]]:
    models = build_same_init_models(
        architecture,
        D=args.D,
        hidden=hidden,
        depth=args.depth,
        time_dim=args.time_dim,
        device=device,
        seed=stable_seed(seed, 101),
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for name, model in models.items():
        model.load_state_dict(payload["models"][name])
        model.eval()
    return models, list(payload["history"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--architectures", type=parse_str_list, default=parse_str_list("residual")
    )
    parser.add_argument(
        "--hidden-dims", type=parse_int_list, default=parse_int_list("64,80,96,128")
    )
    parser.add_argument(
        "--steps",
        type=parse_int_list,
        default=parse_int_list("6000,10000,15000,20000,30000"),
    )
    parser.add_argument(
        "--gammas",
        type=parse_float_list,
        default=parse_float_list("0.01,0.03,0.1,0.2,0.3,0.5,0.78,1.0"),
    )
    parser.add_argument(
        "--geometry-gammas",
        type=parse_float_list,
        default=parse_float_list("0.1,0.5"),
    )
    parser.add_argument(
        "--seeds", type=parse_int_list, default=parse_int_list("20260901")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    device = torch.device(cli.device)
    cli.output_root.mkdir(parents=True, exist_ok=True)
    for seed in cli.seeds:
        input_seed = cli.input_root / f"seed{seed}"
        args = load_run_args(
            manifest_path=input_seed / "manifest.json",
            gammas=list(cli.gammas),
            geometry_gammas=list(cli.geometry_gammas),
            device=cli.device,
            resume=cli.resume,
            save_samples=cli.save_samples,
        )
        output_seed = cli.output_root / f"seed{seed}"
        save_json(
            output_seed / "manifest.json",
            {
                "source": str(input_seed),
                "seed": seed,
                "architectures": cli.architectures,
                "hidden_dims": cli.hidden_dims,
                "steps": cli.steps,
                "gammas": cli.gammas,
                "geometry_gammas": cli.geometry_gammas,
                "role": "post-hoc wide-gamma discovery on frozen checkpoints",
            },
        )
        mixture = TangentGaussianMixture(
            D=args.D,
            components=args.components,
            curvature=args.curvature,
            frequency_scale=args.frequency_scale,
            center_rms=args.center_rms,
            sigma_tangent=args.sigma_tangent,
            sigma_normal=args.sigma_normal,
            seed=args.mixture_seed,
            device=device,
        )
        reference = np.load(input_seed / "common" / "reference.npy")
        bayes = np.load(input_seed / "common" / "bayes.npy")
        for architecture in cli.architectures:
            for hidden in cli.hidden_dims:
                setting_dir = input_seed / architecture / f"H{hidden}"
                for step in cli.steps:
                    checkpoint = setting_dir / "checkpoints" / f"step{step:06d}.pt"
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    models, history = load_models_at_step(
                        checkpoint=checkpoint,
                        architecture=architecture,
                        hidden=hidden,
                        args=args,
                        seed=seed,
                        device=device,
                    )
                    evaluate_milestone(
                        args=args,
                        models=models,
                        mixture=mixture,
                        architecture=architecture,
                        hidden=hidden,
                        seed=seed,
                        step=step,
                        history=history,
                        output_dir=(
                            output_seed
                            / architecture
                            / f"H{hidden}"
                            / f"step{step:06d}"
                        ),
                        reference=reference,
                        bayes=bayes,
                    )
                    del models
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
