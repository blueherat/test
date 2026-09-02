#!/usr/bin/env python3
"""Endpoint-error audit for future-state implicit IG solvers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from experiments.implicit_fixed_point_solvers import integrate_fixed_grid  # noqa: E402
from experiments.run_imagenet100_sit_implicit_ig_solver import (  # noqa: E402
    Condition,
    detect_data,
    detect_repo,
    load_repo_modules,
    runtime_paths,
    segment_step_counts,
)


CONDITIONS = (
    Condition("euler", 64),
    Condition("heun", 32),
    Condition("backward_euler", 32, 1),
    Condition("backward_euler", 32, 2),
    Condition("backward_euler", 32, 3),
    Condition("implicit_midpoint", 32, 1),
    Condition("implicit_midpoint", 32, 2),
    Condition("implicit_midpoint", 32, 3),
    Condition("implicit_trapezoid", 32, 1),
    Condition("implicit_trapezoid", 32, 2),
    Condition("implicit_trapezoid", 32, 3),
)


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    return torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1)


def summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    return {
        "mean": float(value.mean()),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
            "implicit_future_state_ig_solver_v1/trajectory_audit.json"
        ),
    )
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo = detect_repo()
    data = detect_data()
    paths = runtime_paths(repo, data, Path("/data/shared/envs/adm-fid/bin/python"))
    modules = load_repo_modules(repo)
    from torchdiffeq import odeint

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    sit_module, source_metadata = modules["load_official_sit_module"](
        Path(modules["DEFAULT_OFFICIAL_SIT_REPO"]).expanduser().resolve(),
        verify_source=True,
    )
    strong, semantics, _ = modules["load_sit_field_model"](
        checkpoint_path=paths["strong"],
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("expected native velocity model")
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
    heads = {"depth4_v": head}

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.samples, *modules["LATENT_SHAPE"], generator=generator, device=device
    )
    labels = torch.randint(
        0, modules["NUM_CLASSES"], (args.samples,), generator=generator, device=device
    )

    class Field:
        def __init__(self, gamma: float):
            self.gamma = float(gamma)
            self.nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            times = time.expand(len(latent))
            full, trained, _ = modules["evaluate_source_with_heads"](
                strong, latent, times, labels, heads=heads
            )
            if self.gamma == 0.0:
                return full
            return full + self.gamma * (full - trained["depth4_v"])

    segments = ((0.0, 0.25, 0.6), (0.25, 0.5, 0.7), (0.5, 1.0, 0.0))

    def reference_endpoint(rtol: float, atol: float) -> tuple[torch.Tensor, int]:
        state = noise.float()
        nfe = 0
        for start, end, gamma in segments:
            field = Field(gamma)
            state = odeint(
                field,
                state,
                torch.tensor([start, end], device=device),
                method="dopri5",
                rtol=rtol,
                atol=atol,
            )[-1]
            nfe += field.nfe
        return state, nfe

    with torch.inference_mode():
        reference, reference_nfe = reference_endpoint(1e-7, 1e-9)
        default_dopri, default_nfe = reference_endpoint(1e-3, 1e-6)
        rows = []
        default_error = default_dopri - reference
        rows.append(
            {
                "condition": "segmented_dopri5_rtol1e-3",
                "nfe": default_nfe,
                "endpoint_error_rms": summary(sample_rms(default_error)),
                "relative_error_rms": summary(
                    sample_rms(default_error) / sample_rms(reference).clamp_min(1e-12)
                ),
                "transport_cosine": summary(cosine(default_dopri - noise, reference - noise)),
                "picard_last_update_rms_mean": 0.0,
                "picard_last_update_rms_max": 0.0,
            }
        )
        for condition in CONDITIONS:
            state = noise.float()
            nfe = 0
            means: list[float] = []
            maxima: list[float] = []
            for (start, end, gamma), count in zip(
                segments, segment_step_counts(condition.steps)
            ):
                result = integrate_fixed_grid(
                    Field(gamma),
                    state,
                    torch.linspace(start, end, count + 1, device=device),
                    method=condition.method,
                    corrections=condition.corrections,
                    relaxation=condition.relaxation,
                )
                state = result.endpoint
                nfe += result.nfe
                means.append(result.mean_last_update_rms)
                maxima.append(result.max_last_update_rms)
            error = state - reference
            rows.append(
                {
                    "condition": condition.name,
                    "nfe": nfe,
                    "endpoint_error_rms": summary(sample_rms(error)),
                    "relative_error_rms": summary(
                        sample_rms(error) / sample_rms(reference).clamp_min(1e-12)
                    ),
                    "transport_cosine": summary(cosine(state - noise, reference - noise)),
                    "picard_last_update_rms_mean": sum(means) / len(means),
                    "picard_last_update_rms_max": max(maxima),
                }
            )

    payload = {
        "format": "eqvae_implicit_ig_trajectory_audit_v1",
        "samples": args.samples,
        "seed": args.seed,
        "reference": {
            "method": "segmented_dopri5",
            "rtol": 1e-7,
            "atol": 1e-9,
            "nfe": reference_nfe,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
