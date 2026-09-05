#!/usr/bin/env python3
"""Measure cancellation in PFR's Eulerian time response on SiT rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch_seed_schema import batch_seed  # noqa: E402
from experiments.pfr_eulerian_decomposition import (  # noqa: E402
    finite_eulerian_components,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    detect_adm_python,
    detect_data,
    detect_repo,
)
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (  # noqa: E402
    HORIZON,
    gamma_at,
    load_runtime,
)


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    denominator = (
        left_flat.square().sum(dim=1).sqrt()
        * right_flat.square().sum(dim=1).sqrt()
    ).clamp_min(1e-30)
    return (left_flat * right_flat).sum(dim=1) / denominator


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "q10": float(torch.quantile(tensor, 0.1)),
        "q90": float(torch.quantile(tensor, 0.9)),
    }


def ordinary_pair(
    runtime: Any,
    state: torch.Tensor,
    time_value: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    strong, weak = runtime.evaluate_pair(time_value, state, labels)
    gamma = gamma_at(float(time_value.detach().float().item()))
    return strong, weak, strong + gamma * (strong - weak)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home()
        / "data/eqvae/imagenet_sit_flow/pfr_eulerian_decomposition_v1/geometry",
    )
    args = parser.parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts must be positive")
    if args.num_samples % args.batch_size:
        raise ValueError("num-samples must be divisible by batch-size")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this audit requires CUDA")
    repo = detect_repo().resolve()
    data = detect_data().resolve()
    runtime, allocator = load_runtime(
        repo=repo,
        data=data,
        adm_python=detect_adm_python(),
        device=device,
        allocator_limit_gib=args.cuda_allocator_limit_gib,
    )
    times = torch.linspace(0.0, 0.5, 17, device=device)
    values: dict[tuple[int, str], list[float]] = {}
    identity_max = 0.0

    def collect(time_index: int, name: str, tensor: torch.Tensor) -> None:
        values.setdefault((time_index, name), []).extend(
            tensor.detach().float().cpu().tolist()
        )

    with torch.inference_mode():
        for batch_index in range(args.num_samples // args.batch_size):
            generator = torch.Generator(device=device).manual_seed(
                batch_seed(args.seed, batch_index, schema="namespaced_v2")
            )
            state = torch.randn(
                args.batch_size,
                *runtime.modules["LATENT_SHAPE"],
                generator=generator,
                device=device,
            )
            labels = torch.randint(
                0,
                runtime.modules["NUM_CLASSES"],
                (args.batch_size,),
                generator=generator,
                device=device,
            )
            for index in range(len(times) - 1):
                time_value = times[index]
                next_time = times[index + 1]
                step = float((next_time - time_value).item())
                _, weak, guided = ordinary_pair(
                    runtime, state, time_value, labels
                )
                weak_time = runtime.evaluate_weak(next_time, state, labels)
                weak_material = runtime.evaluate_weak(
                    next_time, state + step * guided, labels
                )
                parts = finite_eulerian_components(
                    weak, weak_time, weak_material
                )
                residual = parts.eulerian - parts.material - parts.frame
                identity_max = max(
                    identity_max,
                    float(residual.float().abs().max().item()),
                )

                eulerian_rms = sample_rms(parts.eulerian)
                material_rms = sample_rms(parts.material)
                frame_rms = sample_rms(parts.frame)
                collect(index, "eulerian_rms", eulerian_rms)
                collect(index, "material_rms", material_rms)
                collect(index, "frame_rms", frame_rms)
                collect(
                    index,
                    "material_frame_cosine",
                    sample_cosine(parts.material, parts.frame),
                )
                collect(
                    index,
                    "material_negative_frame_cosine",
                    sample_cosine(parts.material, -parts.frame),
                )
                collect(
                    index,
                    "cancellation_fraction",
                    1.0
                    - eulerian_rms
                    / (material_rms + frame_rms).clamp_min(1e-30),
                )
                collect(
                    index,
                    "eulerian_to_material_ratio",
                    eulerian_rms / material_rms.clamp_min(1e-30),
                )
                collect(
                    index,
                    "eulerian_guided_cosine",
                    sample_cosine(parts.eulerian, guided),
                )

                _, _, guided_predictor = ordinary_pair(
                    runtime, state + step * guided, next_time, labels
                )
                state = state + 0.5 * step * (guided + guided_predictor)

    rows: list[dict[str, Any]] = []
    metric_names = sorted({name for _, name in values})
    for index, time_value in enumerate(times[:-1].cpu().tolist()):
        row: dict[str, Any] = {
            "time": float(time_value),
            "horizon": HORIZON,
            "samples": args.num_samples,
        }
        for name in metric_names:
            stats = summarize(values[(index, name)])
            for statistic, value in stats.items():
                row[f"{name}_{statistic}"] = value
        rows.append(row)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "eulerian_decomposition_by_time.csv", rows)
    summary = {
        "format": "eqvae_pfr_eulerian_decomposition_geometry_v1",
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "horizon": HORIZON,
        "trajectory": "ordinary_internal_guidance_heun32",
        "identity_max_abs_error": identity_max,
        "allocator": allocator,
        "metrics": {
            name: summarize(
                [value for (index, key), group in values.items() if key == name for value in group]
            )
            for name in metric_names
        },
    }
    if not math.isfinite(identity_max) or identity_max > 1e-5:
        raise AssertionError(f"finite decomposition did not close: {identity_max}")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
