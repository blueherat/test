#!/usr/bin/env python3
"""Audit whether SiT weak-to-strong guidance gaps are conservative fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from experiments.finite_guidance_dynamics import jacobian_symmetry_probe
    from experiments.run_imagenet100_sit_finite_guidance import (
        DEFAULT_ANCHOR,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_V270,
        DEFAULT_X400,
        _load_pair,
        _parse_floats,
        _summary,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        sample_sdvae_posterior,
        sha256_file,
    )
except ModuleNotFoundError:
    from finite_guidance_dynamics import jacobian_symmetry_probe
    from run_imagenet100_sit_finite_guidance import (
        DEFAULT_ANCHOR,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_V270,
        DEFAULT_X400,
        _load_pair,
        _parse_floats,
        _summary,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        sample_sdvae_posterior,
        sha256_file,
    )


DEFAULT_CACHE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae"
)


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validation_bank(
    cache_dir: Path,
    *,
    num_samples: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    moments = np.load(cache_dir / "validation_moments.npy", mmap_mode="r")
    labels = np.load(cache_dir / "validation_labels.npy", mmap_mode="r")
    if tuple(moments.shape[1:]) != (8, 32, 32) or len(moments) != len(labels):
        raise ValueError("invalid ImageNet-100 validation latent cache")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(moments), generator=generator)[:num_samples]
    selected_moments = torch.from_numpy(
        np.asarray(moments[indices.numpy()], dtype=np.float32)
    )
    selected_labels = torch.from_numpy(
        np.asarray(labels[indices.numpy()], dtype=np.int64)
    )
    posterior_noise = torch.randn(num_samples, *LATENT_SHAPE, generator=generator)
    bridge_noise = torch.randn(num_samples, *LATENT_SHAPE, generator=generator)
    clean = sample_sdvae_posterior(
        selected_moments,
        posterior_noise,
        scaling_factor=SD_VAE_SCALING_FACTOR,
    )
    return {
        "indices": indices,
        "labels": selected_labels,
        "clean": clean,
        "bridge_noise": bridge_noise,
        "posterior_noise": posterior_noise,
    }


def _collect_rollout_states(
    anchor_field,
    initial_state: torch.Tensor,
    *,
    steps: int,
    requested_times: list[float],
) -> dict[float, torch.Tensor]:
    """Collect Heun baseline states at grid-aligned requested times."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    requested_indices: dict[int, float] = {}
    for value in requested_times:
        index = round(float(value) * steps)
        if not np.isclose(index / steps, value, rtol=0.0, atol=1e-7):
            raise ValueError(f"requested time {value} is not aligned to {steps} steps")
        requested_indices[index] = float(value)
    state = initial_state.float()
    collected: dict[float, torch.Tensor] = {}
    grid = torch.linspace(0.0, 1.0, steps + 1, device=state.device)
    with torch.no_grad():
        for index, (time_value, next_time) in enumerate(
            zip(grid[:-1], grid[1:], strict=True), start=1
        ):
            step = next_time - time_value
            derivative = anchor_field(time_value, state)
            predicted = state + step * derivative
            corrected = anchor_field(next_time, predicted)
            state = state + 0.5 * step * (derivative + corrected)
            if index in requested_indices:
                collected[requested_indices[index]] = state.detach().clone()
    if set(collected) != set(requested_times):
        raise RuntimeError("failed to collect every requested rollout time")
    return collected


def _teacher_states(
    clean: torch.Tensor,
    noise: torch.Tensor,
    requested_times: list[float],
) -> dict[float, torch.Tensor]:
    return {
        value: (1.0 - float(value)) * noise + float(value) * clean
        for value in requested_times
    }


def _rademacher_like(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    cpu = torch.randint(
        0,
        2,
        value.shape,
        generator=generator,
        device="cpu",
        dtype=torch.int8,
    )
    return cpu.to(device=value.device, dtype=value.dtype).mul_(2).sub_(1)


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = [
        "field_rms",
        "jvp_rms",
        "vjp_rms",
        "antisymmetric_rms",
        "antisymmetric_over_jvp_rms",
        "antisymmetric_energy_fraction",
        "jvp_vjp_cosine",
    ]
    keys = sorted(
        {(str(row["source"]), float(row["time"])) for row in rows},
        key=lambda item: (item[0], item[1]),
    )
    aggregate_rows: list[dict[str, object]] = []
    for source, time_value in keys:
        selected = [
            row
            for row in rows
            if row["source"] == source and float(row["time"]) == time_value
        ]
        aggregate: dict[str, object] = {
            "source": source,
            "time": time_value,
            "observations": len(selected),
        }
        for metric in metric_names:
            values = torch.tensor([float(row[metric]) for row in selected])
            for statistic, value in _summary(values).items():
                aggregate[f"{metric}_{statistic}"] = value
        aggregate_rows.append(aggregate)
    return aggregate_rows


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.probes <= 0:
        raise ValueError("sample, batch, and probe counts must be positive")
    if args.num_samples % args.batch_size != 0:
        raise ValueError("num-samples must be divisible by batch-size")
    if any(value <= 0 or value >= 1 for value in args.times):
        raise ValueError("all audit times must lie strictly inside (0, 1)")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    cache_dir = args.cache_dir.expanduser().resolve()
    bank = _validation_bank(cache_dir, num_samples=args.num_samples, seed=args.seed)
    initial_labels = bank["labels"][: args.batch_size].to(device)
    # Forward-mode AD through scaled-dot-product attention requires the math kernel.
    args.math_attention = True
    fields, pair_metadata = _load_pair(args, initial_labels, device)
    base_output = args.output_root.expanduser().resolve()
    if args.component == "direction":
        output_dir = base_output / "conservativity" / args.direction
    else:
        output_dir = (
            base_output / "conservativity_controls" / args.component / args.direction
        )
    output_dir = output_dir / f"n{args.num_samples}_p{args.probes}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    probe_generator = torch.Generator(device="cpu").manual_seed(args.probe_seed)

    for start in range(0, args.num_samples, args.batch_size):
        stop = start + args.batch_size
        fields.labels = bank["labels"][start:stop].to(device)
        clean = bank["clean"][start:stop].to(device)
        noise = bank["bridge_noise"][start:stop].to(device)
        teacher = _teacher_states(clean, noise, args.times)
        rollout = _collect_rollout_states(
            fields.anchor,
            noise,
            steps=args.heun_steps,
            requested_times=args.times,
        )
        for source, source_states in (("teacher", teacher), ("v400_rollout", rollout)):
            for time_value in args.times:
                state = source_states[time_value]
                time_tensor = torch.tensor(time_value, device=device)

                def direction_at_state(current_state: torch.Tensor) -> torch.Tensor:
                    anchor = fields.anchor(time_tensor, current_state)
                    if args.component == "anchor":
                        return anchor
                    direction = fields.direction(time_tensor, current_state, anchor)
                    if args.component == "direction":
                        return direction
                    return anchor - direction

                for probe_index in range(args.probes):
                    probe = _rademacher_like(state, probe_generator)
                    metrics = jacobian_symmetry_probe(direction_at_state, state, probe)
                    for local_index in range(args.batch_size):
                        row: dict[str, object] = {
                            "sample_id": start + local_index,
                            "validation_index": int(bank["indices"][start + local_index]),
                            "label": int(bank["labels"][start + local_index]),
                            "source": source,
                            "time": float(time_value),
                            "probe": probe_index,
                        }
                        for name, values in metrics.items():
                            row[name] = float(values[local_index].detach().cpu())
                        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "samples": [start, stop],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    aggregate_rows = _aggregate(rows)
    _write_csv(rows, output_dir / "per_sample_probe_metrics.csv")
    _write_csv(aggregate_rows, output_dir / "metrics_by_source_time.csv")
    summary = {
        "format": "eqvae_sit400_guidance_conservativity_v1",
        "direction": args.direction,
        "component": args.component,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "probes": args.probes,
        "times": args.times,
        "heun_steps": args.heun_steps,
        "seed": args.seed,
        "probe_seed": args.probe_seed,
        "precision": "fp32",
        "allow_tf32": False,
        "math_attention": True,
        "state_sources": ["teacher", "v400_rollout"],
        "normalization_note": (
            "score-gap scaling t/(1-t) is state-independent at fixed t and "
            "therefore cancels from normalized Jacobian-symmetry metrics"
        ),
        "cache_dir": str(cache_dir),
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "validation_indices_sha256": _tensor_sha256(bank["indices"]),
        "labels_sha256": _tensor_sha256(bank["labels"]),
        "clean_latents_sha256": _tensor_sha256(bank["clean"]),
        "bridge_noise_sha256": _tensor_sha256(bank["bridge_noise"]),
        "pair": pair_metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate_rows": aggregate_rows,
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps({"event": "complete", **summary}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("x400", "v270"), required=True)
    parser.add_argument(
        "--component",
        choices=("direction", "anchor", "other"),
        default="direction",
    )
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--x400-checkpoint", type=Path, default=DEFAULT_X400)
    parser.add_argument("--v270-checkpoint", type=Path, default=DEFAULT_V270)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--times", type=_parse_floats, default=_parse_floats("0.1,0.3,0.5,0.7,0.9"))
    parser.add_argument("--heun-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--probe-seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
