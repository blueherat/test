#!/usr/bin/env python3
"""Audit conservativity of reciprocal common/unique SiT guidance components."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from experiments.run_imagenet100_sit_guidance_conservativity import (
        DEFAULT_CACHE,
        _collect_rollout_states,
        _rademacher_like,
        _teacher_states,
        _validation_bank,
    )
    from experiments.run_imagenet100_sit_guidance_density_action import (
        COMPONENT_NAMES,
        DEFAULT_OUTPUT_ROOT,
        _load_triplet,
        component_jacobian_symmetry_probe,
    )
    from experiments.run_imagenet100_sit_finite_guidance import (
        DEFAULT_ANCHOR,
        DEFAULT_V270,
        DEFAULT_X400,
        _parse_floats,
    )
    from experiments.train_imagenet100_sit_flow import DEFAULT_OFFICIAL_SIT_REPO
except ModuleNotFoundError:
    from run_imagenet100_sit_guidance_conservativity import (
        DEFAULT_CACHE,
        _collect_rollout_states,
        _rademacher_like,
        _teacher_states,
        _validation_bank,
    )
    from run_imagenet100_sit_guidance_density_action import (
        COMPONENT_NAMES,
        DEFAULT_OUTPUT_ROOT,
        _load_triplet,
        component_jacobian_symmetry_probe,
    )
    from run_imagenet100_sit_finite_guidance import (
        DEFAULT_ANCHOR,
        DEFAULT_V270,
        DEFAULT_X400,
        _parse_floats,
    )
    from train_imagenet100_sit_flow import DEFAULT_OFFICIAL_SIT_REPO


DEFAULT_COMPONENTS = (
    "x_common_on_v",
    "x_unique_to_v",
    "v_common_on_x",
    "v_unique_to_x",
)


def _parse_components(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in parsed if item not in COMPONENT_NAMES]
    if not parsed or invalid:
        raise argparse.ArgumentTypeError(f"invalid component names: {invalid}")
    return parsed


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "field_rms",
        "jvp_rms",
        "vjp_rms",
        "antisymmetric_rms",
        "antisymmetric_over_jvp_rms",
        "antisymmetric_energy_fraction",
        "jvp_vjp_cosine",
    )
    output = []
    keys = sorted({(row["source"], row["time"], row["component"]) for row in rows})
    for source, time_value, component in keys:
        selected = [
            row
            for row in rows
            if row["source"] == source
            and row["time"] == time_value
            and row["component"] == component
        ]
        aggregate: dict[str, object] = {
            "source": source,
            "time": time_value,
            "component": component,
            "observations": len(selected),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_median"] = float(np.median(values))
            aggregate[f"{metric}_q10"] = float(np.quantile(values, 0.1))
            aggregate[f"{metric}_q90"] = float(np.quantile(values, 0.9))
        output.append(aggregate)
    return output


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.probes <= 0:
        raise ValueError("sample, batch, and probe counts must be positive")
    if args.num_samples % args.batch_size:
        raise ValueError("num-samples must be divisible by batch-size")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    bank = _validation_bank(
        args.cache_dir.expanduser().resolve(),
        num_samples=args.num_samples,
        seed=args.seed,
    )
    fields, metadata = _load_triplet(
        args, bank["labels"][: args.batch_size].to(device), device
    )
    component_indices = [COMPONENT_NAMES.index(name) for name in args.components]
    rows: list[dict[str, object]] = []
    generator = torch.Generator(device="cpu").manual_seed(args.probe_seed)
    started = time.perf_counter()
    for start in range(0, args.num_samples, args.batch_size):
        stop = start + args.batch_size
        fields.labels = bank["labels"][start:stop].to(device)
        clean = bank["clean"][start:stop].to(device)
        noise = bank["bridge_noise"][start:stop].to(device)
        teacher = _teacher_states(clean, noise, args.times)
        rollout = _collect_rollout_states(
            fields.anchor, noise, steps=args.heun_steps, requested_times=args.times
        )
        for source, source_states in (("teacher", teacher), ("v400_rollout", rollout)):
            for time_value in args.times:
                state = source_states[time_value]
                time_tensor = torch.tensor(time_value, device=device)

                def components(current_state: torch.Tensor) -> torch.Tensor:
                    return fields.components(time_tensor, current_state)

                for probe_index in range(args.probes):
                    probe = _rademacher_like(state, generator)
                    metrics = component_jacobian_symmetry_probe(
                        components, state, probe, component_indices
                    )
                    for selected_index, component in enumerate(args.components):
                        for local_index in range(args.batch_size):
                            row: dict[str, object] = {
                                "sample_id": start + local_index,
                                "validation_index": int(bank["indices"][start + local_index]),
                                "label": int(bank["labels"][start + local_index]),
                                "source": source,
                                "time": float(time_value),
                                "component": component,
                                "probe": probe_index,
                            }
                            for name, value in metrics.items():
                                row[name] = float(
                                    value[selected_index, local_index].detach().cpu()
                                )
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
    output_dir = (
        args.output_root.expanduser().resolve()
        / "component_conservativity"
        / f"n{args.num_samples}_p{args.probes}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate(rows)
    _write_csv(rows, output_dir / "per_sample_probe_metrics.csv")
    _write_csv(aggregate, output_dir / "metrics_by_source_time_component.csv")
    summary = {
        "format": "eqvae_sit400_component_conservativity_v1",
        "components": args.components,
        "samples": args.num_samples,
        "probes": args.probes,
        "times": args.times,
        "state_sources": ["teacher", "v400_rollout"],
        "precision": "fp32",
        "allow_tf32": False,
        "models": metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate_rows": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output_dir": str(output_dir)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--x400-checkpoint", type=Path, default=DEFAULT_X400)
    parser.add_argument("--v270-checkpoint", type=Path, default=DEFAULT_V270)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--components",
        type=_parse_components,
        default=list(DEFAULT_COMPONENTS),
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument(
        "--times", type=_parse_floats, default=_parse_floats("0.1,0.5,0.9")
    )
    parser.add_argument("--heun-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--probe-seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
