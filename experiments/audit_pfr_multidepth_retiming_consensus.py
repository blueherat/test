#!/usr/bin/env python3
"""Test whether cross-depth retiming consensus predicts full-model consensus."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.internal_guidance_path_extrapolation import (  # noqa: E402
    project_per_sample,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
)
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (  # noqa: E402
    HORIZON,
    INTERVENTION_TIME,
    load_runtime,
)
from experiments.run_imagenet100_sit_pfr_query_controls import (  # noqa: E402
    QueryControlledField,
    integrate_times,
    summarize,
)


DEPTHS = (6, 8, 10)


def parse_times(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not result or tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    if any(not 0.0 < item < INTERVENTION_TIME for item in result):
        raise argparse.ArgumentTypeError("times must lie in (0, 0.5)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0.05,0.1,0.2,0.3,0.4,0.46875"),
    )
    parser.add_argument("--horizon", type=float, default=HORIZON)
    parser.add_argument("--depth6-head", type=Path)
    parser.add_argument("--depth8-head", type=Path)
    parser.add_argument("--depth10-head", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    return parser.parse_args()


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def sample_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    denominator = first_flat.norm(dim=1) * second_flat.norm(dim=1)
    return (first_flat * second_flat).sum(1) / denominator.clamp_min(1e-30)


def append(values: dict[str, list[torch.Tensor]], name: str, value: torch.Tensor) -> None:
    values.setdefault(name, []).append(value.detach().float().cpu())


def summarize_metrics(values: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, chunks in values.items():
        summary = summarize(torch.cat(chunks).tolist())
        if summary is None:
            raise RuntimeError(f"empty metric: {name}")
        for statistic, value in summary.items():
            result[f"{name}_{statistic}"] = float(value)
    return result


def resolve_head_paths(args: argparse.Namespace, data: Path) -> dict[int, Path]:
    defaults = {
        6: data
        / "multiscale_guidance_study_v1/runs/depth6_v/checkpoints/step_00050000.pt",
        8: data
        / "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
        "checkpoints/step_00050000.pt",
        10: data
        / "multiscale_guidance_study_v1/runs/depth10_v/checkpoints/step_00050000.pt",
    }
    overrides = {
        6: args.depth6_head,
        8: args.depth8_head,
        10: args.depth10_head,
    }
    paths = {
        depth: (overrides[depth] or defaults[depth]).expanduser().resolve()
        for depth in DEPTHS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing internal heads:\n  " + "\n  ".join(missing))
    return paths


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if args.horizon <= 0.0:
        raise ValueError("horizon must be positive")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = detect_repo()
    data = detect_data()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime, allocator = load_runtime(
        repo=repo,
        data=data,
        adm_python=detect_adm_python(),
        device=device,
        allocator_limit_gib=args.cuda_allocator_limit_gib,
    )
    head_paths = resolve_head_paths(args, data)
    sit_module, source_metadata = runtime.modules["load_official_sit_module"](
        Path(runtime.modules["DEFAULT_OFFICIAL_SIT_REPO"]).expanduser().resolve(),
        verify_source=True,
    )
    heads = {"depth4_v": runtime.head}
    for depth, path in head_paths.items():
        name = f"depth{depth}_v"
        heads[name] = runtime.modules["load_internal_head_for_source"](
            checkpoint_path=path,
            name=name,
            head_weights="ema",
            model=runtime.strong,
            sit_module=sit_module,
            source_checkpoint_path=runtime.paths["strong"],
            source_metadata=source_metadata,
            device=device,
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.samples,
        *runtime.modules["LATENT_SHAPE"],
        generator=generator,
        device=device,
    )
    labels = torch.randint(
        0,
        runtime.modules["NUM_CLASSES"],
        (args.samples,),
        generator=generator,
        device=device,
    )
    ordinary = QueryControlledField(
        runtime, labels, "ordinary_ig", record_diagnostics=False
    )
    states = integrate_times(
        ordinary,
        noise.float(),
        args.times,
        atol=args.atol,
        rtol=args.rtol,
    )

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for time_value, all_state in zip(args.times, states, strict=True):
        horizon = min(args.horizon, INTERVENTION_TIME - time_value)
        metrics: dict[str, list[torch.Tensor]] = {}
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            state = all_state[start:stop]
            batch_labels = labels[start:stop]
            time = torch.full((len(state),), time_value, device=device)
            future_time = torch.full(
                (len(state),), time_value + horizon, device=device
            )
            strong, current_heads, _ = runtime.modules["evaluate_source_with_heads"](
                runtime.strong,
                state,
                time,
                batch_labels,
                heads=heads,
            )
            strong_future, future_heads, _ = runtime.modules[
                "evaluate_source_with_heads"
            ](
                runtime.strong,
                state,
                future_time,
                batch_labels,
                heads=heads,
            )
            revisions = {
                name: current_heads[name] - future_heads[name] for name in heads
            }
            strong_revision = strong - strong_future
            depth4_revision = revisions["depth4_v"]
            true_common = project_per_sample(
                depth4_revision, strong_revision
            ).parallel

            batch_metrics: dict[str, torch.Tensor] = {
                "depth4_strong_cosine": sample_cosine(
                    depth4_revision, strong_revision
                ),
                "true_common_rms": sample_rms(true_common),
                "true_common_energy_fraction": (
                    sample_rms(true_common).square()
                    / sample_rms(depth4_revision).square().clamp_min(1e-30)
                ),
            }
            for depth in DEPTHS:
                candidate = revisions[f"depth{depth}_v"]
                proxy = project_per_sample(depth4_revision, candidate).parallel
                prefix = f"depth{depth}"
                batch_metrics.update(
                    {
                        f"{prefix}_cosine": sample_cosine(
                            depth4_revision, candidate
                        ),
                        f"{prefix}_proxy_true_cosine": sample_cosine(
                            proxy, true_common
                        ),
                        f"{prefix}_proxy_strong_cosine": sample_cosine(
                            proxy, strong_revision
                        ),
                        f"{prefix}_proxy_energy_fraction": (
                            sample_rms(proxy).square()
                            / sample_rms(depth4_revision).square().clamp_min(1e-30)
                        ),
                        f"{prefix}_proxy_true_difference_rms": sample_rms(
                            proxy - true_common
                        ),
                    }
                )
            for name, value in batch_metrics.items():
                append(metrics, name, value)
            cpu_metrics = {
                name: value.detach().float().cpu()
                for name, value in batch_metrics.items()
            }
            for offset in range(len(state)):
                sample_rows.append(
                    {
                        "time": time_value,
                        "sample": start + offset,
                        **{
                            name: float(value[offset])
                            for name, value in cpu_metrics.items()
                        },
                    }
                )

        row: dict[str, Any] = {
            "time": time_value,
            "future_time": time_value + horizon,
            "horizon": horizon,
            "samples": args.samples,
        }
        row.update(summarize_metrics(metrics))
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "time_complete",
                    "time": time_value,
                    "depth4_strong_cosine": row["depth4_strong_cosine_mean"],
                    **{
                        f"depth{depth}_proxy_true_cosine": row[
                            f"depth{depth}_proxy_true_cosine_mean"
                        ]
                        for depth in DEPTHS
                    },
                }
            ),
            flush=True,
        )

    for path, data_rows in (
        (output / "multidepth_summary.csv", rows),
        (output / "per_sample_multidepth.csv", sample_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
            writer.writeheader()
            writer.writerows(data_rows)

    atomic_json(
        output / "manifest.json",
        {
            "format": "eqvae_pfr_multidepth_retiming_consensus_v1",
            "question": (
                "Can agreement between internal depths isolate the depth4/full "
                "shared exponential-retiming defect?"
            ),
            "protocol": {
                "samples": args.samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "times": list(args.times),
                "horizon": args.horizon,
                "trajectory": "ordinary depth4 internal guidance",
                "query": "same latent at a later affine-flow time",
                "weights": "ema",
            },
            "strong": runtime.strong_metadata,
            "heads": {
                name: {
                    "depth": spec.depth,
                    "checkpoint": spec.checkpoint,
                    "checkpoint_sha256": spec.checkpoint_sha256,
                }
                for name, spec in heads.items()
            },
            "allocator": allocator,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "summary": str(output / "multidepth_summary.csv"),
            "per_sample": str(output / "per_sample_multidepth.csv"),
        },
    )
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
