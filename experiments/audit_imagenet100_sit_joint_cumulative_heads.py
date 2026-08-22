#!/usr/bin/env python3
"""Run the held-out depth-convergence audit on one joint-head checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from experiments.audit_imagenet100_sit_depth_convergence import (
        DEFAULT_TIMES,
        analyze_time,
        atomic_json,
        make_figure,
        make_validation_bank,
        parse_times,
        write_csv,
    )
    from experiments.imagenet100_sit_joint_cumulative_heads import (
        create_joint_cumulative_parts,
        source_velocity_from_final_features,
    )
    from experiments.imagenet100_sit_multiscale_models import load_sit_field_model
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        linear_flow_state_target,
        load_official_sit_module,
        sha256_file,
    )
    from experiments.train_imagenet100_sit_joint_cumulative_heads import PROTOCOL
except ModuleNotFoundError:
    from audit_imagenet100_sit_depth_convergence import (
        DEFAULT_TIMES,
        analyze_time,
        atomic_json,
        make_figure,
        make_validation_bank,
        parse_times,
        write_csv,
    )
    from imagenet100_sit_joint_cumulative_heads import (
        create_joint_cumulative_parts,
        source_velocity_from_final_features,
    )
    from imagenet100_sit_multiscale_models import load_sit_field_model
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        linear_flow_state_target,
        load_official_sit_module,
        sha256_file,
    )
    from train_imagenet100_sit_joint_cumulative_heads import PROTOCOL


@torch.inference_mode()
def evaluate_time(
    *,
    prefix,
    readouts,
    source,
    depths: tuple[int, ...],
    clean: torch.Tensor,
    noise: torch.Tensor,
    labels: torch.Tensor,
    time_scalar: float,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {
        **{f"d{depth}": [] for depth in depths},
        "strong": [],
    }
    targets: list[torch.Tensor] = []
    autocast_enabled = precision != "fp32"
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for start in range(0, len(clean), batch_size):
        stop = min(start + batch_size, len(clean))
        clean_batch = clean[start:stop].to(device, non_blocking=True)
        noise_batch = noise[start:stop].to(device, non_blocking=True)
        label_batch = labels[start:stop].to(device, non_blocking=True)
        time_value = torch.full(
            (stop - start,), float(time_scalar), device=device, dtype=torch.float32
        )
        state, target = linear_flow_state_target(clean_batch, noise_batch, time_value)
        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled
        ):
            features, conditioning = prefix(state, time_value, label_batch)
            cumulative, _ = readouts(features, conditioning)
            strong = source_velocity_from_final_features(
                source,
                features[-1],
                conditioning,
                latent_channels=LATENT_SHAPE[0],
            )
        for depth, value in zip(depths, cumulative, strict=True):
            outputs[f"d{depth}"].append(value.float().cpu())
        outputs["strong"].append(strong.float().cpu())
        targets.append(target.float().cpu())
    return (
        {name: torch.cat(values, dim=0) for name, values in outputs.items()},
        torch.cat(targets, dim=0),
    )


def main(args: argparse.Namespace) -> None:
    if args.num_samples < 8 or args.num_samples % 2:
        raise ValueError("--num-samples must be even and at least 8")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("unsupported joint cumulative checkpoint")
    checkpoint_step = int(checkpoint["step"])
    config = checkpoint["config"]
    depths = tuple(int(depth) for depth in config["depths"])
    if depths != (4, 6, 8, 10, 12):
        raise ValueError("the current audit expects depths 4,6,8,10,12")
    source_path = Path(config["source_checkpoint"]).expanduser().resolve()
    if sha256_file(source_path) != config["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint digest changed")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this audit requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.time()
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(), verify_source=True
    )
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("joint checkpoint official SiT metadata changed")
    source, semantics, source_metadata_live = load_sit_field_model(
        checkpoint_path=source_path,
        weights=str(config["source_state_key"]),
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("joint audit requires a native velocity source")
    prefix, readouts = create_joint_cumulative_parts(
        sit_module, source, depths=depths, latent_channels=LATENT_SHAPE[0]
    )
    state_key = "readouts_ema" if args.weights == "ema" else "readouts"
    readouts.load_state_dict(checkpoint[state_key], strict=True)
    prefix.to(device).eval()
    readouts.to(device).eval()
    del checkpoint

    clean, noise, labels = make_validation_bank(
        cache_dir=args.cache_dir.expanduser().resolve(),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )
    output_root = args.output_root
    if output_root is None:
        output_root = checkpoint_path.parents[1] / f"convergence_audit_step_{checkpoint_step:08d}"
    output_root = output_root.expanduser().resolve()
    fit_count = args.num_samples // 2
    output_rows: list[dict] = []
    transition_rows: list[dict] = []
    extrapolation_rows: list[dict] = []
    time_summaries: list[dict] = []
    torch.cuda.reset_peak_memory_stats(device)
    for index, time_scalar in enumerate(args.times, start=1):
        outputs, target = evaluate_time(
            prefix=prefix,
            readouts=readouts,
            source=source,
            depths=depths,
            clean=clean,
            noise=noise,
            labels=labels,
            time_scalar=time_scalar,
            batch_size=args.batch_size,
            device=device,
            precision=args.precision,
        )
        rows_a, rows_b, rows_c, summary = analyze_time(
            time_scalar=time_scalar,
            outputs=outputs,
            target=target,
            fit_count=fit_count,
        )
        output_rows.extend(rows_a)
        transition_rows.extend(rows_b)
        extrapolation_rows.extend(rows_c)
        time_summaries.append(summary)
        print(
            f"[{index}/{len(args.times)}] t={time_scalar:.3f} "
            f"monotonic={summary['strict_internal_mse_monotonic_fraction']:.3f}",
            flush=True,
        )
        del outputs, target

    transition_success = [
        abs(row["cosine_mean"]) > 0.5
        and abs(row["fitted_lambda"]) < 1.0
        and row["relative_transition_residual"] < 0.75
        and row["prediction_over_zero"] < 1.0
        for row in transition_rows
    ]
    valid_limits = [
        row
        for row in extrapolation_rows
        if row["kind"] == "geometric_limit" and row["valid"]
    ]
    aggregate_monotonic_fraction = sum(
        row["aggregate_internal_mse_strictly_decreasing"] for row in time_summaries
    ) / len(time_summaries)
    transition_success_fraction = sum(transition_success) / len(transition_success)
    limit_improvement_fraction = (
        sum(row["candidate_over_base"] < 1.0 for row in valid_limits) / len(valid_limits)
        if valid_limits
        else 0.0
    )
    supported = (
        aggregate_monotonic_fraction >= 0.75
        and transition_success_fraction >= 2.0 / 3.0
        and limit_improvement_fraction > 0.5
    )
    summary = {
        "format": "eqvae_imagenet100_sit_joint_depth_convergence_audit_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "training": config,
        "source": source_metadata_live,
        "protocol": {
            "times": list(args.times),
            "num_samples_per_time": args.num_samples,
            "fit_samples": fit_count,
            "eval_samples": args.num_samples - fit_count,
            "seed": args.seed,
            "precision": args.precision,
            "weights": state_key,
        },
        "time_summaries": time_summaries,
        "decision": {
            "aggregate_mse_monotonic_time_fraction": aggregate_monotonic_fraction,
            "transition_test_success_fraction": transition_success_fraction,
            "valid_geometric_limits": len(valid_limits),
            "geometric_limit_improvement_fraction": limit_improvement_fraction,
            "emergent_convergence_sequence_supported": supported,
        },
        "runtime": {
            "seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    write_csv(output_root / "output_mse.csv", output_rows)
    write_csv(output_root / "transition_metrics.csv", transition_rows)
    write_csv(output_root / "extrapolation_metrics.csv", extrapolation_rows)
    atomic_json(summary, output_root / "summary.json")
    make_figure(
        output_rows,
        transition_rows,
        extrapolation_rows,
        output_root / "depth_convergence_audit.png",
    )
    print(json.dumps(summary["decision"], indent=2), flush=True)
    print(f"wrote {output_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--times", type=parse_times, default=DEFAULT_TIMES)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
