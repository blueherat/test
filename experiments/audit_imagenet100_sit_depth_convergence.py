#!/usr/bin/env python3
"""Audit whether frozen SiT internal heads form an emergent convergence sequence.

The five auxiliary heads were trained independently on a frozen v800 backbone.
This script therefore treats convergence as a hypothesis to test, not as an
architectural guarantee.  All heads are evaluated in one shared-backbone pass
on paired validation states.  Scalar transition coefficients are fitted on one
half of the samples and evaluated on the other half.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

try:
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NpyMomentsDataset,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
        sha256_file,
    )
except ModuleNotFoundError:
    from imagenet100_sit_multiscale_models import (
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NpyMomentsDataset,
        linear_flow_state_target,
        load_official_sit_module,
        sample_sdvae_posterior,
        sha256_file,
    )


DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_STRONG = DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "depth_convergence_audit_v1"
DEPTHS = (4, 6, 8, 10, 12)
DEFAULT_TIMES = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)


def internal_head_checkpoint(depth: int) -> Path:
    if depth == 8:
        return (
            DATA_ROOT
            / "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
            "checkpoints/step_00050000.pt"
        )
    return (
        DATA_ROOT
        / f"multiscale_guidance_study_v1/runs/depth{depth}_v/"
        "checkpoints/step_00050000.pt"
    )


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_times(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(not 0.0 < item < 1.0 for item in result):
        raise argparse.ArgumentTypeError("times must be a non-empty CSV inside (0,1)")
    if tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    return result


def flatten(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1)


def per_sample_mse(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (flatten(value) - flatten(target)).square().mean(dim=1)


def per_sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = flatten(left)
    right_flat = flatten(right)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return numerator / denominator.clamp_min(1e-12)


def fit_scalar_transition(previous: torch.Tensor, following: torch.Tensor) -> float:
    """Least-squares scalar in ``following ~= scalar * previous``."""

    previous_flat = flatten(previous).double()
    following_flat = flatten(following).double()
    denominator = previous_flat.square().sum()
    if float(denominator) <= 1e-20:
        return math.nan
    return float((previous_flat * following_flat).sum() / denominator)


def fit_target_step(base: torch.Tensor, direction: torch.Tensor, target: torch.Tensor) -> float:
    """Fit ``base + gamma * direction`` to a target on a separate fit split."""

    direction_flat = flatten(direction).double()
    residual_flat = (flatten(target) - flatten(base)).double()
    denominator = direction_flat.square().sum()
    if float(denominator) <= 1e-20:
        return math.nan
    return float((direction_flat * residual_flat).sum() / denominator)


def transition_metrics(
    previous: torch.Tensor,
    following: torch.Tensor,
    fitted_scalar: float,
) -> dict[str, float]:
    previous_flat = flatten(previous).double()
    following_flat = flatten(following).double()
    cosine = per_sample_cosine(previous, following).double()
    previous_energy = previous_flat.square().sum(dim=1)
    sample_scalar = (previous_flat * following_flat).sum(dim=1) / previous_energy.clamp_min(
        1e-20
    )
    residual = following_flat - float(fitted_scalar) * previous_flat
    residual_ratio = math.sqrt(
        float(residual.square().sum())
        / max(float(following_flat.square().sum()), 1e-20)
    )
    prediction_mse = float(residual.square().mean())
    zero_increment_mse = float(following_flat.square().mean())
    repeat_increment_mse = float((following_flat - previous_flat).square().mean())
    return {
        "fitted_lambda": float(fitted_scalar),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(cosine.median()),
        "cosine_positive_fraction": float((cosine > 0).double().mean()),
        "sample_lambda_mean": float(sample_scalar.mean()),
        "sample_lambda_std": float(sample_scalar.std(unbiased=True)),
        "sample_lambda_in_0_1_fraction": float(
            ((sample_scalar > 0) & (sample_scalar < 1)).double().mean()
        ),
        "sample_lambda_in_minus1_1_fraction": float(
            ((sample_scalar > -1) & (sample_scalar < 1)).double().mean()
        ),
        "relative_transition_residual": residual_ratio,
        "transition_energy_explained": 1.0 - residual_ratio**2,
        "next_increment_prediction_mse": prediction_mse,
        "zero_increment_prediction_mse": zero_increment_mse,
        "repeat_increment_prediction_mse": repeat_increment_mse,
        "prediction_over_zero": prediction_mse / max(zero_increment_mse, 1e-20),
        "prediction_over_repeat": prediction_mse / max(repeat_increment_mse, 1e-20),
    }


def geometric_limit(
    current: torch.Tensor,
    latest_increment: torch.Tensor,
    fitted_scalar: float,
) -> torch.Tensor | None:
    """Extrapolate a geometric sequence, rejecting unstable coefficients."""

    if not math.isfinite(fitted_scalar) or abs(fitted_scalar) >= 0.95:
        return None
    return current + (fitted_scalar / (1.0 - fitted_scalar)) * latest_increment


def make_validation_bank(
    *, cache_dir: Path, num_samples: int, batch_size: int, workers: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = NpyMomentsDataset(cache_dir, "validation")
    if num_samples > len(dataset):
        raise ValueError(f"requested {num_samples} samples from {len(dataset)} validation rows")
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": False,
        "drop_last": False,
        "persistent_workers": int(workers) > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    clean_rows: list[torch.Tensor] = []
    noise_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    seen = 0
    for moments, labels in loader:
        if seen >= num_samples:
            break
        keep = min(len(moments), num_samples - seen)
        moments = moments[:keep].float()
        labels = labels[:keep].long()
        posterior_noise = torch.randn(
            (keep, *LATENT_SHAPE), generator=generator, dtype=torch.float32
        )
        clean = sample_sdvae_posterior(moments, posterior_noise)
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        clean_rows.append(clean.contiguous())
        noise_rows.append(noise.contiguous())
        label_rows.append(labels.contiguous())
        seen += keep
    if seen != num_samples:
        raise RuntimeError(f"validation loader produced {seen}, expected {num_samples}")
    return (
        torch.cat(clean_rows, dim=0),
        torch.cat(noise_rows, dim=0),
        torch.cat(label_rows, dim=0),
    )


@torch.inference_mode()
def evaluate_time(
    *,
    model,
    heads: dict,
    clean: torch.Tensor,
    noise: torch.Tensor,
    labels: torch.Tensor,
    time_scalar: float,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {
        **{f"d{depth}": [] for depth in DEPTHS},
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
            full, trained, _ = evaluate_source_with_heads(
                model, state, time_value, label_batch, heads=heads
            )
        outputs["strong"].append(full.float().cpu())
        for depth in DEPTHS:
            outputs[f"d{depth}"].append(trained[f"d{depth}"].float().cpu())
        targets.append(target.float().cpu())
    return (
        {name: torch.cat(values, dim=0) for name, values in outputs.items()},
        torch.cat(targets, dim=0),
    )


def analyze_time(
    *,
    time_scalar: float,
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    fit_count: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    names = [f"d{depth}" for depth in DEPTHS]
    eval_slice = slice(fit_count, len(target))
    fit_slice = slice(0, fit_count)
    output_rows: list[dict] = []
    errors: dict[str, torch.Tensor] = {}
    for name in (*names, "strong"):
        values = per_sample_mse(outputs[name][eval_slice], target[eval_slice]).double()
        errors[name] = values
        output_rows.append(
            {
                "time": time_scalar,
                "output": name,
                "samples": len(values),
                "velocity_mse": float(values.mean()),
                "velocity_mse_se": float(values.std(unbiased=True) / math.sqrt(len(values))),
                "output_rms": float(flatten(outputs[name][eval_slice]).square().mean().sqrt()),
            }
        )

    monotonic = torch.ones(len(target) - fit_count, dtype=torch.bool)
    adjacent_improvement: dict[str, float] = {}
    for left, right in zip(names[:-1], names[1:], strict=True):
        improved = errors[right] < errors[left]
        monotonic &= improved
        adjacent_improvement[f"{left}_to_{right}"] = float(improved.double().mean())

    increments = {
        f"d{left}_to_d{right}": outputs[f"d{right}"] - outputs[f"d{left}"]
        for left, right in zip(DEPTHS[:-1], DEPTHS[1:], strict=True)
    }
    increment_names = list(increments)
    transition_rows: list[dict] = []
    extrapolation_rows: list[dict] = []
    for previous_name, following_name in zip(
        increment_names[:-1], increment_names[1:], strict=True
    ):
        fitted_lambda = fit_scalar_transition(
            increments[previous_name][fit_slice], increments[following_name][fit_slice]
        )
        metrics = transition_metrics(
            increments[previous_name][eval_slice],
            increments[following_name][eval_slice],
            fitted_lambda,
        )
        following_depth = int(following_name.rsplit("d", 1)[-1])
        transition_rows.append(
            {
                "time": time_scalar,
                "previous_increment": previous_name,
                "following_increment": following_name,
                "fit_samples": fit_count,
                "eval_samples": len(target) - fit_count,
                **metrics,
            }
        )
        current = outputs[f"d{following_depth}"][eval_slice]
        latest_increment = increments[following_name][eval_slice]
        limit = geometric_limit(current, latest_increment, fitted_lambda)
        if limit is None:
            extrapolation_rows.append(
                {
                    "time": time_scalar,
                    "kind": "geometric_limit",
                    "base": f"d{following_depth}",
                    "direction": following_name,
                    "fit_value": fitted_lambda,
                    "valid": False,
                    "base_mse": float(errors[f"d{following_depth}"].mean()),
                    "candidate_mse": math.nan,
                    "candidate_over_base": math.nan,
                }
            )
        else:
            candidate_mse = float(per_sample_mse(limit, target[eval_slice]).mean())
            base_mse = float(errors[f"d{following_depth}"].mean())
            extrapolation_rows.append(
                {
                    "time": time_scalar,
                    "kind": "geometric_limit",
                    "base": f"d{following_depth}",
                    "direction": following_name,
                    "fit_value": fitted_lambda,
                    "valid": True,
                    "base_mse": base_mse,
                    "candidate_mse": candidate_mse,
                    "candidate_over_base": candidate_mse / max(base_mse, 1e-20),
                }
            )

    for left, right in zip(DEPTHS[:-1], DEPTHS[1:], strict=True):
        base_name = f"d{right}"
        direction_name = f"d{left}_to_d{right}"
        direction = increments[direction_name]
        gamma = fit_target_step(
            outputs[base_name][fit_slice], direction[fit_slice], target[fit_slice]
        )
        candidate = outputs[base_name][eval_slice] + gamma * direction[eval_slice]
        candidate_mse = float(per_sample_mse(candidate, target[eval_slice]).mean())
        base_mse = float(errors[base_name].mean())
        extrapolation_rows.append(
            {
                "time": time_scalar,
                "kind": "target_fitted_heldout",
                "base": base_name,
                "direction": direction_name,
                "fit_value": gamma,
                "valid": math.isfinite(gamma),
                "base_mse": base_mse,
                "candidate_mse": candidate_mse,
                "candidate_over_base": candidate_mse / max(base_mse, 1e-20),
            }
        )

    d12_strong_gap = outputs["strong"] - outputs["d12"]
    readout_cosine = per_sample_cosine(
        d12_strong_gap[eval_slice], target[eval_slice] - outputs["d12"][eval_slice]
    )
    summary = {
        "time": time_scalar,
        "strict_internal_mse_monotonic_fraction": float(monotonic.double().mean()),
        "aggregate_internal_mse_strictly_decreasing": all(
            float(errors[right].mean()) < float(errors[left].mean())
            for left, right in zip(names[:-1], names[1:], strict=True)
        ),
        "adjacent_improvement_fractions": adjacent_improvement,
        "d12_to_strong_mse_ratio": float(errors["strong"].mean())
        / max(float(errors["d12"].mean()), 1e-20),
        "d12_to_strong_gap_rms": float(flatten(d12_strong_gap[eval_slice]).square().mean().sqrt()),
        "d12_to_strong_residual_cosine_mean": float(readout_cosine.mean()),
    }
    return output_rows, transition_rows, extrapolation_rows, summary


def make_figure(
    output_rows: list[dict], transition_rows: list[dict], extrapolation_rows: list[dict], path: Path
) -> None:
    times = sorted({float(row["time"]) for row in output_rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for name in [*(f"d{depth}" for depth in DEPTHS), "strong"]:
        rows = [row for row in output_rows if row["output"] == name]
        axes[0].plot(
            [row["time"] for row in rows],
            [row["velocity_mse"] for row in rows],
            marker="o",
            label=name,
        )
    axes[0].set_title("Held-out velocity MSE")
    axes[0].set_xlabel("t (noise -> data)")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].legend(ncol=2, fontsize=8)

    pairs = sorted({row["following_increment"] for row in transition_rows})
    for pair in pairs:
        rows = [row for row in transition_rows if row["following_increment"] == pair]
        axes[1].plot(
            [row["time"] for row in rows],
            [row["cosine_mean"] for row in rows],
            marker="o",
            label=pair,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_title("Consecutive increment cosine")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("cosine")
    axes[1].legend(fontsize=8)

    geometric = [
        row for row in extrapolation_rows if row["kind"] == "geometric_limit" and row["valid"]
    ]
    bases = sorted({row["base"] for row in geometric})
    if geometric:
        for base in bases:
            rows = [row for row in geometric if row["base"] == base]
            axes[2].plot(
                [row["time"] for row in rows],
                [row["candidate_over_base"] for row in rows],
                marker="o",
                label=base,
            )
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_title("Geometric limit / base MSE")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("ratio (<1 improves)")
    if geometric:
        axes[2].legend(fontsize=8)
    axes[2].set_xticks(times[:: max(1, len(times) // 4)])

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    if args.num_samples < 8 or args.num_samples % 2:
        raise ValueError("--num-samples must be even and at least 8")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this audit requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(), verify_source=True
    )
    strong, strong_semantics, strong_metadata = load_sit_field_model(
        checkpoint_path=args.strong_checkpoint,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if strong_semantics.prediction_target != "velocity":
        raise ValueError("the convergence audit requires a native velocity source")

    heads = {}
    head_metadata = {}
    for depth in DEPTHS:
        path = internal_head_checkpoint(depth).expanduser().resolve()
        spec = load_internal_head_for_source(
            checkpoint_path=path,
            name=f"d{depth}",
            head_weights="ema",
            model=strong,
            sit_module=sit_module,
            source_checkpoint_path=args.strong_checkpoint,
            source_metadata=source_metadata,
            device=device,
        )
        if spec.depth != depth or spec.prediction_target != "velocity":
            raise ValueError(f"unexpected head semantics for depth {depth}")
        heads[spec.name] = spec
        head_metadata[spec.name] = {
            "depth": spec.depth,
            "prediction_target": spec.prediction_target,
            "checkpoint": spec.checkpoint,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "source_checkpoint_sha256": spec.source_checkpoint_sha256,
        }

    clean, noise, labels = make_validation_bank(
        cache_dir=args.cache_dir.expanduser().resolve(),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )
    fit_count = args.num_samples // 2
    output_rows: list[dict] = []
    transition_rows: list[dict] = []
    extrapolation_rows: list[dict] = []
    time_summaries: list[dict] = []
    torch.cuda.reset_peak_memory_stats(device)
    for index, time_scalar in enumerate(args.times, start=1):
        outputs, target = evaluate_time(
            model=strong,
            heads=heads,
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
    limit_improvement_fraction = (
        sum(row["candidate_over_base"] < 1.0 for row in valid_limits) / len(valid_limits)
        if valid_limits
        else 0.0
    )
    aggregate_monotonic_fraction = sum(
        row["aggregate_internal_mse_strictly_decreasing"] for row in time_summaries
    ) / len(time_summaries)
    transition_success_fraction = sum(transition_success) / len(transition_success)
    supported = (
        aggregate_monotonic_fraction >= 0.75
        and transition_success_fraction >= 2.0 / 3.0
        and limit_improvement_fraction > 0.5
    )
    summary = {
        "format": "eqvae_imagenet100_sit_depth_convergence_audit_v1",
        "scope": (
            "post-hoc audit of independently trained frozen-backbone heads; "
            "not an explicitly constrained iterative architecture"
        ),
        "protocol": {
            "times": list(args.times),
            "num_samples_per_time": args.num_samples,
            "fit_samples": fit_count,
            "eval_samples": args.num_samples - fit_count,
            "precision": args.precision,
            "seed": args.seed,
            "paired_clean_noise_across_times": True,
            "head_weights": "ema",
        },
        "strong": {
            **strong_metadata,
            "checkpoint_sha256_live": sha256_file(args.strong_checkpoint),
        },
        "heads": head_metadata,
        "time_summaries": time_summaries,
        "decision": {
            "aggregate_mse_monotonic_time_fraction": aggregate_monotonic_fraction,
            "transition_test_success_fraction": transition_success_fraction,
            "valid_geometric_limits": len(valid_limits),
            "geometric_limit_improvement_fraction": limit_improvement_fraction,
            "emergent_convergence_sequence_supported": supported,
            "thresholds": {
                "aggregate_mse_monotonic_time_fraction": ">=0.75",
                "transition_test_success_fraction": ">=2/3",
                "geometric_limit_improvement_fraction": ">0.5",
            },
        },
        "runtime": {
            "seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--times", type=parse_times, default=DEFAULT_TIMES)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
