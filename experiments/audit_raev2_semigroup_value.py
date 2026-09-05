#!/usr/bin/env python3
"""Held-out diagnostics for the RAEv2 normalized OU-HJB value model.

This audit measures whether a trained value model satisfies its plug-in Bellman
equation on an independent weak-flow switch bank.  It does not treat Bellman
self-consistency as evidence of improved endpoint sample quality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_semigroup_value import (  # noqa: E402
    RAEv2NormalizedOUValue,
    clean_gap_to_ou_score_gap,
    noise_time_from_ou_time,
    ou_potential_gradient_to_clean_correction,
    ou_to_state,
    rae_ou_coefficients,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_pfr_retiming import load_config  # noqa: E402
from experiments.train_raev2_semigroup_value import (  # noqa: E402
    PROTOCOL,
    build_hjb_target,
    load_or_generate_bank,
    source_autocast,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


AUDIT_PROTOCOL = "raev2_normalized_ou_hjb_value_heldout_audit_v1"


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(not math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("times must be a non-empty finite list")
    return parsed


def append_metric(
    storage: dict[str, list[torch.Tensor]], name: str, value: torch.Tensor
) -> None:
    storage[name].append(value.detach().float().flatten().cpu())


def cosine_per_sample(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return (left_flat * right_flat).sum(1) / denominator.clamp_min(1e-12)


def summarize_metric(values: list[torch.Tensor]) -> dict[str, float]:
    merged = torch.cat(values)
    return {
        "mean": float(merged.mean().item()),
        "std": float(merged.std(unbiased=False).item()),
        "rms": float(merged.square().mean().sqrt().item()),
        "q05": float(torch.quantile(merged, 0.05).item()),
        "q50": float(torch.quantile(merged, 0.50).item()),
        "q95": float(torch.quantile(merged, 0.95).item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--bank-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--times",
        type=parse_float_list,
        default=parse_float_list("0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for name in ("samples", "bank_batch_size", "eval_batch_size", "particles"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")

    checkpoint_path = args.value_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("value checkpoint uses an incompatible protocol")
    request = payload["request"]
    training_bank_seed = int(request["bank_seed"])
    if args.seed == training_bank_seed:
        raise ValueError("held-out bank seed must differ from the training bank seed")

    install_raev2_decoder_config_compat()
    config_path = Path(request["config"]).expanduser().resolve()
    source_checkpoint = Path(request["source_checkpoint"]).expanduser().resolve()
    config = load_config(config_path)
    num_classes = int(config.misc.num_classes)
    if args.samples < num_classes:
        raise ValueError(
            f"samples={args.samples} cannot cover all {num_classes} conditioned classes"
        )
    switch_time = float(request["switch_time"])
    maximum_noise_time = float(request["maximum_noise_time"])
    times = sorted(set(float(item) for item in args.times))
    if any(item < switch_time or item > maximum_noise_time for item in times):
        raise ValueError("audit times must lie in the trained semigroup interval")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    precision = str(request["precision"])
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = precision != "fp32"

    source = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    source_payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    source.load_state_dict(source_payload[str(request["source_state_key"])], strict=True)
    del source_payload

    value = RAEv2NormalizedOUValue(
        int(config.misc.latent_size[0]),
        num_classes,
        width=int(request["width"]),
        depth=int(request["depth"]),
        switch_time=switch_time,
    ).to(device).eval().requires_grad_(False)
    value.load_state_dict(payload["value_ema"], strict=True)
    del payload

    bank = load_or_generate_bank(
        output_dir / "heldout_weak_switch_bank.pt",
        source,
        config=config,
        samples=args.samples,
        batch_size=args.bank_batch_size,
        seed=args.seed,
        switch_time=switch_time,
        precision=precision,
        device=device,
    )
    if int(bank["classes_covered"]) != num_classes:
        raise RuntimeError("held-out switch bank does not cover every class")
    switch_states = bank["ou_states"]
    labels = bank["labels"]
    if not isinstance(switch_states, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("held-out bank tensors are missing")

    beta = float(request["beta"])
    ambient_dimension = int(math.prod(config.misc.latent_size))
    switch_tensor = torch.tensor([switch_time], device=device)
    maximum_tensor = torch.tensor([maximum_noise_time], device=device)
    switch_semigroup = rae_ou_coefficients(switch_tensor)[-1][0]
    maximum_semigroup = rae_ou_coefficients(maximum_tensor)[-1][0]
    nominal_bellman_step = (maximum_semigroup - switch_semigroup) / int(
        request["levels"]
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    rows: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)

    for noise_time_scalar in times:
        metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        time_tensor = torch.tensor([noise_time_scalar], device=device)
        semigroup_time = rae_ou_coefficients(time_tensor)[-1][0]
        retention = torch.exp(-(semigroup_time - switch_semigroup))
        bellman_step = torch.minimum(
            nominal_bellman_step, semigroup_time - switch_semigroup
        )

        for start in range(0, args.samples, args.eval_batch_size):
            stop = min(start + args.eval_batch_size, args.samples)
            batch_switch = switch_states[start:stop].to(device=device, dtype=torch.float32)
            batch_labels = labels[start:stop].to(device=device)
            batch_times = torch.full(
                (stop - start,), noise_time_scalar, device=device, dtype=torch.float32
            )
            noise = torch.randn(
                batch_switch.shape,
                generator=generator,
                device=device,
                dtype=batch_switch.dtype,
            )
            ou_state = (
                retention * batch_switch
                + torch.sqrt(1.0 - retention.square()) * noise
            )
            native_state = ou_to_state(ou_state, batch_times)

            with torch.no_grad(), source_autocast(precision):
                full_clean, base_clean = source(
                    native_state,
                    batch_times,
                    context=batch_labels,
                    attn_mask=None,
                )
            full_clean = full_clean.float()
            base_clean = base_clean.float()
            clean_gap = full_clean - base_clean
            score_gap = clean_gap_to_ou_score_gap(
                clean_gap, noise_time=batch_times
            )

            query = ou_state.detach().requires_grad_(True)
            value_prediction = value(query, batch_times, batch_labels)
            value_gradient = torch.autograd.grad(value_prediction.sum(), query)[0]
            score_correction = float(ambient_dimension) * value_gradient
            clean_correction = ou_potential_gradient_to_clean_correction(
                score_correction, noise_time=batch_times
            )
            ig_increment = (beta - 1.0) * clean_gap

            permuted_labels = batch_labels.roll(1)
            if len(batch_labels) == 1:
                permuted_labels = (batch_labels + 1).remainder(num_classes)
            permuted_query = ou_state.detach().requires_grad_(True)
            permuted_value = value(permuted_query, batch_times, permuted_labels)
            permuted_gradient = torch.autograd.grad(
                permuted_value.sum(), permuted_query
            )[0]

            zero_value = value(
                torch.zeros_like(ou_state), batch_times, batch_labels
            ).detach()
            append_metric(metrics, "value", value_prediction)
            append_metric(metrics, "state_dependent_value", value_prediction - zero_value)
            append_metric(
                metrics,
                "score_gap_rms",
                score_gap.flatten(1).square().mean(1).sqrt(),
            )
            append_metric(
                metrics,
                "score_correction_rms",
                score_correction.flatten(1).square().mean(1).sqrt(),
            )
            append_metric(
                metrics,
                "clean_correction_rms",
                clean_correction.flatten(1).square().mean(1).sqrt(),
            )
            append_metric(
                metrics,
                "ig_increment_rms",
                ig_increment.flatten(1).square().mean(1).sqrt(),
            )
            append_metric(
                metrics,
                "correction_gap_cosine",
                cosine_per_sample(clean_correction, clean_gap),
            )
            append_metric(
                metrics,
                "correct_vs_permuted_label_gradient_cosine",
                cosine_per_sample(value_gradient, permuted_gradient),
            )
            append_metric(
                metrics,
                "label_gradient_relative_change",
                (value_gradient - permuted_gradient).flatten(1).norm(dim=1)
                / value_gradient.flatten(1).norm(dim=1).clamp_min(1e-12),
            )

            if noise_time_scalar > switch_time and bellman_step > 0.0:
                old_semigroup = semigroup_time - bellman_step
                old_time_scalar = float(
                    noise_time_from_ou_time(old_semigroup[None])[0].item()
                )
                old_times = torch.full_like(batch_times, old_time_scalar)
                semigroup_steps = torch.full_like(batch_times, float(bellman_step.item()))
                target_values, _ = build_hjb_target(
                    source,
                    value,
                    ou_state=ou_state,
                    old_time=old_times,
                    labels=batch_labels,
                    semigroup_step=semigroup_steps,
                    beta=beta,
                    particles=args.particles,
                    precision=precision,
                    generator=generator,
                )
                old_values = value(ou_state, old_times, batch_labels).detach()
                append_metric(
                    metrics,
                    "bellman_residual",
                    value_prediction.detach() - target_values,
                )
                append_metric(
                    metrics,
                    "bellman_target_increment",
                    target_values - old_values,
                )

        summaries = {name: summarize_metric(parts) for name, parts in metrics.items()}
        score_correction_rms = summaries["score_correction_rms"]["rms"]
        clean_correction_rms = summaries["clean_correction_rms"]["rms"]
        ig_increment_rms = summaries["ig_increment_rms"]["rms"]
        row: dict[str, object] = {
            "noise_time": noise_time_scalar,
            "ou_semigroup_time": float(semigroup_time.item()),
            "ou_retention_from_switch": float(retention.item()),
            "score_correction_to_gap_ratio": score_correction_rms
            / max(summaries["score_gap_rms"]["rms"], 1e-12),
            "clean_correction_to_ig_increment_ratio": clean_correction_rms
            / max(ig_increment_rms, 1e-12),
            "metrics": summaries,
        }
        if "bellman_residual" in summaries:
            row["bellman_residual_to_target_increment_ratio"] = summaries[
                "bellman_residual"
            ]["rms"] / max(summaries["bellman_target_increment"]["rms"], 1e-12)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "protocol": AUDIT_PROTOCOL,
        "interpretation": (
            "Held-out self-consistency audit of the plug-in OU-HJB equation; "
            "not a ground-truth density or endpoint-quality certificate."
        ),
        "value_checkpoint": str(checkpoint_path),
        "value_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "training_bank_seed": training_bank_seed,
        "heldout_bank_seed": args.seed,
        "samples": args.samples,
        "classes_covered": int(bank["classes_covered"]),
        "particles": args.particles,
        "times": times,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "rows": rows,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
