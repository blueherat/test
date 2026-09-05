#!/usr/bin/env python3
"""Train a normalized OU-HJB value on a frozen RAEv2 full/depth4 pair.

The full and base heads remain frozen.  A small raw-latent scalar network learns
the per-dimension free energy ``phi=delta/D`` implied by the semigroup-consistent
power tilt.  Training uses a switch-plane sample bank from the weak/base flow
and exact OU corruption; it does not use ImageNet targets or endpoint labels
beyond the model's ordinary class conditioning.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_fkc_weight_degeneracy import (  # noqa: E402
    finite_audit_grid,
)
from experiments.raev2_pfr_retiming import clean_to_velocity  # noqa: E402
from experiments.raev2_semigroup_value import (  # noqa: E402
    RAEv2NormalizedOUValue,
    clean_gap_to_ou_score_gap,
    clean_prediction_to_ou_score,
    noise_time_from_ou_time,
    normalized_hjb_running_cost,
    normalized_hjb_target,
    ou_to_state,
    rae_ou_coefficients,
    state_to_ou,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_pfr_retiming import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    shifted_time_grid,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_normalized_ou_hjb_value_v1"


def balanced_class_labels(samples: int, num_classes: int) -> torch.Tensor:
    """Return deterministic labels with complete, near-uniform class coverage."""

    if samples < num_classes or num_classes <= 0:
        raise ValueError("switch bank must contain at least one sample per class")
    return torch.arange(samples, dtype=torch.long).remainder(num_classes)


def source_autocast(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def update_ema(target: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must lie in [0,1)")
    with torch.no_grad():
        target_parameters = dict(target.named_parameters())
        for name, parameter in source.named_parameters():
            target_parameters[name].lerp_(parameter, 1.0 - decay)
        target_buffers = dict(target.named_buffers())
        for name, buffer in source.named_buffers():
            target_buffers[name].copy_(buffer)


def generate_weak_switch_bank(
    model: torch.nn.Module,
    *,
    config,
    samples: int,
    batch_size: int,
    seed: int,
    switch_time: float,
    precision: str,
    device: torch.device,
) -> dict[str, torch.Tensor | int | float]:
    """Generate the weak/base marginal at the intervention plane."""

    if samples <= 0 or batch_size <= 0:
        raise ValueError("bank sample counts must be positive")
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(config.misc.latent_size))
        / config.misc.time_dist_shift_base
    )
    native = shifted_time_grid(config.sampler.num_steps, shift, device)
    grid = finite_audit_grid(native, switch_time=switch_time)
    generator = torch.Generator(device=device).manual_seed(seed)
    num_classes = int(config.misc.num_classes)
    labels = balanced_class_labels(samples, num_classes).to(device)
    states: list[torch.Tensor] = []
    t_floor = float(config.transport.t_eps)
    with torch.inference_mode(), source_autocast(precision):
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            state = torch.randn(
                stop - start,
                *config.misc.latent_size,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            batch_labels = labels[start:stop]
            for index in range(len(grid) - 1):
                current = float(grid[index].item())
                following = float(grid[index + 1].item())
                times = torch.full(
                    (stop - start,), current, device=device, dtype=torch.float32
                )
                _, base_clean = model(
                    state,
                    times,
                    context=batch_labels,
                    attn_mask=None,
                )
                drift = clean_to_velocity(
                    base_clean,
                    state,
                    times,
                    denominator_floor=t_floor,
                )
                state = state - (current - following) * drift.float()
            switch_times = torch.full(
                (stop - start,), switch_time, device=device, dtype=torch.float32
            )
            states.append(state_to_ou(state, switch_times).half().cpu())
    return {
        "ou_states": torch.cat(states),
        "labels": labels.cpu(),
        "classes_covered": int(labels.unique().numel()),
        "num_classes": num_classes,
        "seed": int(seed),
        "switch_time": float(switch_time),
    }


def load_or_generate_bank(
    path: Path,
    model: torch.nn.Module,
    *,
    config,
    samples: int,
    batch_size: int,
    seed: int,
    switch_time: float,
    precision: str,
    device: torch.device,
) -> dict[str, torch.Tensor | int | float]:
    if path.is_file():
        bank = torch.load(path, map_location="cpu", weights_only=False)
        if (
            bank.get("protocol") != PROTOCOL
            or int(bank.get("samples", -1)) != samples
            or int(bank.get("seed", -1)) != seed
            or float(bank.get("switch_time", -1.0)) != switch_time
            or int(bank.get("num_classes", -1)) != int(config.misc.num_classes)
            or int(bank.get("classes_covered", -1)) != int(config.misc.num_classes)
        ):
            raise ValueError(f"incompatible switch bank: {path}")
        return bank
    generated = generate_weak_switch_bank(
        model,
        config=config,
        samples=samples,
        batch_size=batch_size,
        seed=seed,
        switch_time=switch_time,
        precision=precision,
        device=device,
    )
    bank = {
        "protocol": PROTOCOL,
        "samples": samples,
        **generated,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)
    return bank


def sample_curriculum_batch(
    bank: dict[str, torch.Tensor | int | float],
    *,
    semigroup_levels: torch.Tensor,
    maximum_level: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample q-marginal OU states and adjacent old/new Bellman times."""

    states = bank["ou_states"]
    labels = bank["labels"]
    if not isinstance(states, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("switch bank tensors are missing")
    if not 1 <= maximum_level < len(semigroup_levels):
        raise ValueError("invalid curriculum level")
    indices = torch.randint(
        0, len(states), (batch_size,), generator=generator, device=device
    )
    levels = torch.randint(
        1, maximum_level + 1, (batch_size,), generator=generator, device=device
    )
    switch_states = states.index_select(0, indices.cpu()).to(
        device=device, dtype=torch.float32
    )
    batch_labels = labels.index_select(0, indices.cpu()).to(device=device)
    new_semigroup = semigroup_levels.index_select(0, levels)
    old_semigroup = semigroup_levels.index_select(0, levels - 1)
    switch_semigroup = semigroup_levels[0]
    retention = torch.exp(-(new_semigroup - switch_semigroup))
    noise = torch.randn(
        switch_states.shape,
        generator=generator,
        device=device,
        dtype=switch_states.dtype,
    )
    ou_state = (
        retention[:, None, None, None] * switch_states
        + torch.sqrt(1.0 - retention.square())[:, None, None, None] * noise
    )
    old_time = noise_time_from_ou_time(old_semigroup)
    new_time = noise_time_from_ou_time(new_semigroup)
    return ou_state, old_time, new_time, batch_labels, new_semigroup - old_semigroup


def build_hjb_target(
    source: torch.nn.Module,
    target_value: RAEv2NormalizedOUValue,
    *,
    ou_state: torch.Tensor,
    old_time: torch.Tensor,
    labels: torch.Tensor,
    semigroup_step: torch.Tensor,
    beta: float,
    particles: int,
    precision: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One first-order fitted-value-iteration target."""

    if particles <= 0:
        raise ValueError("Bellman particle count must be positive")
    old_state = ou_to_state(ou_state, old_time)
    with torch.no_grad(), source_autocast(precision):
        full_clean, base_clean = source(
            old_state,
            old_time,
            context=labels,
            attn_mask=None,
        )
    full_clean = full_clean.float()
    base_clean = base_clean.float()
    base_score = clean_prediction_to_ou_score(
        base_clean,
        ou_state=ou_state,
        noise_time=old_time,
    )
    score_gap = clean_gap_to_ou_score_gap(
        full_clean - base_clean,
        noise_time=old_time,
    )
    guided_score = base_score + beta * score_gap
    drift = ou_state + 2.0 * guided_score
    running_cost = normalized_hjb_running_cost(score_gap, beta=beta)

    query = ou_state.detach().requires_grad_(True)
    old_value = target_value(query, old_time, labels)
    value_gradient = torch.autograd.grad(old_value.sum(), query)[0].detach()
    particle_noise = torch.randn(
        (particles, *ou_state.shape),
        generator=generator,
        device=ou_state.device,
        dtype=ou_state.dtype,
    )
    particle_states = (
        ou_state.unsqueeze(0)
        + semigroup_step[:, None, None, None].unsqueeze(0) * drift.unsqueeze(0)
        + torch.sqrt(2.0 * semigroup_step)[:, None, None, None].unsqueeze(0)
        * particle_noise
    )
    repeated_times = old_time.repeat(particles)
    repeated_labels = labels.repeat(particles)
    with torch.no_grad():
        particle_values = target_value(
            particle_states.flatten(0, 1),
            repeated_times,
            repeated_labels,
        ).reshape(particles, len(ou_state))
    target = normalized_hjb_target(
        particle_values,
        value_gradient,
        running_cost_per_dimension=running_cost,
        semigroup_step=semigroup_step,
        ambient_dimension=ou_state[0].numel(),
    )
    diagnostics = {
        "running_cost": running_cost.detach(),
        "gradient_term": (
            ou_state[0].numel()
            * value_gradient.float().flatten(1).square().sum(1)
        ),
        "score_gap_rms": score_gap.float().flatten(1).square().mean(1).sqrt(),
        "target": target.detach(),
    }
    return target.detach(), diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--beta", type=float, default=1.78)
    parser.add_argument("--switch-time", type=float, default=0.5)
    parser.add_argument("--maximum-noise-time", type=float, default=0.99)
    parser.add_argument("--levels", type=int, default=96)
    parser.add_argument("--steps-per-level", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--bank-size", type=int, default=128)
    parser.add_argument("--bank-batch-size", type=int, default=8)
    parser.add_argument("--bank-seed", type=int, default=20260907)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.beta <= 1.0 or not 0.0 < args.switch_time < args.maximum_noise_time < 1.0:
        raise ValueError("invalid beta or semigroup interval")
    for name in (
        "levels",
        "steps_per_level",
        "max_steps",
        "batch_size",
        "particles",
        "bank_size",
        "bank_batch_size",
        "width",
        "depth",
        "log_every",
        "save_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    install_raev2_decoder_config_compat()
    config = load_config(config_path)
    num_classes = int(config.misc.num_classes)
    if args.bank_size < num_classes:
        raise ValueError(
            f"bank-size={args.bank_size} leaves class embeddings untrained; "
            f"use at least num_classes={num_classes}"
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
    torch.backends.cudnn.allow_tf32 = args.precision != "fp32"
    torch.manual_seed(args.seed)

    source = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    source.load_state_dict(payload[args.state_key], strict=True)
    source_step = int(payload.get("step", 0))
    del payload

    bank_path = output_dir / "weak_switch_bank.pt"
    bank_started = time.perf_counter()
    bank = load_or_generate_bank(
        bank_path,
        source,
        config=config,
        samples=args.bank_size,
        batch_size=args.bank_batch_size,
        seed=args.bank_seed,
        switch_time=args.switch_time,
        precision=args.precision,
        device=device,
    )
    bank_seconds = time.perf_counter() - bank_started

    latent_channels = int(config.misc.latent_size[0])
    value = RAEv2NormalizedOUValue(
        latent_channels,
        int(config.misc.num_classes),
        width=args.width,
        depth=args.depth,
        switch_time=args.switch_time,
    ).to(device)
    target = copy.deepcopy(value).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        value.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    switch_tensor = torch.tensor([args.switch_time], device=device)
    maximum_tensor = torch.tensor([args.maximum_noise_time], device=device)
    switch_semigroup = rae_ou_coefficients(switch_tensor)[-1][0]
    maximum_semigroup = rae_ou_coefficients(maximum_tensor)[-1][0]
    semigroup_levels = torch.linspace(
        switch_semigroup,
        maximum_semigroup,
        args.levels + 1,
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    request = {
        "protocol": PROTOCOL,
        "config": str(config_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_step": source_step,
        "source_state_key": args.state_key,
        "beta": args.beta,
        "switch_time": args.switch_time,
        "maximum_noise_time": args.maximum_noise_time,
        "levels": args.levels,
        "steps_per_level": args.steps_per_level,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "particles": args.particles,
        "bank_size": args.bank_size,
        "bank_classes_covered": int(bank["classes_covered"]),
        "bank_seed": args.bank_seed,
        "seed": args.seed,
        "width": args.width,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "precision": args.precision,
        "ambient_dimension": math.prod(config.misc.latent_size),
        "scientific_inference_scales": 0,
        "target": "switch-plane power tilt q * (p/q)^beta",
        "equation": "normalized OU HJB for phi=delta/D",
        "bank_seconds": bank_seconds,
    }
    (output_dir / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    interval_started = time.perf_counter()
    running: dict[str, float] = {
        "loss": 0.0,
        "prediction": 0.0,
        "target": 0.0,
        "running_cost": 0.0,
        "gradient_term": 0.0,
        "score_gap_rms": 0.0,
    }
    running_steps = 0
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.max_steps + 1):
        maximum_level = min(
            args.levels,
            1 + (step - 1) // args.steps_per_level,
        )
        ou_state, old_time, new_time, labels, semigroup_step = sample_curriculum_batch(
            bank,
            semigroup_levels=semigroup_levels,
            maximum_level=maximum_level,
            batch_size=args.batch_size,
            generator=generator,
            device=device,
        )
        target_values, diagnostics = build_hjb_target(
            source,
            target,
            ou_state=ou_state,
            old_time=old_time,
            labels=labels,
            semigroup_step=semigroup_step,
            beta=args.beta,
            particles=args.particles,
            precision=args.precision,
            generator=generator,
        )
        prediction = value(ou_state, new_time, labels)
        loss = F.mse_loss(prediction.float(), target_values.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite HJB loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        optimizer.step()
        update_ema(target, value, args.ema_decay)

        running["loss"] += float(loss.item())
        running["prediction"] += float(prediction.detach().mean().item())
        running["target"] += float(target_values.mean().item())
        for name in ("running_cost", "gradient_term", "score_gap_rms"):
            running[name] += float(diagnostics[name].mean().item())
        running_steps += 1

        if step % args.log_every == 0 or step == args.max_steps:
            elapsed = time.perf_counter() - interval_started
            row = {
                "step": step,
                "maximum_level": maximum_level,
                **{name: value_sum / running_steps for name, value_sum in running.items()},
                "parameter_gradient_norm": float(gradient_norm.item()),
                "steps_per_second": running_steps / elapsed,
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            running = {name: 0.0 for name in running}
            running_steps = 0
            interval_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)

        if step % args.save_every == 0 or step == args.max_steps:
            checkpoint = {
                "protocol": PROTOCOL,
                "step": step,
                "value": value.state_dict(),
                "value_ema": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "request": request,
            }
            path = checkpoint_dir / f"step_{step:08d}.pt"
            torch.save(checkpoint, path)
            print(json.dumps({"event": "checkpoint", "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
