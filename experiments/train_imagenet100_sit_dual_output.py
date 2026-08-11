"""Train a Dynamic Dual-Output inspired SiT on the ImageNet-100 latent cache.

Everything except the final ``2C+1`` projection and objective is inherited
from the audited SiT-S/2 linear-flow baseline. The two native heads predict
the clean SD-VAE latent and source epsilon. A per-location gate mixes their
implied velocities using a stop-gradient gate loss.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    from experiments import train_imagenet100_sit_flow as base
    from experiments.imagenet100_sit_dual_output import (
        GateActivation,
        dual_output_flow_losses,
        dual_output_velocities,
        retrofit_dual_output_head,
        split_dual_output,
    )
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from imagenet100_sit_dual_output import (
        GateActivation,
        dual_output_flow_losses,
        dual_output_velocities,
        retrofit_dual_output_head,
        split_dual_output,
    )


PROTOCOL = "imagenet100_sit_dual_output_linear_flow_v1"
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "runs/sit-s-2_dual-output_seed0"
)
TIME_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class DualTrainConfig:
    cache_dir: str
    output_dir: str
    official_sit_repo: str
    model_name: str
    global_batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    cfg_dropout: float
    ema_decay: float
    precision: str
    compile: bool
    compile_mode: str
    allow_tf32: bool
    num_workers: int
    prefetch_factor: int
    log_every: int
    validation_every: int
    validation_batches: int
    save_every: int
    seed: int
    gate_activation: str
    epsilon_weight: float
    clean_weight: float
    gate_weight: float
    denominator_floor: float


def create_dual_output_sit(
    sit_module,
    *,
    model_name: str,
    cfg_dropout: float,
) -> nn.Module:
    model = sit_module.SiT_models[model_name](
        input_size=base.LATENT_SHAPE[-1],
        num_classes=base.NUM_CLASSES,
        class_dropout_prob=cfg_dropout,
    )
    return retrofit_dual_output_head(model, latent_channels=base.LATENT_SHAPE[0])


def validate_resume(
    stored: dict,
    current: DualTrainConfig,
    world_size: int,
) -> None:
    immutable = (
        "cache_dir",
        "official_sit_repo",
        "model_name",
        "global_batch_size",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "cfg_dropout",
        "ema_decay",
        "precision",
        "compile",
        "compile_mode",
        "allow_tf32",
        "seed",
        "gate_activation",
        "epsilon_weight",
        "clean_weight",
        "gate_weight",
        "denominator_floor",
    )
    current_values = asdict(current)
    mismatches = [
        f"{key}: checkpoint={stored.get(key)!r}, current={current_values[key]!r}"
        for key in immutable
        if stored.get(key) != current_values[key]
    ]
    if int(stored.get("world_size", world_size)) != world_size:
        mismatches.append(
            f"world_size: checkpoint={stored.get('world_size')!r}, current={world_size}"
        )
    if mismatches:
        raise ValueError("incompatible resume configuration:\n  " + "\n  ".join(mismatches))


def validation_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    context: base.DistributedContext,
    config: DualTrainConfig,
    batches: int,
    seed: int,
) -> dict[str, object]:
    """Validation implementation kept separate for compile-friendly training."""

    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    metric_names = (
        "epsilon_mse",
        "clean_mse",
        "gate_loss",
        "velocity_x_mse",
        "velocity_epsilon_mse",
        "velocity_dynamic_mse",
        "gate_mean",
        "gate_square",
    )
    totals = torch.zeros(len(metric_names) + 1, device=context.device, dtype=torch.float64)
    bin_totals = torch.zeros(
        len(TIME_BIN_EDGES) - 1,
        len(metric_names) + 1,
        device=context.device,
        dtype=torch.float64,
    )
    model.eval()
    with torch.inference_mode():
        for batch_index, (moments, labels) in enumerate(loader):
            if batch_index >= batches:
                break
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            posterior_noise = torch.randn(
                (len(moments), *base.LATENT_SHAPE),
                generator=generator,
                device=context.device,
            )
            clean = base.sample_sdvae_posterior(moments, posterior_noise)
            epsilon = torch.randn(
                clean.shape,
                generator=generator,
                device=context.device,
            )
            time_value = torch.rand(
                (len(clean),),
                generator=generator,
                device=context.device,
            )
            state, target_velocity = base.linear_flow_state_target(
                clean,
                epsilon,
                time_value,
            )
            with base.autocast_context(config.precision):
                output = model(state, time_value, labels)
            epsilon_prediction, clean_prediction, _, gate = split_dual_output(
                output,
                latent_channels=base.LATENT_SHAPE[0],
                gate_activation=config.gate_activation,
            )
            velocities = dual_output_velocities(
                output,
                state=state,
                time_value=time_value,
                gate_activation=config.gate_activation,
                denominator_floor=config.denominator_floor,
            )
            time_image = time_value[:, None, None, None]
            gate_residual = gate.float() * (
                time_image * (clean_prediction.detach().float() - clean)
            ) + (1.0 - gate.float()) * (
                (1.0 - time_image) * (epsilon - epsilon_prediction.detach().float())
            )
            values = torch.stack(
                (
                    (epsilon_prediction.float() - epsilon).square().flatten(1).mean(1),
                    (clean_prediction.float() - clean).square().flatten(1).mean(1),
                    gate_residual.square().flatten(1).mean(1),
                    (velocities["x"] - target_velocity).square().flatten(1).mean(1),
                    (velocities["epsilon"] - target_velocity).square().flatten(1).mean(1),
                    (velocities["dynamic"] - target_velocity).square().flatten(1).mean(1),
                    gate.float().flatten(1).mean(1),
                    gate.float().square().flatten(1).mean(1),
                ),
                dim=1,
            ).double()
            totals[:-1] += values.sum(dim=0)
            totals[-1] += len(values)
            for index, (lower, upper) in enumerate(
                zip(TIME_BIN_EDGES[:-1], TIME_BIN_EDGES[1:], strict=True)
            ):
                mask = (time_value >= lower) & (
                    time_value < upper if upper < 1.0 else time_value <= upper
                )
                if mask.any():
                    bin_totals[index, :-1] += values[mask].sum(dim=0)
                    bin_totals[index, -1] += mask.sum()

    base.reduce_sum(totals, context)
    base.reduce_sum(bin_totals, context)
    if totals[-1].item() == 0:
        raise RuntimeError("validation loader produced no samples")
    means = totals[:-1] / totals[-1]
    result: dict[str, object] = {
        name: float(value.item()) for name, value in zip(metric_names, means, strict=True)
    }
    result["gate_std"] = float(
        torch.sqrt((means[-1] - means[-2].square()).clamp_min(0.0)).item()
    )
    bins: list[dict[str, float]] = []
    for index, (lower, upper) in enumerate(
        zip(TIME_BIN_EDGES[:-1], TIME_BIN_EDGES[1:], strict=True)
    ):
        count = bin_totals[index, -1]
        if count.item() == 0:
            continue
        bin_means = bin_totals[index, :-1] / count
        row = {
            "t_min": lower,
            "t_max": upper,
            "count": int(count.item()),
        }
        row.update(
            {
                name: float(value.item())
                for name, value in zip(metric_names, bin_means, strict=True)
            }
        )
        bins.append(row)
    result["time_bins"] = bins
    return result


def build_metadata(
    *,
    config: DualTrainConfig,
    context: base.DistributedContext,
    source_metadata: dict,
    model: nn.Module,
    cache_manifest: dict,
) -> dict:
    import timm

    return {
        "protocol": PROTOCOL,
        "config": asdict(config),
        "world_size": context.world_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "official_sit": source_metadata,
        "paper": {
            "title": "Dynamic Dual-Output Diffusion Models",
            "venue": "CVPR 2022",
            "arxiv": "2203.04304",
            "adaptation": "DDPM reverse-mean mixing adapted to linear-flow velocity mixing",
            "underspecified_choice": "bounded sigmoid gate; paper does not publish activation code",
        },
        "objective": {
            "path": "x_t=(1-t)*epsilon+t*clean",
            "native_heads": ["epsilon", "clean"],
            "output_channels": "2C+1 with a per-location gate",
            "loss": "L_epsilon + L_clean + L_gate",
            "gate_residual": (
                "r*t*(clean_hat-clean) + "
                "(1-r)*(1-t)*(epsilon-epsilon_hat)"
            ),
            "gate_branch_stop_gradient": True,
            "time_distribution": "Uniform[0,1)",
        },
        "latent": {
            "posterior": "mean+std*N(0,I)",
            "scaling_factor": base.SD_VAE_SCALING_FACTOR,
            "shape": list(base.LATENT_SHAPE),
        },
        "data_manifest": cache_manifest,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "timm": timm.__version__,
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "repo_git_commit": base.git_value(base.REPO_ROOT, "rev-parse", "HEAD"),
        },
    }


def train(args: argparse.Namespace) -> None:
    context = base.initialize_distributed(args.device)
    try:
        if args.global_batch_size % context.world_size:
            raise ValueError("--global-batch-size must be divisible by world size")
        local_batch_size = args.global_batch_size // context.world_size
        config = DualTrainConfig(
            cache_dir=str(args.cache_dir.expanduser().resolve()),
            output_dir=str(args.output_dir.expanduser().resolve()),
            official_sit_repo=str(args.official_sit_repo.expanduser().resolve()),
            model_name=args.model,
            global_batch_size=int(args.global_batch_size),
            max_steps=int(args.max_steps),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            cfg_dropout=float(args.cfg_dropout),
            ema_decay=float(args.ema_decay),
            precision=args.precision,
            compile=bool(args.compile),
            compile_mode=args.compile_mode,
            allow_tf32=bool(args.allow_tf32),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            log_every=int(args.log_every),
            validation_every=int(args.validation_every),
            validation_batches=int(args.validation_batches),
            save_every=int(args.save_every),
            seed=int(args.seed),
            gate_activation=args.gate_activation,
            epsilon_weight=float(args.epsilon_weight),
            clean_weight=float(args.clean_weight),
            gate_weight=float(args.gate_weight),
            denominator_floor=float(args.denominator_floor),
        )
        if min(
            config.global_batch_size,
            config.max_steps,
            config.log_every,
            config.save_every,
            config.epsilon_weight,
            config.clean_weight,
            config.gate_weight,
            config.denominator_floor,
        ) <= 0:
            raise ValueError("batch, step, loss weights and denominator floor must be positive")
        if config.denominator_floor >= 0.5:
            raise ValueError("--denominator-floor must be smaller than 0.5")
        if not 0 <= config.cfg_dropout < 1:
            raise ValueError("--cfg-dropout must be in [0,1)")
        base.configure_runtime(config.seed, context.rank, config.allow_tf32)

        cache_dir = Path(config.cache_dir)
        output_dir = Path(config.output_dir)
        cache_manifest_path = cache_dir / "manifest.json"
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_manifest_sha256 = base.sha256_file(cache_manifest_path)
        if cache_manifest.get("format") != "eqvae_imagenet100_cmc_sdvae_moments_v1":
            raise ValueError(f"unsupported data manifest: {cache_manifest_path}")

        sit_module, source_metadata = base.load_official_sit_module(
            Path(config.official_sit_repo),
            verify_source=args.verify_sit_source,
        )
        if config.model_name not in sit_module.SiT_models:
            raise ValueError(f"unknown official SiT model: {config.model_name}")
        raw_model = create_dual_output_sit(
            sit_module,
            model_name=config.model_name,
            cfg_dropout=config.cfg_dropout,
        ).to(context.device)
        ema = base.ModelEMA(raw_model)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=True,
        )

        train_loader, train_sampler = base.create_loader(
            cache_dir=cache_dir,
            split="train",
            local_batch_size=local_batch_size,
            context=context,
            seed=config.seed,
            shuffle=True,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            drop_last=True,
        )
        validation_loader, validation_sampler = base.create_loader(
            cache_dir=cache_dir,
            split="validation",
            local_batch_size=local_batch_size,
            context=context,
            seed=config.seed + 1,
            shuffle=False,
            num_workers=max(0, min(2, config.num_workers)),
            prefetch_factor=config.prefetch_factor,
            drop_last=False,
        )
        validation_sampler.set_epoch(0)

        resume_path = base.resolve_resume(args.resume, output_dir)
        start_step = 0
        restored_rng: dict | None = None
        if resume_path is not None:
            checkpoint = torch.load(
                resume_path,
                map_location=context.device,
                weights_only=False,
            )
            if checkpoint.get("protocol") != PROTOCOL:
                raise ValueError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
            validate_resume(checkpoint["config"], config, context.world_size)
            if checkpoint.get("data_manifest_sha256") != cache_manifest_sha256:
                raise ValueError("checkpoint data manifest does not match current cache")
            if checkpoint.get("official_sit") != source_metadata:
                raise ValueError("checkpoint official SiT source does not match this run")
            raw_model.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"])
            rng_states = checkpoint.get("rng_states")
            if not isinstance(rng_states, list) or len(rng_states) != context.world_size:
                raise ValueError("checkpoint is missing per-rank RNG states")
            restored_rng = rng_states[context.rank]

        train_model: nn.Module = raw_model
        if config.compile:
            train_model = torch.compile(
                raw_model,
                mode=config.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        if context.world_size > 1:
            train_model = DDP(
                train_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        if resume_path is None:
            ema.load_state_dict(raw_model.state_dict())
        if restored_rng is not None:
            base.restore_rng_state(restored_rng, context.device)

        raw_model.train()
        ema.module.eval()
        batches = base.infinite_train_batches(train_loader, train_sampler, start_step)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            metadata = build_metadata(
                config=config,
                context=context,
                source_metadata=source_metadata,
                model=raw_model,
                cache_manifest=cache_manifest,
            )
            base.atomic_json_dump(metadata, output_dir / "run_config.json")
            print(
                json.dumps(
                    {
                        "event": "start",
                        "step": start_step,
                        "model": config.model_name,
                        "parameters": metadata["parameter_count"],
                        "world_size": context.world_size,
                        "global_batch": config.global_batch_size,
                        "local_batch": local_batch_size,
                        "precision": config.precision,
                        "compile": config.compile,
                        "gate_activation": config.gate_activation,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        base.barrier(context)

        metrics_path = output_dir / "train_metrics.jsonl"
        metric_keys = (
            "total",
            "epsilon",
            "clean",
            "gate",
            "gate_mean",
            "gate_std",
        )
        running = torch.zeros(len(metric_keys), device=context.device, dtype=torch.float64)
        running_steps = 0
        interval_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step in range(start_step + 1, config.max_steps + 1):
            moments, labels = next(batches)
            moments = moments.to(context.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(context.device, dtype=torch.long, non_blocking=True)
            posterior_noise = torch.randn(
                (len(moments), *base.LATENT_SHAPE),
                device=context.device,
            )
            clean = base.sample_sdvae_posterior(moments, posterior_noise)
            epsilon = torch.randn_like(clean)
            time_value = torch.rand((len(clean),), device=context.device)
            state, _ = base.linear_flow_state_target(clean, epsilon, time_value)

            with base.autocast_context(config.precision):
                output = train_model(state, time_value, labels)
            losses = dual_output_flow_losses(
                output,
                clean_target=clean,
                epsilon_target=epsilon,
                time_value=time_value,
                gate_activation=config.gate_activation,
                epsilon_weight=config.epsilon_weight,
                clean_weight=config.clean_weight,
                gate_weight=config.gate_weight,
            )
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(config.ema_decay)
            running += torch.stack([losses[key].detach().double() for key in metric_keys])
            running_steps += 1

            if step % config.log_every == 0 or step == config.max_steps:
                torch.cuda.synchronize(context.device)
                elapsed = time.perf_counter() - interval_started
                values = torch.cat(
                    (
                        running,
                        torch.tensor([running_steps], device=context.device, dtype=torch.float64),
                    )
                )
                base.reduce_sum(values, context)
                elapsed_tensor = torch.tensor(elapsed, device=context.device)
                base.reduce_max(elapsed_tensor, context)
                memory_tensor = torch.tensor(
                    torch.cuda.max_memory_allocated(context.device) / 2**30,
                    device=context.device,
                )
                base.reduce_max(memory_tensor, context)
                denominator = values[-1]
                row = {
                    "step": step,
                    **{
                        f"train_{key}": float((values[index] / denominator).item())
                        for index, key in enumerate(metric_keys)
                    },
                    "steps_per_second": float(running_steps / elapsed_tensor.item()),
                    "images_per_second": float(
                        running_steps * config.global_batch_size / elapsed_tensor.item()
                    ),
                    "max_allocated_gb": float(memory_tensor.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                if context.is_main:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                running.zero_()
                running_steps = 0
                interval_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(context.device)

            should_validate = config.validation_every > 0 and (
                step % config.validation_every == 0 or step == config.max_steps
            )
            if should_validate:
                pause_started = time.perf_counter()
                raw_metrics = validation_metrics(
                    model=raw_model,
                    loader=validation_loader,
                    context=context,
                    config=config,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                )
                ema_metrics = validation_metrics(
                    model=ema.module,
                    loader=validation_loader,
                    context=context,
                    config=config,
                    batches=config.validation_batches,
                    seed=config.seed + 700_000,
                )
                raw_model.train()
                if context.is_main:
                    row = {
                        "step": step,
                        "raw_validation": raw_metrics,
                        "ema_validation": ema_metrics,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                interval_started += time.perf_counter() - pause_started

            should_save = step % config.save_every == 0 or step == config.max_steps
            if should_save:
                pause_started = time.perf_counter()
                rng_states = base.gather_rng_states(context)
                if context.is_main:
                    checkpoint_path = output_dir / "checkpoints" / f"step_{step:08d}.pt"
                    base.atomic_torch_save(
                        {
                            "protocol": PROTOCOL,
                            "step": step,
                            "model": raw_model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                            "config": {**asdict(config), "world_size": context.world_size},
                            "official_sit": source_metadata,
                            "data_manifest_sha256": cache_manifest_sha256,
                        },
                        checkpoint_path,
                    )
                    print(
                        json.dumps({"event": "checkpoint", "path": str(checkpoint_path)}),
                        flush=True,
                    )
                base.barrier(context)
                interval_started += time.perf_counter() - pause_started
    finally:
        base.cleanup_distributed(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=base.DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--model", default="SiT-S/2")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=5_000)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-activation", choices=("sigmoid", "identity", "clamp"), default="sigmoid")
    parser.add_argument("--epsilon-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=1.0)
    parser.add_argument("--denominator-floor", type=float, default=1e-3)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit checkpoint path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
