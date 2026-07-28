"""Continue a frozen Imagenette latent prior with decoder-aware supervision.

This is a narrow causal test, not a replacement for the preregistered latent
prior experiment.  Every branch starts from the same trained prior while the
encoder and stochastic pixel decoder remain frozen.  ``decoder_lpl`` compares
paired decoder responses at the same pixel state and pixel time, so the only
changed input is the predicted clean latent condition.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_latent_prior_tradeoff import (  # noqa: E402
    EMA,
    LatentPriorTradeoffConfig,
    OrthogonalLatentInterface,
    build_prior,
    compute_real_features,
    deterministic_datasets,
    evaluate_rollout,
    fixed_orthogonal_basis,
    fixed_prior_validation_loss,
    latent_distribution_metrics,
    load_frozen_models,
    sample_prior_coordinates,
)
from experiments.imagenette_noise_responsibility import (  # noqa: E402
    ResNet18Evaluator,
    fixed_eval_subset,
    state_dict_sha256,
)
from experiments.mnist_spectral_rollout_toy import configure_fp32  # noqa: E402


DEFAULT_BASE_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"
DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/imagenette_decoder_aware_prior"
OBJECTIVES = ("flow", "clean_mse", "condition", "decoder_lpl")
DEFAULT_DECODER_LAYERS = ("middle", "up2", "up1", "up0")


@dataclass(frozen=True)
class DecoderAwarePriorConfig:
    base_run: Path = DEFAULT_BASE_ROOT / "d256_seed0_p0"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    objective: str = "decoder_lpl"
    continuation_steps: int = 2_000
    batch_size: int = 512
    auxiliary_batch_size: int = 8
    auxiliary_bank_size: int = 1_024
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    auxiliary_weight: float = 0.1
    prior_aux_time_max: float = 0.75
    pixel_time_min: float = 0.5
    pixel_time_max: float = 1.0
    log_every: int = 100
    validation_count: int = 256
    validation_batch_size: int = 8
    sample_count: int = 1_024
    sample_ode_steps: int = 100
    quality_count: int = 0
    pixel_ode_steps: int = 50
    num_workers: int = 4
    decoder_layers: tuple[str, ...] = DEFAULT_DECODER_LAYERS
    seed_offset: int = 40_000
    device: str = "cuda:0"
    save: bool = True
    overwrite: bool = False

    @property
    def branch_seed(self) -> int:
        source = load_source_config(self.base_run)
        return int(source.prior_seed) + int(self.seed_offset)

    @property
    def result_dir(self) -> Path:
        return self.output_root.expanduser() / (
            f"{self.base_run.name}_{self.objective}_s{int(self.continuation_steps)}"
        )

    def validate(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if int(self.continuation_steps) < 1 or int(self.batch_size) < 2:
            raise ValueError("continuation_steps and batch_size must be positive")
        if int(self.auxiliary_batch_size) < 2:
            raise ValueError("auxiliary_batch_size must be at least two")
        if int(self.auxiliary_bank_size) < int(self.auxiliary_batch_size):
            raise ValueError("auxiliary_bank_size must cover one auxiliary batch")
        if not 0.0 < float(self.prior_aux_time_max) <= 1.0:
            raise ValueError("prior_aux_time_max must lie in (0, 1]")
        if not 0.0 <= float(self.pixel_time_min) < float(self.pixel_time_max) <= 1.0:
            raise ValueError("pixel times must satisfy 0 <= min < max <= 1")
        if float(self.auxiliary_weight) < 0.0:
            raise ValueError("auxiliary_weight must be non-negative")
        if int(self.quality_count) < 0 or int(self.pixel_ode_steps) < 1:
            raise ValueError("quality_count must be non-negative and pixel_ode_steps positive")
        state_path = self.base_run.expanduser() / "prior_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing source prior checkpoint: {state_path}")


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=_json_ready) + "\n"
    )


def load_source_config(base_run: str | Path) -> LatentPriorTradeoffConfig:
    path = Path(base_run).expanduser() / "config.json"
    values = json.loads(path.read_text())
    for key in ("data_root", "checkpoint_root", "output_root"):
        values[key] = Path(values[key])
    values["resume"] = False
    values["overwrite"] = False
    values["save"] = True
    return LatentPriorTradeoffConfig(**values)


def clean_estimate_from_velocity(
    state: torch.Tensor,
    predicted_velocity: torch.Tensor,
    time_value: torch.Tensor,
) -> torch.Tensor:
    if state.shape != predicted_velocity.shape:
        raise ValueError("state and predicted_velocity must have equal shapes")
    if time_value.shape != (len(state),):
        raise ValueError("time_value must have shape [batch]")
    return state - time_value[:, None] * predicted_velocity


def decoder_layer_modules(
    decoder: nn.Module,
    layer_names: Sequence[str],
) -> dict[str, nn.Module]:
    available = {
        "down0": decoder.down0[-1],
        "down1": decoder.down1[-1],
        "down2": decoder.down2[-1],
        "middle": decoder.middle[-1],
        "up2": decoder.up2[-1],
        "up1": decoder.up1[-1],
        "up0": decoder.up0[-1],
    }
    unknown = set(layer_names).difference(available)
    if unknown:
        raise ValueError(f"unknown decoder layers: {sorted(unknown)}")
    if not layer_names:
        raise ValueError("at least one decoder layer is required")
    return {name: available[name] for name in layer_names}


@contextmanager
def _capture_outputs(modules: Mapping[str, nn.Module]) -> Iterator[dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in modules.items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=name: captured.__setitem__(key, output)
            )
        )
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def decoder_response_features(
    decoder: nn.Module,
    pixel_state: torch.Tensor,
    pixel_time: torch.Tensor,
    condition: torch.Tensor,
    *,
    layer_names: Sequence[str] = DEFAULT_DECODER_LAYERS,
) -> tuple[torch.Tensor, ...]:
    modules = decoder_layer_modules(decoder, layer_names)
    with _capture_outputs(modules) as captured:
        decoder(pixel_state, pixel_time, condition)
    missing = set(layer_names).difference(captured)
    if missing:
        raise RuntimeError(f"decoder hooks did not capture layers: {sorted(missing)}")
    return tuple(captured[name] for name in layer_names)


def decoder_feature_loss(
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Scale-normalized, per-sample decoder feature discrepancy."""

    if not candidate or len(candidate) != len(reference):
        raise ValueError("candidate and reference feature lists must be non-empty and equal")
    layer_losses = []
    for candidate_layer, reference_layer in zip(candidate, reference):
        if candidate_layer.shape != reference_layer.shape:
            raise ValueError("paired decoder features must have equal shapes")
        dimensions = tuple(range(1, reference_layer.ndim))
        scale = reference_layer.detach().square().mean(dim=dimensions).clamp_min(1e-6)
        discrepancy = (candidate_layer - reference_layer.detach()).square().mean(dim=dimensions)
        layer_losses.append(discrepancy / scale)
    return torch.stack(layer_losses, dim=1).mean(dim=1)


def decoder_condition_loss(
    decoder: nn.Module,
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    candidate_embedding = decoder.condition_embedding(candidate)
    with torch.no_grad():
        reference_embedding = decoder.condition_embedding(reference)
    return (candidate_embedding - reference_embedding).square().mean(dim=1)


def prepare_auxiliary_bank(
    dataset,
    latent: torch.Tensor,
    *,
    count: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(int(count), len(dataset), len(latent))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:count]
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        persistent_workers=int(num_workers) > 0,
    )
    images = torch.cat([batch_images for batch_images, _ in loader])
    return images.contiguous(), latent[indices].contiguous()


def _make_generators(device: torch.device, seed: int) -> dict[str, torch.Generator]:
    offsets = {
        "indices": 11,
        "noise": 13,
        "time": 17,
        "aux_indices": 19,
        "aux_noise": 23,
        "aux_time": 29,
        "pixel_noise": 31,
        "pixel_time": 37,
    }
    return {
        name: torch.Generator(device=device).manual_seed(int(seed) + offset)
        for name, offset in offsets.items()
    }


def _prior_batch(
    model: nn.Module,
    data: torch.Tensor,
    interface: OrthogonalLatentInterface,
    batch_size: int,
    generators: Mapping[str, torch.Generator],
    *,
    auxiliary: bool,
    time_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = "aux_" if auxiliary else ""
    indices = torch.randint(
        len(data),
        (int(batch_size),),
        device=data.device,
        generator=generators[f"{prefix}indices"],
    )
    clean = data[indices]
    base_noise = torch.randn(
        (len(clean), interface.interface_dim),
        device=data.device,
        generator=generators[f"{prefix}noise"],
    )
    noise = base_noise[:, : interface.latent_dim]
    time_value = float(time_max) * torch.rand(
        (len(clean),),
        device=data.device,
        generator=generators[f"{prefix}time"],
    )
    state_coordinates = (1.0 - time_value[:, None]) * clean + time_value[:, None] * noise
    prediction = interface.recover(model(interface.embed(state_coordinates), time_value))
    return (
        indices,
        clean,
        noise - clean,
        prediction,
        clean_estimate_from_velocity(state_coordinates, prediction, time_value),
    )


def _auxiliary_loss(
    objective: str,
    model: nn.Module,
    decoder: nn.Module,
    bank_images: torch.Tensor,
    bank_latents: torch.Tensor,
    interface: OrthogonalLatentInterface,
    config: DecoderAwarePriorConfig,
    generators: Mapping[str, torch.Generator],
) -> torch.Tensor:
    indices, clean, _target, _prediction, estimate = _prior_batch(
        model,
        bank_latents,
        interface,
        config.auxiliary_batch_size,
        generators,
        auxiliary=True,
        time_max=config.prior_aux_time_max,
    )
    if objective == "clean_mse":
        return F.mse_loss(estimate, clean)
    if objective == "condition":
        return decoder_condition_loss(decoder, estimate, clean).mean()
    if objective != "decoder_lpl":
        return estimate.new_zeros(())

    image_indices = indices if bank_images.device == indices.device else indices.cpu()
    images = bank_images[image_indices].to(estimate.device, non_blocking=True)
    pixel_noise = torch.randn(
        images.shape,
        device=images.device,
        generator=generators["pixel_noise"],
    )
    pixel_time = float(config.pixel_time_min) + (
        float(config.pixel_time_max) - float(config.pixel_time_min)
    ) * torch.rand(
        (len(images),),
        device=images.device,
        generator=generators["pixel_time"],
    )
    expanded = pixel_time[:, None, None, None]
    pixel_state = (1.0 - expanded) * images + expanded * pixel_noise
    with torch.no_grad():
        reference = decoder_response_features(
            decoder,
            pixel_state,
            pixel_time,
            clean,
            layer_names=config.decoder_layers,
        )
    candidate = decoder_response_features(
        decoder,
        pixel_state,
        pixel_time,
        estimate,
        layer_names=config.decoder_layers,
    )
    return decoder_feature_loss(candidate, reference).mean()


@torch.no_grad()
def fixed_auxiliary_validation_loss(
    model: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    latent: torch.Tensor,
    interface: OrthogonalLatentInterface,
    config: DecoderAwarePriorConfig,
    *,
    seed: int,
) -> float:
    device = next(model.parameters()).device
    count = min(int(config.validation_count), len(images), len(latent))
    generators = _make_generators(device, int(seed))
    values = []
    for start in range(0, count, int(config.validation_batch_size)):
        end = min(start + int(config.validation_batch_size), count)
        batch_images = images[start:end].to(device)
        batch_latent = latent[start:end].to(device)
        # Use a local config because the helper samples indices from the supplied bank.
        local = DecoderAwarePriorConfig(
            **{
                **asdict(config),
                "objective": "decoder_lpl",
                "auxiliary_batch_size": end - start,
                "auxiliary_bank_size": end - start,
                "save": False,
            }
        )
        values.append(
            float(
                _auxiliary_loss(
                    "decoder_lpl",
                    model,
                    decoder,
                    batch_images,
                    batch_latent,
                    interface,
                    local,
                    generators,
                )
            )
        )
    return float(sum(values) / max(len(values), 1))


def train_continuation(
    source_model: nn.Module,
    decoder: nn.Module,
    train_latent: torch.Tensor,
    val_latent: torch.Tensor,
    bank_images: torch.Tensor,
    bank_latents: torch.Tensor,
    val_images: torch.Tensor,
    val_bank_latents: torch.Tensor,
    interface: OrthogonalLatentInterface,
    config: DecoderAwarePriorConfig,
) -> tuple[EMA, pd.DataFrame, dict]:
    device = torch.device(config.device)
    model = copy.deepcopy(source_model).to(device).train()
    ema = EMA(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    train_data = train_latent.to(device)
    bank_latents_device = bank_latents.to(device)
    generators = _make_generators(device, config.branch_seed)
    parameters = list(model.parameters())
    rows = []
    started = time.monotonic()
    for step in range(1, int(config.continuation_steps) + 1):
        _indices, _clean, target_velocity, predicted_velocity, _estimate = _prior_batch(
            model,
            train_data,
            interface,
            config.batch_size,
            generators,
            auxiliary=False,
        )
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        auxiliary_loss = _auxiliary_loss(
            config.objective,
            model,
            decoder,
            bank_images,
            bank_latents_device,
            interface,
            config,
            generators,
        )
        total_loss = flow_loss + float(config.auxiliary_weight) * auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        ema.update(model, config.ema_decay, step)

        if step == 1 or step % int(config.log_every) == 0 or step == config.continuation_steps:
            ema.module.eval()
            heldout_flow = fixed_prior_validation_loss(
                ema.module,
                val_latent,
                interface,
                batch_size=max(2, int(config.validation_batch_size)),
                seed=config.branch_seed + 101,
            )
            heldout_lpl = fixed_auxiliary_validation_loss(
                ema.module,
                decoder,
                val_images,
                val_bank_latents,
                interface,
                config,
                seed=config.branch_seed + 103,
            )
            row = {
                "step": int(step),
                "flow_loss": float(flow_loss.detach()),
                "auxiliary_loss": float(auxiliary_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "gradient_norm": float(gradient_norm),
                "heldout_flow_loss": float(heldout_flow),
                "heldout_decoder_lpl": float(heldout_lpl),
                "wall_seconds": float(time.monotonic() - started),
            }
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise FloatingPointError("non-finite continuation metric")
            rows.append(row)
            print(json.dumps({"objective": config.objective, **row}), flush=True)
            ema.module.eval()
            model.train()
    ema.module.eval()
    return ema, pd.DataFrame(rows), {
        "continuation_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "continuation_wall_seconds": float(time.monotonic() - started),
    }


def _prepare_output(config: DecoderAwarePriorConfig) -> Path | None:
    if not config.save:
        return None
    output = config.result_dir
    if output.exists() and config.overwrite:
        shutil.rmtree(output)
    if output.exists():
        raise FileExistsError(f"result directory already exists: {output}")
    output.mkdir(parents=True)
    _write_json(output / "config.json", asdict(config))
    return output


def run(config: DecoderAwarePriorConfig) -> Path | None:
    config.validate()
    source = load_source_config(config.base_run)
    configure_fp32(config.branch_seed)
    device = torch.device(config.device)
    output = _prepare_output(config)
    train_dataset, val_dataset = deterministic_datasets(source.data_root, source.image_size)
    encoder, decoder, metadata = load_frozen_models(source, device)
    frozen_before = {
        "encoder": state_dict_sha256(encoder),
        "decoder": state_dict_sha256(decoder),
    }
    cache = torch.load(
        config.base_run.expanduser() / "latent_cache.pt",
        map_location="cpu",
        weights_only=True,
    )
    prior_payload = torch.load(
        config.base_run.expanduser() / "prior_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    source_prior = build_prior(source, device)
    source_prior.load_state_dict(prior_payload["prior_ema"])
    source_prior.eval()
    basis = fixed_orthogonal_basis(source.interface_dim, source.basis_seed)
    interface = OrthogonalLatentInterface(source.latent_dim, basis).to(device)

    bank_images, bank_latents = prepare_auxiliary_bank(
        train_dataset,
        cache["train_latent"],
        count=config.auxiliary_bank_size,
        batch_size=max(config.validation_batch_size, config.auxiliary_batch_size),
        num_workers=config.num_workers,
        seed=config.branch_seed + 43,
    )
    val_images, val_bank_latents = prepare_auxiliary_bank(
        val_dataset,
        cache["val_latent"],
        count=config.validation_count,
        batch_size=config.validation_batch_size,
        num_workers=config.num_workers,
        seed=config.branch_seed + 47,
    )
    source_flow = fixed_prior_validation_loss(
        source_prior,
        cache["val_latent"],
        interface,
        batch_size=config.validation_batch_size,
        seed=config.branch_seed + 101,
    )
    source_lpl = fixed_auxiliary_validation_loss(
        source_prior,
        decoder,
        val_images,
        val_bank_latents,
        interface,
        config,
        seed=config.branch_seed + 103,
    )
    ema, history, training_metadata = train_continuation(
        source_prior,
        decoder,
        cache["train_latent"],
        cache["val_latent"],
        bank_images,
        bank_latents,
        val_images,
        val_bank_latents,
        interface,
        config,
    )
    generated_count = min(
        max(int(config.sample_count), int(config.quality_count)),
        len(cache["val_latent"]),
    )
    generated = sample_prior_coordinates(
        ema.module,
        interface,
        generated_count,
        config.sample_ode_steps,
        seed=config.branch_seed + 211,
        batch_size=config.batch_size,
    )
    count = min(int(config.sample_count), len(generated))
    distribution = latent_distribution_metrics(
        cache["val_latent"][:count], generated[:count], source
    )
    frozen_after = {
        "encoder": state_dict_sha256(encoder),
        "decoder": state_dict_sha256(decoder),
    }
    if frozen_before != frozen_after:
        raise RuntimeError("frozen encoder or decoder changed")
    summary = {
        "objective": config.objective,
        "source_run": str(config.base_run.expanduser()),
        "latent_dim": int(source.latent_dim),
        "frozen_seed": int(source.frozen_seed),
        "source_heldout_flow_loss": float(source_flow),
        "source_heldout_decoder_lpl": float(source_lpl),
        "final_heldout_flow_loss": float(history.iloc[-1].heldout_flow_loss),
        "final_heldout_decoder_lpl": float(history.iloc[-1].heldout_decoder_lpl),
        "heldout_flow_relative_change": float(history.iloc[-1].heldout_flow_loss / source_flow - 1.0),
        "heldout_decoder_lpl_relative_change": float(
            history.iloc[-1].heldout_decoder_lpl / source_lpl - 1.0
        ),
        "frozen_hashes_unchanged": True,
        "frozen_encoder_sha256": metadata["frozen_encoder_sha256"],
        "frozen_decoder_sha256": metadata["frozen_decoder_sha256"],
        **training_metadata,
        **distribution,
    }
    if int(config.quality_count) > 0:
        quality_count = min(int(config.quality_count), len(val_dataset), len(generated))
        evaluator = ResNet18Evaluator().to(device).eval()
        eval_subset = fixed_eval_subset(val_dataset, quality_count, seed=2_027)
        eval_indices = torch.as_tensor(eval_subset.indices, dtype=torch.long)
        real_images, real_labels, real_features = compute_real_features(
            eval_subset,
            evaluator,
            count=quality_count,
            batch_size=source.eval_batch_size,
            num_workers=config.num_workers,
            device=device,
        )
        evaluation_source = LatentPriorTradeoffConfig(
            **{
                **asdict(source),
                "quality_count": quality_count,
                "pixel_ode_steps": int(config.pixel_ode_steps),
                "device": config.device,
            }
        )
        oracle_metrics, _oracle_features, _oracle_preview = evaluate_rollout(
            "oracle",
            decoder,
            cache["val_latent"][eval_indices],
            real_images,
            real_labels,
            real_features,
            evaluator,
            metadata["class_to_idx"],
            evaluation_source,
        )
        prior_metrics, _prior_features, _prior_preview = evaluate_rollout(
            "prior",
            decoder,
            generated[:quality_count],
            real_images,
            real_labels,
            real_features,
            evaluator,
            metadata["class_to_idx"],
            evaluation_source,
        )
        summary.update(
            {
                "quality_count": quality_count,
                "oracle_feature_fid": float(oracle_metrics["feature_fid"]),
                "end_to_end_feature_fid": float(prior_metrics["feature_fid"]),
                "modeling_gap": float(
                    prior_metrics["feature_fid"] - oracle_metrics["feature_fid"]
                ),
                "prior_predicted_class_tv": float(prior_metrics["predicted_class_tv"]),
                "prior_predicted_effective_classes": float(
                    prior_metrics["predicted_effective_classes"]
                ),
            }
        )
    if output is not None:
        history.to_csv(output / "history.csv", index=False)
        torch.save({"prior_ema": ema.module.state_dict()}, output / "prior_state.pt")
        _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=DecoderAwarePriorConfig.base_run)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--objective", choices=OBJECTIVES, default="decoder_lpl")
    parser.add_argument("--continuation-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--auxiliary-batch-size", type=int, default=8)
    parser.add_argument("--auxiliary-bank-size", type=int, default=1_024)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=1_024)
    parser.add_argument("--sample-ode-steps", type=int, default=100)
    parser.add_argument("--quality-count", type=int, default=0)
    parser.add_argument("--pixel-ode-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = build_parser().parse_args(argv)
    return run(
        DecoderAwarePriorConfig(
            base_run=args.base_run,
            output_root=args.output_root,
            objective=args.objective,
            continuation_steps=args.continuation_steps,
            batch_size=args.batch_size,
            auxiliary_batch_size=args.auxiliary_batch_size,
            auxiliary_bank_size=args.auxiliary_bank_size,
            auxiliary_weight=args.auxiliary_weight,
            learning_rate=args.learning_rate,
            log_every=args.log_every,
            validation_count=args.validation_count,
            validation_batch_size=args.validation_batch_size,
            sample_count=args.sample_count,
            sample_ode_steps=args.sample_ode_steps,
            quality_count=args.quality_count,
            pixel_ode_steps=args.pixel_ode_steps,
            num_workers=args.num_workers,
            device=args.device,
            save=not args.no_save,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
