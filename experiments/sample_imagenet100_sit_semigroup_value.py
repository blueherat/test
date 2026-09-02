#!/usr/bin/env python3
"""Sample the parameter-free semigroup value correction on v800/depth-4 IG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint
from torchvision.utils import save_image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.sample_imagenet100_sit_fid import (
    decode_latents_in_chunks,
    official_pixel_quantization,
)
from experiments.sample_imagenet100_sit_frozen_internal_v_head_fid import (
    load_frozen_internal_model,
)
from experiments.semigroup_guidance_value import (
    TokenPotentialHead,
    potential_gradient_to_velocity_correction,
    source_weak_and_final_features,
)
from experiments.train_imagenet100_sit_flow import (
    DEFAULT_OFFICIAL_SIT_REPO,
    LATENT_SHAPE,
    NUM_CLASSES,
    SD_VAE_SCALING_FACTOR,
    atomic_json_dump,
    load_official_sit_module,
    sha256_file,
)


DEFAULT_VALUE_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "semigroup_value_depth4_beta1p6_v1/checkpoints/step_00005000.pt"
)
DEFAULT_OUTPUT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "semigroup_value_depth4_beta1p6_v1/fid1k"
)
FORMAT = "eqvae_imagenet100_sit_semigroup_value_samples_v1"


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


class SemigroupValueField:
    def __init__(
        self,
        source: torch.nn.Module,
        weak: torch.nn.Module,
        value: torch.nn.Module,
        labels: torch.Tensor,
        *,
        internal_depth: int,
        beta: float,
        intervention_time: float,
        minimum_time: float,
        precision: str,
        use_value_correction: bool,
    ) -> None:
        self.source = source
        self.weak = weak
        self.value = value
        self.labels = labels
        self.internal_depth = int(internal_depth)
        self.beta = float(beta)
        self.intervention_time = float(intervention_time)
        self.minimum_time = float(minimum_time)
        self.precision = precision
        self.use_value_correction = bool(use_value_correction)
        self.nfe = 0
        self.value_grad_evals = 0
        self.static_rms_sum = 0.0
        self.correction_rms_sum = 0.0
        self.correction_ratio_sum = 0.0
        self.diagnostic_count = 0

    def _autocast(self):
        if self.precision == "fp32":
            return torch.autocast("cuda", enabled=False)
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def __call__(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        scalar_time = float(time_value.detach().float().item())
        times = time_value.expand(len(state))
        if scalar_time >= self.intervention_time:
            with torch.no_grad(), self._autocast():
                strong, _, _, _ = source_weak_and_final_features(
                    self.source,
                    self.weak,
                    state,
                    times,
                    self.labels,
                    internal_depth=self.internal_depth,
                    latent_channels=LATENT_SHAPE[0],
                )
            return strong.float()

        needs_gradient = self.use_value_correction and scalar_time > self.minimum_time
        with torch.enable_grad() if needs_gradient else torch.no_grad():
            query = state.detach().requires_grad_(needs_gradient)
            with self._autocast():
                strong, weak, tokens, conditioning = source_weak_and_final_features(
                    self.source,
                    self.weak,
                    query,
                    times,
                    self.labels,
                    internal_depth=self.internal_depth,
                    latent_channels=LATENT_SHAPE[0],
                )
                static = weak + self.beta * (strong - weak)
                if needs_gradient:
                    potential = self.value(tokens, conditioning, times)
            static = static.float()
            if needs_gradient:
                gradient = torch.autograd.grad(
                    potential.float().sum(),
                    query,
                    create_graph=False,
                    retain_graph=False,
                )[0].float()
                correction = potential_gradient_to_velocity_correction(
                    gradient,
                    time_value=times.float(),
                )
                self.value_grad_evals += 1
            else:
                correction = torch.zeros_like(static)

        static_rms = sample_rms(static)
        correction_rms = sample_rms(correction)
        self.static_rms_sum += float(static_rms.sum().item())
        self.correction_rms_sum += float(correction_rms.sum().item())
        self.correction_ratio_sum += float(
            (correction_rms / static_rms.clamp_min(1e-8)).sum().item()
        )
        self.diagnostic_count += len(state)
        return (static + correction).detach()

    def summary(self) -> dict[str, float | int]:
        count = max(1, self.diagnostic_count)
        return {
            "nfe": self.nfe,
            "value_gradient_evaluations": self.value_grad_evals,
            "static_velocity_rms_mean": self.static_rms_sum / count,
            "value_correction_rms_mean": self.correction_rms_sum / count,
            "correction_to_static_rms_ratio_mean": self.correction_ratio_sum / count,
        }


def integrate_interval(
    state: torch.Tensor,
    field,
    start: float,
    end: float,
    *,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    return odeint(
        field,
        state.float(),
        torch.tensor([start, end], device=state.device, dtype=torch.float32),
        method="dopri5",
        atol=float(atol),
        rtol=float(rtol),
    )[-1]


def load_value_head(
    path: Path,
    *,
    weights: str,
    source_metadata: dict[str, object],
    hidden_size: int,
    device: torch.device,
) -> tuple[TokenPotentialHead, dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "imagenet100_sit_semigroup_value_v1":
        raise ValueError("unexpected semigroup value checkpoint protocol")
    config = payload["config"]
    if config["source_sha256"] != source_metadata["source_checkpoint_sha256"]:
        raise ValueError("value and source checkpoint hashes differ")
    if config["weak_sha256"] != source_metadata["head_checkpoint_sha256"]:
        raise ValueError("value and weak-head checkpoint hashes differ")
    head = TokenPotentialHead(
        hidden_size,
        intervention_time=float(config["intervention_time"]),
    )
    state_key = "potential_ema" if weights == "ema" else "potential"
    head.load_state_dict(payload[state_key], strict=True)
    head.to(device).eval().requires_grad_(False)
    metadata = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "step": int(payload["step"]),
        "weights": weights,
        "config": config,
    }
    return head, config, metadata


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts and batch size must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    value_path = args.value_checkpoint.expanduser().resolve()
    value_payload = torch.load(value_path, map_location="cpu", weights_only=False)
    weak_path = Path(value_payload["config"]["weak_checkpoint"])
    del value_payload
    sit_module, official_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    source, weak, source_metadata = load_frozen_internal_model(
        head_checkpoint_path=weak_path,
        head_weights="ema",
        sit_module=sit_module,
        source_metadata=official_metadata,
        device=device,
    )
    value, config, value_metadata = load_value_head(
        value_path,
        weights=args.value_weights,
        source_metadata=source_metadata,
        hidden_size=int(source.pos_embed.shape[-1]),
        device=device,
    )

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    ).to(device).eval().requires_grad_(False)
    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    sample_labels = np.empty(args.num_samples, dtype=np.int16)
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    summaries: list[dict[str, float | int]] = []
    preview = None
    cursor = 0
    started = time.perf_counter()
    batch_index = 0
    while cursor < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - cursor)
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
        noise = torch.randn(
            batch_size, *LATENT_SHAPE, generator=generator, device=device
        )
        labels = torch.randint(
            0, NUM_CLASSES, (batch_size,), generator=generator, device=device
        )
        field = SemigroupValueField(
            source,
            weak,
            value,
            labels,
            internal_depth=int(config["internal_depth"]),
            beta=float(config["beta"]),
            intervention_time=float(config["intervention_time"]),
            minimum_time=float(config["minimum_time"]),
            precision=args.precision,
            use_value_correction=args.mode == "semigroup",
        )
        state = integrate_interval(
            noise,
            field,
            0.0,
            float(config["intervention_time"]),
            atol=args.atol,
            rtol=args.rtol,
        )
        endpoint = integrate_interval(
            state,
            field,
            float(config["intervention_time"]),
            1.0,
            atol=args.atol,
            rtol=args.rtol,
        )
        if not torch.isfinite(endpoint).all():
            raise FloatingPointError(f"non-finite endpoint in batch {batch_index}")
        with torch.no_grad():
            decoded = decode_latents_in_chunks(
                vae,
                endpoint,
                scaling_factor=SD_VAE_SCALING_FACTOR,
                chunk_size=args.vae_decode_batch_size,
            )
        stop = cursor + batch_size
        images[cursor:stop] = official_pixel_quantization(decoded)
        sample_labels[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
        noise_digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
        label_digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
        summaries.append(field.summary())
        if preview is None:
            preview = decoded[: min(16, len(decoded))].detach().cpu()
        cursor = stop
        batch_index += 1
        if batch_index == 1 or cursor == args.num_samples or cursor % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "event": "sampling",
                        "mode": args.mode,
                        "generated": cursor,
                        "elapsed_seconds": time.perf_counter() - started,
                        **field.summary(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    samples_path = output_dir / f"samples_n{args.num_samples}.npz"
    np.savez(samples_path, arr_0=images)
    np.save(output_dir / f"labels_n{args.num_samples}.npy", sample_labels)
    assert preview is not None
    save_image(
        preview,
        output_dir / "preview.png",
        nrow=max(1, int(math.sqrt(len(preview)))),
        normalize=True,
        value_range=(-1, 1),
    )
    totals = {
        key: sum(float(row[key]) for row in summaries)
        for key in summaries[0]
    }
    total_diag = sum(int(row["nfe"]) for row in summaries)
    manifest = {
        "format": FORMAT,
        "mode": args.mode,
        "formula": (
            "weak+beta*(strong-weak)+((1-t)/t)*grad_z(delta)"
            if args.mode == "semigroup"
            else "weak+beta*(strong-weak)"
        ),
        "requested_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sample_rng_mode": "per_batch_seed_plus_batch_index",
        "noise_sha256": noise_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "samples": str(samples_path),
        "source": source_metadata,
        "value": value_metadata,
        "sampler": {
            "method": "dopri5",
            "atol": args.atol,
            "rtol": args.rtol,
            "precision": args.precision,
            "intervention_time": config["intervention_time"],
            "minimum_value_time": config["minimum_time"],
            "extra_value_gain": 0,
        },
        "diagnostics": {
            "batch_nfe_sum": int(totals["nfe"]),
            "value_gradient_evaluations": int(totals["value_gradient_evaluations"]),
            "mean_batch_static_velocity_rms": totals["static_velocity_rms_mean"]
            / len(summaries),
            "mean_batch_value_correction_rms": totals["value_correction_rms_mean"]
            / len(summaries),
            "mean_batch_correction_ratio": totals[
                "correction_to_static_rms_ratio_mean"
            ]
            / len(summaries),
            "diagnostic_nfe_denominator": total_diag,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
    print(json.dumps({"event": "complete", **manifest["diagnostics"]}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-checkpoint", type=Path, default=DEFAULT_VALUE_CHECKPOINT)
    parser.add_argument("--value-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--mode", choices=("static", "semigroup"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
