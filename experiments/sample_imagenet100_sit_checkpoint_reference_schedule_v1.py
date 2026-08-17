#!/usr/bin/env python3
"""Sample ImageNet-100 SiT with a time-varying checkpoint guidance reference.

For the active weak checkpoint W_r and its reference-specific gamma_r:

    V(z,t) = S_800(z,t) + gamma_r * (S_800(z,t) - W_r(z,t))

The active reference is piecewise constant over equal time partitions. All
conditions load the same reference set and reseed immediately before sampling,
so paired conditions receive identical initial noise and class labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

try:
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_sit_field,
        load_sit_field_model,
    )
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from experiments.sample_imagenet100_sit_flow import integrate_velocity
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from imagenet100_sit_multiscale_models import evaluate_sit_field, load_sit_field_model
    from sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from sample_imagenet100_sit_flow import integrate_velocity
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )

DEFAULT_DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_STRONG = DEFAULT_DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"


def parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("reference must use NAME=PATH")
    name, raw_path = value.split("=", maxsplit=1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("reference must use non-empty NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def parse_name_float(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("gamma must use NAME=VALUE")
    name, raw_value = value.split("=", maxsplit=1)
    name = name.strip()
    try:
        number = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid gamma: {value!r}") from exc
    if not name or not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("gamma must use NAME=finite_nonnegative_value")
    return name, number


def load_condition(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("condition JSON must contain an object")
    order = payload.get("order")
    gammas = payload.get("gammas")
    if not isinstance(order, list) or not order or not all(isinstance(x, str) for x in order):
        raise ValueError("condition.order must be a non-empty list of reference names")
    if len(set(order)) != len(order):
        raise ValueError("condition.order must not repeat a reference in v1")
    if not isinstance(gammas, dict):
        raise ValueError("condition.gammas must be an object")
    for name in order:
        if name not in gammas:
            raise ValueError(f"missing gamma for active reference {name!r}")
        value = float(gammas[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid gamma for {name!r}: {value}")
    payload["order"] = list(order)
    payload["gammas"] = {str(k): float(v) for k, v in gammas.items()}
    payload.setdefault("name", "checkpoint_reference_schedule")
    return payload


def stage_index(time_value: torch.Tensor, stage_count: int) -> int:
    """Equal-time hard partition; exact boundaries enter the next stage."""
    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    if stage_count == 1:
        return 0
    boundaries = torch.arange(1, stage_count, device=time_value.device, dtype=torch.float32)
    boundaries = boundaries / float(stage_count)
    return int(torch.bucketize(time_value.float().reshape(1), boundaries, right=True).item())


class CheckpointReferenceField:
    def __init__(
        self,
        *,
        condition: dict[str, object],
        strong: torch.nn.Module,
        strong_semantics,
        references: dict[str, torch.nn.Module],
        reference_semantics: dict[str, object],
        labels: torch.Tensor,
    ) -> None:
        self.strong = strong
        self.strong_semantics = strong_semantics
        self.references = references
        self.reference_semantics = reference_semantics
        self.labels = labels
        self.order = tuple(str(x) for x in condition["order"])
        self.gammas = {str(k): float(v) for k, v in condition["gammas"].items()}
        self.nfe = 0
        self.strong_forwards = 0
        self.reference_forwards = {name: 0 for name in references}
        self.stage_evaluations = [0 for _ in self.order]

    def __call__(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        times = time_value.expand(len(state))
        strong_velocity = evaluate_sit_field(
            self.strong, self.strong_semantics, state, times, self.labels
        )
        self.strong_forwards += 1
        index = stage_index(time_value, len(self.order))
        name = self.order[index]
        gamma = self.gammas[name]
        self.stage_evaluations[index] += 1
        if gamma == 0.0:
            return strong_velocity
        weak_velocity = evaluate_sit_field(
            self.references[name], self.reference_semantics[name], state, times, self.labels
        )
        self.reference_forwards[name] += 1
        return strong_velocity + gamma * (strong_velocity - weak_velocity)


def validate_loaded_models(
    strong_semantics,
    strong_metadata: dict[str, object],
    reference_semantics: dict[str, object],
    reference_metadata: dict[str, dict[str, object]],
) -> None:
    if strong_semantics.prediction_target != "velocity":
        raise ValueError("strong checkpoint must be a native velocity model")
    for name, semantics in reference_semantics.items():
        if semantics.prediction_target != "velocity":
            raise ValueError(f"reference {name!r} is not a native velocity model")
        metadata = reference_metadata[name]
        for key in ("model_name", "data_manifest_sha256"):
            if metadata.get(key) != strong_metadata.get(key):
                raise ValueError(
                    f"reference {name!r} differs from strong on {key}: "
                    f"{metadata.get(key)!r} != {strong_metadata.get(key)!r}"
                )


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    condition = load_condition(args.condition_json.expanduser().resolve())
    references_arg = dict(args.reference_checkpoint)
    gamma_arg = dict(args.reference_gamma)
    missing_refs = set(condition["order"]) - set(references_arg)
    missing_gammas = set(condition["order"]) - set(gamma_arg)
    if missing_refs:
        raise ValueError(f"condition requires missing references: {sorted(missing_refs)}")
    if missing_gammas:
        raise ValueError(f"condition requires missing gammas: {sorted(missing_gammas)}")
    for name in condition["order"]:
        if abs(float(condition["gammas"][name]) - float(gamma_arg[name])) > 1e-12:
            raise ValueError(f"condition/CLI gamma mismatch for {name}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(device, limit_gib=args.cuda_allocator_limit_gib)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(), verify_source=args.verify_sit_source
    )
    strong, strong_semantics, strong_metadata = load_sit_field_model(
        checkpoint_path=args.strong_checkpoint.expanduser().resolve(),
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )

    references = {}
    reference_semantics = {}
    reference_metadata = {}
    for name in sorted(references_arg):
        model, semantics, metadata = load_sit_field_model(
            checkpoint_path=references_arg[name].expanduser().resolve(),
            weights="ema",
            sit_module=sit_module,
            source_metadata=source_metadata,
            device=device,
        )
        references[name] = model
        reference_semantics[name] = semantics
        reference_metadata[name] = metadata
    validate_loaded_models(
        strong_semantics, strong_metadata, reference_semantics, reference_metadata
    )

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True)
    vae.to(device).eval().requires_grad_(False)

    # Pairing rule: module construction consumes RNG. Reset only after all
    # strong/reference/VAE modules are loaded, identically for every condition.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    preview = None
    cursor = 0
    total_nfe = 0
    total_strong_forwards = 0
    total_reference_forwards = {name: 0 for name in references}
    total_stage_evaluations = [0 for _ in condition["order"]]
    started = time.perf_counter()

    while cursor < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - cursor)
        noise = torch.randn(batch_size, *LATENT_SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        field = CheckpointReferenceField(
            condition=condition,
            strong=strong,
            strong_semantics=strong_semantics,
            references=references,
            reference_semantics=reference_semantics,
            labels=labels,
        )
        endpoint = integrate_velocity(
            noise,
            field,
            num_output_points=args.num_output_points,
            atol=args.atol,
            rtol=args.rtol,
        )
        if not torch.isfinite(endpoint).all():
            raise FloatingPointError("non-finite endpoint")
        decoded = decode_latents_in_chunks(
            vae,
            endpoint,
            scaling_factor=SD_VAE_SCALING_FACTOR,
            chunk_size=args.vae_decode_batch_size,
        )
        stop = cursor + batch_size
        images[cursor:stop] = official_pixel_quantization(decoded)
        labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
        noise_digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
        label_digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
        if preview is None:
            preview = decoded[: min(16, len(decoded))].detach().cpu()
        total_nfe += field.nfe
        total_strong_forwards += field.strong_forwards
        for name, count in field.reference_forwards.items():
            total_reference_forwards[name] += int(count)
        for index, count in enumerate(field.stage_evaluations):
            total_stage_evaluations[index] += int(count)
        cursor = stop
        if cursor == batch_size or cursor == args.num_samples or cursor % args.log_every == 0:
            print(json.dumps({
                "condition": condition["name"],
                "generated": cursor,
                "total": args.num_samples,
                "elapsed_seconds": time.perf_counter() - started,
                "last_batch_nfe": field.nfe,
            }), flush=True)

    sample_path = output_dir / f"samples_n{args.num_samples}.npz"
    label_path = output_dir / f"labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    assert preview is not None
    save_image(preview, output_dir / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))
    histogram = np.bincount(labels_array.astype(np.int64), minlength=NUM_CLASSES)
    manifest = {
        "format": "eqvae_imagenet100_sit_checkpoint_reference_schedule_samples_v1",
        "condition": condition,
        "formula": "S + gamma_reference * (S - W_reference)",
        "partition": "equal hard time partitions; exact boundary enters next stage",
        "sampling": {
            "num_samples": int(args.num_samples),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "num_output_points": int(args.num_output_points),
            "integrator": "dopri5",
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "precision": "fp32",
            "allow_tf32": bool(args.allow_tf32),
        },
        "strong": strong_metadata,
        "references": reference_metadata,
        "reference_gammas": {name: float(gamma_arg[name]) for name in sorted(gamma_arg)},
        "noise_sha256": noise_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "label_histogram": histogram.tolist(),
        "total_nfe": int(total_nfe),
        "strong_forwards": int(total_strong_forwards),
        "reference_forwards": total_reference_forwards,
        "stage_evaluations": total_stage_evaluations,
        "samples": str(sample_path),
        "labels": str(label_path),
        "elapsed_seconds": time.perf_counter() - started,
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
    print(json.dumps({
        "event": "complete",
        "condition": condition["name"],
        "samples": str(sample_path),
        "noise_sha256": manifest["noise_sha256"],
        "label_sha256": manifest["label_sha256"],
    }, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--reference-checkpoint", action="append", type=parse_name_path, default=[], metavar="NAME=PATH", required=True)
    parser.add_argument("--reference-gamma", action="append", type=parse_name_float, default=[], metavar="NAME=VALUE", required=True)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=4.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=512)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
