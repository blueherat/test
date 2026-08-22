#!/usr/bin/env python3
"""Sample one pre-registered multiscale-guidance condition on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint
from torchvision.utils import save_image

try:
    from experiments.imagenet100_sit_multiscale_guidance import (
        BAND_NAMES,
        interpolate_time_table,
        ordered_band_component,
        per_sample_rms,
        project_frequency_band,
        route_depth_by_target_band,
        schedule_depth,
        select_per_sample,
        band_time_component,
        decompose_weak_head_difference,
        weak_head_difference_field,
    )
    from experiments.imagenet100_sit_multiscale_models import (
        InternalHeadSpec,
        evaluate_sit_field,
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )
except ModuleNotFoundError:
    from imagenet100_sit_multiscale_guidance import (
        BAND_NAMES,
        interpolate_time_table,
        ordered_band_component,
        per_sample_rms,
        project_frequency_band,
        route_depth_by_target_band,
        schedule_depth,
        select_per_sample,
        band_time_component,
        decompose_weak_head_difference,
        weak_head_difference_field,
    )
    from imagenet100_sit_multiscale_models import (
        InternalHeadSpec,
        evaluate_sit_field,
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )


DEFAULT_V800 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_V500 = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_seed0/checkpoints/step_00500000.pt"
)
ADAPTIVE_KINDS = {
    "baseline",
    "full_gap",
    "band_time",
    "ordered_bands",
    "static_depth",
    "depth_schedule",
    "spectral_router",
    "raw_compute_schedule",
    "head_difference",
    "head_difference_component",
}
EULER_KINDS = {"euler_baseline", "euler_depth8", "spectral_delay"}


def parse_head_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("head must use NAME=PATH")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("head must use non-empty NAME=PATH")
    return name, Path(path)


def load_condition(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind = str(payload.get("kind"))
    if kind not in ADAPTIVE_KINDS | EULER_KINDS:
        raise ValueError(f"unsupported condition kind: {kind!r}")
    payload["kind"] = kind
    payload.setdefault("gamma", 0.0)
    if not np.isfinite(float(payload["gamma"])):
        raise ValueError("condition gamma must be finite")
    if kind in {"head_difference", "head_difference_component"}:
        positive = str(payload.get("positive_head", ""))
        negative = str(payload.get("negative_head", ""))
        if not positive or not negative:
            raise ValueError(f"{kind} requires positive_head and negative_head")
        if positive == negative:
            raise ValueError(f"{kind} requires two distinct heads")
    if kind == "head_difference_component":
        component = str(payload.get("component", ""))
        if component not in {"full", "parallel", "orthogonal"}:
            raise ValueError(
                "head_difference_component requires full, parallel, or orthogonal"
            )
    return payload


def provider_needs_head(provider: str) -> bool:
    return provider != "external_v500"


def depth_name_map(heads: dict[str, InternalHeadSpec]) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, spec in heads.items():
        if spec.prediction_target == "velocity" and name.startswith("depth"):
            if spec.depth in result:
                raise ValueError(f"multiple velocity heads at depth {spec.depth}")
            result[spec.depth] = name
    return result


class ConditionField:
    def __init__(
        self,
        *,
        condition: dict[str, object],
        strong: torch.nn.Module,
        strong_semantics,
        external: torch.nn.Module | None,
        external_semantics,
        heads: dict[str, InternalHeadSpec],
        atlas: dict[str, object],
        labels: torch.Tensor,
    ) -> None:
        self.condition = condition
        self.strong = strong
        self.strong_semantics = strong_semantics
        self.external = external
        self.external_semantics = external_semantics
        self.heads = heads
        self.atlas = atlas
        self.labels = labels
        self.depth_names = depth_name_map(heads)
        self.nfe = 0
        self.strong_forwards = 0
        self.external_forwards = 0
        self.router_counts: dict[int, int] = {depth: 0 for depth in self.depth_names}
        self.rms_scale_sum = 0.0
        self.rms_scale_count = 0

    def _evaluate_source(
        self,
        state: torch.Tensor,
        times: torch.Tensor,
        *,
        head_names: set[str] = frozenset(),
        raw_depths: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[int, torch.Tensor]]:
        selected = {name: self.heads[name] for name in head_names}
        self.strong_forwards += 1
        return evaluate_source_with_heads(
            self.strong,
            state,
            times,
            self.labels,
            heads=selected,
            raw_depths=raw_depths,
            source_semantics=self.strong_semantics,
        )

    def _provider_gap(
        self,
        provider: str,
        state: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if provider == "external_v500":
            if self.external is None:
                raise ValueError("external provider requested without v500 model")
            full = evaluate_sit_field(
                self.strong,
                self.strong_semantics,
                state,
                times,
                self.labels,
            )
            self.strong_forwards += 1
            weak = evaluate_sit_field(
                self.external,
                self.external_semantics,
                state,
                times,
                self.labels,
            )
            self.external_forwards += 1
            return full, full - weak
        full, trained, _ = self._evaluate_source(
            state,
            times,
            head_names={provider},
        )
        return full, full - trained[provider]

    def _calibration_scale(self, provider: str, times: torch.Tensor) -> torch.Tensor:
        table = self.atlas["rms_calibration"][provider]
        return interpolate_time_table(
            times,
            table["times"],
            table["scale_to_depth8_v"],
        )

    def __call__(self, time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        times = time_value.expand(len(state))
        kind = str(self.condition["kind"])
        gamma = float(self.condition.get("gamma", 0.0))
        if kind == "baseline":
            self.strong_forwards += 1
            return evaluate_sit_field(
                self.strong,
                self.strong_semantics,
                state,
                times,
                self.labels,
            )
        if kind in {"full_gap", "band_time", "ordered_bands"}:
            provider = str(self.condition["provider"])
            full, gap = self._provider_gap(provider, state, times)
            if kind == "full_gap":
                direction = gap
            elif kind == "band_time":
                scale = 1.0
                if self.condition.get("amplitude") == "equal_action":
                    cell = f"{self.condition['interval']}_{self.condition['band']}"
                    scale = float(
                        self.atlas["action_calibration"][provider][
                            "equal_action_scales"
                        ][cell]
                    )
                direction = band_time_component(
                    gap,
                    times,
                    band=str(self.condition["band"]),
                    interval=str(self.condition["interval"]),
                    scale=scale,
                )
            else:
                cell_scales = None
                if self.condition.get("amplitude") == "equal_action":
                    cell_scales = self.atlas["action_calibration"][provider][
                        "equal_action_scales"
                    ]
                direction = ordered_band_component(
                    gap,
                    times,
                    order=str(self.condition["order"]),
                    cell_scales=cell_scales,
                )
            return full + gamma * direction

        if kind == "head_difference":
            positive_head = str(self.condition["positive_head"])
            negative_head = str(self.condition["negative_head"])
            full, trained, _ = self._evaluate_source(
                state,
                times,
                head_names={positive_head, negative_head},
            )
            return weak_head_difference_field(
                full,
                trained[positive_head],
                trained[negative_head],
                gamma=gamma,
            )

        if kind == "head_difference_component":
            positive_head = str(self.condition["positive_head"])
            negative_head = str(self.condition["negative_head"])
            full, trained, _ = self._evaluate_source(
                state,
                times,
                head_names={positive_head, negative_head},
            )
            difference, parallel, orthogonal, _ = decompose_weak_head_difference(
                full,
                trained[positive_head],
                trained[negative_head],
            )
            components = {
                "full": difference,
                "parallel": parallel,
                "orthogonal": orthogonal,
            }
            return full + gamma * components[str(self.condition["component"])]

        if kind in {"static_depth", "depth_schedule", "spectral_router"}:
            head_names = set(self.depth_names.values())
            full, trained, _ = self._evaluate_source(
                state,
                times,
                head_names=head_names,
            )
            gaps = {
                depth: full - trained[name]
                for depth, name in self.depth_names.items()
            }
            if bool(self.condition.get("rms_matched", False)):
                gaps = {
                    depth: gap
                    * self._calibration_scale(self.depth_names[depth], times)[
                        :, None, None, None
                    ]
                    for depth, gap in gaps.items()
                }
            if kind == "static_depth":
                direction = gaps[int(self.condition["depth"])]
            elif kind == "depth_schedule":
                selected = schedule_depth(
                    times,
                    order=str(self.condition["order"]),
                    # gamma_schedule_sweep_v4_condition_depths
                    depths=tuple(
                        int(value)
                        for value in self.condition.get("depths", (4, 8, 10))
                    ),
                )
                direction = select_per_sample(gaps, selected)
            else:
                direction, selected = route_depth_by_target_band(
                    gaps,
                    times,
                    reverse=bool(self.condition.get("reverse", False)),
                )
                for depth in self.router_counts:
                    self.router_counts[depth] += int((selected == depth).sum().item())
            return full + gamma * direction

        if kind == "raw_compute_schedule":
            raw_depths = tuple(int(value) for value in self.condition["depths"])
            full, _, raw = self._evaluate_source(
                state,
                times,
                raw_depths=raw_depths,
            )
            gaps = {depth: full - raw[depth] for depth in raw_depths}
            if bool(self.condition.get("rms_matched", True)):
                gaps = {
                    depth: gap
                    * self._calibration_scale(f"raw_final_h{depth}", times)[
                        :, None, None, None
                    ]
                    for depth, gap in gaps.items()
                }
            selected = schedule_depth(
                times,
                order=str(self.condition["order"]),
                depths=tuple(int(value) for value in self.condition["depths"]),
            )
            direction = select_per_sample(gaps, selected)
            return full + gamma * direction
        raise ValueError(f"condition {kind!r} requires the fixed-step sampler")


def fixed_euler_endpoint(
    *,
    noise: torch.Tensor,
    labels: torch.Tensor,
    condition: dict[str, object],
    strong: torch.nn.Module,
    strong_semantics,
    heads: dict[str, InternalHeadSpec],
    atlas: dict[str, object],
) -> tuple[torch.Tensor, dict[str, object]]:
    kind = str(condition["kind"])
    steps = int(condition.get("steps", 100))
    if steps < 2:
        raise ValueError("Euler sampling requires at least two steps")
    gamma = float(condition.get("gamma", 0.0))
    depth_names = depth_name_map(heads)
    depth8_name = depth_names.get(8, "depth8_v")
    if kind == "euler_depth8" and 8 not in depth_names:
        raise ValueError("euler_depth8 requires a loaded depth-8 velocity head")
    state = noise.float()
    history: list[torch.Tensor] = []
    nfe = 0
    head_forwards = 0
    rms_scale_sum = 0.0
    rms_scale_count = 0
    fitted_lag_time = {
        band: float(atlas["delay_fit"]["fitted_lag_time"][band])
        for band in BAND_NAMES
    }
    fitted_lags = {
        band: int(round(fitted_lag_time[band] * steps)) for band in BAND_NAMES
    }
    reference_table = atlas["rms_calibration"][depth8_name]

    for step in range(steps):
        time_scalar = float(step / steps)
        times = torch.full(
            (len(state),),
            time_scalar,
            device=state.device,
            dtype=torch.float32,
        )
        if kind == "euler_depth8":
            full, trained, _ = evaluate_source_with_heads(
                strong,
                state,
                times,
                labels,
                heads={depth8_name: heads[depth8_name]},
                source_semantics=strong_semantics,
            )
            direction = full - trained[depth8_name]
            velocity = full + gamma * direction
            head_forwards += 1
        else:
            full = evaluate_sit_field(
                strong,
                strong_semantics,
                state,
                times,
                labels,
            )
            if kind == "euler_baseline":
                velocity = full
            elif kind == "spectral_delay":
                current_clean = state + (1.0 - time_scalar) * full
                synthetic = torch.zeros_like(current_clean)
                for band in BAND_NAMES:
                    lag = int(condition.get("lags", fitted_lags)[band])
                    if lag == 0 or not history:
                        source_clean = current_clean
                    else:
                        source_clean = history[max(0, len(history) - lag)]
                    synthetic = synthetic + project_frequency_band(source_clean, band)
                denominator = max(
                    1.0 - time_scalar,
                    float(condition.get("floor", 1.0 / steps)),
                )
                direction = (current_clean - synthetic) / denominator
                if bool(condition.get("rms_matched", False)):
                    target = interpolate_time_table(
                        times,
                        reference_table["times"],
                        reference_table["rms"],
                    )
                    scale = target / per_sample_rms(direction).clamp_min(1e-12)
                    scale = scale.clamp(max=float(condition.get("max_rms_scale", 8.0)))
                    direction = direction * scale[:, None, None, None]
                    rms_scale_sum += float(scale.sum().item())
                    rms_scale_count += len(scale)
                velocity = full + gamma * direction
                history.append(current_clean.detach())
            else:
                raise ValueError(f"unsupported fixed-step condition: {kind}")
        state = state + velocity.float() / float(steps)
        nfe += 1
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"non-finite Euler state at step {step}")
    return state, {
        "nfe": nfe,
        "internal_head_forwards": head_forwards,
        "fitted_lags": fitted_lags,
        "fitted_lag_time": fitted_lag_time,
        "mean_rms_scale": rms_scale_sum / rms_scale_count if rms_scale_count else None,
    }


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    condition = load_condition(args.condition_json)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    allocator = configure_cuda_allocator(device, limit_gib=args.cuda_allocator_limit_gib)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas = json.loads(args.atlas_summary.read_text(encoding="utf-8"))
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    strong, strong_semantics, strong_metadata = load_sit_field_model(
        checkpoint_path=args.strong_checkpoint,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    head_paths = dict(args.head)
    needed_head_names: set[str] = set()
    kind = str(condition["kind"])
    provider = str(condition.get("provider", ""))
    if provider and provider_needs_head(provider):
        needed_head_names.add(provider)
    if kind == "static_depth":
        needed_head_names.add(f"depth{int(condition['depth'])}_v")
    elif kind == "depth_schedule":
        for depth in condition.get("depths", (4, 8, 10)):
            needed_head_names.add(f"depth{int(depth)}_v")
    elif kind == "spectral_router":
        needed_head_names.update(
            name
            for name in head_paths
            if name.startswith("depth") and name.endswith("_v")
        )
    elif kind == "euler_depth8":
        needed_head_names.add("depth8_v")
    elif kind in {"head_difference", "head_difference_component"}:
        needed_head_names.update(
            (str(condition["positive_head"]), str(condition["negative_head"]))
        )
    missing_heads = sorted(needed_head_names - set(head_paths))
    if missing_heads:
        raise ValueError(f"condition requires missing heads: {missing_heads}")
    heads = {
        name: load_internal_head_for_source(
            checkpoint_path=head_paths[name],
            name=name,
            head_weights="ema",
            model=strong,
            sit_module=sit_module,
            source_checkpoint_path=args.strong_checkpoint,
            source_metadata=source_metadata,
            device=device,
        )
        for name in sorted(needed_head_names)
    }
    external = None
    external_semantics = None
    external_metadata = None
    if provider == "external_v500":
        external, external_semantics, external_metadata = load_sit_field_model(
            checkpoint_path=args.external_weak_checkpoint,
            weights="ema",
            sit_module=sit_module,
            source_metadata=source_metadata,
            device=device,
        )

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)
    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    preview: torch.Tensor | None = None
    cursor = 0
    total_nfe = 0
    total_strong_forwards = 0
    total_external_forwards = 0
    total_head_forwards = 0
    router_counts: dict[int, int] = {}
    fixed_metadata: dict[str, object] | None = None
    started = time.perf_counter()

    while cursor < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - cursor)
        generator = torch.Generator(device=device).manual_seed(
            args.seed + cursor // args.batch_size
        )
        noise = torch.randn(batch_size, *LATENT_SHAPE, generator=generator, device=device)
        labels = torch.randint(
            0,
            NUM_CLASSES,
            (batch_size,),
            generator=generator,
            device=device,
        )
        if kind in ADAPTIVE_KINDS:
            field = ConditionField(
                condition=condition,
                strong=strong,
                strong_semantics=strong_semantics,
                external=external,
                external_semantics=external_semantics,
                heads=heads,
                atlas=atlas,
                labels=labels,
            )
            endpoint = odeint(
                field,
                noise.float(),
                torch.tensor([0.0, 1.0], device=device),
                method="dopri5",
                atol=args.atol,
                rtol=args.rtol,
            )[-1]
            total_nfe += field.nfe
            total_strong_forwards += field.strong_forwards
            total_external_forwards += field.external_forwards
            for depth, count in field.router_counts.items():
                router_counts[depth] = router_counts.get(depth, 0) + count
        else:
            endpoint, fixed_metadata = fixed_euler_endpoint(
                noise=noise,
                labels=labels,
                condition=condition,
                strong=strong,
                strong_semantics=strong_semantics,
                heads=heads,
                atlas=atlas,
            )
            total_nfe += int(fixed_metadata["nfe"])
            total_strong_forwards += int(fixed_metadata["nfe"])
            total_head_forwards += int(fixed_metadata["internal_head_forwards"])
        decoded = decode_latents_in_chunks(
            vae,
            endpoint,
            scaling_factor=SD_VAE_SCALING_FACTOR,
            chunk_size=args.vae_decode_batch_size,
        )
        stop = cursor + batch_size
        images[cursor:stop] = official_pixel_quantization(decoded)
        labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
        noise_digest.update(noise.cpu().contiguous().numpy().tobytes())
        label_digest.update(labels.cpu().contiguous().numpy().tobytes())
        if preview is None:
            preview = decoded[: min(16, len(decoded))].cpu()
        cursor = stop
        print(
            json.dumps(
                {
                    "condition": condition.get("name", kind),
                    "generated": cursor,
                    "total": args.num_samples,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    sample_path = output_dir / f"samples_n{args.num_samples}.npz"
    label_path = output_dir / f"labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    assert preview is not None
    save_image(
        preview,
        output_dir / "preview.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    histogram = np.bincount(labels_array.astype(np.int64), minlength=NUM_CLASSES)
    manifest = {
        "format": "eqvae_imagenet100_sit_multiscale_samples_v1",
        "condition": condition,
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "integrator": "dopri5" if kind in ADAPTIVE_KINDS else "fixed_euler",
        },
        "strong": strong_metadata,
        "external_weak": external_metadata,
        "heads": {
            name: {
                "depth": spec.depth,
                "prediction_target": spec.prediction_target,
                "checkpoint": spec.checkpoint,
                "checkpoint_sha256": spec.checkpoint_sha256,
            }
            for name, spec in heads.items()
        },
        "atlas_summary": str(args.atlas_summary),
        "noise_sha256": noise_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "label_histogram": histogram.tolist(),
        "total_nfe": total_nfe,
        "strong_forwards": total_strong_forwards,
        "external_forwards": total_external_forwards,
        "internal_head_forwards": total_head_forwards,
        "router_counts": router_counts,
        "fixed_step_metadata": fixed_metadata,
        "samples": str(sample_path),
        "labels": str(label_path),
        "elapsed_seconds": time.perf_counter() - started,
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json_dump(manifest, output_dir / "sampling_manifest.json")
    print(json.dumps({"status": "complete", "samples": str(sample_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-json", type=Path, required=True)
    parser.add_argument("--atlas-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_V800)
    parser.add_argument("--external-weak-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument(
        "--head", action="append", type=parse_head_argument, default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
