"""Paired CIFAR-10 DDIM sampling for time-windowed PAG policies.

All policies start from exactly the same Gaussian samples.  The model, solver,
number of function evaluations, and attention layer are shared.  Policies only
change the scalar multiplying ``full - identity_attention`` over a declared
timestep interval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image

from experiments.internal_guidance_direction import guided_prediction
from experiments.run_pag_cifar10_direction_audit import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    attention_modules,
    configure_fp32,
    pag_dual_prediction,
)


DEFAULT_OUTPUT = Path.home() / "data" / "eqvae" / "pag_cifar10_time_window_samples"
DEFAULT_LAYER = "down_blocks.1.attentions.1"


@dataclass(frozen=True)
class SamplingPolicy:
    name: str
    active_scale_s: float
    timestep_low: int
    timestep_high: int

    def scale_at(self, timestep: int) -> float:
        if self.timestep_low <= int(timestep) <= self.timestep_high:
            return float(self.active_scale_s)
        return 1.0


def default_policies() -> tuple[SamplingPolicy, ...]:
    return (
        SamplingPolicy("baseline_full", 1.0, 0, 999),
        SamplingPolicy("pag_all_s1.25", 1.25, 0, 999),
        SamplingPolicy("pag_high_t800_s1.25", 1.25, 800, 999),
        SamplingPolicy("pag_low_t500_s1.25", 1.25, 0, 500),
        SamplingPolicy("interpolate_low_t300_s0.5", 0.5, 0, 300),
    )


def validate_sampling_protocol(
    *,
    samples: int,
    batch_size: int,
    inference_steps: int,
    policies: tuple[SamplingPolicy, ...],
    train_timesteps: int,
) -> None:
    if samples <= 0 or batch_size <= 0 or inference_steps <= 0:
        raise ValueError("samples, batch_size, and inference_steps must be positive")
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    for policy in policies:
        if policy.active_scale_s < 0:
            raise ValueError("policy scales must be non-negative")
        if not 0 <= policy.timestep_low <= policy.timestep_high < train_timesteps:
            raise ValueError("policy interval lies outside the training schedule")


def policy_scale_tensor(
    policies: tuple[SamplingPolicy, ...],
    timestep: int,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.tensor(
        [policy.scale_at(timestep) for policy in policies], device=device, dtype=dtype
    )
    return values.repeat_interleave(batch_size)


@torch.no_grad()
def sample_policy_batch(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    attention_module: torch.nn.Module,
    initial_noise: torch.Tensor,
    policies: tuple[SamplingPolicy, ...],
) -> torch.Tensor:
    policy_count = len(policies)
    batch_size = len(initial_noise)
    states = initial_noise.repeat(policy_count, 1, 1, 1)
    for timestep in scheduler.timesteps:
        timestep_value = int(timestep)
        batch_timestep = torch.full(
            (len(states),), timestep_value, device=states.device, dtype=torch.long
        )
        full, base = pag_dual_prediction(
            model, attention_module, states, batch_timestep
        )
        scales = policy_scale_tensor(
            policies,
            timestep_value,
            batch_size,
            device=states.device,
            dtype=states.dtype,
        )
        prediction = guided_prediction(full, base, scales)
        states = scheduler.step(prediction, timestep, states, eta=0.0).prev_sample
    return states.reshape((policy_count, batch_size) + tuple(initial_noise.shape[1:]))


@torch.no_grad()
def sample_plain_batch(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    state = initial_noise
    for timestep in scheduler.timesteps:
        batch_timestep = torch.full(
            (len(state),), int(timestep), device=state.device, dtype=torch.long
        )
        prediction = getattr(model(state, batch_timestep), "sample")
        state = scheduler.step(prediction, timestep, state, eta=0.0).prev_sample
    return state


def save_batch(
    samples: torch.Tensor,
    policies: tuple[SamplingPolicy, ...],
    output_dir: Path,
    start_index: int,
) -> None:
    arrays = (
        samples.float()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 1, 3, 4, 2)
        .cpu()
        .numpy()
    )
    for policy_index, policy in enumerate(policies):
        policy_dir = output_dir / policy.name
        policy_dir.mkdir(parents=True, exist_ok=True)
        for offset, array in enumerate(arrays[policy_index]):
            Image.fromarray(array, mode="RGB").save(
                policy_dir / f"{start_index + offset:06d}.png"
            )


def compute_fidelity_metrics(
    output_dir: Path,
    policies: tuple[SamplingPolicy, ...],
    *,
    dataset_root: Path,
    cuda: bool,
) -> dict[str, dict[str, float]]:
    from torch_fidelity import calculate_metrics
    from torch_fidelity.datasets import Cifar10_RGB, TransformPILtoRGBTensor

    dataset_path = dataset_root.expanduser().resolve()
    torchvision_root = (
        dataset_path
        if (dataset_path / "cifar-10-batches-py").is_dir()
        else dataset_path.parent
    )
    reference = Cifar10_RGB(
        str(torchvision_root),
        train=True,
        transform=TransformPILtoRGBTensor(),
        download=False,
    )
    result: dict[str, dict[str, float]] = {}
    for policy in policies:
        metrics = calculate_metrics(
            input1=str(output_dir / policy.name),
            input2=reference,
            input2_cache_name="cifar10-train-local-v1",
            cuda=cuda,
            isc=True,
            fid=True,
            kid=True,
            verbose=False,
        )
        result[policy.name] = {key: float(value) for key, value in metrics.items()}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="validate and evaluate already-complete policy PNG folders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fp32(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_root = args.model.expanduser().resolve()
    model = UNet2DModel.from_pretrained(
        model_root, local_files_only=True, use_safetensors=False
    )
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    ddpm = DDPMScheduler.from_pretrained(model_root, local_files_only=True)
    scheduler = DDIMScheduler.from_config(ddpm.config)
    scheduler.set_timesteps(args.inference_steps, device=device)
    policies = default_policies()
    validate_sampling_protocol(
        samples=args.samples,
        batch_size=args.batch_size,
        inference_steps=args.inference_steps,
        policies=policies,
        train_timesteps=int(scheduler.config.num_train_timesteps),
    )
    modules = attention_modules(model)
    if args.layer not in modules:
        raise ValueError(f"unknown attention layer {args.layer!r}; available: {sorted(modules)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    previous_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.skip_sampling and metadata_path.is_file()
        else {}
    )
    baseline_max_abs_difference = previous_metadata.get(
        "paired_baseline_max_abs_difference"
    )
    if args.skip_sampling:
        counts = {
            policy.name: len(list((output_dir / policy.name).glob("*.png")))
            for policy in policies
        }
        if any(value != args.samples for value in counts.values()):
            raise RuntimeError(
                f"cannot skip sampling: expected {args.samples} PNGs per policy, got {counts}"
            )
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
        for start in range(0, args.samples, args.batch_size):
            current_batch = min(args.batch_size, args.samples - start)
            initial = torch.randn(
                (current_batch, 3, 32, 32), generator=generator, dtype=torch.float32
            ).to(device)
            samples = sample_policy_batch(
                model, scheduler, modules[args.layer], initial, policies
            )
            if start == 0:
                plain = sample_plain_batch(model, scheduler, initial)
                baseline_max_abs_difference = float(
                    (plain.float() - samples[0].float()).abs().max().cpu()
                )
                if baseline_max_abs_difference > 2e-4:
                    raise RuntimeError(
                        "paired baseline diverged from plain DDIM by "
                        f"{baseline_max_abs_difference:.3e}"
                    )
            save_batch(samples, policies, output_dir, start)

    metadata = {
        "experiment": "public_ddpm_cifar10_time_window_pag_sampling_v1",
        "training": False,
        "model": str(model_root),
        "prediction_type": str(scheduler.config.prediction_type),
        "solver": "DDIM eta=0",
        "inference_steps": int(args.inference_steps),
        "solver_timesteps": [int(value) for value in scheduler.timesteps],
        "samples_per_policy": int(args.samples),
        "paired_initial_noise": True,
        "attention_layer": args.layer,
        "policies": [asdict(policy) for policy in policies],
        "active_solver_timesteps": {
            policy.name: [
                int(value)
                for value in scheduler.timesteps
                if policy.scale_at(int(value)) != 1.0
            ]
            for policy in policies
        },
        "paired_baseline_max_abs_difference": baseline_max_abs_difference,
        "scale_definition": "guided = base + s * (full - base); standard PAG gamma = s - 1",
        "seed": int(args.seed),
        "precision": "fp32",
        "tf32": False,
        "sampling_complete": True,
        "fidelity": {},
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    fidelity = (
        compute_fidelity_metrics(
            output_dir,
            policies,
            dataset_root=args.dataset,
            cuda=device.type == "cuda",
        )
        if args.compute_fid
        else {}
    )
    metadata["fidelity"] = fidelity
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(fidelity, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"saved paired samples to {output_dir}")


if __name__ == "__main__":
    main()
