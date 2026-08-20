#!/usr/bin/env python3
"""Score existing SiT guidance directions with a frozen CAFM tangent critic.

The audit deliberately does not sample or compute a new FID.  It freezes a
critic selected only by tangent-classification validation loss, evaluates A/B
on a deterministic ImageNet-100 teacher bank, and writes the predictions before
historical quality measurements are joined by the separate summarizer.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.imagenet100_sit_cafm_tangent import (
    critic_from_sit_state,
    per_sample_dot,
)
from experiments.imagenet100_sit_multiscale_models import (
    evaluate_sit_field,
    evaluate_source_with_heads,
    load_internal_head_for_source,
    load_sit_field_model,
)
from experiments.imagenet100_sit_static_pair import (
    common_unique_orthogonal_directions,
    decompose_relative_to_anchor,
)
from experiments.train_imagenet100_sit_cafm_tangent_critic import (
    prepare_teacher_batch,
)
from experiments.train_imagenet100_sit_flow import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OFFICIAL_SIT_REPO,
    NpyMomentsDataset,
    load_official_sit_module,
    sha256_file,
)


DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
STRONG = DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
X800 = (
    DATA_ROOT
    / "runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
)
SAME_TARGET_STEPS = (180_000, 240_000, 270_000, 300_000, 400_000, 500_000, 600_000)
INTERNAL_DEPTHS = (4, 6, 8, 10, 12)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def make_validation_loader(cache_dir: Path, batch_size: int, workers: int) -> DataLoader:
    kwargs = {
        "dataset": NpyMomentsDataset(cache_dir, "validation"),
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": True,
        "drop_last": False,
        "persistent_workers": int(workers) > 0,
        "generator": torch.Generator().manual_seed(72_019),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def load_critic_and_strong(args, device: torch.device):
    critic_checkpoint = torch.load(
        args.critic_checkpoint, map_location="cpu", weights_only=False
    )
    if critic_checkpoint.get("format") != "eqvae_cafm_tangent_critic_v1":
        raise ValueError("unsupported critic checkpoint")
    model_metadata = critic_checkpoint["model"]
    strong_path = Path(model_metadata["strong_checkpoint"]).expanduser().resolve()
    if strong_path != args.strong_checkpoint:
        raise ValueError("critic and requested strong checkpoint differ")
    if sha256_file(strong_path) != model_metadata["strong_checkpoint_sha256"]:
        raise ValueError("strong checkpoint digest changed after critic training")

    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo, verify_source=True
    )
    strong_checkpoint = torch.load(strong_path, map_location="cpu", weights_only=False)
    critic = critic_from_sit_state(
        sit_module=sit_module,
        model_name=str(strong_checkpoint["config"]["model_name"]),
        state_dict=strong_checkpoint["ema"],
        input_size=32,
        num_classes=100,
        class_dropout_prob=float(strong_checkpoint["config"]["cfg_dropout"]),
    )
    critic.load_state_dict(critic_checkpoint["critic"], strict=True)
    critic.to(device).eval().requires_grad_(False)
    strong, strong_semantics, strong_metadata = load_sit_field_model(
        checkpoint_path=strong_path,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    return (
        critic,
        strong,
        strong_semantics,
        strong_metadata,
        sit_module,
        source_metadata,
        critic_checkpoint,
    )


def build_teacher_bank(
    *,
    critic,
    strong,
    strong_semantics,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    fields: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "state",
            "time",
            "labels",
            "real_velocity",
            "strong_velocity",
            "critic_gradient",
            "critic_value",
            "critic_action",
            "euclidean_residual_norm",
        )
    }
    seen = 0
    for moments, labels in loader:
        if seen >= num_samples:
            break
        keep = min(moments.shape[0], num_samples - seen)
        moments = moments[:keep]
        labels = labels[:keep]
        state, time_value, labels, real_velocity = prepare_teacher_batch(
            moments, labels, device=device, generator=generator
        )
        with torch.no_grad():
            strong_velocity = evaluate_sit_field(
                strong, strong_semantics, state, time_value, labels
            ).float()

        state_for_grad = state.detach().float().requires_grad_(True)
        time_for_grad = time_value.detach().float().requires_grad_(True)
        critic_value = critic(state_for_grad, time_for_grad, labels)
        critic_gradient = torch.autograd.grad(
            critic_value.sum(), state_for_grad, create_graph=False
        )[0]
        residual = real_velocity.float() - strong_velocity
        critic_action = per_sample_dot(critic_gradient, residual)

        batch = {
            "state": state.detach().float(),
            "time": time_value.detach().float(),
            "labels": labels.detach(),
            "real_velocity": real_velocity.detach().float(),
            "strong_velocity": strong_velocity.detach().float(),
            "critic_gradient": critic_gradient.detach().float(),
            "critic_value": critic_value.detach().float(),
            "critic_action": critic_action.detach().float(),
            "euclidean_residual_norm": residual.flatten(1).norm(dim=1).detach(),
        }
        for key, value in batch.items():
            fields[key].append(value.cpu())
        seen += keep
    if seen != num_samples:
        raise RuntimeError(f"requested {num_samples} samples but built {seen}")
    return {key: torch.cat(values, dim=0) for key, values in fields.items()}


def batched_model_direction(
    *,
    model,
    semantics,
    bank: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, bank["time"].shape[0], batch_size):
        stop = min(start + batch_size, bank["time"].shape[0])
        state = bank["state"][start:stop].to(device)
        time_value = bank["time"][start:stop].to(device)
        labels = bank["labels"][start:stop].to(device)
        with torch.no_grad():
            weak = evaluate_sit_field(model, semantics, state, time_value, labels)
        strong = bank["strong_velocity"][start:stop].to(device)
        outputs.append((strong - weak.float()).cpu())
    return torch.cat(outputs, dim=0)


def batched_internal_directions(
    *,
    strong,
    heads: dict,
    bank: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in heads}
    max_error = 0.0
    for start in range(0, bank["time"].shape[0], batch_size):
        stop = min(start + batch_size, bank["time"].shape[0])
        state = bank["state"][start:stop].to(device)
        time_value = bank["time"][start:stop].to(device)
        labels = bank["labels"][start:stop].to(device)
        with torch.no_grad():
            full, weak_outputs, _ = evaluate_source_with_heads(
                strong, state, time_value, labels, heads=heads
            )
        expected = bank["strong_velocity"][start:stop].to(device)
        max_error = max(max_error, float((full.float() - expected).abs().max()))
        for name, weak in weak_outputs.items():
            outputs[name].append((expected - weak.float()).cpu())
    if max_error > 0.05:
        raise RuntimeError(f"shared-backbone strong output mismatch: {max_error}")
    return {name: torch.cat(values, dim=0) for name, values in outputs.items()}


def match_norm(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value_norm = value.flatten(1).norm(dim=1).clamp_min(1e-12)
    reference_norm = reference.flatten(1).norm(dim=1)
    scale = (reference_norm / value_norm).reshape(-1, 1, 1, 1)
    return value * scale


def direction_rows(
    name: str,
    direction: torch.Tensor,
    bank: dict[str, torch.Tensor],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    if direction.shape != bank["strong_velocity"].shape:
        raise ValueError(f"direction {name} has the wrong shape")
    gradient = bank["critic_gradient"]
    residual = bank["real_velocity"] - bank["strong_velocity"]
    action = bank["critic_action"].double()
    slope = per_sample_dot(gradient, direction).double()
    euclidean = per_sample_dot(residual, direction).double()
    direction_norm = direction.flatten(1).norm(dim=1).double()
    residual_norm = residual.flatten(1).norm(dim=1).double()
    cosine = euclidean / (direction_norm * residual_norm).clamp_min(1e-12)
    time_value = bank["time"].double()
    bins = (
        ("overall", 0.0, 1.0, True),
        ("high_noise", 0.0, 0.5, False),
        ("low_noise", 0.5, 1.0, True),
        ("q0", 0.0, 0.25, False),
        ("q1", 0.25, 0.5, False),
        ("q2", 0.5, 0.75, False),
        ("q3", 0.75, 1.0, True),
    )
    rows = []
    for bin_name, low, high, include_high in bins:
        mask = (time_value >= low) & (
            (time_value <= high) if include_high else (time_value < high)
        )
        a = action[mask]
        b = slope[mask]
        e = euclidean[mask]
        dnorm = direction_norm[mask]
        cos = cosine[mask]
        count = int(mask.sum())
        a_mean = float(a.mean())
        b_mean = float(b.mean())
        denominator = float((b.square()).mean())
        gamma_sample_ls = float((a * b).mean()) / max(denominator, 1e-20)
        gamma_mean = a_mean / b_mean if abs(b_mean) > 1e-12 else math.nan
        rows.append(
            {
                "direction": name,
                "time_bin": bin_name,
                "time_low": low,
                "time_high": high,
                "samples": count,
                "A_mean": a_mean,
                "A_se": float(a.std(unbiased=True) / math.sqrt(count)),
                "B_mean": b_mean,
                "B_se": float(b.std(unbiased=True) / math.sqrt(count)),
                "AB_mean": float((a * b).mean()),
                "B2_mean": denominator,
                "gamma_hat_mean": gamma_mean,
                "gamma_hat_sample_ls": gamma_sample_ls,
                "predicted_reduction_gamma1_mean_residual": (
                    a_mean * a_mean - (a_mean - b_mean) ** 2
                ),
                "predicted_reduction_gamma1_sample_ls": float(
                    a.square().mean() - (a - b).square().mean()
                ),
                "B_positive_fraction": float((b > 0).double().mean()),
                "direction_rms": float(dnorm.square().mean().sqrt()),
                "euclidean_B_mean": float(e.mean()),
                "euclidean_cosine_mean": float(cos.mean()),
            }
        )
    arrays = {
        "action": action.numpy(),
        "slope": slope.numpy(),
        "euclidean": euclidean.numpy(),
        "direction_norm": direction_norm.numpy(),
        "cosine": cosine.numpy(),
    }
    return rows, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strong-checkpoint", type=Path, default=STRONG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--critic-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=91_003)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.num_samples, args.batch_size, args.critic_batch_size) < 1:
        raise ValueError("sample and batch sizes must be positive")
    args.critic_checkpoint = args.critic_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.strong_checkpoint = args.strong_checkpoint.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.official_sit_repo = args.official_sit_repo.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    (
        critic,
        strong,
        strong_semantics,
        strong_metadata,
        sit_module,
        source_metadata,
        critic_payload,
    ) = load_critic_and_strong(args, device)
    loader = make_validation_loader(args.cache_dir, args.critic_batch_size, args.workers)
    bank = build_teacher_bank(
        critic=critic,
        strong=strong,
        strong_semantics=strong_semantics,
        loader=loader,
        device=device,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "bank_samples": args.num_samples,
                "A_mean": float(bank["critic_action"].mean()),
                "state_sha256": tensor_sha256(bank["state"]),
            }
        ),
        flush=True,
    )

    directions: dict[str, torch.Tensor] = {}
    checkpoint_root = DATA_ROOT / "runs/sit-s-2_seed0/checkpoints"
    for step in SAME_TARGET_STEPS:
        path = checkpoint_root / f"step_{step:08d}.pt"
        weak, semantics, _ = load_sit_field_model(
            checkpoint_path=path,
            weights="ema",
            sit_module=sit_module,
            source_metadata=source_metadata,
            device=device,
        )
        name = f"v{step // 1000}"
        directions[name] = batched_model_direction(
            model=weak,
            semantics=semantics,
            bank=bank,
            device=device,
            batch_size=args.batch_size,
        )
        del weak
        gc.collect()
        torch.cuda.empty_cache()
        print(f"scored field {name}", flush=True)

    x_model, x_semantics, _ = load_sit_field_model(
        checkpoint_path=X800,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    directions["x800"] = batched_model_direction(
        model=x_model,
        semantics=x_semantics,
        bank=bank,
        device=device,
        batch_size=args.batch_size,
    )
    del x_model
    gc.collect()
    torch.cuda.empty_cache()
    print("scored field x800", flush=True)

    heads = {}
    for depth in INTERNAL_DEPTHS:
        name = f"internal_depth{depth}"
        heads[name] = load_internal_head_for_source(
            checkpoint_path=internal_head_checkpoint(depth),
            name=name,
            head_weights="ema",
            model=strong,
            sit_module=sit_module,
            source_checkpoint_path=args.strong_checkpoint,
            source_metadata=source_metadata,
            device=device,
        )
    directions.update(
        batched_internal_directions(
            strong=strong,
            heads=heads,
            bank=bank,
            device=device,
            batch_size=args.batch_size,
        )
    )
    print("scored internal heads", flush=True)

    strong_cpu = bank["strong_velocity"]
    for base_name in ("x800", "v500"):
        parallel, orthogonal = decompose_relative_to_anchor(
            strong_cpu, directions[base_name]
        )
        directions[f"{base_name}_parallel_to_strong"] = parallel
        directions[f"{base_name}_orthogonal_to_strong"] = orthogonal

    common_unique = common_unique_orthogonal_directions(
        strong_cpu,
        strong_cpu - directions["x800"],
        strong_cpu - directions["v500"],
    )
    directions.update(common_unique)

    random_generator = torch.Generator().manual_seed(args.seed + 551)
    random_direction = torch.randn(
        directions["v500"].shape, generator=random_generator
    )
    _, random_orthogonal = decompose_relative_to_anchor(
        strong_cpu, random_direction
    )
    directions["random_orthogonal_matched_v500"] = match_norm(
        random_orthogonal, directions["v500"]
    )
    time_image = bank["time"].reshape(-1, 1, 1, 1)
    directions["v180_high_noise_only"] = directions["v180"] * (time_image < 0.5)
    directions["v180_low_noise_only"] = directions["v180"] * (time_image >= 0.5)

    all_rows: list[dict] = []
    raw_arrays = {"time": bank["time"].numpy(), "A": bank["critic_action"].numpy()}
    for name, direction in directions.items():
        rows, arrays = direction_rows(name, direction, bank)
        all_rows.extend(rows)
        for key, value in arrays.items():
            raw_arrays[f"{name}__{key}"] = value
    write_csv(args.output_dir / "direction_scores.csv", all_rows)
    np.savez_compressed(args.output_dir / "per_sample_scores.npz", **raw_arrays)

    manifest = {
        "format": "eqvae_cafm_tangent_predictivity_audit_v1",
        "scope": "no-rollout no-new-FID preregistered CAFM A/B screen",
        "critic_checkpoint": str(args.critic_checkpoint),
        "critic_checkpoint_sha256": sha256_file(args.critic_checkpoint),
        "critic_training_step": int(critic_payload["step"]),
        "critic_validation": critic_payload.get("validation"),
        "strong": strong_metadata,
        "bank_samples": args.num_samples,
        "bank_seed": args.seed,
        "state_sha256": tensor_sha256(bank["state"]),
        "time_sha256": tensor_sha256(bank["time"]),
        "label_sha256": tensor_sha256(bank["labels"]),
        "directions": sorted(directions),
        "primary_prediction": "gamma_hat_mean=A_mean/B_mean",
        "secondary_prediction": "gamma_hat_sample_ls=E[A_i B_i]/E[B_i^2]",
        "generator_updated": False,
        "new_fid_computed": False,
        "field_compute_precision": "fp32",
    }
    atomic_json(manifest, args.output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
