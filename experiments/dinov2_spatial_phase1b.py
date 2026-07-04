from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.group_action_phase1 import (
    NON_IDENTITY_TRANSFORMS,
    TRANSFORMS,
    DistributedContext,
    ImageTensorDataset,
    Phase1Config,
    apply_spatial_operator,
    assess_success,
    configure_fp32,
    fit_ridge_maps,
    flatten_tokens,
    frobenius_relative_error,
    generator_channel_maps,
    generator_law_loss,
    ensure_square_latent_grid,
    init_ridge_stats,
    load_model_adapter,
    load_named_dataset,
    split_indices,
)


@dataclass
class SpatialPhase1BConfig:
    data_root: str = "/data/shared"
    dataset_name: str = "caltech101"
    dataset_split: str = "train"
    image_size: int = 256
    seed: int = 0
    train_count: int = 512
    val_count: int = 256
    test_count: int = 256
    batch_size: int = 16
    num_workers: int = 4
    artifact_dir: str = "artifacts/group_action_phase1b_dinov2_spatial"
    rae_repo_path: str = "external/RAE"
    steps: int = 1000
    lr: float = 1e-3
    law_weights: tuple[float, ...] = (0.01, 0.1, 1.0)
    channel_anchor_weight: float = 0.01
    spatial_anchor_weight: float = 0.1
    spatial_identity_weight: float = 0.01
    batch_images: int = 32
    grad_clip: float = 10.0
    score_law_weight: float = 0.5


def make_loader(base_dataset, indices: Sequence[int], config: SpatialPhase1BConfig) -> DataLoader:
    dataset = ImageTensorDataset(base_dataset, indices, config.image_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def collect_latents(adapter, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    z_parts = []
    target_parts = {g: [] for g in NON_IDENTITY_TRANSFORMS}
    for x, _ in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        z = adapter.encode(x).float()
        ensure_square_latent_grid(z, "rae_dinov2 latent")
        z_parts.append(z.cpu())
        for g in NON_IDENTITY_TRANSFORMS:
            y = adapter.encode(apply_spatial_operator(x, g)).float()
            target_parts[g].append(y.cpu())
    out = {"z": torch.cat(z_parts, dim=0)}
    for g in NON_IDENTITY_TRANSFORMS:
        out[g] = torch.cat(target_parts[g], dim=0)
    return out


def tokens(z: torch.Tensor) -> torch.Tensor:
    return z.permute(0, 2, 3, 1).reshape(z.shape[0], z.shape[2] * z.shape[3], z.shape[1]).contiguous()


def build_token_map(height: int, width: int, transform: str, device: torch.device) -> torch.Tensor:
    if height != width:
        raise ValueError(
            "Phase1B 的空间生成元实验需要方形 token grid，"
            f"当前 H={height}, W={width}。"
        )
    n = height * width
    basis = torch.eye(n, dtype=torch.float32, device=device).reshape(n, 1, height, width)
    transformed = apply_spatial_operator(basis, transform)
    return transformed.reshape(n, n)


def apply_token_channel(z_tokens: torch.Tensor, spatial_map: torch.Tensor, channel_map: torch.Tensor) -> torch.Tensor:
    spatial = torch.einsum("bnc,nm->bmc", z_tokens, spatial_map)
    return torch.einsum("bnc,cd->bnd", spatial, channel_map)


def fit_independent_channel_maps(train_data: Mapping[str, torch.Tensor], lambdas=(1e-4, 1e-3, 1e-2)) -> Dict[str, torch.Tensor]:
    z = train_data["z"]
    device = z.device
    stats = init_ridge_stats(z.shape[1], TRANSFORMS, device)
    for g in TRANSFORMS:
        if g == "identity":
            x_rows = flatten_tokens(z)
            y_rows = x_rows
        else:
            x_rows = flatten_tokens(apply_spatial_operator(z, g))
            y_rows = flatten_tokens(train_data[g])
        stats[g]["xtx"].add_(x_rows.T @ x_rows)
        stats[g]["xty"].add_(x_rows.T @ y_rows)
        stats[g]["yty"].add_(y_rows.pow(2).sum())
        stats[g]["n_tokens"].add_(float(x_rows.shape[0]))
    maps_by_lambda = fit_ridge_maps(stats, lambdas)
    return maps_by_lambda["1e-02"]


def spatial_generator_maps(rot90: torch.Tensor, flip_h: torch.Tensor) -> Dict[str, torch.Tensor]:
    n = rot90.shape[0]
    identity = torch.eye(n, device=rot90.device, dtype=rot90.dtype)
    rot180 = rot90 @ rot90
    rot270 = rot180 @ rot90
    return {
        "identity": identity,
        "rot90": rot90,
        "rot180": rot180,
        "rot270": rot270,
        "flip_h": flip_h,
        "flip_v": flip_h @ rot180,
    }


def relative_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (pred - target).pow(2).sum()
    denominator = target.pow(2).sum().clamp_min(1e-12)
    return numerator / denominator


def anchor_loss(maps: Mapping[str, torch.Tensor], anchors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for g in NON_IDENTITY_TRANSFORMS:
        target = anchors[g].detach()
        terms.append((maps[g] - target).pow(2).sum() / target.pow(2).sum().clamp_min(1e-12))
    return torch.stack(terms).mean()


def spatial_identity_loss(spatial_rot90: torch.Tensor, spatial_flip_h: torch.Tensor) -> torch.Tensor:
    n = spatial_rot90.shape[0]
    identity = torch.eye(n, device=spatial_rot90.device, dtype=spatial_rot90.dtype)
    return ((spatial_rot90.T @ spatial_rot90 - identity).pow(2).mean() + (spatial_flip_h.T @ spatial_flip_h - identity).pow(2).mean())


def combined_law_metrics(spatial_maps: Mapping[str, torch.Tensor], channel_maps: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    spatial = {
        "spatial_rot180_vs_rot90_rot90": frobenius_relative_error(spatial_maps["rot180"], spatial_maps["rot90"] @ spatial_maps["rot90"]),
        "spatial_rot270_vs_rot90_rot180": frobenius_relative_error(spatial_maps["rot270"], spatial_maps["rot90"] @ spatial_maps["rot180"]),
        "spatial_rot90_cycle4_vs_identity": frobenius_relative_error(spatial_maps["identity"], spatial_maps["rot90"] @ spatial_maps["rot90"] @ spatial_maps["rot90"] @ spatial_maps["rot90"]),
        "spatial_flip_h_square_vs_identity": frobenius_relative_error(spatial_maps["identity"], spatial_maps["flip_h"] @ spatial_maps["flip_h"]),
        "spatial_flip_v_square_vs_identity": frobenius_relative_error(spatial_maps["identity"], spatial_maps["flip_v"] @ spatial_maps["flip_v"]),
    }
    channel = {
        "channel_rot180_vs_rot90_rot90": frobenius_relative_error(channel_maps["rot180"], channel_maps["rot90"] @ channel_maps["rot90"]),
        "channel_rot270_vs_rot90_rot180": frobenius_relative_error(channel_maps["rot270"], channel_maps["rot90"] @ channel_maps["rot180"]),
        "channel_rot90_cycle4_vs_identity": frobenius_relative_error(channel_maps["identity"], channel_maps["rot90"] @ channel_maps["rot90"] @ channel_maps["rot90"] @ channel_maps["rot90"]),
        "channel_flip_h_square_vs_identity": frobenius_relative_error(channel_maps["identity"], channel_maps["flip_h"] @ channel_maps["flip_h"]),
        "channel_flip_v_square_vs_identity": frobenius_relative_error(channel_maps["identity"], channel_maps["flip_v"] @ channel_maps["flip_v"]),
    }
    spatial_mean = float(np.mean(list(spatial.values())))
    channel_mean = float(np.mean(list(channel.values())))
    return {
        **spatial,
        **channel,
        "spatial_mean_composition_error": spatial_mean,
        "channel_mean_composition_error": channel_mean,
        "mean_composition_error": max(spatial_mean, channel_mean),
    }


@torch.no_grad()
def evaluate(
    data: Mapping[str, torch.Tensor],
    spatial_maps: Mapping[str, torch.Tensor],
    channel_maps: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, dict]:
    z = data["z"].to(device=device, dtype=torch.float32)
    z_tokens = tokens(z)
    metrics = {
        "identity": {"err_p": 0.0, "err_pc": 0.0, "ratio_pc_over_p": None, "gain": None}
    }
    for g in NON_IDENTITY_TRANSFORMS:
        target = data[g].to(device=device, dtype=torch.float32)
        target_tokens = tokens(target)
        fixed = tokens(apply_spatial_operator(z, g))
        pred = apply_token_channel(z_tokens, spatial_maps[g], channel_maps[g])
        err_p = torch.sqrt(relative_prediction_loss(fixed, target_tokens))
        err_pc = torch.sqrt(relative_prediction_loss(pred, target_tokens))
        ratio = err_pc / err_p.clamp_min(1e-12)
        metrics[g] = {
            "err_p": float(err_p.cpu()),
            "err_pc": float(err_pc.cpu()),
            "ratio_pc_over_p": float(ratio.cpu()),
            "gain": float((1.0 - ratio).cpu()),
        }
    return metrics


def train_candidate(
    train_data: Mapping[str, torch.Tensor],
    val_data: Mapping[str, torch.Tensor],
    fixed_spatial: Mapping[str, torch.Tensor],
    init_channel: Mapping[str, torch.Tensor],
    config: SpatialPhase1BConfig,
    law_weight: float,
    device: torch.device,
) -> dict:
    z_train = train_data["z"].to(device=device, dtype=torch.float32)
    train_tokens = tokens(z_train)
    targets = {g: tokens(train_data[g].to(device=device, dtype=torch.float32)) for g in NON_IDENTITY_TRANSFORMS}
    sample_count = z_train.shape[0]

    spatial_rot90 = fixed_spatial["rot90"].detach().clone().requires_grad_(True)
    spatial_flip_h = fixed_spatial["flip_h"].detach().clone().requires_grad_(True)
    channel_rot90 = init_channel["rot90"].detach().clone().to(device).requires_grad_(True)
    channel_flip_h = init_channel["flip_h"].detach().clone().to(device).requires_grad_(True)

    params = [spatial_rot90, spatial_flip_h, channel_rot90, channel_flip_h]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.0)
    rng = torch.Generator(device=device).manual_seed(config.seed + int(law_weight * 1000))
    log_every = max(1, config.steps // 5)

    for step in range(config.steps):
        idx = torch.randint(sample_count, (min(config.batch_images, sample_count),), device=device, generator=rng)
        z_batch = train_tokens.index_select(0, idx)
        spatial_maps = spatial_generator_maps(spatial_rot90, spatial_flip_h)
        channel_maps = generator_channel_maps(channel_rot90, channel_flip_h)

        task_terms = []
        for g in NON_IDENTITY_TRANSFORMS:
            pred = apply_token_channel(z_batch, spatial_maps[g], channel_maps[g])
            target = targets[g].index_select(0, idx)
            task_terms.append(relative_prediction_loss(pred, target))
        task = torch.stack(task_terms).mean()
        spatial_law = generator_law_loss(spatial_rot90, spatial_flip_h)
        channel_law = generator_law_loss(channel_rot90, channel_flip_h)
        spatial_anchor = anchor_loss(spatial_maps, fixed_spatial)
        channel_anchor = anchor_loss(channel_maps, init_channel)
        identity = spatial_identity_loss(spatial_rot90, spatial_flip_h)
        loss = (
            task
            + float(law_weight) * (spatial_law + channel_law)
            + config.spatial_anchor_weight * spatial_anchor
            + config.channel_anchor_weight * channel_anchor
            + config.spatial_identity_weight * identity
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        optimizer.step()

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == config.steps:
            print(
                f"law={law_weight:g} step={step + 1}/{config.steps} "
                f"loss={float(loss.detach().cpu()):.6f} task={float(task.detach().cpu()):.6f} "
                f"spatial_law={float(spatial_law.detach().cpu()):.6f} channel_law={float(channel_law.detach().cpu()):.6f}",
                flush=True,
            )

    spatial_maps = {k: v.detach() for k, v in spatial_generator_maps(spatial_rot90, spatial_flip_h).items()}
    channel_maps = {k: v.detach() for k, v in generator_channel_maps(channel_rot90, channel_flip_h).items()}
    val_metrics = evaluate(val_data, spatial_maps, channel_maps, device)
    laws = combined_law_metrics(spatial_maps, channel_maps)
    val_mean_ratio = float(np.mean([val_metrics[g]["ratio_pc_over_p"] for g in NON_IDENTITY_TRANSFORMS]))
    score = val_mean_ratio + config.score_law_weight * laws["mean_composition_error"]
    return {
        "spatial_maps": spatial_maps,
        "channel_maps": channel_maps,
        "val_metrics": val_metrics,
        "group_law": laws,
        "selection_score": float(score),
        "val_mean_ratio": val_mean_ratio,
    }


def run(config: SpatialPhase1BConfig) -> dict:
    configure_fp32()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    artifact_dir = ROOT / config.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    base = load_named_dataset(config.dataset_name, config.data_root, config.dataset_split, download=False)
    splits = split_indices(len(base), config.train_count, config.val_count, config.test_count, config.seed)
    loaders = {name: make_loader(base, indices, config) for name, indices in splits.items()}

    adapter_config = Phase1Config(rae_repo_path=config.rae_repo_path, image_size=config.image_size)
    adapter = load_model_adapter("rae_dinov2", device, adapter_config, dtype=torch.float32)
    print("collect train latents", flush=True)
    train_data = collect_latents(adapter, loaders["train"], device)
    print("collect val latents", flush=True)
    val_data = collect_latents(adapter, loaders["val"], device)
    print("collect test latents", flush=True)
    test_data = collect_latents(adapter, loaders["test"], device)

    train_device_data = {k: v.to(device) for k, v in train_data.items()}
    init_channel = fit_independent_channel_maps(train_device_data)
    _, channels, height, width = train_data["z"].shape
    fixed_spatial = {g: build_token_map(height, width, g, device) for g in TRANSFORMS}

    candidates = {}
    best_name = None
    best = None
    for law_weight in config.law_weights:
        name = f"law_{law_weight:g}"
        print(f"train spatial+channel generator {name}", flush=True)
        candidate = train_candidate(train_data, val_data, fixed_spatial, init_channel, config, law_weight, device)
        candidates[name] = {
            "selection_score": candidate["selection_score"],
            "val_mean_ratio": candidate["val_mean_ratio"],
            "group_law": candidate["group_law"],
            "val_metrics": candidate["val_metrics"],
        }
        if best is None or candidate["selection_score"] < best["selection_score"]:
            best_name = name
            best = candidate

    train_metrics = evaluate(train_data, best["spatial_maps"], best["channel_maps"], device)
    val_metrics = evaluate(val_data, best["spatial_maps"], best["channel_maps"], device)
    test_metrics = evaluate(test_data, best["spatial_maps"], best["channel_maps"], device)
    laws = combined_law_metrics(best["spatial_maps"], best["channel_maps"])
    success = assess_success(train_metrics, test_metrics, laws)
    summary = {
        "config": asdict(config),
        "model": "rae_dinov2",
        "fit_type": "phase1b_spatial_channel_generators",
        "selected_candidate": best_name,
        "candidates": candidates,
        "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        "group_law": laws,
        "success": success,
    }
    torch.save(
        {
            "model_name": "rae_dinov2",
            "fit_type": "phase1b_spatial_channel_generators",
            "spatial_maps": {k: v.detach().cpu() for k, v in best["spatial_maps"].items()},
            "channel_maps": {k: v.detach().cpu() for k, v in best["channel_maps"].items()},
            "summary": summary,
        },
        artifact_dir / "rae_dinov2_phase1b_maps.pt",
    )
    with (artifact_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"selected={best_name} mean_ratio={success['mean_ratio_pc_over_p']:.4f} "
        f"mean_law={laws['mean_composition_error']:.4f} pass={success['passes_thresholds']}",
        flush=True,
    )
    return summary


def parse_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase1B DINOv2 spatial+channel generator search.")
    parser.add_argument("--data-root", default="/data/shared")
    parser.add_argument("--dataset-name", default="caltech101")
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--val-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-dir", default="artifacts/group_action_phase1b_dinov2_spatial")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--law-weights", default="0.01,0.1,1")
    parser.add_argument("--channel-anchor-weight", type=float, default=0.01)
    parser.add_argument("--spatial-anchor-weight", type=float, default=0.1)
    parser.add_argument("--spatial-identity-weight", type=float, default=0.01)
    parser.add_argument("--batch-images", type=int, default=32)
    parser.add_argument("--score-law-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SpatialPhase1BConfig(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        steps=args.steps,
        lr=args.lr,
        law_weights=parse_tuple(args.law_weights),
        channel_anchor_weight=args.channel_anchor_weight,
        spatial_anchor_weight=args.spatial_anchor_weight,
        spatial_identity_weight=args.spatial_identity_weight,
        batch_images=args.batch_images,
        score_law_weight=args.score_law_weight,
    )
    run(config)


if __name__ == "__main__":
    main()
