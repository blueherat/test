"""Distributed layerwise audit of genuine spatial correspondence in RAE-DINOv2."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    P,
    configure_fp32,
    extract_vit_stage_latents,
    grid_coords,
    load_named_dataset,
    pick_dataset_images,
    relative_token_error,
    split_indices,
    token_rows,
    vit_pos_intervention_table,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402


@dataclass(frozen=True)
class CorrespondenceStudyConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    dataset_split: str = "test"
    image_size: int = 256
    count: int = 5196
    batch_size: int = 8
    seed: int = 0
    transforms: tuple[str, ...] = ("rot90", "flip_h")
    hidden_indices: tuple[int, ...] = tuple(range(13))
    random_permutations: int = 4
    position_count: int = 256
    device: str = "cuda:0"
    rae_repo_path: str = "external/RAE"
    output_root: Path = Path.home() / "data/eqvae/artifacts/layerwise_correspondence"
    run_name: str = "dinov2_imagenet_test_n5196"


def spatial_residual(latent: torch.Tensor) -> torch.Tensor:
    return latent - latent.mean(dim=(-2, -1), keepdim=True)


def residual_energy_fraction(latent: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    residual = spatial_residual(latent)
    numerator = residual.flatten(1).square().sum(dim=1)
    denominator = latent.flatten(1).square().sum(dim=1).clamp_min(float(eps))
    return numerator / denominator


def residual_spatial_gram(latent: torch.Tensor) -> torch.Tensor:
    residual = spatial_residual(latent)
    rows = residual.permute(0, 2, 3, 1).reshape(
        latent.shape[0], latent.shape[-2] * latent.shape[-1], latent.shape[1]
    )
    return torch.bmm(rows, rows.transpose(1, 2)).sum(dim=0) / latent.shape[1]


def effective_rank_from_gram(gram: torch.Tensor, eps: float = 1e-12) -> float:
    eigenvalues = torch.linalg.eigvalsh(gram.detach().double().cpu()).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) <= float(eps):
        return 0.0
    return float((total.square() / eigenvalues.square().sum().clamp_min(float(eps))).item())


def _random_permutations(tokens: int, count: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    permutations = []
    identity = torch.arange(tokens)
    while len(permutations) < int(count):
        permutation = torch.randperm(tokens, generator=generator)
        if not torch.equal(permutation, identity):
            permutations.append(permutation.to(device))
    return permutations


def _permute_spatial(latent: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latent.shape
    flat = latent.reshape(batch, channels, height * width)
    return flat.index_select(-1, permutation).reshape(batch, channels, height, width)


def layer_pair_metrics(
    base: torch.Tensor,
    target: torch.Tensor,
    transform: str,
    permutations: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return per-sample correspondence and random-permutation controls."""

    prediction = P(base, transform)
    direct = relative_token_error(target, prediction, center="sample")
    random_errors = torch.stack(
        [
            relative_token_error(target, _permute_spatial(prediction, permutation), center="sample")
            for permutation in permutations
        ],
        dim=0,
    ).mean(dim=0)

    x = F.normalize(token_rows(prediction, center="sample"), dim=-1)
    y = F.normalize(token_rows(target, center="sample"), dim=-1)
    similarity = torch.bmm(x, y.transpose(1, 2))
    batch, tokens, _ = similarity.shape
    side = int(math.sqrt(tokens))
    if side * side != tokens:
        raise ValueError(f"token count must form a square grid, got {tokens}")
    coords = grid_coords(side, side, similarity.device)
    best = similarity.argmax(dim=-1)
    expected = torch.arange(tokens, device=similarity.device).view(1, tokens).expand(batch, tokens)
    displacement = torch.linalg.norm(coords[best] - coords[expected], dim=-1)

    random_exact = []
    random_within = []
    for permutation in permutations:
        random_expected = permutation.view(1, tokens).expand(batch, tokens)
        random_displacement = torch.linalg.norm(
            coords[best] - coords[random_expected],
            dim=-1,
        )
        random_exact.append((best == random_expected).float().mean(dim=1))
        random_within.append((random_displacement <= 1.0).float().mean(dim=1))

    return {
        "residual_direct_error": direct,
        "random_permutation_error": random_errors,
        "diag_cosine": similarity.diagonal(dim1=-2, dim2=-1).mean(dim=1),
        "best_cosine": similarity.max(dim=-1).values.mean(dim=1),
        "best_displacement": displacement.mean(dim=1),
        "exact_match_rate": (best == expected).float().mean(dim=1),
        "within_1_rate": (displacement <= 1.0).float().mean(dim=1),
        "random_exact_match_rate": torch.stack(random_exact).mean(dim=0),
        "random_within_1_rate": torch.stack(random_within).mean(dim=0),
    }


class ScalarAccumulator:
    def __init__(self, device: torch.device):
        self.device = device
        self.values: dict[tuple[str, str, str, str], torch.Tensor] = {}

    def add(self, key: tuple[str, str, str, str], values: torch.Tensor) -> None:
        values = values.detach().to(device=self.device, dtype=torch.float64).reshape(-1)
        if key not in self.values:
            self.values[key] = torch.zeros(2, device=self.device, dtype=torch.float64)
        self.values[key][0] += values.sum()
        self.values[key][1] += values.numel()

    def reduce_rows(self) -> list[dict[str, float | str | int]]:
        keys = sorted(self.values)
        packed = torch.stack([self.values[key] for key in keys])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        rows = []
        for key, (total, count) in zip(keys, packed):
            scope, transform, layer, metric = key
            rows.append(
                {
                    "scope": scope,
                    "transform": transform,
                    "layer": layer,
                    "metric": metric,
                    "value": float((total / count.clamp_min(1.0)).cpu()),
                    "n": int(count.cpu()),
                }
            )
        return rows


def _layer_order(stages: dict[str, torch.Tensor], hidden_indices: Sequence[int]) -> list[str]:
    proposed = ["patch_pre_pos", "post_pos"]
    proposed.extend(f"hidden_{index}" for index in hidden_indices)
    proposed.append("final_raw")
    return [layer for layer in proposed if layer in stages]


def _setup_device(requested: str) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
        return dist.get_rank(), world_size, device
    device = torch.device(requested if torch.cuda.is_available() or requested == "cpu" else "cpu")
    return 0, 1, device


def _reduce_grams(
    grams: dict[str, torch.Tensor],
    counts: dict[str, int],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    layers = sorted(grams)
    for layer in layers:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grams[layer], op=dist.ReduceOp.SUM)
        count = torch.tensor(float(counts[layer]), device=grams[layer].device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        counts[layer] = int(count.item())
    return grams, counts


def _metric_value(rows: pd.DataFrame, transform: str, layer: str, metric: str) -> float:
    match = rows[
        (rows["scope"] == "pair")
        & (rows["transform"] == transform)
        & (rows["layer"] == layer)
        & (rows["metric"] == metric)
    ]
    if len(match) != 1:
        raise RuntimeError(f"expected one metric row for {transform}/{layer}/{metric}, got {len(match)}")
    return float(match.iloc[0]["value"])


def evaluate_acceptance(rows: pd.DataFrame, transforms: Sequence[str]) -> dict[str, object]:
    checks = {}
    for transform in transforms:
        patch_error = _metric_value(rows, transform, "patch_pre_pos", "residual_direct_error")
        final_error = _metric_value(rows, transform, "final_raw", "residual_direct_error")
        random_error = _metric_value(rows, transform, "final_raw", "random_permutation_error")
        exact = _metric_value(rows, transform, "final_raw", "exact_match_rate")
        random_exact = _metric_value(rows, transform, "final_raw", "random_exact_match_rate")
        decline = 1.0 - final_error / max(patch_error, 1e-12)
        aligned_advantage = 1.0 - final_error / max(random_error, 1e-12)
        correspondence_ratio = exact / max(random_exact, 1e-12)
        checks[transform] = {
            "residual_error_decline": decline,
            "residual_error_decline_passed": decline >= 0.15,
            "aligned_vs_random_advantage": aligned_advantage,
            "aligned_vs_random_passed": aligned_advantage >= 0.15,
            "exact_correspondence_over_random": correspondence_ratio,
            "correspondence_passed": correspondence_ratio >= 2.0,
        }
        checks[transform]["passed"] = all(
            checks[transform][name]
            for name in (
                "residual_error_decline_passed",
                "aligned_vs_random_passed",
                "correspondence_passed",
            )
        )
    return {
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
        "interpretation": (
            "deep-layer alignment survives residual, random-permutation and correspondence controls"
            if all(check["passed"] for check in checks.values())
            else "at least one transform fails the preregistered genuine-correspondence gate"
        ),
    }


def _plot(rows: pd.DataFrame, layer_stats: pd.DataFrame, path: Path) -> None:
    def layer_key(name: str) -> tuple[int, int]:
        if name == "patch_pre_pos":
            return (0, 0)
        if name == "post_pos":
            return (1, 0)
        if name.startswith("hidden_"):
            return (2, int(name.removeprefix("hidden_")))
        if name == "final_raw":
            return (3, 0)
        return (4, 0)

    layers = sorted(
        set(rows[rows["scope"] == "pair"]["layer"].tolist()),
        key=layer_key,
    )
    x = np.arange(len(layers))
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True)
    for transform, group in rows[rows["scope"] == "pair"].groupby("transform", sort=False):
        pivot = group.pivot(index="layer", columns="metric", values="value").reindex(layers)
        axes[0, 0].plot(x, pivot["residual_direct_error"], marker="o", label=f"{transform}: aligned")
        axes[0, 0].plot(x, pivot["random_permutation_error"], linestyle="--", label=f"{transform}: random")
        axes[0, 1].plot(x, pivot["exact_match_rate"], marker="o", label=f"{transform}: exact")
        axes[0, 1].plot(x, pivot["random_exact_match_rate"], linestyle="--", label=f"{transform}: random")
        axes[1, 0].plot(x, pivot["diag_cosine"], marker="o", label=transform)
    stats = layer_stats.set_index("layer").reindex(layers)
    axes[1, 1].plot(x, stats["residual_energy_fraction"], marker="o", label="residual energy fraction")
    axes[1, 1].plot(x, stats["spatial_effective_rank_normalized"], marker="s", label="spatial effective rank / tokens")
    titles = (
        "Residual direct error vs random permutation",
        "Exact token correspondence vs random",
        "Diagonal token cosine",
        "Spatial residual non-collapse controls",
    )
    for axis, title in zip(axes.reshape(-1), titles):
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        axis.set_xticks(x)
        axis.set_xticklabels([name.replace("hidden_", "h") for name in layers], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_study(config: CorrespondenceStudyConfig) -> dict[str, object]:
    configure_fp32()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank, world_size, device = _setup_device(config.device)
    dataset = load_named_dataset(
        config.dataset_name,
        config.data_root,
        split=config.dataset_split,
        dataset_path=config.dataset_path,
    )
    indices = split_indices(len(dataset), config.count, config.seed)
    local_indices = indices[rank::world_size]
    position_indices = indices[: min(config.position_count, len(indices))][rank::world_size]
    output_dir = config.output_root / config.run_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(config)
        payload["output_root"] = str(config.output_root)
        payload["indices"] = indices
        payload["world_size"] = world_size
        (output_dir / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    adapter = load_rae_adapter(
        "rae_dinov2",
        repo_path=config.rae_repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=False,
        auto_download=False,
    )
    adapter.model.requires_grad_(False)
    accumulator = ScalarAccumulator(device)
    grams: dict[str, torch.Tensor] = {}
    gram_counts: dict[str, int] = {}
    permutation_cache: dict[tuple[str, int], list[torch.Tensor]] = {}
    processed = 0
    started = time.time()

    for start in range(0, len(local_indices), config.batch_size):
        batch_indices = local_indices[start : start + config.batch_size]
        images, _ = pick_dataset_images(
            dataset,
            indices=batch_indices,
            count=len(batch_indices),
            image_size=config.image_size,
        )
        images = images.to(device=device, dtype=torch.float32)
        base_stages = extract_vit_stage_latents(
            adapter,
            images,
            hidden_indices=config.hidden_indices,
            include_rae_normalized=False,
        )
        layers = _layer_order(base_stages, config.hidden_indices)
        for layer in layers:
            latent = base_stages[layer]
            accumulator.add(
                ("layer", "identity", layer, "residual_energy_fraction"),
                residual_energy_fraction(latent),
            )
            gram = residual_spatial_gram(latent).to(dtype=torch.float64)
            if layer not in grams:
                grams[layer] = torch.zeros_like(gram)
                gram_counts[layer] = 0
            grams[layer] += gram
            gram_counts[layer] += len(latent)

        for transform_index, transform in enumerate(config.transforms):
            target_stages = extract_vit_stage_latents(
                adapter,
                P(images, transform),
                hidden_indices=config.hidden_indices,
                include_rae_normalized=False,
            )
            for layer in layers:
                tokens = base_stages[layer].shape[-1] * base_stages[layer].shape[-2]
                cache_key = (transform, tokens)
                if cache_key not in permutation_cache:
                    permutation_cache[cache_key] = _random_permutations(
                        tokens,
                        config.random_permutations,
                        config.seed + 1009 * (transform_index + 1) + tokens,
                        device,
                    )
                metrics = layer_pair_metrics(
                    base_stages[layer],
                    target_stages[layer],
                    transform,
                    permutation_cache[cache_key],
                )
                for metric, values in metrics.items():
                    accumulator.add(("pair", transform, layer, metric), values)
        processed += len(batch_indices)
        if rank == 0 and (processed % max(8 * config.batch_size, 1) == 0 or processed == len(local_indices)):
            print(
                f"[rank 0] {processed}/{len(local_indices)} local images, elapsed={(time.time()-started)/60:.1f} min",
                flush=True,
            )

    for start in range(0, len(position_indices), config.batch_size):
        batch_indices = position_indices[start : start + config.batch_size]
        images, _ = pick_dataset_images(
            dataset,
            indices=batch_indices,
            count=len(batch_indices),
            image_size=config.image_size,
        )
        position_rows = vit_pos_intervention_table(
            adapter,
            images.to(device=device, dtype=torch.float32),
            transforms=config.transforms,
            center="sample",
        )
        for row in position_rows:
            values = torch.full(
                (len(batch_indices),),
                float(row["error"]),
                device=device,
                dtype=torch.float64,
            )
            accumulator.add(("position", str(row["transform"]), str(row["mode"]), "error"), values)

    metric_rows = accumulator.reduce_rows()
    grams, gram_counts = _reduce_grams(grams, gram_counts)
    result: dict[str, object] = {"rank": rank, "world_size": world_size}
    if rank == 0:
        metric_frame = pd.DataFrame(metric_rows)
        layer_rows = []
        energy_rows = metric_frame[
            (metric_frame["scope"] == "layer")
            & (metric_frame["metric"] == "residual_energy_fraction")
        ].set_index("layer")
        for layer in _layer_order(base_stages, config.hidden_indices):
            effective_rank = effective_rank_from_gram(grams[layer])
            tokens = grams[layer].shape[0]
            layer_rows.append(
                {
                    "layer": layer,
                    "residual_energy_fraction": float(energy_rows.loc[layer, "value"]),
                    "spatial_effective_rank": effective_rank,
                    "spatial_effective_rank_normalized": effective_rank / tokens,
                    "n": gram_counts[layer],
                }
            )
        layer_frame = pd.DataFrame(layer_rows)
        acceptance = evaluate_acceptance(metric_frame, config.transforms)
        metric_path = output_dir / "metrics.csv"
        layer_path = output_dir / "layer_stats.csv"
        plot_path = output_dir / "correspondence_summary.png"
        metric_frame.to_csv(metric_path, index=False)
        layer_frame.to_csv(layer_path, index=False)
        _plot(metric_frame, layer_frame, plot_path)
        (output_dir / "acceptance.json").write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {
            "run_dir": str(output_dir),
            "metrics": str(metric_path),
            "layer_stats": str(layer_path),
            "plot": str(plot_path),
            "acceptance": acceptance,
            "world_size": world_size,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def parse_args() -> CorrespondenceStudyConfig:
    defaults = CorrespondenceStudyConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=defaults.dataset_name)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--dataset-path", default=defaults.dataset_path)
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--count", type=int, default=defaults.count)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--transforms", nargs="+", default=list(defaults.transforms))
    parser.add_argument("--hidden-indices", nargs="+", type=int, default=list(defaults.hidden_indices))
    parser.add_argument("--random-permutations", type=int, default=defaults.random_permutations)
    parser.add_argument("--position-count", type=int, default=defaults.position_count)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--rae-repo-path", default=defaults.rae_repo_path)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    args = parser.parse_args()
    return CorrespondenceStudyConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        image_size=args.image_size,
        count=args.count,
        batch_size=args.batch_size,
        seed=args.seed,
        transforms=tuple(args.transforms),
        hidden_indices=tuple(args.hidden_indices),
        random_permutations=args.random_permutations,
        position_count=args.position_count,
        device=args.device,
        rae_repo_path=args.rae_repo_path,
        output_root=args.output_root,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    run_study(parse_args())
