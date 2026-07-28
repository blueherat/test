"""Run the preregistered no-training gate for a layerwise RAE path.

The projector is fitted only on ImageNet train images.  Validation images are
used for semantic-neighborhood, transform-correspondence, reconstruction, and
random-subspace controls.  The script supports either one process or torchrun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    P,
    configure_fp32,
    correspondence_metrics,
    extract_vit_stage_latents,
    load_named_dataset,
    pick_dataset_images,
    split_indices,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.rae_layerwise_path import (  # noqa: E402
    DetailSubspace,
    MiddleFinalCovariance,
    fit_detail_subspace,
    random_detail_basis,
    split_semantic_detail,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path"


@dataclass(frozen=True)
class GateConfig:
    dataset_path: str = "/data/shared/imagenet-1k"
    train_count: int = 1024
    val_count: int = 256
    batch_size: int = 8
    train_seed: int = 0
    val_seed: int = 10_000
    middle_index: int = 9
    ranks: tuple[int, ...] = (16, 32, 64, 128)
    ridge: float = 1e-3
    transforms: tuple[str, ...] = ("flip_h", "rot90")
    neighborhood_k: int = 10
    random_seed: int = 202_607_18
    rae_repo_path: str = "external/RAE"
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str = ""
    enable_lpips: bool = True


def setup_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world_size, device


def configure_numerics(seed: int) -> None:
    configure_fp32()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def shard(values: list[int], rank: int, world_size: int) -> list[int]:
    return values[rank::world_size]


def batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def sample_label(dataset, index: int) -> int:
    sample = dataset[int(index)]
    if isinstance(sample, (tuple, list)) and len(sample) > 1:
        label = sample[1]
        if isinstance(label, torch.Tensor):
            return int(label.item())
        return int(label)
    return -1


def latent_image_feature(latent: torch.Tensor) -> torch.Tensor:
    """A compact semantic-neighborhood feature retaining mean and dispersion."""

    mean = latent.mean(dim=(-2, -1))
    std = latent.std(dim=(-2, -1), unbiased=False)
    return torch.cat([mean, std], dim=1)


def neighborhood_statistics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate features must share [N,D] shape")
    count = len(reference)
    actual_k = min(int(k), count - 1)
    if actual_k <= 0:
        raise ValueError("at least two validation samples are required")
    reference = torch.nn.functional.normalize(reference.float(), dim=1)
    candidate = torch.nn.functional.normalize(candidate.float(), dim=1)
    ref_similarity = reference @ reference.T
    cand_similarity = candidate @ candidate.T
    diagonal = torch.arange(count)
    ref_similarity[diagonal, diagonal] = -torch.inf
    cand_similarity[diagonal, diagonal] = -torch.inf
    ref_neighbors = ref_similarity.topk(actual_k, dim=1).indices
    cand_neighbors = cand_similarity.topk(actual_k, dim=1).indices
    overlap = []
    for ref_row, cand_row in zip(ref_neighbors, cand_neighbors):
        overlap.append(torch.isin(cand_row, ref_row).float().mean())
    reference_purity = (labels[ref_neighbors] == labels[:, None]).float().mean()
    candidate_purity = (labels[cand_neighbors] == labels[:, None]).float().mean()
    return {
        "neighborhood_recall": float(torch.stack(overlap).mean()),
        "reference_label_purity": float(reference_purity),
        "candidate_label_purity": float(candidate_purity),
        "label_purity_ratio": float(candidate_purity / reference_purity.clamp_min(1e-12)),
    }


def reduce_moments(moments: MiddleFinalCovariance, world_size: int) -> None:
    if world_size <= 1:
        return
    for value in (moments.middle_gram, moments.middle_final, moments.final_gram):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    count = torch.tensor([moments.token_count], device=moments.middle_gram.device, dtype=torch.int64)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    moments.token_count = int(count.item())


def gather_objects(value: Any, world_size: int) -> list[Any]:
    if world_size <= 1:
        return [value]
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def make_lpips(device: torch.device, enabled: bool):
    if not enabled:
        return None
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        reduction="none",
        normalize=False,
    ).to(device)
    metric.requires_grad_(False).eval()
    return metric


def lpips_values(metric, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if metric is None:
        return torch.full((len(prediction),), math.nan, device=prediction.device)
    value = metric(prediction.clamp(-1, 1), target.clamp(-1, 1)).reshape(-1)
    metric.reset()
    return value


def fit_subspaces(
    adapter,
    dataset,
    indices: list[int],
    config: GateConfig,
    device: torch.device,
    world_size: int,
) -> dict[int, DetailSubspace]:
    moments = MiddleFinalCovariance.zeros(768, device=device, dtype=torch.float32)
    local = 0
    for index_batch in batches(indices, config.batch_size):
        images, _ = pick_dataset_images(
            dataset,
            indices=index_batch,
            image_size=256,
        )
        stages = extract_vit_stage_latents(
            adapter,
            images.to(device=device, dtype=torch.float32),
            hidden_indices=(config.middle_index,),
        )
        moments.update(stages[f"hidden_{config.middle_index}"], stages["rae_normalized"])
        local += len(index_batch)
        if local % max(1, config.batch_size * 16) == 0 or local == len(indices):
            print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] fit {local}/{len(indices)}", flush=True)
    reduce_moments(moments, world_size)
    cpu_moments = MiddleFinalCovariance(
        moments.middle_gram.detach().double().cpu(),
        moments.middle_final.detach().double().cpu(),
        moments.final_gram.detach().double().cpu(),
        moments.token_count,
    )
    return {
        int(rank): fit_detail_subspace(cpu_moments, int(rank), ridge=config.ridge)
        for rank in config.ranks
    }


@torch.no_grad()
def evaluate_validation(
    adapter,
    dataset,
    indices: list[int],
    subspaces: dict[int, DetailSubspace],
    config: GateConfig,
    device: torch.device,
) -> dict[str, Any]:
    metric_sums: dict[tuple[int, str, str, str], float] = defaultdict(float)
    metric_counts: dict[tuple[int, str, str, str], int] = defaultdict(int)
    recon_sums: dict[tuple[int, str], float] = defaultdict(float)
    feature_rows: dict[int, dict[str, list[torch.Tensor]]] = {
        rank: {"full": [], "semantic": [], "random_semantic": []}
        for rank in subspaces
    }
    labels: list[int] = []
    lpips_metric = make_lpips(device, config.enable_lpips)
    random_bases = {
        rank: random_detail_basis(768, rank, seed=config.random_seed + rank)
        for rank in subspaces
    }

    done = 0
    for index_batch in batches(indices, config.batch_size):
        images_cpu, _ = pick_dataset_images(dataset, indices=index_batch, image_size=256)
        images = images_cpu.to(device=device, dtype=torch.float32)
        full = adapter.encode(images)
        full_reconstruction = adapter.decode(full)
        labels.extend(sample_label(dataset, index) for index in index_batch)
        transformed = {
            transform: adapter.encode(P(images, transform))
            for transform in config.transforms
        }
        batch = len(images)
        for rank, subspace in subspaces.items():
            semantic, detail = split_semantic_detail(full, subspace)
            random_semantic, random_detail = split_semantic_detail(full, random_bases[rank])
            feature_rows[rank]["full"].append(latent_image_feature(full).cpu())
            feature_rows[rank]["semantic"].append(latent_image_feature(semantic).cpu())
            feature_rows[rank]["random_semantic"].append(
                latent_image_feature(random_semantic).cpu()
            )

            semantic_reconstruction = adapter.decode(semantic)
            random_semantic_reconstruction = adapter.decode(random_semantic)
            values = {
                "full_l1": (full_reconstruction - images).abs().flatten(1).mean(1),
                "semantic_l1": (semantic_reconstruction - images).abs().flatten(1).mean(1),
                "random_semantic_l1": (
                    random_semantic_reconstruction - images
                ).abs().flatten(1).mean(1),
                "full_lpips": lpips_values(lpips_metric, full_reconstruction, images),
                "semantic_lpips": lpips_values(lpips_metric, semantic_reconstruction, images),
                "random_semantic_lpips": lpips_values(
                    lpips_metric, random_semantic_reconstruction, images
                ),
            }
            for name, value in values.items():
                finite = value[torch.isfinite(value)]
                if len(finite):
                    recon_sums[(rank, name)] += float(finite.sum().cpu())
                    recon_sums[(rank, f"{name}_count")] += int(len(finite))

            for transform, target_full in transformed.items():
                target_semantic, target_detail = split_semantic_detail(target_full, subspace)
                target_random_semantic, target_random_detail = split_semantic_detail(
                    target_full, random_bases[rank]
                )
                components = {
                    "full": (full, target_full),
                    "semantic": (semantic, target_semantic),
                    "detail": (detail, target_detail),
                    "random_semantic": (random_semantic, target_random_semantic),
                    "random_detail": (random_detail, target_random_detail),
                }
                for component, (base_value, target_value) in components.items():
                    metrics = correspondence_metrics(
                        base_value,
                        target_value,
                        transform,
                        center="sample",
                    )
                    for metric_name, metric_value in metrics.items():
                        key = (rank, transform, component, metric_name)
                        metric_sums[key] += float(metric_value) * batch
                        metric_counts[key] += batch
        done += batch
        if done % max(1, config.batch_size * 8) == 0 or done == len(indices):
            print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] val {done}/{len(indices)}", flush=True)

    return {
        "metric_sums": dict(metric_sums),
        "metric_counts": dict(metric_counts),
        "recon_sums": dict(recon_sums),
        "features": {
            rank: {name: torch.cat(rows).numpy() for name, rows in values.items()}
            for rank, values in feature_rows.items()
        },
        "labels": np.asarray(labels, dtype=np.int64),
    }


def merge_validation(
    shards: list[dict[str, Any]],
    subspaces: dict[int, DetailSubspace],
    config: GateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sums: dict[tuple[int, str, str, str], float] = defaultdict(float)
    counts: dict[tuple[int, str, str, str], int] = defaultdict(int)
    reconstruction: dict[tuple[int, str], float] = defaultdict(float)
    for result in shards:
        for key, value in result["metric_sums"].items():
            sums[key] += float(value)
        for key, value in result["metric_counts"].items():
            counts[key] += int(value)
        for key, value in result["recon_sums"].items():
            reconstruction[key] += float(value)

    metric_rows = [
        {
            "rank": rank,
            "transform": transform,
            "component": component,
            "metric": metric,
            "value": value / counts[(rank, transform, component, metric)],
            "n": counts[(rank, transform, component, metric)],
        }
        for (rank, transform, component, metric), value in sums.items()
    ]
    metric_frame = pd.DataFrame(metric_rows)
    recon_rows = []
    neighborhood: dict[int, dict[str, float]] = {}
    labels = torch.from_numpy(np.concatenate([result["labels"] for result in shards]))
    for rank, subspace in subspaces.items():
        features = {
            name: torch.from_numpy(
                np.concatenate([result["features"][rank][name] for result in shards])
            )
            for name in ("full", "semantic", "random_semantic")
        }
        neighborhood[rank] = neighborhood_statistics(
            features["full"],
            features["semantic"],
            labels,
            config.neighborhood_k,
        )
        random_stats = neighborhood_statistics(
            features["full"],
            features["random_semantic"],
            labels,
            config.neighborhood_k,
        )
        neighborhood[rank].update(
            {f"random_{name}": value for name, value in random_stats.items()}
        )
        row: dict[str, Any] = {
            "rank": rank,
            "explained_predictable_fraction": subspace.explained_predictable_fraction,
            "explained_final_fraction": subspace.explained_final_fraction,
            **neighborhood[rank],
        }
        for name in (
            "full_l1",
            "semantic_l1",
            "random_semantic_l1",
            "full_lpips",
            "semantic_lpips",
            "random_semantic_lpips",
        ):
            denominator = reconstruction[(rank, f"{name}_count")]
            row[name] = reconstruction[(rank, name)] / denominator if denominator else math.nan
        row["l1_recovery"] = 1.0 - row["full_l1"] / max(row["semantic_l1"], 1e-12)
        row["lpips_recovery"] = (
            1.0 - row["full_lpips"] / max(row["semantic_lpips"], 1e-12)
            if math.isfinite(row["full_lpips"]) and math.isfinite(row["semantic_lpips"])
            else math.nan
        )
        recon_rows.append(row)
    return metric_frame, pd.DataFrame(recon_rows), {str(k): v for k, v in neighborhood.items()}


def metric_value(
    frame: pd.DataFrame,
    rank: int,
    transform: str,
    component: str,
    metric: str,
) -> float:
    selected = frame[
        frame["rank"].eq(rank)
        & frame["transform"].eq(transform)
        & frame["component"].eq(component)
        & frame["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise KeyError((rank, transform, component, metric))
    return float(selected.iloc[0]["value"])


def acceptance_table(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    config: GateConfig,
) -> pd.DataFrame:
    rows = []
    for _, values in summary.iterrows():
        rank = int(values["rank"])
        geometry_transforms = []
        random_advantages = []
        for transform in config.transforms:
            sem_error = metric_value(metrics, rank, transform, "semantic", "direct_error")
            detail_error = metric_value(metrics, rank, transform, "detail", "direct_error")
            sem_cosine = metric_value(metrics, rank, transform, "semantic", "mean_diag_cosine")
            detail_cosine = metric_value(metrics, rank, transform, "detail", "mean_diag_cosine")
            random_error = metric_value(metrics, rank, transform, "random_detail", "direct_error")
            random_cosine = metric_value(metrics, rank, transform, "random_detail", "mean_diag_cosine")
            geometry_transforms.append(
                detail_error <= 0.9 * sem_error or detail_cosine >= sem_cosine + 0.05
            )
            random_advantages.append(
                detail_error < random_error or detail_cosine > random_cosine
            )
        semantic_pass = float(values["neighborhood_recall"]) >= 0.95
        geometry_pass = any(geometry_transforms)
        reconstruction_pass = float(values["full_l1"]) < float(values["semantic_l1"])
        if math.isfinite(float(values["full_lpips"])):
            reconstruction_pass = reconstruction_pass and (
                float(values["full_lpips"]) < float(values["semantic_lpips"])
            )
        random_control_pass = any(random_advantages)
        rows.append(
            {
                "rank": rank,
                "semantic_pass": semantic_pass,
                "geometry_pass": geometry_pass,
                "reconstruction_pass": reconstruction_pass,
                "random_control_pass": random_control_pass,
                "gate_pass": semantic_pass
                and geometry_pass
                and reconstruction_pass
                and random_control_pass,
            }
        )
    return pd.DataFrame(rows)


def build_run_dir(config: GateConfig) -> Path:
    name = config.run_name.strip() or time.strftime("gate1_%Y%m%d_%H%M%S")
    path = Path(config.output_root).expanduser() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(config: GateConfig) -> dict[str, Any] | None:
    rank, world_size, device = setup_distributed()
    configure_numerics(config.train_seed + rank)
    train_dataset = load_named_dataset(
        "imagenet_parquet", "/data/shared", split="train", dataset_path=config.dataset_path
    )
    val_dataset = load_named_dataset(
        "imagenet_parquet", "/data/shared", split="validation", dataset_path=config.dataset_path
    )
    train_indices = split_indices(len(train_dataset), config.train_count, config.train_seed)
    val_indices = split_indices(len(val_dataset), config.val_count, config.val_seed)
    local_train = shard(train_indices, rank, world_size)
    local_val = shard(val_indices, rank, world_size)
    adapter = load_rae_adapter(
        "rae_dinov2",
        repo_path=config.rae_repo_path,
        device=device,
        dtype=torch.float32,
        auto_clone=False,
        auto_download=False,
    )
    adapter.model.requires_grad_(False).eval()
    subspaces = fit_subspaces(
        adapter,
        train_dataset,
        local_train,
        config,
        device,
        world_size,
    )
    validation = evaluate_validation(
        adapter,
        val_dataset,
        local_val,
        subspaces,
        config,
        device,
    )
    gathered = gather_objects(validation, world_size)
    payload = None
    if rank == 0:
        run_dir = build_run_dir(config)
        metrics, summary, neighborhood = merge_validation(gathered, subspaces, config)
        acceptance = acceptance_table(metrics, summary, config)
        torch.save(
            {
                "subspaces": {
                    key: {
                        "basis": value.basis,
                        "explained_predictable_fraction": value.explained_predictable_fraction,
                        "explained_final_fraction": value.explained_final_fraction,
                        "ridge_scale": value.ridge_scale,
                        "token_count": value.token_count,
                    }
                    for key, value in subspaces.items()
                },
                "config": asdict(config),
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
            run_dir / "subspaces.pt",
        )
        metrics.to_csv(run_dir / "correspondence.csv", index=False)
        summary.to_csv(run_dir / "summary.csv", index=False)
        acceptance.to_csv(run_dir / "acceptance.csv", index=False)
        payload = {
            "run_dir": str(run_dir),
            "config": asdict(config),
            "world_size": world_size,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "neighborhood": neighborhood,
            "summary": summary.to_dict(orient="records"),
            "acceptance": acceptance.to_dict(orient="records"),
            "any_rank_passed": bool(acceptance["gate_pass"].any()),
        }
        (run_dir / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return payload


def parse_args() -> GateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=GateConfig.dataset_path)
    parser.add_argument("--train-count", type=int, default=GateConfig.train_count)
    parser.add_argument("--val-count", type=int, default=GateConfig.val_count)
    parser.add_argument("--batch-size", type=int, default=GateConfig.batch_size)
    parser.add_argument("--train-seed", type=int, default=GateConfig.train_seed)
    parser.add_argument("--val-seed", type=int, default=GateConfig.val_seed)
    parser.add_argument("--middle-index", type=int, default=GateConfig.middle_index)
    parser.add_argument("--ranks", nargs="+", type=int, default=list(GateConfig.ranks))
    parser.add_argument("--ridge", type=float, default=GateConfig.ridge)
    parser.add_argument("--transforms", nargs="+", default=list(GateConfig.transforms))
    parser.add_argument("--neighborhood-k", type=int, default=GateConfig.neighborhood_k)
    parser.add_argument("--random-seed", type=int, default=GateConfig.random_seed)
    parser.add_argument("--rae-repo-path", default=GateConfig.rae_repo_path)
    parser.add_argument("--output-root", default=GateConfig.output_root)
    parser.add_argument("--run-name", default=GateConfig.run_name)
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()
    return GateConfig(
        dataset_path=args.dataset_path,
        train_count=args.train_count,
        val_count=args.val_count,
        batch_size=args.batch_size,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        middle_index=args.middle_index,
        ranks=tuple(args.ranks),
        ridge=args.ridge,
        transforms=tuple(args.transforms),
        neighborhood_k=args.neighborhood_k,
        random_seed=args.random_seed,
        rae_repo_path=args.rae_repo_path,
        output_root=args.output_root,
        run_name=args.run_name,
        enable_lpips=not args.no_lpips,
    )


if __name__ == "__main__":
    run(parse_args())
