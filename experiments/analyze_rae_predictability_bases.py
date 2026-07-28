"""Compare absolute-energy and variance-normalized RAE predictability bases.

The original SPC basis maximizes absolute middle-predictable energy in the
final latent.  This diagnostic asks whether that basis is distinct from top
PCA after final variance is removed.  Train moments fit every basis and every
regression map; ImageNet validation moments are used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    extract_vit_stage_latents,
    load_named_dataset,
    pick_dataset_images,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.rae_layerwise_path import (  # noqa: E402
    MiddleFinalCovariance,
    fit_detail_subspace,
    fit_final_pca_subspace,
    fit_fractional_predictability_subspace,
    random_detail_basis,
    subspace_regression_metrics,
)
from experiments.run_rae_layerwise_path_gate import (  # noqa: E402
    configure_numerics,
    reduce_moments,
    setup_distributed,
)


DEFAULT_REFERENCE = (
    Path.home()
    / "data/eqvae/experiments/rae_layerwise_path"
    / "gate1_imagenet_train1024_val256_mid9/subspaces.pt"
)
DEFAULT_OUTPUT = (
    Path.home()
    / "data/eqvae/experiments/rae_spc_multiseed_v1/evaluation"
    / "predictability_basis_v1"
)


def _empty_moments(channels: int, device: torch.device) -> MiddleFinalCovariance:
    return MiddleFinalCovariance.zeros(channels, device=device, dtype=torch.float32)


def _cpu_double(moments: MiddleFinalCovariance) -> MiddleFinalCovariance:
    return MiddleFinalCovariance(
        middle_gram=moments.middle_gram.detach().double().cpu(),
        middle_final=moments.middle_final.detach().double().cpu(),
        final_gram=moments.final_gram.detach().double().cpu(),
        token_count=int(moments.token_count),
    )


def _moments_payload(moments: MiddleFinalCovariance) -> dict[str, object]:
    return {
        "middle_gram": moments.middle_gram,
        "middle_final": moments.middle_final,
        "final_gram": moments.final_gram,
        "token_count": int(moments.token_count),
    }


def _moments_from_payload(payload: dict[str, object]) -> MiddleFinalCovariance:
    return MiddleFinalCovariance(
        middle_gram=torch.as_tensor(payload["middle_gram"]).double(),
        middle_final=torch.as_tensor(payload["middle_final"]).double(),
        final_gram=torch.as_tensor(payload["final_gram"]).double(),
        token_count=int(payload["token_count"]),
    )


def _batches(values: list[tuple[int, int]], batch_size: int) -> Iterable[list[tuple[int, int]]]:
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


@torch.no_grad()
def accumulate_split_moments(
    adapter,
    dataset,
    indexed_samples: list[tuple[int, int]],
    *,
    total_count: int,
    middle_index: int,
    batch_size: int,
    device: torch.device,
    include_halves: bool,
) -> tuple[MiddleFinalCovariance, MiddleFinalCovariance | None, MiddleFinalCovariance | None]:
    full = _empty_moments(768, device)
    first = _empty_moments(768, device) if include_halves else None
    second = _empty_moments(768, device) if include_halves else None
    completed = 0
    for batch in _batches(indexed_samples, batch_size):
        ordinals = [ordinal for ordinal, _ in batch]
        indices = [index for _, index in batch]
        images, _ = pick_dataset_images(dataset, indices=indices, image_size=256)
        stages = extract_vit_stage_latents(
            adapter,
            images.to(device=device, dtype=torch.float32),
            hidden_indices=(int(middle_index),),
        )
        middle = stages[f"hidden_{int(middle_index)}"]
        final = stages["rae_normalized"]
        full.update(middle, final)
        if include_halves:
            midpoint = int(total_count) // 2
            first_mask = torch.tensor(
                [ordinal < midpoint for ordinal in ordinals], device=device
            )
            second_mask = ~first_mask
            if bool(first_mask.any()):
                assert first is not None
                first.update(middle[first_mask], final[first_mask])
            if bool(second_mask.any()):
                assert second is not None
                second.update(middle[second_mask], final[second_mask])
        completed += len(batch)
        if completed % max(1, batch_size * 16) == 0 or completed == len(indexed_samples):
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[rank {rank}] extracted {completed}/{len(indexed_samples)}",
                flush=True,
            )
    return full, first, second


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    left, _ = torch.linalg.qr(left.double(), mode="reduced")
    right, _ = torch.linalg.qr(right.double(), mode="reduced")
    cosines = torch.linalg.svdvals(left.transpose(0, 1) @ right).clamp(0, 1)
    return float(cosines.square().mean())


def _fit_named_basis(
    name: str,
    moments: MiddleFinalCovariance,
    rank: int,
    *,
    ridge: float,
    final_ridge: float,
) -> torch.Tensor:
    if name == "absolute_refit":
        return fit_detail_subspace(moments, rank, ridge=ridge).basis
    if name == "fractional":
        return fit_fractional_predictability_subspace(
            moments,
            rank,
            ridge=ridge,
            final_ridge=final_ridge,
        ).basis
    if name == "top_pca":
        return fit_final_pca_subspace(moments, rank)
    raise KeyError(name)


def analyze_moments(
    train: MiddleFinalCovariance,
    validation: MiddleFinalCovariance,
    train_first: MiddleFinalCovariance,
    train_second: MiddleFinalCovariance,
    reference_bases: dict[int, torch.Tensor],
    *,
    ranks: tuple[int, ...],
    ridge: float,
    final_ridge: float,
    random_controls: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, torch.Tensor]]]:
    rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    saved_bases: dict[int, dict[str, torch.Tensor]] = {}
    for rank in ranks:
        reference = reference_bases[int(rank)].float().cpu()
        named = {
            "reference_guided": reference,
            "absolute_refit": _fit_named_basis(
                "absolute_refit", train, rank, ridge=ridge, final_ridge=final_ridge
            ),
            "fractional": _fit_named_basis(
                "fractional", train, rank, ridge=ridge, final_ridge=final_ridge
            ),
            "top_pca": _fit_named_basis(
                "top_pca", train, rank, ridge=ridge, final_ridge=final_ridge
            ),
        }
        for control in range(int(random_controls)):
            named[f"random_{control}"] = random_detail_basis(
                train.middle_gram.shape[0],
                rank,
                seed=int(random_seed) + 1000 * int(rank) + control,
            )
        saved_bases[int(rank)] = {key: value.cpu() for key, value in named.items()}

        for name, basis in named.items():
            train_metrics = subspace_regression_metrics(
                train, train, basis, ridge=ridge
            )
            val_metrics = subspace_regression_metrics(
                train, validation, basis, ridge=ridge
            )
            row: dict[str, object] = {
                "rank": int(rank),
                "basis": name,
                "basis_family": "random" if name.startswith("random_") else name,
                "overlap_reference": subspace_overlap(basis, reference),
                "overlap_top_pca": subspace_overlap(basis, named["top_pca"]),
            }
            for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
                for key, value in metrics.items():
                    if key != "rank":
                        row[f"{prefix}_{key}"] = float(value)
            val_variance = float(val_metrics["final_variance_per_dimension"])
            for time in (0.5, 0.7, 0.85, 0.95):
                row[f"val_static_snr_t{str(time).replace('.', '')}"] = (
                    ((1.0 - time) ** 2) * val_variance / (time**2)
                )
            rows.append(row)

        for name in ("absolute_refit", "fractional", "top_pca"):
            first_basis = _fit_named_basis(
                name,
                train_first,
                rank,
                ridge=ridge,
                final_ridge=final_ridge,
            )
            second_basis = _fit_named_basis(
                name,
                train_second,
                rank,
                ridge=ridge,
                final_ridge=final_ridge,
            )
            stability_rows.append(
                {
                    "rank": int(rank),
                    "basis": name,
                    "half_split_overlap": subspace_overlap(first_basis, second_basis),
                    "first_vs_full": subspace_overlap(first_basis, named[name]),
                    "second_vs_full": subspace_overlap(second_basis, named[name]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(stability_rows), saved_bases


def aggregate_random_controls(rows: pd.DataFrame) -> pd.DataFrame:
    random_rows = rows[rows["basis_family"] == "random"]
    numeric = [
        column
        for column in random_rows.columns
        if column not in {"rank", "basis", "basis_family"}
    ]
    grouped = random_rows.groupby("rank", as_index=False)[numeric].mean()
    grouped["basis"] = "random_mean"
    grouped["basis_family"] = "random"
    return pd.concat(
        [rows[rows["basis_family"] != "random"], grouped], ignore_index=True
    )


def build_block_atlas(
    train: MiddleFinalCovariance,
    validation: MiddleFinalCovariance,
    *,
    block_rank: int,
    max_rank: int,
    ridge: float,
    final_ridge: float,
    random_controls: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    if int(max_rank) % int(block_rank):
        raise ValueError("max_rank must be divisible by block_rank")
    families = {
        "absolute": _fit_named_basis(
            "absolute_refit",
            train,
            max_rank,
            ridge=ridge,
            final_ridge=final_ridge,
        ),
        "fractional": _fit_named_basis(
            "fractional",
            train,
            max_rank,
            ridge=ridge,
            final_ridge=final_ridge,
        ),
        "pca": _fit_named_basis(
            "top_pca",
            train,
            max_rank,
            ridge=ridge,
            final_ridge=final_ridge,
        ),
    }
    blocks: dict[str, torch.Tensor] = {}
    metadata: dict[str, tuple[str, int, int, int]] = {}
    for family, basis in families.items():
        for start in range(0, int(max_rank), int(block_rank)):
            end = start + int(block_rank)
            name = f"{family}_{start:03d}_{end - 1:03d}"
            blocks[name] = basis[:, start:end].contiguous()
            metadata[name] = (family, start // int(block_rank), start, end)
    channels = int(train.middle_gram.shape[0])
    for control in range(int(random_controls)):
        name = f"random_{control}"
        blocks[name] = random_detail_basis(
            channels,
            block_rank,
            seed=int(random_seed) + 50_000 + control,
        )
        metadata[name] = ("random", -1, -1, -1)

    rows: list[dict[str, object]] = []
    for name, basis in blocks.items():
        family, block_index, start, end = metadata[name]
        train_metrics = subspace_regression_metrics(
            train, train, basis, ridge=ridge
        )
        val_metrics = subspace_regression_metrics(
            train, validation, basis, ridge=ridge
        )
        row: dict[str, object] = {
            "basis": name,
            "basis_family": family,
            "block_index": block_index,
            "direction_start": start,
            "direction_stop": end,
            "rank": int(block_rank),
        }
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metrics.items():
                if key != "rank":
                    row[f"{prefix}_{key}"] = float(value)
        val_variance = float(val_metrics["final_variance_per_dimension"])
        for time in (0.5, 0.7, 0.85, 0.95):
            row[f"val_static_snr_t{str(time).replace('.', '')}"] = (
                ((1.0 - time) ** 2) * val_variance / (time**2)
            )
        rows.append(row)
    return pd.DataFrame(rows), blocks


def plot_block_atlas(rows: pd.DataFrame, output: Path) -> None:
    colors = {"absolute": "#c84c32", "fractional": "#2678a8", "pca": "#2f855a"}
    labels = {
        "absolute": "absolute predictable energy",
        "fractional": "whitened predictability",
        "pca": "PCA",
    }
    selected = rows[rows["basis_family"] != "random"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fields = (
        ("val_r2", "validation R2"),
        ("val_final_variance_per_dimension", "final variance per dimension"),
        ("val_static_snr_t085", "static state SNR at t=0.85"),
    )
    for family in ("absolute", "fractional", "pca"):
        values = selected[selected["basis_family"] == family].sort_values("block_index")
        for axis, (field, ylabel) in zip(axes, fields):
            axis.plot(
                values["block_index"],
                values[field],
                marker="o",
                color=colors[family],
                label=labels[family],
            )
            axis.set_xlabel("rank-16 block index")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(frameon=False)
    fig.savefig(output / "predictability_block_atlas.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    for family in ("absolute", "fractional", "pca"):
        values = selected[selected["basis_family"] == family]
        axis.scatter(
            values["val_final_variance_per_dimension"],
            values["val_r2"],
            s=75,
            color=colors[family],
            label=labels[family],
        )
        for _, row in values.iterrows():
            axis.annotate(
                str(int(row["block_index"])),
                (row["val_final_variance_per_dimension"], row["val_r2"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xscale("log")
    axis.set_xlabel("validation final variance per dimension")
    axis.set_ylabel("validation R2")
    axis.set_title("Rank-16 blocks separate variance from predictability")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output / "block_variance_vs_predictability.png", dpi=180)
    plt.close(fig)


def plot_comparison(rows: pd.DataFrame, stability: pd.DataFrame, output: Path) -> None:
    display = aggregate_random_controls(rows)
    names = ["reference_guided", "fractional", "top_pca", "random_mean"]
    labels = {
        "reference_guided": "original guided",
        "fractional": "whitened predictability",
        "top_pca": "top PCA",
        "random_mean": "random mean",
    }
    colors = {
        "reference_guided": "#c84c32",
        "fractional": "#2678a8",
        "top_pca": "#2f855a",
        "random_mean": "#777777",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for name in names:
        selected = display[display["basis"] == name].sort_values("rank")
        axes[0, 0].plot(
            selected["rank"], selected["val_r2"], marker="o", label=labels[name], color=colors[name]
        )
        axes[0, 1].plot(
            selected["rank"], selected["val_final_energy_fraction"], marker="o", label=labels[name], color=colors[name]
        )
        axes[1, 0].plot(
            selected["rank"], selected["val_static_snr_t085"], marker="o", label=labels[name], color=colors[name]
        )
    for name, color in (("absolute_refit", "#c84c32"), ("fractional", "#2678a8"), ("top_pca", "#2f855a")):
        selected = stability[stability["basis"] == name].sort_values("rank")
        axes[1, 1].plot(
            selected["rank"], selected["half_split_overlap"], marker="o", label=name, color=color
        )
    axes[0, 0].set_title("Held-out middle-to-final predictability")
    axes[0, 0].set_ylabel("validation R2")
    axes[0, 1].set_title("Held-out final latent energy")
    axes[0, 1].set_ylabel("fraction of residual energy")
    axes[1, 0].set_title("Signal-to-noise ratio at t=0.85")
    axes[1, 0].set_ylabel("static state SNR")
    axes[1, 1].set_title("Basis stability across train halves")
    axes[1, 1].set_ylabel("mean squared principal cosine")
    for axis in axes.flat:
        axis.set_xlabel("rank")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output / "predictability_basis_comparison.png", dpi=180)
    plt.close(fig)

    rank = int(display["rank"].min())
    selected = display[(display["rank"] == rank) & (display["basis"].isin(names))]
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for _, row in selected.iterrows():
        name = str(row["basis"])
        axis.scatter(
            row["val_final_variance_per_dimension"],
            row["val_r2"],
            s=100,
            color=colors[name],
        )
        axis.annotate(labels[name], (row["val_final_variance_per_dimension"], row["val_r2"]), xytext=(7, 5), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_xlabel("validation final variance per active dimension (log scale)")
    axis.set_ylabel("validation R2")
    axis.set_title(f"Variance and predictability are different quantities (rank {rank})")
    axis.grid(alpha=0.25)
    fig.savefig(output / f"variance_vs_predictability_rank{rank}.png", dpi=180)
    plt.close(fig)


def build_summary(rows: pd.DataFrame, stability: pd.DataFrame) -> dict[str, object]:
    display = aggregate_random_controls(rows)
    rank = int(display["rank"].min())

    def record(name: str) -> dict[str, object]:
        row = display[(display["rank"] == rank) & (display["basis"] == name)].iloc[0]
        return {
            "val_r2": float(row["val_r2"]),
            "val_energy_fraction": float(row["val_final_energy_fraction"]),
            "val_variance_per_dimension": float(row["val_final_variance_per_dimension"]),
            "val_snr_t085": float(row["val_static_snr_t085"]),
            "overlap_top_pca": float(row["overlap_top_pca"]),
            "overlap_reference": float(row["overlap_reference"]),
        }

    stability_rank = stability[stability["rank"] == rank].set_index("basis")
    return {
        "primary_rank": rank,
        "reference_guided": record("reference_guided"),
        "fractional": record("fractional"),
        "top_pca": record("top_pca"),
        "random_mean": record("random_mean"),
        "half_split_stability": {
            name: float(stability_rank.loc[name, "half_split_overlap"])
            for name in stability_rank.index
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-path", default="/data/shared/imagenet-1k")
    parser.add_argument("--rae-repo-path", default="external/RAE")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--final-ridge", type=float, default=1e-3)
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260720)
    parser.add_argument("--block-rank", type=int, default=16)
    parser.add_argument("--block-max-rank", type=int, default=128)
    parser.add_argument("--reuse-moments", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    configure_numerics(args.random_seed + rank)
    output = args.output.expanduser().resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    reference_payload = torch.load(
        args.reference.expanduser(), map_location="cpu", weights_only=False
    )
    config = reference_payload["config"]
    ranks = tuple(sorted(int(value) for value in reference_payload["subspaces"]))
    middle_index = int(config["middle_index"])
    moments_path = output / "moments.pt"

    if args.reuse_moments:
        if rank == 0 and not moments_path.exists():
            raise FileNotFoundError(moments_path)
        if world_size > 1:
            dist.barrier()
        payload = torch.load(moments_path, map_location="cpu", weights_only=False)
        train = _moments_from_payload(payload["train"])
        validation = _moments_from_payload(payload["validation"])
        train_first = _moments_from_payload(payload["train_first"])
        train_second = _moments_from_payload(payload["train_second"])
    else:
        train_dataset = load_named_dataset(
            "imagenet_parquet", "/data/shared", split="train", dataset_path=args.dataset_path
        )
        val_dataset = load_named_dataset(
            "imagenet_parquet", "/data/shared", split="validation", dataset_path=args.dataset_path
        )
        train_indices = [int(value) for value in reference_payload["train_indices"]]
        val_indices = [int(value) for value in reference_payload["val_indices"]]
        train_pairs = list(enumerate(train_indices))[rank::world_size]
        val_pairs = list(enumerate(val_indices))[rank::world_size]
        adapter = load_rae_adapter(
            "rae_dinov2",
            repo_path=args.rae_repo_path,
            device=device,
            dtype=torch.float32,
            auto_clone=False,
            auto_download=False,
        )
        adapter.model.requires_grad_(False).eval()
        train_gpu, first_gpu, second_gpu = accumulate_split_moments(
            adapter,
            train_dataset,
            train_pairs,
            total_count=len(train_indices),
            middle_index=middle_index,
            batch_size=args.batch_size,
            device=device,
            include_halves=True,
        )
        val_gpu, _, _ = accumulate_split_moments(
            adapter,
            val_dataset,
            val_pairs,
            total_count=len(val_indices),
            middle_index=middle_index,
            batch_size=args.batch_size,
            device=device,
            include_halves=False,
        )
        assert first_gpu is not None and second_gpu is not None
        for value in (train_gpu, first_gpu, second_gpu, val_gpu):
            reduce_moments(value, world_size)
        train = _cpu_double(train_gpu)
        train_first = _cpu_double(first_gpu)
        train_second = _cpu_double(second_gpu)
        validation = _cpu_double(val_gpu)
        if rank == 0:
            torch.save(
                {
                    "train": _moments_payload(train),
                    "validation": _moments_payload(validation),
                    "train_first": _moments_payload(train_first),
                    "train_second": _moments_payload(train_second),
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "middle_index": middle_index,
                    "world_size": world_size,
                },
                moments_path,
            )

    if rank == 0:
        reference_bases = {
            int(key): value["basis"]
            for key, value in reference_payload["subspaces"].items()
        }
        rows, stability, bases = analyze_moments(
            train,
            validation,
            train_first,
            train_second,
            reference_bases,
            ranks=ranks,
            ridge=args.ridge,
            final_ridge=args.final_ridge,
            random_controls=args.random_controls,
            random_seed=args.random_seed,
        )
        block_rows, block_bases = build_block_atlas(
            train,
            validation,
            block_rank=args.block_rank,
            max_rank=args.block_max_rank,
            ridge=args.ridge,
            final_ridge=args.final_ridge,
            random_controls=args.random_controls,
            random_seed=args.random_seed,
        )
        rows.to_csv(output / "basis_metrics.csv", index=False)
        aggregate_random_controls(rows).to_csv(
            output / "basis_metrics_display.csv", index=False
        )
        stability.to_csv(output / "basis_stability.csv", index=False)
        block_rows.to_csv(output / "basis_block_metrics.csv", index=False)
        torch.save(
            {
                "bases": bases,
                "blocks": block_bases,
                "block_rank": args.block_rank,
                "block_max_rank": args.block_max_rank,
                "ranks": ranks,
                "ridge": args.ridge,
                "final_ridge": args.final_ridge,
                "reference": str(args.reference.expanduser().resolve()),
            },
            output / "bases.pt",
        )
        plot_comparison(rows, stability, output)
        plot_block_atlas(block_rows, output)
        summary = build_summary(rows, stability)
        summary.update(
            {
                "train_images": len(reference_payload["train_indices"]),
                "validation_images": len(reference_payload["val_indices"]),
                "middle_index": middle_index,
                "ridge": args.ridge,
                "final_ridge": args.final_ridge,
                "random_controls": args.random_controls,
                "world_size": world_size,
            }
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
