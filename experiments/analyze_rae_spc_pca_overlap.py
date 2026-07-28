"""Measure whether the SPC guided basis overlaps high-variance RAE directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


from experiments.analyze_rae_spc_subspace_signal import (
    DEFAULT_CACHE,
    DEFAULT_OUTPUT,
    DEFAULT_SUBSPACES,
)
from experiments.rae_latent_cache import CachedRAELatentDataset
from experiments.rae_layerwise_path import spatial_center
from experiments.run_rae_spc_directional_sensitivity import orthogonal_control_basis


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    left, _ = torch.linalg.qr(left.double(), mode="reduced")
    right, _ = torch.linalg.qr(right.double(), mode="reduced")
    cosines = torch.linalg.svdvals(left.transpose(0, 1) @ right).clamp(0, 1)
    return {
        "mean_squared_principal_cosine": float(cosines.square().mean()),
        "mean_principal_cosine": float(cosines.mean()),
        "min_principal_cosine": float(cosines.min()),
        "max_principal_cosine": float(cosines.max()),
        "principal_cosines": [float(value) for value in cosines.cpu()],
    }


@torch.no_grad()
def residual_covariance(
    dataset: CachedRAELatentDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    first, _ = dataset[0]
    channels = int(first.shape[0])
    gram = torch.zeros(channels, channels, device=device, dtype=torch.float32)
    token_count = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        latent = torch.stack([dataset[index][0] for index in range(start, end)]).to(
            device
        )
        _, residual = spatial_center(latent)
        rows = residual.permute(0, 2, 3, 1).reshape(-1, channels)
        gram.addmm_(rows.transpose(0, 1), rows)
        token_count += int(len(rows))
    covariance = gram.double().cpu() / float(token_count)
    return 0.5 * (covariance + covariance.transpose(0, 1))


def pca_overlap_metrics(
    covariance: torch.Tensor,
    guided_basis: torch.Tensor,
    control_basis: torch.Tensor,
) -> dict[str, object]:
    covariance = covariance.double()
    rank = int(guided_basis.shape[1])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    pca_basis = eigenvectors[:, order[:rank]]
    total = eigenvalues.sum().clamp_min(1e-20)

    def energy_fraction(basis: torch.Tensor) -> float:
        basis = basis.double()
        return float(torch.trace(basis.T @ covariance @ basis) / total)

    guided_overlap = subspace_overlap(guided_basis, pca_basis)
    control_overlap = subspace_overlap(control_basis, pca_basis)
    return {
        "channels": int(covariance.shape[0]),
        "rank": rank,
        "top_pca_energy_fraction": float(eigenvalues[:rank].sum() / total),
        "guided_energy_fraction": energy_fraction(guided_basis),
        "control_energy_fraction": energy_fraction(control_basis),
        "guided_over_top_pca_energy": energy_fraction(guided_basis)
        / float(eigenvalues[:rank].sum() / total),
        "guided_vs_top_pca": guided_overlap,
        "control_vs_top_pca": control_overlap,
        "effective_rank": float(total.square() / eigenvalues.square().sum()),
        "top_eigenvalues": [float(value) for value in eigenvalues[:rank].cpu()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=100_288)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--control-seed", type=int, default=20_260_731)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the covariance accumulation")
    device = torch.device("cuda", args.device)
    payload = torch.load(
        args.subspaces.expanduser(), map_location="cpu", weights_only=False
    )
    entry = payload["subspaces"].get(
        args.rank, payload["subspaces"].get(str(args.rank))
    )
    if entry is None:
        raise KeyError(f"rank {args.rank} is absent from {args.subspaces}")
    guided = entry["basis"].double().contiguous()
    control = orthogonal_control_basis(
        guided.float(), seed=args.control_seed
    ).double().cpu()
    dataset = CachedRAELatentDataset(
        args.cache.expanduser(), start=args.start, stop=args.start + args.count
    )
    covariance = residual_covariance(
        dataset, batch_size=args.batch_size, device=device
    )
    metrics = pca_overlap_metrics(covariance, guided, control)
    metrics["sample_count"] = int(args.count)
    metrics["token_count"] = int(args.count * 16 * 16)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pca_overlap.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
