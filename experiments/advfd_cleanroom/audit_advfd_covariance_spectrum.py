#!/usr/bin/env python3
"""Decompose AdvFD's real-whitened covariance witness into eigenmodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


EQVAE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-moments", type=Path, required=True)
    parser.add_argument("--fake-moments", type=Path, required=True)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_fresh_variants(path: Path):
    from experiments.advfd_cleanroom.temporal_gauge import (
        PopulationMoments,
        merge_population_moments,
    )

    payload = np.load(path, allow_pickle=False)

    def load(prefix: str) -> PopulationMoments:
        return PopulationMoments(
            mean=np.asarray(payload[f"{prefix}_mean"], dtype=np.float64),
            covariance=np.asarray(
                payload[f"{prefix}_covariance"],
                dtype=np.float64,
            ),
            count=int(payload[f"{prefix}_count"]),
        )

    first = load("fresh_a")
    second = load("fresh_b")
    return {
        "current_a": first,
        "current_b": second,
        "current_full": merge_population_moments(first, second),
    }


def contribution_summary(eigenvalues: np.ndarray) -> dict:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    contributions = np.square(np.sqrt(np.clip(eigenvalues, 0.0, None)) - 1.0)
    descending = np.sort(contributions)[::-1]
    total = float(descending.sum())
    top_ks = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    top_shares = {
        str(k): float(descending[: min(k, len(descending))].sum() / total)
        if total > 0.0
        else 0.0
        for k in top_ks
        if k <= len(descending)
    }
    cumulative = np.cumsum(descending)
    modes_for_share = {}
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        if total <= 0.0:
            count = 0
        else:
            count = int(np.searchsorted(cumulative, threshold * total) + 1)
        modes_for_share[f"{threshold:.2f}"] = count
    squared_sum = float(np.square(descending).sum())
    participation_ratio = total * total / squared_sum if squared_sum > 0.0 else 0.0
    if total > 0.0:
        probabilities = descending / total
        positive = probabilities[probabilities > 0.0]
        entropy_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    else:
        entropy_rank = 0.0
    low_mask = eigenvalues < 1.0
    high_mask = eigenvalues > 1.0
    return {
        "total": total,
        "top_k_share": top_shares,
        "modes_for_cumulative_share": modes_for_share,
        "participation_ratio": float(participation_ratio),
        "entropy_effective_rank": entropy_rank,
        "maximum_contribution": float(descending[0]),
        "median_contribution": float(np.median(descending)),
        "low_eigenvalue_contribution_share": float(contributions[low_mask].sum() / total)
        if total > 0.0
        else 0.0,
        "high_eigenvalue_contribution_share": float(contributions[high_mask].sum() / total)
        if total > 0.0
        else 0.0,
    }


@torch.inference_mode()
def analyze_pair(real, fake, *, epsilon: float) -> dict:
    from experiments.advfd_cleanroom.temporal_gauge import (
        real_whitened_fd_components_from_stats,
        torch_population_moments,
    )

    real_mean, real_covariance = torch_population_moments(real, device="cuda")
    fake_mean, fake_covariance = torch_population_moments(fake, device="cuda")
    mean_term, covariance_term, eigenvalues = real_whitened_fd_components_from_stats(
        real_mean,
        real_covariance,
        fake_mean,
        fake_covariance,
        epsilon=epsilon,
    )
    eigenvalues_np = eigenvalues.cpu().numpy()
    spectrum = contribution_summary(eigenvalues_np)
    if not np.isclose(
        spectrum["total"],
        float(covariance_term),
        rtol=5e-6,
        atol=5e-5,
    ):
        raise RuntimeError("eigenmode contributions do not reconstruct covariance FD")
    quantile_levels = np.asarray([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    eigen_quantiles = np.quantile(eigenvalues_np, quantile_levels)
    return {
        "real_count": int(real.count),
        "fake_count": int(fake.count),
        "feature_dim": int(eigenvalues.numel()),
        "mean_term": float(mean_term),
        "covariance_term": float(covariance_term),
        "total_fd": float(mean_term + covariance_term),
        "covariance_fraction": float(
            covariance_term / (mean_term + covariance_term).clamp_min(1e-30)
        ),
        "eigenvalue_quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "q01", "q10", "q50", "q90", "q99", "max"),
                eigen_quantiles,
            )
        },
        "covariance_contributions": spectrum,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EQVAE_ROOT))
    from experiments.advfd_cleanroom.audit_advfd_temporal_gauge_stats import (
        moments_from_ema_state,
        select_adv_state,
    )

    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    adv_state = select_adv_state(checkpoint, None)
    real_variants = {"ema": moments_from_ema_state(adv_state["real_stats"])}
    fake_variants = {"ema": moments_from_ema_state(adv_state["fake_stats"])}
    real_variants.update(load_fresh_variants(args.real_moments))
    fake_variants.update(load_fresh_variants(args.fake_moments))
    metadata = {
        "saved_step": int(checkpoint.get("step", -1)),
        "current_step": int(checkpoint.get("current_step", -1)),
        "samples_seen": int(checkpoint.get("samples_seen", -1)),
    }
    del checkpoint, adv_state

    pair_names = {
        "ema": ("ema", "ema"),
        "current_a": ("current_a", "current_a"),
        "current_b": ("current_b", "current_b"),
        "current_full": ("current_full", "current_full"),
        "current_real_with_ema_fake": ("current_full", "ema"),
        "ema_real_with_current_fake": ("ema", "current_full"),
    }
    analyses = {
        name: analyze_pair(
            real_variants[real_name],
            fake_variants[fake_name],
            epsilon=args.whiten_eps,
        )
        for name, (real_name, fake_name) in pair_names.items()
    }
    result = {
        "protocol": "advfd_generalized_covariance_spectrum_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_metadata": metadata,
        "real_moments": str(args.real_moments.expanduser().resolve()),
        "fake_moments": str(args.fake_moments.expanduser().resolve()),
        "whiten_epsilon": float(args.whiten_eps),
        "analyses": analyses,
        "interpretation_boundary": (
            "Eigenmode contributions are exact for AdvFD's real-whitened, "
            "epsilon-regularized objective. Current A/B agreement bounds finite-bank noise."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
