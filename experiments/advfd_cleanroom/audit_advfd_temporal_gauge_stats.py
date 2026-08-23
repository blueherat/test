#!/usr/bin/env python3
"""Audit whether AdvFD real EMA moments remain in the current critic frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


EQVAE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--packed-imagenet-root",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument("--bank-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--repr-model", default="inception")
    parser.add_argument("--adv-state-name", default=None)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-moments", type=Path, default=None)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    return parser.parse_args()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def select_adv_state(
    checkpoint: dict[str, Any], state_name: str | None
) -> dict[str, Any]:
    states = checkpoint.get("fd_adv_states")
    if not isinstance(states, list) or not states:
        raise KeyError("checkpoint does not contain fd_adv_states")
    if state_name is None:
        if len(states) != 1:
            raise ValueError("checkpoint has multiple adaptive states; choose one")
        return states[0]
    matches = [state for state in states if state.get("name") == state_name]
    if len(matches) != 1:
        raise KeyError(f"expected one adaptive state named {state_name!r}")
    return matches[0]


def moments_from_ema_state(stats: dict[str, torch.Tensor]):
    from experiments.advfd_cleanroom.temporal_gauge import PopulationMoments

    mean = stats["mu_ema"].detach().double().cpu().numpy()
    second = stats["m2_ema"].detach().double().cpu().numpy()
    covariance = second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return PopulationMoments(mean=mean, covariance=covariance, count=-1)


@torch.inference_mode()
def extract_bank_moments(
    model: torch.nn.Module,
    dataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
):
    from experiments.advfd_cleanroom.temporal_gauge import population_moments_from_sums

    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    feature_sum = None
    feature_outer_sum = None
    count = 0
    for batch_index, batch in enumerate(loader):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.cuda(non_blocking=True)
        use_amp = not getattr(model, "is_inception", False)
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            features, _ = model(images)
        features = features.double()
        if feature_sum is None:
            dimension = int(features.shape[-1])
            feature_sum = torch.zeros(dimension, dtype=torch.float64, device="cuda")
            feature_outer_sum = torch.zeros(
                dimension, dimension, dtype=torch.float64, device="cuda"
            )
        feature_sum.add_(features.sum(dim=0))
        feature_outer_sum.addmm_(features.T, features)
        count += int(features.shape[0])
        if (batch_index + 1) % 25 == 0 or count == len(indices):
            print(f"fresh real features: {count}/{len(indices)}", flush=True)
    if feature_sum is None or feature_outer_sum is None:
        raise RuntimeError("empty feature bank")
    return population_moments_from_sums(
        feature_sum.cpu().numpy(), feature_outer_sum.cpu().numpy(), count
    )


def average_pair_metrics(first: dict, second: dict) -> dict[str, float]:
    scalar_keys = [
        "mean_mahalanobis_sq",
        "mean_mahalanobis_rms_per_dim",
        "covariance_identity_frobenius_per_sqrt_dim",
        "covariance_log_eigen_rms",
        "covariance_trace_per_dim",
        "covariance_bures_to_identity_per_dim",
        "regularized_whitened_fd_per_dim",
    ]
    return {key: 0.5 * (float(first[key]) + float(second[key])) for key in scalar_keys}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EQVAE_ROOT))
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))

    from experiments.advfd_cleanroom.temporal_gauge import (
        build_regularized_whitener,
        merge_population_moments,
        regularized_whitening_consistency,
    )
    from experiments.raev2_training_core import DeterministicImageNetPacked
    from frechet_distance.repr_models import load_repr_model

    checkpoint_path = args.checkpoint.expanduser().resolve()
    packed_root = args.packed_imagenet_root.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    adv_state = select_adv_state(checkpoint, args.adv_state_name)
    ema_real = moments_from_ema_state(adv_state["real_stats"])
    model, feature_dim, _, _ = load_repr_model(args.repr_model, device="cuda")
    model.load_state_dict(adv_state["model"], strict=True)
    model.eval()
    metadata = {
        "saved_step": int(checkpoint.get("step", -1)),
        "current_step": int(checkpoint.get("current_step", -1)),
        "samples_seen": int(checkpoint.get("samples_seen", -1)),
    }
    del checkpoint, adv_state

    dataset = DeterministicImageNetPacked(
        packed_root,
        split="train",
        image_size=args.img_size,
        augmentation_seed=1,
        horizontal_flip=False,
    )
    required = 2 * int(args.bank_size)
    if required > len(dataset):
        raise ValueError("two fresh banks exceed dataset size")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=required, replace=False)
    indices_a = np.sort(indices[: args.bank_size])
    indices_b = np.sort(indices[args.bank_size :])

    fresh_a = extract_bank_moments(
        model,
        dataset,
        indices_a,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    fresh_b = extract_bank_moments(
        model,
        dataset,
        indices_b,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    fresh_full = merge_population_moments(fresh_a, fresh_b)
    epsilon = float(args.whiten_eps)

    fresh_a_whitener = build_regularized_whitener(
        fresh_a, epsilon=epsilon, device="cuda"
    )
    fresh_b_whitener = build_regularized_whitener(
        fresh_b, epsilon=epsilon, device="cuda"
    )
    ema_whitener = build_regularized_whitener(
        ema_real, epsilon=epsilon, device="cuda"
    )

    comparisons = {
        "self_fresh_a": regularized_whitening_consistency(
            fresh_a,
            fresh_a,
            epsilon=epsilon,
            whitener=fresh_a_whitener,
        ),
        "fresh_a_to_b": regularized_whitening_consistency(
            fresh_a,
            fresh_b,
            epsilon=epsilon,
            whitener=fresh_a_whitener,
        ),
        "fresh_b_to_a": regularized_whitening_consistency(
            fresh_b,
            fresh_a,
            epsilon=epsilon,
            whitener=fresh_b_whitener,
        ),
        "ema_to_fresh_a": regularized_whitening_consistency(
            ema_real,
            fresh_a,
            epsilon=epsilon,
            whitener=ema_whitener,
        ),
        "ema_to_fresh_b": regularized_whitening_consistency(
            ema_real,
            fresh_b,
            epsilon=epsilon,
            whitener=ema_whitener,
        ),
        "ema_to_fresh_full": regularized_whitening_consistency(
            ema_real,
            fresh_full,
            epsilon=epsilon,
            whitener=ema_whitener,
        ),
    }
    result = {
        "protocol": "advfd_temporal_gauge_real_stats_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "official_root": str(official_root),
        "official_commit": git_head(official_root),
        "packed_imagenet_root": str(packed_root),
        "real_distribution": {
            "split": "train",
            "crop": "ADM deterministic center crop",
            "horizontal_flip": False,
        },
        "feature_dim": int(feature_dim),
        "bank_size": int(args.bank_size),
        "seed": int(args.seed),
        "bank_a_indices_sha256": sha256_array(indices_a),
        "bank_b_indices_sha256": sha256_array(indices_b),
        "whiten_epsilon": epsilon,
        "comparisons": comparisons,
        "fresh_split_noise_floor_average": average_pair_metrics(
            comparisons["fresh_a_to_b"], comparisons["fresh_b_to_a"]
        ),
        "notes": [
            "Fresh banks use the same packed ImageNet train distribution and preprocessing as training.",
            "Population covariance matches official FeatureStatsEMA semantics.",
            "Whitening regularizes both anchor and probe covariance by epsilon times identity.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_moments is not None:
        args.output_moments.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_moments,
            fresh_a_mean=fresh_a.mean,
            fresh_a_covariance=fresh_a.covariance,
            fresh_a_count=np.asarray(fresh_a.count, dtype=np.int64),
            fresh_b_mean=fresh_b.mean,
            fresh_b_covariance=fresh_b.covariance,
            fresh_b_count=np.asarray(fresh_b.count, dtype=np.int64),
            indices_a=indices_a,
            indices_b=indices_b,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
