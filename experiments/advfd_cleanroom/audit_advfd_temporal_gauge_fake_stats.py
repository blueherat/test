#!/usr/bin/env python3
"""Legacy PNG-based audit of fake EMA moments.

The canonical audit generates current fake samples in memory via
``audit_advfd_temporal_gauge_current_fake_stats.py``. This entry point is kept
only to reproduce the earlier experiment from an existing PNG folder; PNG
quantization prevents it from being the source of the formal reported values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


EQVAE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generated-folder", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--repr-model", default="inception")
    parser.add_argument("--adv-state-name", default=None)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-moments", type=Path, default=None)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EQVAE_ROOT))
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))

    from experiments.advfd_cleanroom.audit_advfd_temporal_gauge_stats import (
        average_pair_metrics,
        extract_bank_moments,
        moments_from_ema_state,
        select_adv_state,
    )
    from experiments.advfd_cleanroom.temporal_gauge import (
        build_regularized_whitener,
        merge_population_moments,
        regularized_whitening_consistency,
    )
    from frechet_distance.datasets import ImageFolderDataset
    from frechet_distance.repr_models import load_repr_model

    checkpoint_path = args.checkpoint.expanduser().resolve()
    generated_folder = args.generated_folder.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    adv_state = select_adv_state(checkpoint, args.adv_state_name)
    ema_fake = moments_from_ema_state(adv_state["fake_stats"])
    model, feature_dim, _, _ = load_repr_model(args.repr_model, device="cuda")
    model.load_state_dict(adv_state["model"], strict=True)
    model.eval()
    metadata = {
        "saved_step": int(checkpoint.get("step", -1)),
        "current_step": int(checkpoint.get("current_step", -1)),
        "samples_seen": int(checkpoint.get("samples_seen", -1)),
    }
    del checkpoint, adv_state

    dataset = ImageFolderDataset(str(generated_folder), img_size=args.img_size)
    required = 2 * int(args.bank_size)
    if required > len(dataset):
        raise ValueError(
            f"two fake banks require {required} images, folder has {len(dataset)}"
        )
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
        ema_fake, epsilon=epsilon, device="cuda"
    )
    comparisons = {
        "self_fresh_a": regularized_whitening_consistency(
            fresh_a, fresh_a, epsilon=epsilon, whitener=fresh_a_whitener
        ),
        "fresh_a_to_b": regularized_whitening_consistency(
            fresh_a, fresh_b, epsilon=epsilon, whitener=fresh_a_whitener
        ),
        "fresh_b_to_a": regularized_whitening_consistency(
            fresh_b, fresh_a, epsilon=epsilon, whitener=fresh_b_whitener
        ),
        "ema_to_fresh_a": regularized_whitening_consistency(
            ema_fake, fresh_a, epsilon=epsilon, whitener=ema_whitener
        ),
        "ema_to_fresh_b": regularized_whitening_consistency(
            ema_fake, fresh_b, epsilon=epsilon, whitener=ema_whitener
        ),
        "ema_to_fresh_full": regularized_whitening_consistency(
            ema_fake, fresh_full, epsilon=epsilon, whitener=ema_whitener
        ),
    }
    result = {
        "protocol": "advfd_temporal_gauge_fake_stats_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "generated_folder": str(generated_folder),
        "feature_dim": int(feature_dim),
        "bank_size": int(args.bank_size),
        "seed": int(args.seed),
        "whiten_epsilon": epsilon,
        "comparisons": comparisons,
        "fresh_split_noise_floor_average": average_pair_metrics(
            comparisons["fresh_a_to_b"], comparisons["fresh_b_to_a"]
        ),
        "notes": [
            "Fake EMA mismatch includes both critic-coordinate drift and generator-distribution history.",
            "Generated PNG quantization is shared by both fresh split banks but not historical EMA moments.",
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
