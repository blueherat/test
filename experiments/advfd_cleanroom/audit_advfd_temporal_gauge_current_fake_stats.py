#!/usr/bin/env python3
"""Build current-generator fake moments without PNG clipping or quantization."""

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
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--ema-beta", type=float, default=0.99)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--cfg-omega", type=float, default=8.5)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=0.7)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-moments", type=Path, required=True)
    return parser.parse_args()


def build_generator(checkpoint: dict, manifest: dict, device: torch.device):
    import models

    model_args = manifest["official_arguments"]
    model_name = str(model_args["model"])
    if model_name not in models.pMFDenoiser_models:
        raise ValueError(f"expected a pMF generator, got {model_name!r}")
    generator = models.pMFDenoiser_models[model_name](
        img_size=int(model_args["img_size"]),
        patch_size=int(model_args["patch_size"]),
        in_channels=int(model_args["token_channels"]),
        tokenizer_patch_size=int(model_args["tokenizer_patch_size"]),
        num_classes=int(model_args["num_classes"]),
        label_drop_prob=float(model_args["label_drop_prob"]),
        P_mean=float(model_args["P_mean"]),
        P_std=float(model_args["P_std"]),
        ratio_r_neq_t=float(model_args["ratio_r_neq_t"]),
        cfg_beta=float(model_args["cfg_beta"]),
        tr_uniform=bool(model_args["tr_uniform"]),
        cfg_omega_max=float(model_args["cfg_omega_max"]),
        aux_head_depth=int(model_args["aux_head_depth"]),
        class_tokens=int(model_args["class_tokens"]),
        time_tokens=int(model_args["time_tokens"]),
        guidance_tokens=int(model_args["guidance_tokens"]),
        interval_tokens=int(model_args["interval_tokens"]),
        t_eps=float(model_args["t_eps"]),
        perceptual_threshold=float(model_args["perceptual_threshold"]),
        perceptual_loss_on_aux=bool(model_args["perceptual_loss_on_aux"]),
        rope_2d=bool(model_args["rope_2d"]),
        learned_pe=bool(model_args["learned_pe"]),
        disable_v_head=bool(model_args["disable_v_head"]),
        noise_scale=float(model_args["noise_scale"]),
        norm_eps=float(model_args["norm_eps"]),
        norm_p=float(model_args["norm_p"]),
        grad_checkpointing=False,
    ).to(device)
    incompatible = generator.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"generator checkpoint mismatch: {incompatible}")
    return generator.eval().requires_grad_(False), model_args


@torch.inference_mode()
def generate_bank_moments(
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    *,
    count: int,
    batch_size: int,
    seed: int,
    num_classes: int,
    cfg_omega: float,
    interval_min: float,
    interval_max: float,
):
    from experiments.advfd_cleanroom.temporal_gauge import population_moments_from_sums

    device = next(generator.parameters()).device
    noise_rng = torch.Generator(device=device).manual_seed(seed)
    label_rng = torch.Generator(device=device).manual_seed(seed + 1_000_000)
    feature_sum = None
    feature_outer_sum = None
    below_zero = 0
    above_one = 0
    image_values = 0
    completed = 0
    while completed < count:
        current_batch = min(batch_size, count - completed)
        noise = torch.randn(
            current_batch,
            generator.in_channels,
            generator.input_size,
            generator.input_size,
            generator=noise_rng,
            device=device,
        ) * float(generator.noise_scale)
        labels = torch.randint(
            0,
            num_classes,
            (current_batch,),
            generator=label_rng,
            device=device,
        )
        model_space = generator.sample_images_with_grad(
            noise,
            labels,
            sampling_args={
                "cfg": cfg_omega,
                "t_min": interval_min,
                "t_max": interval_max,
                "num_steps": 1,
            },
        )
        images = model_space.mul(0.5).add(0.5)
        below_zero += int((images < 0.0).sum())
        above_one += int((images > 1.0).sum())
        image_values += int(images.numel())
        features, _ = critic(images)
        features = features.double()
        if feature_sum is None:
            feature_dim = int(features.shape[-1])
            feature_sum = torch.zeros(feature_dim, dtype=torch.float64, device=device)
            feature_outer_sum = torch.zeros(
                feature_dim,
                feature_dim,
                dtype=torch.float64,
                device=device,
            )
        feature_sum.add_(features.sum(0))
        feature_outer_sum.addmm_(features.T, features)
        completed += current_batch
        if completed == count or completed % max(10 * batch_size, 1) == 0:
            print(f"current fake features: {completed}/{count}", flush=True)
    if feature_sum is None or feature_outer_sum is None:
        raise RuntimeError("empty generated feature bank")
    moments = population_moments_from_sums(
        feature_sum.cpu().numpy(),
        feature_outer_sum.cpu().numpy(),
        completed,
    )
    range_stats = {
        "fraction_below_zero": below_zero / image_values,
        "fraction_above_one": above_one / image_values,
    }
    return moments, range_stats


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EQVAE_ROOT))
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))

    from experiments.advfd_cleanroom.audit_advfd_temporal_gauge_stats import (
        average_pair_metrics,
        moments_from_ema_state,
        select_adv_state,
    )
    from experiments.advfd_cleanroom.temporal_gauge import (
        build_regularized_whitener,
        merge_population_moments,
        regularized_whitening_consistency,
    )
    from frechet_distance.repr_models import load_repr_model

    device = torch.device("cuda")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    adv_state = select_adv_state(checkpoint, None)
    ema_fake = moments_from_ema_state(adv_state["fake_stats"])
    critic, feature_dim, _, _ = load_repr_model("inception", device="cuda")
    critic.load_state_dict(adv_state["model"], strict=True)
    critic.eval().requires_grad_(False)
    manifest = json.loads(
        args.adapter_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    generator, model_args = build_generator(checkpoint, manifest, device)
    metadata = {
        "saved_step": int(checkpoint.get("step", -1)),
        "current_step": int(checkpoint.get("current_step", -1)),
        "samples_seen": int(checkpoint.get("samples_seen", -1)),
    }
    del checkpoint, adv_state

    fresh_a, range_a = generate_bank_moments(
        generator,
        critic,
        count=args.bank_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_classes=int(model_args["num_classes"]),
        cfg_omega=args.cfg_omega,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )
    fresh_b, range_b = generate_bank_moments(
        generator,
        critic,
        count=args.bank_size,
        batch_size=args.batch_size,
        seed=args.seed + 10_000,
        num_classes=int(model_args["num_classes"]),
        cfg_omega=args.cfg_omega,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )
    fresh_full = merge_population_moments(fresh_a, fresh_b)
    epsilon = float(args.whiten_eps)
    fresh_a_whitener = build_regularized_whitener(
        fresh_a,
        epsilon=epsilon,
        device=device,
    )
    fresh_b_whitener = build_regularized_whitener(
        fresh_b,
        epsilon=epsilon,
        device=device,
    )
    ema_whitener = build_regularized_whitener(
        ema_fake,
        epsilon=epsilon,
        device=device,
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
            ema_fake,
            fresh_a,
            epsilon=epsilon,
            whitener=ema_whitener,
        ),
        "ema_to_fresh_b": regularized_whitening_consistency(
            ema_fake,
            fresh_b,
            epsilon=epsilon,
            whitener=ema_whitener,
        ),
        "ema_to_fresh_full": regularized_whitening_consistency(
            ema_fake,
            fresh_full,
            epsilon=epsilon,
            whitener=ema_whitener,
        ),
    }
    result = {
        "protocol": "advfd_temporal_gauge_current_fake_stats_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "adapter_manifest": str(args.adapter_manifest.expanduser().resolve()),
        "feature_dim": int(feature_dim),
        "bank_size": int(args.bank_size),
        "seed_a": int(args.seed),
        "seed_b": int(args.seed + 10_000),
        "sampling": {
            "cfg": float(args.cfg_omega),
            "num_steps": 1,
            "interval_min": float(args.interval_min),
            "interval_max": float(args.interval_max),
            "online_generator": True,
            "image_clipping": False,
            "image_quantization": False,
        },
        "range_stats": {"fresh_a": range_a, "fresh_b": range_b},
        "whiten_epsilon": epsilon,
        "comparisons": comparisons,
        "fresh_split_noise_floor_average": average_pair_metrics(
            comparisons["fresh_a_to_b"],
            comparisons["fresh_b_to_a"],
        ),
        "notes": [
            "Fresh fake banks are generated in memory from the online checkpoint.",
            "The critic sees the same unclipped [0,1]-mapped tensor used by training.",
            "Fake EMA mismatch still mixes critic-frame drift and generator-history drift.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_moments.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_moments,
        fresh_a_mean=fresh_a.mean,
        fresh_a_covariance=fresh_a.covariance,
        fresh_a_count=np.asarray(fresh_a.count, dtype=np.int64),
        fresh_b_mean=fresh_b.mean,
        fresh_b_covariance=fresh_b.covariance,
        fresh_b_count=np.asarray(fresh_b.count, dtype=np.int64),
        seed_a=np.asarray(args.seed, dtype=np.int64),
        seed_b=np.asarray(args.seed + 10_000, dtype=np.int64),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
