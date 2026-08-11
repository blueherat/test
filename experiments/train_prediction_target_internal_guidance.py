#!/usr/bin/env python3
"""Train shared-trunk guidance heads on the spiral toy.

The Internal Guidance control predicts the same clean ``x`` target from its
intermediate and final output heads.  The cross-target variant keeps the final
head on ``x`` while training the intermediate head to predict velocity ``v``.
That second variant is our diagnostic, not the original IG method.  Both
losses are evaluated in the same velocity space to match the source toy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.run_prediction_target_bayes_oracle_v5 import (
    ResidualBlock,
    TangentGaussianMixture,
    build_model,
    loss_for_output,
    save_csv,
    save_json,
    stable_seed,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import TimeEmbedding
from experiments.run_prediction_target_extrapolation_toy_v4 import (
    clean_from_output,
    direct_target,
)


class InternalResidualDenoiseMLP(nn.Module):
    """Residual denoiser with an intermediate and a final output head.

    Final-head modules are constructed in exactly the same order as the
    baseline ``ResidualDenoiseMLP``.  With the same seed, its trunk and final
    head therefore start bit-identically to the original x-predictor; the
    auxiliary head is initialized only afterwards.
    """

    def __init__(
        self,
        *,
        D: int,
        hidden: int,
        depth: int,
        time_dim: int,
        intermediate_after: int,
    ) -> None:
        super().__init__()
        block_count = max(depth - 2, 0)
        if not 1 <= intermediate_after <= block_count:
            raise ValueError(
                "intermediate_after must identify one of the residual blocks"
            )
        self.D = int(D)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.time_dim = int(time_dim)
        self.intermediate_after = int(intermediate_after)
        self.time = TimeEmbedding(time_dim)
        self.in_proj = nn.Linear(D + time_dim, hidden)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden) for _ in range(block_count)
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, D)
        self.intermediate_norm = nn.LayerNorm(hidden)
        self.intermediate_proj = nn.Linear(hidden, D)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time = self.time(t)
        features = F.silu(self.in_proj(torch.cat([x, time], dim=1)))
        intermediate_features = None
        for index, block in enumerate(self.blocks, start=1):
            features = block(features)
            if index == self.intermediate_after:
                intermediate_features = features
        if intermediate_features is None:
            raise RuntimeError("intermediate features were not produced")
        intermediate = self.intermediate_proj(
            self.intermediate_norm(intermediate_features)
        )
        final = self.out_proj(self.out_norm(features))
        return intermediate, final


def build_mixture(manifest: dict[str, object], device: torch.device) -> TangentGaussianMixture:
    return TangentGaussianMixture(
        D=int(manifest["D"]),
        components=int(manifest["components"]),
        curvature=float(manifest["curvature"]),
        frequency_scale=float(manifest["frequency_scale"]),
        center_rms=float(manifest["center_rms"]),
        sigma_tangent=float(manifest["sigma_tangent"]),
        sigma_normal=float(manifest["sigma_normal"]),
        seed=int(manifest["mixture_seed"]),
        device=device,
    )


def build_internal_same_final_init(
    *,
    manifest: dict[str, object],
    hidden: int,
    intermediate_after: int,
    device: torch.device,
    seed: int,
) -> InternalResidualDenoiseMLP:
    torch.manual_seed(seed)
    return InternalResidualDenoiseMLP(
        D=int(manifest["D"]),
        hidden=hidden,
        depth=int(manifest["depth"]),
        time_dim=int(manifest["time_dim"]),
        intermediate_after=intermediate_after,
    ).to(device)


def assert_baseline_initialization_matches(
    *,
    internal: InternalResidualDenoiseMLP,
    manifest: dict[str, object],
    hidden: int,
    device: torch.device,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    baseline = build_model(
        "residual",
        D=int(manifest["D"]),
        hidden=hidden,
        depth=int(manifest["depth"]),
        time_dim=int(manifest["time_dim"]),
    ).to(device)
    shared_names = (
        "time.",
        "in_proj.",
        "blocks.",
        "out_norm.",
        "out_proj.",
    )
    baseline_state = baseline.state_dict()
    internal_state = internal.state_dict()
    for name, value in baseline_state.items():
        if not name.startswith(shared_names):
            continue
        if not torch.equal(value, internal_state[name]):
            raise RuntimeError(f"baseline initialization mismatch at {name}")


def checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step{step:06d}.pt"


def save_checkpoint(
    *,
    path: Path,
    step: int,
    model: InternalResidualDenoiseMLP,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    history: list[dict[str, float]],
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "history": history,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate-after", type=int, default=1)
    parser.add_argument(
        "--intermediate-target",
        choices=("x", "v"),
        default="x",
        help="Direct prediction target of the weak/intermediate head.",
    )
    parser.add_argument("--intermediate-weight", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--checkpoint-every", type=int, default=6000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.steps <= 0 or args.checkpoint_every <= 0:
        raise ValueError("steps and checkpoint_every must be positive")
    if args.intermediate_weight <= 0:
        raise ValueError("intermediate_weight must be positive")
    manifest = json.loads(
        (args.source_seed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    device = torch.device(args.device)
    mixture = build_mixture(manifest, device)
    seed = int(manifest["seed"])
    initialization_seed = stable_seed(seed, 101)
    model = build_internal_same_final_init(
        manifest=manifest,
        hidden=args.hidden,
        intermediate_after=args.intermediate_after,
        device=device,
        seed=initialization_seed,
    )
    assert_baseline_initialization_matches(
        internal=model,
        manifest=manifest,
        hidden=args.hidden,
        device=device,
        seed=initialization_seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(manifest["lr"]),
        weight_decay=float(manifest["weight_decay"]),
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(stable_seed(seed, 211))
    validation_generator = torch.Generator(device=device.type)
    validation_generator.manual_seed(stable_seed(seed, 223))
    val_x, val_eps, val_t, val_x_t, _ = mixture.noised_batch(
        int(manifest["validation_samples"]),
        t_min=float(manifest["t_min"]),
        t_max=float(manifest["t_max"]),
        time_sampler=str(manifest.get("time_sampler", "uniform")),
        time_logit_mean=float(manifest.get("time_logit_mean", 0.8)),
        time_logit_std=float(manifest.get("time_logit_std", 0.8)),
        generator=validation_generator,
    )
    with torch.inference_mode():
        val_bayes = mixture.posterior_clean(val_x_t, val_t)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "protocol": "prediction_target_internal_guidance_v2",
        "method_family": (
            "internal_guidance_same_target"
            if args.intermediate_target == "x"
            else "cross_target_internal_guidance_diagnostic"
        ),
        "source_seed_dir": str(args.source_seed_dir.resolve()),
        "seed": seed,
        "hidden": args.hidden,
        "intermediate_after": args.intermediate_after,
        "intermediate_target": args.intermediate_target,
        "final_target": "x",
        "intermediate_weight": args.intermediate_weight,
        "steps": args.steps,
        "checkpoint_every": args.checkpoint_every,
        "batch_size": int(manifest["batch_size"]),
        "lr": float(manifest["lr"]),
        "weight_decay": float(manifest["weight_decay"]),
        "grad_clip": float(manifest["grad_clip"]),
        "loss_space": str(manifest["loss_space"]),
        "same_final_initialization_as_baseline": True,
        "intermediate_output_head": (
            "independent LayerNorm+Linear head after the selected residual block"
        ),
        "paper_correspondence": (
            "IG also attaches an independent output layer after an intermediate "
            "block; unlike the official same-target objective, intermediate_target=v "
            "with final_target=x is a cross-target diagnostic"
        ),
        "training_data": "fresh samples from the same analytic mixture",
        "validation_data": "fixed independent RNG stream",
    }
    save_json(output_dir / "training_manifest.json", config)
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        clean, noise, time, state, _ = mixture.noised_batch(
            int(manifest["batch_size"]),
            t_min=float(manifest["t_min"]),
            t_max=float(manifest["t_max"]),
            time_sampler=str(manifest.get("time_sampler", "uniform")),
            time_logit_mean=float(manifest.get("time_logit_mean", 0.8)),
            time_logit_std=float(manifest.get("time_logit_std", 0.8)),
            generator=generator,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        intermediate, final = model(state, time)
        intermediate_loss = loss_for_output(
            intermediate,
            x_t=state,
            t=time,
            x=clean,
            eps=noise,
            target=args.intermediate_target,
            loss_space=str(manifest["loss_space"]),
            conversion_clip=float(manifest["conversion_clip"]),
        )
        final_loss = loss_for_output(
            final,
            x_t=state,
            t=time,
            x=clean,
            eps=noise,
            target="x",
            loss_space=str(manifest["loss_space"]),
            conversion_clip=float(manifest["conversion_clip"]),
        )
        total_loss = final_loss + args.intermediate_weight * intermediate_loss
        total_loss.backward()
        if float(manifest["grad_clip"]) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), float(manifest["grad_clip"]))
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                val_intermediate, val_final = model(val_x_t, val_t)
                val_intermediate_clean = clean_from_output(
                    val_intermediate,
                    val_x_t,
                    val_t,
                    args.intermediate_target,
                    float(manifest["conversion_clip"]),
                )
                val_final_clean = clean_from_output(
                    val_final,
                    val_x_t,
                    val_t,
                    "x",
                    float(manifest["conversion_clip"]),
                )
                final_excess = (val_final_clean - val_bayes).square().mean()
                intermediate_excess = (
                    val_intermediate_clean - val_bayes
                ).square().mean()
                final_paired = (val_final_clean - val_x).square().mean()
                intermediate_paired = (
                    val_intermediate_clean - val_x
                ).square().mean()
                intermediate_direct_mse = (
                    val_intermediate
                    - direct_target(val_x, val_eps, args.intermediate_target)
                ).square().mean()
            row = {
                "step": float(step),
                "train_final_loss": float(final_loss.detach()),
                "train_intermediate_loss": float(intermediate_loss.detach()),
                "train_total_loss": float(total_loss.detach()),
                "val_final_excess_mse": float(final_excess),
                "val_intermediate_excess_mse": float(intermediate_excess),
                "val_final_paired_mse": float(final_paired),
                "val_intermediate_paired_mse": float(intermediate_paired),
                "val_intermediate_direct_target_mse": float(
                    intermediate_direct_mse
                ),
            }
            history.append(row)
            save_csv(output_dir / "train_history.csv", history)
            print(
                f"[IG spiral] {step}/{args.steps} "
                f"final={row['train_final_loss']:.5g} "
                f"inter={row['train_intermediate_loss']:.5g} "
                f"final_excess={row['val_final_excess_mse']:.5g} "
                f"inter_excess={row['val_intermediate_excess_mse']:.5g}",
                flush=True,
            )
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_checkpoint(
                path=checkpoint_path(output_dir, step),
                step=step,
                model=model,
                optimizer=optimizer,
                generator=generator,
                history=history,
                config=config,
            )


if __name__ == "__main__":
    main()
