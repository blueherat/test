#!/usr/bin/env python3
"""Compare raw hidden-state and trained-head guidance gaps on v800 rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_hidden_state_extrapolation import (
        internal_and_final_hidden_states,
        velocity_from_hidden_state,
    )
    from experiments.imagenet100_sit_internal_v_head import (
        internal_velocity_from_features,
    )
    from experiments.sample_imagenet100_sit_frozen_internal_v_head_fid import (
        load_frozen_internal_model,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )
except ModuleNotFoundError:
    from imagenet100_sit_hidden_state_extrapolation import (
        internal_and_final_hidden_states,
        velocity_from_hidden_state,
    )
    from imagenet100_sit_internal_v_head import internal_velocity_from_features
    from sample_imagenet100_sit_frozen_internal_v_head_fid import (
        load_frozen_internal_model,
    )
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        atomic_json_dump,
        load_official_sit_module,
        sha256_file,
    )


PROTOCOL = "imagenet100_sit_hidden_state_gap_audit_v1"
DEFAULT_HEAD_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/"
    "sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "fid1k_v800_hidden_state_depth8_ema"
)
DEFAULT_TIMES = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)


def root_mean_square(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().mean(dim=1).sqrt()


def flattened_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        left.float().flatten(1),
        right.float().flatten(1),
        dim=1,
    )


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples < 2:
        raise ValueError("at least two samples are required")
    if not args.times or any(not 0 < value < 1 for value in args.times):
        raise ValueError("audit times must lie strictly inside (0,1)")
    if sorted(set(args.times)) != list(args.times):
        raise ValueError("audit times must be unique and strictly increasing")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    head_checkpoint = args.head_checkpoint.expanduser().resolve()
    sit_module, source_metadata = load_official_sit_module(
        args.official_sit_repo.expanduser().resolve(),
        verify_source=args.verify_sit_source,
    )
    model, head, model_metadata = load_frozen_internal_model(
        head_checkpoint_path=head_checkpoint,
        head_weights=args.head_weights,
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if str(model_metadata["prediction_target"]) != "velocity":
        raise ValueError("the comparison head must predict native velocity")
    if int(model_metadata["internal_depth"]) != args.internal_depth:
        raise ValueError("head checkpoint and requested internal depth disagree")

    noise = torch.randn(args.samples, *LATENT_SHAPE, device=device)
    labels = torch.randint(0, NUM_CLASSES, (args.samples,), device=device)
    integration_times = torch.tensor(
        [0.0, *args.times],
        device=device,
        dtype=torch.float32,
    )

    def baseline_velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return model(
            state,
            time_value.expand(len(state)),
            labels,
        ).float()

    trajectory = odeint(
        baseline_velocity,
        noise,
        integration_times,
        method="dopri5",
        atol=args.atol,
        rtol=args.rtol,
    )
    rows: list[dict[str, float]] = []
    for time_value, state in zip(integration_times[1:], trajectory[1:]):
        times = time_value.expand(args.samples)
        internal_hidden, final_hidden, conditioning = internal_and_final_hidden_states(
            model,
            state,
            times,
            labels,
            internal_depth=args.internal_depth,
        )
        full_velocity = velocity_from_hidden_state(
            model,
            final_hidden,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
        ).float()
        raw_internal_velocity = velocity_from_hidden_state(
            model,
            internal_hidden,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
        ).float()
        trained_internal_velocity = internal_velocity_from_features(
            model,
            head,
            internal_hidden,
            conditioning,
            latent_channels=LATENT_SHAPE[0],
        ).float()

        raw_gap = full_velocity - raw_internal_velocity
        trained_gap = full_velocity - trained_internal_velocity
        raw_gap_flat = raw_gap.flatten(1)
        trained_gap_flat = trained_gap.flatten(1)
        full_flat = full_velocity.flatten(1)
        raw_gap_rms = root_mean_square(raw_gap_flat)
        trained_gap_rms = root_mean_square(trained_gap_flat)
        gap_cosine = flattened_cosine(raw_gap, trained_gap)
        rows.append(
            {
                "time": float(time_value),
                "hidden_cosine_mean": float(
                    flattened_cosine(internal_hidden, final_hidden).mean()
                ),
                "hidden_gap_over_final_rms": float(
                    (
                        root_mean_square((final_hidden - internal_hidden).flatten(1))
                        / root_mean_square(final_hidden.flatten(1))
                    ).mean()
                ),
                "raw_gap_over_full_rms": float(
                    (raw_gap_rms / root_mean_square(full_flat)).mean()
                ),
                "trained_gap_over_full_rms": float(
                    (trained_gap_rms / root_mean_square(full_flat)).mean()
                ),
                "trained_gap_over_raw_gap_rms": float(
                    (trained_gap_rms / raw_gap_rms).mean()
                ),
                "raw_trained_gap_cosine_mean": float(gap_cosine.mean()),
                "raw_trained_gap_cosine_std": float(gap_cosine.std()),
                "raw_trained_gap_positive_fraction": float(
                    (gap_cosine > 0).float().mean()
                ),
            }
        )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "hidden_state_gap_audit.csv"
    json_path = output_root / "hidden_state_gap_audit.json"
    write_csv(csv_path, rows)
    payload = {
        "protocol": PROTOCOL,
        "source": model_metadata,
        "head_checkpoint_sha256": sha256_file(head_checkpoint),
        "internal_depth": args.internal_depth,
        "samples": args.samples,
        "seed": args.seed,
        "trajectory": {
            "initial_time": 0.0,
            "audit_times": list(args.times),
            "method": "dopri5",
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "definitions": {
            "raw_internal": "source FinalLayer(h_internal, conditioning)",
            "trained_internal": "trained auxiliary FinalLayer(h_internal, conditioning)",
            "raw_gap": "v_full - v_raw_internal",
            "trained_gap": "v_full - v_trained_internal",
        },
        "rows": rows,
        "csv": str(csv_path),
    }
    atomic_json_dump(payload, json_path)
    print(json.dumps(payload, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-checkpoint", type=Path, default=DEFAULT_HEAD_CHECKPOINT)
    parser.add_argument("--head-weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--internal-depth", type=int, default=8)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--times", nargs="+", type=float, default=list(DEFAULT_TIMES))
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-sit-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
