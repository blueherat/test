#!/usr/bin/env python3
"""Test whether PFR benefits from a deliberately off-channel weak query.

For the straight conditional path

    Z_t = (1 - t) E + t X,        U = X - E,

the same particle at ``t+h`` is exactly ``Z_t + h U``.  PFR's time-only
query instead presents the unchanged state ``Z_t`` with the later clock.
This audit interpolates between the two:

    Q_rho = Z_t + rho h U,        rho in [0, 1].

Thus ``rho=0`` is maximally clock/state-mismatched within this family and
``rho=1`` is the exact teacher-path future.  The audit measures prediction
errors only; it does not use these unavailable teacher targets at sampling
time and therefore is not a generation algorithm.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_imagenet100_sit_conserved_velocity_revision import (  # noqa: E402
    gamma_at,
    load_validation_latents,
    per_sample_cosine,
    per_sample_mse,
    summarize,
)
from experiments.internal_guidance_path_extrapolation import (  # noqa: E402
    project_to_forward_ray,
    split_internal_guidance,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    load_repo_modules,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import (  # noqa: E402
    parse_float_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae"
        ),
    )
    parser.add_argument(
        "--official-sit-repo",
        type=Path,
        default=Path("/home/zhoushunyu/data/research_repos/SiT"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--times",
        type=parse_float_list,
        default=parse_float_list("0.05,0.1,0.2,0.3,0.4,0.46875"),
    )
    parser.add_argument(
        "--rhos",
        type=parse_float_list,
        default=parse_float_list("0,0.125,0.25,0.5,0.75,1"),
    )
    parser.add_argument("--horizon", type=float, default=1.0 / 32.0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def teacher_channel_query(
    state: torch.Tensor,
    conserved_velocity: torch.Tensor,
    *,
    horizon: float,
    rho: float,
) -> torch.Tensor:
    """Interpolate from a clock-only query to the exact particle future."""

    if state.shape != conserved_velocity.shape:
        raise ValueError("state and conserved_velocity must have identical shapes")
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if float(horizon) < 0.0:
        raise ValueError("horizon must be non-negative")
    return state + float(rho) * float(horizon) * conserved_velocity


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(1).sqrt()


def summarize_metrics(
    values: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    row: dict[str, float] = {}
    for name, chunks in values.items():
        for statistic, value in summarize(chunks).items():
            row[f"{name}_{statistic}"] = value
    return row


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if not 0.0 < args.horizon < 0.5:
        raise ValueError("horizon must lie in (0, 0.5)")
    if any(not 0.0 < value < 0.5 for value in args.times):
        raise ValueError("times must lie in (0, 0.5)")
    if any(not 0.0 <= value <= 1.0 for value in args.rhos):
        raise ValueError("rhos must lie in [0, 1]")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    official_repo = args.official_sit_repo.expanduser().resolve()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    modules = load_repo_modules(REPO_ROOT)
    paths = {
        "strong": data_root
        / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt",
        "depth4": data_root
        / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))

    sit_module, source_metadata = modules["load_official_sit_module"](
        official_repo, verify_source=True
    )
    strong, semantics, strong_metadata = modules["load_sit_field_model"](
        checkpoint_path=paths["strong"],
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("source checkpoint must predict native velocity")
    head = modules["load_internal_head_for_source"](
        checkpoint_path=paths["depth4"],
        name="depth4_v",
        head_weights="ema",
        model=strong,
        sit_module=sit_module,
        source_checkpoint_path=paths["strong"],
        source_metadata=source_metadata,
        device=device,
    )
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_internal_head_only,
    )

    clean, flow_noise, labels = load_validation_latents(
        cache_dir,
        samples=args.samples,
        seed=args.seed,
        scaling_factor=float(modules["SD_VAE_SCALING_FACTOR"]),
        device=device,
    )
    target = clean - flow_noise
    rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []

    for time_value in args.times:
        horizon = min(float(args.horizon), 0.5 - float(time_value))
        query_time_value = float(time_value) + horizon
        gamma = gamma_at(float(time_value))
        beta = 1.0 + gamma
        by_rho: dict[float, dict[str, list[torch.Tensor]]] = {
            float(rho): {} for rho in args.rhos
        }

        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            x = clean[start:stop]
            e = flow_noise[start:stop]
            y = labels[start:stop]
            u = target[start:stop]
            time = torch.full((len(x),), float(time_value), device=device)
            time_image = time.reshape(-1, 1, 1, 1)
            state = (1.0 - time_image) * e + time_image * x
            full, heads, _ = modules["evaluate_source_with_heads"](
                strong,
                state,
                time,
                y,
                heads={"depth4_v": head},
            )
            weak = heads["depth4_v"]
            _, calibration = split_internal_guidance(full, weak, gamma=gamma)
            ordinary = weak + calibration
            projected = project_to_forward_ray(calibration, ordinary)
            exact_future = state + horizon * u
            query_time = torch.full(
                (len(x),), query_time_value, device=device
            )

            for rho_value in args.rhos:
                rho = float(rho_value)
                query_state = teacher_channel_query(
                    state, u, horizon=horizon, rho=rho
                )
                weak_query = evaluate_internal_head_only(
                    strong,
                    query_state,
                    query_time,
                    y,
                    spec=head,
                )
                revision = weak - weak_query
                candidate = ordinary + beta * revision
                mismatch = query_state - exact_future
                metrics = by_rho[rho]
                for name, value in {
                    "weak_query_mse": per_sample_mse(weak_query, u),
                    "candidate_mse": per_sample_mse(candidate, u),
                    "revision_rms": sample_rms(revision),
                    "channel_mismatch_rms": sample_rms(mismatch),
                    "revision_weak_error_cosine": per_sample_cosine(
                        revision, u - weak
                    ),
                    "revision_strong_error_cosine": per_sample_cosine(
                        revision, u - full
                    ),
                    "revision_depth_gap_cosine": per_sample_cosine(
                        revision, full - weak
                    ),
                    "query_shift_projected_cosine": per_sample_cosine(
                        query_state - state, projected.parallel
                    ),
                }.items():
                    metrics.setdefault(name, []).append(value)

                sample_values = {
                    name: value.detach().float().cpu()
                    for name, value in {
                        "weak_query_mse": per_sample_mse(weak_query, u),
                        "candidate_mse": per_sample_mse(candidate, u),
                        "revision_rms": sample_rms(revision),
                        "channel_mismatch_rms": sample_rms(mismatch),
                    }.items()
                }
                for offset in range(len(x)):
                    per_sample_rows.append(
                        {
                            "time": float(time_value),
                            "query_time": query_time_value,
                            "rho": rho,
                            "sample": start + offset,
                            **{
                                name: float(value[offset])
                                for name, value in sample_values.items()
                            },
                        }
                    )

        for rho in map(float, args.rhos):
            row: dict[str, Any] = {
                "time": float(time_value),
                "query_time": query_time_value,
                "horizon": horizon,
                "rho": rho,
                "gamma": gamma,
                "beta": beta,
                "samples": args.samples,
            }
            row.update(summarize_metrics(by_rho[rho]))
            rows.append(row)
        print(
            json.dumps(
                {
                    "event": "time_complete",
                    "time": float(time_value),
                    "mse_rho0": rows[-len(args.rhos)]["weak_query_mse_mean"],
                    "mse_rho1": rows[-1]["weak_query_mse_mean"],
                }
            ),
            flush=True,
        )

    with (output / "channel_consistency_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_sample_channel_consistency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0]))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    manifest = {
        "format": "eqvae_pfr_channel_consistency_audit_v1",
        "scope": "teacher-path diagnostic; oracle U is never available to the sampler",
        "query_family": "Q_rho=Z_t+rho*h*U at time t+h",
        "rho_endpoints": {
            "0": "unchanged state with a later clock",
            "1": "exact same-particle future on the straight conditional path",
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "strong": strong_metadata,
        "head": {
            "checkpoint": head.checkpoint,
            "checkpoint_sha256": head.checkpoint_sha256,
            "depth": head.depth,
            "prediction_target": head.prediction_target,
        },
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
