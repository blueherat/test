#!/usr/bin/env python3
"""Audit the score conservativity assumed by counterfactual-ratio IG.

The deployed PFR field treats ``Q(z,t)=W(q(z,t),t+h)`` as a current-time
velocity reference.  A literal density-ratio interpretation requires the
corresponding current-time score fields to be conservative.  This script
measures Jacobian symmetry for ``S``, ``W``, ``Q``, their contrasts, ordinary
IG, and PFR on paired teacher and unguided strong-rollout states.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_imagenet100_sit_guidance_density_action import (  # noqa: E402
    component_jacobian_symmetry_probe,
)
from experiments.internal_guidance_path_extrapolation import (  # noqa: E402
    affine_counterfactual_ratio_velocity,
    project_to_forward_ray,
    split_internal_guidance,
)
from experiments.run_imagenet100_sit_guidance_conservativity import (  # noqa: E402
    _collect_rollout_states,
    _rademacher_like,
    _teacher_states,
    _validation_bank,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    load_repo_modules,
)
from experiments.run_prediction_target_extrapolation_toy_v4 import (  # noqa: E402
    parse_float_list,
)


FIELD_NAMES = (
    "strong",
    "weak",
    "counterfactual_weak",
    "depth_gap",
    "counterfactual_ratio_gap",
    "ordinary_ig",
    "pfr",
    "pfr_revision",
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--times",
        type=parse_float_list,
        default=parse_float_list("0.125,0.25,0.375,0.46875"),
    )
    parser.add_argument("--horizon", type=float, default=0.03125)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--heun-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--probe-seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def gamma_at(time_value: float) -> float:
    if time_value < 0.25:
        return 0.6
    if time_value < 0.5:
        return 0.7
    return 0.0


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_compatible(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "field_rms",
        "jvp_rms",
        "vjp_rms",
        "antisymmetric_rms",
        "antisymmetric_over_jvp_rms",
        "antisymmetric_energy_fraction",
        "jvp_vjp_cosine",
    )
    output: list[dict[str, object]] = []
    keys = sorted(
        {(str(row["source"]), float(row["time"]), str(row["field"])) for row in rows}
    )
    for source, time_value, field in keys:
        selected = [
            row
            for row in rows
            if row["source"] == source
            and float(row["time"]) == time_value
            and row["field"] == field
        ]
        result: dict[str, object] = {
            "source": source,
            "time": time_value,
            "field": field,
            "observations": len(selected),
        }
        for metric in metrics:
            values = torch.tensor([float(row[metric]) for row in selected])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_median"] = float(values.median())
        output.append(result)
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.probes <= 0:
        raise ValueError("sample, batch, and probe counts must be positive")
    if args.num_samples % args.batch_size:
        raise ValueError("num-samples must be divisible by batch-size")
    if any(not 0.0 < value < 0.5 for value in args.times):
        raise ValueError("times must lie in (0, 0.5)")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.official_sit_repo = args.official_sit_repo.expanduser().resolve()

    modules = load_repo_modules(REPO_ROOT)
    sit_module, source_metadata = modules["load_official_sit_module"](
        args.official_sit_repo,
        verify_source=True,
    )
    strong_path = (
        args.data_root / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
    )
    head_path = (
        args.data_root
        / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
    )
    strong, semantics, strong_metadata = modules["load_sit_field_model"](
        checkpoint_path=strong_path,
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("PFR conservativity audit requires native velocity")
    head = modules["load_internal_head_for_source"](
        checkpoint_path=head_path,
        name="depth4_v",
        head_weights="ema",
        model=strong,
        sit_module=sit_module,
        source_checkpoint_path=strong_path,
        source_metadata=source_metadata,
        device=device,
    )
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_internal_head_only,
    )

    bank = _validation_bank(
        args.cache_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    rows: list[dict[str, object]] = []
    probe_generator = torch.Generator(device="cpu").manual_seed(args.probe_seed)
    started = time.perf_counter()

    def evaluate_pair(state: torch.Tensor, time_tensor: torch.Tensor, labels: torch.Tensor):
        times = time_tensor.expand(len(state))
        with sdpa_kernel(SDPBackend.MATH):
            full, trained, _ = modules["evaluate_source_with_heads"](
                strong,
                state,
                times,
                labels,
                heads={"depth4_v": head},
            )
        return full, trained["depth4_v"]

    for start in range(0, args.num_samples, args.batch_size):
        stop = start + args.batch_size
        labels = bank["labels"][start:stop].to(device)
        clean = bank["clean"][start:stop].to(device)
        noise = bank["bridge_noise"][start:stop].to(device)
        teacher_states = _teacher_states(clean, noise, args.times)

        def strong_field(time_tensor: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return evaluate_pair(state, time_tensor, labels)[0]

        rollout_states = _collect_rollout_states(
            strong_field,
            noise,
            steps=args.heun_steps,
            requested_times=list(args.times),
        )
        for source, states in (
            ("teacher", teacher_states),
            ("strong_rollout", rollout_states),
        ):
            for time_value in args.times:
                state = states[float(time_value)].detach()
                time_tensor = torch.tensor(float(time_value), device=device)
                horizon = min(args.horizon, 0.5 - float(time_value))
                gamma = gamma_at(float(time_value))

                def score_components(current_state: torch.Tensor) -> torch.Tensor:
                    full, weak = evaluate_pair(current_state, time_tensor, labels)
                    weak_base, calibration = split_internal_guidance(
                        full,
                        weak,
                        gamma=gamma,
                    )
                    ordinary = weak_base + calibration
                    projected = project_to_forward_ray(calibration, ordinary)
                    query_state = current_state + horizon * projected.parallel
                    query_time = time_tensor + time_tensor.new_tensor(horizon)
                    with sdpa_kernel(SDPBackend.MATH):
                        query = evaluate_internal_head_only(
                            strong,
                            query_state,
                            query_time.expand(len(current_state)),
                            labels,
                            spec=head,
                        )
                    pfr = affine_counterfactual_ratio_velocity(
                        full,
                        weak_base,
                        (query,),
                        (1.0,),
                        gamma=gamma,
                    )
                    score_scale = time_tensor / (1.0 - time_tensor)
                    score_offset = -current_state / (1.0 - time_tensor)

                    def complete_field_score(velocity: torch.Tensor) -> torch.Tensor:
                        return score_offset + score_scale * velocity

                    # Complete affine fields have coefficient sum one and retain
                    # the common score offset. Contrasts have coefficient sum
                    # zero, so that offset cancels exactly.
                    return torch.stack(
                        (
                            complete_field_score(full),
                            complete_field_score(weak),
                            complete_field_score(query),
                            score_scale * (full - weak),
                            score_scale * (full - query),
                            complete_field_score(ordinary),
                            complete_field_score(pfr),
                            score_scale * (pfr - ordinary),
                        )
                    )

                for probe_index in range(args.probes):
                    probe = _rademacher_like(state, probe_generator)
                    metrics = component_jacobian_symmetry_probe(
                        score_components,
                        state,
                        probe,
                        list(range(len(FIELD_NAMES))),
                    )
                    for field_index, field_name in enumerate(FIELD_NAMES):
                        for local_index in range(args.batch_size):
                            row: dict[str, object] = {
                                "sample_id": start + local_index,
                                "source": source,
                                "time": float(time_value),
                                "probe": probe_index,
                                "field": field_name,
                            }
                            for metric, values in metrics.items():
                                row[metric] = float(
                                    values[field_index, local_index].detach().cpu()
                                )
                            rows.append(row)
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "samples": [start, stop],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    aggregate = summarize(rows)
    write_csv(rows, args.output_root / "per_sample_probe_metrics.csv")
    write_csv(aggregate, args.output_root / "metrics_by_source_time.csv")
    summary = {
        "format": "eqvae_imagenet100_sit_pfr_conservativity_v1",
        "args": vars(args),
        "fields": FIELD_NAMES,
        "strong": strong_metadata,
        "head": {
            "checkpoint": str(head_path),
            "depth": 4,
            "prediction_target": "velocity",
        },
        "score_interpretation": (
            "each numerical field is mapped through the current-time affine "
            "velocity-to-score relation before Jacobian symmetry is tested"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate": aggregate,
    }
    summary = json_compatible(summary)
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"event": "complete", "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
