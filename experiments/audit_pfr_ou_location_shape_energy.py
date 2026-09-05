#!/usr/bin/env python3
"""Audit class-constant versus state-varying OU revision energy.

At high noise the normalized linear-flow coordinate is close to the standard
Gaussian reference measure. A degree-1 density-ratio mode has a relative score
that is constant in state for a fixed class. This audit therefore evaluates all
classes on the same Gaussian coordinate bank and decomposes every candidate
field into a class-constant component and a within-class state-varying one.

The decomposition is exact for the sampled tensors. Calling the two terms
"location" and "shape" uses the near-Gaussian Hermite interpretation and is
reported as a diagnostic, not as an unconditional identity for neural fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_ou_semigroup_controls import (  # noqa: E402
    split_raw_revision_against_ou_degree1,
)
from experiments.pfr_ou_semigroup_spectrum import (  # noqa: E402
    ou_bridge_coordinates,
    ou_degree_retiming_velocity_defect,
    transport_state_at_fixed_ou_coordinate,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
)
from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import (  # noqa: E402
    HORIZON,
    INTERVENTION_TIME,
    load_runtime,
)


def parse_times(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not result or tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    if any(not 0.0 < item < INTERVENTION_TIME for item in result):
        raise argparse.ArgumentTypeError("times must lie in (0, 0.5)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0.02,0.05,0.1,0.2,0.3,0.4"),
    )
    parser.add_argument("--horizon", type=float, default=HORIZON)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    return parser.parse_args()


@dataclass
class GroupedEnergyMoments:
    sums: torch.Tensor
    squared_norm_sums: torch.Tensor
    counts: torch.Tensor

    @classmethod
    def create(cls, num_groups: int, dimension: int) -> "GroupedEnergyMoments":
        return cls(
            sums=torch.zeros(num_groups, dimension, dtype=torch.float64),
            squared_norm_sums=torch.zeros(num_groups, dtype=torch.float64),
            counts=torch.zeros(num_groups, dtype=torch.int64),
        )

    def update(self, values: torch.Tensor, groups: torch.Tensor) -> None:
        if len(values) != len(groups):
            raise ValueError("values and groups must have the same batch size")
        flat = values.detach().float().flatten(1).cpu().double()
        group_ids = groups.detach().cpu().long()
        self.sums.index_add_(0, group_ids, flat)
        self.squared_norm_sums.index_add_(0, group_ids, flat.square().sum(1))
        self.counts.index_add_(
            0, group_ids, torch.ones_like(group_ids, dtype=torch.int64)
        )

    def summarize(self) -> tuple[dict[str, float], list[dict[str, float]]]:
        if torch.any(self.counts <= 0):
            raise RuntimeError("every group must have at least one sample")
        dimension = self.sums.shape[1]
        counts = self.counts.double()
        means = self.sums / counts[:, None]
        total_per_group = self.squared_norm_sums / (counts * dimension)
        constant_per_group = means.square().mean(1)
        varying_per_group = (total_per_group - constant_per_group).clamp_min(0.0)
        unbiased_varying_per_group = (
            counts / (counts - 1.0) * varying_per_group
        )
        unbiased_constant_per_group = (
            constant_per_group - varying_per_group / (counts - 1.0)
        )
        total_count = counts.sum()
        weights = counts / total_count
        total = (weights * total_per_group).sum()
        constant = (weights * constant_per_group).sum()
        varying = (weights * varying_per_group).sum()
        unbiased_constant = (weights * unbiased_constant_per_group).sum()
        unbiased_varying = (weights * unbiased_varying_per_group).sum()
        aggregate = {
            "total_energy": float(total),
            "class_constant_energy": float(constant),
            "state_varying_energy": float(varying),
            "class_constant_fraction": float(constant / total.clamp_min(1e-30)),
            "recomposition_error": float((total - constant - varying).abs()),
            "unbiased_class_constant_energy": float(unbiased_constant),
            "unbiased_state_varying_energy": float(unbiased_varying),
            "unbiased_class_constant_fraction": float(
                unbiased_constant / total.clamp_min(1e-30)
            ),
            "unbiased_recomposition_error": float(
                (total - unbiased_constant - unbiased_varying).abs()
            ),
        }
        per_group = [
            {
                "class": float(group),
                "samples": float(self.counts[group]),
                "total_energy": float(total_per_group[group]),
                "class_constant_energy": float(constant_per_group[group]),
                "state_varying_energy": float(varying_per_group[group]),
                "class_constant_fraction": float(
                    constant_per_group[group]
                    / total_per_group[group].clamp_min(1e-30)
                ),
                "unbiased_class_constant_energy": float(
                    unbiased_constant_per_group[group]
                ),
                "unbiased_state_varying_energy": float(
                    unbiased_varying_per_group[group]
                ),
                "unbiased_class_constant_fraction": float(
                    unbiased_constant_per_group[group]
                    / total_per_group[group].clamp_min(1e-30)
                ),
            }
            for group in range(len(self.counts))
        ]
        return aggregate, per_group


@dataclass
class GroupedProjectionMoments:
    """Sufficient statistics for pointwise and field-level projections."""

    cross_sums: torch.Tensor
    reference_energy_sums: torch.Tensor
    value_energy_sums: torch.Tensor
    pointwise_projection_energy_sums: torch.Tensor
    counts: torch.Tensor

    @classmethod
    def create(cls, num_groups: int) -> "GroupedProjectionMoments":
        return cls(
            cross_sums=torch.zeros(num_groups, dtype=torch.float64),
            reference_energy_sums=torch.zeros(num_groups, dtype=torch.float64),
            value_energy_sums=torch.zeros(num_groups, dtype=torch.float64),
            pointwise_projection_energy_sums=torch.zeros(
                num_groups, dtype=torch.float64
            ),
            counts=torch.zeros(num_groups, dtype=torch.int64),
        )

    def update(
        self,
        values: torch.Tensor,
        references: torch.Tensor,
        groups: torch.Tensor,
    ) -> None:
        if values.shape != references.shape or len(values) != len(groups):
            raise ValueError("values, references and groups must have matching batches")
        value_flat = values.detach().float().flatten(1).cpu().double()
        reference_flat = references.detach().float().flatten(1).cpu().double()
        group_ids = groups.detach().cpu().long()
        cross = (value_flat * reference_flat).sum(1)
        reference_energy = reference_flat.square().sum(1)
        value_energy = value_flat.square().sum(1)
        pointwise_projection_energy = cross.square() / reference_energy.clamp_min(
            1e-30
        )
        self.cross_sums.index_add_(0, group_ids, cross)
        self.reference_energy_sums.index_add_(0, group_ids, reference_energy)
        self.value_energy_sums.index_add_(0, group_ids, value_energy)
        self.pointwise_projection_energy_sums.index_add_(
            0, group_ids, pointwise_projection_energy
        )
        self.counts.index_add_(
            0, group_ids, torch.ones_like(group_ids, dtype=torch.int64)
        )

    def summarize(self) -> tuple[dict[str, float], list[dict[str, float]]]:
        if torch.any(self.counts <= 0):
            raise RuntimeError("every group must have at least one sample")
        reference = self.reference_energy_sums.clamp_min(1e-30)
        value = self.value_energy_sums.clamp_min(1e-30)
        class_coefficients = self.cross_sums / reference
        class_projection_energy = self.cross_sums.square() / reference
        total_value_energy = value.sum()
        total_reference_energy = reference.sum()
        total_cross = self.cross_sums.sum()
        pointwise_projection_energy = self.pointwise_projection_energy_sums.sum()
        classwise_projection_energy = class_projection_energy.sum()
        global_projection_energy = total_cross.square() / total_reference_energy
        aggregate = {
            "samples": int(self.counts.sum()),
            "groups": int(len(self.counts)),
            "pointwise_explained_fraction": float(
                pointwise_projection_energy / total_value_energy
            ),
            "classwise_explained_fraction": float(
                classwise_projection_energy / total_value_energy
            ),
            "global_explained_fraction": float(
                global_projection_energy / total_value_energy
            ),
            "global_coefficient": float(total_cross / total_reference_energy),
            "class_coefficient_mean": float(class_coefficients.mean()),
            "class_coefficient_std": float(class_coefficients.std(unbiased=True)),
            "class_coefficient_min": float(class_coefficients.min()),
            "class_coefficient_max": float(class_coefficients.max()),
        }
        per_group = [
            {
                "class": int(group),
                "samples": int(self.counts[group]),
                "coefficient": float(class_coefficients[group]),
                "explained_fraction": float(
                    class_projection_energy[group] / value[group]
                ),
                "pointwise_explained_fraction": float(
                    self.pointwise_projection_energy_sums[group] / value[group]
                ),
            }
            for group in range(len(self.counts))
        ]
        return aggregate, per_group


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples_per_class < 2 or args.batch_size <= 0:
        raise ValueError("samples-per-class must be at least two and batch-size positive")
    if not 0.0 < args.horizon < INTERVENTION_TIME:
        raise ValueError("horizon must lie in (0, 0.5)")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime, allocator = load_runtime(
        repo=detect_repo(),
        data=detect_data(),
        adm_python=detect_adm_python(),
        device=device,
        allocator_limit_gib=args.cuda_allocator_limit_gib,
    )
    num_classes = int(runtime.modules["NUM_CLASSES"])
    latent_shape = tuple(runtime.modules["LATENT_SHAPE"])
    dimension = int(torch.tensor(latent_shape).prod().item())

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    gaussian_bank = torch.randn(
        args.samples_per_class,
        *latent_shape,
        generator=generator,
        dtype=torch.float32,
    )
    gaussian_hash = hashlib.sha256(
        gaussian_bank.contiguous().numpy().tobytes()
    ).hexdigest()
    total_samples = num_classes * args.samples_per_class

    aggregate_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    per_class_projection_rows: list[dict[str, Any]] = []
    metric_names = (
        "raw_revision",
        "degree1_defect",
        "degree1_common",
        "degree1_unique",
        "degree2_defect",
        "degree2_common",
        "degree2_unique",
    )

    for time_value in args.times:
        future_value = time_value + min(
            float(args.horizon), INTERVENTION_TIME - time_value
        )
        moments = {
            name: GroupedEnergyMoments.create(num_classes, dimension)
            for name in metric_names
        }
        projection_moments = GroupedProjectionMoments.create(num_classes)
        for start in range(0, total_samples, args.batch_size):
            stop = min(start + args.batch_size, total_samples)
            indices = torch.arange(start, stop, dtype=torch.int64)
            labels_cpu = torch.div(
                indices, args.samples_per_class, rounding_mode="floor"
            )
            bank_indices = indices.remainder(args.samples_per_class)
            normalized_state = gaussian_bank[bank_indices].to(device)
            labels = labels_cpu.to(device)
            time = torch.full((len(indices),), time_value, device=device)
            future_time = torch.full((len(indices),), future_value, device=device)
            current_scale = ou_bridge_coordinates(time, normalized_state).scale
            state = current_scale * normalized_state

            weak = runtime.evaluate_weak(time, state, labels)
            weak_future_raw = runtime.evaluate_weak(future_time, state, labels)
            future_ou_state = transport_state_at_fixed_ou_coordinate(
                state, time, future_time
            )
            weak_future_ou = runtime.evaluate_weak(
                future_time, future_ou_state, labels
            )

            raw = weak - weak_future_raw
            degree1 = ou_degree_retiming_velocity_defect(
                weak,
                weak_future_ou,
                state,
                time,
                future_time,
                degree=1.0,
            )
            degree2 = ou_degree_retiming_velocity_defect(
                weak,
                weak_future_ou,
                state,
                time,
                future_time,
                degree=2.0,
            )
            split1 = split_raw_revision_against_ou_degree1(raw, degree1)
            split2 = split_raw_revision_against_ou_degree1(raw, degree2)
            values = {
                "raw_revision": raw,
                "degree1_defect": degree1,
                "degree1_common": split1.common,
                "degree1_unique": split1.unique,
                "degree2_defect": degree2,
                "degree2_common": split2.common,
                "degree2_unique": split2.unique,
            }
            for name, value in values.items():
                moments[name].update(value, labels_cpu)
            projection_moments.update(raw, degree1, labels_cpu)

        for name, accumulator in moments.items():
            aggregate, per_class = accumulator.summarize()
            aggregate_rows.append(
                {
                    "time": time_value,
                    "future_time": future_value,
                    "metric": name,
                    "samples": total_samples,
                    "samples_per_class": args.samples_per_class,
                    **aggregate,
                }
            )
            per_class_rows.extend(
                {
                    "time": time_value,
                    "future_time": future_value,
                    "metric": name,
                    **row,
                }
                for row in per_class
            )
        projection, per_class_projection = projection_moments.summarize()
        projection_rows.append(
            {
                "time": time_value,
                "future_time": future_value,
                **projection,
            }
        )
        per_class_projection_rows.extend(
            {
                "time": time_value,
                "future_time": future_value,
                **row,
            }
            for row in per_class_projection
        )
        print(
            json.dumps(
                {
                    "event": "time_complete",
                    "time": time_value,
                    "degree1_constant_fraction": aggregate_rows[-6][
                        "unbiased_class_constant_fraction"
                    ],
                    "degree2_constant_fraction": aggregate_rows[-3][
                        "unbiased_class_constant_fraction"
                    ],
                }
            ),
            flush=True,
        )

    for path, rows in (
        (output / "location_shape_energy.csv", aggregate_rows),
        (output / "per_class_location_shape_energy.csv", per_class_rows),
        (output / "projection_granularity.csv", projection_rows),
        (
            output / "per_class_projection_granularity.csv",
            per_class_projection_rows,
        ),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "format": "eqvae_pfr_ou_location_shape_energy_v1",
        "scope": (
            "exact sampled class-constant/state-varying decomposition; "
            "location/shape semantics require the high-noise Hermite approximation"
        ),
        "protocol": {
            "samples_per_class": args.samples_per_class,
            "num_classes": num_classes,
            "total_samples": total_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "times": list(args.times),
            "horizon": args.horizon,
            "state_measure": "same standard-Gaussian OU-coordinate bank for every class",
            "gaussian_bank_sha256": gaussian_hash,
            "strong": str(runtime.paths["strong"]),
            "weak": str(runtime.paths["depth4"]),
            "weights": "ema",
        },
        "allocator": allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "summary": str(output / "location_shape_energy.csv"),
        "per_class": str(output / "per_class_location_shape_energy.csv"),
        "projection_granularity": str(output / "projection_granularity.csv"),
        "per_class_projection_granularity": str(
            output / "per_class_projection_granularity.csv"
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
