"""Measure inverse-conditioning errors induced by RAE layerwise data paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external" / "RAE" / "src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_clean_estimate_trajectory import (  # noqa: E402
    PATHS,
    endpoint_observation_factors,
    invert_path_endpoint_observation,
)
from experiments.rae_cycle_direction_intervention import sample_rms  # noqa: E402
from experiments.rae_layerwise_path import random_detail_basis, spatial_center  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset, load_cache_manifest  # noqa: E402
from experiments.rae_path_difference_intervention import (  # noqa: E402
    projected_frechet_distance,
    random_unit_directions,
    standardized_sliced_wasserstein,
)
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    load_models,
    official_time_grid,
)
from experiments.run_rae_clean_estimate_trajectory_probe import (  # noqa: E402
    DEFAULT_BRANCH_ROOT,
    DEFAULT_CACHE,
    DEFAULT_ENDPOINT_ROOT,
    _clean_estimate_rollout,
    _endpoint,
    _paired_noise_and_labels,
)
from experiments.run_rae_path_difference_intervention import (  # noqa: E402
    _decode_probe,
    _load_inception_probe,
)


DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_path_conditioning"
SELECTED_STEPS = (8, 16, 24, 32, 40, 44, 46, 48)
DECODE_STEPS = (32, 40, 44, 46, 48, 49)
ORACLE_STEPS = (32, 40, 44)


@dataclass(frozen=True)
class ConditioningConfig:
    branch_root: Path = DEFAULT_BRANCH_ROOT
    endpoint_root: Path = DEFAULT_ENDPOINT_ROOT
    audit_cache: Path = DEFAULT_CACHE
    output_root: Path = DEFAULT_OUTPUT
    run_name: str = "explore_seed20260718_start128_n32"
    endpoint_seed: int = 20_260_718
    evaluation_seed: int = 20_260_721
    endpoint_total: int = 256
    sample_start: int = 128
    count: int = 32
    batch_size: int = 2
    reference_start: int = 1024
    projection_dim: int = 32
    swd_directions: int = 64


def _distributed(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        raise RuntimeError("launch this probe with torchrun")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configure_fp32(int(seed) * world_size + rank)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rank, world_size, device


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_manifest_and_basis(branch: Path) -> tuple[dict[str, object], torch.Tensor]:
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    rank = int(manifest["subspace_rank"])
    if manifest["subspace_kind"] == "random_energy_matched":
        basis = random_detail_basis(
            768, rank, seed=int(manifest["random_subspace_seed"])
        )
    else:
        payload = torch.load(
            Path(str(manifest["subspace_path"])), map_location="cpu", weights_only=False
        )
        entry = payload["subspaces"].get(rank, payload["subspaces"].get(str(rank)))
        if entry is None:
            raise KeyError(f"rank {rank} is absent from subspace payload")
        basis = entry["basis"].float()
    digest = hashlib.sha256(basis.contiguous().numpy().tobytes()).hexdigest()
    if digest != str(manifest["basis_sha256"]):
        raise RuntimeError("loaded basis does not match branch manifest")
    return manifest, basis


def _basis_projection(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    _, residual = spatial_center(value.float())
    basis = basis.to(value)
    rows = residual.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
    projected = (rows @ basis) @ basis.transpose(0, 1)
    return projected.reshape(
        value.shape[0], value.shape[2], value.shape[3], value.shape[1]
    ).permute(0, 3, 1, 2).contiguous()


def _component_errors(
    estimate: torch.Tensor,
    endpoint: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    estimate_basis = _basis_projection(estimate, basis)
    endpoint_basis = _basis_projection(endpoint, basis)
    estimate_semantic = estimate - estimate_basis
    endpoint_semantic = endpoint - endpoint_basis
    return {
        "total": sample_rms(estimate - endpoint) / sample_rms(endpoint, 1e-12),
        "semantic": sample_rms(estimate_semantic - endpoint_semantic)
        / sample_rms(endpoint_semantic, 1e-12),
        "basis": sample_rms(estimate_basis - endpoint_basis)
        / sample_rms(endpoint_basis, 1e-12),
    }


@torch.no_grad()
def _reference_features(
    decoder: torch.nn.Module,
    inception: torch.nn.Module,
    cache: Path,
    *,
    start: int,
    count: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = CachedRAELatentDataset(cache, start=start, stop=start + count)
    latents = torch.stack([dataset[index][0] for index in range(len(dataset))])
    _, _, features, _ = _decode_probe(
        decoder, inception, latents, batch_size=batch_size, device=device
    )
    return features


def _prediction_summary(
    metrics: pd.DataFrame,
    distribution: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    medians = (
        metrics.groupby(["path", "step_index"], as_index=False)
        .agg(
            semantic_error=("semantic_relative_error", "median"),
            basis_error=("basis_relative_error", "median"),
            total_error=("total_relative_error", "median"),
            semantic_factor=("semantic_factor_abs", "first"),
            basis_factor=("basis_factor_abs", "first"),
        )
    )
    indexed = medians.set_index(["path", "step_index"])
    p1 = all(
        indexed.loc[("reverse", step), "semantic_error"]
        > indexed.loc[("annealed", step), "semantic_error"]
        for step in (32, 40)
    )
    annealed = medians[medians.path == "annealed"]
    p2 = float((annealed.basis_error > annealed.semantic_error).mean()) > 0.5
    random = indexed.loc["random"]
    p3 = bool(
        random.loc[44, "basis_error"]
        > max(random.loc[40, "basis_error"], random.loc[46, "basis_error"])
    )
    corrected_quality = distribution[distribution.method == "corrected"]
    quality = corrected_quality.pivot(
        index="step_index", columns="path", values="projected_frechet"
    )
    p4 = all(
        float(quality.loc[step, ["static", "annealed"]].mean())
        < float(quality.loc[step, "reverse"])
        for step in (32, 40)
    )
    oracle = distribution[distribution.step_index.isin((32, 40))].pivot(
        index=["path", "step_index"], columns="method", values="endpoint_feature_rms"
    )
    p5_reverse = all(
        oracle.loc[("reverse", step), "semantic_oracle"]
        < oracle.loc[("reverse", step), "basis_oracle"]
        for step in (32, 40)
    )
    p5_annealed = all(
        oracle.loc[("annealed", step), "basis_oracle"]
        < oracle.loc[("annealed", step), "semantic_oracle"]
        for step in (32, 40)
    )
    return medians, {
        "p1_reverse_semantic_lag": bool(p1),
        "p2_annealed_detail_localized": bool(p2),
        "p3_random_zero_crossing_spike": bool(p3),
        "p4_decoder_ordering": bool(p4),
        "p5_reverse_semantic_oracle": bool(p5_reverse),
        "p5_annealed_basis_oracle": bool(p5_annealed),
    }


def _plot(metrics: pd.DataFrame, distribution: pd.DataFrame, output: Path) -> None:
    summary = metrics.groupby(["path", "step_index"], as_index=False).agg(
        total=("total_relative_error", "median"),
        semantic=("semantic_relative_error", "median"),
        basis=("basis_relative_error", "median"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    for axis, path in zip(axes.flat, PATHS):
        rows = summary[summary.path == path].sort_values("step_index")
        for column, marker in (("total", "o"), ("semantic", "s"), ("basis", "^")):
            axis.plot(rows.step_index, rows[column], marker=marker, label=column)
        axis.set_yscale("log")
        axis.set_title(path)
        axis.set_xlabel("Euler state index")
        axis.set_ylabel("Relative error to final generated endpoint")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Path-aware endpoint inversion error", fontsize=18)
    figure.savefig(output / "conditioning_errors.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for path in PATHS:
        rows = distribution[
            (distribution.path == path) & (distribution.method == "corrected")
        ].sort_values("step_index")
        axis.plot(
            rows.step_index,
            rows.projected_frechet,
            marker="o",
            linewidth=2,
            label=path,
        )
    axis.set_title("Decoded path-aware endpoint estimates")
    axis.set_xlabel("Euler state index")
    axis.set_ylabel("Projected Frechet (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output / "conditioning_decoder_proxy.png", dpi=180)
    plt.close(figure)


def run(config: ConditioningConfig) -> Path | None:
    rank, world_size, device = _distributed(config.evaluation_seed)
    if world_size != len(PATHS):
        raise ValueError(f"expected {len(PATHS)} GPUs, got {world_size}")
    path = PATHS[rank]
    output = config.output_root.expanduser().resolve() / config.run_name
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    branch = config.branch_root.expanduser().resolve() / f"seed3407_{path}_rank16_s0_to_10000"
    manifest, basis = _load_manifest_and_basis(branch)
    endpoint_payload = _endpoint(config.endpoint_root.expanduser().resolve(), path)
    initial, labels = _paired_noise_and_labels(
        endpoint_payload,
        seed=config.endpoint_seed,
        total=config.endpoint_total,
        start=config.sample_start,
        count=config.count,
    )
    endpoint = endpoint_payload["latents"][
        config.sample_start : config.sample_start + config.count
    ].float()
    model, decoder, model_config = load_models(branch, device)
    inception = _load_inception_probe(device)
    shift = math.sqrt(
        float(model_config.misc.time_dist_shift_dim)
        / float(model_config.misc.time_dist_shift_base)
    )
    times = official_time_grid(50, time_shift=shift).to(device)
    started = perf_counter()
    observations, reproduced_endpoint = _clean_estimate_rollout(
        model,
        initial,
        labels,
        times,
        selected_steps=SELECTED_STEPS,
        batch_size=config.batch_size,
        device=device,
    )
    reproduction_relative = float(
        sample_rms(reproduced_endpoint - endpoint).mean()
        / sample_rms(endpoint, 1e-12).mean()
    )
    if reproduction_relative > 1e-6:
        raise RuntimeError(f"endpoint reproduction failed: {reproduction_relative}")

    mode = str(manifest["path_mode"])
    power = float(manifest["path_power"])
    family = str(manifest.get("path_family", "power"))
    floor = float(manifest.get("path_floor", 0.0))
    alpha = float(manifest.get("path_alpha", 1.0))
    detail_scale = float(manifest["detail_scale"])
    rows = []
    corrected: dict[int, torch.Tensor] = {}
    for step, observation in observations.items():
        batch_time = torch.full((len(observation),), float(times[step]))
        estimate = invert_path_endpoint_observation(
            observation,
            batch_time,
            basis,
            mode=mode,
            power=power,
            family=family,
            floor=floor,
            alpha=alpha,
            detail_scale=detail_scale,
        )
        corrected[step] = estimate
        semantic_factor, detail_factor = endpoint_observation_factors(
            batch_time,
            mode,
            power=power,
            family=family,
            floor=floor,
            alpha=alpha,
        )
        basis_factor = semantic_factor * (1.0 - detail_scale) + detail_factor * detail_scale
        errors = _component_errors(estimate, endpoint, basis)
        for offset in range(len(estimate)):
            rows.append(
                {
                    "path": path,
                    "sample_index": config.sample_start + offset,
                    "step_index": step,
                    "time": float(times[step]),
                    "semantic_factor_abs": float(semantic_factor[offset].abs()),
                    "basis_factor_abs": float(basis_factor[offset].abs()),
                    "total_relative_error": float(errors["total"][offset]),
                    "semantic_relative_error": float(errors["semantic"][offset]),
                    "basis_relative_error": float(errors["basis"][offset]),
                    "estimate_rms": float(sample_rms(estimate)[offset]),
                }
            )
    corrected[49] = endpoint

    cache = config.audit_cache.expanduser().resolve()
    cache_manifest = load_cache_manifest(cache)
    if config.reference_start + config.count > int(cache_manifest["sample_count"]):
        raise ValueError("clean reference slice exceeds cache")
    reference = torch.zeros((config.count, 2048), device=device, dtype=torch.float32)
    if rank == 0:
        reference.copy_(
            _reference_features(
                decoder,
                inception,
                cache,
                start=config.reference_start,
                count=config.count,
                batch_size=config.batch_size,
                device=device,
            ).to(device)
        )
    dist.broadcast(reference, src=0)
    endpoint_basis = _basis_projection(endpoint, basis)
    endpoint_semantic = endpoint - endpoint_basis
    decode_candidates: dict[tuple[int, str], torch.Tensor] = {
        (49, "corrected"): endpoint,
    }
    for step in DECODE_STEPS:
        if step == 49:
            continue
        estimate = corrected[step]
        decode_candidates[(step, "corrected")] = estimate
        if step in ORACLE_STEPS:
            estimate_basis = _basis_projection(estimate, basis)
            estimate_semantic = estimate - estimate_basis
            decode_candidates[(step, "semantic_oracle")] = (
                endpoint_semantic + estimate_basis
            )
            decode_candidates[(step, "basis_oracle")] = (
                estimate_semantic + endpoint_basis
            )
    features = {}
    for (step, method), candidate in decode_candidates.items():
        _, _, value, _ = _decode_probe(
            decoder,
            inception,
            candidate,
            batch_size=config.batch_size,
            device=device,
        )
        features[f"step{step:02d}_{method}"] = value
    pd.DataFrame(rows).to_csv(output / f"conditioning_rank{rank:02d}.csv", index=False)
    torch.save(
        {
            "path": path,
            "features": features,
            "reproduction_relative_rms": reproduction_relative,
            "manifest": {
                "path_mode": mode,
                "path_power": power,
                "path_family": family,
                "path_floor": floor,
                "path_alpha": alpha,
                "detail_scale": detail_scale,
                "subspace_kind": manifest["subspace_kind"],
            },
        },
        output / f"features_rank{rank:02d}.pt",
    )
    print(f"rank{rank} {path} complete in {(perf_counter()-started)/60:.1f}m", flush=True)
    dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return None

    metrics = pd.concat(
        [pd.read_csv(output / f"conditioning_rank{index:02d}.csv") for index in range(world_size)],
        ignore_index=True,
    )
    metrics.to_csv(output / "conditioning_metrics.csv", index=False)
    projection = random_unit_directions(2048, config.projection_dim, config.evaluation_seed + 401)
    directions = random_unit_directions(2048, config.swd_directions, config.evaluation_seed + 409)
    distribution_rows = []
    run_manifests = []
    for index in range(world_size):
        payload = torch.load(
            output / f"features_rank{index:02d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        current_path = str(payload["path"])
        run_manifests.append(
            {
                "path": current_path,
                "reproduction_relative_rms": payload["reproduction_relative_rms"],
                **payload["manifest"],
            }
        )
        endpoint_features = payload["features"]["step49_corrected"]
        for key, value in payload["features"].items():
            step_text, method = key.split("_", maxsplit=1)
            step = int(step_text.removeprefix("step"))
            distribution_rows.append(
                {
                    "path": current_path,
                    "step_index": step,
                    "time": float(times[step]),
                    "method": method,
                    "sample_count": len(value),
                    "projected_frechet": projected_frechet_distance(
                        reference.cpu(), value, projection
                    ),
                    "swd": standardized_sliced_wasserstein(
                        reference.cpu(), value, directions
                    ),
                    "endpoint_feature_rms": float(
                        sample_rms(value - endpoint_features).median()
                    ),
                }
            )
    distribution = pd.DataFrame(distribution_rows).sort_values(
        ["path", "step_index", "method"]
    )
    distribution.to_csv(output / "decoder_proxy.csv", index=False)
    summary, predictions = _prediction_summary(metrics, distribution)
    summary.to_csv(output / "conditioning_summary.csv", index=False)
    _plot(metrics, distribution, output)
    result = {
        "config": {
            **asdict(config),
            "branch_root": str(config.branch_root),
            "endpoint_root": str(config.endpoint_root),
            "audit_cache": str(config.audit_cache),
            "output_root": str(config.output_root),
            "selected_steps": list(SELECTED_STEPS),
            "decode_steps": list(DECODE_STEPS),
            "oracle_steps": list(ORACLE_STEPS),
        },
        "runs": run_manifests,
        "predictions": predictions,
        "all_predictions_hold": bool(all(predictions.values())),
        "scope": f"single-seed, {config.count}-sample exploratory conditioning probe",
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("\nConditioning summary:\n", summary.to_string(index=False), flush=True)
    print("\nDecoder proxy:\n", distribution.to_string(index=False), flush=True)
    dist.destroy_process_group()
    return output


def parse_args() -> ConditioningConfig:
    defaults = ConditioningConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root", type=Path, default=defaults.branch_root)
    parser.add_argument("--endpoint-root", type=Path, default=defaults.endpoint_root)
    parser.add_argument("--audit-cache", type=Path, default=defaults.audit_cache)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--endpoint-seed", type=int, default=defaults.endpoint_seed)
    parser.add_argument("--evaluation-seed", type=int, default=defaults.evaluation_seed)
    parser.add_argument("--endpoint-total", type=int, default=defaults.endpoint_total)
    parser.add_argument("--sample-start", type=int, default=defaults.sample_start)
    parser.add_argument("--count", type=int, default=defaults.count)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--reference-start", type=int, default=defaults.reference_start)
    parser.add_argument("--projection-dim", type=int, default=defaults.projection_dim)
    parser.add_argument("--swd-directions", type=int, default=defaults.swd_directions)
    args = parser.parse_args()
    return ConditioningConfig(
        branch_root=args.branch_root,
        endpoint_root=args.endpoint_root,
        audit_cache=args.audit_cache,
        output_root=args.output_root,
        run_name=args.run_name,
        endpoint_seed=args.endpoint_seed,
        evaluation_seed=args.evaluation_seed,
        endpoint_total=args.endpoint_total,
        sample_start=args.sample_start,
        count=args.count,
        batch_size=args.batch_size,
        reference_start=args.reference_start,
        projection_dim=args.projection_dim,
        swd_directions=args.swd_directions,
    )


if __name__ == "__main__":
    run(parse_args())
