"""Explore RAE clean-estimate trajectories against endpoint chord shortcuts."""

from __future__ import annotations

import argparse
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
    clean_estimate,
    project_to_endpoint_chord,
    rms_matched_projection,
    trajectory_prediction_summary,
)
from experiments.rae_cycle_direction_intervention import sample_rms  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset, load_cache_manifest  # noqa: E402
from experiments.rae_path_difference_intervention import (  # noqa: E402
    feature_progress,
    projected_frechet_distance,
    random_unit_directions,
    standardized_sliced_wasserstein,
)
from experiments.rae_teacher_rollout_gap import (  # noqa: E402
    configure_fp32,
    load_models,
    official_time_grid,
)
from experiments.run_rae_path_difference_intervention import (  # noqa: E402
    _decode_probe,
    _load_inception_probe,
)


DEFAULT_BRANCH_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
DEFAULT_ENDPOINT_ROOT = Path.home() / "data/eqvae/experiments/rae_decoder_risk_phase0"
DEFAULT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_clean_estimate_trajectory"
SELECTED_STEPS = (0, 8, 16, 24, 32, 40, 48)


@dataclass(frozen=True)
class TrajectoryConfig:
    branch_root: Path = DEFAULT_BRANCH_ROOT
    endpoint_root: Path = DEFAULT_ENDPOINT_ROOT
    audit_cache: Path = DEFAULT_CACHE
    output_root: Path = DEFAULT_OUTPUT
    run_name: str = "explore_seed20260718_start128_n32"
    endpoint_seed: int = 20_260_718
    evaluation_seed: int = 20_260_720
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


def _branch(root: Path, path: str) -> Path:
    branch = root / f"seed3407_{path}_rank16_s0_to_10000"
    if not (branch / "generation/ema_step-0010000.pt").exists():
        raise FileNotFoundError(branch)
    return branch


def _endpoint(root: Path, path: str) -> dict[str, object]:
    endpoint = root / f"0b_generated_latents_{path}_n256_s50.pt"
    if not endpoint.exists():
        raise FileNotFoundError(endpoint)
    return torch.load(endpoint, map_location="cpu", weights_only=True)


def _paired_noise_and_labels(
    endpoint: dict[str, object],
    *,
    seed: int,
    total: int,
    start: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = endpoint["latents"]
    if len(latents) != total or start + count > total:
        raise ValueError("endpoint size or requested slice is invalid")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randn(
        (total, *latents.shape[1:]), generator=generator, dtype=torch.float32
    )
    generated_labels = torch.randint(
        0, 1000, (total,), generator=generator, dtype=torch.long
    )
    labels = endpoint["labels"].long()
    if not torch.equal(generated_labels, labels):
        raise RuntimeError("recreated labels disagree with the saved endpoint payload")
    return initial[start : start + count].clone(), labels[start : start + count].clone()


@torch.no_grad()
def _clean_estimate_rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    *,
    selected_steps: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    selected = set(int(value) for value in selected_steps)
    chunks: dict[int, list[torch.Tensor]] = {index: [] for index in selected}
    endpoint_chunks = []
    for start in range(0, len(initial), batch_size):
        stop = min(start + batch_size, len(initial))
        state = initial[start:stop].to(device=device, dtype=torch.float32)
        batch_labels = labels[start:stop].to(device)
        for index, (current, following) in enumerate(zip(times[:-1], times[1:])):
            batch_time = torch.full(
                (len(state),), float(current), device=device, dtype=torch.float32
            )
            velocity = model(state, batch_time, y=batch_labels)
            if index in selected:
                chunks[index].append(clean_estimate(state, velocity, batch_time).cpu())
            state = state + (following.to(state) - current.to(state)) * velocity
        endpoint_chunks.append(state.cpu())
    return (
        {index: torch.cat(chunks[index]) for index in sorted(chunks)},
        torch.cat(endpoint_chunks),
    )


def _latent_rows(
    *,
    path: str,
    step: int,
    time: float,
    actual: torch.Tensor,
    chord: torch.Tensor,
    rms_chord: torch.Tensor,
    endpoint: torch.Tensor,
    progress: torch.Tensor,
    curvature: torch.Tensor,
    first_index: int,
) -> list[dict[str, object]]:
    rows = []
    for kind, value in (("actual", actual), ("chord", chord), ("rms_chord", rms_chord)):
        rms = sample_rms(value)
        endpoint_distance = sample_rms(value - endpoint) / sample_rms(endpoint, 1e-12)
        for offset in range(len(value)):
            rows.append(
                {
                    "path": path,
                    "sample_index": first_index + offset,
                    "step_index": int(step),
                    "time": float(time),
                    "kind": kind,
                    "progress": float(progress[offset]),
                    "progress_outside_unit": bool(
                        progress[offset] < 0 or progress[offset] > 1
                    ),
                    "curvature_ratio": float(curvature[offset]) if kind == "actual" else 0.0,
                    "latent_rms": float(rms[offset]),
                    "relative_endpoint_distance": float(endpoint_distance[offset]),
                }
            )
    return rows


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


def _plot(distribution: pd.DataFrame, latent: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    styles = {
        "actual": ("o", "actual clean estimate"),
        "chord": ("s", "endpoint chord"),
        "rms_chord": ("^", "RMS-matched chord"),
    }
    for axis, path in zip(axes.flat, PATHS):
        selected = distribution[distribution.path == path]
        for method, (marker, label) in styles.items():
            rows = selected[selected.method == method].sort_values("step_index")
            axis.plot(
                rows.step_index,
                rows.projected_frechet,
                marker=marker,
                linewidth=2,
                label=label,
            )
        axis.set_title(path)
        axis.set_xlabel("Euler state index")
        axis.set_ylabel("Projected Frechet (lower is better)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("RAE clean-estimate trajectory vs endpoint chord", fontsize=18)
    figure.savefig(output / "trajectory_quality.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    actual = latent[latent.kind == "actual"]
    for path in PATHS:
        rows = (
            actual[actual.path == path]
            .groupby("step_index", as_index=False)
            .agg(curvature=("curvature_ratio", "median"), progress=("progress", "median"))
            .sort_values("step_index")
        )
        axes[0].plot(rows.step_index, rows.curvature, marker="o", label=path)
        axes[1].plot(rows.step_index, rows.progress, marker="o", label=path)
    axes[0].set_title("Median orthogonal curvature")
    axes[0].set_ylabel("||actual - chord|| / ||endpoint chord||")
    axes[1].set_title("Median chord progress")
    axes[1].set_ylabel("projection coefficient")
    for axis in axes:
        axis.set_xlabel("Euler state index")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(output / "trajectory_geometry.png", dpi=180)
    plt.close(figure)


def run(config: TrajectoryConfig) -> Path | None:
    rank, world_size, device = _distributed(config.evaluation_seed)
    if world_size != len(PATHS):
        raise ValueError(f"expected {len(PATHS)} GPUs, got {world_size}")
    path = PATHS[rank]
    output = config.output_root.expanduser().resolve() / config.run_name
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    branch = _branch(config.branch_root.expanduser().resolve(), path)
    endpoint_payload = _endpoint(config.endpoint_root.expanduser().resolve(), path)
    initial, labels = _paired_noise_and_labels(
        endpoint_payload,
        seed=config.endpoint_seed,
        total=config.endpoint_total,
        start=config.sample_start,
        count=config.count,
    )
    saved_endpoint = endpoint_payload["latents"][
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
    estimates, reproduced_endpoint = _clean_estimate_rollout(
        model,
        initial,
        labels,
        times,
        selected_steps=SELECTED_STEPS,
        batch_size=config.batch_size,
        device=device,
    )
    endpoint_difference = reproduced_endpoint - saved_endpoint
    reproduction = {
        "path": path,
        "max_abs": float(endpoint_difference.abs().max()),
        "relative_rms": float(
            sample_rms(endpoint_difference).mean()
            / sample_rms(saved_endpoint, 1e-12).mean()
        ),
    }
    if reproduction["max_abs"] > 1e-4 or reproduction["relative_rms"] > 1e-6:
        raise RuntimeError(f"saved endpoint reproduction failed: {reproduction}")

    reference = torch.zeros((config.count, 2048), device=device, dtype=torch.float32)
    cache = config.audit_cache.expanduser().resolve()
    manifest = load_cache_manifest(cache)
    if config.reference_start + config.count > int(manifest["sample_count"]):
        raise ValueError("clean reference slice exceeds cache")
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

    first = estimates[min(estimates)]
    candidates: dict[tuple[int, str], torch.Tensor] = {}
    latent_rows: list[dict[str, object]] = []
    for step, actual in estimates.items():
        progress, chord, curvature = project_to_endpoint_chord(
            actual, first, saved_endpoint
        )
        rms_chord = rms_matched_projection(actual, chord)
        for method, value in (
            ("actual", actual),
            ("chord", chord),
            ("rms_chord", rms_chord),
        ):
            candidates[(step, method)] = value
        latent_rows.extend(
            _latent_rows(
                path=path,
                step=step,
                time=float(times[step]),
                actual=actual,
                chord=chord,
                rms_chord=rms_chord,
                endpoint=saved_endpoint,
                progress=progress,
                curvature=curvature,
                first_index=config.sample_start,
            )
        )
    endpoint_progress = torch.ones(config.count)
    endpoint_curvature = torch.zeros(config.count)
    for method in ("actual", "chord", "rms_chord"):
        candidates[(49, method)] = saved_endpoint
    latent_rows.extend(
        _latent_rows(
            path=path,
            step=49,
            time=float(times[-1]),
            actual=saved_endpoint,
            chord=saved_endpoint,
            rms_chord=saved_endpoint,
            endpoint=saved_endpoint,
            progress=endpoint_progress,
            curvature=endpoint_curvature,
            first_index=config.sample_start,
        )
    )

    feature_payload: dict[str, torch.Tensor] = {}
    for (step, method), candidate in candidates.items():
        key = f"step{step:02d}_{method}"
        _, _, features, _ = _decode_probe(
            decoder,
            inception,
            candidate,
            batch_size=config.batch_size,
            device=device,
        )
        feature_payload[key] = features
    pd.DataFrame(latent_rows).to_csv(output / f"latent_rank{rank:02d}.csv", index=False)
    torch.save(
        {
            "path": path,
            "features": feature_payload,
            "reproduction": reproduction,
        },
        output / f"features_rank{rank:02d}.pt",
    )
    print(
        f"rank{rank} {path} complete in {(perf_counter() - started) / 60:.1f}m; "
        f"endpoint max_abs={reproduction['max_abs']:.3e}",
        flush=True,
    )
    dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return None

    latent = pd.concat(
        [pd.read_csv(output / f"latent_rank{index:02d}.csv") for index in range(world_size)],
        ignore_index=True,
    )
    latent.to_csv(output / "latent_metrics.csv", index=False)
    projection = random_unit_directions(2048, config.projection_dim, config.evaluation_seed + 401)
    directions = random_unit_directions(2048, config.swd_directions, config.evaluation_seed + 409)
    distribution_rows = []
    reproductions = []
    for index in range(world_size):
        payload = torch.load(
            output / f"features_rank{index:02d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        current_path = str(payload["path"])
        reproductions.append(payload["reproduction"])
        first_features = payload["features"]["step00_actual"]
        endpoint_features = payload["features"]["step49_actual"]
        for key, features in payload["features"].items():
            step_text, method = key.split("_", maxsplit=1)
            step = int(step_text.removeprefix("step"))
            distribution_rows.append(
                {
                    "path": current_path,
                    "step_index": step,
                    "time": float(times[step]),
                    "method": method,
                    "sample_count": len(features),
                    "projected_frechet": projected_frechet_distance(
                        reference.cpu(), features, projection
                    ),
                    "swd": standardized_sliced_wasserstein(
                        reference.cpu(), features, directions
                    ),
                    "feature_progress_median": float(
                        feature_progress(features, first_features, endpoint_features).median()
                    ),
                }
            )
    distribution = pd.DataFrame(distribution_rows).sort_values(
        ["path", "step_index", "method"]
    )
    distribution.to_csv(output / "distribution_metrics.csv", index=False)
    summary, gate = trajectory_prediction_summary(distribution, latent)
    summary.to_csv(output / "prediction_summary.csv", index=False)
    _plot(distribution, latent, output)
    result = {
        "config": {
            **asdict(config),
            "branch_root": str(config.branch_root),
            "endpoint_root": str(config.endpoint_root),
            "audit_cache": str(config.audit_cache),
            "output_root": str(config.output_root),
            "selected_steps": list(SELECTED_STEPS),
        },
        "endpoint_reproduction": reproductions,
        "prediction_gate": gate,
        "scope": (
            f"single-seed, {config.count}-sample exploratory proxy study; "
            "not FID confirmation"
        ),
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("\nPrediction summary:\n", summary.to_string(index=False), flush=True)
    dist.destroy_process_group()
    return output


def parse_args() -> TrajectoryConfig:
    defaults = TrajectoryConfig()
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
    return TrajectoryConfig(
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
