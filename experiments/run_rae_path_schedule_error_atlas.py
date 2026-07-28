"""Held-out time/subspace error atlas for static, annealed, and floor RAE paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
for import_path in (ROOT, RAE_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.rae_clean_estimate_trajectory import endpoint_observation_factors  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_layerwise_path import plan_layerwise_path, spatial_center  # noqa: E402
from experiments.rae_teacher_rollout_gap import configure_fp32  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


BASELINE_ROOT = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
CANDIDATE_ROOT = Path.home() / "data/eqvae/experiments/rae_path_schedule_train"
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_path_schedule_error_atlas"
CONDITIONS = (
    ("static", BASELINE_ROOT / "seed3407_static_rank16_s0_to_10000"),
    ("annealed", BASELINE_ROOT / "seed3407_annealed_rank16_s0_to_10000"),
    ("floor020_p2", CANDIDATE_ROOT / "seed3407_floor020_p2_rank16_s0_to_2000"),
)
TIMES = (0.97, 0.85, 0.70, 0.50, 0.30, 0.10)


def sample_rms(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    return value.float().square().flatten(1).mean(1).add(float(eps)).sqrt()


def basis_projection(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    _, residual = spatial_center(value.float())
    basis = basis.to(value)
    rows = residual.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
    projected = (rows @ basis) @ basis.transpose(0, 1)
    return projected.reshape(
        value.shape[0], value.shape[2], value.shape[3], value.shape[1]
    ).permute(0, 3, 1, 2).contiguous()


def component_error_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    clean: torch.Tensor,
    basis: torch.Tensor,
    *,
    time: float,
    semantic_factor: float,
    basis_factor: float,
) -> dict[str, torch.Tensor]:
    error = prediction.float() - target.float()
    error_basis = basis_projection(error, basis)
    error_semantic = error - error_basis
    target_basis = basis_projection(target, basis)
    target_semantic = target - target_basis
    clean_basis = basis_projection(clean, basis)
    clean_semantic = clean - clean_basis
    return {
        "velocity_relative": sample_rms(error) / sample_rms(target, 1e-12),
        "semantic_velocity_relative": sample_rms(error_semantic)
        / sample_rms(target_semantic, 1e-12),
        "basis_velocity_relative": sample_rms(error_basis)
        / sample_rms(target_basis, 1e-12),
        "semantic_endpoint_relative": float(time)
        * sample_rms(error_semantic)
        / (float(semantic_factor) * sample_rms(clean_semantic, 1e-12)),
        "basis_endpoint_relative": float(time)
        * sample_rms(error_basis)
        / (float(basis_factor) * sample_rms(clean_basis, 1e-12)),
    }


def _load_basis(manifest: dict[str, object]) -> torch.Tensor:
    payload = torch.load(
        Path(str(manifest["subspace_path"])), map_location="cpu", weights_only=False
    )
    rank = int(manifest["subspace_rank"])
    entry = payload["subspaces"].get(rank, payload["subspaces"].get(str(rank)))
    return entry["basis"].float().contiguous()


def _load_model(branch: Path, device: torch.device) -> torch.nn.Module:
    config = OmegaConf.load(branch / "config.yaml")
    model = instantiate_from_config(config.stage_2).to(device=device, dtype=torch.float32)
    state = torch.load(
        branch / "generation/ema_step-0005000.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval()


@torch.inference_mode()
def evaluate_condition(
    condition: str,
    branch: Path,
    *,
    start: int,
    count: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
    dataset = CachedRAELatentDataset(
        Path(str(manifest["latent_cache"])), start=start, stop=start + count
    )
    clean = torch.stack([dataset[index][0] for index in range(len(dataset))])
    labels = torch.tensor([dataset[index][1] for index in range(len(dataset))], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    basis = _load_basis(manifest).to(device)
    model = _load_model(branch, device)
    mode = str(manifest["path_mode"])
    power = float(manifest["path_power"])
    family = str(manifest.get("path_family", "power"))
    floor = float(manifest.get("path_floor", 0.0))
    alpha = float(manifest.get("path_alpha", 1.0))
    detail_scale = float(manifest["detail_scale"])
    rows = []
    for time_value in TIMES:
        scalar_time = torch.tensor([time_value], dtype=torch.float32)
        semantic_factor, detail_factor = endpoint_observation_factors(
            scalar_time,
            mode,
            power=power,
            family=family,
            floor=floor,
            alpha=alpha,
        )
        basis_factor = semantic_factor * (1.0 - detail_scale) + detail_factor * detail_scale
        for offset in range(0, count, batch_size):
            stop = min(offset + batch_size, count)
            clean_batch = clean[offset:stop].to(device)
            noise_batch = noise[offset:stop].to(device)
            labels_batch = labels[offset:stop].to(device)
            time_batch = torch.full(
                (stop - offset,), time_value, device=device, dtype=torch.float32
            )
            plan = plan_layerwise_path(
                clean_batch,
                noise_batch,
                time_batch,
                basis,
                mode=mode,
                power=power,
                family=family,
                floor=floor,
                alpha=alpha,
                detail_scale=detail_scale,
            )
            prediction = model(plan.state, time_batch, y=labels_batch)
            metrics = component_error_metrics(
                prediction,
                plan.target,
                clean_batch,
                basis,
                time=time_value,
                semantic_factor=float(semantic_factor),
                basis_factor=float(basis_factor),
            )
            for local_index in range(stop - offset):
                rows.append(
                    {
                        "condition": condition,
                        "sample_index": start + offset + local_index,
                        "time": time_value,
                        "semantic_factor": float(semantic_factor),
                        "basis_factor": float(basis_factor),
                        **{
                            name: float(value[local_index])
                            for name, value in metrics.items()
                        },
                    }
                )
    return pd.DataFrame(rows)


def _plot(table: pd.DataFrame, output: Path) -> None:
    summary = table.groupby(["condition", "time"], as_index=False).median(numeric_only=True)
    columns = (
        ("semantic_velocity_relative", "Semantic velocity error"),
        ("basis_velocity_relative", "Rank-16 velocity error"),
        ("semantic_endpoint_relative", "Semantic endpoint error"),
        ("basis_endpoint_relative", "Rank-16 endpoint error"),
    )
    colors = {"static": "#4C78A8", "annealed": "#E45756", "floor020_p2": "#54A24B"}
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, columns):
        for condition in ("static", "annealed", "floor020_p2"):
            rows = summary[summary.condition == condition].sort_values("time")
            axis.plot(
                rows.time,
                rows[column],
                marker="o",
                linewidth=2.3,
                label=condition,
                color=colors[condition],
            )
        axis.set_title(title)
        axis.set_xlabel("t (higher is noisier)")
        axis.set_ylabel("median relative RMS")
        if column == "basis_endpoint_relative":
            axis.set_yscale("log")
            axis.set_title(f"{title} (log scale)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(output / "heldout_error_atlas.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="step5000_seed3407_holdout128")
    parser.add_argument("--start", type=int, default=100000)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20_260_723)
    args = parser.parse_args()
    if "RANK" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("launch with torchrun on three GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != len(CONDITIONS):
        raise ValueError("expected exactly three processes")
    configure_fp32(args.seed)
    condition, branch = CONDITIONS[rank]
    output = args.output.expanduser().resolve() / args.run_name
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    table = evaluate_condition(
        condition,
        branch,
        start=args.start,
        count=args.count,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    table.to_csv(output / f"error_rank{rank:02d}.csv", index=False)
    print(f"rank{rank} {condition} complete", flush=True)
    dist.barrier()
    if rank == 0:
        combined = pd.concat(
            [pd.read_csv(output / f"error_rank{index:02d}.csv") for index in range(3)],
            ignore_index=True,
        )
        combined.to_csv(output / "heldout_errors.csv", index=False)
        summary = combined.groupby(["condition", "time"], as_index=False).median(
            numeric_only=True
        )
        summary.to_csv(output / "heldout_error_summary.csv", index=False)
        _plot(combined, output)
        print(summary.to_string(index=False), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
