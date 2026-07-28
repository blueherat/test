"""Distributed no-training audit of latent transport compatibility.

Run with four GPUs, for example:

    torchrun --standalone --nproc_per_node=4 \
      experiments/audit_latent_transport_compatibility.py

Large artifacts are written below ``~/data/eqvae`` by default.  The audit uses
ImageNet validation indices after the old adapter evaluation prefix, so no
image used to train the adapter enters this study.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.dinov2_token_diagnostics import (  # noqa: E402
    center_crop_resize,
    configure_fp32,
    load_named_dataset,
    pil_to_tensor_m11,
)
from baselines.visual_adapters import load_rae_adapter  # noqa: E402
from experiments.decoder_adapted_rae import _load_adapter_from_checkpoint  # noqa: E402
from experiments.latent_transport_audit_metrics import (  # noqa: E402
    AnisotropicChannelTransform,
    IdentityLatentTransform,
    LatentSketch,
    SignedChannelOrthogonalTransform,
    apply_linear_or_jvp,
    bootstrap_spearman,
    covariance_mismatch,
    knn_overlap,
    local_velocity_ambiguity,
    projected_shared_class_viv,
    relative_l2_per_sample,
    sliced_wasserstein_1,
    spearman_correlation,
)
from experiments.latent_transport_paths import ScaledAdditiveCouplingTransform  # noqa: E402


@dataclass(frozen=True)
class TransportAuditConfig:
    adapter_checkpoint: str = str(
        Path.home()
        / "data/eqvae/artifacts/latent_adapter/"
        "dinov2_adapter_imagenet_train32768_val2048_testval2048_e6_seq_noleak/adapter.pt"
    )
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    dataset_split: str = "validation"
    excluded_prefix: int = 2048
    count: int = 2048
    path_count: int = 512
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 20260718
    repeats: tuple[int, ...] = (0, 1, 2)
    times: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    anisotropic_conditions: tuple[float, ...] = (1.5, 2.0, 4.0, 8.0)
    adapter_scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    sketch_channels: int = 16
    sketch_spatial_size: int = 4
    sketch_seed: int = 3701
    sliced_directions: int = 64
    neighbors: int = 8
    rae_repo_path: str = "external/RAE"
    output_root: str = str(Path.home() / "data/eqvae/artifacts/latent_transport_audit")
    run_name: str = ""
    overwrite: bool = False


class IndexedLabeledImages(Dataset):
    def __init__(self, dataset, indices: Sequence[int], image_size: int):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        index = self.indices[item]
        image, label = self.dataset[index]
        image = center_crop_resize(image.convert("RGB"), self.image_size)
        return pil_to_tensor_m11(image), int(label), index


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _jsonable_config(config: TransportAuditConfig) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(config).items()
    }


def _broadcast_run_name(config: TransportAuditConfig, rank: int) -> str:
    if rank == 0:
        run_name = config.run_name.strip() or time.strftime("phase2_%Y%m%d_%H%M%S")
    else:
        run_name = ""
    if dist.is_initialized():
        values = [run_name]
        dist.broadcast_object_list(values, src=0)
        run_name = str(values[0])
    return run_name


def _select_indices(dataset_length: int, config: TransportAuditConfig) -> list[int]:
    candidates = np.arange(int(config.excluded_prefix), int(dataset_length))
    if len(candidates) < config.count:
        raise ValueError("not enough samples after excluded_prefix")
    generator = np.random.default_rng(int(config.seed))
    return [int(index) for index in generator.permutation(candidates)[: config.count]]


def _encode_local_shard(
    config: TransportAuditConfig,
    local_indices: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = load_named_dataset(
        config.dataset_name,
        config.data_root,
        split=config.dataset_split,
        dataset_path=config.dataset_path,
    )
    loader = DataLoader(
        IndexedLabeledImages(dataset, local_indices, config.image_size),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    rae_repo = Path(config.rae_repo_path)
    if not rae_repo.is_absolute():
        rae_repo = ROOT / rae_repo
    rae = load_rae_adapter(
        "rae_dinov2",
        repo_path=rae_repo,
        device=device,
        dtype=torch.float32,
        auto_clone=False,
        auto_download=False,
    )
    for parameter in rae.model.parameters():
        parameter.requires_grad_(False)
    latents = []
    flipped_latents = []
    labels = []
    observed_indices = []
    with torch.no_grad():
        for images_cpu, labels_cpu, indices_cpu in loader:
            images = images_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            latents.append(rae.encode(images).cpu())
            flipped_latents.append(rae.encode(images.flip(-1)).cpu())
            labels.append(labels_cpu.to(torch.long))
            observed_indices.append(indices_cpu.to(torch.long))
    del rae, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return (
        torch.cat(latents),
        torch.cat(flipped_latents),
        torch.cat(labels),
        torch.cat(observed_indices),
    )


def _gather_tensor(value: torch.Tensor, device: torch.device, cat_dim: int = 0) -> torch.Tensor | None:
    value = value.to(device=device, non_blocking=True).contiguous()
    if not dist.is_initialized():
        return value.cpu()
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    if dist.get_rank() != 0:
        return None
    return torch.cat(gathered, dim=cat_dim).cpu()


def _full_latent_knn_indices(
    values_cpu: torch.Tensor,
    neighbors: int,
    device: torch.device,
    query_batch: int = 32,
) -> torch.Tensor:
    """Exact kNN within one rank shard, evaluated in the full latent space."""

    n = len(values_cpu)
    k = min(max(1, int(neighbors)), n - 1)
    values = values_cpu.flatten(1).to(device=device, dtype=torch.float32)
    norms = values.square().sum(dim=1)
    rows = []
    for start in range(0, n, int(query_batch)):
        stop = min(start + int(query_batch), n)
        distance = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * (values[start:stop] @ values.T)
        ).clamp_min_(0.0)
        local_rows = torch.arange(stop - start, device=device)
        global_rows = torch.arange(start, stop, device=device)
        distance[local_rows, global_rows] = float("inf")
        rows.append(distance.topk(k=k, largest=False).indices.cpu())
    del values, norms
    return torch.cat(rows)


def _full_latent_knn_overlap(
    reference_indices: torch.Tensor,
    transformed_cpu: torch.Tensor,
    neighbors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    transformed_indices = _full_latent_knn_indices(
        transformed_cpu,
        neighbors,
        device,
    )
    matches = (
        reference_indices[:, :, None] == transformed_indices[:, None, :]
    ).any(dim=2).sum(dim=1).float()
    k = reference_indices.shape[1]
    recall = matches / k
    jaccard = matches / (2 * k - matches).clamp_min(1.0)
    return recall, jaccard


def _condition_specs(config: TransportAuditConfig) -> list[dict[str, float | int | str]]:
    specs: list[dict[str, float | int | str]] = []
    for repeat in config.repeats:
        specs.append(
            {
                "family": "identity",
                "strength": 0.0,
                "map_seed": 0,
                "noise_seed": int(repeat),
            }
        )
        specs.append(
            {
                "family": "signed_orthogonal",
                "strength": 1.0,
                "map_seed": int(repeat),
                "noise_seed": int(repeat),
            }
        )
        for condition in config.anisotropic_conditions:
            specs.append(
                {
                    "family": "anisotropic_linear",
                    "strength": float(condition),
                    "map_seed": int(repeat),
                    "noise_seed": int(repeat),
                }
            )
        for scale in config.adapter_scales:
            specs.append(
                {
                    "family": "nonlinear_adapter",
                    "strength": float(scale),
                    "map_seed": 0,
                    "noise_seed": int(repeat),
                }
            )
    for index, spec in enumerate(specs):
        spec["condition_id"] = f"c{index:03d}_{spec['family']}_s{spec['strength']}_r{spec['noise_seed']}"
    return specs


def _make_transform(
    spec: dict[str, float | int | str],
    adapter: torch.nn.Module,
    channels: int,
    device: torch.device,
) -> torch.nn.Module:
    family = str(spec["family"])
    if family == "identity":
        transform = IdentityLatentTransform()
    elif family == "signed_orthogonal":
        transform = SignedChannelOrthogonalTransform(channels, int(spec["map_seed"]))
    elif family == "anisotropic_linear":
        transform = AnisotropicChannelTransform(
            channels,
            float(spec["strength"]),
            int(spec["map_seed"]),
        )
    elif family == "nonlinear_adapter":
        transform = ScaledAdditiveCouplingTransform(adapter, float(spec["strength"]))
        transform.is_linear = False
    else:
        raise ValueError(f"unknown transform family: {family}")
    return transform.to(device=device, dtype=torch.float32).eval()


def _condition_local_tensors(
    z_cpu: torch.Tensor,
    z_flip_cpu: torch.Tensor,
    transform: torch.nn.Module,
    sketch: LatentSketch,
    config: TransportAuditConfig,
    rank: int,
    device: torch.device,
    noise_seed: int,
    reference_knn_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    local_count = len(z_cpu)
    local_path_count = config.path_count // (
        dist.get_world_size() if dist.is_initialized() else 1
    )
    noise_generator = torch.Generator(device=device).manual_seed(
        config.seed + 100_003 * int(noise_seed) + 1_009 * rank
    )
    reference_generator = torch.Generator(device=device).manual_seed(
        config.seed + 200_003 * int(noise_seed) + 2_009 * rank
    )
    outputs: dict[str, list[torch.Tensor]] = {
        "base_sketch": [],
        "target_sketch": [],
        "source_sketch": [],
        "reference_source_sketch": [],
        "equivariance": [],
        "cycle": [],
        "target_full": [],
    }
    path_states = {
        branch: [[] for _ in config.times]
        for branch in ("gaussian_straight", "matched_chord", "pushforward")
    }
    path_velocities = {
        branch: [[] for _ in config.times]
        for branch in ("gaussian_straight", "matched_chord", "pushforward")
    }
    bridge = [[] for _ in config.times]
    velocity_gap = [[] for _ in config.times]
    source_path_gap = [[] for _ in config.times]
    jacobian_gain = [[] for _ in config.times]

    seen = 0
    for start in range(0, local_count, config.batch_size):
        stop = min(start + config.batch_size, local_count)
        z = z_cpu[start:stop].to(device=device, non_blocking=True)
        z_flip = z_flip_cpu[start:stop].to(device=device, non_blocking=True)
        epsilon = torch.randn(
            z.shape,
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        )
        independent_gaussian = torch.randn(
            z.shape,
            generator=reference_generator,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            target = transform(z)
            transformed_flip = transform(z_flip)
            transformed_source = transform(epsilon)
            reconstructed = transform.inverse(target)
            outputs["base_sketch"].append(sketch(z).cpu())
            outputs["target_sketch"].append(sketch(target).cpu())
            outputs["target_full"].append(target.cpu())
            outputs["source_sketch"].append(sketch(transformed_source).cpu())
            outputs["reference_source_sketch"].append(
                sketch(independent_gaussian).cpu()
            )
            outputs["equivariance"].append(
                relative_l2_per_sample(transformed_flip, target.flip(-1)).cpu()
            )
            outputs["cycle"].append(relative_l2_per_sample(reconstructed, z).cpu())

        active = max(0, min(stop, local_path_count) - start)
        if active <= 0:
            seen = stop
            continue
        z_path = z[:active]
        epsilon_path = epsilon[:active]
        target_path = target[:active]
        source_path = transformed_source[:active]
        direction = epsilon_path - z_path
        for time_index, time_value in enumerate(config.times):
            t = float(time_value)
            base_state = (1.0 - t) * z_path + t * epsilon_path
            push_state, push_velocity = apply_linear_or_jvp(
                transform,
                base_state,
                direction,
            )
            chord_state = (1.0 - t) * target_path + t * source_path
            chord_velocity = source_path - target_path
            gaussian_state = (1.0 - t) * target_path + t * epsilon_path
            gaussian_velocity = epsilon_path - target_path

            states = {
                "gaussian_straight": gaussian_state,
                "matched_chord": chord_state,
                "pushforward": push_state,
            }
            velocities = {
                "gaussian_straight": gaussian_velocity,
                "matched_chord": chord_velocity,
                "pushforward": push_velocity,
            }
            with torch.no_grad():
                for branch in states:
                    path_states[branch][time_index].append(sketch(states[branch]).cpu())
                    path_velocities[branch][time_index].append(
                        sketch(velocities[branch]).cpu()
                    )
                bridge[time_index].append(
                    relative_l2_per_sample(chord_state, push_state).cpu()
                )
                velocity_gap[time_index].append(
                    relative_l2_per_sample(chord_velocity, push_velocity).cpu()
                )
                source_path_gap[time_index].append(
                    relative_l2_per_sample(gaussian_state, chord_state).cpu()
                )
                gain = push_velocity.flatten(1).norm(dim=1) / direction.flatten(1).norm(
                    dim=1
                ).clamp_min(1e-12)
                jacobian_gain[time_index].append(gain.cpu())
        seen = stop
    if seen != local_count:
        raise RuntimeError("local condition loop did not consume every latent")

    target_full = torch.cat(outputs.pop("target_full"))
    exact_recall, exact_jaccard = _full_latent_knn_overlap(
        reference_knn_indices,
        target_full,
        config.neighbors,
        device,
    )
    del target_full
    packed = {key: torch.cat(value) for key, value in outputs.items()}
    packed["full_knn_recall"] = exact_recall
    packed["full_knn_jaccard"] = exact_jaccard
    for branch in path_states:
        packed[f"{branch}_states"] = torch.stack(
            [torch.cat(chunks) for chunks in path_states[branch]]
        )
        packed[f"{branch}_velocities"] = torch.stack(
            [torch.cat(chunks) for chunks in path_velocities[branch]]
        )
    packed["bridge"] = torch.stack([torch.cat(chunks) for chunks in bridge])
    packed["velocity_gap"] = torch.stack(
        [torch.cat(chunks) for chunks in velocity_gap]
    )
    packed["source_path_gap"] = torch.stack(
        [torch.cat(chunks) for chunks in source_path_gap]
    )
    packed["jacobian_gain"] = torch.stack(
        [torch.cat(chunks) for chunks in jacobian_gain]
    )
    return packed


def _gather_condition(
    local: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    global_values: dict[str, torch.Tensor] = {}
    for key, value in local.items():
        cat_dim = 1 if value.ndim >= 2 and key not in {
            "base_sketch",
            "target_sketch",
            "source_sketch",
            "reference_source_sketch",
        } and (key.endswith("_states") or key.endswith("_velocities") or key in {
            "bridge",
            "velocity_gap",
            "source_path_gap",
            "jacobian_gain",
        }) else 0
        gathered = _gather_tensor(value, device, cat_dim=cat_dim)
        if gathered is not None:
            global_values[key] = gathered
    return global_values if (not dist.is_initialized() or dist.get_rank() == 0) else None


def _summarize_condition(
    spec: dict[str, float | int | str],
    values: dict[str, torch.Tensor],
    labels: torch.Tensor,
    config: TransportAuditConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    prior = covariance_mismatch(
        values["source_sketch"],
        values["reference_source_sketch"],
    )
    prior_swd = sliced_wasserstein_1(
        values["source_sketch"],
        values["reference_source_sketch"],
        directions=config.sliced_directions,
        seed=config.sketch_seed + 101,
    )
    viv = projected_shared_class_viv(values["target_sketch"], labels)
    neighborhood = knn_overlap(
        values["base_sketch"],
        values["target_sketch"],
        neighbors=config.neighbors,
    )
    bridge = values["bridge"].float()
    velocity_gap = values["velocity_gap"].float()
    source_path_gap = values["source_path_gap"].float()
    jacobian_gain = values["jacobian_gain"].float()
    row: dict[str, object] = dict(spec)
    row.update(
        {
            "equivariance_error_mean": float(values["equivariance"].mean()),
            "equivariance_error_std": float(values["equivariance"].std(unbiased=False)),
            "cycle_relative_l2_mean": float(values["cycle"].mean()),
            "cycle_relative_l2_max": float(values["cycle"].max()),
            "prior_sliced_w1": prior_swd,
            **{f"prior_{key}": value for key, value in prior.items()},
            **viv,
            "full_knn_recall": float(values["full_knn_recall"].mean()),
            "full_knn_jaccard": float(values["full_knn_jaccard"].mean()),
            "projected_knn_recall": neighborhood["recall"],
            "projected_knn_jaccard": neighborhood["jaccard"],
            "bridge_defect_mean": float(bridge.mean()),
            "bridge_defect_max": float(bridge.max()),
            "chord_push_velocity_gap_mean": float(velocity_gap.mean()),
            "gaussian_matched_state_gap_mean": float(source_path_gap.mean()),
            "jacobian_gain_mean": float(jacobian_gain.mean()),
            "jacobian_gain_std": float(jacobian_gain.std(unbiased=False)),
            "jacobian_gain_p05": float(torch.quantile(jacobian_gain, 0.05)),
            "jacobian_gain_p95": float(torch.quantile(jacobian_gain, 0.95)),
        }
    )
    row["jacobian_directional_spread"] = float(
        row["jacobian_gain_p95"] / max(float(row["jacobian_gain_p05"]), 1e-12)
    )

    path_rows: list[dict[str, object]] = []
    for branch in ("gaussian_straight", "matched_chord", "pushforward"):
        for time_index, time_value in enumerate(config.times):
            ambiguity = local_velocity_ambiguity(
                values[f"{branch}_states"][time_index],
                values[f"{branch}_velocities"][time_index],
                neighbors=config.neighbors,
            )
            path_row: dict[str, object] = dict(spec)
            path_row.update(
                {
                    "branch": branch,
                    "time": float(time_value),
                    **ambiguity,
                    "bridge_defect_mean": float(bridge[time_index].mean()),
                    "chord_push_velocity_gap_mean": float(
                        velocity_gap[time_index].mean()
                    ),
                    "gaussian_matched_state_gap_mean": float(
                        source_path_gap[time_index].mean()
                    ),
                }
            )
            path_rows.append(path_row)
    for branch in ("gaussian_straight", "matched_chord", "pushforward"):
        branch_values = [
            float(item["ambiguity_ratio"])
            for item in path_rows
            if item["branch"] == branch
        ]
        row[f"{branch}_ambiguity_mean"] = float(np.mean(branch_values))
    return row, path_rows


def _correlations_and_acceptance(condition_frame: pd.DataFrame) -> tuple[dict, dict]:
    grouped = (
        condition_frame.groupby(["family", "strength"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["family", "strength"])
    )
    anisotropic = grouped[grouped.family == "anisotropic_linear"]
    nonlinear = condition_frame[condition_frame.family == "nonlinear_adapter"]
    nonlinear_grouped = grouped[grouped.family == "nonlinear_adapter"]

    def safe_spearman(left, right) -> float:
        if len(left) < 2:
            return float("nan")
        return spearman_correlation(left, right)

    anisotropic_prior = safe_spearman(
        anisotropic.strength,
        anisotropic.prior_sliced_w1,
    )
    nonlinear_scale_bridge = safe_spearman(
        nonlinear_grouped.strength,
        nonlinear_grouped.bridge_defect_mean,
    )
    if len(nonlinear) >= 3:
        nonlinear_bridge_velocity = bootstrap_spearman(
            nonlinear.bridge_defect_mean,
            nonlinear.chord_push_velocity_gap_mean,
            resamples=2000,
            seed=451,
        )
        nonlinear_bridge_velocity_dict = asdict(nonlinear_bridge_velocity)
    else:
        nonlinear_bridge_velocity = None
        nonlinear_bridge_velocity_dict = {
            "correlation": None,
            "ci_low": None,
            "ci_high": None,
            "valid_bootstraps": 0,
        }
    anisotropic_prior_output = (
        float(anisotropic_prior) if np.isfinite(anisotropic_prior) else None
    )
    nonlinear_scale_bridge_output = (
        float(nonlinear_scale_bridge)
        if np.isfinite(nonlinear_scale_bridge)
        else None
    )
    correlations = {
        "anisotropic_condition_vs_prior_sliced_w1": anisotropic_prior_output,
        "adapter_scale_vs_bridge_defect": nonlinear_scale_bridge_output,
        "adapter_bridge_vs_velocity_gap": nonlinear_bridge_velocity_dict,
        "note": (
            "Projected VIV and kNN ambiguity are diagnostics, not generation metrics. "
            "The mixed-family defect-VIV correlation is intentionally undefined."
        ),
    }

    identity = condition_frame[condition_frame.family == "identity"]
    orthogonal = condition_frame[condition_frame.family == "signed_orthogonal"]
    linear = condition_frame[condition_frame.family == "anisotropic_linear"]
    identity_prior = float(identity.prior_sliced_w1.median())
    orthogonal_prior = float(orthogonal.prior_sliced_w1.median())
    checks = {
        "identity_bridge_numerical_zero": {
            "value": float(identity.bridge_defect_max.max()),
            "threshold": 1e-6,
            "passed": float(identity.bridge_defect_max.max()) <= 1e-6,
        },
        "identity_velocity_gap_numerical_zero": {
            "value": float(identity.chord_push_velocity_gap_mean.max()),
            "threshold": 1e-6,
            "passed": float(identity.chord_push_velocity_gap_mean.max()) <= 1e-6,
        },
        "orthogonal_bridge_numerical_zero": {
            "value": float(orthogonal.bridge_defect_max.max()),
            "threshold": 1e-6,
            "passed": float(orthogonal.bridge_defect_max.max()) <= 1e-6,
        },
        "orthogonal_prior_within_sampling_control": {
            "value": orthogonal_prior / max(identity_prior, 1e-12),
            "threshold": 1.5,
            "passed": orthogonal_prior <= 1.5 * identity_prior,
        },
        "anisotropic_bridge_numerical_zero": {
            "value": float(linear.bridge_defect_max.max()),
            "threshold": 1e-6,
            "passed": float(linear.bridge_defect_max.max()) <= 1e-6,
        },
        "anisotropic_prior_monotonic": {
            "value": anisotropic_prior_output,
            "threshold": 0.8,
            "passed": bool(np.isfinite(anisotropic_prior) and anisotropic_prior >= 0.8),
            "evaluable": bool(np.isfinite(anisotropic_prior)),
        },
        "adapter_cycle": {
            "value": float(nonlinear.cycle_relative_l2_max.max()),
            "threshold": 1e-5,
            "passed": float(nonlinear.cycle_relative_l2_max.max()) <= 1e-5,
        },
        "adapter_bridge_monotonic": {
            "value": nonlinear_scale_bridge_output,
            "threshold": 0.8,
            "passed": bool(
                np.isfinite(nonlinear_scale_bridge)
                and nonlinear_scale_bridge >= 0.8
            ),
            "evaluable": bool(np.isfinite(nonlinear_scale_bridge)),
        },
        "adapter_bridge_velocity_relation": {
            "value": nonlinear_bridge_velocity_dict["correlation"],
            "ci_low": nonlinear_bridge_velocity_dict["ci_low"],
            "threshold": 0.6,
            "passed": bool(
                nonlinear_bridge_velocity is not None
                and nonlinear_bridge_velocity.correlation >= 0.6
                and nonlinear_bridge_velocity.ci_low > 0.0
            ),
            "evaluable": nonlinear_bridge_velocity is not None,
        },
    }
    acceptance = {
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "decision": (
            "phase2_controls_passed"
            if all(item["passed"] for item in checks.values())
            else (
                "smoke_not_evaluable"
                if any(item.get("evaluable") is False for item in checks.values())
                else "inspect_failed_controls_before_causal_toy"
            )
        ),
    }
    return correlations, acceptance


def _plot_summary(condition_frame: pd.DataFrame, path_frame: pd.DataFrame, path: Path) -> None:
    grouped = (
        condition_frame.groupby(["family", "strength"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["family", "strength"])
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for family, frame in grouped.groupby("family"):
        axes[0, 0].plot(frame.strength, frame.prior_sliced_w1, marker="o", label=family)
        axes[0, 1].plot(frame.strength, frame.bridge_defect_mean, marker="o", label=family)
        axes[1, 0].plot(
            frame.strength,
            frame.chord_push_velocity_gap_mean,
            marker="o",
            label=family,
        )
    nonlinear_path = path_frame[path_frame.family == "nonlinear_adapter"]
    nonlinear_path = (
        nonlinear_path.groupby(["strength", "branch", "time"], as_index=False)
        .mean(numeric_only=True)
    )
    for (strength, branch), frame in nonlinear_path.groupby(["strength", "branch"]):
        if strength not in {0.25, 1.0}:
            continue
        axes[1, 1].plot(
            frame.time,
            frame.ambiguity_ratio,
            marker="o",
            label=f"alpha={strength:g} {branch}",
        )
    titles = (
        "Source-prior mismatch (projected SW1)",
        "Chord vs pushforward state defect",
        "Chord vs pushforward velocity gap",
        "Local velocity ambiguity proxy",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    axes[0, 0].set_xlabel("condition / alpha")
    axes[0, 1].set_xlabel("condition / alpha")
    axes[1, 0].set_xlabel("condition / alpha")
    axes[1, 1].set_xlabel("t (data=0, noise=1)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: TransportAuditConfig) -> dict[str, object] | None:
    configure_fp32()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    rank, world_size, local_rank = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("this real RAE audit requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(config.seed + rank)
    if config.count % world_size or config.path_count % world_size:
        raise ValueError("count and path_count must be divisible by world_size")
    if config.path_count > config.count:
        raise ValueError("path_count cannot exceed count")

    run_name = _broadcast_run_name(config, rank)
    run_dir = Path(config.output_root).expanduser() / run_name
    if rank == 0:
        if run_dir.exists() and not config.overwrite:
            raise FileExistsError(f"run exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(_jsonable_config(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        dist.barrier()

    dataset = load_named_dataset(
        config.dataset_name,
        config.data_root,
        split=config.dataset_split,
        dataset_path=config.dataset_path,
    )
    all_indices = _select_indices(len(dataset), config)
    del dataset
    local_indices = all_indices[rank::world_size]
    start_time = time.time()
    z_cpu, z_flip_cpu, labels_local, observed_indices = _encode_local_shard(
        config,
        local_indices,
        device,
    )
    labels_global = _gather_tensor(labels_local, device)
    indices_global = _gather_tensor(observed_indices, device)
    reference_knn_indices = _full_latent_knn_indices(
        z_cpu,
        config.neighbors,
        device,
    )

    adapter = _load_adapter_from_checkpoint(
        config.adapter_checkpoint,
        channels=z_cpu.shape[1],
        hidden_channels=None,
        blocks=None,
    ).to(device=device, dtype=torch.float32).eval()
    adapter.requires_grad_(False)
    sketch = LatentSketch(
        z_cpu.shape[1],
        projected_channels=config.sketch_channels,
        spatial_size=config.sketch_spatial_size,
        seed=config.sketch_seed,
    ).to(device=device, dtype=torch.float32).eval()

    condition_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    specs = _condition_specs(config)
    for condition_index, spec in enumerate(specs):
        transform = _make_transform(spec, adapter, z_cpu.shape[1], device)
        local_values = _condition_local_tensors(
            z_cpu,
            z_flip_cpu,
            transform,
            sketch,
            config,
            rank,
            device,
            int(spec["noise_seed"]),
            reference_knn_indices,
        )
        global_values = _gather_condition(local_values, device)
        if rank == 0:
            row, rows = _summarize_condition(
                spec,
                global_values,
                labels_global,
                config,
            )
            condition_rows.append(row)
            path_rows.extend(rows)
            print(
                f"[{condition_index + 1}/{len(specs)}] {spec['condition_id']} "
                f"prior={row['prior_sliced_w1']:.4f} "
                f"bridge={row['bridge_defect_mean']:.4f} "
                f"cycle={row['cycle_relative_l2_max']:.2e}",
                flush=True,
            )
        del transform, local_values, global_values
        torch.cuda.empty_cache()
    del adapter, sketch
    gc.collect()
    torch.cuda.empty_cache()

    if rank != 0:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return None

    condition_frame = pd.DataFrame(condition_rows)
    path_frame = pd.DataFrame(path_rows)
    correlations, acceptance = _correlations_and_acceptance(condition_frame)
    condition_frame.to_csv(run_dir / "condition_metrics.csv", index=False)
    path_frame.to_csv(run_dir / "path_metrics.csv", index=False)
    (run_dir / "correlations.json").write_text(
        json.dumps(correlations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_summary(condition_frame, path_frame, run_dir / "transport_audit.png")
    result = {
        "protocol_version": 2,
        "timestamp": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "elapsed_seconds": time.time() - start_time,
        "world_size": world_size,
        "sample_count": config.count,
        "path_sample_count": config.path_count,
        "latent_shape": list(z_cpu.shape[1:]),
        "selected_indices": [int(value) for value in indices_global.tolist()],
        "leakage_audit": {
            "adapter_train_split": "ImageNet train indices 0..32767",
            "adapter_old_external_eval": "ImageNet validation indices 0..2047",
            "current_split": config.dataset_split,
            "current_minimum_index": int(indices_global.min()),
            "excluded_prefix": config.excluded_prefix,
            "overlap_with_old_external_eval": bool(
                (indices_global < config.excluded_prefix).any()
            ),
        },
        "metric_scope": {
            "full_latent": [
                "equivariance_error",
                "cycle_error",
                "bridge_defect",
                "chord_push_velocity_gap",
                "directional_jacobian_gain",
                "within-rank-shard exact kNN overlap",
            ],
            "fixed_projection": [
                "prior_sliced_w1",
                "projected_shared_class_viv",
                "projected_knn_overlap_auxiliary_only",
                "local_velocity_ambiguity_proxy",
            ],
            "warning": (
                "Projected shared-class VIV is not full class-specific VIV; "
                "kNN ambiguity is a proxy and is never reported as VIV."
            ),
        },
        "correlations": correlations,
        "acceptance": acceptance,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def parse_args() -> TransportAuditConfig:
    defaults = TransportAuditConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", default=defaults.adapter_checkpoint)
    parser.add_argument("--dataset-name", default=defaults.dataset_name)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--dataset-path", default=defaults.dataset_path)
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--excluded-prefix", type=int, default=defaults.excluded_prefix)
    parser.add_argument("--count", type=int, default=defaults.count)
    parser.add_argument("--path-count", type=int, default=defaults.path_count)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--repeats", nargs="+", type=int, default=list(defaults.repeats))
    parser.add_argument("--times", nargs="+", type=float, default=list(defaults.times))
    parser.add_argument(
        "--anisotropic-conditions",
        nargs="+",
        type=float,
        default=list(defaults.anisotropic_conditions),
    )
    parser.add_argument(
        "--adapter-scales",
        nargs="+",
        type=float,
        default=list(defaults.adapter_scales),
    )
    parser.add_argument("--sketch-channels", type=int, default=defaults.sketch_channels)
    parser.add_argument(
        "--sketch-spatial-size",
        type=int,
        default=defaults.sketch_spatial_size,
    )
    parser.add_argument("--sketch-seed", type=int, default=defaults.sketch_seed)
    parser.add_argument("--sliced-directions", type=int, default=defaults.sliced_directions)
    parser.add_argument("--neighbors", type=int, default=defaults.neighbors)
    parser.add_argument("--rae-repo-path", default=defaults.rae_repo_path)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return TransportAuditConfig(
        adapter_checkpoint=args.adapter_checkpoint,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        excluded_prefix=args.excluded_prefix,
        count=args.count,
        path_count=args.path_count,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        repeats=tuple(args.repeats),
        times=tuple(args.times),
        anisotropic_conditions=tuple(args.anisotropic_conditions),
        adapter_scales=tuple(args.adapter_scales),
        sketch_channels=args.sketch_channels,
        sketch_spatial_size=args.sketch_spatial_size,
        sketch_seed=args.sketch_seed,
        sliced_directions=args.sliced_directions,
        neighbors=args.neighbors,
        rae_repo_path=args.rae_repo_path,
        output_root=args.output_root,
        run_name=args.run_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
