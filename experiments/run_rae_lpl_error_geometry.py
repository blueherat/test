"""Run a paired four-seed decoder-geometry audit for RAE Flow vs LPL errors.

Each torchrun rank owns one independently trained Flow/LPL checkpoint pair.
All ranks evaluate the same held-out ImageNet latents, labels, noise, and time
points.  No model is trained and no ImageNet validation sample is used for
checkpoint selection or calibration.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_decoder_risk_phase0 import decoder_hidden_features  # noqa: E402
from experiments.rae_latent_cache import CachedRAELatentDataset  # noqa: E402
from experiments.rae_lpl_error_geometry import (  # noqa: E402
    cosine_per_sample,
    finite_difference_feature_gain,
    paired_amplitudes,
    raw_feature_layer_losses,
    sample_rms,
    scale_direction_to_rms,
    shuffled_direction,
    summarize_geometry_rows,
    unit_rms_direction,
)
from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_hidden_indices,
    flow_clean_estimate,
    strict_lpl_per_sample,
)
from experiments.rae_teacher_rollout_gap import configure_fp32, load_frozen_decoder  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "experiments/configs/rae_strict_lpl_ditdh_s_dinov2.yaml"
DEFAULT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_decoder_risk_phase0/seed20260718_cal1024_test2048_fp32"
)
DEFAULT_OUTPUT = Path.home() / "data/eqvae/experiments/rae_lpl_error_geometry"


@dataclass(frozen=True)
class CheckpointPair:
    name: str
    training_seed: int
    flow: Path
    lpl: Path


def parse_pair(value: str) -> CheckpointPair:
    """Parse ``NAME=FLOW_CHECKPOINT,LPL_CHECKPOINT``."""

    if "=" not in value or "," not in value:
        raise argparse.ArgumentTypeError(
            "pair must be NAME=FLOW_CHECKPOINT,LPL_CHECKPOINT"
        )
    name, paths = value.split("=", 1)
    flow, lpl = paths.split(",", 1)
    match = re.search(r"(\d+)$", name)
    if not name or match is None:
        raise argparse.ArgumentTypeError("pair NAME must end in its integer training seed")
    pair = CheckpointPair(
        name=name,
        training_seed=int(match.group(1)),
        flow=Path(flow).expanduser().resolve(),
        lpl=Path(lpl).expanduser().resolve(),
    )
    for path in (pair.flow, pair.lpl):
        if not path.exists():
            raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return pair


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def distributed_context(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("the RAE decoder geometry audit requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "RANK" in os.environ:
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    configure_fp32(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rank, world_size, device


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def finish_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def load_manifest(cache: Path) -> dict[str, object]:
    path = cache / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"complete", "calibration_count", "test_count", "latent_shape"}
    if missing := required.difference(payload):
        raise KeyError(f"cache manifest is missing {sorted(missing)}")
    if not bool(payload["complete"]):
        raise RuntimeError(f"cache is incomplete: {cache}")
    return payload


def load_stage2(
    stage2_config: OmegaConf,
    checkpoint: Path,
    state_key: str,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    model = instantiate_from_config(stage2_config).to(device=device, dtype=torch.float32)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state_key not in payload:
        raise KeyError(f"{checkpoint} lacks state key {state_key!r}")
    model.load_state_dict(payload[state_key], strict=True)
    model.requires_grad_(False).eval()
    step = int(payload.get("step", -1))
    del payload
    gc.collect()
    return model, step


def deterministic_tensor(
    shape: Sequence[int],
    *,
    seed: int,
    sample_index: int,
    offset: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(offset) + int(sample_index) * 1_000_003
    )
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)


def spatial_features(features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    result = []
    for feature in features:
        if feature.ndim != 3:
            raise ValueError("decoder hidden features must have shape [B,N,C]")
        side = int(round(feature.shape[1] ** 0.5))
        if side * side != feature.shape[1]:
            raise ValueError(f"decoder token count is not square: {feature.shape[1]}")
        result.append(
            feature.transpose(1, 2).reshape(
                feature.shape[0], feature.shape[2], side, side
            )
        )
    return tuple(result)


@torch.no_grad()
def batched_decoder_features(
    rae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    hidden_indices: Sequence[int],
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    parts: list[list[torch.Tensor]] | None = None
    for start in range(0, len(latents), int(batch_size)):
        features = decoder_hidden_features(
            rae,
            latents[start : start + int(batch_size)],
            hidden_indices=hidden_indices,
        )
        if parts is None:
            parts = [[] for _ in features]
        for layer_parts, feature in zip(parts, features, strict=True):
            layer_parts.append(feature)
    if parts is None:
        raise ValueError("cannot decode an empty latent batch")
    return tuple(torch.cat(layer_parts, dim=0) for layer_parts in parts)


def select_features(
    features: Sequence[torch.Tensor],
    index: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(feature[index : index + 1] for feature in features)


def repeated_features(
    features: Sequence[torch.Tensor],
    count: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(feature.expand(int(count), *feature.shape[1:]) for feature in features)


def _candidate_index(names: Sequence[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as error:
        raise KeyError(name) from error


@torch.no_grad()
def evaluate_pair_observation(
    *,
    rae: torch.nn.Module,
    flow_model: torch.nn.Module,
    lpl_model: torch.nn.Module,
    clean: torch.Tensor,
    label: torch.Tensor,
    noise: torch.Tensor,
    ratio: float,
    random_direction: torch.Tensor,
    hidden_indices: Sequence[int],
    decoder_batch_size: int,
    local_fraction: float,
    finite_difference_fraction: float,
    strict_lpl_config: dict[str, object],
) -> dict[str, float]:
    """Evaluate one exactly paired sample/time observation."""

    if len(clean) != 1:
        raise ValueError("pair observations are evaluated one sample at a time")
    time_value = float(ratio) / (1.0 + float(ratio))
    time = clean.new_full((1,), time_value)
    noisy = (1.0 - time_value) * clean + time_value * noise
    target_velocity = noise - clean
    flow_velocity = flow_model(noisy, time, y=label)
    lpl_velocity = lpl_model(noisy, time, y=label)
    flow_clean = flow_clean_estimate(noisy, flow_velocity, time)
    lpl_clean = flow_clean_estimate(noisy, lpl_velocity, time)
    flow_error = flow_clean - clean
    lpl_error = lpl_clean - clean
    flow_direction = unit_rms_direction(flow_error)
    lpl_direction = unit_rms_direction(lpl_error)
    random_unit = unit_rms_direction(random_direction)
    shuffled_lpl = unit_rms_direction(shuffled_direction(lpl_error))

    realistic_rms, local_rms = paired_amplitudes(
        clean,
        flow_error,
        lpl_error,
        local_fraction=float(local_fraction),
    )
    finite_difference_rms = (
        float(finite_difference_fraction) * sample_rms(clean)
    ).clamp_min(1e-6)

    candidate_map = {
        "flow_actual": flow_clean,
        "lpl_actual": lpl_clean,
        "flow_matched": clean + scale_direction_to_rms(flow_direction, realistic_rms),
        "lpl_matched": clean + scale_direction_to_rms(lpl_direction, realistic_rms),
        "flow_local": clean + scale_direction_to_rms(flow_direction, local_rms),
        "lpl_local": clean + scale_direction_to_rms(lpl_direction, local_rms),
        "random_local": clean + scale_direction_to_rms(random_unit, local_rms),
        "shuffled_lpl_local": clean + scale_direction_to_rms(shuffled_lpl, local_rms),
        "flow_fd_plus": clean
        + scale_direction_to_rms(flow_direction, finite_difference_rms),
        "flow_fd_minus": clean
        - scale_direction_to_rms(flow_direction, finite_difference_rms),
        "lpl_fd_plus": clean
        + scale_direction_to_rms(lpl_direction, finite_difference_rms),
        "lpl_fd_minus": clean
        - scale_direction_to_rms(lpl_direction, finite_difference_rms),
        "random_fd_plus": clean
        + scale_direction_to_rms(random_unit, finite_difference_rms),
        "random_fd_minus": clean
        - scale_direction_to_rms(random_unit, finite_difference_rms),
    }
    candidate_names = list(candidate_map)
    candidates = torch.cat(list(candidate_map.values()), dim=0)

    reference = decoder_hidden_features(rae, clean, hidden_indices=hidden_indices)
    candidate_features = batched_decoder_features(
        rae,
        candidates,
        hidden_indices=hidden_indices,
        batch_size=int(decoder_batch_size),
    )
    reference_repeated = repeated_features(reference, len(candidates))
    raw_layers = raw_feature_layer_losses(candidate_features, reference_repeated)
    raw_losses = raw_layers.mean(dim=1)

    strict_target = repeated_features(spatial_features(reference), len(candidates))
    strict_candidate = spatial_features(candidate_features)
    strict_losses, strict_details = strict_lpl_per_sample(
        strict_target,
        strict_candidate,
        layer_weights=[1.0] * len(hidden_indices),
        outlier_quantile=float(strict_lpl_config["outlier_quantile"]),
        outlier_opening=int(strict_lpl_config["outlier_opening"]),
        outlier_closing=int(strict_lpl_config["outlier_closing"]),
        eps=float(strict_lpl_config["normalization_eps"]),
    )

    flow_fd_layers, flow_fd_gain = finite_difference_feature_gain(
        select_features(candidate_features, _candidate_index(candidate_names, "flow_fd_plus")),
        select_features(candidate_features, _candidate_index(candidate_names, "flow_fd_minus")),
        finite_difference_rms,
    )
    lpl_fd_layers, lpl_fd_gain = finite_difference_feature_gain(
        select_features(candidate_features, _candidate_index(candidate_names, "lpl_fd_plus")),
        select_features(candidate_features, _candidate_index(candidate_names, "lpl_fd_minus")),
        finite_difference_rms,
    )
    random_fd_layers, random_fd_gain = finite_difference_feature_gain(
        select_features(candidate_features, _candidate_index(candidate_names, "random_fd_plus")),
        select_features(candidate_features, _candidate_index(candidate_names, "random_fd_minus")),
        finite_difference_rms,
    )

    def raw(name: str) -> float:
        return float(raw_losses[_candidate_index(candidate_names, name)].item())

    def strict(name: str) -> float:
        return float(strict_losses[_candidate_index(candidate_names, name)].item())

    def symmetric_strict_gain(prefix: str) -> float:
        plus = strict(f"{prefix}_fd_plus")
        minus = strict(f"{prefix}_fd_minus")
        return 0.5 * (plus + minus) / max(
            float(finite_difference_rms.square().item()), 1e-30
        )

    def layer(name: str, layer_index: int) -> float:
        candidate_index = _candidate_index(candidate_names, name)
        return float(raw_layers[candidate_index, layer_index].item())

    flow_error_rms = sample_rms(flow_error)
    lpl_error_rms = sample_rms(lpl_error)
    local_denominator = float(local_rms.square().item())
    realistic_denominator = float(realistic_rms.square().item())
    flow_actual_raw = raw("flow_actual")
    lpl_actual_raw = raw("lpl_actual")
    flow_fd_strict_gain = symmetric_strict_gain("flow")
    lpl_fd_strict_gain = symmetric_strict_gain("lpl")
    random_fd_strict_gain = symmetric_strict_gain("random")
    row = {
        "time": time_value,
        "noise_to_signal_ratio": float(ratio),
        "clean_rms": float(sample_rms(clean).item()),
        "flow_velocity_mse": float(
            (flow_velocity - target_velocity).square().flatten(1).mean().item()
        ),
        "lpl_velocity_mse": float(
            (lpl_velocity - target_velocity).square().flatten(1).mean().item()
        ),
        "flow_latent_mse": float(flow_error_rms.square().item()),
        "lpl_latent_mse": float(lpl_error_rms.square().item()),
        "flow_lpl_error_cosine": float(cosine_per_sample(flow_error, lpl_error).item()),
        "realistic_matched_rms": float(realistic_rms.item()),
        "local_matched_rms": float(local_rms.item()),
        "finite_difference_rms": float(finite_difference_rms.item()),
        "flow_actual_raw_loss": flow_actual_raw,
        "lpl_actual_raw_loss": lpl_actual_raw,
        "flow_actual_amplification": flow_actual_raw
        / max(float(flow_error_rms.square().item()), 1e-30),
        "lpl_actual_amplification": lpl_actual_raw
        / max(float(lpl_error_rms.square().item()), 1e-30),
        "flow_actual_strict_lpl": strict("flow_actual"),
        "lpl_actual_strict_lpl": strict("lpl_actual"),
        "flow_matched_raw_loss": raw("flow_matched"),
        "lpl_matched_raw_loss": raw("lpl_matched"),
        "flow_matched_gain": raw("flow_matched") / max(realistic_denominator, 1e-30),
        "lpl_matched_gain": raw("lpl_matched") / max(realistic_denominator, 1e-30),
        "flow_matched_strict_lpl": strict("flow_matched"),
        "lpl_matched_strict_lpl": strict("lpl_matched"),
        "flow_matched_strict_gain": strict("flow_matched")
        / max(realistic_denominator, 1e-30),
        "lpl_matched_strict_gain": strict("lpl_matched")
        / max(realistic_denominator, 1e-30),
        "flow_local_raw_loss": raw("flow_local"),
        "lpl_local_raw_loss": raw("lpl_local"),
        "random_local_raw_loss": raw("random_local"),
        "shuffled_lpl_local_raw_loss": raw("shuffled_lpl_local"),
        "flow_local_gain": raw("flow_local") / max(local_denominator, 1e-30),
        "lpl_local_gain": raw("lpl_local") / max(local_denominator, 1e-30),
        "random_local_gain": raw("random_local") / max(local_denominator, 1e-30),
        "shuffled_lpl_local_gain": raw("shuffled_lpl_local")
        / max(local_denominator, 1e-30),
        "flow_local_strict_lpl": strict("flow_local"),
        "lpl_local_strict_lpl": strict("lpl_local"),
        "random_local_strict_lpl": strict("random_local"),
        "shuffled_lpl_local_strict_lpl": strict("shuffled_lpl_local"),
        "flow_local_strict_gain": strict("flow_local")
        / max(local_denominator, 1e-30),
        "lpl_local_strict_gain": strict("lpl_local")
        / max(local_denominator, 1e-30),
        "random_local_strict_gain": strict("random_local")
        / max(local_denominator, 1e-30),
        "shuffled_lpl_local_strict_gain": strict("shuffled_lpl_local")
        / max(local_denominator, 1e-30),
        "flow_fd_gain": float(flow_fd_gain.item()),
        "lpl_fd_gain": float(lpl_fd_gain.item()),
        "random_fd_gain": float(random_fd_gain.item()),
        "flow_fd_strict_gain": flow_fd_strict_gain,
        "lpl_fd_strict_gain": lpl_fd_strict_gain,
        "random_fd_strict_gain": random_fd_strict_gain,
        "flow_quadratic_prediction": float(
            flow_fd_gain.item() * flow_error_rms.square().item()
        ),
        "lpl_quadratic_prediction": float(
            lpl_fd_gain.item() * lpl_error_rms.square().item()
        ),
        "flow_strict_quadratic_prediction": float(
            flow_fd_strict_gain * flow_error_rms.square().item()
        ),
        "lpl_strict_quadratic_prediction": float(
            lpl_fd_strict_gain * lpl_error_rms.square().item()
        ),
        "strict_mask_keep_fraction": float(
            strict_details["mask_keep_fraction"].mean().item()
        ),
    }
    for layer_index in range(len(hidden_indices)):
        row[f"flow_actual_raw_layer{layer_index}"] = layer(
            "flow_actual", layer_index
        )
        row[f"lpl_actual_raw_layer{layer_index}"] = layer("lpl_actual", layer_index)
        row[f"flow_local_gain_layer{layer_index}"] = layer(
            "flow_local", layer_index
        ) / max(local_denominator, 1e-30)
        row[f"lpl_local_gain_layer{layer_index}"] = layer(
            "lpl_local", layer_index
        ) / max(local_denominator, 1e-30)
        row[f"flow_fd_gain_layer{layer_index}"] = float(
            flow_fd_layers[0, layer_index].item()
        )
        row[f"lpl_fd_gain_layer{layer_index}"] = float(
            lpl_fd_layers[0, layer_index].item()
        )
        row[f"random_fd_gain_layer{layer_index}"] = float(
            random_fd_layers[0, layer_index].item()
        )
    return row


def ratio_summary(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (seed, ratio), group in rows.groupby(
        ["training_seed", "noise_to_signal_ratio"], sort=True
    ):
        local_ratio = group["lpl_local_gain"] / group["flow_local_gain"].clip(lower=1e-30)
        fd_ratio = group["lpl_fd_gain"] / group["flow_fd_gain"].clip(lower=1e-30)
        strict_local_ratio = group["lpl_local_strict_gain"] / group[
            "flow_local_strict_gain"
        ].clip(lower=1e-30)
        strict_fd_ratio = group["lpl_fd_strict_gain"] / group[
            "flow_fd_strict_gain"
        ].clip(lower=1e-30)
        result.append(
            {
                "training_seed": int(seed),
                "noise_to_signal_ratio": float(ratio),
                "observations": int(len(group)),
                "latent_mse_lpl_over_flow_mean": float(
                    group["lpl_latent_mse"].mean()
                    / max(group["flow_latent_mse"].mean(), 1e-30)
                ),
                "actual_raw_lpl_over_flow_mean": float(
                    group["lpl_actual_raw_loss"].mean()
                    / max(group["flow_actual_raw_loss"].mean(), 1e-30)
                ),
                "local_gain_lpl_over_flow_gmean": float(
                    np.exp(np.log(local_ratio.clip(lower=1e-30)).mean())
                ),
                "local_lpl_better_fraction": float((local_ratio < 1.0).mean()),
                "fd_gain_lpl_over_flow_gmean": float(
                    np.exp(np.log(fd_ratio.clip(lower=1e-30)).mean())
                ),
                "local_strict_gain_lpl_over_flow_gmean": float(
                    np.exp(np.log(strict_local_ratio.clip(lower=1e-30)).mean())
                ),
                "fd_strict_gain_lpl_over_flow_gmean": float(
                    np.exp(np.log(strict_fd_ratio.clip(lower=1e-30)).mean())
                ),
            }
        )
    return pd.DataFrame(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pair", type=parse_pair, action="append", required=True)
    parser.add_argument("--state-key", choices=("model", "ema"), default="ema")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument(
        "--noise-ratios", type=float, nargs="+", default=(0.5, 1.0, 2.0, 3.0)
    )
    parser.add_argument("--decoder-batch-size", type=int, default=2)
    parser.add_argument("--local-fraction", type=float, default=0.01)
    parser.add_argument("--finite-difference-fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, device = distributed_context(args.seed)
    if len(args.pair) != world_size:
        finish_distributed()
        raise ValueError(
            f"received {len(args.pair)} checkpoint pairs for world size {world_size}"
        )
    if int(args.sample_count) < 1:
        finish_distributed()
        raise ValueError("sample-count must be positive")
    if int(args.decoder_batch_size) < 1:
        finish_distributed()
        raise ValueError("decoder-batch-size must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.expanduser().resolve()
    manifest = load_manifest(cache)
    calibration_count = int(manifest["calibration_count"])
    test_count = int(manifest["test_count"])
    if int(args.sample_count) > test_count:
        finish_distributed()
        raise ValueError("sample-count exceeds held-out cache size")
    dataset = CachedRAELatentDataset(
        cache,
        start=calibration_count,
        stop=calibration_count + int(args.sample_count),
    )

    config = OmegaConf.load(args.config.expanduser().resolve())
    stage1_config = OmegaConf.create(OmegaConf.to_container(config.stage_1, resolve=True))
    stage2_config = OmegaConf.create(OmegaConf.to_container(config.stage_2, resolve=True))
    strict_config = OmegaConf.to_container(config.training.strict_lpl, resolve=True)
    rae = load_frozen_decoder(stage1_config)
    rae = rae.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
    hidden_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        tuple(float(value) for value in strict_config["layer_fractions"]),
    )

    pair = args.pair[rank]
    flow_model, flow_step = load_stage2(
        stage2_config, pair.flow, args.state_key, device
    )
    lpl_model, lpl_step = load_stage2(stage2_config, pair.lpl, args.state_key, device)
    if flow_step != lpl_step:
        finish_distributed()
        raise ValueError(f"paired checkpoint steps differ: {flow_step} vs {lpl_step}")

    started = perf_counter()
    rows = []
    for sample_index in range(int(args.sample_count)):
        clean_cpu, label_value = dataset[sample_index]
        clean = clean_cpu[None].to(device=device, dtype=torch.float32)
        label = torch.tensor([label_value], device=device, dtype=torch.long)
        noise = deterministic_tensor(
            clean.shape,
            seed=int(args.seed),
            sample_index=sample_index,
            offset=10_000_019,
        ).to(device)
        for ratio_index, ratio in enumerate(args.noise_ratios):
            random_direction = deterministic_tensor(
                clean.shape,
                seed=int(args.seed),
                sample_index=sample_index,
                offset=20_000_033 + ratio_index * 10_007,
            ).to(device)
            row = evaluate_pair_observation(
                rae=rae,
                flow_model=flow_model,
                lpl_model=lpl_model,
                clean=clean,
                label=label,
                noise=noise,
                ratio=float(ratio),
                random_direction=random_direction,
                hidden_indices=hidden_indices,
                decoder_batch_size=int(args.decoder_batch_size),
                local_fraction=float(args.local_fraction),
                finite_difference_fraction=float(args.finite_difference_fraction),
                strict_lpl_config=dict(strict_config),
            )
            row.update(
                {
                    "checkpoint_pair": pair.name,
                    "training_seed": int(pair.training_seed),
                    "checkpoint_step": int(flow_step),
                    "state_key": args.state_key,
                    "sample_index": int(sample_index),
                    "label": int(label_value),
                }
            )
            rows.append(row)
        print(
            f"[rank {rank}] {pair.name}: sample {sample_index + 1}/{args.sample_count}",
            flush=True,
        )

    rank_table = pd.DataFrame(rows)
    rank_path = output / f"rows_rank{rank:02d}.csv"
    rank_table.to_csv(rank_path, index=False)
    atomic_json(
        output / f"manifest_rank{rank:02d}.json",
        {
            "checkpoint_pair": pair.name,
            "training_seed": int(pair.training_seed),
            "flow_checkpoint": str(pair.flow),
            "lpl_checkpoint": str(pair.lpl),
            "checkpoint_step": int(flow_step),
            "state_key": args.state_key,
            "sample_count": int(args.sample_count),
            "noise_ratios": [float(value) for value in args.noise_ratios],
            "hidden_indices": list(hidden_indices),
            "local_fraction": float(args.local_fraction),
            "finite_difference_fraction": float(args.finite_difference_fraction),
            "elapsed_seconds": perf_counter() - started,
            "rows": int(len(rank_table)),
        },
    )
    barrier()

    if rank == 0:
        table = pd.concat(
            [pd.read_csv(output / f"rows_rank{index:02d}.csv") for index in range(world_size)],
            ignore_index=True,
        )
        table.to_csv(output / "rows.csv", index=False)
        ratio_table = ratio_summary(table)
        ratio_table.to_csv(output / "by_seed_ratio.csv", index=False)
        seed_table, gate = summarize_geometry_rows(
            table, required_seed_count=world_size
        )
        seed_table.to_csv(output / "by_seed.csv", index=False)
        atomic_json(
            output / "summary.json",
            {
                "experiment": "RAE Flow-vs-LPL matched-error decoder geometry",
                "dataset": "ImageNet-1k validation latent cache",
                "cache": str(cache),
                "precision": "fp32",
                "tf32": False,
                "sample_count_per_seed": int(args.sample_count),
                "observation_count": int(len(table)),
                "training_seed_count": int(world_size),
                "noise_ratios": [float(value) for value in args.noise_ratios],
                "hidden_indices": list(hidden_indices),
                "gate": dict(gate),
            },
        )
        print("\nPer-seed mechanism summary")
        print(seed_table.to_string(index=False))
        print("\nPer-seed/time summary")
        print(ratio_table.to_string(index=False))
        print("\nMechanism gate")
        print(json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=True))
        print(output)

    del flow_model, lpl_model, rae
    gc.collect()
    torch.cuda.empty_cache()
    finish_distributed()


if __name__ == "__main__":
    main()
