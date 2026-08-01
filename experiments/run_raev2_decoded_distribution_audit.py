"""Decode matched RAEv2 trajectories and compare them in Inception space.

The experiment is inference-only.  For each image/noise/class tuple it builds
the training-path state p_t and captures the official sampler states q_t with
and without internal guidance.  A frozen RAE decoder maps all three states to
images; held-out linear C2ST and FID then measure the decoder-side effect.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetPacked,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    HeldoutStates,
    SamplerStateRecorder,
    autocast_context,
    bootstrap_auc_delta,
    bootstrap_paired_auc,
    build_requested_labels,
    class_group_split,
    load_config,
    match_requested_times,
    paired_auc,
    select_matching_imagenet_rows,
    shifted_solver_grid,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from utils.guidance_utils import forward_with_internalguidance  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


DEFAULT_DECODE_TIMES = (0.0, 0.2, 0.4, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/"
            "dinov3l-k7/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--packed-data-path",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument(
        "--parquet-data-path",
        type=Path,
        default=Path("/data/shared/imagenet-1k"),
    )
    parser.add_argument(
        "--fid-reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--time", action="append", type=float, dest="times")
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--inception-feature", choices=("64", "192", "768", "2048"), default="2048"
    )
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument("--dino-repo-dir", type=Path)
    return parser.parse_args()


def time_suffix(value: float) -> str:
    return f"t{float(value):.6f}".replace(".", "p")


def fit_feature_probe(
    negative: np.ndarray,
    positive: np.ndarray,
    train_mask: np.ndarray,
    ridge_ratio: float,
) -> tuple[np.ndarray, float, float]:
    """Fit diagonal LDA using only the designated training examples."""

    if negative.shape != positive.shape or negative.ndim != 2:
        raise ValueError("feature arrays must be matching matrices")
    if train_mask.shape != (negative.shape[0],):
        raise ValueError("train mask shape does not match features")
    if train_mask.sum() < 2:
        raise ValueError("at least two training examples are required")
    neg = negative[train_mask].astype(np.float64, copy=False)
    pos = positive[train_mask].astype(np.float64, copy=False)
    mean_neg = neg.mean(axis=0)
    mean_pos = pos.mean(axis=0)
    var_neg = neg.var(axis=0, ddof=1)
    var_pos = pos.var(axis=0, ddof=1)
    pooled = 0.5 * (var_neg + var_pos)
    positive_scale = pooled[pooled > 0]
    base_scale = float(np.median(positive_scale)) if positive_scale.size else 1.0
    ridge = max(float(ridge_ratio) * base_scale, 1e-12)
    weight = (mean_pos - mean_neg) / (pooled + ridge)
    norm = float(np.linalg.norm(weight))
    if norm > 0:
        weight /= norm
    intercept = -0.5 * float(weight @ (mean_pos + mean_neg))
    return weight.astype(np.float32), intercept, ridge


def feature_probe_scores(
    values: np.ndarray, weight: np.ndarray, intercept: float
) -> np.ndarray:
    return values.astype(np.float32, copy=False) @ weight + float(intercept)


def feature_statistics(features: np.ndarray) -> dict[str, np.ndarray]:
    stats = fid_features_to_statistics(torch.from_numpy(features.astype(np.float32)))
    return {
        "mu": np.asarray(stats["mu"], dtype=np.float64),
        "sigma": np.asarray(stats["sigma"], dtype=np.float64),
    }


def fid_between_statistics(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> float:
    metric = fid_statistics_to_metric(first, second, verbose=False)
    value = float(metric["frechet_inception_distance"])
    if not math.isfinite(value):
        raise FloatingPointError("non-finite FID")
    return max(value, 0.0)


def load_reference_statistics(path: Path, feature_name: str) -> dict[str, np.ndarray]:
    if feature_name != "2048":
        raise ValueError("real-reference FID requires --inception-feature 2048")
    with np.load(path) as payload:
        return {
            "mu": np.asarray(payload["mu"], dtype=np.float64),
            "sigma": np.asarray(payload["sigma"], dtype=np.float64),
        }


def _collect_sampler_states(
    *,
    branch: str,
    ig_scale: float,
    model: torch.nn.Module,
    sample_fn: Callable[..., Any],
    config: Any,
    matched_positive_times: list[dict[str, float | int]],
    include_endpoint: bool,
    local_ids: np.ndarray,
    local_labels: np.ndarray,
    local_noise: torch.Tensor,
    per_rank_batch: int,
    log_every_batches: int,
    precision: str,
    device: torch.device,
) -> HeldoutStates:
    store = HeldoutStates()
    model_fn = partial(forward_with_internalguidance, model)
    interval = (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max))
    total_batches = math.ceil(local_ids.size / per_rank_batch)

    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, local_ids.size, per_rank_batch)):
            end = min(start + per_rank_batch, local_ids.size)
            ids = local_ids[start:end]
            labels = torch.from_numpy(local_labels[start:end]).to(device=device)
            noise = local_noise[start:end].to(device=device)
            null = torch.full(
                (noise.shape[0],),
                int(config.misc.num_classes),
                device=device,
                dtype=torch.long,
            )
            context = torch.cat((labels.long(), null), dim=0)
            doubled_noise = torch.cat((noise, noise), dim=0)

            recorder = SamplerStateRecorder(
                model_fn,
                matched_positive_times,
                real_batch_size=noise.shape[0],
                callback=lambda key, state: store.add(key, state, ids),
            )
            with autocast_context(precision):
                trajectory = sample_fn(
                    doubled_noise,
                    recorder,
                    context=context,
                    attn_mask=None,
                    ig_scale=float(ig_scale),
                    ig_interval=interval,
                )
            recorder.validate(int(config.sampler.num_steps))
            if include_endpoint:
                endpoint = trajectory[-1]
                if endpoint.shape[0] == 2 * noise.shape[0]:
                    endpoint = endpoint.chunk(2, dim=0)[0]
                if endpoint.shape[0] != noise.shape[0]:
                    raise RuntimeError("unexpected endpoint batch size")
                store.add(0.0, endpoint, ids)
            if dist.get_rank() == 0 and (
                (batch_index + 1) % log_every_batches == 0
                or batch_index + 1 == total_batches
            ):
                print(
                    f"[{branch}] sampler batches {batch_index + 1}/{total_batches}",
                    flush=True,
                )
    return store


def _decode_features(
    rae: torch.nn.Module,
    extractor: torch.nn.Module,
    states: torch.Tensor,
    ids: np.ndarray,
    *,
    decode_batch: int,
    precision: str,
    example_ids: set[int],
    device: torch.device,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, float]]:
    features: list[np.ndarray] = []
    examples: dict[int, np.ndarray] = {}
    clipped_low = 0
    clipped_high = 0
    pixel_count = 0
    raw_min = math.inf
    raw_max = -math.inf
    with torch.inference_mode():
        for start in range(0, states.shape[0], decode_batch):
            end = min(start + decode_batch, states.shape[0])
            latent = states[start:end].to(device=device, dtype=torch.float32)
            with autocast_context(precision):
                decoded = rae.decode(latent).float()
            if not torch.isfinite(decoded).all():
                raise FloatingPointError("decoder produced non-finite pixels")
            raw_min = min(raw_min, float(decoded.min().item()))
            raw_max = max(raw_max, float(decoded.max().item()))
            clipped_low += int((decoded < 0).sum().item())
            clipped_high += int((decoded > 1).sum().item())
            pixel_count += decoded.numel()
            images = decoded.clamp(0, 1).mul(255).to(torch.uint8)
            feature = extractor(images)[0].float().cpu().numpy()
            features.append(feature)
            for offset, sample_id in enumerate(ids[start:end].tolist()):
                if int(sample_id) in example_ids:
                    examples[int(sample_id)] = (
                        images[offset].permute(1, 2, 0).cpu().numpy()
                    )
    return (
        np.concatenate(features).astype(np.float32, copy=False),
        examples,
        {
            "raw_min": raw_min,
            "raw_max": raw_max,
            "clipped_low_fraction": clipped_low / max(pixel_count, 1),
            "clipped_high_fraction": clipped_high / max(pixel_count, 1),
        },
    )


def _load_feature_shards(
    output_dir: Path, world_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    shards = [
        np.load(output_dir / f"decoded_features_rank{rank:02d}.npz")
        for rank in range(world_size)
    ]
    ids = np.concatenate([item["ids"] for item in shards])
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(ids.size)):
        raise RuntimeError("decoded feature shards do not contain every sample exactly once")
    labels = np.concatenate([item["labels"] for item in shards])[order]
    test_mask = np.concatenate([item["test_mask"] for item in shards])[order].astype(bool)
    feature_keys = sorted(key for key in shards[0].files if key.startswith("feat_"))
    features = {
        key: np.concatenate([item[key] for item in shards], axis=0)[order]
        for key in feature_keys
    }
    examples: dict[str, Any] = {}
    for shard in shards:
        for key in shard.files:
            if key.startswith("example_"):
                examples[key] = shard[key]
    for shard in shards:
        shard.close()
    return ids[order], labels, test_mask, features, examples


def _plot_decoded_examples(
    output_dir: Path,
    world_size: int,
    requested_times: tuple[float, ...],
) -> None:
    originals: dict[int, np.ndarray] = {}
    decoded: dict[tuple[str, float, int], np.ndarray] = {}
    for rank in range(world_size):
        with np.load(output_dir / f"decoded_features_rank{rank:02d}.npz") as shard:
            for sample_id, image in zip(
                shard["example_ids"].tolist(), shard["example_original"]
            ):
                originals[int(sample_id)] = image
            for condition in ("p", "full", "ig"):
                for requested_time in requested_times:
                    suffix = time_suffix(requested_time)
                    ids = shard[f"example_ids_{condition}_{suffix}"]
                    images = shard[f"example_{condition}_{suffix}"]
                    for sample_id, image in zip(ids.tolist(), images):
                        decoded[(condition, requested_time, int(sample_id))] = image

    ordered_times = sorted(requested_times, reverse=True)
    for sample_id in sorted(originals):
        fig, axes = plt.subplots(
            len(ordered_times), 4, figsize=(11.5, 2.8 * len(ordered_times)), squeeze=False
        )
        for row, requested_time in enumerate(ordered_times):
            images = [
                originals[sample_id],
                decoded[("p", requested_time, sample_id)],
                decoded[("full", requested_time, sample_id)],
                decoded[("ig", requested_time, sample_id)],
            ]
            for column, image in enumerate(images):
                axes[row, column].imshow(image)
                axes[row, column].axis("off")
            axes[row, 0].set_ylabel(f"t={requested_time:g}", rotation=0, labelpad=28)
        for column, title in enumerate(("original x", "D(p_t)", "D(q_full)", "D(q_IG)")):
            axes[0, column].set_title(title)
        fig.suptitle(f"Matched decoded trajectories, sample ID {sample_id}")
        fig.tight_layout()
        fig.savefig(output_dir / f"decoded_example_id{sample_id:04d}.png", dpi=170)
        plt.close(fig)


def _plot_metric_curves(summary: pd.DataFrame, output: Path) -> None:
    frame = summary.sort_values("actual_time", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    axes[0].plot(frame["actual_time"], frame["auc_full"], "o-", label="full")
    axes[0].plot(frame["actual_time"], frame["auc_ig"], "s-", label="IG")
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Decoded Inception-feature AUC")
    axes[1].plot(frame["actual_time"], frame["fid_real_full"], "o-", label="full")
    axes[1].plot(frame["actual_time"], frame["fid_real_ig"], "s-", label="IG")
    axes[1].plot(frame["actual_time"], frame["fid_real_p"], "^-", label="p_t")
    axes[1].set_ylabel("FID to ImageNet reference")
    axes[2].plot(frame["actual_time"], frame["fid_p_full"], "o-", label="full")
    axes[2].plot(frame["actual_time"], frame["fid_p_ig"], "s-", label="IG")
    axes[2].set_ylabel("FID to decoded p_t")
    for axis in axes:
        axis.set_xlabel("Solver time t (sampling: 1 to 0)")
        axis.invert_xaxis()
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False)
    fig.suptitle("RAEv2 Internal Guidance After the Frozen Decoder")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.samples <= 0 or args.per_rank_batch <= 0 or args.decode_batch <= 0:
        raise ValueError("sample and batch counts must be positive")
    if args.example_count < 0:
        raise ValueError("--example-count cannot be negative")
    requested_times = tuple(sorted(set(args.times or DEFAULT_DECODE_TIMES)))
    if any(value < 0.0 or value > 1.0 for value in requested_times):
        raise ValueError("decode times must lie in [0, 1]")
    include_endpoint = 0.0 in requested_times
    positive_times = tuple(value for value in requested_times if value > 0.0)
    if not positive_times:
        raise ValueError("at least one positive solver time is required")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    if args.dino_repo_dir is not None:
        os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    config = load_config(args.config.expanduser().resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_solver_grid(int(config.sampler.num_steps), shift)
    matched_positive = match_requested_times(positive_times, grid)
    time_rows = ([{"requested_time": 0.0, "solver_index": int(config.sampler.num_steps), "actual_time": 0.0,
                   "absolute_time_error": 0.0}] if include_endpoint else []) + matched_positive
    actual_by_requested = {
        float(item["requested_time"]): float(item["actual_time"]) for item in time_rows
    }

    labels = build_requested_labels(args.samples, int(config.misc.num_classes))
    test_mask = class_group_split(labels, args.test_fraction, args.seed + 17)
    if rank == 0:
        real_rows = select_matching_imagenet_rows(
            args.parquet_data_path, labels, args.seed + 31
        )
    else:
        real_rows = np.empty(args.samples, dtype=np.int64)
    row_tensor = torch.from_numpy(real_rows).to(device=device)
    dist.broadcast(row_tensor, src=0)
    real_rows = row_tensor.cpu().numpy().astype(np.int64, copy=True)

    local_ids = np.arange(rank, args.samples, world_size, dtype=np.int64)
    local_labels = labels[local_ids]
    local_test_mask = test_mask[local_ids]
    local_rows = real_rows[local_ids]
    generator = torch.Generator(device="cpu").manual_seed(
        int(args.seed) + 1_000_003 * rank
    )
    local_noise = torch.randn(
        (local_ids.size, *latent_size), generator=generator, dtype=torch.float32
    )
    example_ids = set(range(min(args.example_count, args.samples)))
    original_examples: dict[int, np.ndarray] = {}

    p_store = HeldoutStates()
    dataset = DeterministicImageNetPacked(
        args.packed_data_path,
        split="train",
        image_size=int(config.training.image_size),
        horizontal_flip=False,
    )
    rae = instantiate_from_config(config.stage_1)
    del rae.decoder
    rae = rae.to(device).eval()
    rae.requires_grad_(False)
    total_batches = math.ceil(local_ids.size / args.per_rank_batch)
    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(0, local_ids.size, args.per_rank_batch)
        ):
            end = min(start + args.per_rank_batch, local_ids.size)
            images_list = []
            for source_row, expected_label in zip(
                local_rows[start:end].tolist(), local_labels[start:end].tolist()
            ):
                image, actual_label, _ = dataset[int(source_row)]
                if int(actual_label) != int(expected_label):
                    raise RuntimeError("ImageNet source label mismatch")
                images_list.append(image)
            images = torch.stack(images_list).to(device=device)
            ids = local_ids[start:end]
            for offset, sample_id in enumerate(ids.tolist()):
                if int(sample_id) in example_ids:
                    original_examples[int(sample_id)] = (
                        images[offset].clamp(0, 1).mul(255).to(torch.uint8)
                        .permute(1, 2, 0).cpu().numpy()
                    )
            with autocast_context(args.precision):
                clean = rae.encode(images).float()
            noise = local_noise[start:end].to(device=device)
            for requested_time, actual_time in actual_by_requested.items():
                state = clean if requested_time == 0.0 else (
                    (1.0 - actual_time) * clean + actual_time * noise
                )
                p_store.add(requested_time, state, ids)
            if rank == 0 and (
                (batch_index + 1) % args.log_every_batches == 0
                or batch_index + 1 == total_batches
            ):
                print(f"[p_t] encoder batches {batch_index + 1}/{total_batches}", flush=True)
    del rae, dataset
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    del checkpoint
    transport = create_transport(config=config.transport, time_dist_shift=shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    official_scale = (
        float(args.ig_scale) if args.ig_scale is not None else float(config.guidance.ig.scale)
    )
    q_stores = {}
    for branch, scale in (("full", 1.0), ("ig", official_scale)):
        q_stores[branch] = _collect_sampler_states(
            branch=branch,
            ig_scale=scale,
            model=model,
            sample_fn=sample_fn,
            config=config,
            matched_positive_times=matched_positive,
            include_endpoint=include_endpoint,
            local_ids=local_ids,
            local_labels=local_labels,
            local_noise=local_noise,
            per_rank_batch=args.per_rank_batch,
            log_every_batches=args.log_every_batches,
            precision=args.precision,
            device=device,
        )
        dist.barrier()

    if 1.0 in requested_times:
        p_t1, p_ids = p_store.tensors(1.0)
        for branch in ("full", "ig"):
            q_t1, q_ids = q_stores[branch].tensors(1.0)
            if not np.array_equal(p_ids, q_ids) or not torch.equal(p_t1, q_t1):
                raise RuntimeError(f"t=1 hard control failed for {branch}")

    del model, sampler, transport
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    decoder_rae = instantiate_from_config(config.stage_1)
    del decoder_rae.encoder
    decoder_rae = decoder_rae.to(device).eval()
    decoder_rae.requires_grad_(False)
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", [args.inception_feature], verbose=False
    ).to(device)

    archive: dict[str, np.ndarray] = {
        "ids": local_ids,
        "labels": local_labels,
        "test_mask": local_test_mask,
        "example_ids": np.asarray(sorted(original_examples), dtype=np.int64),
        "example_original": np.stack(
            [original_examples[key] for key in sorted(original_examples)], axis=0
        ) if original_examples else np.empty((0, 256, 256, 3), dtype=np.uint8),
    }
    decode_diagnostics: dict[str, Any] = {}
    stores = {"p": p_store, **q_stores}
    for condition, store in stores.items():
        for requested_time in requested_times:
            states, ids = store.tensors(float(requested_time))
            if not np.array_equal(ids, local_ids):
                raise RuntimeError(f"state IDs changed for {condition} t={requested_time}")
            features, examples, diagnostics = _decode_features(
                decoder_rae,
                extractor,
                states,
                ids,
                decode_batch=args.decode_batch,
                precision=args.precision,
                example_ids=example_ids,
                device=device,
            )
            suffix = time_suffix(requested_time)
            archive[f"feat_{condition}_{suffix}"] = features
            local_example_ids = np.asarray(sorted(examples), dtype=np.int64)
            archive[f"example_ids_{condition}_{suffix}"] = local_example_ids
            archive[f"example_{condition}_{suffix}"] = (
                np.stack([examples[key] for key in local_example_ids.tolist()], axis=0)
                if local_example_ids.size
                else np.empty((0, 256, 256, 3), dtype=np.uint8)
            )
            decode_diagnostics[f"{condition}_{suffix}"] = diagnostics
            if rank == 0:
                print(f"[decode] {condition} {suffix} complete", flush=True)

    np.savez(output_dir / f"decoded_features_rank{rank:02d}.npz", **archive)
    (output_dir / f"decode_diagnostics_rank{rank:02d}.json").write_text(
        json.dumps(decode_diagnostics, indent=2), encoding="utf-8"
    )
    del decoder_rae, extractor
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        ids, ordered_labels, ordered_test_mask, features, _ = _load_feature_shards(
            output_dir, world_size
        )
        if not np.array_equal(ordered_labels, labels):
            raise RuntimeError("feature labels differ from sampling protocol")
        if not np.array_equal(ordered_test_mask, test_mask):
            raise RuntimeError("feature split differs from sampling protocol")
        train_mask = ~test_mask
        reference = None
        stats: dict[str, dict[str, np.ndarray]] = {}
        if not args.skip_fid:
            reference = load_reference_statistics(
                args.fid_reference.expanduser().resolve(), args.inception_feature
            )
            stats = {key: feature_statistics(value) for key, value in features.items()}
        rows = []
        score_archive: dict[str, np.ndarray] = {}
        for time_index, requested_time in enumerate(requested_times):
            suffix = time_suffix(requested_time)
            p = features[f"feat_p_{suffix}"]
            branch_scores = {}
            for branch_index, branch in enumerate(("full", "ig")):
                q = features[f"feat_{branch}_{suffix}"]
                weight, intercept, ridge = fit_feature_probe(
                    p, q, train_mask, args.ridge_ratio
                )
                p_scores = feature_probe_scores(p[test_mask], weight, intercept)
                q_scores = feature_probe_scores(q[test_mask], weight, intercept)
                auc = paired_auc(p_scores, q_scores)
                ci_low, ci_high = bootstrap_paired_auc(
                    p_scores,
                    q_scores,
                    args.bootstrap_repeats,
                    args.seed + 1000 * time_index + branch_index,
                )
                branch_scores[branch] = (p_scores, q_scores, auc)
                score_archive[f"p_{branch}_{suffix}"] = p_scores
                score_archive[f"q_{branch}_{suffix}"] = q_scores
                rows.append(
                    {
                        "branch": branch,
                        "requested_time": requested_time,
                        "actual_time": actual_by_requested[requested_time],
                        "auc": auc,
                        "auc_ci_low": ci_low,
                        "auc_ci_high": ci_high,
                        "ridge": ridge,
                        "heldout_pairs": int(test_mask.sum()),
                    }
                )
        auc_frame = pd.DataFrame(rows)
        auc_frame.to_csv(output_dir / "decoded_auc_results.csv", index=False)
        np.savez_compressed(output_dir / "decoded_heldout_scores.npz", **score_archive)

        summary_rows = []
        for time_index, requested_time in enumerate(requested_times):
            suffix = time_suffix(requested_time)
            p_key = f"feat_p_{suffix}"
            full_key = f"feat_full_{suffix}"
            ig_key = f"feat_ig_{suffix}"
            auc_rows = auc_frame[auc_frame["requested_time"] == requested_time]
            auc_full = float(auc_rows[auc_rows["branch"] == "full"]["auc"].iloc[0])
            auc_ig = float(auc_rows[auc_rows["branch"] == "ig"]["auc"].iloc[0])
            p_full = score_archive[f"p_full_{suffix}"]
            q_full = score_archive[f"q_full_{suffix}"]
            p_ig = score_archive[f"p_ig_{suffix}"]
            q_ig = score_archive[f"q_ig_{suffix}"]
            delta_low, delta_high = bootstrap_auc_delta(
                p_full,
                q_full,
                p_ig,
                q_ig,
                args.bootstrap_repeats,
                args.seed + 10_000 + time_index,
            )
            if reference is None:
                fid_real_p = fid_real_full = fid_real_ig = float("nan")
                fid_p_full = fid_p_ig = float("nan")
            else:
                fid_real_p = fid_between_statistics(stats[p_key], reference)
                fid_real_full = fid_between_statistics(stats[full_key], reference)
                fid_real_ig = fid_between_statistics(stats[ig_key], reference)
                fid_p_full = fid_between_statistics(stats[p_key], stats[full_key])
                fid_p_ig = fid_between_statistics(stats[p_key], stats[ig_key])
            summary_rows.append(
                {
                    "requested_time": requested_time,
                    "actual_time": actual_by_requested[requested_time],
                    "auc_full": auc_full,
                    "auc_ig": auc_ig,
                    "auc_delta_ig_minus_full": auc_ig - auc_full,
                    "auc_delta_ci_low": delta_low,
                    "auc_delta_ci_high": delta_high,
                    "fid_real_p": fid_real_p,
                    "fid_real_full": fid_real_full,
                    "fid_real_ig": fid_real_ig,
                    "fid_real_delta_ig_minus_full": fid_real_ig - fid_real_full,
                    "fid_p_full": fid_p_full,
                    "fid_p_ig": fid_p_ig,
                    "fid_p_delta_ig_minus_full": fid_p_ig - fid_p_full,
                }
            )
        summary = pd.DataFrame(summary_rows).sort_values("actual_time", ascending=False)
        summary.to_csv(output_dir / "decoded_distribution_summary.csv", index=False)
        _plot_metric_curves(summary, output_dir / "decoded_distribution_curves.png")
        _plot_decoded_examples(output_dir, world_size, requested_times)
        manifest = {
            "protocol": "raev2_decoded_distribution_audit_v1",
            "inference_only": True,
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "checkpoint_epoch": checkpoint_epoch,
            "state_key": args.state_key,
            "samples": args.samples,
            "train_pairs": int(train_mask.sum()),
            "heldout_pairs": int(test_mask.sum()),
            "split_unit": "ImageNet class",
            "seed": args.seed,
            "world_size": world_size,
            "requested_times": requested_times,
            "matched_times": time_rows,
            "ig_scale": official_scale,
            "inception_feature": args.inception_feature,
            "fid_skipped": bool(args.skip_fid),
            "fid_reference": str(args.fid_reference.expanduser().resolve()),
            "fid_scope": (
                "standard Inception statistics with a finite generated sample count; "
                "relative branch comparison, not a 50k official gFID claim"
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(summary.to_string(index=False))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
