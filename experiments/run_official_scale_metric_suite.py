#!/usr/bin/env python3
"""
Official RAEv2 scale sweep with paired, latent-distribution, decoder, and image metrics.

Fixed default scales
--------------------
    0.85, 0.92, 1.0, 1.2, 1.4, 1.5, 1.6, 1.78, 1.85

The program is an inference-only, resumable experiment for the official
DINOv3-L-K7 RAEv2 checkpoint. One command performs all stages:

1. Paired clean-latent error
   - Uses teacher-forced states z_t=(1-t)z_0+t*epsilon.
   - Evaluates every actual 100-step solver input inside the official IG interval.
   - Computes paired error for every fixed scale without separate forwards:
         z_hat_s = base + s*(full-base).
   - Outputs per-time and globally delta-t-weighted nMSE/nRMSE.

2. Clean reference distribution (5k)
   - Selects 5,000 real ImageNet images with the same class histogram as generation.
   - Encodes clean latents.
   - Runs the frozen decoder once and captures five internal decoder depths.
   - Saves compact distribution features; full 1024x16x16 latents are not saved.

3. Free endpoint distribution for each scale (5k)
   - Uses identical initial noises and labels for every scale.
   - Runs the official 100-step ODE sampler.
   - At the endpoint, in one pass:
       * projects the latent with fixed CountSketch maps;
       * accumulates full latent/channel moments;
       * decodes the latent;
       * captures decoder hidden features;
       * extracts Inception-2048 features.

4. Analysis
   Core scale-level outputs:
       * paired clean-latent nMSE/nRMSE;
       * projected latent FID and sliced Wasserstein distance;
       * latent global/channel variance and mean/covariance mismatch;
       * gFID to the full ADM ImageNet reference statistics;
       * FID to clean reconstructions;
       * KID to real and reconstruction features;
       * improved precision/recall to real and reconstruction features;
       * decoder within-image spatial variance ratio;
       * decoder between-image pooled-feature variance ratio;
       * decoder projected feature FID/SWD at five depths;
       * decoder clipping diagnostics.

Why compact projections?
------------------------
A raw RAEv2 latent has 1024*16*16=262,144 coordinates. A 5k-sample full
covariance is neither computationally feasible nor statistically well estimated.
The program therefore reports explicitly named *projected* latent FID/SWD using
four fixed CountSketch projections, while retaining interpretable full-space
global and per-channel moment diagnostics.

Main outputs
------------
    scale_summary.csv
    pair_error_by_time.csv
    decoder_feature_distribution.csv
    reference_baselines.json
    final_report.json
    final_report.txt
    plots/*.png

Default command
---------------
    cd /home/zhoushunyu/eqvae
    /data/users/zhoushunyu/eqvae/envs/raev2/bin/python \
      experiments/run_official_scale_metric_suite.py

Smoke test
----------
    .../python experiments/run_official_scale_metric_suite.py \
      --pair-samples 8 \
      --distribution-samples 1000 \
      --scales 0.92,1.0,1.4 \
      --output-root /data/users/zhoushunyu/eqvae/experiments/official_scale_suite_smoke

The output root is protocol-locked. Re-running the same command resumes;
changing scales, seeds, sample counts, checkpoint, or projection settings
requires a new output root.

Real Inception features
-----------------------
FID uses the full ADM reference mean/covariance from:
    /data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz

KID and precision/recall require individual real Inception-2048 vectors. The
default path is the torch-fidelity cache already created by earlier repository
evaluations:
    ~/.cache/torch/fidelity_cache/
      raev2_imagenet256_virtual_reference-inception-v3-compat-features-2048.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPO = Path("/home/zhoushunyu/eqvae")
DEFAULT_DATA = Path("/data/users/zhoushunyu/eqvae")
DEFAULT_SCALES = "0.85,0.92,1.0,1.2,1.4,1.5,1.6,1.78,1.85"
EPS = 1e-12


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one numeric scale is required")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("guidance scales must be nonnegative")
    return sorted(set(values))


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def scale_tag(scale: float) -> str:
    text = f"{float(scale):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def shell_join(command: Sequence[str]) -> str:
    import shlex
    return " ".join(shlex.quote(str(item)) for item in command)


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    command = [str(item) for item in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = shell_join(command)
    print(f"\n$ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {printable}\n")
        log.flush()
        if dry_run:
            return
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, command)


def autocast_context(precision: str):
    import torch
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def initialize_repo_imports(repo: Path) -> None:
    raev2_src = repo / "external" / "RAEv2" / "src"
    for path in (repo, raev2_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_config(config_path: Path, repo: Path):
    from omegaconf import OmegaConf

    initialize_repo_imports(repo)
    from configs.stage2 import Stage2Config
    from stage2.utils import validate_stage2_config

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(config_path))
    )
    config.post_process()
    validate_stage2_config(config)
    config.prepare_model_params()
    if config.transport.prediction != "x":
        raise ValueError("The official experiment requires clean-latent x-prediction.")
    return config


def shifted_solver_grid(num_steps: int, shift: float):
    import torch

    grid = torch.linspace(1.0, 0.0, int(num_steps) + 1, dtype=torch.float64)
    return shift * grid / (1.0 + (shift - 1.0) * grid)


def build_labels(sample_count: int, num_classes: int) -> np.ndarray:
    return np.arange(int(sample_count), dtype=np.int64) % int(num_classes)


def sorted_shard_paths(directory: Path, prefix: str) -> list[Path]:
    paths = sorted(directory.glob(f"{prefix}_rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_rank*.npz under {directory}")
    return paths


def extract_tensor_from_cache(payload: Any):
    import torch

    if isinstance(payload, torch.Tensor) and payload.ndim == 2:
        return payload
    if isinstance(payload, dict):
        candidates = [
            value for value in payload.values()
            if isinstance(value, torch.Tensor) and value.ndim == 2
        ]
        if len(candidates) == 1:
            return candidates[0]
    raise ValueError("Real Inception feature cache has an unexpected format.")


# ---------------------------------------------------------------------------
# Fixed latent projection and moments
# ---------------------------------------------------------------------------


class CountSketchProjector:
    """Fixed CountSketch maps shared by reference and every scale."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        repeats: int,
        seed: int,
        device,
    ) -> None:
        import torch

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.repeats = int(repeats)
        self.scale = math.sqrt(max(self.input_dim / self.output_dim, 1.0))
        buckets = []
        signs = []
        for repeat in range(self.repeats):
            rng = np.random.default_rng(int(seed) + 104729 * repeat)
            buckets.append(
                torch.from_numpy(
                    rng.integers(
                        0,
                        self.output_dim,
                        size=self.input_dim,
                        dtype=np.int64,
                    )
                ).to(device=device)
            )
            signs.append(
                torch.from_numpy(
                    rng.choice(
                        np.asarray([-1.0, 1.0], dtype=np.float32),
                        size=self.input_dim,
                    )
                ).to(device=device)
            )
        self.buckets = buckets
        self.signs = signs

    def __call__(self, latent):
        import torch

        flat = latent.float().flatten(1)
        if flat.shape[1] != self.input_dim:
            raise ValueError(
                f"CountSketch expected {self.input_dim} coordinates, got {flat.shape[1]}"
            )
        result = torch.zeros(
            flat.shape[0],
            self.repeats,
            self.output_dim,
            device=flat.device,
            dtype=torch.float32,
        )
        for repeat, (bucket, sign) in enumerate(zip(self.buckets, self.signs)):
            result[:, repeat].scatter_add_(
                1,
                bucket.unsqueeze(0).expand(flat.shape[0], -1),
                flat * sign.unsqueeze(0),
            )
        return result / self.scale


class LatentMomentAccumulator:
    def __init__(self, channels: int, device) -> None:
        import torch

        self.global_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.global_sumsq = torch.zeros((), device=device, dtype=torch.float64)
        self.global_count = torch.zeros((), device=device, dtype=torch.float64)
        self.channel_sum = torch.zeros(channels, device=device, dtype=torch.float64)
        self.channel_sumsq = torch.zeros(channels, device=device, dtype=torch.float64)
        self.channel_count = torch.zeros((), device=device, dtype=torch.float64)

    def update(self, latent) -> None:
        value = latent.double()
        self.global_sum += value.sum()
        self.global_sumsq += value.square().sum()
        self.global_count += value.numel()
        self.channel_sum += value.sum(dim=(0, 2, 3))
        self.channel_sumsq += value.square().sum(dim=(0, 2, 3))
        self.channel_count += value.shape[0] * value.shape[2] * value.shape[3]

    def save(self, path: Path) -> None:
        np.savez(
            path,
            global_sum=self.global_sum.cpu().numpy(),
            global_sumsq=self.global_sumsq.cpu().numpy(),
            global_count=self.global_count.cpu().numpy(),
            channel_sum=self.channel_sum.cpu().numpy(),
            channel_sumsq=self.channel_sumsq.cpu().numpy(),
            channel_count=self.channel_count.cpu().numpy(),
        )


# ---------------------------------------------------------------------------
# Decoder feature capture in the actual decode pass
# ---------------------------------------------------------------------------


def first_tensor(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Decoder block hook did not receive a tensor output.")


class DecoderFeatureCapture:
    """Capture selected decoder block outputs while rae.decode runs once."""

    def __init__(self, rae, hidden_indices: Sequence[int]) -> None:
        self.rae = rae
        self.hidden_indices = tuple(int(value) for value in hidden_indices)
        self.pending: dict[int, Any] = {}
        self.handles = []
        layers = rae.decoder.decoder_layers
        for hidden_index in self.hidden_indices:
            layer_position = hidden_index - 1
            if layer_position < 0 or layer_position >= len(layers):
                raise ValueError(
                    f"Invalid hidden index {hidden_index} for decoder depth {len(layers)}"
                )

            def hook(_module, _inputs, output, key=hidden_index):
                if key in self.pending:
                    raise RuntimeError(f"Decoder layer {key} fired more than once.")
                self.pending[key] = first_tensor(output)

            self.handles.append(layers[layer_position].register_forward_hook(hook))

    def clear(self) -> None:
        self.pending.clear()

    def consume(self, expected_batch: int):
        if set(self.pending) != set(self.hidden_indices):
            raise RuntimeError(
                f"Decoder hooks incomplete: got {sorted(self.pending)}, "
                f"expected {sorted(self.hidden_indices)}"
            )
        pooled = {}
        within = {}
        for hidden_index in self.hidden_indices:
            tokens = self.pending[hidden_index]
            if tokens.ndim != 3 or tokens.shape[0] != expected_batch:
                raise ValueError(
                    f"Unexpected decoder tokens at layer {hidden_index}: {tuple(tokens.shape)}"
                )
            token_count = tokens.shape[1]
            side_without_cls = math.isqrt(max(token_count - 1, 0))
            side_with_cls = math.isqrt(token_count)
            if side_without_cls * side_without_cls == token_count - 1:
                patches = tokens[:, 1:, :]
            elif side_with_cls * side_with_cls == token_count:
                patches = tokens
            else:
                raise ValueError(
                    f"Cannot identify patch tokens at layer {hidden_index}: N={token_count}"
                )
            patches = patches.float()
            pooled[hidden_index] = patches.mean(dim=1)
            within[hidden_index] = patches.var(dim=1, unbiased=False).mean(dim=1)
        self.pending.clear()
        return pooled, within

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def decode_and_extract(
    *,
    rae,
    extractor,
    capture: DecoderFeatureCapture,
    latent,
    precision: str,
):
    import torch

    capture.clear()
    with autocast_context(precision):
        decoded = rae.decode(latent).float()
    pooled, within = capture.consume(latent.shape[0])
    if not torch.isfinite(decoded).all():
        raise FloatingPointError("Decoder produced non-finite pixels.")
    raw_min = float(decoded.min().item())
    raw_max = float(decoded.max().item())
    low = float((decoded < 0).float().mean().item())
    high = float((decoded > 1).float().mean().item())
    images = decoded.clamp(0, 1).mul(255).to(torch.uint8)
    with torch.autocast(device_type="cuda", enabled=False):
        inception = extractor(images)[0].float()
    return inception, pooled, within, {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "clipped_low_fraction": low,
        "clipped_high_fraction": high,
    }


# ---------------------------------------------------------------------------
# Worker output helpers
# ---------------------------------------------------------------------------


def concatenate_chunks(chunks: list[np.ndarray], dtype=None) -> np.ndarray:
    if not chunks:
        raise ValueError("Cannot concatenate an empty chunk list.")
    value = np.concatenate(chunks, axis=0)
    return value.astype(dtype, copy=False) if dtype is not None else value


def save_feature_shard(
    path: Path,
    *,
    ids: np.ndarray,
    labels: np.ndarray,
    latent_projection: np.ndarray,
    inception: np.ndarray,
    decoder_pooled: dict[int, np.ndarray],
    decoder_within: dict[int, np.ndarray],
    example_ids: np.ndarray,
    example_images: np.ndarray,
) -> None:
    payload: dict[str, np.ndarray] = {
        "ids": ids.astype(np.int64, copy=False),
        "labels": labels.astype(np.int64, copy=False),
        "latent_projection": latent_projection.astype(np.float32, copy=False),
        "inception": inception.astype(np.float32, copy=False),
        "example_ids": example_ids.astype(np.int64, copy=False),
        "example_images": example_images.astype(np.uint8, copy=False),
    }
    for hidden_index, value in decoder_pooled.items():
        payload[f"decoder_pooled_l{hidden_index}"] = value.astype(np.float16, copy=False)
    for hidden_index, value in decoder_within.items():
        payload[f"decoder_within_l{hidden_index}"] = value.astype(np.float32, copy=False)
    np.savez(path, **payload)


def merge_shards(directory: Path) -> dict[str, np.ndarray]:
    paths = sorted_shard_paths(directory, "features")
    archives = [np.load(path) for path in paths]
    try:
        ids = np.concatenate([archive["ids"] for archive in archives])
        order = np.argsort(ids)
        expected = np.arange(ids.size, dtype=np.int64)
        if not np.array_equal(ids[order], expected):
            raise RuntimeError(
                f"Feature shards under {directory} do not contain IDs 0..N-1 exactly once."
            )
        common_keys = set(archives[0].files)
        for archive in archives[1:]:
            common_keys &= set(archive.files)
        result: dict[str, np.ndarray] = {}
        for key in sorted(common_keys):
            if key.startswith("example_"):
                continue
            result[key] = np.concatenate([archive[key] for archive in archives], axis=0)[
                order
            ]
        examples: dict[int, np.ndarray] = {}
        for archive in archives:
            for sample_id, image in zip(
                archive["example_ids"].tolist(), archive["example_images"]
            ):
                examples[int(sample_id)] = image
        result["example_ids"] = np.asarray(sorted(examples), dtype=np.int64)
        result["example_images"] = (
            np.stack([examples[key] for key in sorted(examples)], axis=0)
            if examples
            else np.empty((0, 256, 256, 3), dtype=np.uint8)
        )
        return result
    finally:
        for archive in archives:
            archive.close()


def aggregate_moments(directory: Path) -> dict[str, np.ndarray | float]:
    paths = sorted(directory.glob("moments_rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"No moments_rank*.npz under {directory}")
    totals: dict[str, np.ndarray] = {}
    for path in paths:
        with np.load(path) as payload:
            for key in payload.files:
                value = np.asarray(payload[key], dtype=np.float64)
                totals[key] = totals.get(key, np.zeros_like(value)) + value
    return totals


# ---------------------------------------------------------------------------
# Pair worker
# ---------------------------------------------------------------------------


def pair_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset

    initialize_repo_imports(args.repo.resolve())
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())

    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.raev2_training_core import (
        DeterministicImageNetPacked,
        split_internal_guidance_output,
        validate_full_stage2_checkpoint,
    )
    from utils.model_utils import instantiate_from_config

    install_raev2_decoder_config_compat()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config.resolve(), args.repo.resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    grid = shifted_solver_grid(int(config.sampler.num_steps), shift)
    input_times = grid[:-1]
    delta_t = grid[:-1] - grid[1:]
    active = (
        (input_times >= float(config.guidance.ig.t_min))
        & (input_times <= float(config.guidance.ig.t_max))
    )
    times = input_times[active].to(torch.float32)
    widths = delta_t[active].to(torch.float64)

    dataset = DeterministicImageNetPacked(
        args.packed_data_path.resolve(),
        split="train",
        image_size=int(config.training.image_size),
        augmentation_seed=int(args.pair_seed),
        horizontal_flip=False,
        index_map_path=args.index_map.resolve(),
    )
    local_ids = np.arange(rank, args.pair_samples, world_size, dtype=np.int64)
    loader = DataLoader(
        Subset(dataset, local_ids.tolist()),
        batch_size=int(args.pair_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
        multiprocessing_context="spawn" if int(args.num_workers) > 0 else None,
    )

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    del rae.decoder

    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint

    n_times = int(times.numel())
    a_sum = torch.zeros(n_times, device=device, dtype=torch.float64)
    b_sum = torch.zeros(n_times, device=device, dtype=torch.float64)
    c_sum = torch.zeros(n_times, device=device, dtype=torch.float64)
    z_sum = torch.zeros(n_times, device=device, dtype=torch.float64)
    count = torch.zeros(n_times, device=device, dtype=torch.float64)

    with torch.inference_mode():
        for batch_index, (images, labels, data_ids) in enumerate(loader):
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            data_id_list = [int(value) for value in data_ids.tolist()]
            with autocast_context(args.precision):
                clean = rae.encode(images).float()

            noises = []
            for data_id in data_id_list:
                generator = torch.Generator(device="cpu").manual_seed(
                    int(args.pair_seed) + 1_000_003 * data_id
                )
                noises.append(
                    torch.randn(latent_size, generator=generator, dtype=torch.float32)
                )
            noise = torch.stack(noises).to(device=device)

            batch_size = clean.shape[0]
            for start in range(0, n_times, int(args.pair_time_chunk)):
                end = min(start + int(args.pair_time_chunk), n_times)
                current_times = times[start:end].to(device=device)
                chunk = current_times.numel()
                clean_rep = (
                    clean[:, None]
                    .expand(batch_size, chunk, *clean.shape[1:])
                    .reshape(batch_size * chunk, *clean.shape[1:])
                )
                noise_rep = (
                    noise[:, None]
                    .expand(batch_size, chunk, *noise.shape[1:])
                    .reshape(batch_size * chunk, *noise.shape[1:])
                )
                time_rep = (
                    current_times[None, :]
                    .expand(batch_size, chunk)
                    .reshape(batch_size * chunk)
                )
                label_rep = (
                    labels[:, None].expand(batch_size, chunk).reshape(batch_size * chunk)
                )
                t_view = time_rep.reshape(
                    time_rep.shape[0], *([1] * (clean.ndim - 1))
                )
                noisy = (1.0 - t_view) * clean_rep + t_view * noise_rep
                with autocast_context(args.precision):
                    output = model(
                        noisy,
                        time_rep,
                        context=label_rep,
                        attn_mask=None,
                    )
                full, base = split_internal_guidance_output(output)
                if base is None:
                    raise RuntimeError("The official IG model did not return Full/Base.")
                in_channels = int(model.in_channels)
                full = full[:, :in_channels].float()
                base = base[:, :in_channels].float()
                direction = (full - base).double().flatten(1)
                residual = (base - clean_rep).double().flatten(1)
                clean_flat = clean_rep.double().flatten(1)

                sl = slice(start, end)
                a_sum[sl] += direction.square().sum(1).reshape(batch_size, chunk).sum(0)
                b_sum[sl] += (
                    residual * direction
                ).sum(1).reshape(batch_size, chunk).sum(0)
                c_sum[sl] += residual.square().sum(1).reshape(batch_size, chunk).sum(0)
                z_sum[sl] += clean_flat.square().sum(1).reshape(batch_size, chunk).sum(0)
                count[sl] += clean_rep[0].numel() * batch_size

            if rank == 0 and (
                (batch_index + 1) % args.log_every_batches == 0
                or (batch_index + 1) * args.pair_batch_size >= len(local_ids)
            ):
                print(f"[pair] rank0 batch {batch_index + 1}", flush=True)

    for tensor in (a_sum, b_sum, c_sum, z_sum, count):
        dist.all_reduce(tensor)

    if rank == 0:
        output = args.output_root.resolve() / "pair"
        output.mkdir(parents=True, exist_ok=True)
        a = a_sum.cpu().numpy()
        b = b_sum.cpu().numpy()
        c = c_sum.cpu().numpy()
        z = z_sum.cpu().numpy()
        times_np = times.double().cpu().numpy()
        widths_np = widths.cpu().numpy()

        rows = []
        for scale in args.scales:
            for index, time_value in enumerate(times_np):
                error = a[index] * scale * scale + 2.0 * b[index] * scale + c[index]
                rows.append(
                    {
                        "scale": float(scale),
                        "solver_index_in_ig_interval": index,
                        "time": float(time_value),
                        "delta_t": float(widths_np[index]),
                        "paired_nmse": float(error / max(z[index], EPS)),
                        "paired_nrmse": float(
                            math.sqrt(max(error, 0.0) / max(z[index], EPS))
                        ),
                    }
                )
        by_time = pd.DataFrame(rows)
        by_time.to_csv(output / "pair_error_by_time.csv", index=False)

        summary_rows = []
        for scale in args.scales:
            errors = a * scale * scale + 2.0 * b * scale + c
            summary_rows.append(
                {
                    "scale": float(scale),
                    "paired_nmse_equal_call": float(errors.sum() / max(z.sum(), EPS)),
                    "paired_nrmse_equal_call": float(
                        math.sqrt(max(errors.sum(), 0.0) / max(z.sum(), EPS))
                    ),
                    "paired_nmse_delta_t": float(
                        np.sum(widths_np * errors) / max(np.sum(widths_np * z), EPS)
                    ),
                    "paired_nrmse_delta_t": float(
                        math.sqrt(
                            max(np.sum(widths_np * errors), 0.0)
                            / max(np.sum(widths_np * z), EPS)
                        )
                    ),
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output / "pair_scale_summary.csv", index=False)
        manifest = {
            "protocol": "official_scale_suite_pair_v1",
            "samples": int(args.pair_samples),
            "seed": int(args.pair_seed),
            "scales": args.scales,
            "solver_steps": int(config.sampler.num_steps),
            "ig_interval": [
                float(config.guidance.ig.t_min),
                float(config.guidance.ig.t_max),
            ],
            "active_times": n_times,
        }
        (output / "complete.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(summary.to_string(index=False), flush=True)

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Reference worker
# ---------------------------------------------------------------------------


class SelectedImageDataset:
    def __init__(self, base, rows: np.ndarray, labels: np.ndarray, ids: np.ndarray):
        self.base = base
        self.rows = rows
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image, actual_label, _ = self.base[int(self.rows[index])]
        expected = int(self.labels[index])
        if int(actual_label) != expected:
            raise RuntimeError(
                f"Selected ImageNet row has label {actual_label}, expected {expected}"
            )
        return image, expected, int(self.ids[index])


def reference_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    initialize_repo_imports(args.repo.resolve())
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())

    from experiments.rae_strict_lpl import decoder_hidden_indices
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.raev2_training_core import DeterministicImageNetPacked
    from experiments.run_raev2_distribution_auc import select_matching_imagenet_rows
    from utils.model_utils import instantiate_from_config

    install_raev2_decoder_config_compat()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config.resolve(), args.repo.resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    labels = build_labels(args.distribution_samples, int(config.misc.num_classes))

    if rank == 0:
        selected_rows = select_matching_imagenet_rows(
            args.parquet_data_path.resolve(),
            labels,
            int(args.reference_seed),
        )
    else:
        selected_rows = np.empty(args.distribution_samples, dtype=np.int64)
    row_tensor = torch.from_numpy(selected_rows).to(device=device)
    dist.broadcast(row_tensor, src=0)
    selected_rows = row_tensor.cpu().numpy().astype(np.int64, copy=True)

    local_ids = np.arange(rank, args.distribution_samples, world_size, dtype=np.int64)
    local_labels = labels[local_ids]
    local_rows = selected_rows[local_ids]

    base = DeterministicImageNetPacked(
        args.packed_data_path.resolve(),
        split="train",
        image_size=int(config.training.image_size),
        horizontal_flip=False,
    )
    loader = DataLoader(
        SelectedImageDataset(base, local_rows, local_labels, local_ids),
        batch_size=int(args.distribution_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
        multiprocessing_context="spawn" if int(args.num_workers) > 0 else None,
    )

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    hidden_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        fractions=args.decoder_layer_fractions,
    )
    capture = DecoderFeatureCapture(rae, hidden_indices)
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).to(device).eval()
    extractor.requires_grad_(False)

    projector = CountSketchProjector(
        input_dim=math.prod(latent_size),
        output_dim=args.latent_projection_dim,
        repeats=args.latent_projection_repeats,
        seed=args.latent_projection_seed,
        device=device,
    )
    moments = LatentMomentAccumulator(latent_size[0], device)

    ids_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    latent_chunks: list[np.ndarray] = []
    inception_chunks: list[np.ndarray] = []
    pooled_chunks: dict[int, list[np.ndarray]] = defaultdict(list)
    within_chunks: dict[int, list[np.ndarray]] = defaultdict(list)
    example_ids: list[int] = []
    example_images: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_index, (images, batch_labels, batch_ids) in enumerate(loader):
            images = images.to(device=device, non_blocking=True)
            with autocast_context(args.precision):
                clean = rae.encode(images).float()
            moments.update(clean)
            latent_projection = projector(clean)
            inception, pooled, within, _ = decode_and_extract(
                rae=rae,
                extractor=extractor,
                capture=capture,
                latent=clean,
                precision=args.precision,
            )

            ids_np = batch_ids.numpy().astype(np.int64, copy=False)
            ids_chunks.append(ids_np)
            label_chunks.append(batch_labels.numpy().astype(np.int64, copy=False))
            latent_chunks.append(latent_projection.cpu().numpy())
            inception_chunks.append(inception.cpu().numpy())
            for hidden_index in hidden_indices:
                pooled_chunks[hidden_index].append(
                    pooled[hidden_index].cpu().numpy().astype(np.float16)
                )
                within_chunks[hidden_index].append(
                    within[hidden_index].cpu().numpy()
                )

            decoded_examples = min(
                max(args.example_count - len(example_ids), 0), len(ids_np)
            )
            if decoded_examples:
                with autocast_context(args.precision):
                    decoded = rae.decode(clean[:decoded_examples]).float()
                pixels = (
                    decoded.clamp(0, 1)
                    .mul(255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                example_ids.extend(ids_np[:decoded_examples].tolist())
                example_images.extend(list(pixels))

            if rank == 0 and (
                (batch_index + 1) % args.log_every_batches == 0
                or (batch_index + 1) * args.distribution_batch_size >= len(local_ids)
            ):
                print(f"[reference] rank0 batch {batch_index + 1}", flush=True)

    output = args.output_root.resolve() / "reference"
    output.mkdir(parents=True, exist_ok=True)
    save_feature_shard(
        output / f"features_rank{rank:02d}.npz",
        ids=concatenate_chunks(ids_chunks),
        labels=concatenate_chunks(label_chunks),
        latent_projection=concatenate_chunks(latent_chunks),
        inception=concatenate_chunks(inception_chunks),
        decoder_pooled={
            key: concatenate_chunks(value) for key, value in pooled_chunks.items()
        },
        decoder_within={
            key: concatenate_chunks(value) for key, value in within_chunks.items()
        },
        example_ids=np.asarray(example_ids, dtype=np.int64),
        example_images=(
            np.stack(example_images)
            if example_images
            else np.empty((0, 256, 256, 3), dtype=np.uint8)
        ),
    )
    moments.save(output / f"moments_rank{rank:02d}.npz")
    capture.close()
    dist.barrier()

    if rank == 0:
        manifest = {
            "protocol": "official_scale_suite_reference_v1",
            "samples": int(args.distribution_samples),
            "seed": int(args.reference_seed),
            "world_size": world_size,
            "hidden_indices": list(hidden_indices),
            "latent_projection_dim": int(args.latent_projection_dim),
            "latent_projection_repeats": int(args.latent_projection_repeats),
            "latent_projection_seed": int(args.latent_projection_seed),
            "matching_class_histogram": True,
        }
        (output / "complete.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Scale worker
# ---------------------------------------------------------------------------


def scale_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from functools import partial
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    initialize_repo_imports(args.repo.resolve())
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())

    from experiments.rae_strict_lpl import decoder_hidden_indices
    from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
    from experiments.raev2_training_core import validate_full_stage2_checkpoint
    from stage2.transport import create_sampler, create_transport
    from utils.guidance_utils import forward_with_internalguidance
    from utils.model_utils import instantiate_from_config

    install_raev2_decoder_config_compat()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config.resolve(), args.repo.resolve())
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    local_ids = np.arange(rank, args.distribution_samples, world_size, dtype=np.int64)
    local_labels = local_ids % int(config.misc.num_classes)

    rae = instantiate_from_config(config.stage_1).to(device).eval()
    rae.requires_grad_(False)
    del rae.encoder
    hidden_indices = decoder_hidden_indices(
        len(rae.decoder.decoder_layers),
        fractions=args.decoder_layer_fractions,
    )
    capture = DecoderFeatureCapture(rae, hidden_indices)
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).to(device).eval()
    extractor.requires_grad_(False)

    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint

    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    model_fn = partial(forward_with_internalguidance, model)
    interval = (
        float(config.guidance.ig.t_min),
        float(config.guidance.ig.t_max),
    )

    projector = CountSketchProjector(
        input_dim=math.prod(latent_size),
        output_dim=args.latent_projection_dim,
        repeats=args.latent_projection_repeats,
        seed=args.latent_projection_seed,
        device=device,
    )
    moments = LatentMomentAccumulator(latent_size[0], device)

    ids_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    latent_chunks: list[np.ndarray] = []
    inception_chunks: list[np.ndarray] = []
    pooled_chunks: dict[int, list[np.ndarray]] = defaultdict(list)
    within_chunks: dict[int, list[np.ndarray]] = defaultdict(list)
    example_ids: list[int] = []
    example_images: list[np.ndarray] = []
    diagnostic_sums = {
        "samples": 0.0,
        "raw_min": math.inf,
        "raw_max": -math.inf,
        "clipped_low_weighted": 0.0,
        "clipped_high_weighted": 0.0,
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.sampling_seed) * world_size + rank)

    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(0, len(local_ids), int(args.distribution_batch_size))
        ):
            end = min(start + int(args.distribution_batch_size), len(local_ids))
            ids = local_ids[start:end]
            labels_np = local_labels[start:end]
            noise = torch.randn(
                len(ids),
                *latent_size,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
            null = torch.full(
                (len(ids),),
                int(config.misc.num_classes),
                device=device,
                dtype=torch.long,
            )
            with autocast_context(args.precision):
                trajectory = sample_fn(
                    torch.cat((noise, noise), dim=0),
                    model_fn,
                    context=torch.cat((labels, null), dim=0),
                    attn_mask=None,
                    ig_scale=float(args.worker_scale),
                    ig_interval=interval,
                )
                endpoint = trajectory[-1]
                if endpoint.shape[0] == 2 * len(ids):
                    endpoint = endpoint.chunk(2, dim=0)[0]
                endpoint = endpoint.float()
                del trajectory

            moments.update(endpoint)
            latent_projection = projector(endpoint)
            inception, pooled, within, diagnostics = decode_and_extract(
                rae=rae,
                extractor=extractor,
                capture=capture,
                latent=endpoint,
                precision=args.precision,
            )

            ids_chunks.append(ids)
            label_chunks.append(labels_np)
            latent_chunks.append(latent_projection.cpu().numpy())
            inception_chunks.append(inception.cpu().numpy())
            for hidden_index in hidden_indices:
                pooled_chunks[hidden_index].append(
                    pooled[hidden_index].cpu().numpy().astype(np.float16)
                )
                within_chunks[hidden_index].append(
                    within[hidden_index].cpu().numpy()
                )

            diagnostic_sums["samples"] += len(ids)
            diagnostic_sums["raw_min"] = min(
                diagnostic_sums["raw_min"], diagnostics["raw_min"]
            )
            diagnostic_sums["raw_max"] = max(
                diagnostic_sums["raw_max"], diagnostics["raw_max"]
            )
            diagnostic_sums["clipped_low_weighted"] += (
                diagnostics["clipped_low_fraction"] * len(ids)
            )
            diagnostic_sums["clipped_high_weighted"] += (
                diagnostics["clipped_high_fraction"] * len(ids)
            )

            needed = min(max(args.example_count - len(example_ids), 0), len(ids))
            if needed:
                with autocast_context(args.precision):
                    decoded = rae.decode(endpoint[:needed]).float()
                pixels = (
                    decoded.clamp(0, 1)
                    .mul(255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                example_ids.extend(ids[:needed].tolist())
                example_images.extend(list(pixels))

            if rank == 0 and (
                (batch_index + 1) % args.log_every_batches == 0
                or end == len(local_ids)
            ):
                print(
                    f"[scale={args.worker_scale:g}] rank0 "
                    f"batch {batch_index + 1}, local {end}/{len(local_ids)}",
                    flush=True,
                )

    output = (
        args.output_root.resolve()
        / "scales"
        / f"ig_{scale_tag(args.worker_scale)}"
    )
    output.mkdir(parents=True, exist_ok=True)
    save_feature_shard(
        output / f"features_rank{rank:02d}.npz",
        ids=concatenate_chunks(ids_chunks),
        labels=concatenate_chunks(label_chunks),
        latent_projection=concatenate_chunks(latent_chunks),
        inception=concatenate_chunks(inception_chunks),
        decoder_pooled={
            key: concatenate_chunks(value) for key, value in pooled_chunks.items()
        },
        decoder_within={
            key: concatenate_chunks(value) for key, value in within_chunks.items()
        },
        example_ids=np.asarray(example_ids, dtype=np.int64),
        example_images=(
            np.stack(example_images)
            if example_images
            else np.empty((0, 256, 256, 3), dtype=np.uint8)
        ),
    )
    moments.save(output / f"moments_rank{rank:02d}.npz")
    (output / f"decode_diagnostics_rank{rank:02d}.json").write_text(
        json.dumps(diagnostic_sums, indent=2),
        encoding="utf-8",
    )
    capture.close()
    dist.barrier()

    if rank == 0:
        manifest = {
            "protocol": "official_scale_suite_free_v1",
            "scale": float(args.worker_scale),
            "samples": int(args.distribution_samples),
            "sampling_seed": int(args.sampling_seed),
            "world_size": world_size,
            "state_key": args.state_key,
            "sampler_steps": int(config.sampler.num_steps),
            "hidden_indices": list(hidden_indices),
            "latent_projection_dim": int(args.latent_projection_dim),
            "latent_projection_repeats": int(args.latent_projection_repeats),
            "latent_projection_seed": int(args.latent_projection_seed),
            "same_noise_and_labels_across_scales": True,
        }
        (output / "complete.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------


def covariance_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(features, dtype=np.float64)
    return value.mean(axis=0), np.cov(value, rowvar=False)


def frechet_distance(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> float:
    from scipy import linalg

    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    cov_a = np.asarray(cov_a, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)
    diff = mean_a - mean_b
    product = cov_a @ cov_b
    covmean = linalg.sqrtm(product)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_a.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_a + offset) @ (cov_b + offset))
    if np.iscomplexobj(covmean):
        maximum_imaginary = float(np.max(np.abs(covmean.imag)))
        if maximum_imaginary > 1e-3:
            raise ValueError(
                f"FID covariance square root has imaginary component {maximum_imaginary}"
            )
        covmean = covmean.real
    value = (
        diff @ diff
        + np.trace(cov_a)
        + np.trace(cov_b)
        - 2.0 * np.trace(covmean)
    )
    return float(max(value, 0.0))


def fid_from_features(first: np.ndarray, second: np.ndarray) -> float:
    mean_a, cov_a = covariance_statistics(first)
    mean_b, cov_b = covariance_statistics(second)
    return frechet_distance(mean_a, cov_a, mean_b, cov_b)


def sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    *,
    directions: int,
    seed: int,
) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    count = min(len(first), len(second))
    first = first[:count]
    second = second[:count]
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((first.shape[1], int(directions)))
    vectors /= np.linalg.norm(vectors, axis=0, keepdims=True) + EPS
    projected_a = np.sort(first @ vectors, axis=0)
    projected_b = np.sort(second @ vectors, axis=0)
    return float(np.sqrt(np.mean((projected_a - projected_b) ** 2)))


def standardize_from_reference(
    reference: np.ndarray,
    generated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    mean = reference.mean(axis=0, keepdims=True)
    std = reference.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (reference - mean) / std, (generated - mean) / std


def repeated_projected_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    swd_directions: int,
    seed: int,
) -> dict[str, float]:
    if reference.ndim != 3 or generated.ndim != 3:
        raise ValueError("Repeated projections must have shape [N,R,D].")
    if reference.shape[1:] != generated.shape[1:]:
        raise ValueError("Reference/generated projection shapes differ.")
    fid_values = []
    swd_values = []
    mean_values = []
    covariance_values = []
    for repeat in range(reference.shape[1]):
        ref, gen = standardize_from_reference(
            reference[:, repeat],
            generated[:, repeat],
        )
        fid_values.append(fid_from_features(ref, gen))
        swd_values.append(
            sliced_wasserstein(
                ref,
                gen,
                directions=swd_directions,
                seed=seed + 1009 * repeat,
            )
        )
        mean_values.append(
            float(np.linalg.norm(gen.mean(0) - ref.mean(0)) / math.sqrt(ref.shape[1]))
        )
        covariance_values.append(
            float(
                np.linalg.norm(np.cov(gen, rowvar=False) - np.cov(ref, rowvar=False), ord="fro")
                / (np.linalg.norm(np.cov(ref, rowvar=False), ord="fro") + EPS)
            )
        )
    return {
        "projected_fid_mean": float(np.mean(fid_values)),
        "projected_fid_std": float(np.std(fid_values, ddof=1)) if len(fid_values) > 1 else 0.0,
        "projected_swd_mean": float(np.mean(swd_values)),
        "projected_swd_std": float(np.std(swd_values, ddof=1)) if len(swd_values) > 1 else 0.0,
        "projected_mean_shift_mean": float(np.mean(mean_values)),
        "projected_covariance_shift_mean": float(np.mean(covariance_values)),
    }


def moment_metrics(
    reference: dict[str, np.ndarray | float],
    generated: dict[str, np.ndarray | float],
) -> dict[str, float]:
    ref_count = float(reference["global_count"])
    gen_count = float(generated["global_count"])
    ref_mean = float(reference["global_sum"]) / ref_count
    gen_mean = float(generated["global_sum"]) / gen_count
    ref_var = float(reference["global_sumsq"]) / ref_count - ref_mean * ref_mean
    gen_var = float(generated["global_sumsq"]) / gen_count - gen_mean * gen_mean

    ref_channel_count = float(reference["channel_count"])
    gen_channel_count = float(generated["channel_count"])
    ref_channel_mean = np.asarray(reference["channel_sum"]) / ref_channel_count
    gen_channel_mean = np.asarray(generated["channel_sum"]) / gen_channel_count
    ref_channel_var = (
        np.asarray(reference["channel_sumsq"]) / ref_channel_count
        - ref_channel_mean**2
    )
    gen_channel_var = (
        np.asarray(generated["channel_sumsq"]) / gen_channel_count
        - gen_channel_mean**2
    )
    ratios = (gen_channel_var + EPS) / (ref_channel_var + EPS)
    return {
        "clean_latent_global_mean": float(ref_mean),
        "generated_latent_global_mean": float(gen_mean),
        "clean_latent_global_variance": float(ref_var),
        "generated_latent_global_variance": float(gen_var),
        "latent_global_mean_shift_normalized": float(
            abs(gen_mean - ref_mean) / math.sqrt(max(ref_var, EPS))
        ),
        "latent_global_variance_ratio": float(gen_var / max(ref_var, EPS)),
        "latent_channel_variance_ratio_median": float(np.median(ratios)),
        "latent_channel_variance_ratio_p10": float(np.quantile(ratios, 0.10)),
        "latent_channel_variance_ratio_p90": float(np.quantile(ratios, 0.90)),
        "latent_channel_log_variance_ratio_rms": float(
            np.sqrt(np.mean(np.log(np.maximum(ratios, EPS)) ** 2))
        ),
    }


def random_project_features(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    output_dim: int,
    repeats: int,
    seed: int,
    swd_directions: int,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    clean_mean = reference.mean(axis=0, keepdims=True)
    clean_std = reference.std(axis=0, keepdims=True)
    clean_std = np.where(clean_std < 1e-6, 1.0, clean_std)
    ref_standard = (reference - clean_mean) / clean_std
    gen_standard = (generated - clean_mean) / clean_std

    fid_values = []
    swd_values = []
    for repeat in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + 15485863 * repeat)
        matrix = rng.standard_normal((reference.shape[1], int(output_dim)))
        matrix /= math.sqrt(int(output_dim))
        ref_projected = ref_standard @ matrix
        gen_projected = gen_standard @ matrix
        fid_values.append(fid_from_features(ref_projected, gen_projected))
        swd_values.append(
            sliced_wasserstein(
                ref_projected,
                gen_projected,
                directions=swd_directions,
                seed=seed + 1009 * repeat,
            )
        )
    return {
        "projected_fid_mean": float(np.mean(fid_values)),
        "projected_fid_std": float(np.std(fid_values, ddof=1)) if len(fid_values) > 1 else 0.0,
        "projected_swd_mean": float(np.mean(swd_values)),
        "projected_swd_std": float(np.std(swd_values, ddof=1)) if len(swd_values) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# KID and improved precision/recall
# ---------------------------------------------------------------------------


def polynomial_mmd_unbiased(first, second):
    import torch

    dimension = first.shape[1]
    kernel_xx = (first @ first.T / dimension + 1.0).pow(3)
    kernel_yy = (second @ second.T / dimension + 1.0).pow(3)
    kernel_xy = (first @ second.T / dimension + 1.0).pow(3)
    m = first.shape[0]
    n = second.shape[0]
    xx = (kernel_xx.sum() - kernel_xx.diag().sum()) / (m * (m - 1))
    yy = (kernel_yy.sum() - kernel_yy.diag().sum()) / (n * (n - 1))
    xy = kernel_xy.mean()
    return xx + yy - 2.0 * xy


def kid_from_features(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    subsets: int,
    subset_size: int,
    seed: int,
    device,
) -> tuple[float, float]:
    import torch

    generated_tensor = torch.from_numpy(
        generated.astype(np.float32, copy=False)
    ).to(device=device)
    reference_tensor = torch.from_numpy(
        reference.astype(np.float32, copy=False)
    ).to(device=device)
    size = min(int(subset_size), len(generated), len(reference))
    if size < 2:
        raise ValueError("KID requires at least two samples.")
    rng = np.random.default_rng(seed)
    values = []
    with torch.inference_mode():
        for _ in range(int(subsets)):
            generated_indices = torch.from_numpy(
                rng.choice(len(generated), size=size, replace=False)
            ).to(device=device)
            reference_indices = torch.from_numpy(
                rng.choice(len(reference), size=size, replace=False)
            ).to(device=device)
            value = polynomial_mmd_unbiased(
                generated_tensor.index_select(0, generated_indices),
                reference_tensor.index_select(0, reference_indices),
            )
            values.append(float(value.item()))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def manifold_radii(features, *, neighborhood: int, batch_size: int):
    import torch

    values = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], int(batch_size)):
            distances = torch.cdist(features[start : start + batch_size], features)
            values.append(distances.kthvalue(int(neighborhood) + 1, dim=1).values)
    return torch.cat(values)


def manifold_coverage(
    queries,
    centers,
    center_radii,
    *,
    batch_size: int,
) -> float:
    covered = 0
    with __import__("torch").inference_mode():
        for start in range(0, queries.shape[0], int(batch_size)):
            distances = __import__("torch").cdist(
                queries[start : start + batch_size], centers
            )
            covered += int(
                (distances <= center_radii.unsqueeze(0)).any(dim=1).sum().item()
            )
    return covered / queries.shape[0]


def precision_recall(
    generated: np.ndarray,
    reference_tensor,
    reference_radii,
    *,
    neighborhood: int,
    distance_batch: int,
    device,
) -> tuple[float, float]:
    import torch

    generated_tensor = torch.from_numpy(
        generated.astype(np.float32, copy=False)
    ).to(device=device)
    generated_radii = manifold_radii(
        generated_tensor,
        neighborhood=neighborhood,
        batch_size=distance_batch,
    )
    precision = manifold_coverage(
        generated_tensor,
        reference_tensor,
        reference_radii,
        batch_size=distance_batch,
    )
    recall = manifold_coverage(
        reference_tensor,
        generated_tensor,
        generated_radii,
        batch_size=distance_batch,
    )
    return precision, recall


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_decode_diagnostics(directory: Path) -> dict[str, float]:
    paths = sorted(directory.glob("decode_diagnostics_rank*.json"))
    if not paths:
        return {
            "decoder_raw_min": float("nan"),
            "decoder_raw_max": float("nan"),
            "decoder_clipped_low_fraction": float("nan"),
            "decoder_clipped_high_fraction": float("nan"),
        }
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    total = sum(float(item["samples"]) for item in payloads)
    return {
        "decoder_raw_min": min(float(item["raw_min"]) for item in payloads),
        "decoder_raw_max": max(float(item["raw_max"]) for item in payloads),
        "decoder_clipped_low_fraction": sum(
            float(item["clipped_low_weighted"]) for item in payloads
        ) / max(total, 1.0),
        "decoder_clipped_high_fraction": sum(
            float(item["clipped_high_weighted"]) for item in payloads
        ) / max(total, 1.0),
    }


def analyze(args: argparse.Namespace) -> None:
    import torch

    output_root = args.output_root.resolve()
    pair_source = output_root / "pair" / "pair_scale_summary.csv"
    pair_time_source = output_root / "pair" / "pair_error_by_time.csv"
    pair = pd.read_csv(pair_source)
    # Expose the two paired tables beside the main output tables.
    pair.to_csv(output_root / "pair_scale_summary.csv", index=False)
    pd.read_csv(pair_time_source).to_csv(
        output_root / "pair_error_by_time.csv", index=False
    )
    reference = merge_shards(output_root / "reference")
    reference_moments = aggregate_moments(output_root / "reference")
    reference_manifest = json.loads(
        (output_root / "reference" / "complete.json").read_text(encoding="utf-8")
    )
    hidden_indices = [int(value) for value in reference_manifest["hidden_indices"]]

    with np.load(args.fid_reference.resolve()) as payload:
        if "mu" not in payload or "sigma" not in payload:
            raise KeyError(
                f"{args.fid_reference} must contain ADM reference mu and sigma."
            )
        real_mu = np.asarray(payload["mu"], dtype=np.float64)
        real_sigma = np.asarray(payload["sigma"], dtype=np.float64)

    cache_payload = torch.load(
        args.real_feature_cache.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    real_all = extract_tensor_from_cache(cache_payload).float()
    if len(real_all) < args.distribution_samples:
        raise ValueError(
            f"Real feature cache has {len(real_all)} samples, fewer than "
            f"--distribution-samples={args.distribution_samples}"
        )
    rng = np.random.default_rng(args.real_feature_seed)
    real_indices = rng.choice(
        len(real_all),
        size=args.distribution_samples,
        replace=False,
    )
    real_features = real_all[real_indices].numpy().astype(np.float32, copy=False)
    clean_inception = reference["inception"].astype(np.float32, copy=False)

    analysis_device = torch.device(args.analysis_device)
    real_tensor = torch.from_numpy(real_features).to(device=analysis_device)
    clean_tensor = torch.from_numpy(clean_inception).to(device=analysis_device)
    real_radii = manifold_radii(
        real_tensor,
        neighborhood=args.pr_neighborhood,
        batch_size=args.pr_distance_batch,
    )
    clean_radii = manifold_radii(
        clean_tensor,
        neighborhood=args.pr_neighborhood,
        batch_size=args.pr_distance_batch,
    )

    clean_mu, clean_sigma = covariance_statistics(clean_inception)
    clean_fid_real = frechet_distance(clean_mu, clean_sigma, real_mu, real_sigma)
    clean_kid_real, clean_kid_real_std = kid_from_features(
        clean_inception,
        real_features,
        subsets=args.kid_subsets,
        subset_size=args.kid_subset_size,
        seed=args.metric_seed + 11,
        device=analysis_device,
    )
    clean_precision_real, clean_recall_real = precision_recall(
        clean_inception,
        real_tensor,
        real_radii,
        neighborhood=args.pr_neighborhood,
        distance_batch=args.pr_distance_batch,
        device=analysis_device,
    )

    scale_rows = []
    decoder_rows = []

    pair_index = pair.set_index("scale")
    for scale_index, scale in enumerate(args.scales):
        scale_dir = output_root / "scales" / f"ig_{scale_tag(scale)}"
        generated = merge_shards(scale_dir)
        generated_moments = aggregate_moments(scale_dir)
        generated_inception = generated["inception"].astype(np.float32, copy=False)

        latent_distribution = repeated_projected_metrics(
            reference["latent_projection"].astype(np.float32, copy=False),
            generated["latent_projection"].astype(np.float32, copy=False),
            swd_directions=args.swd_directions,
            seed=args.metric_seed + 10000 * scale_index,
        )
        latent_moments = moment_metrics(reference_moments, generated_moments)

        generated_mu, generated_sigma = covariance_statistics(generated_inception)
        gfid = frechet_distance(generated_mu, generated_sigma, real_mu, real_sigma)
        fid_reconstruction = frechet_distance(
            generated_mu,
            generated_sigma,
            clean_mu,
            clean_sigma,
        )
        kid_real, kid_real_std = kid_from_features(
            generated_inception,
            real_features,
            subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            seed=args.metric_seed + 1000 * scale_index + 21,
            device=analysis_device,
        )
        kid_reconstruction, kid_reconstruction_std = kid_from_features(
            generated_inception,
            clean_inception,
            subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            seed=args.metric_seed + 1000 * scale_index + 31,
            device=analysis_device,
        )
        precision_real, recall_real = precision_recall(
            generated_inception,
            real_tensor,
            real_radii,
            neighborhood=args.pr_neighborhood,
            distance_batch=args.pr_distance_batch,
            device=analysis_device,
        )
        precision_reconstruction, recall_reconstruction = precision_recall(
            generated_inception,
            clean_tensor,
            clean_radii,
            neighborhood=args.pr_neighborhood,
            distance_batch=args.pr_distance_batch,
            device=analysis_device,
        )

        decoder_scale_rows = []
        for layer_position, hidden_index in enumerate(hidden_indices):
            clean_pooled = reference[f"decoder_pooled_l{hidden_index}"].astype(
                np.float32, copy=False
            )
            generated_pooled = generated[f"decoder_pooled_l{hidden_index}"].astype(
                np.float32, copy=False
            )
            clean_within = reference[f"decoder_within_l{hidden_index}"].astype(
                np.float64, copy=False
            )
            generated_within = generated[f"decoder_within_l{hidden_index}"].astype(
                np.float64, copy=False
            )
            clean_between = np.var(clean_pooled.astype(np.float64), axis=0, ddof=1).sum()
            generated_between = np.var(
                generated_pooled.astype(np.float64), axis=0, ddof=1
            ).sum()
            decoder_projected = random_project_features(
                clean_pooled,
                generated_pooled,
                output_dim=args.decoder_projection_dim,
                repeats=args.decoder_projection_repeats,
                seed=args.decoder_projection_seed + 10007 * layer_position,
                swd_directions=args.swd_directions,
            )
            row = {
                "scale": float(scale),
                "layer_position": layer_position,
                "decoder_hidden_index": hidden_index,
                "clean_within_spatial_variance": float(clean_within.mean()),
                "generated_within_spatial_variance": float(generated_within.mean()),
                "within_spatial_variance_ratio": float(
                    generated_within.mean() / max(clean_within.mean(), EPS)
                ),
                "clean_between_sample_trace_variance": float(clean_between),
                "generated_between_sample_trace_variance": float(generated_between),
                "between_sample_trace_variance_ratio": float(
                    generated_between / max(clean_between, EPS)
                ),
                "pooled_mean_shift_normalized": float(
                    np.linalg.norm(
                        generated_pooled.mean(0) - clean_pooled.mean(0)
                    )
                    / (
                        math.sqrt(
                            np.var(clean_pooled.astype(np.float64), axis=0, ddof=1).sum()
                        )
                        + EPS
                    )
                ),
                "decoder_projected_fid_mean": decoder_projected["projected_fid_mean"],
                "decoder_projected_fid_std": decoder_projected["projected_fid_std"],
                "decoder_projected_swd_mean": decoder_projected["projected_swd_mean"],
                "decoder_projected_swd_std": decoder_projected["projected_swd_std"],
            }
            decoder_rows.append(row)
            decoder_scale_rows.append(row)

        decoder_frame = pd.DataFrame(decoder_scale_rows)
        diagnostics = load_decode_diagnostics(scale_dir)
        pair_row = pair_index.loc[float(scale)]

        scale_rows.append(
            {
                "scale": float(scale),
                "paired_nmse_equal_call": float(pair_row["paired_nmse_equal_call"]),
                "paired_nrmse_equal_call": float(pair_row["paired_nrmse_equal_call"]),
                "paired_nmse_delta_t": float(pair_row["paired_nmse_delta_t"]),
                "paired_nrmse_delta_t": float(pair_row["paired_nrmse_delta_t"]),
                "projected_latent_fid_mean": latent_distribution["projected_fid_mean"],
                "projected_latent_fid_std": latent_distribution["projected_fid_std"],
                "projected_latent_swd_mean": latent_distribution["projected_swd_mean"],
                "projected_latent_swd_std": latent_distribution["projected_swd_std"],
                "projected_latent_mean_shift": latent_distribution[
                    "projected_mean_shift_mean"
                ],
                "projected_latent_covariance_shift": latent_distribution[
                    "projected_covariance_shift_mean"
                ],
                **latent_moments,
                "gfid_to_real": gfid,
                "fid_to_clean_reconstruction": fid_reconstruction,
                "kid_to_real_mean": kid_real,
                "kid_to_real_std": kid_real_std,
                "kid_to_clean_reconstruction_mean": kid_reconstruction,
                "kid_to_clean_reconstruction_std": kid_reconstruction_std,
                "precision_to_real": precision_real,
                "recall_to_real": recall_real,
                "precision_to_clean_reconstruction": precision_reconstruction,
                "recall_to_clean_reconstruction": recall_reconstruction,
                "decoder_within_variance_ratio_mean_layers": float(
                    decoder_frame["within_spatial_variance_ratio"].mean()
                ),
                "decoder_between_variance_ratio_mean_layers": float(
                    decoder_frame["between_sample_trace_variance_ratio"].mean()
                ),
                "decoder_projected_fid_mean_layers": float(
                    decoder_frame["decoder_projected_fid_mean"].mean()
                ),
                "decoder_projected_swd_mean_layers": float(
                    decoder_frame["decoder_projected_swd_mean"].mean()
                ),
                **diagnostics,
            }
        )
        print(f"[analysis] scale={scale:g} complete", flush=True)

    scale_frame = pd.DataFrame(scale_rows).sort_values("scale")
    decoder_frame = pd.DataFrame(decoder_rows).sort_values(
        ["scale", "layer_position"]
    )
    scale_frame.to_csv(output_root / "scale_summary.csv", index=False)
    decoder_frame.to_csv(
        output_root / "decoder_feature_distribution.csv", index=False
    )

    baselines = {
        "clean_reconstruction_fid_to_real": clean_fid_real,
        "clean_reconstruction_kid_to_real_mean": clean_kid_real,
        "clean_reconstruction_kid_to_real_std": clean_kid_real_std,
        "clean_reconstruction_precision_to_real": clean_precision_real,
        "clean_reconstruction_recall_to_real": clean_recall_real,
        "real_feature_samples_for_kid_pr": int(args.distribution_samples),
        "real_feature_cache": str(args.real_feature_cache.resolve()),
    }
    (output_root / "reference_baselines.json").write_text(
        json.dumps(baselines, indent=2), encoding="utf-8"
    )

    make_plots(args, scale_frame, decoder_frame)

    minima = {}
    for column in (
        "paired_nrmse_delta_t",
        "projected_latent_swd_mean",
        "projected_latent_fid_mean",
        "gfid_to_real",
        "fid_to_clean_reconstruction",
        "decoder_projected_fid_mean_layers",
    ):
        best = scale_frame.loc[scale_frame[column].idxmin()]
        minima[column] = {
            "best_scale": float(best["scale"]),
            "best_value": float(best[column]),
        }
    report = {
        "protocol": "official_raev2_fixed_scale_metric_suite_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "state_key": args.state_key,
        "scales": args.scales,
        "pair_samples": int(args.pair_samples),
        "distribution_samples_per_scale": int(args.distribution_samples),
        "sampling_seed": int(args.sampling_seed),
        "reference_seed": int(args.reference_seed),
        "metric_seed": int(args.metric_seed),
        "minima": minima,
        "reference_baselines": baselines,
        "interpretation": {
            "paired_error": "Teacher-forced per-sample clean-latent regression.",
            "latent_distribution": (
                "Free endpoint latent marginal vs clean encoder marginal. "
                "FID/SWD are explicitly projected diagnostics."
            ),
            "decoded_distribution": (
                "gFID uses full ADM real statistics; reconstruction-relative FID "
                "uses the matched 5k clean decoded reference."
            ),
            "decoder_variance": (
                "Within-image spatial variance and between-image pooled-feature "
                "variance are reported separately."
            ),
        },
    }
    (output_root / "final_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "Official RAEv2 fixed-scale metric suite",
        "=======================================",
        "",
        f"Scales: {args.scales}",
        f"Pair samples: {args.pair_samples}",
        f"Free/reference samples per scale: {args.distribution_samples}",
        "",
        "Observed minima:",
    ]
    for metric, item in minima.items():
        lines.append(
            f"  {metric}: scale={item['best_scale']:.6g}, "
            f"value={item['best_value']:.8g}"
        )
    lines.extend(
        [
            "",
            "Clean reconstruction baseline:",
            f"  FID to real: {clean_fid_real:.8g}",
            f"  KID to real: {clean_kid_real:.8g} ± {clean_kid_real_std:.8g}",
            f"  Precision/Recall to real: "
            f"{clean_precision_real:.6g} / {clean_recall_real:.6g}",
            "",
            "See scale_summary.csv and decoder_feature_distribution.csv "
            "for the complete table.",
        ]
    )
    (output_root / "final_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


def make_plots(
    args: argparse.Namespace,
    scales: pd.DataFrame,
    decoder: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_root.resolve() / "plots"
    output.mkdir(parents=True, exist_ok=True)

    def line_plot(columns: Sequence[str], labels: Sequence[str], filename: str, ylabel: str):
        figure, axis = plt.subplots(figsize=(8.6, 5.4))
        for column, label in zip(columns, labels):
            axis.plot(scales["scale"], scales[column], marker="o", label=label)
        axis.set_xlabel("IG scale")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=185)
        plt.close(figure)

    line_plot(
        ["paired_nrmse_delta_t"],
        ["paired nRMSE (delta-t)"],
        "scale_vs_pair_error.png",
        "Paired clean-latent relative RMSE",
    )
    line_plot(
        ["projected_latent_swd_mean", "projected_latent_fid_mean"],
        ["projected latent SWD", "projected latent FID"],
        "scale_vs_latent_distribution.png",
        "Latent distribution distance",
    )
    line_plot(
        ["gfid_to_real", "fid_to_clean_reconstruction"],
        ["gFID to real", "FID to clean reconstruction"],
        "scale_vs_decoded_fid.png",
        "FID",
    )
    line_plot(
        ["kid_to_real_mean", "kid_to_clean_reconstruction_mean"],
        ["KID to real", "KID to clean reconstruction"],
        "scale_vs_kid.png",
        "KID",
    )
    line_plot(
        ["precision_to_real", "recall_to_real"],
        ["precision to real", "recall to real"],
        "precision_recall_vs_scale.png",
        "Improved precision / recall",
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for hidden_index, frame in decoder.groupby("decoder_hidden_index"):
        ordered = frame.sort_values("scale")
        axes[0].plot(
            ordered["scale"],
            ordered["within_spatial_variance_ratio"],
            marker="o",
            label=f"layer {hidden_index}",
        )
        axes[1].plot(
            ordered["scale"],
            ordered["between_sample_trace_variance_ratio"],
            marker="o",
            label=f"layer {hidden_index}",
        )
    axes[0].axhline(1.0, linewidth=1)
    axes[1].axhline(1.0, linewidth=1)
    axes[0].set_ylabel("Within-image spatial variance ratio")
    axes[1].set_ylabel("Between-image trace variance ratio")
    for axis in axes:
        axis.set_xlabel("IG scale")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "scale_vs_decoder_feature_variance.png", dpi=185)
    plt.close(figure)

    scatter_pairs = (
        ("paired_nrmse_delta_t", "gfid_to_real", "pair_error_vs_gfid.png"),
        ("projected_latent_swd_mean", "gfid_to_real", "latent_swd_vs_gfid.png"),
        (
            "decoder_projected_fid_mean_layers",
            "gfid_to_real",
            "decoder_feature_fid_vs_gfid.png",
        ),
    )
    for x_column, y_column, filename in scatter_pairs:
        figure, axis = plt.subplots(figsize=(6.4, 5.4))
        axis.scatter(scales[x_column], scales[y_column])
        for _, row in scales.iterrows():
            axis.annotate(
                f"s={row['scale']:g}",
                (row[x_column], row[y_column]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=185)
        plt.close(figure)


# ---------------------------------------------------------------------------
# Orchestration and protocol locking
# ---------------------------------------------------------------------------


def worker_command(args: argparse.Namespace, mode: str, scale: float | None = None):
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    command = [
        str(args.python.resolve()),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(devices)}",
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--repo",
        str(args.repo.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--python",
        str(args.python.resolve()),
        "--config",
        str(args.config.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--index-map",
        str(args.index_map.resolve()),
        "--packed-data-path",
        str(args.packed_data_path.resolve()),
        "--parquet-data-path",
        str(args.parquet_data_path.resolve()),
        "--fid-reference",
        str(args.fid_reference.resolve()),
        "--real-feature-cache",
        str(args.real_feature_cache.resolve()),
        "--dino-ckpt-dir",
        str(args.dino_ckpt_dir.resolve()),
        "--dino-repo-dir",
        str(args.dino_repo_dir.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--devices",
        args.devices,
        "--state-key",
        args.state_key,
        "--precision",
        args.precision,
        "--pair-samples",
        str(args.pair_samples),
        "--pair-batch-size",
        str(args.pair_batch_size),
        "--pair-time-chunk",
        str(args.pair_time_chunk),
        "--pair-seed",
        str(args.pair_seed),
        "--distribution-samples",
        str(args.distribution_samples),
        "--distribution-batch-size",
        str(args.distribution_batch_size),
        "--sampling-seed",
        str(args.sampling_seed),
        "--reference-seed",
        str(args.reference_seed),
        "--metric-seed",
        str(args.metric_seed),
        "--latent-projection-dim",
        str(args.latent_projection_dim),
        "--latent-projection-repeats",
        str(args.latent_projection_repeats),
        "--latent-projection-seed",
        str(args.latent_projection_seed),
        "--decoder-layer-fractions",
        ",".join(str(value) for value in args.decoder_layer_fractions),
        "--example-count",
        str(args.example_count),
        "--num-workers",
        str(args.num_workers),
        "--log-every-batches",
        str(args.log_every_batches),
        "--scales",
        ",".join(str(value) for value in args.scales),
    ]
    if scale is not None:
        command.extend(["--worker-scale", str(scale)])
    return command


def preflight(args: argparse.Namespace) -> None:
    require_dir(args.repo.resolve(), "repository")
    require_file(args.python.resolve(), "RAEv2 Python")
    require_file(args.config.resolve(), "RAEv2 config")
    require_file(args.checkpoint.resolve(), "official checkpoint")
    require_file(args.index_map.resolve(), "ImageNet index map")
    require_dir(args.packed_data_path.resolve(), "packed ImageNet")
    require_dir(args.parquet_data_path.resolve(), "ImageNet parquet root")
    require_file(args.fid_reference.resolve(), "ADM reference")
    require_file(args.real_feature_cache.resolve(), "real Inception feature cache")
    require_dir(args.dino_ckpt_dir.resolve(), "DINOv3 checkpoint directory")
    require_dir(args.dino_repo_dir.resolve(), "DINOv3 repository")
    if args.distribution_samples <= 3:
        raise ValueError("--distribution-samples must exceed 3.")
    if args.pair_samples <= 0:
        raise ValueError("--pair-samples must be positive.")
    if args.latent_projection_dim <= 1:
        raise ValueError("--latent-projection-dim must exceed one.")
    if args.latent_projection_repeats <= 0:
        raise ValueError("--latent-projection-repeats must be positive.")


def protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": "official_raev2_fixed_scale_metric_suite_v1",
        "repo": str(args.repo.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "state_key": args.state_key,
        "scales": args.scales,
        "pair_samples": args.pair_samples,
        "pair_seed": args.pair_seed,
        "distribution_samples": args.distribution_samples,
        "sampling_seed": args.sampling_seed,
        "reference_seed": args.reference_seed,
        "metric_seed": args.metric_seed,
        "latent_projection_dim": args.latent_projection_dim,
        "latent_projection_repeats": args.latent_projection_repeats,
        "latent_projection_seed": args.latent_projection_seed,
        "decoder_layer_fractions": args.decoder_layer_fractions,
        "decoder_projection_dim": args.decoder_projection_dim,
        "decoder_projection_repeats": args.decoder_projection_repeats,
        "decoder_projection_seed": args.decoder_projection_seed,
        "swd_directions": args.swd_directions,
        "kid_subsets": args.kid_subsets,
        "kid_subset_size": args.kid_subset_size,
        "pr_neighborhood": args.pr_neighborhood,
        "pr_distance_batch": args.pr_distance_batch,
        "real_feature_seed": args.real_feature_seed,
        "real_feature_cache": str(args.real_feature_cache.resolve()),
        "fid_reference": str(args.fid_reference.resolve()),
        "precision": args.precision,
        "devices": args.devices,
    }


def orchestrate(args: argparse.Namespace) -> None:
    preflight(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "study_manifest.json"
    requested = protocol_manifest(args)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != requested:
            raise ValueError(
                "The output root is locked to a different protocol. "
                "Use a new --output-root after changing scales, seeds, counts, "
                "checkpoint, or projection settings."
            )
    else:
        manifest_path.write_text(
            json.dumps(requested, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = (
        f"{args.repo.resolve()}:{base_env.get('PYTHONPATH', '')}"
    )
    base_env["CUDA_VISIBLE_DEVICES"] = args.devices

    pair_complete = args.output_root / "pair" / "complete.json"
    if not pair_complete.is_file():
        run_logged(
            worker_command(args, "pair-worker"),
            cwd=args.repo.resolve(),
            log_path=args.output_root / "pair" / "run.log",
            env=base_env,
            dry_run=args.dry_run,
        )
    else:
        print(f"[resume] pair complete: {pair_complete}")

    reference_complete = args.output_root / "reference" / "complete.json"
    if not reference_complete.is_file():
        run_logged(
            worker_command(args, "reference-worker"),
            cwd=args.repo.resolve(),
            log_path=args.output_root / "reference" / "run.log",
            env=base_env,
            dry_run=args.dry_run,
        )
    else:
        print(f"[resume] reference complete: {reference_complete}")

    for scale in args.scales:
        scale_dir = args.output_root / "scales" / f"ig_{scale_tag(scale)}"
        complete = scale_dir / "complete.json"
        if not complete.is_file():
            run_logged(
                worker_command(args, "scale-worker", scale=scale),
                cwd=args.repo.resolve(),
                log_path=scale_dir / "run.log",
                env=base_env,
                dry_run=args.dry_run,
            )
        else:
            print(f"[resume] scale={scale:g} complete: {complete}")

    if args.dry_run:
        print("Dry run complete.")
        return
    analyze(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("all", "pair-worker", "reference-worker", "scale-worker", "analyze"),
        default="all",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_DATA / "envs/raev2/bin/python",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REPO
        / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_DATA
        / "models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt",
    )
    parser.add_argument(
        "--index-map",
        type=Path,
        default=DEFAULT_DATA
        / "datasets/raev2_imagenet_train_lexicographic_indices.npy",
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
    parser.add_argument(
        "--real-feature-cache",
        type=Path,
        default=Path.home()
        / ".cache/torch/fidelity_cache/"
        "raev2_imagenet256_virtual_reference-inception-v3-compat-features-2048.pt",
    )
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=DEFAULT_DATA / "models/RAEv2/encoders/dinov3",
    )
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=DEFAULT_DATA / "models/RAEv2/dinov3_repo",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA
        / "experiments/official_scale_metric_suite_seed20260805",
    )

    parser.add_argument(
        "--scales",
        type=parse_csv_floats,
        default=parse_csv_floats(DEFAULT_SCALES),
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--state-key", choices=("model", "ema"), default="model")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--example-count", type=int, default=16)

    parser.add_argument("--pair-samples", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=1)
    parser.add_argument("--pair-time-chunk", type=int, default=4)
    parser.add_argument("--pair-seed", type=int, default=20260805)

    parser.add_argument("--distribution-samples", type=int, default=5000)
    parser.add_argument("--distribution-batch-size", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=20260806)
    parser.add_argument("--reference-seed", type=int, default=20260805)
    parser.add_argument("--metric-seed", type=int, default=20260805)
    parser.add_argument("--real-feature-seed", type=int, default=20260805)

    parser.add_argument("--latent-projection-dim", type=int, default=256)
    parser.add_argument("--latent-projection-repeats", type=int, default=4)
    parser.add_argument("--latent-projection-seed", type=int, default=8309)
    parser.add_argument("--swd-directions", type=int, default=256)

    parser.add_argument(
        "--decoder-layer-fractions",
        type=parse_csv_floats,
        default=parse_csv_floats("0.2,0.4,0.6,0.8,1.0"),
    )
    parser.add_argument("--decoder-projection-dim", type=int, default=128)
    parser.add_argument("--decoder-projection-repeats", type=int, default=2)
    parser.add_argument("--decoder-projection-seed", type=int, default=14243)

    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--pr-neighborhood", type=int, default=3)
    parser.add_argument("--pr-distance-batch", type=int, default=250)
    parser.add_argument("--analysis-device", default="cuda:0")

    parser.add_argument("--worker-scale", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "pair-worker":
        pair_worker(args)
    elif args.mode == "reference-worker":
        reference_worker(args)
    elif args.mode == "scale-worker":
        scale_worker(args)
    elif args.mode == "analyze":
        preflight(args)
        analyze(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()