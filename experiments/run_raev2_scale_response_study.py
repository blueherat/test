"""Resumable RAEv2 internal-guidance scale response study.

The frozen official model is evaluated in three spaces for a shared set of
ImageNet classes and initial noises:

1. sampled endpoint latent ``z_s`` versus real encoder latent ``E(x)``;
2. round-trip latent ``E(D(z_s))`` versus ``E(D(E(x)))``;
3. decoded images ``D(z_s)`` versus source and reconstructed images.

Every expensive phase writes rank-local shards and an atomic completion
marker. Rerunning the same command resumes at the first incomplete phase.
This is an inference-only experiment; no parameter receives gradients.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.compute_raev2_predicted_clean_kid import (  # noqa: E402
    polynomial_mmd_unbiased,
)
from experiments.raev2_stage1_compat import (  # noqa: E402
    install_raev2_decoder_config_compat,
)
from experiments.raev2_training_core import (  # noqa: E402
    DeterministicImageNetPacked,
    validate_full_stage2_checkpoint,
)
from experiments.run_raev2_decoded_distribution_audit import (  # noqa: E402
    feature_probe_scores,
    feature_statistics,
    fid_between_statistics,
    fit_feature_probe,
    load_reference_statistics,
)
from experiments.run_raev2_distribution_auc import (  # noqa: E402
    autocast_context,
    bootstrap_paired_auc,
    build_requested_labels,
    class_group_split,
    load_config,
    select_matching_imagenet_rows,
)
from stage2.transport import create_sampler, create_transport  # noqa: E402
from utils.guidance_utils import forward_with_internalguidance  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402


PROTOCOL = "raev2_ig_scale_response_v1"
DEFAULT_SCALES = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.78, 2.0, 2.2)


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
    parser.add_argument("--scale", action="append", type=float, dest="scales")
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--encode-batch", type=int, default=4)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--state-key", choices=("ema", "model"), default="ema")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--metric-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument(
        "--dino-ckpt-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3"),
    )
    parser.add_argument(
        "--dino-repo-dir",
        type=Path,
        default=Path("/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo"),
    )
    return parser.parse_args()


def scale_key(scale: float) -> str:
    if not math.isfinite(float(scale)) or float(scale) < 0:
        raise ValueError("guidance scales must be finite and non-negative")
    return f"scale_s{float(scale):.6f}".replace(".", "p")


def normalized_scales(values: Iterable[float] | None) -> tuple[float, ...]:
    scales = tuple(sorted(set(float(value) for value in (values or DEFAULT_SCALES))))
    if not scales:
        raise ValueError("at least one guidance scale is required")
    for scale in scales:
        scale_key(scale)
    if 1.0 not in scales:
        raise ValueError("scale sweep must include s=1.0 as the unguided full-head control")
    return scales


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def marker_path(output_dir: Path, phase: str) -> Path:
    return output_dir / "markers" / f"{phase}.json"


def marker_exists(output_dir: Path, phase: str) -> bool:
    return marker_path(output_dir, phase).is_file()


def complete_phase(
    output_dir: Path, phase: str, *, rank: int, world_size: int, **details: Any
) -> None:
    dist.barrier()
    if rank == 0:
        atomic_write_json(
            marker_path(output_dir, phase),
            {"phase": phase, "world_size": world_size, **details},
        )
    dist.barrier()


def rank_file(directory: Path, condition: str, rank: int) -> Path:
    return directory / f"{condition}_rank{rank:02d}.npy"


def local_ids_for_rank(samples: int, rank: int, world_size: int) -> np.ndarray:
    return np.arange(rank, samples, world_size, dtype=np.int64)


def validate_local_array(path: Path, expected_rows: int, sample_shape: tuple[int, ...]) -> None:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != (expected_rows, *sample_shape):
        raise RuntimeError(f"unexpected array shape in {path}: {value.shape}")
    if value.dtype not in (np.float16, np.float32, np.uint8):
        raise RuntimeError(f"unexpected array dtype in {path}: {value.dtype}")


def write_initial_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    scales: tuple[float, ...],
    config: Any,
    checkpoint_path: Path,
    labels: np.ndarray,
    source_rows: np.ndarray,
    test_mask: np.ndarray,
    world_size: int,
) -> None:
    manifest_path = output_dir / "manifest.json"
    expected = {
        "protocol": PROTOCOL,
        "status": "running",
        "inference_only": True,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "scales": list(scales),
        "code_scale_convention": "base + s * (full - base); s=1 is full",
        "world_size": int(world_size),
        "latent_size": [int(value) for value in config.misc.latent_size],
        "sampler_steps": int(config.sampler.num_steps),
        "ig_interval": [
            float(config.guidance.ig.t_min),
            float(config.guidance.ig.t_max),
        ],
        "precision": args.precision,
        "state_key": args.state_key,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_path.stat().st_size,
        "packed_data_path": str(args.packed_data_path.expanduser().resolve()),
        "parquet_data_path": str(args.parquet_data_path.expanduser().resolve()),
        "fid_reference": str(args.fid_reference.expanduser().resolve()),
        "test_fraction": float(args.test_fraction),
        "train_samples": int((~test_mask).sum()),
        "heldout_samples": int(test_mask.sum()),
        "class_disjoint_split": True,
        "same_noise_and_labels_across_scales": True,
        "skip_fid": bool(args.skip_fid),
    }
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "protocol",
            "samples",
            "seed",
            "scales",
            "world_size",
            "latent_size",
            "sampler_steps",
            "ig_interval",
            "precision",
            "state_key",
            "checkpoint",
        )
        mismatched = [
            key for key in immutable_keys if current.get(key) != expected.get(key)
        ]
        if mismatched:
            raise RuntimeError(
                f"cannot resume with a changed protocol; mismatched keys: {mismatched}"
            )
    else:
        atomic_write_json(manifest_path, expected)
        np.savez_compressed(
            output_dir / "sample_protocol.npz",
            sample_ids=np.arange(args.samples, dtype=np.int64),
            labels=labels,
            real_source_rows=source_rows,
            test_mask=test_mask,
        )


def release_model(model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def collect_real_latents(
    *,
    args: argparse.Namespace,
    config: Any,
    output_dir: Path,
    local_ids: np.ndarray,
    labels: np.ndarray,
    source_rows: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    phase = "real_latents"
    if marker_exists(output_dir, phase):
        return
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
    latent_parts: list[np.ndarray] = []
    total_batches = math.ceil(local_ids.size / args.encode_batch)
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, local_ids.size, args.encode_batch)):
            ids = local_ids[start : start + args.encode_batch]
            images = []
            for sample_id in ids.tolist():
                image, actual_label, _ = dataset[int(source_rows[sample_id])]
                if int(actual_label) != int(labels[sample_id]):
                    raise RuntimeError(f"ImageNet label mismatch for sample {sample_id}")
                images.append(image)
            batch = torch.stack(images).to(device=device)
            with autocast_context(args.precision):
                latent = rae.encode(batch).float()
            latent_parts.append(latent.cpu().to(torch.float16).numpy())
            if rank == 0 and (
                (batch_index + 1) % args.log_every_batches == 0
                or batch_index + 1 == total_batches
            ):
                print(f"[real latent] {batch_index + 1}/{total_batches}", flush=True)
    path = rank_file(output_dir / "latents", "real", rank)
    atomic_save_npy(path, np.concatenate(latent_parts, axis=0))
    validate_local_array(
        path,
        local_ids.size,
        tuple(int(value) for value in config.misc.latent_size),
    )
    dataset.close()
    release_model(rae)
    complete_phase(
        output_dir,
        phase,
        rank=rank,
        world_size=world_size,
        rows=int(args.samples),
    )


def sample_scale_endpoints(
    *,
    args: argparse.Namespace,
    config: Any,
    scales: tuple[float, ...],
    output_dir: Path,
    local_ids: np.ndarray,
    labels: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    incomplete = [
        scale
        for scale in scales
        if not marker_exists(output_dir, f"sample_{scale_key(scale)}")
    ]
    if not incomplete:
        return
    model = instantiate_from_config(config.stage_2).to(device).eval()
    model.requires_grad_(False)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    validate_full_stage2_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    del checkpoint
    gc.collect()

    latent_size = tuple(int(value) for value in config.misc.latent_size)
    shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size))
        / config.misc.time_dist_shift_base
    )
    transport = create_transport(config=config.transport, time_dist_shift=shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    sample_fn = sampler.sample_ode(**dataclasses.asdict(config.sampler))
    model_fn = partial(forward_with_internalguidance, model)
    interval = (float(config.guidance.ig.t_min), float(config.guidance.ig.t_max))
    generator = torch.Generator(device="cpu").manual_seed(
        int(args.seed) + 1_000_003 * rank
    )
    local_noise = torch.randn(
        (local_ids.size, *latent_size), generator=generator, dtype=torch.float32
    )
    noise_hash = hashlib.sha256(local_noise.numpy().tobytes()).hexdigest()

    for scale in incomplete:
        endpoint_parts: list[np.ndarray] = []
        total_batches = math.ceil(local_ids.size / args.per_rank_batch)
        with torch.inference_mode():
            for batch_index, start in enumerate(
                range(0, local_ids.size, args.per_rank_batch)
            ):
                ids = local_ids[start : start + args.per_rank_batch]
                noise = local_noise[start : start + args.per_rank_batch].to(device=device)
                doubled_noise = torch.cat([noise, noise], dim=0)
                batch_labels = torch.from_numpy(labels[ids]).to(
                    device=device, dtype=torch.long
                )
                null_labels = torch.full(
                    (ids.size,),
                    int(config.misc.num_classes),
                    device=device,
                    dtype=torch.long,
                )
                context = torch.cat([batch_labels, null_labels], dim=0)
                with autocast_context(args.precision):
                    trajectory = sample_fn(
                        doubled_noise,
                        model_fn,
                        context=context,
                        attn_mask=None,
                        ig_scale=float(scale),
                        ig_interval=interval,
                    )
                endpoint = trajectory[-1]
                if endpoint.shape[0] == 2 * ids.size:
                    endpoint = endpoint.chunk(2, dim=0)[0]
                if endpoint.shape != (ids.size, *latent_size):
                    raise RuntimeError(
                        f"unexpected endpoint shape for scale {scale}: {endpoint.shape}"
                    )
                if not torch.isfinite(endpoint).all():
                    raise FloatingPointError(f"non-finite endpoint for scale {scale}")
                endpoint_parts.append(endpoint.float().cpu().to(torch.float16).numpy())
                if rank == 0 and (
                    (batch_index + 1) % args.log_every_batches == 0
                    or batch_index + 1 == total_batches
                ):
                    print(
                        f"[sample s={scale:g}] {batch_index + 1}/{total_batches}",
                        flush=True,
                    )
        path = rank_file(output_dir / "latents", scale_key(scale), rank)
        atomic_save_npy(path, np.concatenate(endpoint_parts, axis=0))
        validate_local_array(path, local_ids.size, latent_size)
        complete_phase(
            output_dir,
            f"sample_{scale_key(scale)}",
            rank=rank,
            world_size=world_size,
            scale=float(scale),
            rank_noise_sha256=noise_hash if rank == 0 else None,
        )
    release_model(model)


def extract_inception(
    extractor: torch.nn.Module, images_uint8: torch.Tensor
) -> np.ndarray:
    return extractor(images_uint8)[0].float().cpu().numpy().astype(np.float32)


def decode_condition(
    *,
    condition: str,
    latent_path: Path,
    image_path: Path,
    feature_path: Path,
    decoder: torch.nn.Module,
    extractor: torch.nn.Module,
    batch_size: int,
    precision: str,
    log_every: int,
    rank: int,
    device: torch.device,
) -> None:
    latents = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    image_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    total_batches = math.ceil(latents.shape[0] / batch_size)
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, latents.shape[0], batch_size)):
            batch_np = np.asarray(latents[start : start + batch_size], dtype=np.float32)
            latent = torch.from_numpy(batch_np).to(device=device)
            with autocast_context(precision):
                decoded = decoder.decode(latent).float()
            if not torch.isfinite(decoded).all():
                raise FloatingPointError(f"decoder produced non-finite pixels for {condition}")
            clamped = decoded.clamp(0, 1)
            # Preserve the continuous decoder output for the scientific
            # round-trip. Inception still receives the standard uint8 image,
            # but E(D(z)) must not silently include an extra Q8 operation.
            image_parts.append(
                clamped.permute(0, 2, 3, 1).cpu().to(torch.float16).numpy()
            )
            feature_parts.append(
                extract_inception(extractor, clamped.mul(255).to(torch.uint8))
            )
            if rank == 0 and (
                (batch_index + 1) % log_every == 0
                or batch_index + 1 == total_batches
            ):
                print(f"[decode {condition}] {batch_index + 1}/{total_batches}", flush=True)
    atomic_save_npy(image_path, np.concatenate(image_parts, axis=0))
    atomic_save_npy(feature_path, np.concatenate(feature_parts, axis=0))


def decode_all_conditions(
    *,
    args: argparse.Namespace,
    config: Any,
    scales: tuple[float, ...],
    output_dir: Path,
    local_ids: np.ndarray,
    labels: np.ndarray,
    source_rows: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    conditions = ("real", *(scale_key(scale) for scale in scales))
    incomplete = [
        condition
        for condition in conditions
        if not marker_exists(output_dir, f"decode_{condition}")
    ]
    source_phase = "source_inception"
    if not incomplete and marker_exists(output_dir, source_phase):
        return

    decoder = instantiate_from_config(config.stage_1)
    del decoder.encoder
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).to(device)

    if not marker_exists(output_dir, source_phase):
        dataset = DeterministicImageNetPacked(
            args.packed_data_path,
            split="train",
            image_size=int(config.training.image_size),
            horizontal_flip=False,
        )
        feature_parts: list[np.ndarray] = []
        total_batches = math.ceil(local_ids.size / args.decode_batch)
        with torch.inference_mode():
            for batch_index, start in enumerate(
                range(0, local_ids.size, args.decode_batch)
            ):
                ids = local_ids[start : start + args.decode_batch]
                images = []
                for sample_id in ids.tolist():
                    image, actual_label, _ = dataset[int(source_rows[sample_id])]
                    if int(actual_label) != int(labels[sample_id]):
                        raise RuntimeError(f"ImageNet label mismatch for sample {sample_id}")
                    images.append(image)
                batch = torch.stack(images).to(device=device)
                uint8 = batch.clamp(0, 1).mul(255).to(torch.uint8)
                feature_parts.append(extract_inception(extractor, uint8))
                if rank == 0 and (
                    (batch_index + 1) % args.log_every_batches == 0
                    or batch_index + 1 == total_batches
                ):
                    print(
                        f"[source inception] {batch_index + 1}/{total_batches}",
                        flush=True,
                    )
        source_feature_path = rank_file(output_dir / "inception", "source", rank)
        atomic_save_npy(source_feature_path, np.concatenate(feature_parts, axis=0))
        dataset.close()
        complete_phase(
            output_dir,
            source_phase,
            rank=rank,
            world_size=world_size,
        )

    for condition in incomplete:
        decode_condition(
            condition=condition,
            latent_path=rank_file(output_dir / "latents", condition, rank),
            image_path=rank_file(output_dir / "decoded", condition, rank),
            feature_path=rank_file(output_dir / "inception", condition, rank),
            decoder=decoder,
            extractor=extractor,
            batch_size=args.decode_batch,
            precision=args.precision,
            log_every=args.log_every_batches,
            rank=rank,
            device=device,
        )
        validate_local_array(
            rank_file(output_dir / "decoded", condition, rank),
            local_ids.size,
            (int(config.training.image_size), int(config.training.image_size), 3),
        )
        validate_local_array(
            rank_file(output_dir / "inception", condition, rank),
            local_ids.size,
            (2048,),
        )
        complete_phase(
            output_dir,
            f"decode_{condition}",
            rank=rank,
            world_size=world_size,
            condition=condition,
        )
    del extractor
    release_model(decoder)


def encode_decoded_condition(
    *,
    condition: str,
    image_path: Path,
    latent_path: Path,
    encoder: torch.nn.Module,
    batch_size: int,
    precision: str,
    log_every: int,
    rank: int,
    device: torch.device,
) -> None:
    images = np.load(image_path, mmap_mode="r", allow_pickle=False)
    latent_parts: list[np.ndarray] = []
    total_batches = math.ceil(images.shape[0] / batch_size)
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, images.shape[0], batch_size)):
            # The source is a read-only mmap. Copy before exposing it to
            # PyTorch so later tensor operations can never mutate mmap-backed
            # storage through an unsafe non-writable NumPy view.
            batch_np = np.array(
                images[start : start + batch_size], copy=True
            )
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(
                device=device, dtype=torch.float32
            )
            # Retain compatibility with early smoke artifacts while all new
            # studies use float16 clamped decoder outputs.
            if batch_np.dtype == np.uint8:
                batch.div_(255.0)
            with autocast_context(precision):
                latent = encoder.encode(batch).float()
            latent_parts.append(latent.cpu().to(torch.float16).numpy())
            if rank == 0 and (
                (batch_index + 1) % log_every == 0
                or batch_index + 1 == total_batches
            ):
                print(
                    f"[roundtrip {condition}] {batch_index + 1}/{total_batches}",
                    flush=True,
                )
    atomic_save_npy(latent_path, np.concatenate(latent_parts, axis=0))


def encode_roundtrips(
    *,
    args: argparse.Namespace,
    config: Any,
    scales: tuple[float, ...],
    output_dir: Path,
    local_ids: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    conditions = ("real", *(scale_key(scale) for scale in scales))
    incomplete = [
        condition
        for condition in conditions
        if not marker_exists(output_dir, f"roundtrip_{condition}")
    ]
    if not incomplete:
        return
    encoder = instantiate_from_config(config.stage_1)
    del encoder.decoder
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    latent_size = tuple(int(value) for value in config.misc.latent_size)
    for condition in incomplete:
        path = rank_file(output_dir / "roundtrip", condition, rank)
        encode_decoded_condition(
            condition=condition,
            image_path=rank_file(output_dir / "decoded", condition, rank),
            latent_path=path,
            encoder=encoder,
            batch_size=args.encode_batch,
            precision=args.precision,
            log_every=args.log_every_batches,
            rank=rank,
            device=device,
        )
        validate_local_array(path, local_ids.size, latent_size)
        complete_phase(
            output_dir,
            f"roundtrip_{condition}",
            rank=rank,
            world_size=world_size,
            condition=condition,
        )
    release_model(encoder)


def load_ordered_rank_arrays(
    directory: Path,
    condition: str,
    *,
    samples: int,
    world_size: int,
) -> np.ndarray:
    arrays = []
    ids = []
    for rank in range(world_size):
        value = np.load(
            rank_file(directory, condition, rank), mmap_mode=None, allow_pickle=False
        )
        expected_ids = local_ids_for_rank(samples, rank, world_size)
        if value.shape[0] != expected_ids.size:
            raise RuntimeError(f"rank row count mismatch for {condition} rank {rank}")
        arrays.append(value)
        ids.append(expected_ids)
    joined_ids = np.concatenate(ids)
    order = np.argsort(joined_ids)
    if not np.array_equal(joined_ids[order], np.arange(samples)):
        raise RuntimeError(f"sample IDs are incomplete for {condition}")
    return np.concatenate(arrays, axis=0)[order]


def diagonal_lda_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    test_mask: np.ndarray,
    *,
    ridge_ratio: float,
) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.shape[0] != test_mask.size:
        raise ValueError("latent arrays and split mask must align")
    if reference.ndim < 2 or (~test_mask).sum() < 2 or test_mask.sum() < 1:
        raise ValueError("insufficient train/test samples for latent C2ST")
    ref = reference.reshape(reference.shape[0], -1)
    cand = candidate.reshape(candidate.shape[0], -1)
    train = ~test_mask
    ref_mean = ref[train].mean(axis=0, dtype=np.float64)
    cand_mean = cand[train].mean(axis=0, dtype=np.float64)
    ref_second = np.square(ref[train], dtype=np.float32).mean(axis=0, dtype=np.float64)
    cand_second = np.square(cand[train], dtype=np.float32).mean(axis=0, dtype=np.float64)
    ref_var = np.maximum(ref_second - np.square(ref_mean), 0.0)
    cand_var = np.maximum(cand_second - np.square(cand_mean), 0.0)
    pooled = 0.5 * (ref_var + cand_var)
    positive = pooled[pooled > 0]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    ridge = max(float(ridge_ratio) * base_scale, 1e-12)
    weight = ((cand_mean - ref_mean) / (pooled + ridge)).astype(np.float32)
    norm = float(np.linalg.norm(weight))
    if norm > 0:
        weight /= norm
    intercept = -0.5 * float(weight.astype(np.float64) @ (ref_mean + cand_mean))
    ref_scores = ref[test_mask].astype(np.float32) @ weight + intercept
    cand_scores = cand[test_mask].astype(np.float32) @ weight + intercept
    labels = np.concatenate(
        [np.zeros(ref_scores.size, dtype=np.int8), np.ones(cand_scores.size, dtype=np.int8)]
    )
    auc = float(roc_auc_score(labels, np.concatenate([ref_scores, cand_scores])))
    mean_shift_rms = float(np.sqrt(np.mean(np.square(cand_mean - ref_mean))))
    reference_std_rms = float(np.sqrt(np.mean(ref_var)))
    variance_relative = float(
        np.linalg.norm(cand_var - ref_var) / max(np.linalg.norm(ref_var), 1e-30)
    )
    return {
        "auc": auc,
        "auc_separability": 0.5 + abs(auc - 0.5),
        "ridge": ridge,
        "mean_shift_rms": mean_shift_rms,
        "mean_shift_over_reference_std": mean_shift_rms
        / max(reference_std_rms, 1e-30),
        "diagonal_variance_relative_l2": variance_relative,
        "mean_variance_ratio": float(np.mean(cand_var) / max(np.mean(ref_var), 1e-30)),
    }


def fixed_random_sketch(
    values: np.ndarray,
    projection: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    if flat.shape[1] != projection.shape[0]:
        raise ValueError("projection input dimension does not match latent")
    outputs = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], batch_size):
            batch = torch.from_numpy(
                flat[start : start + batch_size].astype(np.float32, copy=False)
            ).to(device=device)
            outputs.append((batch @ projection).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def rbf_mmd_squared(
    reference: np.ndarray, candidate: np.ndarray, bandwidth_sq: float
) -> float:
    x = torch.from_numpy(reference.astype(np.float32, copy=False))
    y = torch.from_numpy(candidate.astype(np.float32, copy=False))
    if x.shape != y.shape or x.shape[0] < 2:
        raise ValueError("RBF MMD requires matching matrices with at least two rows")
    gamma = 0.5 / max(float(bandwidth_sq), 1e-12)
    d_xx = torch.cdist(x, x).square()
    d_yy = torch.cdist(y, y).square()
    d_xy = torch.cdist(x, y).square()
    k_xx = torch.exp(-gamma * d_xx)
    k_yy = torch.exp(-gamma * d_yy)
    k_xy = torch.exp(-gamma * d_xy)
    # Use the empirical squared RKHS distance. Unlike the unbiased estimator,
    # this is exactly zero for identical empirical distributions and is easier
    # to interpret as a response curve across paired sampling scales.
    return max(float((k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).item()), 0.0)


def sketch_distance_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    bandwidth_sq: float | None = None,
) -> tuple[dict[str, float], float]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("sketch arrays must be matching matrices")
    mean = reference.mean(axis=0, dtype=np.float64)
    std = reference.std(axis=0, dtype=np.float64)
    ref = ((reference - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    cand = ((candidate - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    if bandwidth_sq is None:
        distances = torch.cdist(torch.from_numpy(ref), torch.from_numpy(ref)).square()
        upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
        positive = upper[upper > 0]
        bandwidth_sq = float(positive.median().item()) if positive.numel() else 1.0
    sorted_ref = np.sort(ref, axis=0)
    sorted_cand = np.sort(cand, axis=0)
    sliced_wasserstein = float(np.mean(np.abs(sorted_ref - sorted_cand)))
    ref_cov = np.cov(ref, rowvar=False)
    cand_cov = np.cov(cand, rowvar=False)
    covariance_relative = float(
        np.linalg.norm(cand_cov - ref_cov) / max(np.linalg.norm(ref_cov), 1e-30)
    )
    metrics = {
        "sketch_sliced_wasserstein": sliced_wasserstein,
        "sketch_rbf_mmd_squared": rbf_mmd_squared(ref, cand, bandwidth_sq),
        "sketch_mean_shift_l2": float(np.linalg.norm(cand.mean(0) - ref.mean(0))),
        "sketch_covariance_relative_frobenius": covariance_relative,
        "sketch_bandwidth_sq": float(bandwidth_sq),
    }
    return metrics, float(bandwidth_sq)


def latent_space_rows(
    *,
    space: str,
    directory: Path,
    scales: tuple[float, ...],
    samples: int,
    world_size: int,
    test_mask: np.ndarray,
    ridge_ratio: float,
    projection: torch.Tensor,
    metric_batch: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    reference = load_ordered_rank_arrays(
        directory, "real", samples=samples, world_size=world_size
    )
    reference_sketch = fixed_random_sketch(
        reference, projection, batch_size=metric_batch, device=device
    )
    bandwidth_sq = None
    rows = []
    for scale in scales:
        candidate = load_ordered_rank_arrays(
            directory, scale_key(scale), samples=samples, world_size=world_size
        )
        metrics = diagonal_lda_metrics(
            reference, candidate, test_mask, ridge_ratio=ridge_ratio
        )
        candidate_sketch = fixed_random_sketch(
            candidate, projection, batch_size=metric_batch, device=device
        )
        sketch_metrics, bandwidth_sq = sketch_distance_metrics(
            reference_sketch, candidate_sketch, bandwidth_sq=bandwidth_sq
        )
        rows.append({"space": space, "scale": scale, **metrics, **sketch_metrics})
        del candidate, candidate_sketch
        gc.collect()
    del reference, reference_sketch
    gc.collect()
    return rows


def image_metric_rows(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    scales: tuple[float, ...],
    samples: int,
    world_size: int,
    test_mask: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    source = load_ordered_rank_arrays(
        output_dir / "inception", "source", samples=samples, world_size=world_size
    ).astype(np.float32)
    reconstruction = load_ordered_rank_arrays(
        output_dir / "inception", "real", samples=samples, world_size=world_size
    ).astype(np.float32)
    source_stats = reconstruction_stats = reference_stats = None
    baseline: dict[str, float] = {}
    if not args.skip_fid:
        reference_stats = load_reference_statistics(
            args.fid_reference.expanduser().resolve(), "2048"
        )
        source_stats = feature_statistics(source)
        reconstruction_stats = feature_statistics(reconstruction)
        baseline = {
            "fid_source_to_official": fid_between_statistics(
                source_stats, reference_stats
            ),
            "fid_reconstruction_to_official": fid_between_statistics(
                reconstruction_stats, reference_stats
            ),
            "fid_reconstruction_to_source": fid_between_statistics(
                reconstruction_stats, source_stats
            ),
        }
    rows = []
    train_mask = ~test_mask
    for index, scale in enumerate(scales):
        candidate = load_ordered_rank_arrays(
            output_dir / "inception",
            scale_key(scale),
            samples=samples,
            world_size=world_size,
        ).astype(np.float32)
        weight, intercept, ridge = fit_feature_probe(
            reconstruction, candidate, train_mask, args.ridge_ratio
        )
        p_scores = feature_probe_scores(
            reconstruction[test_mask], weight, intercept
        )
        q_scores = feature_probe_scores(candidate[test_mask], weight, intercept)
        labels_auc = np.concatenate(
            [np.zeros(p_scores.size, dtype=np.int8), np.ones(q_scores.size, dtype=np.int8)]
        )
        auc = float(roc_auc_score(labels_auc, np.concatenate([p_scores, q_scores])))
        ci_low, ci_high = bootstrap_paired_auc(
            p_scores,
            q_scores,
            args.bootstrap_repeats,
            args.seed + 20_000 + index,
        )
        with torch.inference_mode():
            kid_reconstruction = float(
                polynomial_mmd_unbiased(
                    torch.from_numpy(reconstruction).to(device=device),
                    torch.from_numpy(candidate).to(device=device),
                ).cpu().item()
            )
            kid_source = float(
                polynomial_mmd_unbiased(
                    torch.from_numpy(source).to(device=device),
                    torch.from_numpy(candidate).to(device=device),
                ).cpu().item()
            )
        row: dict[str, Any] = {
            "space": "decoded_inception",
            "scale": scale,
            "auc": auc,
            "auc_separability": 0.5 + abs(auc - 0.5),
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            "ridge": ridge,
            "kid_to_reconstruction": kid_reconstruction,
            "kid_to_source": kid_source,
        }
        if reference_stats is not None and source_stats is not None and reconstruction_stats is not None:
            candidate_stats = feature_statistics(candidate)
            row.update(
                {
                    "fid_to_official": fid_between_statistics(
                        candidate_stats, reference_stats
                    ),
                    "fid_to_reconstruction": fid_between_statistics(
                        candidate_stats, reconstruction_stats
                    ),
                    "fid_to_source": fid_between_statistics(
                        candidate_stats, source_stats
                    ),
                }
            )
        rows.append(row)
        del candidate
        gc.collect()
    return rows, baseline


def plot_summary(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    colors = {"latent": "#2a6fbb", "roundtrip": "#c4513a", "decoded_inception": "#2d8a57"}
    for space in ("latent", "roundtrip", "decoded_inception"):
        part = frame[frame["space"] == space].sort_values("scale")
        if not part.empty:
            axes[0].plot(
                part["scale"],
                part["auc_separability"],
                marker="o",
                linewidth=2.2,
                label=space,
                color=colors[space],
            )
    axes[0].set_ylabel("C2ST separability (0.5 is best)")
    for space in ("latent", "roundtrip"):
        part = frame[frame["space"] == space].sort_values("scale")
        axes[1].plot(
            part["scale"],
            part["sketch_sliced_wasserstein"],
            marker="o",
            linewidth=2.2,
            label=space,
            color=colors[space],
        )
    axes[1].set_ylabel("Projected sliced Wasserstein")
    decoded = frame[frame["space"] == "decoded_inception"].sort_values("scale")
    if "fid_to_official" in decoded and decoded["fid_to_official"].notna().any():
        axes[2].plot(
            decoded["scale"], decoded["fid_to_official"], "o-", label="to official"
        )
        axes[2].plot(
            decoded["scale"],
            decoded["fid_to_reconstruction"],
            "s-",
            label="to D(E(x))",
        )
        axes[2].set_ylabel("Decoded FID")
    else:
        axes[2].plot(
            decoded["scale"],
            decoded["kid_to_reconstruction"] * 1000.0,
            "o-",
            label="to D(E(x))",
        )
        axes[2].set_ylabel("Decoded KID x1000")
    for axis in axes:
        axis.axvline(1.0, color="#222222", linestyle="--", alpha=0.55, label="full s=1")
        axis.axvline(1.78, color="#7c4aa5", linestyle=":", alpha=0.7, label="official s=1.78")
        axis.set_xlabel("Code guidance scale s")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("RAEv2 Internal Guidance: Scale Response Across Measurement Spaces")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def compute_all_metrics(
    *,
    args: argparse.Namespace,
    config: Any,
    scales: tuple[float, ...],
    output_dir: Path,
    test_mask: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    phase = "metrics"
    if marker_exists(output_dir, phase):
        return
    if rank != 0:
        dist.barrier()
        dist.barrier()
        return
    latent_dim = math.prod(int(value) for value in config.misc.latent_size)
    generator = np.random.default_rng(args.seed + 70_001)
    projection_np = generator.standard_normal(
        (latent_dim, args.sketch_dim), dtype=np.float32
    ) / math.sqrt(latent_dim)
    projection = torch.from_numpy(projection_np).to(device=device)
    rows = []
    rows.extend(
        latent_space_rows(
            space="latent",
            directory=output_dir / "latents",
            scales=scales,
            samples=args.samples,
            world_size=world_size,
            test_mask=test_mask,
            ridge_ratio=args.ridge_ratio,
            projection=projection,
            metric_batch=args.metric_batch,
            device=device,
        )
    )
    rows.extend(
        latent_space_rows(
            space="roundtrip",
            directory=output_dir / "roundtrip",
            scales=scales,
            samples=args.samples,
            world_size=world_size,
            test_mask=test_mask,
            ridge_ratio=args.ridge_ratio,
            projection=projection,
            metric_batch=args.metric_batch,
            device=device,
        )
    )
    image_rows, baseline = image_metric_rows(
        args=args,
        output_dir=output_dir,
        scales=scales,
        samples=args.samples,
        world_size=world_size,
        test_mask=test_mask,
        device=device,
    )
    rows.extend(image_rows)
    frame = pd.DataFrame(rows).sort_values(["space", "scale"])
    frame.to_csv(output_dir / "scale_response_metrics.csv", index=False)
    atomic_write_json(output_dir / "image_baseline_metrics.json", baseline)
    plot_summary(frame, output_dir / "scale_response_curves.png")
    complete_phase(
        output_dir,
        phase,
        rank=rank,
        world_size=world_size,
        rows=len(frame),
    )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["metric_spaces"] = ["latent", "roundtrip", "decoded_inception"]
    manifest["roundtrip_definition"] = (
        "E(clamp(D(z_s))) versus E(clamp(D(E(x)))); clamped decoder images "
        "are stored as float16 without uint8 quantization"
    )
    manifest["measurement_warning"] = (
        "A scale ranking reversal after round-trip supports an E-compose-D effect; "
        "decoded Inception rankings still combine decoder and feature-metric geometry."
    )
    atomic_write_json(manifest_path, manifest)
    print(frame.to_string(index=False), flush=True)


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    scales = normalized_scales(args.scales)
    positive_ints = (
        args.samples,
        args.per_rank_batch,
        args.decode_batch,
        args.encode_batch,
        args.log_every_batches,
        args.sketch_dim,
        args.metric_batch,
    )
    if any(value <= 0 for value in positive_ints):
        raise ValueError("sample, batch, logging, and sketch values must be positive")

    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
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
    checkpoint_path = args.checkpoint.expanduser().resolve()
    labels = build_requested_labels(args.samples, int(config.misc.num_classes))
    test_mask = class_group_split(labels, args.test_fraction, args.seed + 17)
    if np.intersect1d(np.unique(labels[~test_mask]), np.unique(labels[test_mask])).size:
        raise RuntimeError("ImageNet classes leak across the train/test split")
    if rank == 0:
        source_rows = select_matching_imagenet_rows(
            args.parquet_data_path, labels, args.seed + 31
        )
    else:
        source_rows = np.empty(args.samples, dtype=np.int64)
    source_tensor = torch.from_numpy(source_rows).to(device=device)
    dist.broadcast(source_tensor, src=0)
    source_rows = source_tensor.cpu().numpy().astype(np.int64, copy=True)
    if rank == 0:
        write_initial_manifest(
            output_dir=output_dir,
            args=args,
            scales=scales,
            config=config,
            checkpoint_path=checkpoint_path,
            labels=labels,
            source_rows=source_rows,
            test_mask=test_mask,
            world_size=world_size,
        )
    dist.barrier()

    local_ids = local_ids_for_rank(args.samples, rank, world_size)
    collect_real_latents(
        args=args,
        config=config,
        output_dir=output_dir,
        local_ids=local_ids,
        labels=labels,
        source_rows=source_rows,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    sample_scale_endpoints(
        args=args,
        config=config,
        scales=scales,
        output_dir=output_dir,
        local_ids=local_ids,
        labels=labels,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    decode_all_conditions(
        args=args,
        config=config,
        scales=scales,
        output_dir=output_dir,
        local_ids=local_ids,
        labels=labels,
        source_rows=source_rows,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    encode_roundtrips(
        args=args,
        config=config,
        scales=scales,
        output_dir=output_dir,
        local_ids=local_ids,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    compute_all_metrics(
        args=args,
        config=config,
        scales=scales,
        output_dir=output_dir,
        test_mask=test_mask,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
