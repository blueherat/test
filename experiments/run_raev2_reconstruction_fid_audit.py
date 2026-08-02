"""Measure the RAEv2 reconstruction FID floor on matched ImageNet samples.

This audit reuses ``D(E(x))`` Inception features from a completed decoded
distribution audit and extracts features for the exact source images again.
It separates three quantities that are otherwise easy to conflate:

* finite-sample/reference mismatch: source images vs official ImageNet stats;
* total reconstruction FID: reconstructions vs official ImageNet stats;
* decoder distribution shift: reconstructions vs their matched source images.

No encoder, decoder, or Stage-2 generator is loaded by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_training_core import DeterministicImageNetPacked
from experiments.run_raev2_decoded_distribution_audit import (
    feature_statistics,
    fid_between_statistics,
    load_reference_statistics,
    time_suffix,
)
from experiments.run_raev2_distribution_auc import select_matching_imagenet_rows

EXPECTED_PROTOCOL = "raev2_decoded_distribution_audit_v1"
AUDIT_PROTOCOL = "raev2_reconstruction_fid_audit_v1"
RECONSTRUCTION_KEY = f"feat_p_{time_suffix(0.0)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoded-run",
        type=Path,
        action="append",
        required=True,
        help="Completed decoded-distribution run; repeat for multiple seeds.",
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every-batches", type=int, default=20)
    return parser.parse_args()


def load_decoded_protocol(
    decoded_run: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Load and validate the sample protocol from decoded feature shards."""

    manifest_path = decoded_run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"decoded manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError(
            f"expected protocol {EXPECTED_PROTOCOL!r}, got {manifest.get('protocol')!r}"
        )
    if manifest.get("inception_feature") != "2048":
        raise ValueError("the reconstruction FID audit requires 2048-d Inception features")

    samples = int(manifest["samples"])
    source_world_size = int(manifest["world_size"])
    shard_ids: list[np.ndarray] = []
    shard_labels: list[np.ndarray] = []
    for rank in range(source_world_size):
        shard_path = decoded_run / f"decoded_features_rank{rank:02d}.npz"
        if not shard_path.is_file():
            raise FileNotFoundError(f"decoded feature shard not found: {shard_path}")
        with np.load(shard_path) as shard:
            if RECONSTRUCTION_KEY not in shard.files:
                raise KeyError(f"{RECONSTRUCTION_KEY!r} missing from {shard_path}")
            shard_ids.append(np.asarray(shard["ids"], dtype=np.int64))
            shard_labels.append(np.asarray(shard["labels"], dtype=np.int64))

    ids = np.concatenate(shard_ids)
    labels = np.concatenate(shard_labels)
    order = np.argsort(ids)
    expected_ids = np.arange(samples, dtype=np.int64)
    if ids.size != samples or not np.array_equal(ids[order], expected_ids):
        raise RuntimeError("decoded shards do not contain every sample exactly once")
    labels = labels[order]
    if labels.shape != (samples,) or np.any((labels < 0) | (labels >= 1000)):
        raise RuntimeError("decoded ImageNet labels are invalid")
    return manifest, expected_ids, labels


def load_reconstruction_features(
    decoded_run: Path, source_world_size: int, samples: int
) -> np.ndarray:
    """Load only clean-reconstruction features instead of every saved condition."""

    ids_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for rank in range(source_world_size):
        with np.load(decoded_run / f"decoded_features_rank{rank:02d}.npz") as shard:
            ids_parts.append(np.asarray(shard["ids"], dtype=np.int64))
            feature_parts.append(
                np.asarray(shard[RECONSTRUCTION_KEY], dtype=np.float32)
            )
    ids = np.concatenate(ids_parts)
    features = np.concatenate(feature_parts, axis=0)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(samples, dtype=np.int64)):
        raise RuntimeError("reconstruction features are not aligned to sample IDs")
    features = features[order]
    if features.shape != (samples, 2048) or not np.isfinite(features).all():
        raise RuntimeError(f"invalid reconstruction feature shape: {features.shape}")
    return features


def load_raw_feature_shards(
    audit_dir: Path, world_size: int, samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for rank in range(world_size):
        shard_path = audit_dir / f"raw_features_rank{rank:02d}.npz"
        with np.load(shard_path) as shard:
            ids_parts.append(np.asarray(shard["ids"], dtype=np.int64))
            labels_parts.append(np.asarray(shard["labels"], dtype=np.int64))
            feature_parts.append(np.asarray(shard["features"], dtype=np.float32))
    ids = np.concatenate(ids_parts)
    labels = np.concatenate(labels_parts)
    features = np.concatenate(feature_parts, axis=0)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(samples, dtype=np.int64)):
        raise RuntimeError("raw feature shards do not contain every sample exactly once")
    features = features[order]
    if features.shape != (samples, 2048) or not np.isfinite(features).all():
        raise RuntimeError(f"invalid raw feature shape: {features.shape}")
    return labels[order], features, ids[order]


def paired_feature_metrics(
    source: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    if source.shape != reconstruction.shape or source.ndim != 2:
        raise ValueError("paired feature arrays must be matching matrices")
    difference = reconstruction.astype(np.float64) - source.astype(np.float64)
    numerator = np.sum(source.astype(np.float64) * reconstruction.astype(np.float64), axis=1)
    denominator = np.linalg.norm(source, axis=1) * np.linalg.norm(reconstruction, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    return {
        "paired_feature_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "paired_feature_cosine_mean": float(np.mean(cosine)),
        "paired_feature_cosine_std": float(np.std(cosine, ddof=1)),
    }


def compute_audit_metrics(
    source: np.ndarray,
    reconstruction: np.ndarray,
    official_reference: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute the three non-additive FIDs and matched feature diagnostics."""

    source_stats = feature_statistics(source)
    reconstruction_stats = feature_statistics(reconstruction)
    result = {
        "fid_source_to_official": fid_between_statistics(
            source_stats, official_reference
        ),
        "fid_reconstruction_to_official": fid_between_statistics(
            reconstruction_stats, official_reference
        ),
        "fid_reconstruction_to_matched_source": fid_between_statistics(
            reconstruction_stats, source_stats
        ),
    }
    result["fid_official_delta_reconstruction_minus_source"] = (
        result["fid_reconstruction_to_official"]
        - result["fid_source_to_official"]
    )
    result.update(paired_feature_metrics(source, reconstruction))
    return result


def extract_source_features(
    *,
    dataset: DeterministicImageNetPacked,
    extractor: torch.nn.Module,
    source_rows: np.ndarray,
    expected_labels: np.ndarray,
    local_ids: np.ndarray,
    batch_size: int,
    log_every_batches: int,
    rank: int,
    device: torch.device,
) -> np.ndarray:
    features: list[np.ndarray] = []
    total_batches = (local_ids.size + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, local_ids.size, batch_size)):
            ids = local_ids[start : start + batch_size]
            images = []
            for sample_id in ids.tolist():
                image, actual_label, _ = dataset[int(source_rows[sample_id])]
                if int(actual_label) != int(expected_labels[sample_id]):
                    raise RuntimeError(
                        f"ImageNet label mismatch for sample {sample_id}: "
                        f"expected {expected_labels[sample_id]}, got {actual_label}"
                    )
                images.append(image)
            image_batch = torch.stack(images).to(device=device)
            uint8_batch = image_batch.clamp(0, 1).mul(255).to(torch.uint8)
            feature = extractor(uint8_batch)[0].float().cpu().numpy()
            features.append(feature)
            if rank == 0 and (
                (batch_index + 1) % log_every_batches == 0
                or batch_index + 1 == total_batches
            ):
                print(
                    f"[raw features] batches {batch_index + 1}/{total_batches}",
                    flush=True,
                )
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.log_every_batches <= 0:
        raise ValueError("batch sizes and logging interval must be positive")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    packed_path = args.packed_data_path.expanduser().resolve()
    parquet_path = args.parquet_data_path.expanduser().resolve()
    reference_path = args.fid_reference.expanduser().resolve()
    official_reference = load_reference_statistics(reference_path, "2048")
    dataset = DeterministicImageNetPacked(
        packed_path,
        split="train",
        image_size=256,
        horizontal_flip=False,
    )
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).to(device)

    for decoded_run_arg in args.decoded_run:
        decoded_run = decoded_run_arg.expanduser().resolve()
        manifest, _, labels = load_decoded_protocol(decoded_run)
        samples = int(manifest["samples"])
        seed = int(manifest["seed"])
        source_world_size = int(manifest["world_size"])
        audit_dir = decoded_run / "reconstruction_fid_audit"
        if rank == 0:
            audit_dir.mkdir(parents=True, exist_ok=True)
            source_rows = select_matching_imagenet_rows(
                parquet_path, labels, seed + 31
            )
        else:
            source_rows = np.empty(samples, dtype=np.int64)
        dist.barrier()
        rows_tensor = torch.from_numpy(source_rows).to(device=device)
        dist.broadcast(rows_tensor, src=0)
        source_rows = rows_tensor.cpu().numpy().astype(np.int64, copy=True)

        local_ids = np.arange(rank, samples, world_size, dtype=np.int64)
        local_features = extract_source_features(
            dataset=dataset,
            extractor=extractor,
            source_rows=source_rows,
            expected_labels=labels,
            local_ids=local_ids,
            batch_size=args.batch_size,
            log_every_batches=args.log_every_batches,
            rank=rank,
            device=device,
        )
        np.savez(
            audit_dir / f"raw_features_rank{rank:02d}.npz",
            ids=local_ids,
            labels=labels[local_ids],
            source_rows=source_rows[local_ids],
            features=local_features,
        )
        dist.barrier()

        if rank == 0:
            ordered_labels, source_features, ordered_ids = load_raw_feature_shards(
                audit_dir, world_size, samples
            )
            if not np.array_equal(ordered_ids, np.arange(samples)):
                raise RuntimeError("raw feature IDs changed during collection")
            if not np.array_equal(ordered_labels, labels):
                raise RuntimeError("raw feature labels differ from decoded protocol")
            reconstruction = load_reconstruction_features(
                decoded_run, source_world_size, samples
            )
            metrics = compute_audit_metrics(
                source_features, reconstruction, official_reference
            )

            existing_summary = pd.read_csv(
                decoded_run / "decoded_distribution_summary.csv"
            )
            existing_t0 = existing_summary.loc[
                np.isclose(existing_summary["requested_time"], 0.0), "fid_real_p"
            ]
            if len(existing_t0) != 1:
                raise RuntimeError("decoded summary does not contain exactly one t=0 row")
            existing_fid = float(existing_t0.iloc[0])
            reproduced_fid = metrics["fid_reconstruction_to_official"]
            if not np.isclose(existing_fid, reproduced_fid, rtol=0.0, atol=1e-8):
                raise RuntimeError(
                    "reconstruction FID did not reproduce the decoded audit: "
                    f"existing={existing_fid}, reproduced={reproduced_fid}"
                )

            row = {
                "decoded_run": str(decoded_run),
                "seed": seed,
                "samples": samples,
                **metrics,
                "existing_fid_reproduced_absolute_error": abs(
                    existing_fid - reproduced_fid
                ),
            }
            pd.DataFrame([row]).to_csv(
                audit_dir / "reconstruction_fid_audit.csv", index=False
            )
            audit_manifest = {
                "protocol": AUDIT_PROTOCOL,
                "inference_only": True,
                "decoded_run": str(decoded_run),
                "source_protocol": manifest["protocol"],
                "samples": samples,
                "seed": seed,
                "world_size": world_size,
                "source_split": "ImageNet-1k train",
                "image_size": 256,
                "horizontal_flip": False,
                "inception_feature": "2048",
                "fid_reference": str(reference_path),
                "interpretation": {
                    "fid_source_to_official": (
                        "finite-sample and train/reference mismatch floor"
                    ),
                    "fid_reconstruction_to_official": (
                        "total reconstruction FID under the same official reference"
                    ),
                    "fid_reconstruction_to_matched_source": (
                        "distribution shift introduced by E then D on the matched sample"
                    ),
                    "non_additivity_warning": (
                        "FID values are not additive; the official-reference delta is "
                        "descriptive and is not an isolated decoder FID"
                    ),
                },
            }
            (audit_dir / "manifest.json").write_text(
                json.dumps(audit_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(pd.DataFrame([row]).to_string(index=False), flush=True)
        dist.barrier()

    dataset.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
