"""Decode paired RAEv2 IG-window endpoints for small-sample quality screening."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torchvision.models import Inception_V3_Weights, inception_v3


ROOT = Path(__file__).resolve().parents[1]
RAEV2_SRC = ROOT / "external" / "RAEv2" / "src"
for path in (RAEV2_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.analyze_raev2_ig_window_response import (  # noqa: E402
    load_manifest,
    validate_paired_protocol,
)
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat  # noqa: E402
from experiments.run_raev2_distribution_auc import autocast_context, load_config  # noqa: E402
from experiments.run_raev2_scale_response_study import (  # noqa: E402
    atomic_save_npy,
    scale_key,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_condition(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("conditions must use NAME=/path/to/run")
    name = name.strip()
    if not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("condition names may contain letters, numbers, _ and -")
    return name, Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=parse_condition, action="append", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
    )
    parser.add_argument("--decode-batch", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
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


def local_ids(samples: int, rank: int, world_size: int) -> np.ndarray:
    return np.arange(rank, samples, world_size, dtype=np.int64)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels have incompatible shapes")
    tensor = torch.from_numpy(logits.astype(np.float32, copy=False))
    targets = torch.from_numpy(labels.astype(np.int64, copy=False))
    log_probs = tensor.log_softmax(dim=1)
    probs = log_probs.exp()
    true_log_prob = log_probs.gather(1, targets[:, None]).squeeze(1)
    marginal = probs.mean(dim=0).clamp_min(1e-30)
    inception_score = torch.exp((probs * (log_probs - marginal.log())).sum(dim=1).mean())
    return {
        "top1_accuracy": float(tensor.argmax(dim=1).eq(targets).float().mean()),
        "true_class_log_probability": float(true_log_prob.mean()),
        "maximum_probability": float(probs.max(dim=1).values.mean()),
        "predictive_entropy": float(-(probs * log_probs).sum(dim=1).mean()),
        "inception_score_screening": float(inception_score),
    }


def classifier_forward(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False, antialias=True
    )
    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device).view(1, 3, 1, 1)
    return model((resized - mean) / std)


def load_global_shards(
    directory: Path,
    condition: str,
    *,
    samples: int,
    world_size: int,
) -> np.ndarray:
    result: np.ndarray | None = None
    for rank in range(world_size):
        ids = local_ids(samples, rank, world_size)
        local = np.load(directory / condition / f"rank{rank:02d}.npy", allow_pickle=False)
        if len(local) != len(ids):
            raise RuntimeError(f"unexpected shard rows for {condition}, rank {rank}")
        if result is None:
            result = np.empty((samples, *local.shape[1:]), dtype=local.dtype)
        result[ids] = local
    if result is None:
        raise RuntimeError(f"no shards found for {condition}")
    return result


def main() -> None:
    install_raev2_decoder_config_compat()
    args = parse_args()
    if args.decode_batch <= 0:
        raise ValueError("decode batch must be positive")
    condition_map = dict(args.condition)
    if len(condition_map) != len(args.condition) or args.baseline_name not in condition_map:
        raise ValueError("condition names must be unique and include the baseline name")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.expanduser().resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.expanduser().resolve())

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    resolved = {name: path.expanduser().resolve() for name, path in args.condition}
    reference = load_manifest(resolved[args.baseline_name])
    protocol = np.load(resolved[args.baseline_name] / "sample_protocol.npz")
    labels = protocol["labels"].astype(np.int64)
    samples = int(reference["samples"])
    if len(labels) != samples or int(reference["world_size"]) != world_size:
        raise ValueError("decode world size and sample protocol must match endpoint generation")
    for name, directory in resolved.items():
        manifest = load_manifest(directory)
        validate_paired_protocol(reference, manifest)
        candidate_protocol = np.load(directory / "sample_protocol.npz")
        for key in ("sample_ids", "labels"):
            if not np.array_equal(protocol[key], candidate_protocol[key]):
                raise ValueError(f"condition {name} does not share {key}")

    config = load_config(args.config.expanduser().resolve())
    decoder = instantiate_from_config(config.stage_1)
    del decoder.encoder
    decoder = decoder.to(device).eval().requires_grad_(False)
    feature_extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).to(device).eval()
    classifier = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    classifier.requires_grad_(False)

    ids = local_ids(samples, rank, world_size)
    for name, directory in resolved.items():
        manifest = load_manifest(directory)
        scales = tuple(float(value) for value in manifest["scales"])
        if len(scales) != 1:
            raise ValueError(f"condition {name} must contain one scale")
        key = scale_key(scales[0])
        latents = np.load(
            directory / "latents" / f"{key}_rank{rank:02d}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        image_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        logit_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(ids), args.decode_batch):
                latent = torch.from_numpy(
                    np.array(
                        latents[start : start + args.decode_batch],
                        dtype=np.float32,
                        copy=True,
                    )
                ).to(device)
                with autocast_context(args.precision):
                    images = decoder.decode(latent).float().clamp(0, 1)
                image_parts.append(images.cpu().permute(0, 2, 3, 1).to(torch.float16).numpy())
                feature_parts.append(
                    feature_extractor(images.mul(255).to(torch.uint8))[0]
                    .float()
                    .cpu()
                    .numpy()
                )
                logit_parts.append(classifier_forward(classifier, images).float().cpu().numpy())
        for subdir, values in (
            ("images", image_parts),
            ("features", feature_parts),
            ("logits", logit_parts),
        ):
            atomic_save_npy(
                output_dir / subdir / name / f"rank{rank:02d}.npy",
                np.concatenate(values, axis=0),
            )
        dist.barrier()

    if rank == 0:
        baseline_features = load_global_shards(
            output_dir / "features", args.baseline_name, samples=samples, world_size=world_size
        ).astype(np.float32)
        rows: list[dict[str, Any]] = []
        for name in condition_map:
            features = load_global_shards(
                output_dir / "features", name, samples=samples, world_size=world_size
            ).astype(np.float32)
            logits = load_global_shards(
                output_dir / "logits", name, samples=samples, world_size=world_size
            ).astype(np.float32)
            delta = features - baseline_features
            rows.append(
                {
                    "condition": name,
                    **classification_metrics(logits, labels),
                    "feature_delta_rms": float(np.sqrt(np.mean(np.square(delta), axis=1)).mean()),
                }
            )
        frame = pd.DataFrame(rows)
        baseline_row = frame[frame["condition"].eq(args.baseline_name)].iloc[0]
        for metric in ("top1_accuracy", "true_class_log_probability", "maximum_probability"):
            frame[f"delta_{metric}"] = frame[metric] - float(baseline_row[metric])
        frame.to_csv(output_dir / "quality_screening.csv", index=False)
        manifest = {
            "format_version": 1,
            "scope": "small_sample_paired_window_quality_screening",
            "warning": "N=64 classification and feature metrics are screening signals, not FID claims.",
            "samples": samples,
            "conditions": {name: str(path) for name, path in resolved.items()},
            "baseline": args.baseline_name,
            "precision": args.precision,
            "world_size": world_size,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(frame.to_string(index=False))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
