#!/usr/bin/env python3
"""Blindly score CFG-Rejection EDM2 samples with a local ImageNet classifier.

The classifier is evaluated before its predictions are joined with the stored
CFG-Rejection signals.  Symmetric evidence tails are selected within each
ImageNet class, which prevents class-dependent signal scales from creating a
spurious global tail comparison.  CFG-Rejection treats the low-ASD tail as
suspicious.  High-minus-low changes therefore have the following quality
interpretation when its ranking works:

* lower target-class probability is worse;
* lower top-1 accuracy is worse;
* higher predictive entropy is worse.

This script never downloads weights.  It requires TorchVision's official
ConvNeXt-Tiny ImageNet-1K V1 checkpoint to be present in the local torch hub
cache (or at an explicitly supplied ``--weights`` path).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


EVIDENCE_METRICS = (
    "official_notebook_metric_tau5",
    "denoiser_asd_tau5",
    "denoiser_asd_full",
    "score_asd_tau5",
    "score_asd_full",
)
MODEL_WEIGHTS = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
MODEL_NAME = "torchvision_convnext_tiny_imagenet1k_v1"
NUM_IMAGENET_CLASSES = 1_000


@dataclass(frozen=True)
class InputRecord:
    class_id: int
    seed: int
    image_path: Path
    signal_path: Path
    evidence: dict[str, float]

    @property
    def key(self) -> tuple[int, int]:
        return self.class_id, self.seed


@dataclass(frozen=True)
class ClassifierScore:
    target_probability: float
    top1_class_id: int
    top1_probability: float
    top1_correct: bool
    entropy_nats: float
    normalized_entropy: float


class ImageRecordDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: Sequence[InputRecord], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing reproduction manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    if manifest.get("experiment") != "cfg_rejection_edm2_reproduction":
        raise ValueError(
            "unexpected manifest experiment: "
            f"{manifest.get('experiment')!r} in {path}"
        )
    return manifest, sha256_file(path)


def _scalar(payload: Any, name: str, path: Path) -> float:
    if name not in payload:
        raise KeyError(f"signal {path} is missing {name!r}")
    value = np.asarray(payload[name])
    if value.ndim != 0:
        raise ValueError(f"signal {path} field {name!r} must be scalar, got {value.shape}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"signal {path} field {name!r} is non-finite: {result}")
    return result


def load_records(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    allow_incomplete: bool,
) -> tuple[list[InputRecord], dict[str, Any]]:
    signals_dir = run_dir / "signals"
    images_dir = run_dir / "images"
    signal_paths = sorted(signals_dir.glob("class_*/*.npz"))
    if not signal_paths:
        raise FileNotFoundError(f"no signal NPZ files found under {signals_dir}")

    declared_count = manifest.get("sample_count")
    if (
        declared_count is not None
        and int(declared_count) != len(signal_paths)
        and not allow_incomplete
    ):
        raise RuntimeError(
            "reproduction output is incomplete: "
            f"signal_files={len(signal_paths)}, declared={int(declared_count)}. "
            "Pass --allow-incomplete only for a clearly labeled partial diagnostic."
        )

    records: list[InputRecord] = []
    seen: set[tuple[int, int]] = set()
    for signal_path in signal_paths:
        with np.load(signal_path, allow_pickle=False) as payload:
            class_id = int(_scalar(payload, "class_id", signal_path))
            seed = int(_scalar(payload, "seed", signal_path))
            evidence = {
                metric: _scalar(payload, metric, signal_path)
                for metric in EVIDENCE_METRICS
            }
        if not 0 <= class_id < NUM_IMAGENET_CLASSES:
            raise ValueError(f"invalid ImageNet class {class_id} in {signal_path}")
        key = (class_id, seed)
        if key in seen:
            raise ValueError(f"duplicate class/seed signal key: {key}")
        seen.add(key)
        image_path = images_dir / f"class_{class_id:04d}" / f"{seed:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"signal has no matching PNG: {image_path}")
        records.append(InputRecord(class_id, seed, image_path, signal_path, evidence))

    records.sort(key=lambda record: (record.class_id, record.seed, str(record.image_path)))
    manifest_classes = {int(value) for value in manifest.get("class_ids", [])}
    manifest_seeds = {int(value) for value in manifest.get("seeds", [])}
    expected_keys = {
        (class_id, seed)
        for class_id in manifest_classes
        for seed in manifest_seeds
    }
    observed_keys = {record.key for record in records}
    unexpected = sorted(observed_keys - expected_keys) if expected_keys else []
    missing = sorted(expected_keys - observed_keys) if expected_keys else []
    count_matches = declared_count is None or int(declared_count) == len(records)
    complete = count_matches and not unexpected and not missing
    if unexpected:
        raise ValueError(
            f"signals contain {len(unexpected)} class/seed keys absent from the manifest; "
            f"first={unexpected[:5]}"
        )
    if not complete and not allow_incomplete:
        raise RuntimeError(
            "reproduction output is incomplete: "
            f"records={len(records)}, declared={declared_count}, "
            f"missing_keys={len(missing)}. Pass --allow-incomplete only for a "
            "clearly labeled partial diagnostic."
        )
    audit = {
        "allow_incomplete": bool(allow_incomplete),
        "complete_against_manifest": bool(complete),
        "declared_sample_count": None if declared_count is None else int(declared_count),
        "observed_sample_count": len(records),
        "expected_cartesian_count": len(expected_keys),
        "missing_key_count": len(missing),
        "missing_key_preview": [list(key) for key in missing[:20]],
        "unexpected_key_count": len(unexpected),
    }
    return records, audit


def default_cached_weights_path() -> Path:
    filename = Path(urlparse(MODEL_WEIGHTS.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def load_local_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    if not weights_path.is_file():
        raise FileNotFoundError(
            "local ConvNeXt-Tiny weights are missing; refusing to download: "
            f"{weights_path}"
        )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model = convnext_tiny(weights=None)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval().requires_grad_(False)


def configure_reproducibility(seed: int, allow_tf32: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def score_records(
    records: Sequence[InputRecord],
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[ClassifierScore]:
    dataset = ImageRecordDataset(records, MODEL_WEIGHTS.transforms())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    scores: list[ClassifierScore | None] = [None] * len(records)
    entropy_normalizer = math.log(NUM_IMAGENET_CLASSES)
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).float()
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        target_ids = torch.tensor(
            [records[int(index)].class_id for index in indices],
            dtype=torch.long,
            device=device,
        )
        target_probabilities = probabilities.gather(1, target_ids[:, None]).squeeze(1)
        top1_probabilities, top1_ids = probabilities.max(dim=1)
        entropies = -(probabilities * log_probabilities).sum(dim=1)
        for offset, raw_index in enumerate(indices.tolist()):
            entropy = float(entropies[offset].item())
            top1_id = int(top1_ids[offset].item())
            scores[int(raw_index)] = ClassifierScore(
                target_probability=float(target_probabilities[offset].item()),
                top1_class_id=top1_id,
                top1_probability=float(top1_probabilities[offset].item()),
                top1_correct=top1_id == records[int(raw_index)].class_id,
                entropy_nats=entropy,
                normalized_entropy=entropy / entropy_normalizer,
            )
    if any(score is None for score in scores):
        raise RuntimeError("classifier did not return a score for every input image")
    return [score for score in scores if score is not None]


def select_symmetric_tails(
    records: Sequence[InputRecord],
    *,
    tail_fraction: float,
) -> tuple[dict[str, dict[tuple[int, int], str]], dict[str, Any]]:
    by_class: dict[int, list[InputRecord]] = {}
    for record in records:
        by_class.setdefault(record.class_id, []).append(record)

    assignments: dict[str, dict[tuple[int, int], str]] = {}
    audit: dict[str, Any] = {}
    for metric in EVIDENCE_METRICS:
        metric_assignment = {record.key: "middle" for record in records}
        class_audit: dict[str, Any] = {}
        for class_id, class_records in sorted(by_class.items()):
            ordered = sorted(
                class_records,
                key=lambda record: (
                    record.evidence[metric],
                    record.seed,
                    str(record.image_path),
                ),
            )
            tail_count = min(
                len(ordered) // 2,
                max(1, int(math.floor(len(ordered) * tail_fraction))),
            )
            if tail_count == 0:
                class_audit[str(class_id)] = {
                    "sample_count": len(ordered),
                    "tail_count_each": 0,
                    "excluded": True,
                }
                continue
            low = ordered[:tail_count]
            high = ordered[-tail_count:]
            for record in low:
                metric_assignment[record.key] = "low"
            for record in high:
                if metric_assignment[record.key] != "middle":
                    raise RuntimeError(f"overlapping tails for {metric}, class {class_id}")
                metric_assignment[record.key] = "high"
            low_boundary_tied = (
                tail_count < len(ordered)
                and low[-1].evidence[metric] == ordered[tail_count].evidence[metric]
            )
            high_start = len(ordered) - tail_count
            high_boundary_tied = (
                high_start > 0
                and ordered[high_start - 1].evidence[metric]
                == high[0].evidence[metric]
            )
            class_audit[str(class_id)] = {
                "sample_count": len(ordered),
                "tail_count_each": tail_count,
                "excluded": False,
                "low_max": low[-1].evidence[metric],
                "high_min": high[0].evidence[metric],
                "low_boundary_tied": low_boundary_tied,
                "high_boundary_tied": high_boundary_tied,
            }
        assignments[metric] = metric_assignment
        audit[metric] = {
            "selection": "equal-count low/high tails ranked separately within each class",
            "class_details": class_audit,
            "classes_used": sum(
                not details["excluded"] for details in class_audit.values()
            ),
            "boundary_tie_class_count": sum(
                bool(details.get("low_boundary_tied"))
                or bool(details.get("high_boundary_tied"))
                for details in class_audit.values()
            ),
        }
    return assignments, audit


def distribution_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty collection")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def quality_summary(scores: Sequence[ClassifierScore]) -> dict[str, Any]:
    if not scores:
        raise ValueError("cannot summarize zero classifier scores")
    return {
        "sample_count": len(scores),
        "target_probability": distribution_summary(
            score.target_probability for score in scores
        ),
        "top1_probability": distribution_summary(
            score.top1_probability for score in scores
        ),
        "top1_accuracy": float(np.mean([score.top1_correct for score in scores])),
        "top1_incorrect_count": int(sum(not score.top1_correct for score in scores)),
        "entropy_nats": distribution_summary(score.entropy_nats for score in scores),
        "normalized_entropy": distribution_summary(
            score.normalized_entropy for score in scores
        ),
    }


def _differences(high: dict[str, Any], low: dict[str, Any]) -> dict[str, float]:
    return {
        "target_probability_mean": float(
            high["target_probability"]["mean"] - low["target_probability"]["mean"]
        ),
        "top1_accuracy": float(high["top1_accuracy"] - low["top1_accuracy"]),
        "entropy_nats_mean": float(
            high["entropy_nats"]["mean"] - low["entropy_nats"]["mean"]
        ),
        "normalized_entropy_mean": float(
            high["normalized_entropy"]["mean"]
            - low["normalized_entropy"]["mean"]
        ),
    }


def tail_enrichment_summary(
    records: Sequence[InputRecord],
    scores: Sequence[ClassifierScore],
    assignments: dict[str, dict[tuple[int, int], str]],
    selection_audit: dict[str, Any],
) -> dict[str, Any]:
    if len(records) != len(scores):
        raise ValueError("record/score length mismatch")
    by_class_indices: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        by_class_indices.setdefault(record.class_id, []).append(index)

    output: dict[str, Any] = {}
    for metric in EVIDENCE_METRICS:
        metric_assignments = assignments[metric]
        low_indices = [
            index
            for index, record in enumerate(records)
            if metric_assignments[record.key] == "low"
        ]
        high_indices = [
            index
            for index, record in enumerate(records)
            if metric_assignments[record.key] == "high"
        ]
        if not low_indices or len(low_indices) != len(high_indices):
            raise RuntimeError(f"metric {metric} has invalid symmetric tails")
        low = quality_summary([scores[index] for index in low_indices])
        high = quality_summary([scores[index] for index in high_indices])
        high_wrong = int(high["top1_incorrect_count"])
        low_wrong = int(low["top1_incorrect_count"])
        high_count = int(high["sample_count"])
        low_count = int(low["sample_count"])
        raw_wrong_rate_ratio = (
            None
            if low_wrong == 0
            else (high_wrong / high_count) / (low_wrong / low_count)
        )
        smoothed_high_wrong_rate = (high_wrong + 0.5) / (high_count + 1.0)
        smoothed_low_wrong_rate = (low_wrong + 0.5) / (low_count + 1.0)
        smoothed_wrong_odds_ratio = (
            (high_wrong + 0.5) * (low_count - low_wrong + 0.5)
        ) / (
            (high_count - high_wrong + 0.5) * (low_wrong + 0.5)
        )

        per_class: dict[str, Any] = {}
        class_differences: list[dict[str, float]] = []
        for class_id, class_indices in sorted(by_class_indices.items()):
            class_low_indices = [
                index
                for index in class_indices
                if metric_assignments[records[index].key] == "low"
            ]
            class_high_indices = [
                index
                for index in class_indices
                if metric_assignments[records[index].key] == "high"
            ]
            if not class_low_indices:
                continue
            class_low = quality_summary([scores[index] for index in class_low_indices])
            class_high = quality_summary([scores[index] for index in class_high_indices])
            differences = _differences(class_high, class_low)
            class_differences.append(differences)
            per_class[str(class_id)] = {
                "low": class_low,
                "high": class_high,
                "high_minus_low": differences,
            }

        class_balanced = {
            name: float(np.mean([item[name] for item in class_differences]))
            for name in class_differences[0]
        }
        output[metric] = {
            **selection_audit[metric],
            "low": low,
            "high": high,
            "pooled_high_minus_low": _differences(high, low),
            "class_balanced_high_minus_low": class_balanced,
            "wrong_class_enrichment": {
                "high_incorrect_rate": high_wrong / high_count,
                "low_incorrect_rate": low_wrong / low_count,
                "high_minus_low": high_wrong / high_count - low_wrong / low_count,
                "low_minus_high": low_wrong / low_count - high_wrong / high_count,
                "raw_risk_ratio": raw_wrong_rate_ratio,
                "raw_low_over_high_risk_ratio": (
                    None
                    if high_wrong == 0
                    else (low_wrong / low_count) / (high_wrong / high_count)
                ),
                "haldane_smoothed_risk_ratio": (
                    smoothed_high_wrong_rate / smoothed_low_wrong_rate
                ),
                "haldane_smoothed_odds_ratio": smoothed_wrong_odds_ratio,
            },
            "per_class": per_class,
        }
    return output


def write_score_csv(
    records: Sequence[InputRecord],
    scores: Sequence[ClassifierScore],
    assignments: dict[str, dict[tuple[int, int], str]],
    path: Path,
) -> None:
    if len(records) != len(scores):
        raise ValueError("record/score length mismatch")
    fields = (
        "class_id",
        "seed",
        "image_path",
        "signal_path",
        "target_probability",
        "top1_class_id",
        "top1_probability",
        "top1_correct",
        "entropy_nats",
        "normalized_entropy",
        *EVIDENCE_METRICS,
        *(f"tail_{metric}" for metric in EVIDENCE_METRICS),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, score in zip(records, scores, strict=True):
            writer.writerow(
                {
                    "class_id": record.class_id,
                    "seed": record.seed,
                    "image_path": str(record.image_path),
                    "signal_path": str(record.signal_path),
                    "target_probability": score.target_probability,
                    "top1_class_id": score.top1_class_id,
                    "top1_probability": score.top1_probability,
                    "top1_correct": int(score.top1_correct),
                    "entropy_nats": score.entropy_nats,
                    "normalized_entropy": score.normalized_entropy,
                    **record.evidence,
                    **{
                        f"tail_{metric}": assignments[metric][record.key]
                        for metric in EVIDENCE_METRICS
                    },
                }
            )
    temporary.replace(path)


def build_summary(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    records: Sequence[InputRecord],
    scores: Sequence[ClassifierScore],
    weights_path: Path,
    weights_sha256: str,
    input_audit: dict[str, Any],
    assignments: dict[str, dict[tuple[int, int], str]],
    selection_audit: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    by_class: dict[int, list[ClassifierScore]] = {}
    for record, score in zip(records, scores, strict=True):
        by_class.setdefault(record.class_id, []).append(score)
    return {
        "schema_version": 1,
        "analysis": "blind_cfg_rejection_edm2_convnext_imagenet_scoring",
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "blindness": (
            "Classifier predictions use only PNG pixels and manifest target class IDs; "
            "evidence values are joined only for prespecified within-class tail analysis."
        ),
        "run_dir": str(args.run_dir),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "protocol": manifest.get("protocol"),
            "role": manifest.get("role"),
        },
        "input_audit": input_audit,
        "classifier": {
            "name": MODEL_NAME,
            "weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "weights_path": str(weights_path),
            "weights_bytes": weights_path.stat().st_size,
            "weights_sha256": weights_sha256,
            "weights_source_url_recorded_only_no_download": MODEL_WEIGHTS.url,
            "preprocessing": repr(MODEL_WEIGHTS.transforms()),
            "entropy_units": "natural logarithm nats",
            "normalized_entropy_denominator": "log(1000)",
        },
        "execution": {
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "allow_tf32": bool(args.allow_tf32),
            "elapsed_seconds": float(elapsed_seconds),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "cuda": torch.version.cuda,
        },
        "tail_protocol": {
            "requested_fraction_each": float(args.tail_fraction),
            "ranking": "within class, ascending evidence, deterministic seed/path tie break",
            "symmetry": "equal low and high counts in every included class",
            "interpretation": (
                "CFG-Rejection treats LOW ASD as suspicious. If its ranking tracks class "
                "consistency, high-minus-low target probability and accuracy should be "
                "positive, entropy should be negative, and low-minus-high incorrect rate "
                "should be positive. These are semantic weak labels, not artifact labels."
            ),
            "boundary_ties": (
                "Fixed-count tails remain symmetric; boundary tie counts disclose where "
                "membership is determined by the recorded deterministic tie break."
            ),
        },
        "overall": quality_summary(scores),
        "per_class": {
            str(class_id): quality_summary(class_scores)
            for class_id, class_scores in sorted(by_class.items())
        },
        "tail_enrichment": tail_enrichment_summary(
            records, scores, assignments, selection_audit
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="CFG-Rejection EDM2 run containing manifest.json, images/, and signals/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest override; defaults to RUN_DIR/manifest.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Per-image output; defaults to RUN_DIR/analysis/convnext_tiny_scores.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Summary output; defaults to RUN_DIR/analysis/convnext_tiny_summary.json.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Local ConvNeXt-Tiny V1 checkpoint; defaults to the torch hub cache.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.10,
        help="Within-class fraction assigned to each evidence tail.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Score a partial run while recording the manifest mismatch prominently.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TF32; disabled by default for tighter numerical reproducibility.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if not 0.0 < args.tail_fraction <= 0.5:
        parser.error("--tail-fraction must be in (0, 0.5]")
    return args


def resolve_device(argument: str) -> torch.device:
    if argument == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else args.run_dir / "manifest.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else args.run_dir / "analysis" / "convnext_tiny_scores.csv"
    )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else args.run_dir / "analysis" / "convnext_tiny_summary.json"
    )
    if output_csv == output_json:
        raise ValueError("--output-csv and --output-json must be different paths")
    weights_path = (
        args.weights.expanduser().resolve()
        if args.weights is not None
        else default_cached_weights_path()
    )
    if not weights_path.is_file():
        raise FileNotFoundError(
            "local ConvNeXt-Tiny weights are missing; refusing to download: "
            f"{weights_path}"
        )
    device = resolve_device(args.device)
    args.device = str(device)
    configure_reproducibility(args.seed, args.allow_tf32)

    manifest, manifest_sha256 = read_manifest(manifest_path)
    records, input_audit = load_records(
        args.run_dir,
        manifest,
        allow_incomplete=args.allow_incomplete,
    )
    assignments, selection_audit = select_symmetric_tails(
        records,
        tail_fraction=args.tail_fraction,
    )
    weights_sha256 = sha256_file(weights_path)
    model = load_local_model(weights_path, device)
    started = time.perf_counter()
    scores = score_records(
        records,
        model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    write_score_csv(records, scores, assignments, output_csv)
    summary = build_summary(
        args=args,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        records=records,
        scores=scores,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        input_audit=input_audit,
        assignments=assignments,
        selection_audit=selection_audit,
        elapsed_seconds=elapsed_seconds,
    )
    summary["outputs"] = {
        "per_image_csv": str(output_csv),
        "summary_json": str(output_json),
    }
    atomic_json_dump(summary, output_json)
    print(
        json.dumps(
            {
                "records": len(records),
                "complete_against_manifest": input_audit["complete_against_manifest"],
                "top1_accuracy": summary["overall"]["top1_accuracy"],
                "elapsed_seconds": elapsed_seconds,
                "output_csv": str(output_csv),
                "output_json": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
