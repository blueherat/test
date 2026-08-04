"""Extract ADM Inception features for multiple SiT interval image NPZs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train_gen"))

PROTOCOL = "sit_ig_interval_adm_features_v2"


def load_adm_modules() -> tuple[Any, Any]:
    import evaluator as adm_evaluator
    import tensorflow.compat.v1 as tf

    return adm_evaluator, tf


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    parser.add_argument("--reference-stats-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--inception-path",
        default="/data/shared/adm_refs/classify_image_graph_def.pb",
    )
    return parser.parse_args()


def load_statistics(path: Path, adm_evaluator: Any) -> tuple[Any, Any]:
    with np.load(path) as data:
        return (
            adm_evaluator.FIDStatistics(data["mu"], data["sigma"]),
            adm_evaluator.FIDStatistics(data["mu_s"], data["sigma_s"]),
        )


def reference_statistics(
    evaluator: Any,
    *,
    reference: Path,
    cache: Path,
    adm_evaluator: Any,
) -> tuple[Any, Any]:
    if cache.is_file():
        return load_statistics(cache, adm_evaluator)
    with np.load(reference) as embedded:
        required = {"mu", "sigma", "mu_s", "sigma_s"}
        if required.issubset(embedded.files):
            stats = adm_evaluator.FIDStatistics(embedded["mu"], embedded["sigma"])
            spatial = adm_evaluator.FIDStatistics(
                embedded["mu_s"], embedded["sigma_s"]
            )
        else:
            activations = evaluator.read_activations(str(reference))
            stats, spatial = evaluator.read_statistics(str(reference), activations)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            mu=stats.mu,
            sigma=stats.sigma,
            mu_s=spatial.mu,
            sigma_s=spatial.sigma,
        )
    os.replace(temporary, cache)
    return stats, spatial


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def extract_one(
    evaluator: Any,
    *,
    samples: Path,
    feature_path: Path,
) -> dict[str, Any]:
    activations = evaluator.read_activations(str(samples))
    pool3 = np.asarray(activations[0], dtype=np.float32)
    atomic_npy(feature_path, pool3)
    return {
        "activations": str(feature_path),
        "activation_rows": int(pool3.shape[0]),
        "activation_dims": int(pool3.shape[1]),
        "inception_score": float(evaluator.compute_inception_score(pool3)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    decode_dir = args.decode_dir.expanduser().resolve()
    decode_manifest = json.loads(
        (decode_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if decode_manifest.get("status") != "complete":
        raise RuntimeError("decoded interval run is incomplete")
    if decode_manifest.get("protocol") != "sit_ig_interval_decode_v1":
        raise RuntimeError("unexpected decode protocol")
    conditions = [str(value) for value in decode_manifest["conditions"]]
    missing = [name for name in conditions if not (decode_dir / f"{name}.npz").is_file()]
    if missing:
        raise RuntimeError(f"missing decoded conditions: {missing}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adm_evaluator, tf = load_adm_modules()
    adm_evaluator.INCEPTION_V3_PATH = args.inception_path
    Path(args.inception_path).expanduser().resolve().parent.mkdir(
        parents=True, exist_ok=True
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = adm_evaluator.Evaluator(
        tf.Session(config=config), batch_size=args.batch_size
    )
    evaluator.warmup()
    reference_statistics(
        evaluator,
        reference=args.reference.expanduser().resolve(),
        cache=args.reference_stats_cache.expanduser().resolve(),
        adm_evaluator=adm_evaluator,
    )
    rows = []
    for name in conditions:
        feature_path = output_dir / f"{name}_pool3.npy"
        record_path = output_dir / f"{name}.json"
        if record_path.is_file() and feature_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            record = {
                "condition": name,
                "samples": str(decode_dir / f"{name}.npz"),
                **extract_one(
                    evaluator,
                    samples=decode_dir / f"{name}.npz",
                    feature_path=feature_path,
                ),
            }
            atomic_json(record_path, record)
        rows.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    with (output_dir / "feature_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        output_dir / "manifest.json",
        {
            "protocol": PROTOCOL,
            "status": "complete",
            "decode_dir": str(decode_dir),
            "reference": str(args.reference.expanduser().resolve()),
            "reference_stats_cache": str(
                args.reference_stats_cache.expanduser().resolve()
            ),
            "batch_size": args.batch_size,
            "conditions": conditions,
            "sample_count": int(decode_manifest["samples"]),
            "feature_backend": "ADM TensorFlow Inception pool_3",
            "scope": "diagnostic FID; public XL/2 trend replication, not Table 3 B/2",
        },
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
