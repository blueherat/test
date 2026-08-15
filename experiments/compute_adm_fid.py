#!/usr/bin/env python3
"""Compute ADM-style FID for generated ImageNet samples.

This intentionally uses the ADM/guided-diffusion TensorFlow Inception graph
through the repository's ``train_gen/evaluator.py`` implementation.  It skips
precision/recall by default because gFID only needs the Frechet distance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "train_gen"))

import evaluator as adm_evaluator  # noqa: E402
import tensorflow.compat.v1 as tf  # noqa: E402


def _npz_keys(path: str) -> set[str]:
    with np.load(path) as obj:
        return set(obj.keys())


def _stats_or_activations(evalr: adm_evaluator.Evaluator, path: str):
    keys = _npz_keys(path)
    if {"mu", "sigma", "mu_s", "sigma_s"}.issubset(keys):
        obj = np.load(path)
        return (
            adm_evaluator.FIDStatistics(obj["mu"], obj["sigma"]),
            adm_evaluator.FIDStatistics(obj["mu_s"], obj["sigma_s"]),
            None,
        )

    acts = evalr.read_activations(path)
    stats, stats_spatial = evalr.read_statistics(path, acts)
    return stats, stats_spatial, acts


def _save_statistics(
    path: Path,
    stats: adm_evaluator.FIDStatistics,
    stats_spatial: adm_evaluator.FIDStatistics,
) -> None:
    """Save the exact four-array ADM reference format understood above."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez(
        temporary,
        mu=stats.mu,
        sigma=stats.sigma,
        mu_s=stats_spatial.mu,
        sigma_s=stats_spatial.sigma,
    )
    temporary.replace(path)


def _save_activations(
    path: Path,
    activations: tuple[np.ndarray, np.ndarray],
) -> None:
    """Persist ADM pool and spatial features without changing FID evaluation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez(
        temporary,
        pool_3=np.asarray(activations[0]),
        spatial=np.asarray(activations[1]),
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ADM/guided-diffusion FID.")
    parser.add_argument("--reference", required=True, help="ADM reference .npz")
    parser.add_argument("--samples", required=True, help="Generated samples .npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.30,
        help="Hard TensorFlow fraction of the single visible GPU.",
    )
    parser.add_argument(
        "--inception-path",
        default="/data/shared/adm_refs/classify_image_graph_def.pb",
        help="Where to cache ADM's classify_image_graph_def.pb.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--reference-stats-output",
        default=None,
        help=(
            "Optionally cache the reference mu/sigma arrays as an NPZ. Future "
            "evaluations can pass that NPZ as --reference without recomputing "
            "reference Inception activations."
        ),
    )
    parser.add_argument(
        "--precision-recall",
        action="store_true",
        help="Also compute ADM precision/recall. This is expensive for 50k samples.",
    )
    parser.add_argument(
        "--activations-output",
        default=None,
        help="Optionally save sample pool_3 and spatial activations as an NPZ.",
    )
    args = parser.parse_args()
    if not 0.0 < args.gpu_memory_fraction < 1.0:
        raise ValueError("--gpu-memory-fraction must be between 0 and 1")

    adm_evaluator.INCEPTION_V3_PATH = args.inception_path
    os.makedirs(os.path.dirname(os.path.abspath(args.inception_path)), exist_ok=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    evalr = adm_evaluator.Evaluator(tf.Session(config=config), batch_size=args.batch_size)

    print("Warmup...")
    evalr.warmup()

    print("Reference statistics...")
    ref_stats, ref_stats_spatial, ref_acts = _stats_or_activations(evalr, args.reference)
    if args.reference_stats_output:
        stats_path = Path(args.reference_stats_output).expanduser().resolve()
        if stats_path == Path(args.reference).expanduser().resolve():
            raise ValueError("reference statistics output must differ from --reference")
        _save_statistics(stats_path, ref_stats, ref_stats_spatial)
        print(f"Cached reference statistics: {stats_path}")

    print("Sample activations/statistics...")
    sample_stats, sample_stats_spatial, sample_acts = _stats_or_activations(evalr, args.samples)
    if sample_acts is None:
        raise ValueError("Sample npz must contain images, not only precomputed statistics.")
    if args.activations_output:
        activation_path = Path(args.activations_output).expanduser().resolve()
        _save_activations(activation_path, sample_acts)
        print(f"Saved sample activations: {activation_path}")

    metrics: dict[str, Any] = {
        "reference": args.reference,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "fid": float(sample_stats.frechet_distance(ref_stats)),
        "sfid": float(sample_stats_spatial.frechet_distance(ref_stats_spatial)),
        "inception_score": float(evalr.compute_inception_score(sample_acts[0])),
    }

    if args.precision_recall:
        if ref_acts is None:
            print("Reference activations are required for precision/recall; recomputing.")
            ref_acts = evalr.read_activations(args.reference)
        prec, recall = evalr.compute_prec_recall(ref_acts[0], sample_acts[0])
        metrics["precision"] = float(prec)
        metrics["recall"] = float(recall)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
