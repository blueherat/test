#!/usr/bin/env python3
"""Evaluate RAEv2 sample archives with the official nanogen evaluator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_training_core import file_sha256  # noqa: E402


DEFAULT_EVALUATOR_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals"
)


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("branch must be NAME=SAMPLES_NPZ")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid branch or missing archive: {value}")
    return name, path


def evaluator_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", action="append", type=parse_branch, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--fid-reference", default="imagenet_256_fid_stats")
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument(
        "--evaluator-root", type=Path, default=DEFAULT_EVALUATOR_ROOT
    )
    args = parser.parse_args()

    evaluator_root = args.evaluator_root.expanduser().resolve()
    package_root = evaluator_root / "fd_evaluator"
    if not package_root.is_dir():
        raise FileNotFoundError(f"fd_evaluator source is missing: {package_root}")
    sys.path.insert(0, str(package_root))
    from fd_evaluator import compute_metrics

    cache_dir = (
        args.feature_cache_dir.expanduser().resolve()
        if args.feature_cache_dir is not None
        else args.output.expanduser().resolve().parent / "official_feature_cache"
    )
    source_commit = evaluator_commit(evaluator_root)
    rows: list[dict[str, object]] = []
    for name, sample_path in args.branch:
        sample_hash = file_sha256(sample_path)
        metrics = compute_metrics(
            images=str(sample_path),
            metrics=["fid", "inception_score"],
            fid_reference=args.fid_reference,
            device=args.device,
            batch_size=args.batch_size,
            rng_seed=args.seed,
            feature_cache_dir=str(cache_dir),
            feature_cache_key=f"{name}-{sample_hash[:16]}",
            verbose=True,
        )
        rows.append(
            {
                "branch": name,
                "sample_path": str(sample_path),
                "sample_sha256": sample_hash,
                "fid_reference": args.fid_reference,
                "evaluator_root": str(evaluator_root),
                "evaluator_commit": source_commit,
                **metrics,
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
