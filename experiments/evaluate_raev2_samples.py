"""Compute one consistent torch-fidelity table for RAEv2 sample archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_rae_spectral_generation import (  # noqa: E402
    NumpyRGBDataset,
    file_sha256,
    torch_fidelity_metrics,
)


def parse_branch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("branch must be NAME=SAMPLES_NPZ")
    name, path = value.split("=", 1)
    return name, Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", action="append", type=parse_branch, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reference_path = args.reference.expanduser().resolve()
    reference = NumpyRGBDataset(reference_path)
    rows = []
    for name, sample_path in args.branch:
        samples = NumpyRGBDataset(sample_path)
        metrics = torch_fidelity_metrics(
            samples,
            reference,
            batch_size=args.batch_size,
            cache_name="raev2_imagenet256_virtual_reference",
            rng_seed=args.seed,
        )
        rows.append(
            {
                "branch": name,
                "sample_path": str(sample_path),
                "sample_sha256": file_sha256(sample_path),
                "sample_count": len(samples),
                "reference_path": str(reference_path),
                "reference_sha256": file_sha256(reference_path),
                **metrics,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
