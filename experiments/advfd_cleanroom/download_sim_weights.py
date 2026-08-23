#!/usr/bin/env python3
"""Populate an isolated timm cache with the exact FD-SIM checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import timm

from experiments.advfd_cleanroom.feature_extractors import (
    MAE_LARGE_224,
    SIGLIP2_SO400M_224,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=50)
    parser.add_argument("--retry-delay", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for spec in (MAE_LARGE_224, SIGLIP2_SO400M_224):
        model = None
        for attempt in range(1, args.max_retries + 1):
            print(
                f"loading {spec.model_name} attempt={attempt}/{args.max_retries}",
                flush=True,
            )
            try:
                model = timm.create_model(
                    spec.model_name,
                    pretrained=True,
                    num_classes=0,
                    **dict(spec.model_kwargs),
                )
                break
            except Exception as error:
                print(
                    f"download/load failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt == args.max_retries:
                    raise
                time.sleep(args.retry_delay)
        assert model is not None
        record = {
            "name": spec.name,
            "model_name": spec.model_name,
            "output_dim": int(model.num_features),
            "input_size": spec.input_size,
            "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", 0)),
            "global_pool": str(getattr(model, "global_pool", "unknown")),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del model
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
