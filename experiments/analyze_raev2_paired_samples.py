"""Analyze paired RAEv2 samples generated from identical noise and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=SAMPLES_NPZ")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("name cannot be empty")
    return name, Path(path).expanduser().resolve()


def load_rgb_samples(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        images = np.asarray(payload["arr_0"])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB samples in {path}, got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"expected uint8 samples in {path}, got {images.dtype}")
    return images


def _summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p99": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1 if values.size > 1 else 0)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
    }


def _rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right, dtype=np.float64)
    left_norm = np.sqrt(np.einsum("ij,ij->i", left, left, dtype=np.float64))
    right_norm = np.sqrt(np.einsum("ij,ij->i", right, right, dtype=np.float64))
    denominator = left_norm * right_norm
    cosine = np.full(len(left), np.nan, dtype=np.float64)
    valid = denominator > 0
    cosine[valid] = numerator[valid] / denominator[valid]
    return cosine


def paired_metrics(
    *,
    origin: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    chunk_size: int = 16,
) -> dict[str, float]:
    """Measure a candidate against a same-noise continuation control."""

    if origin.shape != control.shape or origin.shape != candidate.shape:
        raise ValueError(
            "origin, control, and candidate must have identical shapes, got "
            f"{origin.shape}, {control.shape}, and {candidate.shape}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    candidate_control_mae = []
    candidate_origin_mae = []
    control_origin_mae = []
    update_cosine = []
    extra_vs_flow_cosine = []
    for start in range(0, len(origin), int(chunk_size)):
        stop = min(start + int(chunk_size), len(origin))
        origin_chunk = origin[start:stop].astype(np.float32) / 255.0
        control_chunk = control[start:stop].astype(np.float32) / 255.0
        candidate_chunk = candidate[start:stop].astype(np.float32) / 255.0

        flow_update = (control_chunk - origin_chunk).reshape(stop - start, -1)
        candidate_update = (candidate_chunk - origin_chunk).reshape(stop - start, -1)
        lpl_increment = (candidate_chunk - control_chunk).reshape(stop - start, -1)

        candidate_control_mae.append(np.mean(np.abs(lpl_increment), axis=1))
        candidate_origin_mae.append(np.mean(np.abs(candidate_update), axis=1))
        control_origin_mae.append(np.mean(np.abs(flow_update), axis=1))
        update_cosine.append(_rowwise_cosine(candidate_update, flow_update))
        extra_vs_flow_cosine.append(_rowwise_cosine(lpl_increment, flow_update))

    candidate_control_mae_array = np.concatenate(candidate_control_mae)
    candidate_origin_mae_array = np.concatenate(candidate_origin_mae)
    control_origin_mae_array = np.concatenate(control_origin_mae)
    update_cosine_array = np.concatenate(update_cosine)
    extra_vs_flow_cosine_array = np.concatenate(extra_vs_flow_cosine)

    result = {
        "sample_count": int(len(origin)),
        **_summarize(candidate_control_mae_array, "mae_to_flow"),
        **_summarize(candidate_origin_mae_array, "mae_to_official"),
        **_summarize(control_origin_mae_array, "flow_mae_to_official"),
        **_summarize(
            update_cosine_array[np.isfinite(update_cosine_array)],
            "total_update_cosine_with_flow",
        ),
        **_summarize(
            extra_vs_flow_cosine_array[np.isfinite(extra_vs_flow_cosine_array)],
            "lpl_increment_cosine_with_flow",
        ),
    }
    mean_flow_update = float(np.mean(control_origin_mae_array))
    result["mae_to_flow_over_flow_update"] = (
        float(np.mean(candidate_control_mae_array) / mean_flow_update)
        if mean_flow_update > 0
        else float("nan")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--branch", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    official = load_rgb_samples(args.official.expanduser().resolve())
    flow = load_rgb_samples(args.flow.expanduser().resolve())
    rows = []
    for name, path in args.branch:
        candidate = load_rgb_samples(path)
        rows.append(
            {
                "branch": name,
                "sample_path": str(path),
                **paired_metrics(
                    origin=official,
                    control=flow,
                    candidate=candidate,
                    chunk_size=args.chunk_size,
                ),
            }
        )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
