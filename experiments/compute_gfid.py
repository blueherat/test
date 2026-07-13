from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = REPO_ROOT / "external" / "RAE" / "src"
if str(RAE_SRC) not in sys.path:
    sys.path.insert(0, str(RAE_SRC))


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        data = np.load(path)
        if "arr_0" in data.files:
            return data["arr_0"]
        if "images" in data.files:
            return data["images"]
        raise KeyError(f"{path} has no arr_0/images array. Keys={data.files}")
    raise ValueError(f"Unsupported array format: {path}")


def load_stats(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".npz":
        raise ValueError(f"Reference stats must be .npz, got {path}")
    data = np.load(path)
    if "mu" not in data.files or "sigma" not in data.files:
        raise KeyError(f"{path} must contain mu and sigma. Keys={data.files}")
    return {"mu": data["mu"], "sigma": data["sigma"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RAE-style generation FID for sampled npy/npz arrays.")
    parser.add_argument("--samples", required=True, help="Generated samples .npy/.npz, usually sample_ddp output.")
    parser.add_argument("--reference", default=None, help="Reference image array .npy/.npz. Uses torch-fidelity two-input FID.")
    parser.add_argument("--reference-stats", default=None, help="Reference stats .npz with mu/sigma. Uses RAE calculate_gfid.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if (args.reference is None) == (args.reference_stats is None):
        raise SystemExit("Specify exactly one of --reference or --reference-stats.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    from eval.fid import calculate_gfid, calculate_rfid

    samples = load_array(args.samples)
    result: dict[str, Any] = {
        "samples": str(Path(args.samples).expanduser()),
        "sample_shape": list(samples.shape),
        "sample_dtype": str(samples.dtype),
        "batch_size": int(args.batch_size),
        "device": args.device,
    }
    if args.reference is not None:
        reference = load_array(args.reference)
        fid = float(calculate_rfid(reference, samples, bs=args.batch_size, device=args.device))
        result.update(
            {
                "mode": "two_input_reference_images",
                "reference": str(Path(args.reference).expanduser()),
                "reference_shape": list(reference.shape),
                "reference_dtype": str(reference.dtype),
                "gfid": fid,
            }
        )
    else:
        stats = load_stats(args.reference_stats)
        fid = float(calculate_gfid(samples, stats, batch_size=args.batch_size, device=args.device))
        result.update(
            {
                "mode": "precomputed_reference_stats",
                "reference_stats": str(Path(args.reference_stats).expanduser()),
                "gfid": fid,
            }
        )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
