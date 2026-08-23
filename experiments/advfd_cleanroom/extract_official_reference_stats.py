"""Extract AdvFD's paper reference-statistics bundle into exact NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


REQUIRED_STATS = (
    "guided_diffusion_stats.npz",
    "vit_large_patch16_224_mae_in256_t224_stats.npz",
    "vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/research_repos/FD-Loss-assets/"
            "paper_ref_stats.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/research_deps/advfd_reference_stats"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.bundle.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError("Reference-statistics bundle must be a dictionary")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for filename in REQUIRED_STATS:
        if filename not in bundle:
            raise KeyError(f"Missing {filename!r} in {args.bundle}")
        arrays = bundle[filename]
        if not isinstance(arrays, dict) or not {"mu", "sigma"} <= set(arrays):
            raise ValueError(f"Malformed statistics entry: {filename}")
        output = args.output_dir / filename
        np.savez(output, **arrays)
        with np.load(output) as loaded:
            for key, expected in arrays.items():
                if key not in loaded or not np.array_equal(loaded[key], expected):
                    raise RuntimeError(f"Round-trip mismatch for {filename}:{key}")
        records.append(
            {
                "filename": filename,
                "keys": sorted(arrays),
                "mean_dimension": int(np.asarray(arrays["mu"]).size),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            }
        )

    manifest = {
        "source_bundle": str(args.bundle.resolve()),
        "source_bundle_sha256": sha256(args.bundle),
        "files": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
