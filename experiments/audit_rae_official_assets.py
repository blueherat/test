"""Verify local RAE checkpoints against official Hugging Face LFS hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "nyu-visionx/RAE-collections"
MODEL_ROOT = Path.home() / "data/eqvae/models/RAE"
FILES = (
    "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep14/stage2_model.pt",
    "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep20/stage2_model.pt",
    "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL_ep20/stage2_model.pt",
    "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL_ep80/stage2_model.pt",
    "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt",
    "decoders/dinov2/wReg_small/ViTXL_n08/model.pt",
    "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
    "decoders/dinov2/wReg_large/ViTXL_n08/model.pt",
    "stats/dinov2/wReg_small/imagenet1k/stat.pt",
    "stats/dinov2/wReg_base/imagenet1k/stat.pt",
    "stats/dinov2/wReg_large/imagenet1k/stat.pt",
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / "data/eqvae/experiments/rae_lpl_authenticity/official_assets.json",
    )
    args = parser.parse_args()

    info = HfApi().model_info(REPO_ID, files_metadata=True)
    metadata = {sibling.rfilename: sibling for sibling in info.siblings}
    rows = []
    errors = []
    for name in FILES:
        local = args.model_root.expanduser().resolve() / name
        sibling = metadata.get(name)
        if sibling is None or sibling.lfs is None:
            errors.append(f"official metadata lacks an LFS hash for {name}")
            continue
        if not local.exists():
            errors.append(f"missing local asset: {local}")
            continue
        local_hash = file_sha256(local)
        official_hash = sibling.lfs.sha256
        matches = (
            local.stat().st_size == sibling.size and local_hash == official_hash
        )
        rows.append(
            {
                "remote_path": name,
                "local_path": str(local),
                "bytes": local.stat().st_size,
                "official_bytes": sibling.size,
                "local_sha256": local_hash,
                "official_sha256": official_hash,
                "matches_official": matches,
            }
        )
        if not matches:
            errors.append(f"asset differs from official LFS object: {name}")

    result = {
        "repository": REPO_ID,
        "revision": info.sha,
        "all_assets_match": not errors,
        "assets": rows,
        "errors": errors,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
