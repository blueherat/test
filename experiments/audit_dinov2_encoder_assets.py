"""Hash the exact cached DINOv2 encoder snapshots used by RAE experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REVISIONS = {
    "small": "0d9846e56b43a21fa46d7f3f5070f0506a5795a9",
    "base": "a1d738ccfa7ae170945f210395d99dde8adb1805",
    "large": "e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_snapshot(cache_root: Path, size: str, revision: str) -> dict[str, object]:
    repository = f"facebook/dinov2-with-registers-{size}"
    root = cache_root / f"models--facebook--dinov2-with-registers-{size}"
    ref = root / "refs/main"
    snapshot = root / "snapshots" / revision
    if not ref.exists():
        raise FileNotFoundError(ref)
    resolved_ref = ref.read_text(encoding="utf-8").strip()
    if resolved_ref != revision:
        raise RuntimeError(
            f"{repository}: refs/main={resolved_ref}, expected pinned {revision}"
        )

    files = {}
    for filename in ("model.safetensors", "config.json", "preprocessor_config.json"):
        path = snapshot / filename
        if not path.exists():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        sha256 = file_sha256(resolved)
        files[filename] = {
            "snapshot_path": str(path),
            "resolved_path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256,
        }
        if filename == "model.safetensors" and resolved.name != sha256:
            raise RuntimeError(
                f"{repository}: model blob name {resolved.name} != SHA256 {sha256}"
            )

    return {
        "size": size,
        "repository": repository,
        "revision": revision,
        "refs_main_matches_pin": True,
        "model_blob_name_matches_sha256": True,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / "data/eqvae/experiments/rae_lpl_authenticity"
        / "dinov2_encoder_asset_audit.json",
    )
    args = parser.parse_args()

    cache_root = args.cache_root.expanduser().resolve()
    assets = [
        audit_snapshot(cache_root, size, revision)
        for size, revision in REVISIONS.items()
    ]
    payload = {
        "offline_asset_audit": True,
        "all_assets_match": True,
        "assets": assets,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
