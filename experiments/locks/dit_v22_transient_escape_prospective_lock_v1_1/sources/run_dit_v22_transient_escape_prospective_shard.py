#!/usr/bin/env python3
"""Run one shard of the frozen DiT-v2.2 h10 transient-escape experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1_1"
LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_LOCK_V1_1"
RUNNER_NAME = "intervene_dit_v22_transient_escape_suffix"
RNG_NAMESPACE = "eqvae-dit-v22-h10-max-nonconformity-prospective-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"self hash failed: {path}")
    return value


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid prospective lock tree: {root}")
    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != LOCK_KIND or manifest.get("status") != "complete":
        raise RuntimeError("wrong prospective lock kind/status")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("prospective lock exact tree changed")
    for name, record in records.items():
        path = root / name
        expected = {
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if record != expected:
            raise RuntimeError(f"prospective lock member changed: {name}")
    protocol = load_self_hashed(root / "protocol.json", "identity_sha256")
    if (
        protocol.get("identity_sha256") != manifest.get("protocol_identity_sha256")
        or protocol.get("status") != "EXECUTION_READY_UNOBSERVED_PROSPECTIVE_SUFFIXES"
        or protocol.get("fresh_rng", {}).get("namespace") != RNG_NAMESPACE
        or len(protocol.get("jobs", [])) != 128
    ):
        raise RuntimeError("prospective protocol scope changed")
    return manifest, protocol


def command_for_job(lock: Path, protocol: Mapping[str, Any], job: Mapping[str, Any]) -> list[str]:
    lineage = protocol["lineage"]
    return [
        sys.executable,
        str(lock / "sources/intervene_dit_v22_transient_escape_suffix.py"),
        "--trace-dir",
        str(job["trace_dir"]),
        "--target-slot",
        str(job["class_slot"]),
        "--target-class-id",
        str(job["class_id"]),
        "--rollback-sampling-step",
        str(job["rollback_sampling_step"]),
        "--prospective-lock",
        str(lock),
        "--dit-root",
        str(lineage["dit_root"]),
        "--checkpoint",
        str(lineage["checkpoint"]),
        "--vae-snapshot",
        str(lineage["vae_snapshot"]),
        "--outdir",
        str(job["outdir"]),
    ]


def validate_output(
    path: Path,
    job: Mapping[str, Any],
    *,
    lock_identity_sha256: str,
    protocol_identity_sha256: str,
) -> dict[str, Any]:
    manifest = load_self_hashed(path / "manifest.json", "identity_sha256")
    completion = load_self_hashed(path / "completion.json", "payload_sha256")
    target = manifest.get("target", {})
    rollback = manifest.get("rollback", {})
    streams = [row.get("seed") for row in manifest.get("branches", {}).get("fresh_stream_seeds", [])]
    binding = manifest.get("prospective_binding", {})
    if (
        manifest.get("runner") != RUNNER_NAME
        or manifest.get("posthoc_exploratory") is not False
        or manifest.get("method_claim_eligible") is not True
        or manifest.get("rng", {}).get("namespace") != RNG_NAMESPACE
        or binding.get("lock_identity_sha256") != lock_identity_sha256
        or binding.get("protocol_identity_sha256") != protocol_identity_sha256
        or binding.get("job_index") != job["job_index"]
        or binding.get("trace_identity_sha256") != job["trace_identity_sha256"]
        or target.get("global_seed") != job["global_seed"]
        or target.get("class_id") != job["class_id"]
        or target.get("slot") != job["class_slot"]
        or rollback.get("sampling_step_index_zero_based") != job["rollback_sampling_step"]
        or streams != job["fresh_stream_seeds"]
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
    ):
        raise RuntimeError(f"prospective output does not match frozen job: {path}")
    return {
        "job_index": job["job_index"],
        "outdir": str(path),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(path / "manifest.json"),
        "completion_payload_sha256": completion["payload_sha256"],
        "completion_file_sha256": sha256_file(path / "completion.json"),
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard_count)")
    lock = args.lock.expanduser().resolve()
    lock_manifest, protocol = validate_lock(lock)
    jobs = [
        row
        for row in protocol["jobs"]
        if int(row["job_index"]) % args.shard_count == args.shard_index
    ]
    if not jobs:
        raise RuntimeError("selected shard has no jobs")
    records = []
    for ordinal, job in enumerate(jobs, start=1):
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "progress": f"{ordinal}/{len(jobs)}",
                    "job_index": job["job_index"],
                    "seed": job["global_seed"],
                    "class_id": job["class_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(command_for_job(lock, protocol, job), check=True)
        records.append(
            validate_output(
                Path(job["outdir"]),
                job,
                lock_identity_sha256=lock_manifest["identity_sha256"],
                protocol_identity_sha256=protocol["identity_sha256"],
            )
        )

    receipt_root = Path(protocol["outputs"]["receipt_root"])
    receipt_dir = receipt_root / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_SHARD_RECEIPT_V1",
        "status": "complete",
        "execution_lock_identity_sha256": lock_manifest["identity_sha256"],
        "protocol_identity_sha256": protocol["identity_sha256"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "job_indices": [row["job_index"] for row in jobs],
        "outputs": records,
        "png_label_quality_B_E_O_FID_embedding_or_attempt_selection_used": False,
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    if receipt_dir.exists():
        if load_self_hashed(receipt_dir / "receipt.json", "identity_sha256") != receipt:
            raise RuntimeError(f"existing shard receipt differs: {receipt_dir}")
        print(f"validated existing receipt: {receipt_dir}")
        return
    receipt_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{receipt_dir.name}.tmp-", dir=receipt_root))
    try:
        write_json(staging / "receipt.json", receipt)
        os.replace(staging, receipt_dir)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    print(json.dumps({"status": "complete", "receipt": str(receipt_dir)}, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
