#!/usr/bin/env python3
"""Run one deterministic shard of the frozen v2.2 repairability job matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_v22_repairability_execution_source_lock_v1"
RECEIPT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_receipts"
)


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
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"invalid self hash: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"invalid execution lock: {root}")
    manifest = self_hashed(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != "DIT_V22_REPAIRABILITY_EXECUTION_SOURCE_LOCK_V1":
        raise RuntimeError("wrong execution lock kind")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("execution lock exact tree changed")
    for name, record in records.items():
        path = root / name
        if (
            path.is_symlink()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"execution lock member changed: {name}")
    contract = self_hashed(root / "execution_contract.json", "identity_sha256")
    if contract.get("identity_sha256") != manifest.get(
        "execution_contract_identity_sha256"
    ):
        raise RuntimeError("execution contract identity changed")
    if contract.get("status") != "EXECUTION_READY_RETROSPECTIVE_EXPLORATORY_ONLY":
        raise RuntimeError("execution contract is not ready")
    return manifest, contract


def command_for_job(lock: Path, contract: dict[str, Any], job: dict[str, Any]) -> list[str]:
    lineage = contract["lineage"]
    return [
        sys.executable,
        str(lock / "sources/intervene_dit_v22_custom_trace_suffix.py"),
        "--trace-dir",
        str(job["trace_dir"]),
        "--target-slot",
        str(job["class_slot"]),
        "--target-class-id",
        str(job["class_id"]),
        "--rollback-sampling-step",
        str(job["rollback_sampling_step"]),
        "--pilot-lock",
        str(contract["selection_lock"]["path"]),
        "--dit-root",
        str(lineage["dit_source"]["root"]),
        "--checkpoint",
        str(lineage["checkpoint"]["path"]),
        "--vae-snapshot",
        str(lineage["vae_snapshot"]["snapshot"]),
        "--outdir",
        str(job["outdir"]),
    ]


def output_record(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    manifest = self_hashed(path / "manifest.json", "identity_sha256")
    completion = self_hashed(path / "completion.json", "payload_sha256")
    if (
        manifest.get("target", {}).get("global_seed") != job["global_seed"]
        or manifest.get("target", {}).get("class_id") != job["class_id"]
        or manifest.get("target", {}).get("slot") != job["class_slot"]
        or manifest.get("rollback", {}).get("sampling_step_index_zero_based")
        != job["rollback_sampling_step"]
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
    ):
        raise RuntimeError(f"completed output does not match job: {path}")
    return {
        "job_index": job["job_index"],
        "outdir": str(path),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(path / "manifest.json"),
        "completion_payload_sha256": completion["payload_sha256"],
        "completion_file_sha256": sha256_file(path / "completion.json"),
    }


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard_count)")
    lock = args.lock.expanduser().resolve()
    manifest, contract = validate_lock(lock)
    jobs = [
        row
        for row in contract["jobs"]
        if int(row["job_index"]) % args.shard_count == args.shard_index
    ]
    if not jobs:
        raise RuntimeError("selected shard has no jobs")
    receipt_dir = RECEIPT_ROOT / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    output_records: list[dict[str, Any]] = []
    for ordinal, job in enumerate(jobs, start=1):
        command = command_for_job(lock, contract, job)
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "progress": f"{ordinal}/{len(jobs)}",
                    "job_index": job["job_index"],
                    "seed": job["global_seed"],
                    "class_id": job["class_id"],
                    "step": job["rollback_sampling_step"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(command, check=True)
        output_records.append(output_record(Path(job["outdir"]), job))
    receipt: dict[str, Any] = {
        "status": "complete",
        "artifact_kind": "DIT_V22_REPAIRABILITY_EXECUTION_SHARD_RECEIPT_V1",
        "execution_lock_identity_sha256": manifest["identity_sha256"],
        "execution_contract_identity_sha256": contract["identity_sha256"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "job_indices": [row["job_index"] for row in jobs],
        "outputs": output_records,
        "quality_scores_labels_features_or_attempt_selection_used": False,
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    if receipt_dir.exists():
        stored = self_hashed(receipt_dir / "receipt.json", "identity_sha256")
        if stored != receipt:
            raise RuntimeError(f"existing shard receipt differs: {receipt_dir}")
        print(f"validated existing receipt: {receipt_dir}")
        return
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{receipt_dir.name}.tmp-", dir=RECEIPT_ROOT))
    try:
        write_json(staging / "receipt.json", receipt)
        os.replace(staging, receipt_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    print(json.dumps({"status": "complete", "receipt": str(receipt_dir)}, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
