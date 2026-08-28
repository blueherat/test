#!/usr/bin/env python3
"""Freeze the prospective blind-review builder, qualification gate, and analyzer.

This source-only lock must be created before any prospective endpoint is opened
for external judging and before any real review response exists.  It binds the
already-frozen V1.2 sampler protocol to the exact three downstream programs and
their fixed review/statistical contract.  It never opens a PNG, selector row,
review response, or private blind mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2"
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_v22_transient_escape_blind_evaluation_lock_v1_1"
SOURCES = (
    "prepare_dit_v22_transient_escape_blind_review.py",
    "check_dit_v22_transient_escape_reviewer_qualification.py",
    "analyze_dit_v22_transient_escape_blind_review.py",
)
UPSTREAM_LOCK_ID = "cd8154479f5f6f883ae21d6657a61ec91ff6d2c77f569e18ea589d83517671a9"
UPSTREAM_PROTOCOL_ID = "54b11c1ebb6e310c73bb14e27c18e0f1810b5598212e2dc0c9be915f861155c1"
LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_BLIND_EVALUATION_LOCK_V1_1"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_self_hashed(path: Path) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get("identity_sha256")
    payload = dict(value)
    payload.pop("identity_sha256", None)
    if observed != canonical_sha256(payload):
        raise RuntimeError(f"self hash failed: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "name": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_upstream(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("invalid upstream prospective lock")
    manifest = load_self_hashed(root / "manifest.json")
    protocol = load_self_hashed(root / "protocol.json")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("upstream prospective lock exact tree changed")
    for name, row in records.items():
        path = root / str(name)
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"upstream prospective lock member changed: {name}")
    if (
        manifest.get("identity_sha256") != UPSTREAM_LOCK_ID
        or manifest.get("protocol_identity_sha256") != UPSTREAM_PROTOCOL_ID
        or protocol.get("identity_sha256") != UPSTREAM_PROTOCOL_ID
        or protocol.get("status") != "EXECUTION_READY_UNOBSERVED_PROSPECTIVE_SUFFIXES"
    ):
        raise RuntimeError("upstream prospective identity changed")
    return manifest, protocol


def freeze(args: argparse.Namespace) -> None:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite analysis lock: {output}")
    upstream_manifest, upstream_protocol = validate_upstream(args.upstream)
    source_paths = [ROOT / "experiments" / name for name in SOURCES]
    if any(path.is_symlink() or not path.is_file() for path in source_paths):
        raise RuntimeError("one or more blind-evaluation sources are missing or symlinks")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        source_root = staging / "sources"
        source_root.mkdir()
        for path in source_paths:
            shutil.copy2(path, source_root / path.name)

        protocol: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_V22_TRANSIENT_ESCAPE_BLIND_EVALUATION_PROTOCOL_V1_1",
            "status": "FROZEN_BEFORE_ANY_REAL_REVIEW_OR_ENDPOINT_OPENING",
            "upstream_lock_identity_sha256": upstream_manifest["identity_sha256"],
            "upstream_protocol_identity_sha256": upstream_protocol["identity_sha256"],
            "source_sha256": {
                path.name: sha256_file(source_root / path.name) for path in source_paths
            },
            "review_axis_per_reviewer": {
                "absolute": 640,
                "attempt0_vs_fresh_pairs": 512,
                "qualification": 7,
                "reviewers": 3,
            },
            "qualification": "at least 6 of 7 exact frozen-anchor severities before full review",
            "primary": {
                "population": "96 frozen class795 prefixes",
                "estimand": "mean_four_fresh(severity/2)-sealed_h10_max_A(severity/2)",
                "minimum_effect": 0.10,
                "monte_carlo_replicates": 1_000_000,
                "monte_carlo_seed": 2026082807,
                "bootstrap_replicates": 100_000,
                "bootstrap_seed": 2026082808,
            },
            "safety": {
                "margin": 0.05,
                "bootstrap_replicates": 100_000,
                "bootstrap_seed": 2026082809,
                "comparison": "sealed selected fresh versus exact uniform policy over the same four fresh branches",
            },
            "decision_order": [
                "reliability_or_attempt_index_failure_implies_INCONCLUSIVE",
                "interpretable_nonpositive_effect_or_material_point_harm_implies_STOP",
                "all_effect_inference_integrity_and_safety_gates_imply_NARROW_GO",
                "otherwise_INCONCLUSIVE",
            ],
            "invalid_vote_rule": "any absolute or pair valid=no is code-enforced as INCONCLUSIVE",
            "external_metrics_are_method_inputs": False,
            "same_pool_failed_primary_may_be_rescued_by_other_horizon_direction_class_or_embedding": False,
        }
        protocol["identity_sha256"] = canonical_sha256(protocol)
        write_json(staging / "protocol.json", protocol)

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": LOCK_KIND,
            "status": "complete",
            "scientific_role": "immutable_pre_review_external_evaluation_code_and_decision_contract",
            "upstream_lock_identity_sha256": upstream_manifest["identity_sha256"],
            "upstream_protocol_identity_sha256": upstream_protocol["identity_sha256"],
            "protocol_identity_sha256": protocol["identity_sha256"],
            "files": [file_record(path, staging) for path in files],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "identity_sha256": manifest["identity_sha256"],
                "protocol_identity_sha256": protocol["identity_sha256"],
                "source_sha256": protocol["source_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def self_test() -> None:
    payload = {"schema_version": 1, "status": "synthetic"}
    payload["identity_sha256"] = canonical_sha256(payload)
    observed = dict(payload)
    identity = observed.pop("identity_sha256")
    if identity != canonical_sha256(observed):
        raise AssertionError("canonical self-hash contract changed")
    if len(SOURCES) != 3 or len(set(SOURCES)) != 3:
        raise AssertionError("frozen source axis changed")
    print("blind-evaluation freezer self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    else:
        freeze(parsed)
