#!/usr/bin/env python3
"""Freeze the v2.2 repairability blind-review sources before GPU outputs exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from analyze_dit_v22_repairability_blind_review import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONSENSUS_DEFINITIONS,
)
from prepare_dit_v22_repairability_blind_review import (
    NAMESPACE,
    RESPONSE_COLUMNS,
    review_rubric,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION_LOCK = ROOT / "experiments/locks/dit_v22_repairability_pilot_lock_v1_2"
EXECUTION_LOCK = ROOT / "experiments/locks/dit_v22_repairability_execution_source_lock_v1"
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_v22_repairability_review_source_lock_v1"
EXPECTED_SELECTION_MANIFEST = "16acd0bffda207ed73ef78a62909e53997bef68baae66cdffedede1bb207fbd0"
EXPECTED_SELECTION_PROTOCOL = "f39c5a8bfbbc129d6e80ca5e38a07dfd886c6c41faff15337042127e78b3ae77"
EXPECTED_EXECUTION_MANIFEST = "c71ac783f2f72b9ec599b20ec7134c0ea1ebad642dbb1155fc7d873cc63d1cb6"
EXPECTED_EXECUTION_CONTRACT = "e5a585dd2a4c850a9543192d8dac69c3d9aebfa41171263e377f515e5c8e52bf"
ARTIFACT_ROOT = Path("/data/users/zhoushunyu/eqvae/cross_scale_evidence")
OUTPUT_ROOT = ARTIFACT_ROOT / "dit_v22_repairability_pilot_v1_2_outputs"
RECEIPT_ROOT = ARTIFACT_ROOT / "dit_v22_repairability_pilot_v1_2_receipts"
DELIVERY = ARTIFACT_ROOT / "dit_v22_repairability_pilot_v1_2_blind_review_v1_delivery"
PRIVATE = ARTIFACT_ROOT / "dit_v22_repairability_pilot_v1_2_blind_review_v1_private"
REVIEWS = ARTIFACT_ROOT / "dit_v22_repairability_pilot_v1_2_blind_review_v1_reviews"
ANALYSIS = ARTIFACT_ROOT / "dit_v22_repairability_pilot_v1_2_blind_review_v1_analysis"
SOURCE_NAMES = (
    "prepare_dit_v22_repairability_blind_review.py",
    "analyze_dit_v22_repairability_blind_review.py",
    "freeze_dit_v22_repairability_review_sources.py",
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
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular JSON: {path}")
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
        raise RuntimeError(f"self hash changed: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_upstream() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection_manifest = self_hashed(SELECTION_LOCK / "manifest.json", "identity_sha256")
    selection_protocol = self_hashed(SELECTION_LOCK / "protocol.json", "identity_sha256")
    execution_manifest = self_hashed(EXECUTION_LOCK / "manifest.json", "identity_sha256")
    execution_contract = self_hashed(EXECUTION_LOCK / "execution_contract.json", "identity_sha256")
    observed = (
        selection_manifest.get("identity_sha256"),
        selection_protocol.get("identity_sha256"),
        execution_manifest.get("identity_sha256"),
        execution_contract.get("identity_sha256"),
    )
    expected = (
        EXPECTED_SELECTION_MANIFEST,
        EXPECTED_SELECTION_PROTOCOL,
        EXPECTED_EXECUTION_MANIFEST,
        EXPECTED_EXECUTION_CONTRACT,
    )
    if observed != expected:
        raise RuntimeError(f"upstream lock identity changed: {observed}")
    if (
        selection_manifest.get("protocol_identity_sha256") != EXPECTED_SELECTION_PROTOCOL
        or execution_manifest.get("execution_contract_identity_sha256") != EXPECTED_EXECUTION_CONTRACT
        or execution_contract.get("selection_lock", {}).get("manifest_identity_sha256")
        != EXPECTED_SELECTION_MANIFEST
    ):
        raise RuntimeError("upstream lock cross-binding changed")
    return selection_manifest, selection_protocol, execution_manifest, execution_contract


def validate_absence() -> dict[str, bool]:
    paths = {
        "formal_gpu_output_root": OUTPUT_ROOT,
        "formal_execution_receipt_root": RECEIPT_ROOT,
        "blind_delivery": DELIVERY,
        "private_mapping": PRIVATE,
        "review_responses": REVIEWS,
        "analysis": ANALYSIS,
    }
    observed = {name: not os.path.lexists(path) for name, path in paths.items()}
    if not all(observed.values()):
        present = [name for name, absent in observed.items() if not absent]
        raise RuntimeError(f"freeze must precede all formal outputs/review artifacts: {present}")
    return observed


def build_contract(source_records: list[dict[str, Any]], absence: dict[str, bool]) -> dict[str, Any]:
    locked_root = DEFAULT_OUTPUT.resolve()
    reviews = [REVIEWS / f"reviewer_{index}.csv" for index in range(1, 4)]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "status": "REVIEW_EXECUTION_READY_BEFORE_FORMAL_GPU_OUTPUTS",
        "artifact_kind": "DIT_V22_REPAIRABILITY_REVIEW_CONTRACT_V1",
        "upstream": {
            "selection_lock": {
                "path": str(SELECTION_LOCK.resolve()),
                "manifest_identity_sha256": EXPECTED_SELECTION_MANIFEST,
                "protocol_identity_sha256": EXPECTED_SELECTION_PROTOCOL,
            },
            "execution_lock": {
                "path": str(EXECUTION_LOCK.resolve()),
                "manifest_identity_sha256": EXPECTED_EXECUTION_MANIFEST,
                "contract_identity_sha256": EXPECTED_EXECUTION_CONTRACT,
            },
        },
        "absence_at_freeze": absence,
        "fixed_paths": {
            "formal_gpu_outputs": str(OUTPUT_ROOT),
            "formal_execution_receipts": str(RECEIPT_ROOT),
            "blind_delivery": str(DELIVERY),
            "private_mapping": str(PRIVATE),
            "review_responses_root": str(REVIEWS),
            "reviewer_responses": [str(path) for path in reviews],
            "analysis": str(ANALYSIS),
        },
        "blind_pack": {
            "job_count_required_and_validated": 32,
            "fresh_attempts_per_job": 4,
            "comparison_count": 128,
            "native_left_right_png_count": 256,
            "reviewer_count": 3,
            "deterministic_randomization_namespace": NAMESPACE,
            "delivery_and_private_mapping_physically_separate": True,
            "mapping_unsealed_only_after_three_complete_responses": True,
            "forbidden_delivery_metadata": [
                "seed/class/role/rollback-step/fresh-attempt mapping",
                "internal scores or alarms",
                "prior visual labels",
                "FID or endpoint representations",
            ],
        },
        "response_schema": list(RESPONSE_COLUMNS),
        "response_values": {
            "quality": ["clean_good", "mild_or_uncertain", "clear_bad"],
            "boolean": ["true", "false"],
            "identity_composition_preserved": ["yes", "no", "uncertain"],
            "preferred_side": ["left", "right", "tie"],
            "localized_reason_nonempty": True,
        },
        "rubric": review_rubric(),
        "consensus": CONSENSUS_DEFINITIONS,
        "analysis": {
            "fixed_seed_path_cluster_bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "interval": "percentile_95_CI",
                "cluster": "original selected path",
            },
            "required_summaries": [
                "role by rollback step",
                "each original path's eight-attempt rates",
                "baseline repair-opportunity rates",
                "joint-E-and-B minus matched-B-only descriptive differences",
                "three-reviewer agreement",
            ],
            "claim_limit": "exploratory observational effect-modification only; no causal, deployment, rollback-authorization, FID, or e-process-guarantee claim",
        },
        "source_records": source_records,
        "canonical_commands": {
            "prepare": [
                "python",
                str(locked_root / "sources/prepare_dit_v22_repairability_blind_review.py"),
                "--selection-lock",
                str(SELECTION_LOCK.resolve()),
                "--execution-lock",
                str(EXECUTION_LOCK.resolve()),
                "--receipts",
                str(RECEIPT_ROOT),
                "--delivery",
                str(DELIVERY),
                "--private",
                str(PRIVATE),
            ],
            "analyze": [
                "python",
                str(locked_root / "sources/analyze_dit_v22_repairability_blind_review.py"),
                "--delivery",
                str(DELIVERY),
                "--mapping",
                str(PRIVATE / "sealed_mapping.json"),
                "--reviewers",
                *[str(path) for path in reviews],
                "--output",
                str(ANALYSIS),
            ],
        },
    }
    contract["identity_sha256"] = canonical_sha256(contract)
    return contract


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid review source lock: {root}")
    manifest = self_hashed(root / "manifest.json", "identity_sha256")
    contract = self_hashed(root / "review_contract.json", "identity_sha256")
    if (
        manifest.get("artifact_kind") != "DIT_V22_REPAIRABILITY_REVIEW_SOURCE_LOCK_V1"
        or manifest.get("status") != "complete"
        or manifest.get("review_contract_identity_sha256") != contract["identity_sha256"]
        or contract.get("status") != "REVIEW_EXECUTION_READY_BEFORE_FORMAL_GPU_OUTPUTS"
    ):
        raise RuntimeError("review source lock contract changed")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != observed:
        raise RuntimeError("review source lock exact tree changed")
    for name, record in records.items():
        path = root / name
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"review source lock member changed: {name}")
    for source in contract.get("source_records", []):
        locked = root / source["locked_name"]
        if sha256_file(locked) != source.get("sha256"):
            raise RuntimeError(f"contract source binding changed: {locked}")
    return manifest, contract


def freeze(output: Path) -> None:
    output = output.expanduser().resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("review source lock path is fixed")
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite review source lock: {output}")
    validate_upstream()
    absence = validate_absence()
    sources = [ROOT / "experiments" / name for name in SOURCE_NAMES]
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"missing regular source: {source}")
    source_records = [
        {
            "name": source.name,
            "origin": str(source.resolve()),
            "locked_name": f"sources/{source.name}",
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for source in sources
    ]
    contract = build_contract(source_records, absence)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (staging / "sources").mkdir()
        for source in sources:
            shutil.copyfile(source, staging / "sources" / source.name)
        write_json(staging / "review_contract.json", contract)
        manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_REPAIRABILITY_REVIEW_SOURCE_LOCK_V1",
            "selection_lock_identity_sha256": EXPECTED_SELECTION_MANIFEST,
            "selection_protocol_identity_sha256": EXPECTED_SELECTION_PROTOCOL,
            "execution_lock_identity_sha256": EXPECTED_EXECUTION_MANIFEST,
            "execution_contract_identity_sha256": EXPECTED_EXECUTION_CONTRACT,
            "review_contract_identity_sha256": contract["identity_sha256"],
            "files": [
                {
                    "name": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        validate_lock(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest, contract = validate_lock(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "lock": str(output),
                "lock_identity_sha256": manifest["identity_sha256"],
                "review_contract_identity_sha256": contract["identity_sha256"],
                "source_sha256": {row["name"]: row["sha256"] for row in source_records},
            },
            indent=2,
            sort_keys=True,
        )
    )


def self_test() -> None:
    if canonical_sha256({"b": 2, "a": 1}) != canonical_sha256({"a": 1, "b": 2}):
        raise AssertionError("canonical hash ordering changed")
    if len(RESPONSE_COLUMNS) != 10 or BOOTSTRAP_REPLICATES != 50_000:
        raise AssertionError("frozen review constants changed")
    if DEFAULT_OUTPUT.exists():
        validate_lock(DEFAULT_OUTPUT)
    print("repairability review-source freezer self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
    elif args.validate_only:
        manifest, contract = validate_lock(args.output)
        print(json.dumps({"lock_identity_sha256": manifest["identity_sha256"], "review_contract_identity_sha256": contract["identity_sha256"]}, indent=2, sort_keys=True))
    else:
        freeze(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
