#!/usr/bin/env python3
"""Prepare the score/role-blind review pack for frozen v1.2 repairability jobs.

The program is fail closed: both frozen locks, every execution receipt, and all
32 completed output bundles are authenticated before any image is published.
Each of the four fresh endpoints from a job is paired with its attempt-0 replay,
giving 128 comparisons.  A domain-separated hash fixes comparison order,
anonymous IDs, and left/right placement.  Scientific roles and source lineage
exist only in a physically separate sealed mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_LOCK = ROOT / "experiments/locks/dit_v22_repairability_pilot_lock_v1_2"
DEFAULT_EXECUTION_LOCK = ROOT / "experiments/locks/dit_v22_repairability_execution_source_lock_v1"
DEFAULT_RECEIPTS = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_receipts"
)
NAMESPACE = "eqvae.dit.v22.repairability.blind-review.v1"
RESPONSE_COLUMNS = (
    "anonymous_pair_id",
    "left_quality",
    "right_quality",
    "left_blur",
    "right_blur",
    "left_topology",
    "right_topology",
    "identity_composition_preserved",
    "preferred_side",
    "localized_reason",
)


def review_rubric() -> dict[str, Any]:
    """Return the exact public rubric frozen with the review source."""
    return {
        "task": "Compare each anonymous LEFT/RIGHT endpoint at native resolution, relative to ordinary quality in this frozen model batch.",
        "quality_values": ["clean_good", "mild_or_uncertain", "clear_bad"],
        "clear_bad_rule": "Only an obvious defect below ordinary batch quality: conspicuous blur/melting/fusion, gross misregistration, or malformed/missing/duplicated/misattached anatomy or object structure. Do not call normal model roughness clear_bad.",
        "blur_fields": "true/false independently for LEFT and RIGHT; true for conspicuous blur, melting, smearing, or soft fusion.",
        "topology_fields": "true/false independently for LEFT and RIGHT; true for conspicuous malformed, duplicated, fused, missing, or misattached anatomy/object geometry.",
        "identity_composition_preserved": "yes only when the two sides preserve class/identity, object count, main pose, and composition; otherwise no, or uncertain when genuinely ambiguous.",
        "preferred_side": "left/right/tie by overall visible quality; do not infer how either side was produced.",
        "localized_reason": "required short localized explanation for every pair.",
        "review_order": "Open both native PNGs at 100% first; pair images and sheets are navigation aids.",
        "forbidden_context": [
            "any production provenance or hidden experimental assignment",
            "any internal measurement, prior judgment, or other reviewer response",
            "any external representation or batch-level metric",
        ],
    }


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
        raise RuntimeError(f"invalid self hash: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def hidden_key(domain: str, *parts: Any) -> str:
    payload = "\0".join([NAMESPACE, domain, *map(str, parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_lock(root: Path, expected_kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"invalid frozen lock: {root}")
    manifest = self_hashed(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != expected_kind or manifest.get("status") != "complete":
        raise RuntimeError(f"wrong or incomplete frozen lock: {root}")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != observed:
        raise RuntimeError(f"frozen lock exact tree changed: {root}")
    for name, record in records.items():
        path = root / name
        if (
            path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"frozen lock member changed: {path}")
    document_name = "protocol.json" if "PILOT_LOCK" in expected_kind else "execution_contract.json"
    document = self_hashed(root / document_name, "identity_sha256")
    expected_identity_key = (
        "protocol_identity_sha256" if document_name == "protocol.json" else "execution_contract_identity_sha256"
    )
    if document["identity_sha256"] != manifest.get(expected_identity_key):
        raise RuntimeError(f"frozen document identity changed: {root}")
    return manifest, document


def validate_receipts(
    root: Path, execution_manifest: dict[str, Any], contract: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"all execution receipts are required before review: {root}")
    expected = {int(row["job_index"]) for row in contract["jobs"]}
    observed: dict[int, dict[str, Any]] = {}
    receipt_paths = sorted(root.glob("*/receipt.json"))
    if not receipt_paths:
        raise RuntimeError("no repairability execution receipts found")
    for path in receipt_paths:
        receipt = self_hashed(path, "identity_sha256")
        if (
            receipt.get("status") != "complete"
            or receipt.get("artifact_kind") != "DIT_V22_REPAIRABILITY_EXECUTION_SHARD_RECEIPT_V1"
            or receipt.get("execution_lock_identity_sha256") != execution_manifest["identity_sha256"]
            or receipt.get("execution_contract_identity_sha256") != contract["identity_sha256"]
            or receipt.get("quality_scores_labels_features_or_attempt_selection_used") is not False
        ):
            raise RuntimeError(f"invalid execution receipt: {path}")
        indices = receipt.get("job_indices")
        outputs = receipt.get("outputs")
        if not isinstance(indices, list) or not isinstance(outputs, list) or len(indices) != len(outputs):
            raise RuntimeError(f"malformed execution receipt: {path}")
        output_by_job = {int(row.get("job_index")): row for row in outputs}
        if set(map(int, indices)) != set(output_by_job):
            raise RuntimeError(f"receipt job/output axis differs: {path}")
        for index in map(int, indices):
            if index in observed:
                raise RuntimeError(f"duplicate execution receipt for job {index}")
            observed[index] = output_by_job[index]
    if set(observed) != expected or len(expected) != 32:
        raise RuntimeError(f"all 32 unique jobs must be receipted: got {len(observed)}")
    return observed


def inspect_png(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular PNG: {path}")
    if expected.get("sha256") != sha256_file(path) or expected.get("bytes") != path.stat().st_size:
        raise RuntimeError(f"PNG file identity changed: {path}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
            raise RuntimeError(f"native endpoint contract changed: {path}")


def validate_output(job: dict[str, Any], receipt: dict[str, Any]) -> dict[int, Path]:
    root = Path(job["outdir"]).expanduser().resolve()
    if not root.is_dir() or root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid completed output bundle: {root}")
    manifest = self_hashed(root / "manifest.json", "identity_sha256")
    results = self_hashed(root / "results.json", "payload_sha256")
    completion = self_hashed(root / "completion.json", "payload_sha256")
    target = manifest.get("target", {})
    rollback = manifest.get("rollback", {})
    if (
        target.get("global_seed") != job["global_seed"]
        or target.get("class_id") != job["class_id"]
        or target.get("slot") != job["class_slot"]
        or rollback.get("sampling_step_index_zero_based") != job["rollback_sampling_step"]
        or manifest.get("attempt_ranking_or_selection") is not False
        or manifest.get("quality_scores_or_labels_used_by_runner") is not False
        or results.get("selection_performed") is not False
        or results.get("selected_attempt") is not None
        or results.get("quality_scores_or_features_computed") is not False
    ):
        raise RuntimeError(f"output scientific contract changed: {root}")
    if (
        receipt.get("outdir") != str(root)
        or receipt.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or receipt.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or receipt.get("completion_payload_sha256") != completion["payload_sha256"]
        or receipt.get("completion_file_sha256") != sha256_file(root / "completion.json")
    ):
        raise RuntimeError(f"receipt/output binding changed: {root}")
    if (
        completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("results_payload_sha256") != results["payload_sha256"]
        or completion.get("results_file_sha256") != sha256_file(root / "results.json")
        or completion.get("branch_count") != 5
        or results.get("branch_count") != 5
        or results.get("fresh_attempt_count") != 4
    ):
        raise RuntimeError(f"output completion binding changed: {root}")
    branches = results.get("branches")
    if not isinstance(branches, list) or [row.get("attempt_index") for row in branches] != list(range(5)):
        raise RuntimeError(f"output branch axis changed: {root}")
    pngs: dict[int, Path] = {}
    expected_files = {
        (root / "manifest.json").resolve(),
        (root / "results.json").resolve(),
        (root / "completion.json").resolve(),
        (root / "runner_source.py").resolve(),
    }
    if sha256_file(root / "runner_source.py") != manifest.get("runner_source", {}).get("sha256"):
        raise RuntimeError(f"runner snapshot changed: {root}")
    for attempt, summary in enumerate(branches):
        branch = f"attempt_{attempt:03d}"
        branch_json_path = root / "branches" / branch / "branch.json"
        branch_json = self_hashed(branch_json_path, "payload_sha256")
        png_record = branch_json.get("target_png", {})
        npz_record = branch_json.get("trace_npz", {})
        png = (root / str(png_record.get("relative_path"))).resolve()
        npz = (root / str(npz_record.get("relative_path"))).resolve()
        if root not in png.parents or root not in npz.parents:
            raise RuntimeError(f"branch path escaped output root: {root}")
        if (
            summary.get("branch") != branch
            or branch_json.get("branch") != branch
            or summary.get("attempt_index") != attempt
            or branch_json.get("attempt_index") != attempt
            or summary.get("branch_json_sha256") != sha256_file(branch_json_path)
            or summary.get("branch_payload_sha256") != branch_json["payload_sha256"]
            or summary.get("target_png") != png_record
            or summary.get("trace_npz_sha256") != npz_record.get("sha256")
        ):
            raise RuntimeError(f"branch binding changed: {root}/{branch}")
        inspect_png(png, png_record)
        if not npz.is_file() or npz.is_symlink() or sha256_file(npz) != npz_record.get("sha256"):
            raise RuntimeError(f"branch trace identity changed: {npz}")
        pngs[attempt] = png
        expected_files.update({branch_json_path.resolve(), png, npz})
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(f"output exact tree changed: {root}")
    return pngs


def make_pair(left: Path, right: Path, anonymous_id: str, output: Path) -> None:
    canvas = Image.new("RGB", (520, 282), "white")
    draw = ImageDraw.Draw(canvas)
    for index, source in enumerate((left, right)):
        with Image.open(source) as image:
            canvas.paste(image.convert("RGB"), (0 if index == 0 else 264, 0))
    draw.text((4, 261), f"{anonymous_id}  LEFT", fill="black")
    draw.text((268, 261), "RIGHT", fill="black")
    canvas.save(output)


def make_sheet(pair_paths: list[Path], output: Path) -> None:
    if len(pair_paths) != 4:
        raise ValueError("each review sheet must contain four anonymous pairs")
    canvas = Image.new("RGB", (1040, 2 * 282), "white")
    for index, path in enumerate(pair_paths):
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), ((index % 2) * 520, (index // 2) * 282))
    canvas.save(output)


def publish(args: argparse.Namespace) -> None:
    delivery = args.delivery.expanduser().resolve()
    private = args.private.expanduser().resolve()
    if delivery == private or delivery in private.parents or private in delivery.parents:
        raise RuntimeError("delivery and private mapping must be physically separate trees")
    if os.path.lexists(delivery) or os.path.lexists(private):
        raise RuntimeError("refusing to overwrite blind-review artifacts")
    selection_manifest, protocol = validate_lock(
        args.selection_lock, "DIT_V22_REPAIRABILITY_PILOT_LOCK_V1_2"
    )
    execution_manifest, contract = validate_lock(
        args.execution_lock, "DIT_V22_REPAIRABILITY_EXECUTION_SOURCE_LOCK_V1"
    )
    if (
        contract.get("selection_lock", {}).get("manifest_identity_sha256")
        != selection_manifest["identity_sha256"]
        or contract.get("selection_lock", {}).get("protocol_identity_sha256")
        != protocol["identity_sha256"]
        or len(contract.get("jobs", [])) != 32
        or contract.get("runner_contract", {}).get("fresh_attempts") != 4
    ):
        raise RuntimeError("selection/execution lock binding changed")
    receipts = validate_receipts(args.receipts, execution_manifest, contract)
    units: list[dict[str, Any]] = []
    for job in contract["jobs"]:
        pngs = validate_output(job, receipts[int(job["job_index"])])
        for fresh_attempt in range(1, 5):
            units.append({"job": job, "fresh_attempt": fresh_attempt, "pngs": pngs})
    if len(units) != 128:
        raise RuntimeError("expected exactly 128 baseline/fresh comparisons")
    units.sort(
        key=lambda unit: hidden_key(
            "order", unit["job"]["job_index"], unit["fresh_attempt"]
        )
    )
    delivery.parent.mkdir(parents=True, exist_ok=True)
    private.parent.mkdir(parents=True, exist_ok=True)
    delivery_stage = Path(tempfile.mkdtemp(prefix=f".{delivery.name}.tmp-", dir=delivery.parent))
    private_stage = Path(tempfile.mkdtemp(prefix=f".{private.name}.tmp-", dir=private.parent))
    try:
        native = delivery_stage / "native"
        pairs = delivery_stage / "pairs"
        sheets = delivery_stage / "sheets"
        native.mkdir()
        pairs.mkdir()
        sheets.mkdir()
        mapping: list[dict[str, Any]] = []
        for ordinal, unit in enumerate(units):
            anonymous_id = f"R{ordinal:04d}"
            job = unit["job"]
            fresh_attempt = int(unit["fresh_attempt"])
            baseline_side = (
                "left"
                if int(hidden_key("side", job["job_index"], fresh_attempt)[-2:], 16) % 2 == 0
                else "right"
            )
            fresh_side = "right" if baseline_side == "left" else "left"
            source_by_side = {
                baseline_side: unit["pngs"][0],
                fresh_side: unit["pngs"][fresh_attempt],
            }
            for side in ("left", "right"):
                shutil.copyfile(source_by_side[side], native / f"{anonymous_id}_{side}.png")
            make_pair(
                native / f"{anonymous_id}_left.png",
                native / f"{anonymous_id}_right.png",
                anonymous_id,
                pairs / f"{anonymous_id}.png",
            )
            mapping.append(
                {
                    "anonymous_pair_id": anonymous_id,
                    "job_index": int(job["job_index"]),
                    "pair_index": int(job["pair_index"]),
                    "role": job["role"],
                    "global_seed": int(job["global_seed"]),
                    "class_id": int(job["class_id"]),
                    "class_slot": int(job["class_slot"]),
                    "rollback_sampling_step": int(job["rollback_sampling_step"]),
                    "fresh_attempt": fresh_attempt,
                    "baseline_side": baseline_side,
                    "fresh_side": fresh_side,
                    "baseline_source": str(unit["pngs"][0]),
                    "fresh_source": str(unit["pngs"][fresh_attempt]),
                    "baseline_source_sha256": sha256_file(unit["pngs"][0]),
                    "fresh_source_sha256": sha256_file(unit["pngs"][fresh_attempt]),
                }
            )
        for start in range(0, 128, 4):
            block = mapping[start : start + 4]
            make_sheet(
                [pairs / f"{row['anonymous_pair_id']}.png" for row in block],
                sheets / f"sheet_{start // 4:02d}.png",
            )
        write_json(delivery_stage / "rubric.json", review_rubric())
        for reviewer in range(1, 4):
            with (delivery_stage / f"reviewer_{reviewer}_response.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
                writer.writeheader()
                for row in mapping:
                    writer.writerow({"anonymous_pair_id": row["anonymous_pair_id"]})
        delivery_manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_REPAIRABILITY_SCORE_ROLE_BLIND_DELIVERY_V1",
            "anonymous_pair_count": 128,
            "native_image_count": 256,
            "pair_image_count": 128,
            "sheet_count": 32,
            "reviewer_response_template_count": 3,
            "private_mapping_present": False,
            "nonvisual_measurements_present": False,
            "randomization_namespace_sha256": hashlib.sha256(NAMESPACE.encode()).hexdigest(),
            "files": [
                {
                    "name": path.relative_to(delivery_stage).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(delivery_stage.rglob("*"))
                if path.is_file()
            ],
        }
        delivery_manifest["identity_sha256"] = canonical_sha256(delivery_manifest)
        write_json(delivery_stage / "manifest.json", delivery_manifest)
        private_payload: dict[str, Any] = {
            "status": "SEALED_UNTIL_THREE_COMPLETE_REVIEWS",
            "artifact_kind": "DIT_V22_REPAIRABILITY_PRIVATE_MAPPING_V1",
            "delivery_identity_sha256": delivery_manifest["identity_sha256"],
            "selection_lock_identity_sha256": selection_manifest["identity_sha256"],
            "selection_protocol_identity_sha256": protocol["identity_sha256"],
            "execution_lock_identity_sha256": execution_manifest["identity_sha256"],
            "execution_contract_identity_sha256": contract["identity_sha256"],
            "comparison_count": 128,
            "job_count": 32,
            "fresh_attempts_per_job": 4,
            "mapping": mapping,
            "scores_labels_or_external_features_used_for_pack_order_or_sides": False,
        }
        private_payload["identity_sha256"] = canonical_sha256(private_payload)
        write_json(private_stage / "sealed_mapping.json", private_payload)
        os.replace(delivery_stage, delivery)
        os.replace(private_stage, private)
    except BaseException:
        shutil.rmtree(delivery_stage, ignore_errors=True)
        shutil.rmtree(private_stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "delivery": str(delivery),
                "private": str(private),
                "delivery_identity_sha256": delivery_manifest["identity_sha256"],
                "comparisons": 128,
                "jobs_validated": 32,
            },
            indent=2,
            sort_keys=True,
        )
    )


def self_test() -> None:
    keys = [hidden_key("order", job, attempt) for job in range(32) for attempt in range(1, 5)]
    if len(keys) != 128 or len(set(keys)) != 128:
        raise AssertionError("anonymous ordering keys collide")
    sides = [int(hidden_key("side", job, attempt)[-2:], 16) % 2 for job in range(32) for attempt in range(1, 5)]
    if not 40 <= sum(sides) <= 88:
        raise AssertionError("deterministic side allocation is pathologically imbalanced")
    if len(RESPONSE_COLUMNS) != 10 or RESPONSE_COLUMNS[0] != "anonymous_pair_id":
        raise AssertionError("review response contract changed")
    print("repairability blind-review preparation self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, default=DEFAULT_SELECTION_LOCK)
    parser.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION_LOCK)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--delivery", type=Path, required=False)
    parser.add_argument("--private", type=Path, required=False)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.delivery is None or args.private is None:
        raise SystemExit("--delivery and --private are required unless --self-test is used")
    publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
