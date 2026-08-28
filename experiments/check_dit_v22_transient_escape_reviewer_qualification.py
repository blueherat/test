#!/usr/bin/env python3
"""Fail-closed pre-review qualification gate for the DiT-v2.2 blind review.

This program reads exactly three candidate qualification response CSVs.  It
does not accept or inspect absolute-image or preservation-pair responses.  It
first authenticates the public qualification templates and their exact Q axis,
then authenticates the frozen lock, internal selector product, and physically
separate private mapping before using the seven private gold severities.

The sealed output contains aggregate pass/fail counts and input hashes only.
It never contains a per-Q answer, gold label, image/pair identity, job, seed,
attempt, slot, or method mapping.  The main analyzer must independently repeat
qualification; this receipt is only a gate before formal 640+512 review work.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2"
DEFAULT_INTERNAL_PRODUCT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_internal_v1"
)
DEFAULT_PUBLIC = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_blind_review_v1_public"
)
DEFAULT_PRIVATE = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_blind_review_v1_private"
)

LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_LOCK_V1_2"
LOCK_ID = "cd8154479f5f6f883ae21d6657a61ec91ff6d2c77f569e18ea589d83517671a9"
PROTOCOL_ID = "54b11c1ebb6e310c73bb14e27c18e0f1810b5598212e2dc0c9be915f861155c1"
PRODUCT_KIND = "DIT_V22_TRANSIENT_ESCAPE_INTERNAL_PRODUCT_V1"
PUBLIC_KIND = "DIT_V22_TRANSIENT_ESCAPE_BLIND_DELIVERY_V1"
PRIVATE_KIND = "DIT_V22_TRANSIENT_ESCAPE_PRIVATE_MAPPING_V1"
RECEIPT_KIND = "DIT_V22_TRANSIENT_ESCAPE_REVIEWER_QUALIFICATION_RECEIPT_V1"

REVIEWERS = ("reviewer_1", "reviewer_2", "reviewer_3")
QUALIFICATION_COLUMNS = ("qualification_id", "severity", "valid", "reason")
QUALIFICATION_IDS = tuple(f"Q{index:03d}" for index in range(7))
SEVERITIES = {"0", "1", "2"}
VALID_VALUES = {"yes", "no"}
REQUIRED_CORRECT = 6


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hash(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return canonical_sha256(payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a regular JSON file: {path}")
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    if not isinstance(observed, str) or observed != self_hash(value, key):
        raise RuntimeError(f"self hash failed: {path}")
    return value


def safe_relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise RuntimeError("manifest member name is not a string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"unsafe manifest member name: {value!r}")
    return relative


def resolve_input_root(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise RuntimeError(f"{label} root must not be a symlink: {lexical}")
    root = lexical.resolve()
    if not root.is_dir() or any(member.is_symlink() for member in root.rglob("*")):
        raise RuntimeError(f"invalid {label} tree: {root}")
    return root


def expected_directories(relative_files: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for name in relative_files:
        parent = safe_relative(name).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def validate_manifest_tree(
    root: Path,
    *,
    unlisted_files: set[str],
) -> dict[str, Any]:
    """Validate a self-hashed manifest and its exact byte-addressed tree."""

    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    records: dict[str, dict[str, Any]] = {}
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list):
        raise RuntimeError(f"manifest file records are absent: {root}")
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid manifest member record: {root}")
        relative = safe_relative(raw.get("name")).as_posix()
        if relative in records:
            raise RuntimeError(f"duplicate manifest member: {relative}")
        records[relative] = raw
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in unlisted_files
    }
    if set(records) != actual_files:
        raise RuntimeError(f"sealed exact file tree changed: {root}")
    for name, record in records.items():
        path = root / name
        expected = {
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if record != expected:
            raise RuntimeError(f"sealed member changed: {path}")
    all_files = {*actual_files, *unlisted_files}
    actual_directories = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    }
    if actual_directories != expected_directories(all_files):
        raise RuntimeError(f"sealed exact directory tree changed: {root}")
    return manifest


def validate_public(root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest = validate_manifest_tree(root, unlisted_files={"manifest.json"})
    if (
        manifest.get("artifact_kind") != PUBLIC_KIND
        or manifest.get("status") != "complete"
        or manifest.get("absolute_image_count") != 640
        or manifest.get("preservation_pair_count") != 512
        or manifest.get("qualification_item_count") != 7
        or manifest.get("qualification_required_correct") != REQUIRED_CORRECT
        or manifest.get("reviewer_count") != len(REVIEWERS)
        or manifest.get("private_mapping_present") is not False
        or manifest.get("source_lineage_present") is not False
        or manifest.get("internal_method_assignment_present") is not False
    ):
        raise RuntimeError("public blind-delivery scope changed")
    axes: list[tuple[str, ...]] = []
    for reviewer in REVIEWERS:
        path = root / "templates" / f"{reviewer}_qualification.csv"
        rows, _ = read_qualification_csv(path, response=False)
        axis = tuple(row["qualification_id"] for row in rows)
        axes.append(axis)
    if any(axis != QUALIFICATION_IDS for axis in axes):
        raise RuntimeError("public qualification template Q axis changed")
    return manifest, axes[0]


def validate_lock(root: Path) -> dict[str, Any]:
    manifest = validate_manifest_tree(root, unlisted_files={"manifest.json"})
    if (
        manifest.get("artifact_kind") != LOCK_KIND
        or manifest.get("status") != "complete"
        or manifest.get("identity_sha256") != LOCK_ID
        or manifest.get("protocol_identity_sha256") != PROTOCOL_ID
    ):
        raise RuntimeError("wrong V1.2 prospective lock")
    protocol = load_self_hashed(root / "protocol.json", "identity_sha256")
    if protocol.get("identity_sha256") != PROTOCOL_ID:
        raise RuntimeError("wrong V1.2 prospective protocol")
    return manifest


def validate_internal(root: Path, lock_identity: str) -> tuple[dict[str, Any], str]:
    manifest = validate_manifest_tree(
        root,
        unlisted_files={"manifest.json", "completion.json"},
    )
    completion = load_self_hashed(root / "completion.json", "identity_sha256")
    selection_path = root / "sealed_selections.json"
    selection_sha = sha256_file(selection_path)
    if (
        manifest.get("artifact_kind") != PRODUCT_KIND
        or manifest.get("status") != "complete"
        or manifest.get("lock_identity_sha256") != lock_identity
        or manifest.get("protocol_identity_sha256") != PROTOCOL_ID
        or manifest.get("counts")
        != {"jobs": 128, "feature_rows": 384, "selection_rows": 128}
        or manifest.get("png_pixels_opened") is not False
        or completion.get("complete") is not True
        or completion.get("product_identity_sha256") != manifest["identity_sha256"]
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("sealed_selections_file_sha256") != selection_sha
        or completion.get("external_judging_may_begin_after_this_product") is not True
    ):
        raise RuntimeError("internal selector-product seal changed")
    return manifest, selection_sha


def validate_private(
    root: Path,
    *,
    public: Mapping[str, Any],
    public_root: Path,
    lock: Mapping[str, Any],
    lock_root: Path,
    internal: Mapping[str, Any],
    internal_root: Path,
    selection_sha: str,
    q_axis: Sequence[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    actual = list(root.rglob("*"))
    if actual != [root / "sealed_mapping.json"] or not actual[0].is_file():
        raise RuntimeError("private mapping exact tree changed")
    mapping_path = root / "sealed_mapping.json"
    mapping = load_self_hashed(mapping_path, "identity_sha256")
    if (
        mapping.get("artifact_kind") != PRIVATE_KIND
        or mapping.get("status") != "SEALED_UNTIL_THREE_REVIEWS_AND_QUALIFICATION_LOCKS"
        or mapping.get("public_delivery_identity_sha256") != public["identity_sha256"]
        or mapping.get("public_manifest_file_sha256")
        != sha256_file(public_root / "manifest.json")
        or mapping.get("prospective_lock_identity_sha256") != lock["identity_sha256"]
        or mapping.get("prospective_lock_manifest_file_sha256")
        != sha256_file(lock_root / "manifest.json")
        or mapping.get("prospective_protocol_identity_sha256") != PROTOCOL_ID
        or mapping.get("prospective_protocol_file_sha256")
        != sha256_file(lock_root / "protocol.json")
        or mapping.get("internal_product_identity_sha256") != internal["identity_sha256"]
        or mapping.get("internal_product_manifest_file_sha256")
        != sha256_file(internal_root / "manifest.json")
        or mapping.get("internal_product_completion_file_sha256")
        != sha256_file(internal_root / "completion.json")
        or mapping.get("sealed_selections_file_sha256") != selection_sha
    ):
        raise RuntimeError("private mapping public/lock/internal binding changed")
    rule = mapping.get("qualification_rule")
    if rule != {
        "correct_required": REQUIRED_CORRECT,
        "total": len(QUALIFICATION_IDS),
        "exact_severity_match": True,
        "ordinary_gold": 0,
        "clear_bad_gold": 2,
    }:
        raise RuntimeError("private qualification rule changed")
    rows = mapping.get("qualification_mapping")
    if not isinstance(rows, list) or len(rows) != len(QUALIFICATION_IDS):
        raise RuntimeError("private qualification mapping count changed")
    gold: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid private qualification row")
        qualification_id = row.get("qualification_id")
        severity = row.get("gold_severity")
        if (
            qualification_id not in q_axis
            or qualification_id in gold
            or severity not in {0, 2}
        ):
            raise RuntimeError("private qualification ID/gold changed")
        gold[str(qualification_id)] = int(severity)
    if tuple(sorted(gold)) != tuple(sorted(q_axis)):
        raise RuntimeError("private qualification axis differs from public Q axis")
    return mapping, gold


def read_qualification_csv(
    path: Path,
    *,
    response: bool,
) -> tuple[list[dict[str, str]], str]:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise RuntimeError(f"missing qualification CSV: {lexical}")
    raw = lexical.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"qualification CSV is not UTF-8: {lexical}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != QUALIFICATION_COLUMNS:
        raise RuntimeError(f"qualification CSV schema changed: {lexical}")
    rows = [dict(row) for row in reader]
    if len(rows) != len(QUALIFICATION_IDS) or any(None in row for row in rows):
        raise RuntimeError(f"qualification CSV row shape/count changed: {lexical}")
    ids = tuple(str(row["qualification_id"]).strip() for row in rows)
    if ids != QUALIFICATION_IDS or len(set(ids)) != len(ids):
        raise RuntimeError(f"qualification CSV Q axis changed: {lexical}")
    for row in rows:
        normalized = {key: str(row[key]).strip() for key in QUALIFICATION_COLUMNS}
        row.update(normalized)
        if response:
            if (
                row["severity"] not in SEVERITIES
                or row["valid"] not in VALID_VALUES
                or not row["reason"]
                or "\x00" in row["reason"]
            ):
                raise RuntimeError(f"invalid qualification response enumeration: {lexical}")
        elif any(row[key] for key in QUALIFICATION_COLUMNS[1:]):
            raise RuntimeError(f"public qualification template is not empty: {lexical}")
    return rows, hashlib.sha256(raw).hexdigest()


def score_response(
    rows: Sequence[Mapping[str, str]],
    gold: Mapping[str, int],
) -> int:
    if tuple(row["qualification_id"] for row in rows) != QUALIFICATION_IDS:
        raise RuntimeError("response Q axis changed before scoring")
    return sum(
        row["valid"] == "yes" and int(row["severity"]) == gold[row["qualification_id"]]
        for row in rows
    )


def build_receipt(
    *,
    public: Mapping[str, Any],
    public_manifest_sha: str,
    private: Mapping[str, Any],
    private_file_sha: str,
    lock: Mapping[str, Any],
    internal: Mapping[str, Any],
    responses: Mapping[str, tuple[Sequence[Mapping[str, str]], str]],
    gold: Mapping[str, int],
) -> dict[str, Any]:
    reviewers: dict[str, dict[str, Any]] = {}
    for reviewer in REVIEWERS:
        rows, input_sha = responses[reviewer]
        exact_count = score_response(rows, gold)
        reviewers[reviewer] = {
            "input_sha256": input_sha,
            "exact_count": exact_count,
            "total": len(QUALIFICATION_IDS),
            "pass": exact_count >= REQUIRED_CORRECT,
        }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RECEIPT_KIND,
        "status": "complete",
        "role": "PRE_FORMAL_REVIEW_QUALIFICATION_GATE_ONLY",
        "public_delivery_identity_sha256": public["identity_sha256"],
        "public_manifest_file_sha256": public_manifest_sha,
        "private_mapping_identity_sha256": private["identity_sha256"],
        "private_mapping_file_sha256": private_file_sha,
        "prospective_lock_identity_sha256": lock["identity_sha256"],
        "internal_product_identity_sha256": internal["identity_sha256"],
        "required_correct": REQUIRED_CORRECT,
        "reviewers": reviewers,
        "all_reviewers_pass": all(row["pass"] for row in reviewers.values()),
        "main_analyzer_must_independently_revalidate": True,
        "per_item_answers_or_gold_present": False,
        "private_generation_mapping_present": False,
    }
    receipt["identity_sha256"] = self_hash(receipt, "identity_sha256")
    return receipt


def write_json_atomic_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace output requires Linux renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(temporary), -100, os.fsencode(path), 1) != 0:
            code = ctypes.get_errno()
            if code == errno.EEXIST:
                raise FileExistsError(f"refusing to overwrite: {path}")
            raise OSError(code, os.strerror(code), path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_output_separate(output: Path, roots: Sequence[Path], inputs: Sequence[Path]) -> Path:
    output = output.expanduser().absolute()
    resolved_output = output.resolve(strict=False)
    for root in roots:
        if resolved_output == root or root in resolved_output.parents:
            raise RuntimeError("qualification receipt must not be written into an input seal")
    for path in inputs:
        if resolved_output == path.expanduser().absolute().resolve():
            raise RuntimeError("qualification receipt must not overwrite an input CSV")
    return output


def run(args: argparse.Namespace) -> None:
    public_root = resolve_input_root(args.public, "public blind delivery")
    public, q_axis = validate_public(public_root)

    response_paths = (args.reviewer_1, args.reviewer_2, args.reviewer_3)
    resolved_response_paths = [path.expanduser().absolute().resolve() for path in response_paths]
    if len(set(resolved_response_paths)) != len(REVIEWERS):
        raise RuntimeError("the three reviewer inputs must be distinct files")
    responses: dict[str, tuple[list[dict[str, str]], str]] = {}
    for reviewer, path in zip(REVIEWERS, response_paths):
        rows, file_hash = read_qualification_csv(path, response=True)
        if tuple(row["qualification_id"] for row in rows) != q_axis:
            raise RuntimeError(f"{reviewer} response differs from public Q axis")
        responses[reviewer] = (rows, file_hash)

    # Private gold is unavailable until the complete public/response Q axes pass.
    lock_root = resolve_input_root(args.lock, "prospective lock")
    lock = validate_lock(lock_root)
    internal_root = resolve_input_root(args.internal_product, "internal product")
    internal, selection_sha = validate_internal(internal_root, lock["identity_sha256"])
    private_root = resolve_input_root(args.private, "private blind mapping")
    private, gold = validate_private(
        private_root,
        public=public,
        public_root=public_root,
        lock=lock,
        lock_root=lock_root,
        internal=internal,
        internal_root=internal_root,
        selection_sha=selection_sha,
        q_axis=q_axis,
    )
    output = ensure_output_separate(
        args.output,
        (public_root, private_root, lock_root, internal_root),
        response_paths,
    )
    receipt = build_receipt(
        public=public,
        public_manifest_sha=sha256_file(public_root / "manifest.json"),
        private=private,
        private_file_sha=sha256_file(private_root / "sealed_mapping.json"),
        lock=lock,
        internal=internal,
        responses=responses,
        gold=gold,
    )
    write_json_atomic_noreplace(output, receipt)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "identity_sha256": receipt["identity_sha256"],
                "all_reviewers_pass": receipt["all_reviewers_pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def synthetic_rows(gold: Mapping[str, int], wrong: set[str]) -> list[dict[str, str]]:
    return [
        {
            "qualification_id": qualification_id,
            "severity": str(
                (gold[qualification_id] + 1) % 3
                if qualification_id in wrong
                else gold[qualification_id]
            ),
            "valid": "yes",
            "reason": "synthetic response",
        }
        for qualification_id in QUALIFICATION_IDS
    ]


def self_test() -> None:
    gold = {qualification_id: (2 if index >= 5 else 0) for index, qualification_id in enumerate(QUALIFICATION_IDS)}
    six = synthetic_rows(gold, {"Q000"})
    five = synthetic_rows(gold, {"Q000", "Q001"})
    invalid = synthetic_rows(gold, set())
    invalid[0]["valid"] = "no"
    if score_response(six, gold) != 6 or score_response(five, gold) != 5:
        raise AssertionError("qualification threshold scoring changed")
    if score_response(invalid, gold) != 6:
        raise AssertionError("valid=no must count as an incorrect qualification answer")
    with tempfile.TemporaryDirectory(prefix="transient-escape-qualification-selftest-") as name:
        root = Path(name)
        path = root / "response.csv"
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(QUALIFICATION_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(six)
        observed, file_hash = read_qualification_csv(path, response=True)
        if observed != six or file_hash != sha256_file(path):
            raise AssertionError("synthetic qualification CSV validation changed")
        responses = {
            reviewer: (six if reviewer != "reviewer_3" else five, hashlib.sha256(reviewer.encode()).hexdigest())
            for reviewer in REVIEWERS
        }
        identity = lambda token: {"identity_sha256": hashlib.sha256(token.encode()).hexdigest()}
        receipt = build_receipt(
            public=identity("public"),
            public_manifest_sha=hashlib.sha256(b"public manifest").hexdigest(),
            private=identity("private"),
            private_file_sha=hashlib.sha256(b"private file").hexdigest(),
            lock=identity("lock"),
            internal=identity("internal"),
            responses=responses,
            gold=gold,
        )
        if (
            receipt["reviewers"]["reviewer_1"]["pass"] is not True
            or receipt["reviewers"]["reviewer_3"]["pass"] is not False
            or receipt["all_reviewers_pass"] is not False
            or receipt["identity_sha256"] != self_hash(receipt, "identity_sha256")
        ):
            raise AssertionError("aggregate qualification receipt changed")
        allowed_reviewer_keys = {
            "input_sha256",
            "exact_count",
            "total",
            "pass",
        }
        if any(set(row) != allowed_reviewer_keys for row in receipt["reviewers"].values()):
            raise AssertionError("reviewer aggregate leaked a nonaggregate field")
        serialized = canonical_bytes(receipt).decode("utf-8")
        for forbidden in ("qualification_id", "gold_severity", "job_index", "attempt", "method_roles"):
            if forbidden in serialized:
                raise AssertionError(f"aggregate receipt leaked {forbidden}")
        output = root / "sealed_receipt.json"
        write_json_atomic_noreplace(output, receipt)
        if load_self_hashed(output, "identity_sha256") != receipt:
            raise AssertionError("synthetic sealed receipt round trip changed")
        try:
            write_json_atomic_noreplace(output, receipt)
        except FileExistsError:
            pass
        else:
            raise AssertionError("sealed receipt overwrite was not refused")
    print("transient-escape reviewer qualification self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--internal-product", type=Path, default=DEFAULT_INTERNAL_PRODUCT)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--reviewer-1", type=Path)
    parser.add_argument("--reviewer-2", type=Path)
    parser.add_argument("--reviewer-3", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    required = (args.reviewer_1, args.reviewer_2, args.reviewer_3, args.output)
    if any(value is None for value in required):
        raise SystemExit("--reviewer-1, --reviewer-2, --reviewer-3, and --output are required")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
