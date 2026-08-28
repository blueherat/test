#!/usr/bin/env python3
"""Fail-closed blind analysis for the DiT-v2.2 transient-escape pilot.

This program is deliberately downstream of two immutable products:

1. the frozen V1.2 prospective execution lock; and
2. the sealed sampler-internal h10 selector product.

It may then open one public blind seal, one private blind mapping seal, and
the three frozen review tables (absolute, preservation pair, qualification)
for each of exactly three reviewers.  External judgments are used only to
evaluate the already-sealed selector.  They can never change a prefix, branch,
horizon, score, or threshold.  The output contains aggregate statistics only;
no review-ID, prefix, seed, class/attempt join, or private mapping is emitted.

The blind-pack builder was intentionally not coupled to this analyzer.  The
small schema adaptation surface is isolated in ``adapt_public_items``,
``adapt_private_items``, and ``adapt_review_row``.  If the final pack uses
different field names, those three functions are the only intended patch
site; all identity, statistical, and decision rules below remain frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2"
DEFAULT_INTERNAL_PRODUCT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_internal_v1"
)
DEFAULT_PUBLIC_SEAL = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_blind_review_v1_public"
)
DEFAULT_PRIVATE_SEAL = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_blind_review_v1_private"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_blind_review_v1_analysis"
)

LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_LOCK_V1_2"
LOCK_IDENTITY = "cd8154479f5f6f883ae21d6657a61ec91ff6d2c77f569e18ea589d83517671a9"
PROTOCOL_IDENTITY = "54b11c1ebb6e310c73bb14e27c18e0f1810b5598212e2dc0c9be915f861155c1"
INTERNAL_PRODUCT_KIND = "DIT_V22_TRANSIENT_ESCAPE_INTERNAL_PRODUCT_V1"
RESULT_KIND = "DIT_V22_TRANSIENT_ESCAPE_BLIND_REVIEW_ANALYSIS_V1"

REVIEWER_COUNT = 3
PREFIX_COUNT = 128
PRIMARY_CLASS = 795
PRIMARY_PREFIX_COUNT = 96
ABSOLUTE_ITEMS_PER_REVIEWER = 640
PAIR_ITEMS_PER_REVIEWER = 512
ANCHORS_PER_REVIEWER = 7
REVIEWER_NAMES = ("reviewer_1", "reviewer_2", "reviewer_3")
FRESH_ATTEMPTS = (1, 2, 3, 4)
ALL_ATTEMPTS = (0, 1, 2, 3, 4)

PRIMARY_SEED = 2026082807
PRIMARY_REPLICATES = 1_000_000
PRIMARY_BOOTSTRAP_SEED = 2026082808
PRIMARY_BOOTSTRAP_REPLICATES = 100_000
SAFETY_BOOTSTRAP_SEED = 2026082809
SAFETY_BOOTSTRAP_REPLICATES = 100_000
PRIMARY_MINIMUM_EFFECT = 0.10
PRIMARY_ALPHA = 0.05
KAPPA_MINIMUM = 0.30
ATTEMPT_INDEX_MINIMUM_P = 0.01
SAFETY_MARGIN = 0.05

SEVERITIES = {0, 1, 2}
PRESERVATION_VALUES = {"yes", "no", "uncertain"}
PAIR_VALUES = {"left", "right", "tie"}
NORMALIZED_PAIR_VALUES = {"fresh_better", "attempt0_better", "tie"}
ITEM_KINDS = {"absolute", "pair", "anchor"}


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


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "name": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


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
    keys = [key for key in ("identity_sha256", "payload_sha256") if key in value]
    if len(keys) != 1:
        raise RuntimeError(f"expected exactly one self-hash field: {path}")
    key = keys[0]
    observed = value[key]
    payload = dict(value)
    payload.pop(key)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"self hash failed: {path}")
    return value


def load_member_json(path: Path) -> dict[str, Any]:
    """Load a manifest-hashed JSON member and verify a self hash when present."""

    value = load_json(path)
    self_hash_keys = [key for key in ("identity_sha256", "payload_sha256") if key in value]
    if self_hash_keys:
        if len(self_hash_keys) != 1:
            raise RuntimeError(f"ambiguous member self hash: {path}")
        key = self_hash_keys[0]
        payload = dict(value)
        observed = payload.pop(key)
        if not isinstance(observed, str) or canonical_sha256(payload) != observed:
            raise RuntimeError(f"member self hash failed: {path}")
    return value


def self_identity(value: Mapping[str, Any]) -> str:
    for key in ("identity_sha256", "payload_sha256"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
    raise RuntimeError("self-hashed object has no identity")


def validate_exact_directory(root: Path, *, manifest_name: str = "manifest.json") -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid sealed directory: {root}")
    manifest = load_self_hashed(root / manifest_name)
    records: dict[str, dict[str, Any]] = {}
    for record in manifest.get("files", manifest.get("members", [])):
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid member record in {root}")
        name = record.get("name", record.get("relative_path"))
        if not isinstance(name, str) or name in records:
            raise RuntimeError(f"invalid or duplicate member name in {root}")
        records[name] = record
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_name
    }
    if set(records) != actual:
        raise RuntimeError(f"sealed directory exact member set changed: {root}")
    for name, record in records.items():
        path = root / name
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if expected_bytes != path.stat().st_size or expected_hash != sha256_file(path):
            raise RuntimeError(f"sealed member changed: {path}")
    return manifest


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    manifest = validate_exact_directory(root)
    protocol = load_self_hashed(root / "protocol.json")
    config = load_json(root / "frozen_config.json")
    if (
        manifest.get("artifact_kind") != LOCK_KIND
        or manifest.get("status") != "complete"
        or manifest.get("identity_sha256") != LOCK_IDENTITY
        or manifest.get("protocol_identity_sha256") != PROTOCOL_IDENTITY
        or protocol.get("identity_sha256") != PROTOCOL_IDENTITY
        or protocol.get("status") != "EXECUTION_READY_UNOBSERVED_PROSPECTIVE_SUFFIXES"
        or len(protocol.get("jobs", [])) != PREFIX_COUNT
        or config.get("external_judge_only", {}).get("primary_population")
        != "the 96 frozen class795 prefixes; no posthoc bad-prefix filter"
        or config.get("external_judge_only", {}).get("primary_minimum_effect") != PRIMARY_MINIMUM_EFFECT
    ):
        raise RuntimeError("V1.2 prospective lock identity or frozen analysis contract changed")
    jobs = protocol.get("jobs", [])
    if (
        len({row.get("job_index") for row in jobs}) != PREFIX_COUNT
        or Counter(row.get("class_id") for row in jobs) != Counter({207: 16, 602: 16, 795: 96})
        or len({row.get("global_seed") for row in jobs}) != PREFIX_COUNT
    ):
        raise RuntimeError("V1.2 job axis changed")
    return manifest, protocol, config


def validate_internal_product(
    root: Path,
    *,
    lock_identity: str,
    protocol_identity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid internal selector product directory: {root}")
    manifest = load_self_hashed(root / "manifest.json")
    completion = load_self_hashed(root / "completion.json")
    expected_payloads = {
        "distance_matrices.json",
        "extractor_source.py",
        "features.csv",
        "features.json",
        "frozen_config.json",
        "input_inventory.json",
        "sealed_selections.json",
    }
    records = {row.get("name"): row for row in manifest.get("files", [])}
    actual_payloads = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "completion.json"}
    }
    if set(records) != expected_payloads or actual_payloads != expected_payloads:
        raise RuntimeError("internal selector product payload set changed")
    for name, record in records.items():
        path = root / name
        if (
            not isinstance(record, dict)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"internal selector product member changed: {path}")
    if (
        manifest.get("artifact_kind") != INTERNAL_PRODUCT_KIND
        or manifest.get("status") != "complete"
        or manifest.get("scientific_role")
        != "prospective_sampler_internal_selection_sealed_before_external_judging"
        or manifest.get("lock_identity_sha256") != lock_identity
        or manifest.get("protocol_identity_sha256") != protocol_identity
        or manifest.get("counts") != {"jobs": 128, "feature_rows": 384, "selection_rows": 128}
        or manifest.get("selector") != "step149_h10_argmax_fresh_mean_nonconformity"
        or manifest.get("attempt0_O_lowO_B_E_or_external_metric_computed") is not False
        or manifest.get("png_pixels_opened") is not False
        or manifest.get("all_outputs_retained") is not True
        or completion.get("complete") is not True
        or completion.get("product_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("sealed_selections_file_sha256")
        != sha256_file(root / "sealed_selections.json")
        or completion.get("external_judging_may_begin_after_this_product") is not True
    ):
        raise RuntimeError("internal selector exact seal changed")
    selections_raw = load_json_array(root / "sealed_selections.json")
    if len(selections_raw) != PREFIX_COUNT:
        raise RuntimeError("internal selector row count changed")
    selections: list[dict[str, Any]] = []
    for row in selections_raw:
        if not isinstance(row, dict):
            raise RuntimeError("invalid internal selector row")
        if (
            type(row.get("job_index")) is not int
            or type(row.get("class_id")) is not int
            or row.get("primary_horizon") != 10
            or row.get("selector") != "argmax_h10_fresh_mean_nonconformity"
            or row.get("max_attempt") not in FRESH_ATTEMPTS
            or row.get("medoid_attempt") not in FRESH_ATTEMPTS
            or row.get("hash_random_attempt") not in FRESH_ATTEMPTS
            or row.get("all_four_fresh_endpoints_must_remain_available_to_external_judge") is not True
        ):
            raise RuntimeError("invalid internal selector contract row")
        selections.append(dict(row))
    if (
        len({row["job_index"] for row in selections}) != PREFIX_COUNT
        or Counter(row["class_id"] for row in selections) != Counter({207: 16, 602: 16, 795: 96})
    ):
        raise RuntimeError("internal selector axis changed")
    return manifest, selections


def load_json_array(path: Path) -> list[Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a real JSON array: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise RuntimeError(f"expected JSON array: {path}")
    return value


class SealBundle:
    def __init__(self, root: Path, manifest: dict[str, Any], payloads: list[dict[str, Any]]):
        self.root = root
        self.manifest = manifest
        self.payloads = payloads
        self.identities = {
            self_identity(value)
            for value in [manifest, *payloads]
            if any(isinstance(value.get(key), str) for key in ("identity_sha256", "payload_sha256"))
        }


def load_seal_bundle(path: Path) -> SealBundle:
    path = path.expanduser().resolve()
    if path.is_dir():
        if not (path / "manifest.json").is_file():
            members = sorted(member for member in path.iterdir() if member.is_file())
            if len(members) != 1 or members[0].suffix.lower() != ".json":
                raise RuntimeError(
                    "a manifest-free seal directory must contain exactly one self-hashed JSON"
                )
            payload = load_self_hashed(members[0])
            return SealBundle(path, payload, [])
        manifest = validate_exact_directory(path)
        payloads = []
        for record in manifest.get("files", manifest.get("members", [])):
            name = record.get("name", record.get("relative_path"))
            member = path / str(name)
            if member.suffix.lower() == ".json":
                payloads.append(load_member_json(member))
        return SealBundle(path, manifest, payloads)
    manifest = load_self_hashed(path)
    payloads: list[dict[str, Any]] = []
    records = manifest.get("files", manifest.get("members", []))
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("invalid seal member record")
        name = record.get("name", record.get("relative_path"))
        member = path.parent / str(name)
        if (
            member.is_symlink()
            or not member.is_file()
            or record.get("bytes") != member.stat().st_size
            or record.get("sha256") != sha256_file(member)
        ):
            raise RuntimeError(f"seal member changed: {member}")
        if member.suffix.lower() == ".json":
            payloads.append(load_member_json(member))
    return SealBundle(path, manifest, payloads)


def unique_binding(bundle: SealBundle, aliases: Sequence[str]) -> str | None:
    observed: set[str] = set()
    for payload in [bundle.manifest, *bundle.payloads]:
        for alias in aliases:
            value = payload.get(alias)
            if isinstance(value, str):
                observed.add(value)
    if len(observed) > 1:
        raise RuntimeError(f"conflicting seal bindings for {aliases}: {sorted(observed)}")
    return next(iter(observed)) if observed else None


def candidate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "rows", "mapping", "review_items", "blind_items"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return [dict(row) for row in value]
    reviewers = payload.get("reviewers")
    if isinstance(reviewers, dict):
        rows: list[dict[str, Any]] = []
        for reviewer, value in reviewers.items():
            if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
                continue
            for row in value:
                current = dict(row)
                current.setdefault("reviewer", reviewer)
                rows.append(current)
        if rows:
            return rows
    return []


def normalize_kind(value: Any) -> str:
    token = str(value).strip().lower()
    aliases = {
        "absolute_image": "absolute",
        "absolute_endpoint": "absolute",
        "endpoint_absolute": "absolute",
        "pairwise": "pair",
        "paired": "pair",
        "hidden_anchor": "anchor",
        "gold_anchor": "anchor",
    }
    token = aliases.get(token, token)
    if token not in ITEM_KINDS:
        raise RuntimeError(f"unknown blind item kind: {value!r}")
    return token


def first_field(row: Mapping[str, Any], aliases: Sequence[str], *, required: bool = True) -> Any:
    values = [row[key] for key in aliases if key in row and row[key] not in (None, "")]
    if not values:
        if required:
            raise RuntimeError(f"missing required field aliases: {aliases}")
        return None
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError(f"conflicting alias values: {aliases}")
    return values[0]


def adapt_public_items(bundle: SealBundle) -> list[dict[str, Any]]:
    """Normalize only the public review-ID axis and item kinds.

    Keep pack-specific aliases here.  Public rows must not be used to recover
    jobs or attempts; those identities are private-seal responsibilities.
    """

    # Frozen builder V1 stores the public ID axis in empty, member-hashed CSV
    # templates and deliberately keeps all job/attempt lineage private.
    if bundle.root.is_dir() and (bundle.root / "templates").is_dir():
        normalized: list[dict[str, Any]] = []
        specs = (
            ("absolute", "absolute", "image_id", ABSOLUTE_ITEMS_PER_REVIEWER),
            ("preservation", "pair", "pair_id", PAIR_ITEMS_PER_REVIEWER),
            ("qualification", "anchor", "qualification_id", ANCHORS_PER_REVIEWER),
        )
        for reviewer in REVIEWER_NAMES:
            for filename_kind, item_kind, id_field, expected_count in specs:
                path = bundle.root / "templates" / f"{reviewer}_{filename_kind}.csv"
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"missing frozen public review template: {path}")
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None or id_field not in reader.fieldnames:
                        raise RuntimeError(f"public template ID field changed: {path}")
                    ids = [str(row.get(id_field, "")).strip() for row in reader]
                if (
                    len(ids) != expected_count
                    or len(set(ids)) != expected_count
                    or any(not value for value in ids)
                ):
                    raise RuntimeError(f"public template ID axis changed: {path}")
                normalized.extend(
                    {
                        "reviewer": reviewer,
                        "review_id": review_id,
                        "item_kind": item_kind,
                    }
                    for review_id in ids
                )
        return normalized

    # Compatibility adapter for a future builder that emits explicit public
    # item rows instead of member-hashed templates.
    sets = [candidate_rows(value) for value in [bundle.manifest, *bundle.payloads]]
    rows = max(sets, key=len, default=[])
    if not rows:
        raise RuntimeError("public seal contains neither frozen templates nor adaptable item rows")
    return [
        {
            "reviewer": str(first_field(row, ("reviewer", "reviewer_id"))).strip(),
            "review_id": str(first_field(row, ("review_id", "blind_review_id", "id"))).strip(),
            "item_kind": normalize_kind(
                first_field(row, ("item_kind", "review_kind", "kind", "task_type"))
            ),
        }
        for row in rows
    ]


def adapt_private_items(bundle: SealBundle) -> list[dict[str, Any]]:
    """Normalize the private mapping needed for the aggregate join.

    Keep pack-specific aliases here.  No normalized row is ever serialized.
    """

    # Frozen builder V1 uses three separate mapping axes without reviewer
    # names.  Every reviewer receives the same opaque IDs, so replicate the
    # authenticated mapping mechanically across the fixed reviewer axis.
    payloads = [bundle.manifest, *bundle.payloads]
    frozen = next(
        (
            value
            for value in payloads
            if isinstance(value.get("absolute_mapping"), list)
            and isinstance(value.get("preservation_mapping"), list)
            and isinstance(value.get("qualification_mapping"), list)
        ),
        None,
    )
    if frozen is not None:
        absolute = frozen["absolute_mapping"]
        pairs = frozen["preservation_mapping"]
        anchors = frozen["qualification_mapping"]
        if (
            len(absolute) != ABSOLUTE_ITEMS_PER_REVIEWER
            or len(pairs) != PAIR_ITEMS_PER_REVIEWER
            or len(anchors) != ANCHORS_PER_REVIEWER
            or not all(isinstance(row, dict) for row in [*absolute, *pairs, *anchors])
        ):
            raise RuntimeError("private frozen mapping counts or row types changed")
        base: list[dict[str, Any]] = []
        for row in absolute:
            base.append(
                {
                    "review_id": str(first_field(row, ("image_id", "review_id"))).strip(),
                    "item_kind": "absolute",
                    "job_index": int(first_field(row, ("job_index", "prefix_index"))),
                    "attempt_index": int(
                        first_field(row, ("attempt", "attempt_index", "physical_attempt"))
                    ),
                }
            )
        for row in pairs:
            base.append(
                {
                    "review_id": str(first_field(row, ("pair_id", "review_id"))).strip(),
                    "item_kind": "pair",
                    "pair_key": str(first_field(row, ("pair_id", "comparison_id"))),
                    "job_index": int(first_field(row, ("job_index", "prefix_index"))),
                    "fresh_attempt": int(first_field(row, ("fresh_attempt", "attempt"))),
                    "left_attempt": int(first_field(row, ("left_attempt",))),
                    "right_attempt": int(first_field(row, ("right_attempt",))),
                }
            )
        for row in anchors:
            base.append(
                {
                    "review_id": str(
                        first_field(row, ("qualification_id", "review_id"))
                    ).strip(),
                    "item_kind": "anchor",
                    "expected_severity": int(
                        first_field(row, ("gold_severity", "expected_severity"))
                    ),
                }
            )
        return [dict(row, reviewer=reviewer) for reviewer in REVIEWER_NAMES for row in base]

    # Compatibility adapter for a future builder with one already-expanded
    # mapping row per reviewer.
    sets = [candidate_rows(value) for value in payloads]
    rows = max(sets, key=len, default=[])
    if not rows:
        raise RuntimeError("private seal contains no adaptable mapping rows")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        kind = normalize_kind(first_field(row, ("item_kind", "review_kind", "kind", "task_type")))
        current: dict[str, Any] = {
            "reviewer": str(first_field(row, ("reviewer", "reviewer_id"))).strip(),
            "review_id": str(first_field(row, ("review_id", "blind_review_id", "id"))).strip(),
            "item_kind": kind,
        }
        if kind == "absolute":
            current["job_index"] = int(first_field(row, ("job_index", "prefix_index")))
            current["attempt_index"] = int(
                first_field(row, ("attempt_index", "physical_attempt", "attempt"))
            )
        elif kind == "anchor":
            current["expected_severity"] = int(
                first_field(row, ("expected_severity", "anchor_severity", "gold_severity"))
            )
        else:
            current.update(
                {
                    "pair_key": str(first_field(row, ("pair_id", "comparison_id"))),
                    "job_index": int(first_field(row, ("job_index", "prefix_index"))),
                    "fresh_attempt": int(first_field(row, ("fresh_attempt", "attempt"))),
                    "left_attempt": int(first_field(row, ("left_attempt",))),
                    "right_attempt": int(first_field(row, ("right_attempt",))),
                }
            )
        normalized.append(current)
    return normalized


def normalize_valid(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "valid", "ok"}:
        return True
    if token in {"0", "false", "no", "invalid", "unusable"}:
        return False
    raise RuntimeError(f"invalid validity enum: {value!r}")


def normalize_severity(value: Any) -> int:
    try:
        result = int(str(value).strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid severity: {value!r}") from exc
    if result not in SEVERITIES:
        raise RuntimeError(f"severity outside 0/1/2: {value!r}")
    return result


def normalize_preservation(value: Any) -> str:
    token = str(value).strip().lower()
    aliases = {"1": "yes", "true": "yes", "0": "no", "false": "no"}
    token = aliases.get(token, token)
    if token not in PRESERVATION_VALUES:
        raise RuntimeError(f"invalid preservation enum: {value!r}")
    return token


def normalize_ternary_flag(value: Any, *, field: str) -> str:
    try:
        return normalize_preservation(value)
    except RuntimeError as exc:
        raise RuntimeError(f"invalid {field} enum: {value!r}") from exc


def normalize_pair(value: Any) -> str:
    token = str(value).strip().lower()
    aliases = {
        "left_better": "left",
        "right_better": "right",
        "same": "tie",
        "equal": "tie",
        "neither": "tie",
    }
    token = aliases.get(token, token)
    if token not in PAIR_VALUES:
        raise RuntimeError(f"invalid pair enum: {value!r}")
    return token


def adapt_review_row(row: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Normalize one CSV response without exposing private identities."""

    current: dict[str, Any] = {
        "review_id": str(
            first_field(
                row,
                (
                    "review_id",
                    "blind_review_id",
                    "image_id",
                    "pair_id",
                    "qualification_id",
                    "id",
                ),
            )
        ).strip()
    }
    if kind in {"absolute", "anchor"}:
        current["severity"] = normalize_severity(first_field(row, ("severity", "defect_severity")))
        validity = first_field(
            row,
            ("absolute_valid", "anchor_valid", "valid", "is_valid"),
        )
        current["valid"] = normalize_valid(validity, default=False)
        reason = str(
            first_field(
                row,
                ("localized_reason", "reason")
                if kind == "absolute"
                else ("reason", "localized_reason"),
            )
        ).strip()
        if not reason:
            raise RuntimeError(f"empty {kind} review reason")
        if kind == "absolute":
            current["blur_fusion"] = normalize_ternary_flag(
                first_field(row, ("blur_fusion",)),
                field="blur_fusion",
            )
            current["topology_misalignment"] = normalize_ternary_flag(
                first_field(row, ("topology_misalignment",)),
                field="topology_misalignment",
            )
    else:
        validity = first_field(row, ("pair_valid", "valid", "is_valid"))
        current["valid"] = normalize_valid(validity, default=False)
        reason = str(first_field(row, ("reason", "localized_reason"))).strip()
        if not reason:
            raise RuntimeError("empty pair review reason")
        preserved = first_field(
            row,
            ("preserved", "preservation", "semantic_preservation"),
        )
        current["preservation"] = normalize_preservation(preserved)
        choice = first_field(
            row,
            ("preferred_side", "pair_choice", "pair_preference", "preference", "winner"),
        )
        current["pair_choice"] = normalize_pair(choice)
    return current


def validate_seal_bindings(
    public: SealBundle,
    private: SealBundle,
    *,
    lock_identity: str,
    internal_identity: str,
) -> None:
    lock_aliases = ("lock_identity_sha256", "prospective_lock_identity_sha256")
    internal_aliases = (
        "internal_product_identity_sha256",
        "selector_product_identity_sha256",
    )
    for name, bundle in (("public", public), ("private", private)):
        artifact = str(bundle.manifest.get("artifact_kind", "")).upper()
        status = str(bundle.manifest.get("status", "")).upper()
        if (
            not any(token in artifact for token in ("BLIND", "PRIVATE_MAPPING", "REVIEW"))
            or not any(token in status for token in ("COMPLETE", "SEALED"))
        ):
            raise RuntimeError(f"{name} blind seal kind/status changed")
    # Public delivery intentionally contains no lineage or selector identity.
    # All exact bindings live in the physically separate private seal.
    if unique_binding(public, lock_aliases) is not None or unique_binding(public, internal_aliases) is not None:
        raise RuntimeError("public blind delivery unexpectedly exposes private lineage")
    if unique_binding(private, lock_aliases) != lock_identity:
        raise RuntimeError("private blind seal lock binding changed")
    if unique_binding(private, internal_aliases) != internal_identity:
        raise RuntimeError("private blind seal internal-product binding changed")
    public_binding = unique_binding(
        private,
        (
            "public_delivery_identity_sha256",
            "public_seal_identity_sha256",
            "public_identity_sha256",
            "delivery_identity_sha256",
        ),
    )
    if public_binding not in public.identities:
        raise RuntimeError("private seal does not bind the exact public seal")


def read_review_csv(
    path: Path,
    *,
    id_field: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing real review CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or id_field not in reader.fieldnames:
            raise RuntimeError(f"review CSV lacks frozen ID field {id_field!r}: {path}")
        rows = [dict(row) for row in reader]
    ids = [str(row.get(id_field, "")).strip() for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"review CSV has empty or duplicate review IDs: {path}")
    normalized = [dict(row, review_id=review_id) for row, review_id in zip(rows, ids)]
    return normalized, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_pair_preference(
    pair_choice: str,
    *,
    fresh_attempt: int,
    left_attempt: int,
    right_attempt: int,
) -> str:
    """Resolve randomized left/right only after opening the private mapping."""

    if fresh_attempt not in FRESH_ATTEMPTS or {left_attempt, right_attempt} != {
        0,
        fresh_attempt,
    }:
        raise RuntimeError("pair mapping is not attempt0 versus its stated fresh attempt")
    if pair_choice == "tie":
        return "tie"
    preferred_attempt = left_attempt if pair_choice == "left" else right_attempt
    result = "fresh_better" if preferred_attempt == fresh_attempt else "attempt0_better"
    if result not in NORMALIZED_PAIR_VALUES:
        raise AssertionError("unreachable normalized pair preference")
    return result


def prevalidate_reviews_against_public(
    public_rows: Sequence[Mapping[str, Any]],
    review_paths: Mapping[str, Mapping[str, Path]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, Any]]]], list[dict[str, Any]]]:
    """Validate all nine response tables while only opaque public IDs are open."""

    public_ids: dict[tuple[str, str], set[str]] = {}
    for row in public_rows:
        reviewer = str(row["reviewer"])
        kind = str(row["item_kind"])
        review_id = str(row["review_id"])
        key = (reviewer, kind)
        ids = public_ids.setdefault(key, set())
        if not review_id or review_id in ids:
            raise RuntimeError("public review axis has an empty or duplicate opaque ID")
        ids.add(review_id)
    kind_specs = {
        "absolute": ("image_id", ABSOLUTE_ITEMS_PER_REVIEWER),
        "pair": ("pair_id", PAIR_ITEMS_PER_REVIEWER),
        "anchor": ("qualification_id", ANCHORS_PER_REVIEWER),
    }
    expected_public_keys = {
        (reviewer, kind) for reviewer in REVIEWER_NAMES for kind in kind_specs
    }
    if set(public_ids) != expected_public_keys or any(
        len(public_ids[(reviewer, kind)]) != expected_count
        for reviewer in REVIEWER_NAMES
        for kind, (_, expected_count) in kind_specs.items()
    ):
        raise RuntimeError("public reviewer/kind/count axis changed")
    if set(review_paths) != set(REVIEWER_NAMES):
        raise RuntimeError("supplied reviewer axis differs from the frozen public axis")

    observed_reviews: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    receipts: list[dict[str, Any]] = []
    for reviewer in REVIEWER_NAMES:
        if set(review_paths[reviewer]) != set(kind_specs):
            raise RuntimeError(f"{reviewer} must supply absolute, pair, and anchor CSVs")
        observed_reviews[reviewer] = {}
        for kind, (id_field, expected_count) in kind_specs.items():
            raw_rows, receipt = read_review_csv(
                review_paths[reviewer][kind],
                id_field=id_field,
            )
            responses: dict[str, dict[str, Any]] = {}
            for row in raw_rows:
                response = adapt_review_row(row, kind=kind)
                review_id = str(response["review_id"])
                if review_id in responses:
                    raise RuntimeError(f"duplicate adapted {kind} ID for {reviewer}")
                responses[review_id] = response
            if (
                len(responses) != expected_count
                or set(responses) != public_ids[(reviewer, kind)]
            ):
                raise RuntimeError(f"{kind} public review-ID axis mismatch for {reviewer}")
            observed_reviews[reviewer][kind] = responses
            receipt.update({"reviewer": reviewer, "item_kind": kind})
            receipts.append(receipt)
    return observed_reviews, receipts


def validate_and_join_reviews(
    public_rows: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    observed_reviews: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> tuple[
    dict[str, dict[tuple[int, int], dict[str, Any]]],
    dict[str, dict[tuple[int, int], dict[str, Any]]],
    dict[str, Any],
]:
    public_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    private_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for name, rows, target in (
        ("public", public_rows, public_by_key),
        ("private", private_rows, private_by_key),
    ):
        for row in rows:
            key = (
                str(row["reviewer"]),
                str(row["item_kind"]),
                str(row["review_id"]),
            )
            if key in target:
                raise RuntimeError(f"duplicate {name} review key")
            target[key] = row
    if set(public_by_key) != set(private_by_key):
        raise RuntimeError("public/private review-ID axes differ")
    reviewers = sorted({key[0] for key in public_by_key})
    if reviewers != sorted(REVIEWER_NAMES) or set(reviewers) != set(observed_reviews):
        raise RuntimeError("reviewer axis differs across seals and supplied CSVs")

    absolute_votes: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    pair_votes: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    reliability: dict[str, Any] = {}
    kind_specs = {
        "absolute": ABSOLUTE_ITEMS_PER_REVIEWER,
        "pair": PAIR_ITEMS_PER_REVIEWER,
        "anchor": ANCHORS_PER_REVIEWER,
    }
    for reviewer in reviewers:
        if set(observed_reviews[reviewer]) != set(kind_specs):
            raise RuntimeError(f"prevalidated axes missing for {reviewer}")
        observed_by_kind = observed_reviews[reviewer]
        private_by_kind: dict[str, dict[str, Mapping[str, Any]]] = {}
        for kind, expected_count in kind_specs.items():
            expected_ids = {
                review_id
                for current_reviewer, current_kind, review_id in public_by_key
                if current_reviewer == reviewer and current_kind == kind
            }
            private = {
                review_id: private_by_key[(reviewer, kind, review_id)]
                for review_id in expected_ids
            }
            if (
                len(expected_ids) != expected_count
                or set(observed_by_kind[kind]) != expected_ids
                or len(private) != expected_count
            ):
                raise RuntimeError(f"{kind} review-ID axis mismatch for {reviewer}")
            private_by_kind[kind] = private

        absolute: dict[tuple[int, int], dict[str, Any]] = {}
        pairs: dict[tuple[int, int], dict[str, Any]] = {}
        anchor_exact = 0
        absolute_valid = 0
        pair_valid = 0
        for review_id, mapping in private_by_kind["absolute"].items():
            response = observed_by_kind["absolute"][review_id]
            if response["review_id"] != review_id:
                raise RuntimeError("adapted response review ID changed")
            key = (int(mapping["job_index"]), int(mapping["attempt_index"]))
            if key in absolute:
                raise RuntimeError(f"duplicate absolute item for {reviewer}: {key}")
            if key[0] not in range(PREFIX_COUNT) or key[1] not in ALL_ATTEMPTS:
                raise RuntimeError("absolute mapping outside frozen job/attempt axis")
            # Every row must carry severity; validity is an independent reliability gate.
            absolute[key] = response
            absolute_valid += int(response["valid"])

        for review_id, mapping in private_by_kind["pair"].items():
            response = observed_by_kind["pair"][review_id]
            if response["review_id"] != review_id:
                raise RuntimeError("adapted pair response review ID changed")
            job_index = int(mapping["job_index"])
            fresh_attempt = int(mapping["fresh_attempt"])
            left_attempt = int(mapping["left_attempt"])
            right_attempt = int(mapping["right_attempt"])
            key = (job_index, fresh_attempt)
            if key in pairs:
                raise RuntimeError(f"duplicate pair job/fresh axis for {reviewer}: {key}")
            if job_index not in range(PREFIX_COUNT):
                raise RuntimeError("pair mapping outside frozen job axis")
            normalized_preference = normalize_pair_preference(
                response["pair_choice"],
                fresh_attempt=fresh_attempt,
                left_attempt=left_attempt,
                right_attempt=right_attempt,
            )
            pairs[key] = {
                "preservation": response["preservation"],
                "preference": normalized_preference,
                "valid": response["valid"],
            }
            pair_valid += int(response["valid"])

        for review_id, mapping in private_by_kind["anchor"].items():
            response = observed_by_kind["anchor"][review_id]
            if response["review_id"] != review_id:
                raise RuntimeError("adapted anchor response review ID changed")
            expected_severity = int(mapping["expected_severity"])
            if expected_severity not in SEVERITIES:
                raise RuntimeError("anchor expected severity outside 0/1/2")
            anchor_exact += int(response["valid"] and response["severity"] == expected_severity)

        expected_absolute_axis = {
            (job_index, attempt)
            for job_index in range(PREFIX_COUNT)
            for attempt in ALL_ATTEMPTS
        }
        expected_pair_axis = {
            (job_index, attempt)
            for job_index in range(PREFIX_COUNT)
            for attempt in FRESH_ATTEMPTS
        }
        if set(absolute) != expected_absolute_axis:
            raise RuntimeError(f"absolute job/attempt axis changed for {reviewer}")
        if set(pairs) != expected_pair_axis:
            raise RuntimeError(f"pair job/fresh-attempt axis changed for {reviewer}")
        absolute_rate = absolute_valid / ABSOLUTE_ITEMS_PER_REVIEWER
        pair_rate = pair_valid / PAIR_ITEMS_PER_REVIEWER
        anchor_pass = anchor_exact >= 6
        reliability[reviewer] = {
            "hidden_anchor_exact": anchor_exact,
            "hidden_anchor_total": ANCHORS_PER_REVIEWER,
            "hidden_anchor_pass": anchor_pass,
            "absolute_valid_count": absolute_valid,
            "absolute_total": ABSOLUTE_ITEMS_PER_REVIEWER,
            "absolute_valid_rate": absolute_rate,
            "absolute_valid_pass": absolute_rate >= 0.95,
            "pair_valid_count": pair_valid,
            "pair_total": PAIR_ITEMS_PER_REVIEWER,
            "pair_valid_rate": pair_rate,
            "pair_valid_pass": pair_rate >= 0.95,
        }
        absolute_votes[reviewer] = absolute
        pair_votes[reviewer] = pairs
    return absolute_votes, pair_votes, reliability


def quadratic_weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must be nonempty and aligned")
    matrix = np.zeros((3, 3), dtype=np.float64)
    for a, b in zip(left, right):
        if a not in SEVERITIES or b not in SEVERITIES:
            raise ValueError("kappa values must be 0/1/2")
        matrix[a, b] += 1.0
    n = float(matrix.sum())
    observed = matrix / n
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / (n * n)
    weights = np.fromfunction(lambda i, j: ((i - j) / 2.0) ** 2, (3, 3), dtype=float)
    observed_disagreement = float(np.sum(weights * observed))
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement <= 0:
        return 1.0 if observed_disagreement <= 0 else 0.0
    return float(1.0 - observed_disagreement / expected_disagreement)


def build_consensus(
    absolute_votes: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
    pair_votes: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    dict[str, float],
]:
    reviewers = sorted(absolute_votes)
    if reviewers != sorted(pair_votes) or reviewers != sorted(REVIEWER_NAMES):
        raise RuntimeError("absolute/pair reviewer axes differ")
    absolute_keys = sorted(next(iter(absolute_votes.values())))
    pair_keys = sorted(next(iter(pair_votes.values())))
    if any(set(absolute_votes[name]) != set(absolute_keys) for name in reviewers):
        raise RuntimeError("absolute vote axes differ across reviewers")
    if any(set(pair_votes[name]) != set(pair_keys) for name in reviewers):
        raise RuntimeError("pair vote axes differ across reviewers")

    absolute_consensus: dict[tuple[int, int], dict[str, Any]] = {}
    for key in absolute_keys:
        severities = [int(absolute_votes[reviewer][key]["severity"]) for reviewer in reviewers]
        absolute_consensus[key] = {
            "severity": sorted(severities)[1],
            "blur_fusion": sum(
                absolute_votes[reviewer][key]["blur_fusion"] == "yes"
                for reviewer in reviewers
            )
            >= 2,
            "topology_misalignment": sum(
                absolute_votes[reviewer][key]["topology_misalignment"] == "yes"
                for reviewer in reviewers
            )
            >= 2,
        }

    pair_consensus: dict[tuple[int, int], dict[str, Any]] = {}
    for key in pair_keys:
        preservation_yes = sum(
            pair_votes[reviewer][key]["preservation"] == "yes" for reviewer in reviewers
        )
        preferences = Counter(pair_votes[reviewer][key]["preference"] for reviewer in reviewers)
        if preferences["fresh_better"] >= 2:
            preference = "fresh_better"
        elif preferences["attempt0_better"] >= 2:
            preference = "attempt0_better"
        else:
            preference = "tie"
        pair_consensus[key] = {
            "preservation": preservation_yes >= 2,
            "preference": preference,
        }

    kappas: dict[str, float] = {}
    for left_index in range(len(reviewers)):
        for right_index in range(left_index + 1, len(reviewers)):
            left = reviewers[left_index]
            right = reviewers[right_index]
            kappas[f"{left}__{right}"] = quadratic_weighted_kappa(
                [int(absolute_votes[left][key]["severity"]) for key in absolute_keys],
                [int(absolute_votes[right][key]["severity"]) for key in absolute_keys],
            )
    return absolute_consensus, pair_consensus, kappas


def primary_monte_carlo(
    severities: np.ndarray,
    selected_indices: np.ndarray,
    *,
    replicates: int = PRIMARY_REPLICATES,
    seed: int = PRIMARY_SEED,
) -> dict[str, Any]:
    if severities.ndim != 2 or severities.shape[1] != 4:
        raise ValueError("primary severities must be [prefix,4]")
    if selected_indices.shape != (severities.shape[0],):
        raise ValueError("selected index axis mismatch")
    row_means = np.mean(severities, axis=1)
    observed_d = float(np.mean(row_means - severities[np.arange(len(severities)), selected_indices]))
    rng = np.random.Generator(np.random.PCG64(seed))
    exceed = 0
    completed = 0
    chunk_size = 10_000
    while completed < replicates:
        count = min(chunk_size, replicates - completed)
        choices = rng.integers(0, 4, size=(count, len(severities)), endpoint=False)
        picked = severities[np.arange(len(severities))[None, :], choices]
        permuted_d = np.mean(row_means[None, :] - picked, axis=1)
        exceed += int(np.count_nonzero(permuted_d >= observed_d))
        completed += count
    return {
        "mean_D": observed_d,
        "replicates": replicates,
        "seed": seed,
        "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
        "null": "independent uniform choice among the same four severities within every prefix",
        "tail": "greater_than_or_equal",
        "exceedance_count": exceed,
        "plus_one_p_value": (exceed + 1) / (replicates + 1),
    }


def percentile_bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    lower: float = 0.025,
    upper: float = 0.975,
) -> dict[str, Any]:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires a nonempty finite vector")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(replicates, dtype=np.float64)
    completed = 0
    chunk_size = 5_000
    while completed < replicates:
        count = min(chunk_size, replicates - completed)
        indices = rng.integers(0, len(values), size=(count, len(values)), endpoint=False)
        samples[completed : completed + count] = np.mean(values[indices], axis=1)
        completed += count
    quantiles = np.quantile(samples, [lower, upper], method="linear")
    return {
        "replicates": replicates,
        "seed": seed,
        "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
        "unit": "prefix",
        "interval": [lower, upper],
        "lower": float(quantiles[0]),
        "upper": float(quantiles[1]),
    }


def chi_square_df3_survival(statistic: float) -> float:
    if not math.isfinite(statistic) or statistic < 0:
        raise ValueError("invalid chi-square statistic")
    x = statistic / 2.0
    return float(math.erfc(math.sqrt(x)) + 2.0 * math.sqrt(x) * math.exp(-x) / math.sqrt(math.pi))


def attempt_index_test(selected_attempts: Sequence[int]) -> dict[str, Any]:
    if not selected_attempts or any(value not in FRESH_ATTEMPTS for value in selected_attempts):
        raise ValueError("selected attempts must be 1..4")
    counts = [selected_attempts.count(attempt) for attempt in FRESH_ATTEMPTS]
    expected = len(selected_attempts) / 4.0
    statistic = sum((count - expected) ** 2 / expected for count in counts)
    p_value = chi_square_df3_survival(statistic)
    return {
        "counts_physical_attempt_1_to_4": counts,
        "expected_each": expected,
        "pearson_chi_square": statistic,
        "degrees_of_freedom": 3,
        "p_value": p_value,
        "minimum_p": ATTEMPT_INDEX_MINIMUM_P,
        "passed": p_value >= ATTEMPT_INDEX_MINIMUM_P,
    }


def safety_bootstrap(
    populations: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    replicates: int = SAFETY_BOOTSTRAP_REPLICATES,
    seed: int = SAFETY_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Jointly bootstrap bad excess and preservation loss per population.

    ``bad_excess = selected_bad - exact_uniform_bad`` and
    ``preservation_loss = exact_uniform_preservation - selected_preservation``.
    Positive values are harmful for both outcomes.
    """

    rng = np.random.Generator(np.random.PCG64(seed))
    output: dict[str, Any] = {}
    for name, bad_excess, preservation_loss in populations:
        if (
            bad_excess.ndim != 1
            or preservation_loss.ndim != 1
            or len(bad_excess) != len(preservation_loss)
            or len(bad_excess) == 0
        ):
            output[name] = {
                "prefix_count": int(len(bad_excess)),
                "available": False,
                "passed": False,
            }
            continue
        bad_samples = np.empty(replicates, dtype=np.float64)
        preservation_samples = np.empty(replicates, dtype=np.float64)
        completed = 0
        chunk_size = 5_000
        while completed < replicates:
            count = min(chunk_size, replicates - completed)
            indices = rng.integers(0, len(bad_excess), size=(count, len(bad_excess)), endpoint=False)
            bad_samples[completed : completed + count] = np.mean(bad_excess[indices], axis=1)
            preservation_samples[completed : completed + count] = np.mean(
                preservation_loss[indices], axis=1
            )
            completed += count
        bad_upper = float(np.quantile(bad_samples, 0.95, method="linear"))
        preservation_upper = float(
            np.quantile(preservation_samples, 0.95, method="linear")
        )
        output[name] = {
            "prefix_count": int(len(bad_excess)),
            "available": True,
            "bad_excess_point": float(np.mean(bad_excess)),
            "bad_excess_one_sided_95_upper": bad_upper,
            "preservation_loss_point": float(np.mean(preservation_loss)),
            "preservation_loss_one_sided_95_upper": preservation_upper,
            "noninferiority_margin": SAFETY_MARGIN,
            "bad_bound_passed": bad_upper <= SAFETY_MARGIN,
            "preservation_bound_passed": preservation_upper <= SAFETY_MARGIN,
            "passed": bad_upper <= SAFETY_MARGIN and preservation_upper <= SAFETY_MARGIN,
        }
    return {
        "replicates": replicates,
        "seed": seed,
        "rng": f"numpy.random.Generator(PCG64(seed={seed}))",
        "unit": "prefix",
        "one_sided_upper_quantile": 0.95,
        "populations_in_rng_order": [name for name, _, _ in populations],
        "populations": output,
    }


def decide(
    *,
    mean_d: float,
    primary_p: float,
    primary_lower: float,
    reliability_passed: bool,
    index_passed: bool,
    safety: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    populations = safety.get("populations", {})
    safety_passed = (
        set(populations) == {"class795_all", "attempt0_good_all_classes"}
        and all(bool(row.get("passed")) for row in populations.values())
    )
    material_point_harm = any(
        bool(row.get("available"))
        and (
            float(row.get("bad_excess_point", 0.0)) > SAFETY_MARGIN
            or float(row.get("preservation_loss_point", 0.0)) > SAFETY_MARGIN
        )
        for row in populations.values()
    )
    gates = {
        "mean_D_at_least_0_10": mean_d >= PRIMARY_MINIMUM_EFFECT,
        "one_sided_primary_p_at_most_0_05": primary_p <= PRIMARY_ALPHA,
        "primary_bootstrap_lower_above_zero": primary_lower > 0.0,
        "review_reliability_passed": reliability_passed,
        "attempt_index_gate_passed": index_passed,
        "four_safety_bounds_passed": safety_passed,
        "material_point_harm": material_point_harm,
    }
    # Reliability and randomized-side/index integrity are prerequisites for
    # interpreting direction.  A failed prerequisite cannot become a negative
    # efficacy finding merely because the untrusted point estimate is <= 0.
    if not reliability_passed or not index_passed:
        return "INCONCLUSIVE_NO_PROMOTION", gates
    if mean_d <= 0.0 or material_point_harm:
        return "STOP_MAX_H10", gates
    if all(
        gates[key]
        for key in (
            "mean_D_at_least_0_10",
            "one_sided_primary_p_at_most_0_05",
            "primary_bootstrap_lower_above_zero",
            "review_reliability_passed",
            "attempt_index_gate_passed",
            "four_safety_bounds_passed",
        )
    ):
        return "NARROW_GO", gates
    return "INCONCLUSIVE_NO_PROMOTION", gates


def aggregate(
    *,
    selections: Sequence[Mapping[str, Any]],
    absolute_consensus: Mapping[tuple[int, int], Mapping[str, Any]],
    pair_consensus: Mapping[tuple[int, int], Mapping[str, Any]],
    absolute_votes: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
    pair_votes: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
    reliability: Mapping[str, Any],
    kappas: Mapping[str, float],
) -> dict[str, Any]:
    selection_by_job = {int(row["job_index"]): row for row in selections}
    primary_jobs = sorted(
        job for job, row in selection_by_job.items() if int(row["class_id"]) == PRIMARY_CLASS
    )
    if len(primary_jobs) != PRIMARY_PREFIX_COUNT:
        raise RuntimeError("primary class795 prefix count changed")

    primary_severity = np.asarray(
        [
            [
                float(absolute_consensus[(job, attempt)]["severity"]) / 2.0
                for attempt in FRESH_ATTEMPTS
            ]
            for job in primary_jobs
        ],
        dtype=np.float64,
    )
    primary_selected = np.asarray(
        [int(selection_by_job[job]["max_attempt"]) - 1 for job in primary_jobs],
        dtype=np.int64,
    )
    primary_random = primary_monte_carlo(primary_severity, primary_selected)
    primary_d = np.mean(primary_severity, axis=1) - primary_severity[
        np.arange(len(primary_jobs)), primary_selected
    ]
    primary_ci = percentile_bootstrap_mean(
        primary_d,
        replicates=PRIMARY_BOOTSTRAP_REPLICATES,
        seed=PRIMARY_BOOTSTRAP_SEED,
    )
    index_test = attempt_index_test(
        [int(selection_by_job[job]["max_attempt"]) for job in primary_jobs]
    )

    reviewer_gate = all(
        bool(row["hidden_anchor_pass"])
        and bool(row["absolute_valid_pass"])
        and bool(row["pair_valid_pass"])
        for row in reliability.values()
    )
    all_primary_and_safety_votes_valid = all(
        int(row["absolute_valid_count"]) == ABSOLUTE_ITEMS_PER_REVIEWER
        and int(row["pair_valid_count"]) == PAIR_ITEMS_PER_REVIEWER
        for row in reliability.values()
    )
    kappa_min = min(kappas.values())
    # The frozen 95% rates remain useful reviewer-quality diagnostics.  A
    # technically invalid item, however, must not contribute a filled-in vote
    # to the primary or safety decision, so any such row forces INCONCLUSIVE.
    reliability_passed = (
        reviewer_gate
        and all_primary_and_safety_votes_valid
        and kappa_min >= KAPPA_MINIMUM
    )

    def safety_vectors(jobs: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        bad_values: list[float] = []
        preservation_values: list[float] = []
        for job in jobs:
            selected_attempt = int(selection_by_job[job]["max_attempt"])
            fresh_bad = np.asarray(
                [
                    int(absolute_consensus[(job, attempt)]["severity"] >= 1)
                    for attempt in FRESH_ATTEMPTS
                ],
                dtype=np.float64,
            )
            fresh_preservation = np.asarray(
                [
                    int(bool(pair_consensus[(job, attempt)]["preservation"]))
                    for attempt in FRESH_ATTEMPTS
                ],
                dtype=np.float64,
            )
            selected_index = selected_attempt - 1
            bad_values.append(float(fresh_bad[selected_index] - np.mean(fresh_bad)))
            preservation_values.append(
                float(np.mean(fresh_preservation) - fresh_preservation[selected_index])
            )
        return np.asarray(bad_values), np.asarray(preservation_values)

    sentinel_good_jobs = sorted(
        job
        for job in selection_by_job
        if int(absolute_consensus[(job, 0)]["severity"]) == 0
    )
    class795_safety = safety_vectors(primary_jobs)
    sentinel_good_safety = safety_vectors(sentinel_good_jobs)
    safety = safety_bootstrap(
        [
            ("class795_all", *class795_safety),
            ("attempt0_good_all_classes", *sentinel_good_safety),
        ]
    )

    def population_rates(jobs: Sequence[int]) -> dict[str, Any]:
        selected_bad = []
        uniform_bad = []
        selected_catastrophic = []
        uniform_catastrophic = []
        selected_preservation = []
        uniform_preservation = []
        for job in jobs:
            selected = int(selection_by_job[job]["max_attempt"])
            severities = [
                int(absolute_consensus[(job, attempt)]["severity"])
                for attempt in FRESH_ATTEMPTS
            ]
            preservation = [
                int(bool(pair_consensus[(job, attempt)]["preservation"]))
                for attempt in FRESH_ATTEMPTS
            ]
            selected_bad.append(int(severities[selected - 1] >= 1))
            uniform_bad.append(sum(value >= 1 for value in severities) / 4.0)
            selected_catastrophic.append(int(severities[selected - 1] == 2))
            uniform_catastrophic.append(sum(value == 2 for value in severities) / 4.0)
            selected_preservation.append(preservation[selected - 1])
            uniform_preservation.append(sum(preservation) / 4.0)
        return {
            "prefix_count": len(jobs),
            "selected_obvious_bad_rate": float(np.mean(selected_bad)) if jobs else None,
            "exact_uniform_obvious_bad_rate": float(np.mean(uniform_bad)) if jobs else None,
            "selected_catastrophic_rate": float(np.mean(selected_catastrophic)) if jobs else None,
            "exact_uniform_catastrophic_rate": float(np.mean(uniform_catastrophic)) if jobs else None,
            "selected_catastrophic_excess": (
                float(np.mean(selected_catastrophic) - np.mean(uniform_catastrophic)) if jobs else None
            ),
            "selected_preservation_rate": float(np.mean(selected_preservation)) if jobs else None,
            "exact_uniform_preservation_rate": float(np.mean(uniform_preservation)) if jobs else None,
        }

    def secondary_repair_rates(jobs: Sequence[int]) -> dict[str, Any]:
        """Descriptive external repair readout; never a selector or decision gate."""

        opportunity_jobs = [
            job
            for job in jobs
            if int(absolute_consensus[(job, 0)]["severity"]) >= 1
        ]
        selected_repair: list[float] = []
        uniform_repair: list[float] = []
        selected_preference = Counter()
        uniform_preference = Counter()
        for job in opportunity_jobs:
            attempt0_severity = int(absolute_consensus[(job, 0)]["severity"])
            selected = int(selection_by_job[job]["max_attempt"])
            repairs: list[float] = []
            for attempt in FRESH_ATTEMPTS:
                pair = pair_consensus[(job, attempt)]
                preference = str(pair["preference"])
                uniform_preference[preference] += 0.25
                repairs.append(
                    float(
                        int(absolute_consensus[(job, attempt)]["severity"])
                        < attempt0_severity
                        and int(absolute_consensus[(job, attempt)]["severity"]) < 2
                        and bool(pair["preservation"])
                        and preference == "fresh_better"
                    )
                )
            selected_pair = pair_consensus[(job, selected)]
            selected_preference[str(selected_pair["preference"])] += 1
            selected_repair.append(repairs[selected - 1])
            uniform_repair.append(float(np.mean(repairs)))
        count = len(opportunity_jobs)
        selected_rate = float(np.mean(selected_repair)) if count else None
        uniform_rate = float(np.mean(uniform_repair)) if count else None
        return {
            "status": "secondary_external_only_not_selector_or_primary",
            "opportunity_definition": "attempt0 consensus severity >= 1",
            "strict_repair_definition": (
                "fresh severity < attempt0 severity AND pair preservation consensus yes "
                "AND normalized pair preference fresh_better AND fresh severity < 2"
            ),
            "opportunity_prefix_count": count,
            "selected_strict_repair_rate": selected_rate,
            "exact_uniform_strict_repair_rate": uniform_rate,
            "selected_minus_exact_uniform_strict_repair_rate": (
                selected_rate - uniform_rate if count else None
            ),
            "selected_normalized_preference_rates": {
                value: selected_preference[value] / count if count else None
                for value in sorted(NORMALIZED_PAIR_VALUES)
            },
            "exact_uniform_normalized_preference_rates": {
                value: uniform_preference[value] / count if count else None
                for value in sorted(NORMALIZED_PAIR_VALUES)
            },
        }

    def secondary_subtype_rates(jobs: Sequence[int]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for subtype in ("blur_fusion", "topology_misalignment"):
            selected_values: list[float] = []
            uniform_values: list[float] = []
            for job in jobs:
                selected = int(selection_by_job[job]["max_attempt"])
                flags = [
                    float(bool(absolute_consensus[(job, attempt)][subtype]))
                    for attempt in FRESH_ATTEMPTS
                ]
                selected_values.append(flags[selected - 1])
                uniform_values.append(float(np.mean(flags)))
            output[subtype] = {
                "selected_rate": float(np.mean(selected_values)) if jobs else None,
                "exact_uniform_rate": float(np.mean(uniform_values)) if jobs else None,
                "selected_minus_exact_uniform": (
                    float(np.mean(selected_values) - np.mean(uniform_values)) if jobs else None
                ),
            }
        return {
            "status": "secondary_diagnostic_only_not_selector_primary_or_decision",
            "consensus_definition": "at least two of three absolute reviewers answered yes",
            "rates": output,
        }

    realized_wins = Counter()
    medoid_d: list[float] = []
    class_effects: dict[str, Any] = {}
    for class_id in (207, 602, 795):
        jobs = sorted(
            job for job, row in selection_by_job.items() if int(row["class_id"]) == class_id
        )
        values = []
        for job in jobs:
            severity = np.asarray(
                [
                    float(absolute_consensus[(job, attempt)]["severity"]) / 2.0
                    for attempt in FRESH_ATTEMPTS
                ]
            )
            selected = int(selection_by_job[job]["max_attempt"]) - 1
            values.append(float(np.mean(severity) - severity[selected]))
        class_effects[str(class_id)] = {
            "prefix_count": len(jobs),
            "mean_D_descriptive": float(np.mean(values)),
        }
    for job in primary_jobs:
        row = selection_by_job[job]
        selected_loss = (
            float(absolute_consensus[(job, int(row["max_attempt"]))]["severity"]) / 2.0
        )
        random_loss = (
            float(absolute_consensus[(job, int(row["hash_random_attempt"]))]["severity"])
            / 2.0
        )
        medoid_loss = (
            float(absolute_consensus[(job, int(row["medoid_attempt"]))]["severity"])
            / 2.0
        )
        if selected_loss < random_loss:
            realized_wins["max_better"] += 1
        elif selected_loss > random_loss:
            realized_wins["random_better"] += 1
        else:
            realized_wins["tie"] += 1
        fresh_mean = np.mean(
            [
                float(absolute_consensus[(job, attempt)]["severity"]) / 2.0
                for attempt in FRESH_ATTEMPTS
            ]
        )
        medoid_d.append(float(fresh_mean - medoid_loss))

    decision, gates = decide(
        mean_d=float(primary_random["mean_D"]),
        primary_p=float(primary_random["plus_one_p_value"]),
        primary_lower=float(primary_ci["lower"]),
        reliability_passed=reliability_passed,
        index_passed=bool(index_test["passed"]),
        safety=safety,
    )
    severity_histogram = Counter(
        int(row["severity"]) for row in absolute_consensus.values()
    )

    def reviewer_class795_summary(reviewer: str) -> dict[str, Any]:
        selected_severity: list[float] = []
        uniform_severity: list[float] = []
        selected_bad: list[float] = []
        uniform_bad: list[float] = []
        selected_catastrophic: list[float] = []
        uniform_catastrophic: list[float] = []
        selected_preservation: list[float] = []
        uniform_preservation: list[float] = []
        d_values: list[float] = []
        for job in primary_jobs:
            selected = int(selection_by_job[job]["max_attempt"])
            severities = [
                int(absolute_votes[reviewer][(job, attempt)]["severity"])
                for attempt in FRESH_ATTEMPTS
            ]
            preservation = [
                int(pair_votes[reviewer][(job, attempt)]["preservation"] == "yes")
                for attempt in FRESH_ATTEMPTS
            ]
            selected_value = severities[selected - 1]
            selected_severity.append(float(selected_value))
            uniform_severity.append(float(np.mean(severities)))
            d_values.append(float(np.mean(severities) / 2.0 - selected_value / 2.0))
            selected_bad.append(float(selected_value >= 1))
            uniform_bad.append(float(np.mean([value >= 1 for value in severities])))
            selected_catastrophic.append(float(selected_value == 2))
            uniform_catastrophic.append(float(np.mean([value == 2 for value in severities])))
            selected_preservation.append(float(preservation[selected - 1]))
            uniform_preservation.append(float(np.mean(preservation)))
        return {
            "status": "descriptive_only_not_a_decision_gate",
            "prefix_count": len(primary_jobs),
            "mean_D_severity_over_2": float(np.mean(d_values)),
            "selected_mean_severity_0_to_2": float(np.mean(selected_severity)),
            "exact_uniform_mean_severity_0_to_2": float(np.mean(uniform_severity)),
            "selected_obvious_bad_rate": float(np.mean(selected_bad)),
            "exact_uniform_obvious_bad_rate": float(np.mean(uniform_bad)),
            "selected_catastrophic_rate": float(np.mean(selected_catastrophic)),
            "exact_uniform_catastrophic_rate": float(np.mean(uniform_catastrophic)),
            "selected_preservation_yes_rate": float(np.mean(selected_preservation)),
            "exact_uniform_preservation_yes_rate": float(np.mean(uniform_preservation)),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "scientific_role": "external_blind_evaluation_of_presealed_sampler_internal_selector",
        "method_evaluation_boundary": {
            "external_judgments_are_selector_inputs": False,
            "prefix_branch_horizon_threshold_or_direction_changed_after_reviews": False,
            "private_mapping_opened_only_after_complete_public_axis_reviews": True,
            "output_contains_per_prefix_or_review_id_join": False,
            "primary_selector": "sealed step149 h10 max fresh mean nonconformity",
        },
        "counts": {
            "prefixes": PREFIX_COUNT,
            "class795_primary_prefixes": PRIMARY_PREFIX_COUNT,
            "absolute_items_per_reviewer": ABSOLUTE_ITEMS_PER_REVIEWER,
            "pair_items_per_reviewer": PAIR_ITEMS_PER_REVIEWER,
            "hidden_anchors_per_reviewer": ANCHORS_PER_REVIEWER,
            "consensus_absolute_items": len(absolute_consensus),
            "consensus_pair_items": len(pair_consensus),
            "attempt0_good_all_classes_prefixes": len(sentinel_good_jobs),
        },
        "review_reliability": {
            "reviewers": dict(reliability),
            "pairwise_quadratic_weighted_kappa": dict(kappas),
            "minimum_pairwise_kappa": kappa_min,
            "minimum_required_kappa": KAPPA_MINIMUM,
            "all_primary_and_safety_votes_valid": all_primary_and_safety_votes_valid,
            "invalid_vote_decision_rule": "any absolute or pair valid=no forces INCONCLUSIVE",
            "passed": reliability_passed,
        },
        "class795_each_reviewer_descriptive": {
            reviewer: reviewer_class795_summary(reviewer) for reviewer in REVIEWER_NAMES
        },
        "consensus_severity_histogram_0_1_2_all_absolute": {
            str(level): severity_histogram[level] for level in (0, 1, 2)
        },
        "primary_class795": {
            "estimand": "D_i=mean_four_fresh(severity/2)-selected_max_A(severity/2)",
            "positive_is_better": True,
            "monte_carlo": primary_random,
            "prefix_percentile_bootstrap": primary_ci,
            "minimum_effect": PRIMARY_MINIMUM_EFFECT,
            "realized_hash_random_win_tie_loss": {
                key: realized_wins[key] for key in ("max_better", "tie", "random_better")
            },
            "medoid_mean_D_descriptive_negative_control": float(np.mean(medoid_d)),
        },
        "attempt_index_gate": index_test,
        "safety": {
            **safety,
            "class795_rates": population_rates(primary_jobs),
            "attempt0_good_all_classes_rates": population_rates(sentinel_good_jobs),
            "obvious_bad_definition": "consensus severity >= 1",
            "catastrophic_definition": "consensus severity == 2",
            "preservation_definition": "at least two of three reviewers answered yes",
        },
        "secondary_pair_repair_descriptive": {
            "class795": secondary_repair_rates(primary_jobs),
            "all_classes": secondary_repair_rates(sorted(selection_by_job)),
            "preferred_side_normalization": (
                "left/right is decoded with the sealed private pair mapping into "
                "fresh_better/attempt0_better/tie"
            ),
            "used_by_selector_primary_or_decision": False,
        },
        "secondary_absolute_subtype_diagnostic": {
            "class795": secondary_subtype_rates(primary_jobs),
            "all_classes": secondary_subtype_rates(sorted(selection_by_job)),
            "used_by_selector_primary_or_decision": False,
        },
        "descriptive_class_effects_no_transfer_claim": class_effects,
        "decision": decision,
        "decision_gates": gates,
        "claim_boundary": [
            "NARROW_GO is limited to the frozen class795/new-suffix/sampler design.",
            "Classes207/602 are gross-harm sentinels and are not powered benefit confirmations.",
            "Pair responses, FID, Inception, DINO, CLIP, and any external embedding do not select or rescue the method.",
            "Pair preference and absolute defect subtypes are descriptive secondary readouts with no multiplicity-adjusted confirmatory claim.",
            "A failed primary cannot be rescued on this pool by another horizon, class slice, low-O direction, or external score.",
        ],
    }
    result["identity_sha256"] = canonical_sha256(result)
    return result


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_review_arguments(values: Sequence[str]) -> dict[str, dict[str, Path]]:
    output: dict[str, dict[str, Path]] = {}
    kind_aliases = {
        "absolute": "absolute",
        "pair": "pair",
        "preservation": "pair",
        "anchor": "anchor",
        "qualification": "anchor",
    }
    for value in values:
        if "=" not in value:
            raise ValueError("--review must be REVIEWER=DIR or REVIEWER:KIND=CSV_PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path)
        if ":" not in label:
            reviewer = label
            if reviewer not in REVIEWER_NAMES or reviewer in output:
                raise ValueError("directory-form reviewer names must be reviewer_1..reviewer_3 once")
            output[reviewer] = {
                "absolute": path / f"{reviewer}_absolute.csv",
                "pair": path / f"{reviewer}_preservation.csv",
                "anchor": path / f"{reviewer}_qualification.csv",
            }
            continue
        reviewer, raw_kind = (token.strip() for token in label.rsplit(":", 1))
        kind = kind_aliases.get(raw_kind.lower())
        if reviewer not in REVIEWER_NAMES or kind is None:
            raise ValueError("review file keys must be reviewer_1..3:absolute|pair|anchor")
        reviewer_paths = output.setdefault(reviewer, {})
        if kind in reviewer_paths:
            raise ValueError(f"duplicate review path for {reviewer}:{kind}")
        reviewer_paths[kind] = path
    if set(output) != set(REVIEWER_NAMES) or any(
        set(output[reviewer]) != ITEM_KINDS for reviewer in REVIEWER_NAMES
    ):
        raise ValueError(
            "supply three REVIEWER=DIR arguments or all nine REVIEWER:KIND=CSV_PATH arguments"
        )
    return output


def analyze(args: argparse.Namespace) -> None:
    output = args.output.expanduser().absolute()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis: {output}")
    lock_manifest, protocol, _ = validate_lock(args.lock)
    internal_manifest, selections = validate_internal_product(
        args.internal_product,
        lock_identity=lock_manifest["identity_sha256"],
        protocol_identity=protocol["identity_sha256"],
    )
    locked_axis = {
        (
            int(row["job_index"]),
            int(row["global_seed"]),
            int(row["class_id"]),
            int(row["rollback_sampling_step"]),
        )
        for row in protocol["jobs"]
    }
    selector_axis = {
        (
            int(row["job_index"]),
            int(row["global_seed"]),
            int(row["class_id"]),
            int(row["rollback_step"]),
        )
        for row in selections
    }
    if selector_axis != locked_axis:
        raise RuntimeError("sealed selector job axis differs from the V1.2 lock")
    public = load_seal_bundle(args.public_seal)
    public_rows = adapt_public_items(public)
    review_paths = parse_review_arguments(args.review)
    observed_reviews, review_receipts = prevalidate_reviews_against_public(
        public_rows,
        review_paths,
    )
    # The private lineage, gold anchors, and randomized-side roles are opened
    # only after all nine response tables pass the opaque public-axis checks.
    private = load_seal_bundle(args.private_seal)
    validate_seal_bindings(
        public,
        private,
        lock_identity=lock_manifest["identity_sha256"],
        internal_identity=internal_manifest["identity_sha256"],
    )
    private_rows = adapt_private_items(private)
    absolute_votes, pair_votes, reliability = validate_and_join_reviews(
        public_rows,
        private_rows,
        observed_reviews,
    )
    absolute_consensus, pair_consensus, kappas = build_consensus(
        absolute_votes,
        pair_votes,
    )
    result = aggregate(
        selections=selections,
        absolute_consensus=absolute_consensus,
        pair_consensus=pair_consensus,
        absolute_votes=absolute_votes,
        pair_votes=pair_votes,
        reliability=reliability,
        kappas=kappas,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "results.json", result)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "lock_identity_sha256": lock_manifest["identity_sha256"],
            "protocol_identity_sha256": protocol["identity_sha256"],
            "internal_product_identity_sha256": internal_manifest["identity_sha256"],
            "public_seal_identities_sha256": sorted(public.identities),
            "private_seal_identities_sha256": sorted(private.identities),
            "review_csv_files": sorted(
                review_receipts,
                key=lambda row: (row["reviewer"], row["item_kind"]),
            ),
            "review_rows_or_private_mapping_emitted": False,
            "external_judgment_fed_back_to_selector": False,
            "private_opened_only_after_complete_public_axis_reviews": True,
        }
        receipt["identity_sha256"] = canonical_sha256(receipt)
        write_json(staging / "input_receipt.json", receipt)
        shutil.copy2(Path(__file__).resolve(), staging / "analyzer_source.py")
        payloads = sorted(path for path in staging.iterdir() if path.is_file())
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": RESULT_KIND,
            "status": "complete",
            "scientific_role": "aggregate_only_external_blind_evaluation",
            "result_identity_sha256": result["identity_sha256"],
            "lock_identity_sha256": lock_manifest["identity_sha256"],
            "internal_product_identity_sha256": internal_manifest["identity_sha256"],
            "contains_per_prefix_join": False,
            "files": [file_record(path, staging) for path in payloads],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion: dict[str, Any] = {
            "schema_version": 1,
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "result_identity_sha256": result["identity_sha256"],
            "decision": result["decision"],
        }
        completion["identity_sha256"] = canonical_sha256(completion)
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "result_identity_sha256": result["identity_sha256"],
                "decision": result["decision"],
                "contains_per_prefix_join": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="dit-v22-blind-analyzer-selftest-") as raw_root:
        test_root = Path(raw_root)
        public_root = test_root / "public"
        templates_root = public_root / "templates"
        private_root = test_root / "private"
        reviews_root = test_root / "reviews"
        templates_root.mkdir(parents=True)
        private_root.mkdir()
        reviews_root.mkdir()

        absolute_mapping = [
            {
                "image_id": f"img-{job:03d}-{attempt}",
                "job_index": job,
                "attempt": attempt,
            }
            for job in range(PREFIX_COUNT)
            for attempt in ALL_ATTEMPTS
        ]
        preservation_mapping = []
        for job in range(PREFIX_COUNT):
            for attempt in FRESH_ATTEMPTS:
                fresh_on_left = (job + attempt) % 2 == 0
                preservation_mapping.append(
                    {
                        "pair_id": f"pair-{job:03d}-{attempt}",
                        "job_index": job,
                        "fresh_attempt": attempt,
                        "left_attempt": attempt if fresh_on_left else 0,
                        "right_attempt": 0 if fresh_on_left else attempt,
                        "roles": {"attempt0": 0, "fresh": attempt},
                    }
                )
        qualification_mapping = [
            {"qualification_id": f"anchor-{index}", "gold_severity": index % 3}
            for index in range(ANCHORS_PER_REVIEWER)
        ]

        template_specs = (
            (
                "absolute",
                "image_id",
                [row["image_id"] for row in absolute_mapping],
                [
                    "image_id",
                    "severity",
                    "blur_fusion",
                    "topology_misalignment",
                    "valid",
                    "localized_reason",
                ],
            ),
            (
                "preservation",
                "pair_id",
                [row["pair_id"] for row in preservation_mapping],
                ["pair_id", "preserved", "preferred_side", "valid", "reason"],
            ),
            (
                "qualification",
                "qualification_id",
                [row["qualification_id"] for row in qualification_mapping],
                ["qualification_id", "severity", "valid", "reason"],
            ),
        )
        for reviewer in REVIEWER_NAMES:
            for filename_kind, id_field, ids, columns in template_specs:
                path = templates_root / f"{reviewer}_{filename_kind}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    for review_id in ids:
                        writer.writerow({id_field: review_id})
        public_manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_V22_TRANSIENT_ESCAPE_BLIND_DELIVERY_V1",
            "status": "complete",
            "files": [
                file_record(path, public_root)
                for path in sorted(templates_root.iterdir())
            ],
        }
        public_manifest["identity_sha256"] = canonical_sha256(public_manifest)
        write_json(public_root / "manifest.json", public_manifest)

        private_payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "DIT_V22_TRANSIENT_ESCAPE_PRIVATE_MAPPING_V1",
            "status": "SEALED_UNTIL_THREE_REVIEWS_AND_QUALIFICATION_LOCKS",
            "public_delivery_identity_sha256": public_manifest["identity_sha256"],
            "prospective_lock_identity_sha256": "synthetic-lock",
            "internal_product_identity_sha256": "synthetic-internal",
            "absolute_mapping": absolute_mapping,
            "preservation_mapping": preservation_mapping,
            "qualification_mapping": qualification_mapping,
        }
        private_payload["identity_sha256"] = canonical_sha256(private_payload)
        write_json(private_root / "sealed_mapping.json", private_payload)

        severity_by_attempt = {0: 2, 1: 0, 2: 1, 3: 2, 4: 0}
        for reviewer_index, reviewer in enumerate(REVIEWER_NAMES):
            absolute_path = reviews_root / f"{reviewer}_absolute.csv"
            with absolute_path.open("w", newline="", encoding="utf-8") as handle:
                columns = list(template_specs[0][3])
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for mapping in absolute_mapping:
                    writer.writerow(
                        {
                            "image_id": mapping["image_id"],
                            "severity": severity_by_attempt[int(mapping["attempt"])],
                            "blur_fusion": "yes" if reviewer_index < 2 else "uncertain",
                            "topology_misalignment": "no",
                            "valid": "yes",
                            "localized_reason": "synthetic",
                        }
                    )
            pair_path = reviews_root / f"{reviewer}_preservation.csv"
            with pair_path.open("w", newline="", encoding="utf-8") as handle:
                columns = list(template_specs[1][3])
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for mapping in preservation_mapping:
                    fresh_side = (
                        "left"
                        if int(mapping["left_attempt"]) == int(mapping["fresh_attempt"])
                        else "right"
                    )
                    writer.writerow(
                        {
                            "pair_id": mapping["pair_id"],
                            "preserved": "yes" if reviewer_index < 2 else "uncertain",
                            "preferred_side": fresh_side,
                            "valid": "yes",
                            "reason": "synthetic",
                        }
                    )
            anchor_path = reviews_root / f"{reviewer}_qualification.csv"
            with anchor_path.open("w", newline="", encoding="utf-8") as handle:
                columns = list(template_specs[2][3])
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for mapping in qualification_mapping:
                    writer.writerow(
                        {
                            "qualification_id": mapping["qualification_id"],
                            "severity": mapping["gold_severity"],
                            "valid": "yes",
                            "reason": "synthetic",
                        }
                    )

        public = load_seal_bundle(public_root)
        public_rows = adapt_public_items(public)
        review_paths = parse_review_arguments(
            [f"{reviewer}={reviews_root}" for reviewer in REVIEWER_NAMES]
        )
        observed_reviews, receipts = prevalidate_reviews_against_public(
            public_rows,
            review_paths,
        )
        if len(receipts) != REVIEWER_COUNT * 3:
            raise AssertionError("synthetic nine-table public prevalidation failed")
        private = load_seal_bundle(private_root)
        validate_seal_bindings(
            public,
            private,
            lock_identity="synthetic-lock",
            internal_identity="synthetic-internal",
        )
        private_rows = adapt_private_items(private)
        absolute_votes, pair_votes, reliability = validate_and_join_reviews(
            public_rows,
            private_rows,
            observed_reviews,
        )
        absolute_consensus, pair_consensus, synthetic_kappas = build_consensus(
            absolute_votes,
            pair_votes,
        )
        if (
            len(absolute_consensus) != ABSOLUTE_ITEMS_PER_REVIEWER
            or len(pair_consensus) != PAIR_ITEMS_PER_REVIEWER
            or not all(row["hidden_anchor_pass"] for row in reliability.values())
            or min(synthetic_kappas.values()) != 1.0
        ):
            raise AssertionError("synthetic builder-schema axes or reliability failed")
        if not all(
            row["preservation"]
            and row["preference"] == "fresh_better"
            for row in pair_consensus.values()
        ):
            raise AssertionError("randomized pair-side normalization or consensus failed")
        if not all(
            row["blur_fusion"] and not row["topology_misalignment"]
            for row in absolute_consensus.values()
        ):
            raise AssertionError("absolute subtype consensus failed")

    if abs(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) - 1.0) > 1e-12:
        raise AssertionError("weighted-kappa identity test failed")
    if abs(chi_square_df3_survival(0.0) - 1.0) > 1e-12:
        raise AssertionError("chi-square survival test failed")
    attempts = attempt_index_test([1, 2, 3, 4] * 4)
    if not attempts["passed"] or attempts["p_value"] != 1.0:
        raise AssertionError("attempt-index test failed")
    severity = np.asarray(
        [[0.0, 0.5, 1.0, 0.5], [0.0, 0.0, 0.5, 1.0]], dtype=np.float64
    )
    primary = primary_monte_carlo(severity, np.asarray([0, 0]), replicates=1_000, seed=7)
    if primary["mean_D"] <= 0 or not (0 < primary["plus_one_p_value"] <= 1):
        raise AssertionError("primary Monte Carlo self-test failed")
    bootstrap = percentile_bootstrap_mean(
        np.asarray([0.0, 0.25, 0.5]), replicates=1_000, seed=8
    )
    if bootstrap["lower"] > bootstrap["upper"]:
        raise AssertionError("bootstrap interval self-test failed")
    safety = safety_bootstrap(
        [
            ("class795_all", np.zeros(8), np.zeros(8)),
            ("attempt0_good_all_classes", np.zeros(8), np.zeros(8)),
        ],
        replicates=1_000,
        seed=9,
    )
    decision, _ = decide(
        mean_d=0.2,
        primary_p=0.01,
        primary_lower=0.05,
        reliability_passed=True,
        index_passed=True,
        safety=safety,
    )
    if decision != "NARROW_GO":
        raise AssertionError("decision self-test failed")
    inconclusive, _ = decide(
        mean_d=-0.2,
        primary_p=1.0,
        primary_lower=-0.3,
        reliability_passed=False,
        index_passed=True,
        safety=safety,
    )
    if inconclusive != "INCONCLUSIVE_NO_PROMOTION":
        raise AssertionError("reliability-first decision precedence failed")
    print(
        "self-test passed: synthetic builder schema, nine-table blind order, adapters, "
        "side/subtype consensus, PCG64 inference, safety signs, and decision gates"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--internal-product", type=Path, default=DEFAULT_INTERNAL_PRODUCT)
    parser.add_argument("--public-seal", type=Path, default=DEFAULT_PUBLIC_SEAL)
    parser.add_argument("--private-seal", type=Path, default=DEFAULT_PRIVATE_SEAL)
    parser.add_argument(
        "--review",
        action="append",
        default=[],
        metavar="REVIEWER=DIR|REVIEWER:KIND=CSV_PATH",
        help=(
            "three reviewer directories containing the frozen filenames, or all nine "
            "kind-specific CSV paths"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    else:
        analyze(parsed)
