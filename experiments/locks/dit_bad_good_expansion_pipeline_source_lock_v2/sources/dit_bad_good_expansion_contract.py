#!/usr/bin/env python3
"""Shared frozen constants and integrity helpers for the DiT expansion cohort.

This module contains no data access at import time.  In particular, importing
it never opens the original/expansion score products, calibration thresholds,
alerts, endpoint images, or row-level visual labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_LOCK = ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
CANDIDATE_PROTOCOL_IDENTITY = (
    "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"
)
EXPANSION_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_event_count_expansion_lock_v1"
)
EXPANSION_PROTOCOL_IDENTITY = (
    "78c37ba24922075357ccf459837b2c5695343779c100091036d7f9f4eda9a652"
)
PIPELINE_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_expansion_pipeline_source_lock_v2"
)

CLASSES = (207, 602, 795)
CALIBRATION_SEEDS = tuple(range(30, 50))
ORIGINAL_EVALUATION_SEEDS = tuple(range(50, 130))
EXPANSION_SEEDS = tuple(range(130, 250))
ALL_CONFIRMATION_SEEDS = (*ORIGINAL_EVALUATION_SEEDS, *EXPANSION_SEEDS)

ORIGINAL_CLEAR_BAD_EVENTS = 8
MINIMUM_CLEAR_BAD_EVENTS = 15
REVIEWERS = ("J", "K", "L")

PRIMARY_SCORE = "S_UNION"
PRIMARY_ALERT_010 = "alert_alpha0p10_conformal"
PRIMARY_ALERT_005 = "alert_alpha0p05_conformal"
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026082701
CLUSTER_BOOTSTRAP_DRAWS = 100_000
CLUSTER_BOOTSTRAP_SEED = 2026082702
AUC_BOOTSTRAP_DRAWS = 100_000
AUC_BOOTSTRAP_SEED = 2026082703

LABEL_BAD = "clear_bad"
LABEL_GOOD = "clean_good"
LABEL_EXCLUDED = "mild_or_disputed"
LABELS = (LABEL_BAD, LABEL_GOOD, LABEL_EXCLUDED)

ALLOWED_FLAGS = {
    "none",
    "global_blur",
    "local_blur",
    "fusion_duplication",
    "topology_attachment",
    "limb_object_misalignment",
    "texture_break",
    "other",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sample_key(class_id: int, seed: int) -> str:
    return f"class{class_id:04d}_seed{seed:03d}"


def expected_keys(seeds: tuple[int, ...] = EXPANSION_SEEDS) -> set[str]:
    return {sample_key(class_id, seed) for class_id in CLASSES for seed in seeds}


def require_canonical_identity(value: dict[str, Any], description: str) -> str:
    payload = dict(value)
    identity = payload.pop("identity_sha256", None)
    if not isinstance(identity, str) or identity != canonical_sha256(payload):
        raise RuntimeError(f"{description} canonical identity failed")
    return identity


def validate_candidate_lock(root: Path = CANDIDATE_LOCK) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"candidate lock must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    protocol_path = root / "candidate_protocol.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    protocol = load_json(protocol_path)
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("protocol_identity_sha256") != CANDIDATE_PROTOCOL_IDENTITY
        or protocol.get("identity_sha256") != CANDIDATE_PROTOCOL_IDENTITY
        or protocol.get("schema_version") != 5
        or protocol.get("status")
        != "FROZEN_BEFORE_ANY_FRESH_SCORE_EXTRACTION_OR_ENDPOINT_VISUAL_REVIEW"
    ):
        raise RuntimeError("candidate v5 lock changed")
    for item in manifest.get("files", []):
        path = root / str(item.get("name"))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"candidate lock member changed: {path}")
    fresh = protocol.get("fresh_confirmation", {})
    if tuple(fresh.get("classes", ())) != CLASSES:
        raise RuntimeError("candidate class contract changed")
    gate = protocol.get("evaluation", {}).get("initial_go_gate", {})
    if (
        gate.get("minimum_clear_bad_events_for_decision")
        != MINIMUM_CLEAR_BAD_EVENTS
        or gate.get("S_UNION_class_matched_auc_at_least") != 0.75
        or gate.get("S_UNION_stratified_permutation_one_sided_p_below") != 0.05
        or gate.get("alpha_0p10_TPR_minus_FPR_point_above") != 0.0
        or gate.get("no_class_with_two_or_more_bad_events_has_auc_below") != 0.6
    ):
        raise RuntimeError("candidate v5 evaluation gate changed")
    return protocol


def validate_expansion_lock(root: Path = EXPANSION_LOCK) -> dict[str, Any]:
    """Validate the event-count-only expansion decision without score access."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"expansion lock must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    protocol_path = root / "expansion_protocol.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    protocol = load_json(protocol_path)
    require_canonical_identity(manifest, "expansion manifest")
    require_canonical_identity(protocol, "expansion protocol")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("protocol_identity_sha256") != EXPANSION_PROTOCOL_IDENTITY
        or manifest.get("protocol_identity_sha256") != EXPANSION_PROTOCOL_IDENTITY
        or protocol.get("identity_sha256") != EXPANSION_PROTOCOL_IDENTITY
        or protocol.get("status")
        != "FROZEN_AFTER_EVENT_COUNT_ONLY_GATE_BEFORE_EXPANSION_SAMPLING_OR_SCORE_ACCESS"
    ):
        raise RuntimeError("event-count expansion lock changed")
    files = {str(item.get("name")): item for item in manifest.get("files", [])}
    for name, item in files.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"expansion lock member changed: {path}")
    cohort = protocol.get("expansion_cohort", {})
    selection = protocol.get("selection_basis", {})
    audit = protocol.get("evidence_access_audit", {})
    if (
        tuple(cohort.get("global_seeds", ())) != EXPANSION_SEEDS
        or cohort.get("trajectory_count") != 360
        or cohort.get("one_shared_global_seed_initial_noise_cluster_across_three_classes")
        is not True
        or selection.get("original_locked_clear_bad_events")
        != ORIGINAL_CLEAR_BAD_EVENTS
        or selection.get("frozen_minimum_clear_bad_events")
        != MINIMUM_CLEAR_BAD_EVENTS
        or selection.get("additional_events_needed") != 7
        or selection.get("detector_scores_thresholds_alerts_or_score_label_join_used")
        is not False
        or audit.get("selection_used_only_locked_aggregate_event_count") is not True
    ):
        raise RuntimeError("event-count expansion scientific contract changed")
    return protocol


def validate_pipeline_source_lock(
    source_name: str, root: Path = PIPELINE_SOURCE_LOCK
) -> dict[str, Any]:
    """Validate the immutable pipeline lock and this live source snapshot."""

    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"pipeline source lock must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    protocol_path = root / "pipeline_protocol.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    protocol = load_json(protocol_path)
    require_canonical_identity(manifest, "pipeline source manifest")
    require_canonical_identity(protocol, "pipeline source protocol")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("protocol_identity_sha256") != protocol.get("identity_sha256")
        or manifest.get("protocol_identity_sha256") != protocol.get("identity_sha256")
        or protocol.get("status")
        != "FROZEN_BEFORE_ANY_EXPANSION_ENDPOINT_REVIEW_OR_SCORE_EXTRACTION"
        or protocol.get("candidate_protocol_identity_sha256")
        != CANDIDATE_PROTOCOL_IDENTITY
        or protocol.get("event_count_expansion_protocol_identity_sha256")
        != EXPANSION_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("expansion pipeline source lock changed")
    files = {str(item.get("name")): item for item in manifest.get("files", [])}
    for name, item in files.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"pipeline source-lock member changed: {path}")
    snapshot = protocol.get("source_snapshots", {}).get(source_name)
    live = ROOT / "experiments" / source_name
    locked = root / "sources" / source_name
    if (
        not isinstance(snapshot, dict)
        or not live.is_file()
        or live.is_symlink()
        or not locked.is_file()
        or locked.is_symlink()
        or sha256_file(live) != snapshot.get("sha256")
        or sha256_file(locked) != snapshot.get("sha256")
    ):
        raise RuntimeError(f"live source differs from pipeline freeze: {source_name}")
    return protocol


def require_planned_path(
    protocol: dict[str, Any], key: str, actual: Path, *, reviewer: str | None = None
) -> Path:
    """Require a CLI input/output path to equal its source-frozen planned path."""

    planned: Any = protocol.get("planned_paths", {}).get(key)
    if reviewer is not None:
        if not isinstance(planned, dict):
            raise RuntimeError(f"pipeline plan lacks reviewer paths: {key}")
        planned = planned.get(reviewer)
    if not isinstance(planned, str):
        raise RuntimeError(f"pipeline plan lacks path: {key}")
    expected = Path(planned).expanduser().absolute()
    observed = actual.expanduser().absolute()
    if expected != observed:
        suffix = f"/{reviewer}" if reviewer is not None else ""
        raise RuntimeError(
            f"path differs from pipeline freeze for {key}{suffix}: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def expansion_protocol() -> dict[str, Any]:
    """Return the pre-score expansion contract without reading any evidence."""

    return {
        "schema_version": 1,
        "status": "FROZEN_DISJOINT_EVENT_COUNT_DRIVEN_EXPANSION",
        "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
        "classes": list(CLASSES),
        "calibration_seeds_reused_without_reestimation": list(CALIBRATION_SEEDS),
        "original_evaluation_seeds": list(ORIGINAL_EVALUATION_SEEDS),
        "expansion_seeds": list(EXPANSION_SEEDS),
        "disjointness": {
            "calibration_vs_expansion": not bool(
                set(CALIBRATION_SEEDS) & set(EXPANSION_SEEDS)
            ),
            "original_evaluation_vs_expansion": not bool(
                set(ORIGINAL_EVALUATION_SEEDS) & set(EXPANSION_SEEDS)
            ),
        },
        "expansion_decision_input": {
            "allowed": "locked aggregate original clear-bad event count only",
            "original_clear_bad_events": ORIGINAL_CLEAR_BAD_EVENTS,
            "candidate_scores_thresholds_alerts_or_row_labels_used": False,
        },
        "frozen_score": {
            "A": (
                "learned_range_cond_minus_uncond_logstd_gap_tile4x4_"
                "concentration_guided3__q2_max_positive_jump"
            ),
            "A_orientation": "lower_is_bad; z_A=(-A-median(-A))/scale",
            "B": (
                "pred_xstart_cond_uncond_disagreement_rms_channel4__"
                "q1_centered_cusum_range"
            ),
            "B_orientation": "higher_is_bad; z_B=(B-median(B))/scale",
            "combination": "S_UNION=max(z_A,z_B)",
            "normalizers": "candidate-v5 discovery class medians/MAD scales",
        },
        "fixed_calibration": (
            "reuse the immutable seeds30..49 split-conformal thresholds; "
            "never pool expansion values into calibration"
        ),
        "labels_first_execution_order": (
            "feature extraction may run before review, but expansion A/B/S scoring, "
            "threshold application, and all score access wait until the final expansion "
            "visual-label lock exists"
        ),
        "visual_labels": {
            "reviewers": list(REVIEWERS),
            "independent_endpoint_only": True,
            "scores_metrics_thresholds_alerts_trajectories_visible": False,
            "consensus": "2-of-3; >=2 severity is clear_bad; >=2 zeros is clean_good",
            "adjudication": "raw clear_bad may only be retained or downgraded",
        },
        "opening_gate": {
            "original_clear_bad_events": ORIGINAL_CLEAR_BAD_EVENTS,
            "minimum_combined_clear_bad_events": MINIMUM_CLEAR_BAD_EVENTS,
            "minimum_new_clear_bad_events": (
                MINIMUM_CLEAR_BAD_EVENTS - ORIGINAL_CLEAR_BAD_EVENTS
            ),
            "below_gate": (
                "emit aggregate event-count receipt and exit before opening original "
                "row labels, calibration, or either score/alert product"
            ),
        },
        "combined_evaluation": {
            "primary_statistic": "class-matched pair-weighted ROC AUC of S_UNION",
            "permutation": {
                "method": "within-class label permutation preserving bad/good counts",
                "draws": PERMUTATION_DRAWS,
                "seed": PERMUTATION_SEED,
                "one_sided": True,
            },
            "gates_unchanged_from_candidate_v5": True,
            "published_outputs": "aggregate only; never joined rows or individual ranks",
        },
    }


def self_test() -> None:
    assert not (set(CALIBRATION_SEEDS) & set(EXPANSION_SEEDS))
    assert not (set(ORIGINAL_EVALUATION_SEEDS) & set(EXPANSION_SEEDS))
    assert len(EXPANSION_SEEDS) == 120
    assert len(expected_keys()) == 360
    assert MINIMUM_CLEAR_BAD_EVENTS - ORIGINAL_CLEAR_BAD_EVENTS == 7
    assert expansion_protocol()["combined_evaluation"]["permutation"]["draws"] == 100_000
    assert validate_expansion_lock()["identity_sha256"] == EXPANSION_PROTOCOL_IDENTITY


if __name__ == "__main__":
    self_test()
    print(json.dumps(expansion_protocol(), ensure_ascii=False, indent=2))
