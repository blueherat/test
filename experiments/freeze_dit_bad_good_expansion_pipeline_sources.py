#!/usr/bin/env python3
"""Freeze the post-sampling expansion analysis/review/evaluation pipeline.

This freezer reads only source code, candidate/expansion protocols, and the
already locked aggregate event-count lineage.  It never opens expansion
traces/endpoints, feature products, scores, calibration members, alerts,
review drafts, or row-level labels.  The output directory is immutable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from .dit_bad_good_expansion_contract import (
        CANDIDATE_PROTOCOL_IDENTITY,
        EXPANSION_PROTOCOL_IDENTITY,
        ROOT,
        canonical_sha256,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        write_json,
    )
except ImportError:  # pragma: no cover
    from dit_bad_good_expansion_contract import (
        CANDIDATE_PROTOCOL_IDENTITY,
        EXPANSION_PROTOCOL_IDENTITY,
        ROOT,
        canonical_sha256,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        write_json,
    )


DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
DEFAULT_OUTPUT = (
    ROOT / "experiments/locks/dit_bad_good_expansion_pipeline_source_lock_v3"
)
SOURCE_NAMES = (
    "dit_bad_good_expansion_contract.py",
    "analyze_dit_bad_good_custom_traces.py",
    "analyze_dit_posterior_evidence_metrics.py",
    "score_dit_bad_good_frozen_candidates.py",
    "build_dit_bad_good_fresh_blind_review_pack.py",
    "score_dit_bad_good_expansion_candidates.py",
    "apply_dit_bad_good_expansion_thresholds.py",
    "build_dit_bad_good_expansion_blind_review_pack.py",
    "lock_dit_bad_good_expansion_labels.py",
    "evaluate_dit_bad_good_combined_expansion.py",
    "freeze_dit_bad_good_expansion_pipeline_sources.py",
)


def planned_paths(expansion_protocol: dict[str, Any]) -> dict[str, Any]:
    base = DATA_ROOT / "cross_scale_evidence/bad_good_metric_confirmation_expansion_v1"
    original = DATA_ROOT / "cross_scale_evidence/bad_good_metric_confirmation_v1"
    annotation = ROOT / "experiments/annotations"
    expansion_lineage = expansion_protocol["input_lineage"]
    return {
        "trace_root": str(
            DATA_ROOT
            / "cross_scale_evidence/dit_bad_good_confirmation_expansion_v1_custom_traces_cfg_locked"
        ),
        "label_free_primary_features": str(base / "primary_label_free_v1"),
        "label_free_posterior_features": str(base / "posterior_label_free_v1"),
        "endpoint_only_blind_pack": str(base / "blind_review_expansion_seed130_249_v1"),
        "review_drafts": {
            reviewer: str(
                annotation / f"dit_expansion_eval360_review_{reviewer}_v1_draft.json"
            )
            for reviewer in "JKL"
        },
        "raw_consensus_lock": str(
            annotation / "dit_expansion_eval360_consensus_lock_v1"
        ),
        "adjudication_draft": str(
            annotation / "dit_expansion_eval360_adjudication_v1_draft.json"
        ),
        "final_consensus_lock": str(
            annotation / "dit_expansion_eval360_adjudicated_consensus_lock_v1"
        ),
        "label_free_candidate_scores": str(base / "frozen_candidate_scores_label_free_v1"),
        "label_free_calibrated_alerts": str(base / "calibrated_alerts_label_free_v1"),
        "combined_aggregate_result": str(base / "combined_confirmation_result_v1"),
        "candidate_v5_lock": str(
            ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
        ),
        "event_count_expansion_lock": str(
            ROOT / "experiments/locks/dit_bad_good_event_count_expansion_lock_v1"
        ),
        "calibration_lock": str(
            ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"
        ),
        "original_event_receipt": expansion_lineage["event_count_only_result"]["path"],
        "original_final_consensus_lock": expansion_lineage[
            "final_visual_consensus_aggregate"
        ]["path"],
        "original_label_free_alerts": str(
            original / "calibrated_evaluation_alerts_label_free_v1"
        ),
    }


def build_protocol() -> dict[str, Any]:
    candidate = validate_candidate_lock()
    expansion = validate_expansion_lock()
    sources: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        path = ROOT / "experiments" / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"pipeline source is missing or indirect: {path}")
        sources[name] = {
            "live_path_at_freeze": str(path),
            "sha256": sha256_file(path),
        }
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ANY_EXPANSION_ENDPOINT_REVIEW_OR_SCORE_EXTRACTION",
        "candidate_protocol_identity_sha256": candidate["identity_sha256"],
        "event_count_expansion_protocol_identity_sha256": expansion[
            "identity_sha256"
        ],
        "planned_paths": planned_paths(expansion),
        "source_snapshots": sources,
        "execution_order": [
            "complete and validate observation-only expansion sampling",
            "freeze this pipeline source lock",
            "extract primary and posterior features label-free (no scoring)",
            "build endpoint-only blind pack",
            "three independent endpoint-only reviews",
            "lock raw consensus and visual-only no-promotion adjudication",
            "only after final expansion label lock, compute frozen A/B/S_UNION scores",
            "apply immutable seeds30..49 thresholds without re-estimation",
            "combined evaluator first counts 8 plus expansion bad events",
            "below 15: event-count-only return before original row labels/calibration/scores",
            "at least 15: open both alert products, join in memory, publish aggregate only",
        ],
        "reuse_vs_wrapper": {
            "directly_parameterized": [
                "analyze_dit_bad_good_custom_traces.py --trace-root/--trace-glob/--expected-seeds",
                "analyze_dit_posterior_evidence_metrics.py --trace-root/--trace-glob/--expected-seeds",
            ],
            "new_expansion_wrappers_required_because_original_is_seed_or_lineage_hardcoded": [
                "score_dit_bad_good_expansion_candidates.py",
                "apply_dit_bad_good_expansion_thresholds.py",
                "build_dit_bad_good_expansion_blind_review_pack.py",
                "lock_dit_bad_good_expansion_labels.py",
                "evaluate_dit_bad_good_combined_expansion.py",
            ],
        },
        "immutable_statistical_contract": {
            "candidate": "S_UNION=max(z_A,z_B)",
            "normalizers": "candidate-v5 discovery medians/MAD scales",
            "calibration": "unchanged seeds30..49 class thresholds",
            "permutation": "100000 within-class label permutations; PCG64 seed 2026082701",
            "initial_go_gates_changed": False,
            "aggregate_only_output": True,
        },
        "original_aggregate_lineage_copied_without_row_decode": expansion[
            "input_lineage"
        ],
        "evidence_access_audit": {
            "expansion_trace_or_endpoint_opened": False,
            "feature_or_metric_product_opened": False,
            "candidate_score_file_opened": False,
            "calibration_member_or_threshold_opened": False,
            "alert_file_opened": False,
            "review_or_row_level_label_file_opened": False,
            "label_score_join_performed": False,
            "only_source_and_protocol_inputs": True,
        },
    }
    if (
        protocol["candidate_protocol_identity_sha256"]
        != CANDIDATE_PROTOCOL_IDENTITY
        or protocol["event_count_expansion_protocol_identity_sha256"]
        != EXPANSION_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("upstream protocol identity mismatch")
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol


def publish(output: Path) -> Path:
    protocol = build_protocol()
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite expansion pipeline source lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "pipeline_protocol.json", protocol)
        source_root = staging / "sources"
        source_root.mkdir()
        for name in SOURCE_NAMES:
            shutil.copy2(ROOT / "experiments" / name, source_root / name)
        files = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            files.append(
                {
                    "name": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "protocol_identity_sha256": protocol["identity_sha256"],
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "protocol_file_sha256": sha256_file(staging / "pipeline_protocol.json"),
                "protocol_identity_sha256": protocol["identity_sha256"],
                "source_snapshot_count": len(SOURCE_NAMES),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    protocol = build_protocol()
    assert protocol["candidate_protocol_identity_sha256"] == CANDIDATE_PROTOCOL_IDENTITY
    assert protocol["event_count_expansion_protocol_identity_sha256"] == EXPANSION_PROTOCOL_IDENTITY
    assert len(protocol["source_snapshots"]) == len(SOURCE_NAMES)
    assert protocol["evidence_access_audit"]["only_source_and_protocol_inputs"] is True
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    output = publish(args.output)
    protocol = build_protocol()
    print(
        {
            "output": str(output),
            "protocol_identity_sha256": protocol["identity_sha256"],
            "source_snapshot_count": len(SOURCE_NAMES),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
