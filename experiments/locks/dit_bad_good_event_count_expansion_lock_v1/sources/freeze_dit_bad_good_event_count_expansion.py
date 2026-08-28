#!/usr/bin/env python3
"""Freeze the event-count-only expansion of the DiT bad/good confirmation.

This locker is deliberately blind to candidate scores, thresholds, alerts, and
all sample-level label/score joins.  It reads only the already public aggregate
event-count receipt and the aggregate metadata of the final visual-label lock.
The resulting lock fixes the sampler, classes, and disjoint global seeds before
any expansion trajectory is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))

DEFAULT_CANDIDATE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
)
DEFAULT_EVENT_RESULT = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_confirmation_v1"
    / "prospective_confirmation_result_v1"
)
DEFAULT_CONSENSUS_LOCK = (
    ROOT / "experiments/annotations/dit_fresh_eval240_adjudicated_consensus_lock_v2"
)
DEFAULT_RUNNER_SOURCE = (
    ROOT / "experiments/run_dit_bad_good_event_count_expansion_pool.py"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments/locks/dit_bad_good_event_count_expansion_lock_v1"
)

EXPECTED_CANDIDATE_IDENTITY = (
    "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"
)
EXPECTED_EVENT_RESULT_IDENTITY = (
    "4791dafe591823b22c3b89aaca5bcf287a06493cc65b13be740e81c389a88e31"
)
EXPECTED_CONSENSUS_IDENTITY = (
    "21c242dc796d5c8baa4568c9f82add0d1b64c984477cf8698efbbca5889e166a"
)

CLASSES = (207, 602, 795)
SEEDS = tuple(range(130, 250))
ORIGINAL_EVALUATION_TRAJECTORIES = 240
ORIGINAL_CLEAR_BAD_EVENTS = 8
MINIMUM_CLEAR_BAD_EVENTS = 15
REQUIRED_ADDITIONAL_EVENTS = MINIMUM_CLEAR_BAD_EVENTS - ORIGINAL_CLEAR_BAD_EVENTS
EXPANSION_TRAJECTORIES = len(CLASSES) * len(SEEDS)

SOURCE_PATHS = {
    "trace_dit_imagenet256_custom_batch.py": (
        ROOT / "experiments/trace_dit_imagenet256_custom_batch.py"
    ),
    "sample_dit_imagenet256_custom.py": (
        ROOT / "experiments/sample_dit_imagenet256_custom.py"
    ),
    "reproduce_dit_imagenet256.py": (
        ROOT / "experiments/reproduce_dit_imagenet256.py"
    ),
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def binomial_tail_at_least(n: int, p: float, threshold: int) -> float:
    if type(n) is not int or type(threshold) is not int or not 0 <= threshold <= n:
        raise ValueError("invalid binomial n or threshold")
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("invalid binomial probability")
    return float(
        1.0
        - sum(
            math.comb(n, count)
            * p**count
            * (1.0 - p) ** (n - count)
            for count in range(threshold)
        )
    )


def validate_candidate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"candidate lock must be a real directory: {root}")
    protocol_path = root / "candidate_protocol.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        completion.get("complete") is not True
        or protocol.get("identity_sha256") != EXPECTED_CANDIDATE_IDENTITY
        or completion.get("protocol_identity_sha256") != EXPECTED_CANDIDATE_IDENTITY
        or manifest.get("protocol_identity_sha256") != EXPECTED_CANDIDATE_IDENTITY
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
    ):
        raise RuntimeError("candidate v5 lock failed identity/hash validation")
    fresh = protocol.get("fresh_confirmation", {})
    primary = protocol.get("primary_candidate", {})
    if (
        tuple(fresh.get("classes", ())) != CLASSES
        or fresh.get("model") != "DiT-XL/2 ImageNet-256"
        or fresh.get("sampler") != "official 250-step ancestral DDPM"
        or fresh.get("cfg_scale") != 4.0
        or primary.get("name") != "S_UNION"
        or primary.get("formula") != "max(z_A, z_B)"
    ):
        raise RuntimeError("candidate v5 scientific contract changed")
    original_seeds = fresh.get("seeds", {})
    if (
        original_seeds.get("start_inclusive") != 30
        or original_seeds.get("stop_inclusive") != 129
        or original_seeds.get("count") != 100
        or set(SEEDS).intersection(range(30, 130))
    ):
        raise RuntimeError("expansion seeds are not disjoint from candidate v5")
    return protocol, {
        "path": str(root.resolve()),
        "protocol_identity_sha256": EXPECTED_CANDIDATE_IDENTITY,
        "protocol_file_sha256": sha256_file(protocol_path),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "completion_file_sha256": sha256_file(completion_path),
    }


def validate_event_result(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"event result must be a real directory: {root}")
    result_path = root / "confirmation_results.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    result = load_json(result_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        result.get("identity_sha256") != EXPECTED_EVENT_RESULT_IDENTITY
        or completion.get("complete") is not True
        or completion.get("aggregate_only") is not True
        or completion.get("published_table_count") != 0
        or completion.get("result_identity_sha256") != EXPECTED_EVENT_RESULT_IDENTITY
        or completion.get("result_file_sha256") != sha256_file(result_path)
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or manifest.get("result_identity_sha256") != EXPECTED_EVENT_RESULT_IDENTITY
        or manifest.get("aggregate_only") is not True
    ):
        raise RuntimeError("event-count-only result failed identity/hash validation")
    audit = result.get("evidence_access_audit", {})
    expected_audit = {
        "aggregate_only": True,
        "calibration_lock_or_members_opened": False,
        "evaluation_alert_lock_or_score_CSV_opened": False,
        "individual_labels_or_identifiers_published": False,
        "score_label_join_performed": False,
    }
    if audit != expected_audit:
        raise RuntimeError("event result is not a score-blind aggregate-only receipt")
    counts = result.get("cohort", {}).get("aggregate_label_counts")
    gate = result.get("event_gate", {})
    lineage = result.get("input_lineage", {})
    if (
        counts != {"clean_good": 216, "clear_bad": 8, "mild_or_disputed": 16}
        or result.get("cohort", {}).get("trajectory_count")
        != ORIGINAL_EVALUATION_TRAJECTORIES
        or gate.get("observed_clear_bad_events") != ORIGINAL_CLEAR_BAD_EVENTS
        or gate.get("minimum_clear_bad_events_for_decision")
        != MINIMUM_CLEAR_BAD_EVENTS
        or gate.get("evaluated") is not False
        or gate.get("decision")
        != "PILOT_ONLY_EXPAND_DISJOINT_SEEDS_WITHOUT_CHANGING_FORMULAS"
        or lineage.get("candidate_protocol_identity_sha256")
        != EXPECTED_CANDIDATE_IDENTITY
        or lineage.get("blind_consensus_identity_sha256")
        != EXPECTED_CONSENSUS_IDENTITY
    ):
        raise RuntimeError("event count, gate, or input lineage changed")
    return result, {
        "path": str(root.resolve()),
        "result_identity_sha256": EXPECTED_EVENT_RESULT_IDENTITY,
        "result_file_sha256": sha256_file(result_path),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "completion_file_sha256": sha256_file(completion_path),
        "evidence_access_audit": expected_audit,
    }


def validate_consensus_aggregate(root: Path) -> dict[str, Any]:
    """Validate aggregate metadata without parsing sample-level consensus rows."""

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"consensus lock must be a real directory: {root}")
    consensus_path = root / "consensus_locked.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    # The manifest contains only aggregate counts and member hashes.  The large
    # sample-level consensus JSON is hashed as opaque bytes and never decoded.
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("candidate_protocol_identity_sha256")
        != EXPECTED_CANDIDATE_IDENTITY
        or manifest.get("consensus_identity_sha256") != EXPECTED_CONSENSUS_IDENTITY
        or manifest.get("counts")
        != {"clean_good": 216, "clear_bad": 8, "mild_or_disputed": 16}
        or completion.get("complete") is not True
        or completion.get("locked_row_count") != ORIGINAL_EVALUATION_TRAJECTORIES
        or completion.get("consensus_identity_sha256") != EXPECTED_CONSENSUS_IDENTITY
        or completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
    ):
        raise RuntimeError("final consensus aggregate lock failed validation")
    return {
        "path": str(root.resolve()),
        "consensus_identity_sha256": EXPECTED_CONSENSUS_IDENTITY,
        "consensus_file_sha256": sha256_file(consensus_path),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "completion_file_sha256": sha256_file(completion_path),
        "aggregate_counts": manifest["counts"],
        "sample_level_consensus_decoded_by_this_locker": False,
    }


def validate_sampler_sources(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = candidate.get("sampler_lineage_contract", {}).get(
        "source_snapshot_sha256", {}
    )
    mapping = {
        "trace_dit_imagenet256_custom_batch.py": "runner_source.py",
        "sample_dit_imagenet256_custom.py": "custom_baseline_helper.py",
        "reproduce_dit_imagenet256.py": "strict_reproduction_helper.py",
    }
    records: dict[str, dict[str, Any]] = {}
    for basename, candidate_name in mapping.items():
        path = SOURCE_PATHS[basename]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid sampler source: {path}")
        observed = sha256_file(path)
        if observed != expected.get(candidate_name):
            raise RuntimeError(
                f"sampler source differs from candidate v5: {basename} {observed}"
            )
        records[basename] = {
            "live_path_at_freeze": str(path.resolve()),
            "sha256": observed,
            "candidate_v5_snapshot_name": candidate_name,
        }
    runner = DEFAULT_RUNNER_SOURCE
    if not runner.is_file() or runner.is_symlink():
        raise RuntimeError(f"expansion runner source is missing or indirect: {runner}")
    records[runner.name] = {
        "live_path_at_freeze": str(runner.resolve()),
        "sha256": sha256_file(runner),
        "candidate_v5_snapshot_name": None,
    }
    locker = Path(__file__).resolve()
    records[locker.name] = {
        "live_path_at_freeze": str(locker),
        "sha256": sha256_file(locker),
        "candidate_v5_snapshot_name": None,
    }
    return records


def build_protocol(
    candidate_root: Path,
    event_root: Path,
    consensus_root: Path,
) -> dict[str, Any]:
    candidate, candidate_lineage = validate_candidate_lock(candidate_root)
    _, event_lineage = validate_event_result(event_root)
    consensus_lineage = validate_consensus_aggregate(consensus_root)
    sources = validate_sampler_sources(candidate)
    p_hat = ORIGINAL_CLEAR_BAD_EVENTS / ORIGINAL_EVALUATION_TRAJECTORIES
    planning_probability = binomial_tail_at_least(
        EXPANSION_TRAJECTORIES, p_hat, REQUIRED_ADDITIONAL_EVENTS
    )
    if not math.isclose(planning_probability, 0.956720557701896, abs_tol=1e-15):
        raise RuntimeError("plug-in binomial planning calculation changed")

    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "FROZEN_AFTER_EVENT_COUNT_ONLY_GATE_BEFORE_EXPANSION_SAMPLING_OR_SCORE_ACCESS"
        ),
        "objective": (
            "Increase the number of independently and blindly labeled clear-bad events "
            "without changing the frozen S_UNION candidate or consulting its scores."
        ),
        "scientific_contract": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "sampling_steps": 250,
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "classes_ordered": list(CLASSES),
            "candidate_name": "S_UNION",
            "candidate_formula": "max(z_A, z_B)",
            "candidate_formula_or_normalization_changes_allowed": False,
            "model_sampler_cfg_or_class_changes_allowed": False,
            "observation_only_during_trace_generation": True,
        },
        "expansion_cohort": {
            "global_seed_start_inclusive": SEEDS[0],
            "global_seed_stop_inclusive": SEEDS[-1],
            "global_seed_count": len(SEEDS),
            "global_seeds": list(SEEDS),
            "class_count_per_seed": len(CLASSES),
            "trajectory_count": EXPANSION_TRAJECTORIES,
            "disjoint_from_candidate_v5_seeds_30_129": True,
            "one_shared_global_seed_initial_noise_cluster_across_three_classes": True,
        },
        "selection_basis": {
            "only_trigger": "locked aggregate clear_bad count 8 is below the frozen minimum 15",
            "original_evaluation_trajectory_count": ORIGINAL_EVALUATION_TRAJECTORIES,
            "original_locked_clear_bad_events": ORIGINAL_CLEAR_BAD_EVENTS,
            "frozen_minimum_clear_bad_events": MINIMUM_CLEAR_BAD_EVENTS,
            "additional_events_needed": REQUIRED_ADDITIONAL_EVENTS,
            "plug_in_prevalence": p_hat,
            "chosen_new_global_seed_count": len(SEEDS),
            "chosen_new_trajectory_count": EXPANSION_TRAJECTORIES,
            "plug_in_binomial_probability_at_least_7_new_events": planning_probability,
            "calculation": "Pr[Binomial(n=360,p=8/240)>=7]",
            "interpretation": (
                "Planning heuristic only, not an inferential guarantee; 120 seeds were "
                "selected solely from 8/240 prevalence and the need for seven more events."
            ),
            "detector_scores_thresholds_alerts_or_score_label_join_used": False,
        },
        "post_expansion_rule": {
            "labels_first": (
                "Endpoint labels for all 360 paths must be locked under the same blind rubric "
                "before any expansion candidate score, threshold alert, or label-score join is opened."
            ),
            "if_cumulative_clear_bad_at_least_15": (
                "Stop event-count expansion and run the already frozen prospective evaluation."
            ),
            "if_cumulative_clear_bad_below_15": (
                "Choose any subsequent disjoint seed block only from the newly locked aggregate "
                "event count or visual-label prevalence uncertainty; start at seed 250 or later; "
                "do not inspect detector performance and do not change formulas, normalization, "
                "calibration, orientation, or decision gates."
            ),
        },
        "input_lineage": {
            "candidate_v5": candidate_lineage,
            "event_count_only_result": event_lineage,
            "final_visual_consensus_aggregate": consensus_lineage,
        },
        "source_snapshots": sources,
        "evidence_access_audit": {
            "candidate_score_files_opened": False,
            "calibration_threshold_members_opened": False,
            "evaluation_alert_files_opened": False,
            "sample_level_label_score_mapping_opened": False,
            "selection_used_only_locked_aggregate_event_count": True,
        },
        "forbidden_before_expansion_labels_lock": [
            "opening or computing expansion candidate scores",
            "opening class thresholds or expansion alert decisions",
            "joining any visual label to any trajectory metric",
            "changing S_UNION, A, B, orientation, normalization, or gates",
        ],
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol


def iter_regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"lock staging contains a symlink: {path}")
        if path.is_file():
            yield path


def publish(
    candidate_root: Path,
    event_root: Path,
    consensus_root: Path,
    output: Path,
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite expansion lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        protocol = build_protocol(
            candidate_root.resolve(), event_root.resolve(), consensus_root.resolve()
        )
        write_json(staging / "expansion_protocol.json", protocol)
        sources_dir = staging / "sources"
        sources_dir.mkdir()
        copy_sources = {
            **SOURCE_PATHS,
            DEFAULT_RUNNER_SOURCE.name: DEFAULT_RUNNER_SOURCE,
            Path(__file__).name: Path(__file__).resolve(),
        }
        for basename, source in copy_sources.items():
            shutil.copy2(source, sources_dir / basename)
        members = [
            {
                "name": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in iter_regular_files(staging)
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "protocol_identity_sha256": protocol["identity_sha256"],
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "protocol_identity_sha256": protocol["identity_sha256"],
            "protocol_file_sha256": sha256_file(staging / "expansion_protocol.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "source_snapshot_count": len(copy_sources),
        }
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def run_self_test(
    candidate_root: Path, event_root: Path, consensus_root: Path
) -> None:
    assert len(SEEDS) == 120 and SEEDS[0] == 130 and SEEDS[-1] == 249
    assert EXPANSION_TRAJECTORIES == 360 and REQUIRED_ADDITIONAL_EVENTS == 7
    assert math.isclose(
        binomial_tail_at_least(360, 8 / 240, 7),
        0.956720557701896,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    first = build_protocol(candidate_root, event_root, consensus_root)
    second = build_protocol(candidate_root, event_root, consensus_root)
    assert first == second
    identity = first.pop("identity_sha256")
    assert canonical_sha256(first) == identity
    print(
        "self-test passed: event-only lineage, aggregate count, disjoint seeds, "
        "sampler hashes, deterministic identity, and binomial planning calculation"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, default=DEFAULT_CANDIDATE_LOCK)
    parser.add_argument("--event-result", type=Path, default=DEFAULT_EVENT_RESULT)
    parser.add_argument("--consensus-lock", type=Path, default=DEFAULT_CONSENSUS_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = (
        args.candidate_lock.expanduser().absolute().resolve(),
        args.event_result.expanduser().absolute().resolve(),
        args.consensus_lock.expanduser().absolute().resolve(),
    )
    if args.self_test:
        run_self_test(*roots)
        return 0
    if args.dry_run:
        protocol = build_protocol(*roots)
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "would_write": str(args.output.expanduser().absolute()),
                    "protocol_identity_sha256": protocol["identity_sha256"],
                    "classes": protocol["scientific_contract"]["classes_ordered"],
                    "seeds": [SEEDS[0], SEEDS[-1]],
                    "seed_count": len(SEEDS),
                    "trajectory_count": EXPANSION_TRAJECTORIES,
                    "score_label_join_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = publish(*roots, args.output)
    protocol = load_json(output / "expansion_protocol.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": protocol["status"],
                "protocol_identity_sha256": protocol["identity_sha256"],
                "seed_count": len(SEEDS),
                "trajectory_count": EXPANSION_TRAJECTORIES,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
