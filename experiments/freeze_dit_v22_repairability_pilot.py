#!/usr/bin/env python3
"""Freeze the retrospective v2.2 repairability pilot without opening outcomes.

The pilot asks a deliberately narrower question than bad-image detection:
among paths whose internal B statistic is already high, does an E alarm predict
that a fixed, single-shot suffix resample is more likely to repair the endpoint?

Selection reads only the already immutable, label-free B/E score shards.  It
selects all joint B+E alarms and deterministic same-class, exact-start-schedule
B-only controls.  It never reads endpoint pixels, reviews, labels, FID, or any
endpoint representation.  The result is exploratory because the old paths and
the B/E candidate family have already been viewed; it cannot authorize a
confirmatory claim or deployment intervention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
METHOD_LOCK = ROOT / "experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_2"
METHOD_LOCK_ID = "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"
EXPECTED_CLASSES = (207, 602, 795)
EXPECTED_SEED_START = 270
EXPECTED_SEED_END = 850
EXPECTED_CELL_COUNTS = {"E0B0": 1575, "E0B1": 139, "E1B0": 18, "E1B1": 8}
NAMESPACE = "eqvae.dit.v22.repairability.pilot.v1"
PREVIOUS_LOCK = ROOT / "experiments/locks/dit_v22_repairability_pilot_lock_v1_1"
PREVIOUS_LOCK_ID = "8c601a62b888670b20acd8b888525ea614e15e5d1082dcd52f087091950e39dc"
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_v22_repairability_pilot_lock_v1_2"

SCORE_COLUMNS = (
    "global_seed",
    "class_id",
    "B_persistence",
    "B_alarm",
    "E_blur_gated_running_max_log",
    "E_blur_gated_alarm",
    "E_no_state_gate_running_max_log",
    "E_first_hit_full_budget_running_max_log",
    "G_start_schedule_diagnostic",
    "T_delta1",
    "h_delta1",
    "T_delta4",
    "h_delta4",
    "fallback_steps_delta1",
    "fallback_steps_delta4",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"invalid {key}: {path}")
    return value


def row_schedule(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(
        int(row[name])
        for name in ("T_delta1", "h_delta1", "T_delta4", "h_delta4")
    )


def selection_key(row: dict[str, str]) -> str:
    value = (
        f"{NAMESPACE}\0control\0{row['global_seed']}\0{row['class_id']}"
    ).encode("ascii")
    return hashlib.sha256(value).hexdigest()


def validate_score_shard(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"invalid score shard root: {root}")
    manifest = self_hashed_json(root / "manifest.json", "identity_sha256")
    if manifest.get("artifact_kind") != "RETROSPECTIVE_DIT_V22_LABEL_FREE_SCORE_SHARD":
        raise RuntimeError("wrong score shard kind")
    if manifest.get("method_lock_identity_sha256") != METHOD_LOCK_ID:
        raise RuntimeError("score shard method lock changed")
    if tuple(manifest.get("classes_ordered", ())) != EXPECTED_CLASSES:
        raise RuntimeError("score shard class axis changed")
    records = {item.get("name"): item for item in manifest.get("files", [])}
    for name in ("scores.csv", "mechanics.npz", "calibration.json"):
        path = root / name
        record = records.get(name)
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.is_symlink()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"score shard member changed: {path}")
    calibration = self_hashed_json(root / "calibration.json", "identity_sha256")
    if calibration.get("identity_sha256") != manifest.get("calibration_identity_sha256"):
        raise RuntimeError("calibration identity changed")
    if tuple(calibration.get("classes_ordered", ())) != EXPECTED_CLASSES:
        raise RuntimeError("calibration class axis changed")
    thresholds = calibration.get("B_alarm_threshold_by_class")
    if not isinstance(thresholds, list) or len(thresholds) != len(EXPECTED_CLASSES):
        raise RuntimeError("B alarm thresholds changed")
    with (root / "scores.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCORE_COLUMNS:
            raise RuntimeError("score columns changed")
        rows = list(reader)
    seed_start = int(manifest["seed_start_inclusive"])
    seed_end = int(manifest["seed_end_exclusive"])
    expected_axis = [
        (seed, class_id)
        for seed in range(seed_start, seed_end)
        for class_id in EXPECTED_CLASSES
    ]
    observed_axis = [(int(row["global_seed"]), int(row["class_id"])) for row in rows]
    if observed_axis != expected_axis or len(rows) != int(manifest["row_count"]):
        raise RuntimeError("score shard seed/class axis changed")
    threshold_by_class = dict(zip(EXPECTED_CLASSES, map(float, thresholds), strict=True))
    for row in rows:
        class_id = int(row["class_id"])
        b_score = float(row["B_persistence"])
        e_score = float(row["E_blur_gated_running_max_log"])
        b_alarm = int(row["B_alarm"])
        e_alarm = int(row["E_blur_gated_alarm"])
        numeric = [float(row[name]) for name in SCORE_COLUMNS if name not in {
            "global_seed", "class_id", "B_alarm", "E_blur_gated_alarm",
            "T_delta1", "h_delta1", "T_delta4", "h_delta4",
            "fallback_steps_delta1", "fallback_steps_delta4",
        }]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError("non-finite score row")
        if b_alarm != int(b_score > threshold_by_class[class_id]):
            raise RuntimeError("B alarm formula changed")
        if e_alarm != int(e_score >= math.log(10.0)):
            raise RuntimeError("E alarm formula changed")
    return manifest, rows


def selected_record(
    row: dict[str, str],
    role: str,
    *,
    pair_index: int,
    matched_case: dict[str, str] | None = None,
) -> dict[str, Any]:
    class_id = int(row["class_id"])
    slot = EXPECTED_CLASSES.index(class_id)
    return {
        "role": role,
        "pair_index": pair_index,
        "global_seed": int(row["global_seed"]),
        "class_id": class_id,
        "class_slot": slot,
        "B_persistence": float(row["B_persistence"]),
        "B_alarm": int(row["B_alarm"]),
        "E_running_max_log": float(row["E_blur_gated_running_max_log"]),
        "E_alarm": int(row["E_blur_gated_alarm"]),
        "start_schedule": list(row_schedule(row)),
        "matched_case_global_seed": (
            None if matched_case is None else int(matched_case["global_seed"])
        ),
        "B_persistence_difference_from_case": (
            None
            if matched_case is None
            else float(row["B_persistence"]) - float(matched_case["B_persistence"])
        ),
    }


def minimum_cost_b_matches(
    cases: list[dict[str, str]], candidates: list[dict[str, str]]
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Return the monotone one-dimensional minimum-L1 matching.

    For absolute distance, an optimal one-to-one matching has no crossings, so
    dynamic programming over the two B-sorted axes gives the exact minimum.
    Selection hashes break exact floating-cost ties deterministically.
    """

    ordered_cases = sorted(
        cases, key=lambda row: (float(row["B_persistence"]), selection_key(row))
    )
    ordered_candidates = sorted(
        candidates, key=lambda row: (float(row["B_persistence"]), selection_key(row))
    )
    if len(ordered_candidates) < len(ordered_cases):
        raise RuntimeError("insufficient controls for B matching")

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[float, tuple[int, ...], tuple[str, ...]] | None:
        if i == 0:
            return (0.0, (), ())
        if j == 0 or j < i:
            return None
        skip = solve(i, j - 1)
        prior = solve(i - 1, j - 1)
        match = None
        if prior is not None:
            cost = prior[0] + abs(
                float(ordered_cases[i - 1]["B_persistence"])
                - float(ordered_candidates[j - 1]["B_persistence"])
            )
            match = (
                cost,
                prior[1] + (j - 1,),
                prior[2] + (selection_key(ordered_candidates[j - 1]),),
            )
        options = [value for value in (skip, match) if value is not None]
        return min(options, key=lambda value: (round(value[0], 15), value[2]))

    optimum = solve(len(ordered_cases), len(ordered_candidates))
    if optimum is None:
        raise RuntimeError("B matching failed")
    selected_controls = [ordered_candidates[index] for index in optimum[1]]
    return list(zip(ordered_cases, selected_controls, strict=True))


def freeze(args: argparse.Namespace) -> None:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite lock: {output}")
    method_manifest = self_hashed_json(METHOD_LOCK / "manifest.json", "identity_sha256")
    if method_manifest.get("identity_sha256") != METHOD_LOCK_ID:
        raise RuntimeError("v2.2 method lock identity changed")
    previous_manifest = self_hashed_json(PREVIOUS_LOCK / "manifest.json", "identity_sha256")
    if previous_manifest.get("identity_sha256") != PREVIOUS_LOCK_ID:
        raise RuntimeError("superseded v1 pilot lock identity changed")

    all_rows: list[dict[str, str]] = []
    shard_records: list[dict[str, Any]] = []
    calibration_ids: set[str] = set()
    expected_next = EXPECTED_SEED_START
    for shard in args.score_shards:
        manifest, rows = validate_score_shard(shard)
        if int(manifest["seed_start_inclusive"]) != expected_next:
            raise RuntimeError("score shards are not contiguous in the frozen order")
        expected_next = int(manifest["seed_end_exclusive"])
        calibration_ids.add(str(manifest["calibration_identity_sha256"]))
        all_rows.extend(rows)
        shard_records.append(
            {
                "path": str(shard.expanduser().resolve()),
                "identity_sha256": manifest["identity_sha256"],
                "scores_csv_sha256": sha256_file(shard.expanduser().resolve() / "scores.csv"),
                "seed_start_inclusive": manifest["seed_start_inclusive"],
                "seed_end_exclusive": manifest["seed_end_exclusive"],
            }
        )
    if expected_next != EXPECTED_SEED_END or len(calibration_ids) != 1:
        raise RuntimeError("full retrospective score axis or calibration identity changed")

    observed_counts = Counter(
        f"E{int(row['E_blur_gated_alarm'])}B{int(row['B_alarm'])}" for row in all_rows
    )
    if dict(observed_counts) != EXPECTED_CELL_COUNTS:
        raise RuntimeError(f"frozen 2x2 counts changed: {dict(observed_counts)}")

    joint = [
        row for row in all_rows
        if int(row["E_blur_gated_alarm"]) == 1 and int(row["B_alarm"]) == 1
    ]
    joint.sort(key=lambda row: (int(row["class_id"]), int(row["global_seed"])))
    if len(joint) != 8:
        raise RuntimeError("joint E+B count changed")
    joint_schedule = {row_schedule(row) for row in joint}
    if joint_schedule != {(4, 5, 1, 8)}:
        raise RuntimeError("joint E+B schedule support changed")
    class_targets = Counter(int(row["class_id"]) for row in joint)

    control_pool = [
        row for row in all_rows
        if int(row["E_blur_gated_alarm"]) == 0
        and int(row["B_alarm"]) == 1
        and row_schedule(row) == (4, 5, 1, 8)
        and int(row["class_id"]) in class_targets
    ]
    matched_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for class_id, count in sorted(class_targets.items()):
        class_cases = [row for row in joint if int(row["class_id"]) == class_id]
        candidates = [row for row in control_pool if int(row["class_id"]) == class_id]
        if len(candidates) < count:
            raise RuntimeError(f"insufficient exact-schedule B-only controls: {class_id}")
        matched_pairs.extend(minimum_cost_b_matches(class_cases, candidates))

    matched_pairs.sort(key=lambda pair: (int(pair[0]["class_id"]), int(pair[0]["global_seed"])))
    selected = [
        selected_record(case, "joint_E_and_B", pair_index=index)
        for index, (case, _) in enumerate(matched_pairs)
    ]
    selected += [
        selected_record(
            control,
            "B_only_exact_schedule_B_matched_control",
            pair_index=index,
            matched_case=case,
        )
        for index, (case, control) in enumerate(matched_pairs)
    ]
    if Counter(row["class_id"] for row in selected[: len(joint)]) != Counter(
        row["class_id"] for row in selected[len(joint) :]
    ):
        raise RuntimeError("control class counts do not match joint cases")

    protocol: dict[str, Any] = {
        "schema_version": 1,
        "scientific_revision": "v1.2",
        "status": "RETROSPECTIVE_EXPLORATORY_MECHANISM_ONLY_NO_INTERVENTION_AUTHORITY",
        "supersedes": {
            "lock": "experiments/locks/dit_v22_repairability_pilot_lock_v1_1",
            "lock_identity_sha256": PREVIOUS_LOCK_ID,
            "preserved_immutable": True,
            "repair_outputs_or_repair_reviews_opened_before_supersession": False,
            "reason": (
                "v1.1 incorrectly described step149 as a pre-transition decision even though "
                "the frozen E alarm includes the step149 innovation; v1.2 fixes the decision/"
                "rollback timing and narrows the group comparison to predictive effect moderation"
            ),
        },
        "question": (
            "Conditional on the frozen internal B alarm, does the frozen E>=10 event "
            "predict successful single-shot suffix repair rather than visual severity?"
        ),
        "method_roles": {
            "B": "internal preterminal blur/soft-fusion phenotype monitor; not an e-process",
            "E": "internal anytime innovation-alignment e-process; not a quality probability",
            "J": "E_alarm AND B_alarm; a subset of the E alarm event, not a new e-process",
            "external_review": "sealed outcome readout only; never a trigger or selector",
        },
        "forbidden_selection_or_intervention_inputs": [
            "endpoint pixels",
            "old or new visual labels",
            "FID or batch endpoint metrics",
            "Inception, DINO, CLIP, or other endpoint representations",
            "best-of-N quality ranking",
        ],
        "selection": {
            "case_rule": "all E_alarm=1 and B_alarm=1 paths in the frozen old pool",
            "control_rule": (
                "exact minimum-total-absolute-B-distance matching without replacement from "
                "E_alarm=0,B_alarm=1, within class and exact "
                "(T1,h1,T4,h4)=(4,5,1,8); fixed hashes break exact ties"
            ),
            "selection_namespace": NAMESPACE,
            "endpoint_or_label_opened_for_selection": False,
        },
        "intervention": {
            "sampler": "frozen official DiT-XL/2 250-step ancestral DDPM, CFG=4",
            "rollback_sampling_steps": [109, 149],
            "rollback_meaning": {
                "109": "start-informed deeper rollback: state before the delta1 start transition",
                "149": "shallow rollback: restore the saved state before the ninth-checkpoint transition",
            },
            "decision_time": (
                "after the sampling-step149 transition has been observed, so the complete frozen "
                "E running maximum through that innovation and the ninth-draft B persistence are "
                "both available; then restore the saved state_before at step109 or step149"
            ),
            "online_requirement": (
                "a deployable sampler must retain the relevant checkpoint state; it cannot claim "
                "to know the full E alarm before observing the step149 innovation"
            ),
            "fresh_attempts_per_path_and_step": 4,
            "baseline_replay_attempt": 0,
            "fresh_suffixes_ranked_or_selected": False,
            "retry_loop": False,
            "termination": "every branch runs exactly once to t=0",
        },
        "evaluation_frozen_before_outputs": {
            "unit": "path x rollback_step x fresh_attempt, clustered by original path",
            "blind_pair": "baseline endpoint versus one fresh suffix endpoint, randomized left/right",
            "successful_repair": (
                "visible blur/fusion is clearly reduced AND class/identity/object count/main pose/"
                "composition are preserved AND no new equally severe defect appears"
            ),
            "primary_readout": (
                "per-path successful-repair fraction in joint E+B cases and in matched B-only "
                "controls, with the descriptive between-role difference clustered by original path"
            ),
            "repair_opportunity_guard": (
                "blindly adjudicate the baseline endpoint first; report baseline visible-defect "
                "opportunity by role, and call the between-role comparison inconclusive if either "
                "role has fewer than four paths with a repairable baseline defect"
            ),
            "secondary": [
                "step109 versus step149 repair probability",
                "semantic drift rate",
                "worsening rate",
            ],
            "claim_limit": (
                "exploratory predictive effect-modification screen only: suffix noise is randomized "
                "within a path, but E group membership is observational and is not a causal treatment; "
                "positive results must be frozen and replicated on disjoint fresh paths"
            ),
        },
        "mathematical_limits": {
            "joint_event_budget": "P_P(J)<=P_P(E_alarm)<=alpha=0.1",
            "joint_is_eprocess": False,
            "clean_conditional_fpr_controlled": False,
            "suffix_retry_bound_claimed": False,
            "TV_bound_claimed_for_this_retrospective_screen": False,
        },
        "method_lock_identity_sha256": METHOD_LOCK_ID,
        "score_shards": shard_records,
        "calibration_identity_sha256": next(iter(calibration_ids)),
        "full_pool_cell_counts": EXPECTED_CELL_COUNTS,
        "selected_paths": selected,
        "matching_diagnostics": {
            "pair_count": len(matched_pairs),
            "total_absolute_B_difference": sum(
                abs(float(control["B_persistence"]) - float(case["B_persistence"]))
                for case, control in matched_pairs
            ),
            "maximum_absolute_B_difference": max(
                abs(float(control["B_persistence"]) - float(case["B_persistence"]))
                for case, control in matched_pairs
            ),
        },
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "protocol.json", protocol)
        manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_REPAIRABILITY_PILOT_LOCK_V1_2",
            "protocol_identity_sha256": protocol["identity_sha256"],
            "method_lock_identity_sha256": METHOD_LOCK_ID,
            "files": [
                {
                    "name": "protocol.json",
                    "bytes": (staging / "protocol.json").stat().st_size,
                    "sha256": sha256_file(staging / "protocol.json"),
                },
                {
                    "name": "source.py",
                    "bytes": Path(__file__).stat().st_size,
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            ],
        }
        with Path(__file__).open("rb") as source, (staging / "source.py").open("wb") as target:
            target.write(source.read())
            target.flush()
            os.fsync(target.fileno())
        # Recompute source record after copying, then seal the envelope.
        manifest["files"][1]["bytes"] = (staging / "source.py").stat().st_size
        manifest["files"][1]["sha256"] = sha256_file(staging / "source.py")
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise
    print(json.dumps({
        "status": "frozen",
        "output": str(output),
        "protocol_identity_sha256": protocol["identity_sha256"],
        "selected_path_count": len(selected),
        "joint_count": len(joint),
        "control_count": len(matched_pairs),
    }, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    freeze(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
