#!/usr/bin/env python3
"""Freeze the endpoint-screen -> event-rich B/C confirmation design.

This locker reads only source files and the already frozen *protocol metadata*
for B/C.  It does not open third-pool images, reviews, labels, feature tables,
candidate scores, embeddings, or trajectories.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v1"
OLD_LOCK = ROOT / "experiments/locks/dit_bad_good_third_pool_protocol_lock_v1"
SELECTOR = ROOT / "experiments/select_dit_event_rich_classes.py"


ROSTER = {
    "articulated_furry_mammals": [
        (160, "Afghan hound"), (170, "Irish wolfhound"), (228, "komondor"),
        (229, "Old English sheepdog"), (231, "collie"), (250, "Siberian husky"),
        (258, "Samoyed"), (281, "tabby"), (283, "Persian cat"), (332, "Angora"),
        (366, "gorilla"), (381, "spider monkey"),
    ],
    "thin_anatomy_wings_legs": [
        (9, "ostrich"), (22, "bald eagle"), (72, "black and gold garden spider"),
        (79, "centipede"), (84, "peacock"), (94, "hummingbird"),
        (127, "white stork"), (130, "flamingo"), (315, "mantis"),
        (319, "dragonfly"), (320, "damselfly"), (321, "admiral"),
    ],
    "multipart_animals": [
        (29, "axolotl"), (43, "frilled lizard"), (47, "African chameleon"),
        (69, "trilobite"), (71, "scorpion"), (118, "Dungeness crab"),
        (121, "king crab"), (122, "American lobster"), (125, "hermit crab"),
        (327, "starfish"), (328, "sea urchin"), (363, "armadillo"),
    ],
    "human_garment_interaction": [
        (411, "apron"), (414, "backpack"), (433, "bathing cap"), (445, "bikini"),
        (459, "brassiere"), (578, "gown"), (610, "jersey"), (614, "kimono"),
        (652, "military uniform"), (981, "ballplayer"), (982, "groom"),
        (983, "scuba diver"),
    ],
    "thin_rigid_topology": [
        (401, "accordion"), (402, "acoustic guitar"), (420, "banjo"),
        (444, "bicycle-built-for-two"), (486, "cello"), (488, "chain"),
        (489, "chainlink fence"), (491, "chain saw"), (545, "electric fan"),
        (559, "folding chair"), (594, "harp"), (671, "mountain bike"),
    ],
    "vehicles_multipart_objects": [
        (403, "aircraft carrier"), (404, "airliner"), (407, "ambulance"),
        (476, "carousel"), (517, "crane"), (537, "dogsled"), (555, "fire engine"),
        (561, "forklift"), (603, "horse cart"), (609, "jeep"), (717, "pickup"),
        (817, "sports car"),
    ],
    "soft_clustered_organic": [
        (107, "jellyfish"), (108, "sea anemone"), (109, "brain coral"),
        (115, "sea slug"), (393, "anemone fish"), (396, "lionfish"),
        (599, "honeycomb"), (927, "trifle"), (933, "cheeseburger"),
        (953, "pineapple"), (959, "carbonara"), (963, "pizza"),
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(value)
    out.pop("identity_sha256", None)
    return out


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite lock: {OUTPUT}")
    old_protocol_path = OLD_LOCK / "third_pool_protocol.json"
    old_protocol = json.loads(old_protocol_path.read_text(encoding="utf-8"))
    old_identity = old_protocol.get("identity_sha256")
    if canonical_sha256(without_identity(old_protocol)) != old_identity:
        raise RuntimeError("upstream third-pool protocol identity mismatch")

    class_roster = [
        {"class_id": class_id, "class_name": name, "stratum": stratum}
        for stratum, members in ROSTER.items()
        for class_id, name in members
    ]
    class_ids = [row["class_id"] for row in class_roster]
    if len(class_roster) != 84 or len(set(class_ids)) != 84:
        raise RuntimeError("screen roster must contain 84 unique classes")
    if set(class_ids) & {207, 602, 795}:
        raise RuntimeError("screen roster must exclude the prior three classes")

    b = old_protocol["candidates"]["B_blur_mean"]
    c = old_protocol["candidates"]["C_c3_low_jump"]
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EVENT_RICH_ENDPOINT_SCREEN_SAMPLING_OR_REVIEW",
        "objective": (
            "Use endpoint-only human labels to prospectively enrich rare, model-relative "
            "failure events across a wider ImageNet class roster, then test the unchanged "
            "internal B/C trajectory candidates on new seeds without score-guided selection."
        ),
        "method_boundary": {
            "internal_method_candidates_only": ["B_blur_mean", "C_c3_low_jump"],
            "Inception_DINO_FID_role": (
                "optional endpoint-only descriptive audit after human labels are locked; never "
                "a class-selection input, candidate, gate, intervention signal, or claimed method"
            ),
            "screen_is_sampling_design_not_a_detector": True,
            "no_third_pool_candidate_feature_or_score_access": True,
        },
        "upstream_candidate_lock": {
            "path": str(old_protocol_path),
            "file_sha256": sha256_file(old_protocol_path),
            "identity_sha256": old_identity,
        },
        "candidates": {
            "B_blur_mean": {
                "feature": b["feature"], "formula": b["formula"],
                "checkpoint_sampling_steps": b["checkpoint_sampling_steps"],
                "orientation": b["raw_orientation"],
                "endpoint": b["primary_endpoint"],
                "auc_gate": 0.75,
            },
            "C_c3_low_jump": {
                "feature": c["feature"], "formula": c["formula"],
                "orientation": c["raw_orientation"],
                "endpoint": c["primary_endpoint"],
                "auc_gate": 0.70,
            },
        },
        "endpoint_screen": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "endpoint_only_no_trace_saved": True,
            "class_roster": class_roster,
            "stratum_count": len(ROSTER),
            "class_count": len(class_roster),
            "discovery_seeds": list(range(1000, 1012)),
            "discovery_samples_per_class": 12,
            "discovery_endpoint_count": 1008,
            "classes_selected_per_candidate": 6,
            "ranking_B": "descending blur_clear_bad, then descending all clear_bad, then frozen roster order",
            "ranking_C": "descending all clear_bad, then descending blur_clear_bad, then frozen roster order",
            "anchor_seeds": list(range(1012, 1036)),
            "anchor_samples_per_selected_union_class": 24,
            "maximum_selected_union_classes": 12,
            "maximum_anchor_endpoint_count": 288,
            "maximum_total_screen_endpoints": 1296,
            "rank_and_anchor_seed_disjoint": True,
        },
        "human_anchor": {
            "three_independent_endpoint_only_reviewers": True,
            "reviewers_blind_to_candidate_hypotheses_scores_embeddings_and_each_other": True,
            "native_resolution_review_required_for_every_suspicious_grid_item": True,
            "rubric": old_protocol["phenotype_contract"],
            "labels_locked_before_class_selection_or_anchor_decision": True,
            "anchor_labels_locked_before_trace_plan": True,
            "external_representation_distances_hidden_from_reviewers_and_selector": True,
        },
        "forecast_rule": {
            "confidence_bound": "one-sided Wilson lower bound at 80% (z=0.8416212335729143)",
            "one_sided_wilson_z": 0.8416212335729143,
            "rationale": (
                "The anchor cohort is independent of discovery ranking.  The bound is a planning "
                "guardrail, not a frequentist guarantee after class screening."
            ),
            "B_blur_mean": {
                "minimum_anchor_events": 6,
                "minimum_event_bearing_classes": 3,
                "minimum_anchor_clean_good": 60,
                "conservative_expected_target": 22.5,
                "final_event_gate": 15,
            },
            "C_c3_low_jump": {
                "minimum_anchor_events": 11,
                "minimum_event_bearing_classes": 3,
                "minimum_anchor_clean_good": 60,
                "conservative_expected_target": 45.0,
                "final_event_gate": 30,
            },
        },
        "confirmation": {
            "calibration_seeds": list(range(1100, 1120)),
            "confirmation_seeds": list(range(1200, 1328)),
            "samples_per_selected_class": 128,
            "maximum_union_classes": 12,
            "maximum_confirmation_trace_rows": 1536,
            "calibration_is_label_free_and_class_specific": True,
            "calibration_samples_per_selected_class": 20,
            "screen_discovery_anchor_calibration_confirmation_seeds_all_disjoint": True,
            "exact_same_model_sampler_cfg_as_screen": True,
            "B_scope": "only the six discovery-selected B-risk classes if the independent B anchor GO rule passes",
            "C_scope": "only the six discovery-selected C-risk classes if the independent C anchor GO rule passes",
            "candidate_columns_physically_separate_before_any_label_join": True,
            "no_candidate_combination_or_refitting": True,
            "primary_statistic": old_protocol["confirmatory_statistics"]["primary_statistic"],
            "permutation": old_protocol["confirmatory_statistics"]["randomization_test"],
            "multiple_testing": (
                "Holm over exactly B and C; an event-gated-off candidate receives p=1 without "
                "opening its score product"
            ),
            "candidate_gates": old_protocol["confirmatory_statistics"]["candidate_gates"],
        },
        "separate_score_label_unlock_gates": {
            "B_blur_mean": {
                "minimum_blur_or_soft_fusion_clear_bad": 15,
                "minimum_event_bearing_classes": 3,
                "if_fail": "set B p=1 and do not open the B score product",
            },
            "C_c3_low_jump": {
                "minimum_total_clear_bad": 30,
                "minimum_event_bearing_classes": 3,
                "if_fail": "set C p=1 and do not open the C score product",
            },
            "labels_must_be_immutable_before_either_gate": True,
            "one_candidate_can_be_evaluated_when_the_other_is_gated_off": True,
        },
        "scope_and_stopping": {
            "confirmatory_claim_scope": (
                "prospectively screen-selected high-risk ImageNet classes under the frozen DiT "
                "sampler; no universal-image or random-ImageNet claim"
            ),
            "stop_without_full_traces": "if both independent anchor GO rules fail",
            "sample_only_GO_candidate_union": True,
            "stop_without_score_join": "separately for B or C when its final event gate fails",
            "no_additional_classes_seeds_or_candidate_changes_after_anchor_plan_is_locked": True,
            "intervention_authority": (
                "only B passing its original statistical gates authorizes a blur/fusion-specific "
                "matched-seed intervention; C never authorizes intervention by itself"
            ),
        },
        "independence_audit": {
            "prior_seed_max": 849,
            "new_seed_min": 1000,
            "prior_classes_excluded_from_screen": [207, 602, 795],
            "third_pool_aggregate_counts_used_only_for_power_motivation": {
                "total_clear_bad": 6, "blur_clear_bad": 4, "trajectory_count": 1800,
            },
            "third_pool_images_labels_scores_features_embeddings_or_traces_opened": False,
            "screen_selection_inputs": ["blind endpoint severity consensus", "blind endpoint phenotype consensus"],
            "forbidden_selection_inputs": ["B score", "C score", "any trajectory feature", "Inception", "DINO", "FID"],
        },
        "selector_source": {
            "path": str(SELECTOR),
            "sha256": sha256_file(SELECTOR),
        },
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "sources").mkdir()
    write_json(OUTPUT / "protocol.json", protocol)
    shutil.copy2(SELECTOR, OUTPUT / "sources" / SELECTOR.name)
    shutil.copy2(Path(__file__), OUTPUT / "sources" / Path(__file__).name)
    files = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "completion.json"}:
            files.append({
                "name": path.relative_to(OUTPUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "dit_event_rich_confirmation_protocol_lock_v1",
        "protocol_identity_sha256": protocol["identity_sha256"],
        "files": files,
    }
    manifest["identity_sha256"] = canonical_sha256(manifest)
    write_json(OUTPUT / "manifest.json", manifest)
    write_json(OUTPUT / "completion.json", {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(OUTPUT / "manifest.json"),
        "protocol_identity_sha256": protocol["identity_sha256"],
        "protocol_file_sha256": sha256_file(OUTPUT / "protocol.json"),
    })
    print(json.dumps({
        "output": str(OUTPUT),
        "protocol_identity_sha256": protocol["identity_sha256"],
        "manifest_identity_sha256": manifest["identity_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
