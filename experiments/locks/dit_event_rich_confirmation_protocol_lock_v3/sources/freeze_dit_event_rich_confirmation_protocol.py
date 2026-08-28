#!/usr/bin/env python3
"""Freeze the endpoint-screen -> event-rich B/C confirmation design.

This locker freezes the B/C design, the revised label-quality contract, and a
small endpoint-only instructional anchor set selected from the completed old
pool.  It never reads a trajectory, candidate feature/score, threshold, alert,
embedding, or external representation.  The result remains a scientific
design/source lock, not an executable sampling or evaluation lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v3"
OLD_LOCK = ROOT / "experiments/locks/dit_bad_good_third_pool_protocol_lock_v1"
SELECTOR = ROOT / "experiments/select_dit_event_rich_classes.py"
LABEL_AUDIT = ROOT / "experiments/audits/dit_bad_good_third_pool_label_reliability_v1"
OLD_REVIEW_PACK = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_third_pool_v1_blind_review_pack"
)


# These are visible instructional anchors, not qualification items and not
# confirmatory labels.  Reasons are frozen so a reviewer learns the intended
# decision boundary rather than merely memorizing a severity number.
ANCHORS = [
    (0, 250, 207, "clean_good", "none", "Sharp coherent dog portrait: attached anatomy, resolved eyes/muzzle and natural fur texture."),
    (1, 250, 602, "clean_good", "none", "Difficult gymnast pose remains anatomically connected to the apparatus with resolved edges and no fusion."),
    (2, 250, 795, "clean_good", "none", "Two skiers and their skis/poles remain separately readable; small scale alone is not a defect."),
    (3, 251, 207, "clean_good", "none", "Full dog body is coherent and adequately sharp for this model's class band."),
    (5, 251, 795, "clean_good", "none", "Small skiers and equipment are coherent; ordinary distance-related detail loss is acceptable."),
    (1560, 770, 207, "mild_or_disputed", "mild_softness", "Sleeping pose and shallow-focus softness are visible, but face/body boundaries remain coherent and the softness is not clearly below the class band."),
    (1620, 790, 207, "mild_or_disputed", "mild_softness", "Portrait has modest low-resolution softness while eyes, muzzle and fur boundary remain readable."),
    (1623, 791, 207, "mild_or_disputed", "mild_rendering", "Oversaturated painterly rendering is imperfect but does not erase anatomy or fuse subject parts."),
    (1746, 832, 207, "mild_or_disputed", "mild_structure", "Body proportions are slightly unusual yet still plausible, attached and not clearly malformed."),
    (1395, 715, 207, "mild_or_disputed", "mild_softness", "Old-photo softness is present, but the dog silhouette, limbs and face remain sufficiently coherent for a boundary/mild call."),
    (201, 317, 207, "clear_bad", "blur_or_soft_fusion", "Strong global smoothing produces a plastic face and erases normal fur/facial detail: recognizable subject does not make this quality-normal."),
    (216, 322, 207, "clear_bad", "blur_or_soft_fusion", "Eyes and head contour are visibly blurred/melted below the ordinary class band."),
    (927, 559, 207, "clear_bad", "blur_or_soft_fusion", "Global blur removes facial and fur structure despite the dog remaining recognizable."),
    (1509, 753, 207, "clear_bad", "blur_or_soft_fusion", "Severe global blur and body/background fusion materially degrade the whole subject."),
    (778, 509, 602, "clear_bad", "blur_or_soft_fusion", "Gymnast and apparatus are grossly stretched and fused; this is a clear fusion failure, not a hard-but-valid pose."),
    (399, 383, 207, "clear_bad", "topology_or_attachment", "Hind limbs merge and attach incorrectly, producing an unmistakable topology/misalignment defect."),
    (715, 488, 602, "clear_bad", "topology_or_attachment", "Gymnast limbs are duplicated/incoherent around the bars."),
    (989, 579, 795, "clear_bad", "topology_or_attachment", "A ski-like object floats detached high above the people and slope."),
    (1329, 693, 207, "clear_bad", "topology_or_attachment", "Dog anatomy is grossly fused/duplicated with incompatible body parts."),
    (1510, 753, 602, "clear_bad", "topology_or_attachment", "Gymnast arms are missing/fused into symmetric blobs rather than valid articulated limbs."),
]


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


def validate_upstream_lock() -> tuple[Path, dict[str, Any]]:
    if not OLD_LOCK.is_dir() or OLD_LOCK.is_symlink():
        raise RuntimeError("upstream third-pool lock must be a real directory")
    protocol_path = OLD_LOCK / "third_pool_protocol.json"
    manifest_path = OLD_LOCK / "manifest.json"
    completion_path = OLD_LOCK / "completion.json"
    old_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    old_identity = old_protocol.get("identity_sha256")
    if canonical_sha256(without_identity(old_protocol)) != old_identity:
        raise RuntimeError("upstream third-pool protocol identity mismatch")
    if (
        canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != old_identity
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != old_identity
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("upstream third-pool lock manifest/completion mismatch")
    return protocol_path, old_protocol


def prepare_instructional_anchors() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not OLD_REVIEW_PACK.is_dir() or OLD_REVIEW_PACK.is_symlink():
        raise RuntimeError("old endpoint-only review pack must be a real directory")
    manifest_path = OLD_REVIEW_PACK / "manifest.json"
    completion_path = OLD_REVIEW_PACK / "completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("old endpoint review-pack receipt mismatch")
    listed = manifest.get("files")
    if not isinstance(listed, list) or not all(isinstance(row, dict) for row in listed):
        raise RuntimeError("old endpoint review-pack manifest is malformed")
    by_name = {row.get("name"): row for row in listed}
    if len(by_name) != len(listed):
        raise RuntimeError("old endpoint review-pack manifest has duplicate names")

    catalog: list[dict[str, Any]] = []
    for ordinal, (sample_index, global_seed, class_id, severity, group, reason) in enumerate(ANCHORS):
        source_relative = f"native/endpoint_{sample_index:04d}.png"
        source = OLD_REVIEW_PACK / source_relative
        record = by_name.get(source_relative)
        if (
            not isinstance(record, dict)
            or not source.is_file()
            or source.is_symlink()
            or record.get("bytes") != source.stat().st_size
            or record.get("sha256") != sha256_file(source)
        ):
            raise RuntimeError(f"instructional anchor source mismatch: {source_relative}")
        expected_class = (207, 602, 795)[sample_index % 3]
        expected_seed = 250 + sample_index // 3
        if class_id != expected_class or global_seed != expected_seed:
            raise RuntimeError("instructional anchor sample/class/seed mapping changed")
        catalog.append({
            "anchor_id": f"qa_{ordinal:02d}",
            "source_sample_index": sample_index,
            "source_global_seed": global_seed,
            "source_class_id": class_id,
            "severity": severity,
            "component_group": group,
            "reason": reason,
            "source_relative_path": source_relative,
            "frozen_relative_path": f"anchors/qa_{ordinal:02d}.png",
            "bytes": source.stat().st_size,
            "sha256": record["sha256"],
        })
    return catalog, {
        "old_review_pack_path": str(OLD_REVIEW_PACK),
        "old_review_pack_manifest_identity_sha256": manifest["identity_sha256"],
        "old_review_pack_manifest_file_sha256": sha256_file(manifest_path),
        "old_review_pack_completion_file_sha256": sha256_file(completion_path),
        "label_reliability_audit_summary_sha256": sha256_file(LABEL_AUDIT / "audit_summary.json"),
        "label_reliability_audit_report_sha256": sha256_file(LABEL_AUDIT / "AUDIT_REPORT.md"),
        "row_level_inputs_used_only_for_visible_instructional_anchors": True,
        "trajectory_candidate_score_embedding_or_external_representation_opened": False,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite lock: {OUTPUT}")
    old_protocol_path, old_protocol = validate_upstream_lock()
    old_identity = old_protocol.get("identity_sha256")

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
    anchor_catalog, anchor_lineage = prepare_instructional_anchors()
    anchor_catalog_artifact: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_VISIBLE_INSTRUCTIONAL_ANCHORS_NOT_QUALIFICATION_GOLD",
        "anchors": anchor_catalog,
    }
    anchor_catalog_artifact["identity_sha256"] = canonical_sha256(anchor_catalog_artifact)
    protocol: dict[str, Any] = {
        "schema_version": 3,
        "status": "FROZEN_BEFORE_REVIEWER_QUALIFICATION_OR_EVENT_RICH_SCREEN",
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
            "no_row_level_third_pool_input_to_new_class_selection_or_forecast": True,
            "historical_B_C_selection_is_prior_data_adaptive": True,
            "endpoint_labels_and_instructional_anchors_are_external_evaluation_only": True,
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
            "batch_rng_contract": {
                "rng_unit": ["global_seed", "class_id"],
                "domain": "eqvae.dit.event-rich.endpoint.v1",
                "seed_derivation": (
                    "payload=UTF8(domain)+0x00+ASCII(str(global_seed))+0x00+"
                    "ASCII(str(class_id)); digest=SHA256(payload); pair_seed="
                    "int.from_bytes(digest[0:8],byteorder='big',signed=False) & 0x7fffffffffffffff"
                ),
                "pair_seed_range": "unsigned 63-bit integer [0,2^63-1]",
                "classes_per_invocation": 1,
                "manual_seed_timing": (
                    "call torch.manual_seed(pair_seed) after the frozen DiT model and VAE are "
                    "fully loaded, immediately before drawing the singleton initial latent"
                ),
                "cfg_batch_contract": (
                    "B=1 initial latent is duplicated into the ordered 2B conditional/null batch"
                ),
                "transition_rng_contract": (
                    "every one of the 250 ancestral DDPM steps draws the full ordered 2B "
                    "randn_like tensor; the t=0 draw is consumed before it is multiplied by zero"
                ),
                "same_global_seed_classes_share_initial_or_transition_innovation": False,
                "batch_order_shard_and_resume_invariant": True,
                "same_rule_for_discovery_anchor_calibration_and_confirmation": True,
                "compatibility_scope": (
                    "same frozen model, 250-step sampler and full-2B draw semantics as the old "
                    "pool; no claim of reproducing its three-class batch-correlated realizations"
                ),
            },
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
        "label_system": {
            "motivation_from_completed_label_audit": {
                "third_pool_binary_fleiss_kappa": 0.1471879593843114,
                "raw_majority_clear_bad": 29,
                "single_adjudicator_downgraded": 23,
                "all_downgrades_class_id": 207,
                "independent_single_auditor_obvious_bad_misses_at_least": 7,
                "conclusion": (
                    "old final labels are high-precision but too inconsistent and conservative "
                    "to calibrate event prevalence or justify the next screen's label system"
                ),
            },
            "instructional_anchor_catalog": anchor_catalog,
            "instructional_anchor_catalog_artifact_identity_sha256": anchor_catalog_artifact[
                "identity_sha256"
            ],
            "instructional_anchor_lineage": anchor_lineage,
            "instructional_anchor_role": (
                "visible training examples with frozen written reasons; excluded from reviewer "
                "qualification statistics, screen ranking, anchor GO and confirmation testing"
            ),
            "core_rubric": {
                "recognizable_subject_does_not_imply_quality_normal": True,
                "two_question_rule": [
                    "is a visible defect present?",
                    "is it materially below the frozen model/class comparison band?",
                ],
                "severity_0": "ordinary/clean for the frozen model and class; no material defect",
                "severity_1": "visible imperfection or genuine boundary ambiguity, not clearly material",
                "severity_2": (
                    "clear material defect below the class comparison band; obvious global soft "
                    "blur, lost facial/object detail, fusion, melting or misattachment qualifies "
                    "even when the subject is recognizable"
                ),
                "severity_3": "severe/catastrophic version of a severity-2 defect",
                "required_for_severity_2_or_3": (
                    "at least one frozen component plus a one-sentence localization/reason"
                ),
                "rubric": old_protocol["phenotype_contract"],
            },
            "reviewer_qualification": {
                "must_pass_before_any_formal_screen_image_is_released": True,
                "visible_instructional_anchors_are_not_qualification_items": True,
                "hidden_set_size": 60,
                "hidden_set_balance": {
                    "clean_good": 15,
                    "mild_or_disputed": 15,
                    "clear_blur_or_soft_fusion": 15,
                    "clear_topology_or_attachment": 15,
                },
                "hidden_set_is_cross_class_endpoint_only_and_score_blind": True,
                "hidden_gold_creation": (
                    "two independent expert curators label every item against frozen class-band "
                    "references; disagreements go to a third independent resolver; all images, "
                    "gold labels and written reasons are hashed before reviewer qualification"
                ),
                "reviewer_blinding": (
                    "reviewers do not see gold labels, other reviews, metrics, candidate "
                    "hypotheses, trajectories, embeddings or external representation distances"
                ),
                "binary_definition": "clear_bad iff severity >=2; clean/mild are binary negative",
                "positive_agreement_formula": "2*n11/(2*n11+n10+n01)",
                "binary_cohen_kappa_formula": "(observed_agreement-expected_chance_agreement)/(1-expected_chance_agreement)",
                "every_pair_minimum_positive_agreement": 0.60,
                "every_pair_minimum_binary_cohen_kappa": 0.50,
                "individual_minimum_clear_bad_recall_against_hidden_gold": 0.80,
                "individual_minimum_non_clear_bad_specificity_against_hidden_gold": 0.80,
                "if_any_gate_fails": (
                    "STOP; do not release formal screen images. Replace/retrain the failed "
                    "reviewer and qualify the complete proposed panel on a new, disjoint, "
                    "pre-hashed reserve form; the failed form cannot be reused"
                ),
                "qualification_result_must_be_immutable_before_screen_pack": True,
            },
            "production_review": {
                "three_independent_endpoint_only_reviewers": True,
                "reviewers_blind_to_candidate_hypotheses_scores_embeddings_and_each_other": True,
                "native_resolution_review_required_for_every_suspicious_grid_item": True,
                "class_comparison_band_visible_but_no_future_or_metric_selected_examples": True,
                "discovery_labels_locked_before_class_selection": True,
                "anchor_labels_locked_before_trace_plan": True,
                "confirmation_labels_locked_before_any_score_product_is_opened": True,
                "report_pairwise_positive_agreement_binary_kappa_prevalence_and_confusion_counts_per_phase": True,
            },
            "dual_adjudication_and_miss_audit": {
                "two_independent_qualified_adjudicators": True,
                "single_adjudicator_can_change_final_severity": False,
                "adjudicators_blind_to_candidate_scores_features_embeddings_and_external_representations": True,
                "audit_pack_includes": [
                    "every raw 2-of-3 clear-bad item",
                    "every item with any reviewer severity >=2 but no raw clear-bad majority",
                    "a frozen simple-random equal-count decoy sample from items with no severity >=2 vote",
                ],
                "audit_pack_selection_uses_model_score_or_embedding": False,
                "adjudicators_blind_to_trigger_stratum_reviewer_identity_vote_count_and_each_other": True,
                "unanimous_three_reviewer_clear_bad_is_never_downgradable": True,
                "raw_two_of_three_clear_bad_downgrade_rule": (
                    "downgrade only when both adjudicators independently choose mild/non-bad; "
                    "otherwise retain clear_bad"
                ),
                "raw_nonmajority_promotion_rule": (
                    "promote to clear_bad only when both adjudicators independently choose "
                    "clear_bad; otherwise preserve the raw clean/mild status"
                ),
                "decoy_discovery_rule": (
                    "a decoy receives the same unanimous-two-adjudicator promotion rule; report "
                    "its blind false-negative audit rate separately"
                ),
                "all_changes_require_component_and_written_reason_from_both_adjudicators": True,
                "per_phase_counts_before_and_after_each_rule_must_be_published": True,
            },
            "external_representation_distances_hidden_from_every_label_role": True,
        },
        "human_anchor": {
            "status": "SUPERSEDED_COMPATIBILITY_ALIAS",
            "authoritative_contract": "label_system",
        },
        "forecast_rule": {
            "confidence_bound": "one-sided Wilson lower bound at 80% (z=0.8416212335729143)",
            "one_sided_wilson_z": 0.8416212335729143,
            "rationale": (
                "The anchor cohort is independent of discovery ranking.  The bound is a planning "
                "guardrail conditional on the discovery-selected class set, not a guarantee of "
                "event count or power for the AUC hypothesis."
            ),
            "iid_binomial_planning_assumption": (
                "The reported tail probabilities treat confirmation endpoints as iid Bernoulli "
                "within the equally weighted selected-class mixture; deterministic seeded model "
                "runs need not satisfy this assumption."
            ),
            "label_uncertainty_caveat": (
                "The third pool's final 6/1800 and 4/1800 subtype counts are excluded from "
                "prevalence estimation because the completed reliability audit found material "
                "reviewer disagreement and at least seven obvious adjudication misses."
            ),
            "formal_auc_power_analysis_completed": False,
            "sample_size_status": (
                "fixed cost cap and event-count planning design only; qualification, dual "
                "adjudication and seed-block dependence prevent claiming guaranteed AUC power"
            ),
            "B_blur_mean": {
                "minimum_anchor_events": 6,
                "minimum_event_bearing_classes": 3,
                "minimum_anchor_clean_good": 60,
                "conservative_expected_target": 22.5,
                "final_event_gate": 15,
                "tail_probability_at_boundary_under_iid_binomial_model": 0.968693754019694,
            },
            "C_c3_low_jump": {
                "minimum_anchor_events": 11,
                "minimum_event_bearing_classes": 3,
                "minimum_anchor_clean_good": 60,
                "conservative_expected_target": 45.0,
                "final_event_gate": 30,
                "tail_probability_at_boundary_under_iid_binomial_model": 0.9958265266210399,
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
            "permutation": {
                "alternative": "frozen-direction class-matched AUC greater than chance",
                "draws": 100000,
                "method": (
                    "for each draw, apply one common random permutation of the 128 complete "
                    "global-seed label/phenotype blocks; within each candidate, retain only its "
                    "ordered six-class scope, thereby preserving every selected class's label "
                    "counts and available cross-class label dependence"
                ),
                "p_value": "(1 + exceedances)/(1 + draws)",
                "rng": "numpy.default_rng(PCG64(seed=2026082801))",
                "unit": "one intact ordered active-union selected-class block per global seed",
                "same_seed_permutation_used_for_B_and_C": True,
            },
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
                "minimum_comparable_classes_with_event_and_clean_good": 3,
                "minimum_clean_good": 60,
                "if_fail": "set B p=1 and do not open the B score product",
            },
            "C_c3_low_jump": {
                "minimum_total_clear_bad": 30,
                "minimum_event_bearing_classes": 3,
                "minimum_comparable_classes_with_event_and_clean_good": 3,
                "minimum_clean_good": 60,
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
            "third_pool_aggregate_counts_recorded_for_failure_diagnosis_not_power": {
                "total_clear_bad": 6, "blur_clear_bad": 4, "trajectory_count": 1800,
            },
            "third_pool_final_counts_eligible_for_prevalence_or_power_estimation": False,
            "historical_context_is_not_an_untouched_replication": (
                "B/C and the failure phenotypes were chosen after earlier pools; only the new "
                "discovery, anchor, and confirmation rows are prospectively independent"
            ),
            "row_level_third_pool_inputs_read_by_this_locker": (
                "only the 20 endpoint PNGs explicitly frozen as visible instructional anchors; "
                "never trajectories, model scores/features, embeddings or future selection rows"
            ),
            "screen_selection_inputs": ["blind endpoint severity consensus", "blind endpoint phenotype consensus"],
            "forbidden_selection_inputs": ["B score", "C score", "any trajectory feature", "Inception", "DINO", "FID"],
        },
        "execution_readiness": {
            "ready_for_real_sampling": False,
            "reason": (
                "Visible anchors are frozen, but they still require two-expert ratification and "
                "the hidden qualification/reserve forms, qualification evaluator, pair-keyed "
                "singleton sampler/source lock, production three-reviewer plus dual-adjudicator "
                "pipeline, dynamic selected-union trace runner, physically separate B/C score "
                "extractors, and two-stage fail-closed evaluator do not yet exist."
            ),
            "must_exist_and_be_frozen_before_real_sampling": [
                "two-expert ratification of every visible anchor and hidden qualification gold set",
                "qualification and disjoint reserve forms plus fail-closed qualification evaluator",
                "pair-keyed singleton endpoint sampler with the exact RNG and asset contract",
                "discovery, anchor and confirmation blind-review/dual-adjudication/consensus locks",
                "dynamic trace runner bound to the immutable anchor plan",
                "separate B-only and C-only score products",
                "stage-A label-only gate evaluator and stage-B score-opening evaluator",
            ],
        },
        "selector_source": {
            "path": str(SELECTOR),
            "sha256": sha256_file(SELECTOR),
        },
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "sources").mkdir()
    (OUTPUT / "anchors").mkdir()
    (OUTPUT / "label_audit").mkdir()
    write_json(OUTPUT / "protocol.json", protocol)
    write_json(OUTPUT / "instructional_anchor_catalog.json", anchor_catalog_artifact)
    shutil.copy2(SELECTOR, OUTPUT / "sources" / SELECTOR.name)
    shutil.copy2(Path(__file__), OUTPUT / "sources" / Path(__file__).name)
    shutil.copy2(LABEL_AUDIT / "AUDIT_REPORT.md", OUTPUT / "label_audit/AUDIT_REPORT.md")
    shutil.copy2(LABEL_AUDIT / "audit_summary.json", OUTPUT / "label_audit/audit_summary.json")
    for row in anchor_catalog:
        shutil.copy2(
            OLD_REVIEW_PACK / row["source_relative_path"],
            OUTPUT / row["frozen_relative_path"],
        )
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
        "experiment": "dit_event_rich_confirmation_protocol_lock_v3",
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
