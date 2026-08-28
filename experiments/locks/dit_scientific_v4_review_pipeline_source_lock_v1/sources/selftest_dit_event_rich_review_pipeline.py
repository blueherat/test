#!/usr/bin/env python3
"""Synthetic, non-scientific fail-closed tests for the review infrastructure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image

import dit_event_rich_review_contract as contract
import run_dit_event_rich_blind_label_pipeline as pipeline


def must_fail(name: str, function: Callable[[], Any]) -> str:
    try:
        function()
    except RuntimeError:
        return name
    raise AssertionError(f"expected fail-closed rejection: {name}")


def run_tests(*, source_lock: Path | None = None) -> dict[str, Any]:
    passed: list[str] = []
    protocol, _ = contract.validate_event_protocol_lock()
    assert protocol["identity_sha256"] == contract.EVENT_PROTOCOL_IDENTITY
    passed.append("frozen_corrected_scientific_v4_identity")

    for poison in (
        "B_score",
        "E_metric",
        "trajectory_feature",
        "DINO_embedding",
        "FID",
        "class_rank",
        "label",
    ):
        passed.append(
            must_fail(
                f"poison_column_{poison}",
                lambda poison=poison: contract.reject_forbidden_columns(
                    ("class_id", poison)
                ),
            )
        )

    rows = [{"id": "a"}, {"id": "b"}]
    contract.validate_unique_axis(rows, ("id",), (("a",), ("b",)))
    passed.append("exact_axis_accepts_complete_unique_order")
    passed.append(
        must_fail(
            "duplicate_row_rejected",
            lambda: contract.validate_unique_axis(
                [{"id": "a"}, {"id": "a"}], ("id",)
            ),
        )
    )
    passed.append(
        must_fail(
            "missing_row_rejected",
            lambda: contract.validate_unique_axis(
                [{"id": "a"}], ("id",), (("a",), ("b",))
            ),
        )
    )
    passed.append(
        must_fail(
            "extra_row_rejected",
            lambda: contract.validate_unique_axis(
                [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                ("id",),
                (("a",), ("b",)),
            ),
        )
    )

    with tempfile.TemporaryDirectory(prefix="event-review-selftest-") as raw:
        root = Path(raw)
        good = root / "good.csv"
        contract.write_csv(good, ("id",), ({"id": "a"},))
        assert contract.read_csv_exact(good, ("id",)) == [{"id": "a"}]
        passed.append("strict_csv_schema_accepts_exact")
        poisoned = root / "poisoned.csv"
        with poisoned.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "B_score"])
            writer.writerow(["a", "1.0"])
        passed.append(
            must_fail(
                "strict_csv_extra_poison_column_rejected",
                lambda: contract.read_csv_exact(poisoned, ("id",)),
            )
        )

    gold = [0] * 30 + [1] * 30
    perfect = {slot: list(gold) for slot in contract.ROLE_SLOTS}
    metrics = contract.qualification_metrics(gold, perfect)
    assert metrics["panel_passed"] is True and len(metrics["pairs"]) == 10
    passed.append("five_role_all_pairs_qualification_pass")
    serialized_token_map = {
        slot: f"token_{slot}" for slot in sorted(contract.ROLE_SLOTS)
    }
    assert tuple(serialized_token_map) != contract.ROLE_SLOTS
    assert contract.qualified_role_tokens_valid(serialized_token_map)
    passed.append("qualified_role_map_is_json_order_invariant")
    failed = {slot: list(gold) for slot in contract.ROLE_SLOTS}
    failed["reviewer_2"] = [0] * 60
    metrics = contract.qualification_metrics(gold, failed)
    assert metrics["panel_passed"] is False
    assert metrics["individuals"]["reviewer_2"]["clear_bad_recall"] == 0.0
    passed.append("qualification_any_individual_or_pair_failure_stops")

    # Consensus invariants, including attempted unilateral changes.
    assert pipeline.consensus_policy(
        3, 0, ("mild_or_not_clear_bad", "mild_or_not_clear_bad")
    ) == ("clear_bad", "unanimous_3of3_never_downgradable", False)
    passed.append("three_of_three_never_downgradable")
    assert pipeline.consensus_policy(
        2, 1, ("mild_or_not_clear_bad", "clear_bad")
    )[0] == "clear_bad"
    passed.append("single_adjudicator_cannot_downgrade_two_of_three")
    assert pipeline.consensus_policy(
        2, 1, ("mild_or_not_clear_bad", "mild_or_not_clear_bad")
    )[0] == "mild_or_disputed"
    passed.append("dual_unanimous_can_downgrade_two_of_three")
    assert pipeline.consensus_policy(1, 2, ("clear_bad", "mild_or_not_clear_bad"))[0] == "clean_good"
    passed.append("single_adjudicator_cannot_promote_minority_positive")
    assert pipeline.consensus_policy(1, 2, ("clear_bad", "clear_bad"))[0] == "clear_bad"
    passed.append("dual_unanimous_can_promote_minority_positive")
    assert pipeline.consensus_policy(0, 3, ("clear_bad", "clear_bad"))[0] == "clear_bad"
    passed.append("decoy_uses_same_dual_promotion_rule")

    passed.append(
        must_fail(
            "severity_two_without_component_rejected",
            lambda: contract.validate_severity_row(
                {"severity": "2", "components": "none", "localization_reason": "clear localized defect"}
            ),
        )
    )
    if source_lock is not None:
        manifest, frozen_contract = contract.validate_source_lock(source_lock)
        assert frozen_contract["real_expert_or_reviewer_results_present"] is False
        assert manifest["identity"]["ready_for_real_sampling"] is False
        passed.append("frozen_source_lock_valid_and_contains_no_results")
    return {
        "schema_version": 1,
        "status": "PASS",
        "synthetic_test_only": True,
        "scientific_labels_or_results_created": False,
        "test_count": len(passed),
        "tests": passed,
    }


def _fill_review_form(
    pack: Path,
    slot: str,
    clear_keys: set[tuple[str, str]],
    output: Path,
) -> None:
    mapping_fields = (
        "blind_id",
        "class_id",
        "global_seed",
        "class_name",
        "source_image_path",
        "image_sha256",
        "image_pixel_sha256",
    )
    mapping = contract.read_csv_exact(pack / f"custody/{slot}_mapping.csv", mapping_fields)
    template = contract.read_csv_exact(
        pack / f"delivery/{slot}/response_template.csv",
        contract.REVIEW_RESPONSE_FIELDS,
    )
    key_by_id = {
        row["blind_id"]: (row["class_id"], row["global_seed"]) for row in mapping
    }
    rows = []
    for row in template:
        clear = key_by_id[row["blind_id"]] in clear_keys
        rows.append(
            {
                **row,
                "severity": "2" if clear else "0",
                "components": "global_blur" if clear else "none",
                "localization_reason": (
                    "synthetic localized blur defect for fail-closed integration testing"
                    if clear
                    else ""
                ),
                "independence_attestation": pipeline.ATTESTATION,
            }
        )
    contract.write_csv(output, contract.REVIEW_RESPONSE_FIELDS, rows)


def _fill_adjudication_form(
    pack: Path,
    slot: str,
    *,
    unanimous_key: tuple[str, str],
    raw_two_key: tuple[str, str],
    minority_key: tuple[str, str],
    output: Path,
) -> tuple[str, str] | None:
    mapping = contract.read_csv_exact(
        pack / f"custody/{slot}_mapping.csv",
        pipeline.ADJUDICATION_MAPPING_FIELDS,
        allow_empty=True,
    )
    template = contract.read_csv_exact(
        pack / f"delivery/{slot}/response_template.csv",
        contract.ADJUDICATION_RESPONSE_FIELDS,
        allow_empty=not mapping,
    )
    key_by_id = {
        row["adjudication_id"]: (row["class_id"], row["global_seed"])
        for row in mapping
    }
    decoys = [
        (row["class_id"], row["global_seed"])
        for row in mapping
        if row["selection_kind"] == "zero_positive_random_decoy"
    ]
    promoted_decoy = sorted(decoys, key=lambda key: (int(key[1]), int(key[0])))[0] if decoys else None
    clear_keys = {minority_key}
    if promoted_decoy is not None:
        clear_keys.add(promoted_decoy)
    # Both adjudicators deliberately request a downgrade for unanimous_key and
    # raw_two_key.  Consensus must ignore the former and apply the latter.
    assert unanimous_key in key_by_id.values()
    assert raw_two_key in key_by_id.values()
    rows = []
    for row in template:
        clear = key_by_id[row["adjudication_id"]] in clear_keys
        rows.append(
            {
                **row,
                "decision": "clear_bad" if clear else "mild_or_not_clear_bad",
                "components": "global_blur" if clear else "none",
                "localization_reason": (
                    "synthetic dual adjudication reason for clear blur promotion"
                    if clear
                    else "synthetic dual adjudication reason for non-clear decision"
                ),
                "independence_attestation": pipeline.ATTESTATION,
            }
        )
    contract.write_csv(output, contract.ADJUDICATION_RESPONSE_FIELDS, rows)
    return promoted_decoy


def run_synthetic_end_to_end(source_lock: Path) -> dict[str, Any]:
    """Exercise the complete production chain with ephemeral synthetic pixels.

    Nothing produced here is a scientific label, gold form, or reusable review.
    All artifacts live under a temporary directory and are destroyed on return.
    """

    passed: list[str] = []
    source_lock = source_lock.expanduser().absolute()
    contract.validate_source_lock(source_lock)
    contract.REVIEW_SOURCE_LOCK = source_lock
    protocol, _ = contract.validate_event_protocol_lock()
    roster = protocol["endpoint_screen"]["class_roster"]
    classes = tuple(int(row["class_id"]) for row in roster)
    seeds = tuple(int(seed) for seed in protocol["endpoint_screen"]["discovery_seeds"])
    expected_axis = tuple((class_id, seed) for seed in seeds for class_id in classes)

    with tempfile.TemporaryDirectory(prefix="scientific-v4-review-e2e-") as raw:
        root = Path(raw)
        # Exact-tree and immutable-member tamper tests operate on disposable copies.
        extra_tree = root / "source_lock_extra"
        shutil.copytree(source_lock, extra_tree)
        (extra_tree / "unexpected.bin").write_bytes(b"tamper")
        passed.append(
            must_fail(
                "review_source_exact_tree_extra_member_rejected",
                lambda: contract.validate_source_lock(extra_tree),
            )
        )
        changed_tree = root / "source_lock_changed"
        shutil.copytree(source_lock, changed_tree)
        with (changed_tree / "sources/run_dit_event_rich_blind_label_pipeline.py").open(
            "ab"
        ) as handle:
            handle.write(b"\n# synthetic tamper\n")
        passed.append(
            must_fail(
                "review_source_snapshot_mutation_rejected",
                lambda: contract.validate_source_lock(changed_tree),
            )
        )

        phase_plan = {
            "schema_version": 1,
            "status": "FROZEN_ENDPOINT_PHASE_PLAN",
            "phase": "discovery",
            "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
            "class_ids_ordered": list(classes),
            "global_seeds_ordered": list(seeds),
            "upstream_plan_identity_sha256": None,
            "labels_locked_before_plan": False,
            "candidate_scores_features_trajectories_embeddings_or_ranks_used": False,
        }
        phase_plan["identity_sha256"] = contract.canonical_sha256(phase_plan)
        phase_plan_path = root / "phase_plan.json"
        contract.write_json(phase_plan_path, phase_plan)

        image_root = root / "endpoints"
        image_root.mkdir()
        class_names = {int(row["class_id"]): row["class_name"] for row in roster}
        endpoint_rows = []
        pair_rows = []
        for index, (class_id, seed) in enumerate(expected_axis):
            color_value = index + 1
            color = (
                color_value & 255,
                (color_value >> 8) & 255,
                (color_value >> 16) & 255,
            )
            image_path = image_root / f"seed{seed:04d}_class{class_id:04d}.png"
            Image.new("RGB", (256, 256), color).save(image_path)
            image_sha = contract.sha256_file(image_path)
            pixel_sha = hashlib.sha256(Image.new("RGB", (256, 256), color).tobytes()).hexdigest()
            pair_identity = hashlib.sha256(
                f"synthetic-pair\0{seed}\0{class_id}".encode("ascii")
            ).hexdigest()
            pair_manifest = hashlib.sha256(
                f"synthetic-pair-manifest\0{seed}\0{class_id}".encode("ascii")
            ).hexdigest()
            endpoint_rows.append(
                {
                    "class_id": class_id,
                    "global_seed": seed,
                    "class_name": class_names[class_id],
                    "image_path": str(image_path),
                    "image_sha256": image_sha,
                    "image_pixel_sha256": pixel_sha,
                    "width": 256,
                    "height": 256,
                    "mode": "RGB",
                    "source_pair_identity_sha256": pair_identity,
                    "source_manifest_sha256": pair_manifest,
                }
            )
            pair_rows.append(
                {
                    "global_seed": seed,
                    "class_id": class_id,
                    "identity_sha256": pair_identity,
                    "manifest_sha256": pair_manifest,
                    "endpoint_sha256": image_sha,
                    "endpoint_pixel_sha256": pixel_sha,
                }
            )
        endpoint_index = root / "endpoint_index.csv"
        contract.write_csv(endpoint_index, pipeline.COHORT_INDEX_FIELDS, endpoint_rows)
        source_pool = root / "source_pool"
        source_pool.mkdir()
        pool_manifest = {
            "schema_version": 1,
            "status": "complete",
            "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
            "pair_outputs": pair_rows,
            "endpoint_only": True,
            "synthetic_selftest_only": True,
        }
        pool_manifest["identity_sha256"] = contract.canonical_sha256(pool_manifest)
        contract.write_json(source_pool / "pool_manifest.json", pool_manifest)
        source_receipt = {
            "schema_version": 1,
            "status": "complete",
            "phase": "discovery",
            "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
            "phase_plan_identity_sha256": phase_plan["identity_sha256"],
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "cfg_scale": 4.0,
            "endpoint_count": len(expected_axis),
            "endpoint_only_review_payload": True,
            "source_artifact_path": str(source_pool),
            "source_artifact_identity_sha256": pool_manifest["identity_sha256"],
            "source_manifest_sha256": contract.sha256_file(source_pool / "pool_manifest.json"),
            "labels_reviews_metrics_features_embeddings_or_scores_opened_for_sampling": False,
        }
        source_receipt_path = root / "source_receipt.json"
        contract.write_json(source_receipt_path, source_receipt)

        poisoned_index = root / "poisoned_endpoint_index.csv"
        with poisoned_index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[*pipeline.COHORT_INDEX_FIELDS, "DINO_embedding"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow({**endpoint_rows[0], "DINO_embedding": "forbidden"})
        passed.append(
            must_fail(
                "external_representation_column_rejected_at_cohort_boundary",
                lambda: pipeline.lock_cohort(
                    argparse.Namespace(
                        phase="discovery",
                        phase_plan=phase_plan_path,
                        endpoint_index=poisoned_index,
                        source_receipt=source_receipt_path,
                        output=root / "poisoned_cohort",
                    )
                ),
            )
        )
        changed_rows = [dict(row) for row in endpoint_rows]
        changed_rows[0]["source_manifest_sha256"] = "f" * 64
        pair_poison = root / "pair_poison.csv"
        contract.write_csv(pair_poison, pipeline.COHORT_INDEX_FIELDS, changed_rows)
        passed.append(
            must_fail(
                "source_pair_manifest_reconciliation_tamper_rejected",
                lambda: pipeline.lock_cohort(
                    argparse.Namespace(
                        phase="discovery",
                        phase_plan=phase_plan_path,
                        endpoint_index=pair_poison,
                        source_receipt=source_receipt_path,
                        output=root / "pair_poison_cohort",
                    )
                ),
            )
        )

        cohort = pipeline.lock_cohort(
            argparse.Namespace(
                phase="discovery",
                phase_plan=phase_plan_path,
                endpoint_index=endpoint_index,
                source_receipt=source_receipt_path,
                output=root / "cohort",
            )
        )
        _, cohort_rows = pipeline.validate_cohort(cohort, "discovery")
        assert [(row["class_id"], row["global_seed"]) for row in cohort_rows] == [
            (str(class_id), str(seed)) for class_id, seed in expected_axis
        ]
        passed.append("global_seed_major_class_minor_axis_round_trip")

        role_tokens = {slot: f"synthetic_token_{slot}" for slot in contract.ROLE_SLOTS}
        qualification = contract.publish_artifact(
            root / "qualification",
            identity={
                "schema_version": 1,
                "artifact_kind": "EVENT_RICH_PANEL_QUALIFICATION_EVALUATION_V1",
                "status": "PASS_FORMAL_SCREEN_RELEASE_AUTHORIZED",
                "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
                "panel_passed": True,
                "formal_screen_release_authorized": True,
                "qualified_role_tokens": role_tokens,
                "synthetic_selftest_only": True,
            },
            builder=lambda output: None,
        )
        review_pack = pipeline.build_review_pack(
            argparse.Namespace(
                phase="discovery",
                cohort_lock=cohort,
                qualification_evaluation=qualification,
                output=root / "review_pack",
            )
        )
        unanimous_key = (str(expected_axis[0][0]), str(expected_axis[0][1]))
        raw_two_key = (str(expected_axis[1][0]), str(expected_axis[1][1]))
        minority_key = (str(expected_axis[2][0]), str(expected_axis[2][1]))
        clear_by_slot = {
            "reviewer_1": {unanimous_key, raw_two_key},
            "reviewer_2": {unanimous_key, raw_two_key},
            "reviewer_3": {unanimous_key, minority_key},
        }
        review_locks: dict[str, Path] = {}
        for slot in contract.REVIEWER_SLOTS:
            form = root / f"{slot}_completed.csv"
            _fill_review_form(review_pack, slot, clear_by_slot[slot], form)
            review_locks[slot] = pipeline.lock_review(
                argparse.Namespace(
                    phase="discovery",
                    review_pack=review_pack,
                    slot=slot,
                    completed_form=form,
                    output=root / f"{slot}_lock",
                )
            )
        adjudication_pack = pipeline.build_adjudication_pack(
            argparse.Namespace(
                phase="discovery",
                review_pack=review_pack,
                cohort_lock=cohort,
                qualification_evaluation=qualification,
                output=root / "adjudication_pack",
                **review_locks,
            )
        )
        adjudication_locks: dict[str, Path] = {}
        promoted_decoys = []
        for slot in contract.ADJUDICATOR_SLOTS:
            form = root / f"{slot}_completed.csv"
            promoted_decoys.append(
                _fill_adjudication_form(
                    adjudication_pack,
                    slot,
                    unanimous_key=unanimous_key,
                    raw_two_key=raw_two_key,
                    minority_key=minority_key,
                    output=form,
                )
            )
            adjudication_locks[slot] = pipeline.lock_adjudication(
                argparse.Namespace(
                    phase="discovery",
                    adjudication_pack=adjudication_pack,
                    slot=slot,
                    completed_form=form,
                    output=root / f"{slot}_lock",
                )
            )
        assert promoted_decoys[0] == promoted_decoys[1] and promoted_decoys[0] is not None
        consensus = pipeline.lock_consensus(
            argparse.Namespace(
                phase="discovery",
                review_pack=review_pack,
                adjudication_pack=adjudication_pack,
                output=root / "consensus",
                **review_locks,
                **adjudication_locks,
            )
        )
        rows = contract.read_csv_exact(consensus / "consensus_rows.csv", pipeline.CONSENSUS_FIELDS)
        by_key = {(row["class_id"], row["global_seed"]): row for row in rows}
        assert by_key[unanimous_key]["consensus_rule"] == "unanimous_3of3_never_downgradable"
        assert by_key[raw_two_key]["consensus_rule"] == "raw_2of3_downgraded_by_two_unanimous_adjudicators"
        assert by_key[minority_key]["consensus_rule"] == "raw_nonmajority_promoted_by_two_unanimous_adjudicators"
        assert by_key[promoted_decoys[0]]["consensus_rule"] == "raw_nonmajority_promoted_by_two_unanimous_adjudicators"
        evaluation_rows = contract.read_csv_exact(
            consensus / "evaluation_labels.csv", pipeline.EVALUATION_LABEL_FIELDS
        )
        assert len(evaluation_rows) == len(expected_axis)
        assert not any(
            any(fragment in field.lower() for fragment in ("score", "embedding", "fid", "dino", "inception"))
            for field in pipeline.EVALUATION_LABEL_FIELDS
        )
        passed.extend(
            [
                "synthetic_cohort_review_adjudication_consensus_end_to_end",
                "three_reviewer_two_adjudicator_policy_end_to_end",
                "union_positive_equal_decoy_and_dual_promotion_end_to_end",
                "external_evaluation_export_contains_no_method_or_representation_values",
            ]
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "synthetic_test_only": True,
        "ephemeral_outputs_destroyed": True,
        "scientific_gold_labels_reviews_or_results_created": False,
        "test_count": len(passed),
        "tests": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--full-end-to-end", action="store_true")
    args = parser.parse_args()
    if args.full_end_to_end:
        if args.source_lock is None:
            parser.error("--full-end-to-end requires --source-lock")
        result = run_synthetic_end_to_end(args.source_lock)
    else:
        result = run_tests(source_lock=args.source_lock)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
