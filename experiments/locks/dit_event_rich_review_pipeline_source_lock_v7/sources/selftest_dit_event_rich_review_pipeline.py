#!/usr/bin/env python3
"""Synthetic, non-scientific fail-closed tests for the review infrastructure."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

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
    passed.append("frozen_protocol_v3_identity")

    for poison in (
        "B_score",
        "C_metric",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_tests(source_lock=args.source_lock), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
