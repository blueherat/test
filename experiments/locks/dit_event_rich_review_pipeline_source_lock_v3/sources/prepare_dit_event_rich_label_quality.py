#!/usr/bin/env python3
"""Freeze anchor ratification, hidden gold forms, and panel qualification.

No expert or reviewer result is fabricated by this program.  It only accepts
completed external forms, validates their exact axes and schemas, and emits an
immutable result.  Hidden-gold reviewer delivery lives in the
``reviewer_release`` subdirectory; curators' gold remains in ``custody`` and
must never be distributed to the candidate panel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

import dit_event_rich_review_contract as contract


RUNNER = "prepare_dit_event_rich_label_quality"
ATTESTATION = "I independently reviewed endpoint pixels only under the frozen rubric"
HIDDEN_BALANCE = {
    "clean_good": 15,
    "mild_or_disputed": 15,
    "clear_blur_or_soft_fusion": 15,
    "clear_topology_or_attachment": 15,
}


def source_binding() -> dict[str, Any]:
    manifest, _ = contract.validate_source_lock(invoked_source=Path(__file__).resolve())
    return {
        "review_source_lock": str(contract.REVIEW_SOURCE_LOCK),
        "review_source_lock_identity_sha256": manifest["identity_sha256"],
        "runner_source_sha256": contract.sha256_file(Path(__file__).resolve()),
    }


def one_token(rows: Sequence[Mapping[str, str]], field: str, description: str) -> str:
    values = {row[field].strip() for row in rows}
    if len(values) != 1 or not next(iter(values)):
        raise RuntimeError(f"{description} must contain exactly one nonempty {field}")
    return next(iter(values))


def anchor_catalog() -> tuple[dict[str, Any], ...]:
    protocol, _ = contract.validate_event_protocol_lock()
    rows = protocol["label_system"]["instructional_anchor_catalog"]
    if not isinstance(rows, list) or len(rows) != 20:
        raise RuntimeError("event protocol no longer contains exactly 20 visible anchors")
    ids = [row.get("anchor_id") for row in rows]
    if ids != [f"qa_{index:02d}" for index in range(20)]:
        raise RuntimeError("visible-anchor axis/order changed")
    for row in rows:
        path = contract.EVENT_PROTOCOL_LOCK / row["frozen_relative_path"]
        if contract.sha256_file(contract.require_regular(path, "visible anchor PNG")) != row["sha256"]:
            raise RuntimeError(f"visible anchor bytes changed: {row['anchor_id']}")
    return tuple(rows)


def validate_ratification_form(path: Path, catalog: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    rows = contract.read_csv_exact(path, contract.VISIBLE_RATIFICATION_FIELDS)
    expected_ids = [(row["anchor_id"],) for row in catalog]
    contract.validate_unique_axis(rows, ("anchor_id",), expected_ids)
    token = one_token(rows, "expert_role_token", "visible-anchor expert form")
    for observed, expected in zip(rows, catalog):
        frozen = {
            "image_sha256": expected["sha256"],
            "proposed_severity": expected["severity"],
            "proposed_component_group": expected["component_group"],
            "proposed_reason": expected["reason"],
        }
        if any(observed[key] != value for key, value in frozen.items()):
            raise RuntimeError(f"visible anchor proposal was edited: {expected['anchor_id']}")
        contract.validate_attestation(observed["independence_attestation"])
        if observed["decision"] != "ratify":
            raise RuntimeError(
                f"expert did not ratify {expected['anchor_id']}; revise the frozen protocol before continuing"
            )
        if any(observed[key].strip() for key in ("correction_severity", "correction_component_group", "correction_reason")):
            raise RuntimeError("ratified rows must leave all correction fields empty")
    return token, rows


def lock_visible(args: argparse.Namespace) -> Path:
    catalog = anchor_catalog()
    token_1, rows_1 = validate_ratification_form(args.expert_1, catalog)
    token_2, rows_2 = validate_ratification_form(args.expert_2, catalog)
    if token_1 == token_2:
        raise RuntimeError("the two expert role tokens must be distinct")
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_VISIBLE_ANCHOR_RATIFICATION_LOCK_V1",
        "status": "TWO_EXPERTS_RATIFIED_ALL_20_VISIBLE_ANCHORS",
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "anchor_count": 20,
        "expert_role_tokens": [token_1, token_2],
        "actual_personhood_or_organizational_independence_verified_by_software": False,
        "required_external_fact": "role tokens must represent two genuinely independent qualified experts",
        "evidence_access": {
            "endpoint_pixels": True,
            "visible_frozen_rubric_and_reasons": True,
            "candidate_scores_features_trajectories_embeddings_or_ranks": False,
        },
        **source_binding(),
    }

    def builder(root: Path) -> None:
        contract.write_csv(root / "expert_1_ratification.csv", contract.VISIBLE_RATIFICATION_FIELDS, rows_1)
        contract.write_csv(root / "expert_2_ratification.csv", contract.VISIBLE_RATIFICATION_FIELDS, rows_2)
        contract.write_json(
            root / "ratification_summary.json",
            {
                "all_20_ratified_by_both": True,
                "anchor_count": 20,
                "expert_count": 2,
                "protocol_change_required_if_any_expert_rejects": True,
            },
        )

    return contract.publish_artifact(args.output, identity=identity, builder=builder)


def validate_visible_lock(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, completion = contract.validate_artifact(
        path, expected_kind="EVENT_RICH_VISIBLE_ANCHOR_RATIFICATION_LOCK_V1"
    )
    identity = manifest["identity"]
    if identity.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY or identity.get("anchor_count") != 20:
        raise RuntimeError("visible-anchor ratification lock binding changed")
    return manifest, completion


def inspect_image(path: Path) -> dict[str, Any]:
    path = contract.require_regular(path, "hidden-form endpoint PNG")
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = tuple(image.size)
        pixel_sha256 = __import__("hashlib").sha256(image.tobytes()).hexdigest()
    if mode != "RGB" or size != (256, 256):
        raise RuntimeError(f"hidden-form image must be RGB 256x256: {path}")
    return {
        "path": path,
        "sha256": contract.sha256_file(path),
        "pixel_sha256": pixel_sha256,
        "width": size[0],
        "height": size[1],
        "mode": mode,
    }


def load_hidden_items(path: Path) -> list[dict[str, Any]]:
    rows = contract.read_csv_exact(path, contract.HIDDEN_ITEM_FIELDS)
    if len(rows) != 60:
        raise RuntimeError("each hidden qualification/reserve form requires exactly 60 items")
    contract.validate_unique_axis(rows, ("item_id",))
    normalized: list[dict[str, Any]] = []
    seen_pixels: set[str] = set()
    for row in rows:
        if not row["item_id"].strip() or len(row["item_id"]) > 80:
            raise RuntimeError("hidden item_id is empty or too long")
        class_id = contract.parse_int(row["class_id"], "class_id", 0, 999)
        if not row["class_name"].strip():
            raise RuntimeError("hidden item class_name is required")
        inspected = inspect_image(Path(row["image_path"]))
        expected = {
            "image_sha256": inspected["sha256"],
            "image_pixel_sha256": inspected["pixel_sha256"],
            "width": "256",
            "height": "256",
            "mode": "RGB",
        }
        if any(row[key] != value for key, value in expected.items()):
            raise RuntimeError(f"hidden item provenance mismatch: {row['item_id']}")
        if inspected["pixel_sha256"] in seen_pixels:
            raise RuntimeError("hidden form contains duplicate endpoint pixels")
        seen_pixels.add(inspected["pixel_sha256"])
        normalized.append(
            {
                "item_id": row["item_id"],
                "class_id": class_id,
                "class_name": row["class_name"].strip(),
                "source_path": str(inspected["path"]),
                "image_sha256": inspected["sha256"],
                "image_pixel_sha256": inspected["pixel_sha256"],
            }
        )
    return normalized


def load_expert_labels(path: Path, item_ids: Sequence[str], description: str) -> tuple[str, dict[str, dict[str, Any]]]:
    rows = contract.read_csv_exact(path, contract.EXPERT_LABEL_FIELDS)
    contract.validate_unique_axis(rows, ("item_id",), [(value,) for value in item_ids])
    token = one_token(rows, "expert_role_token", description)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        contract.validate_attestation(row["independence_attestation"])
        result[row["item_id"]] = contract.validate_severity_row(row, always_reason=True)
    return token, result


def label_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (value["severity"], tuple(value["components"]))


def gold_stratum(label: Mapping[str, Any]) -> str:
    severity = int(label["severity"])
    components = set(label["components"])
    if severity == 0:
        return "clean_good"
    if severity == 1:
        return "mild_or_disputed"
    if components & contract.BLUR_COMPONENTS:
        return "clear_blur_or_soft_fusion"
    if components & contract.STRUCTURE_COMPONENTS:
        return "clear_topology_or_attachment"
    raise RuntimeError("clear-bad hidden gold must contain a blur/fusion or topology/attachment component")


def lock_hidden_gold(args: argparse.Namespace) -> Path:
    visible_manifest, _ = validate_visible_lock(args.visible_anchor_lock)
    items = load_hidden_items(args.item_index)
    visible_hashes = {row["sha256"] for row in anchor_catalog()}
    if visible_hashes & {row["image_sha256"] for row in items}:
        raise RuntimeError("visible instructional anchors cannot be hidden qualification items")
    item_ids = [row["item_id"] for row in items]
    token_1, curator_1 = load_expert_labels(args.curator_1, item_ids, "curator 1 form")
    token_2, curator_2 = load_expert_labels(args.curator_2, item_ids, "curator 2 form")
    if token_1 == token_2:
        raise RuntimeError("curator role tokens must be distinct")
    disagreement_ids = [
        item_id
        for item_id in item_ids
        if label_signature(curator_1[item_id]) != label_signature(curator_2[item_id])
    ]
    resolver_rows = contract.read_csv_exact(
        args.resolver,
        contract.EXPERT_LABEL_FIELDS,
        allow_empty=not disagreement_ids,
    )
    contract.validate_unique_axis(
        resolver_rows,
        ("item_id",),
        [(value,) for value in disagreement_ids],
    )
    resolver_token = ""
    resolver: dict[str, dict[str, Any]] = {}
    if disagreement_ids:
        resolver_token = one_token(resolver_rows, "expert_role_token", "resolver form")
        if resolver_token in {token_1, token_2}:
            raise RuntimeError("resolver role token must differ from both curator tokens")
        for row in resolver_rows:
            contract.validate_attestation(row["independence_attestation"])
            resolver[row["item_id"]] = contract.validate_severity_row(row, always_reason=True)
    gold: list[dict[str, Any]] = []
    for item in items:
        item_id = item["item_id"]
        final = resolver[item_id] if item_id in resolver else curator_1[item_id]
        stratum = gold_stratum(final)
        gold.append({**item, **final, "gold_stratum": stratum})
    counts = {name: sum(row["gold_stratum"] == name for row in gold) for name in HIDDEN_BALANCE}
    if counts != HIDDEN_BALANCE:
        raise RuntimeError(f"hidden gold balance must be exactly {HIDDEN_BALANCE}; observed={counts}")

    disjoint_identity = None
    if args.form_kind == "reserve":
        if args.disjoint_from is None:
            raise RuntimeError("reserve gold requires --disjoint-from the primary hidden-gold lock")
        other_manifest, _ = contract.validate_artifact(
            args.disjoint_from, expected_kind="EVENT_RICH_HIDDEN_QUALIFICATION_GOLD_LOCK_V1"
        )
        if other_manifest["identity"].get("form_kind") != "qualification":
            raise RuntimeError("reserve must be disjoint from the primary qualification form")
        other_rows = contract.read_csv_exact(
            args.disjoint_from / "custody/gold_rows.csv",
            (
                "qualification_id",
                "source_item_id",
                "class_id",
                "class_name",
                "image_sha256",
                "image_pixel_sha256",
                "severity",
                "components",
                "localization_reason",
                "gold_stratum",
            ),
        )
        other_pixels = {row["image_pixel_sha256"] for row in other_rows}
        if other_pixels & {row["image_pixel_sha256"] for row in gold}:
            raise RuntimeError("qualification and reserve forms reuse endpoint pixels")
        disjoint_identity = other_manifest["identity_sha256"]
    elif args.disjoint_from is not None:
        raise RuntimeError("--disjoint-from is only valid for reserve")

    output = args.output.expanduser().absolute()
    authorized_evaluation_path = str(output.parent / f"{output.name}_evaluation")
    form_salt = contract.canonical_sha256(
        {
            "form_kind": args.form_kind,
            "event_protocol": contract.EVENT_PROTOCOL_IDENTITY,
            "pixel_hashes": [row["image_pixel_sha256"] for row in gold],
        }
    )
    released = []
    for row in gold:
        qid = "q_" + contract.stable_blind_id(
            "eqvae.dit.event-rich.hidden-qualification.v1",
            args.form_kind,
            form_salt,
            row["item_id"],
        )
        released.append({**row, "qualification_id": qid})
    released.sort(key=lambda row: contract.stable_blind_id(form_salt, row["qualification_id"]))
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_HIDDEN_QUALIFICATION_GOLD_LOCK_V1",
        "status": "HIDDEN_GOLD_FROZEN_BEFORE_PANEL_QUALIFICATION",
        "form_kind": args.form_kind,
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "visible_anchor_ratification_identity_sha256": visible_manifest["identity_sha256"],
        "disjoint_primary_gold_identity_sha256": disjoint_identity,
        "item_count": 60,
        "gold_balance": counts,
        "curator_role_tokens": [token_1, token_2],
        "resolver_role_token": resolver_token or None,
        "curator_disagreement_count": len(disagreement_ids),
        "authorized_evaluation_path": authorized_evaluation_path,
        "actual_personhood_or_organizational_independence_verified_by_software": False,
        "required_external_fact": "curator/resolver tokens must represent genuinely independent qualified experts",
        "distribution_rule": "release only reviewer_release; never custody or artifact root",
        "candidate_scores_features_trajectories_embeddings_ranks_opened": False,
        **source_binding(),
    }

    gold_fields = (
        "qualification_id",
        "source_item_id",
        "class_id",
        "class_name",
        "image_sha256",
        "image_pixel_sha256",
        "severity",
        "components",
        "localization_reason",
        "gold_stratum",
    )
    release_fields = (
        "qualification_id",
        "class_id",
        "class_name",
        "image_relative_path",
        "image_sha256",
        "image_pixel_sha256",
        "mode",
        "width",
        "height",
    )

    def builder(root: Path) -> None:
        custody = root / "custody"
        release = root / "reviewer_release"
        images = release / "images"
        custody.mkdir()
        release.mkdir()
        images.mkdir()
        gold_rows = []
        release_rows = []
        for row in released:
            relative = f"images/{row['qualification_id']}.png"
            destination = release / relative
            shutil.copyfile(row["source_path"], destination)
            if contract.sha256_file(destination) != row["image_sha256"]:
                raise RuntimeError("hidden-form image copy changed bytes")
            gold_rows.append(
                {
                    "qualification_id": row["qualification_id"],
                    "source_item_id": row["item_id"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "image_sha256": row["image_sha256"],
                    "image_pixel_sha256": row["image_pixel_sha256"],
                    "severity": row["severity"],
                    "components": ";".join(row["components"]),
                    "localization_reason": row["localization_reason"],
                    "gold_stratum": row["gold_stratum"],
                }
            )
            release_rows.append(
                {
                    "qualification_id": row["qualification_id"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "image_relative_path": relative,
                    "image_sha256": row["image_sha256"],
                    "image_pixel_sha256": row["image_pixel_sha256"],
                    "mode": "RGB",
                    "width": 256,
                    "height": 256,
                }
            )
        contract.write_csv(custody / "gold_rows.csv", gold_fields, gold_rows)
        contract.write_csv(release / "item_index.csv", release_fields, release_rows)
        for slot in contract.ROLE_SLOTS:
            template = [
                {
                    "qualification_id": row["qualification_id"],
                    "severity": "",
                    "components": "",
                    "localization_reason": "",
                    "role_slot": slot,
                    "role_token": "",
                    "independence_attestation": "",
                }
                for row in release_rows
            ]
            contract.write_csv(
                release / f"{slot}_response_template.csv",
                contract.QUALIFICATION_RESPONSE_FIELDS,
                template,
            )
        contract.write_json(
            release / "READ_THIS_BEFORE_DISTRIBUTION.json",
            {
                "distribute_only_this_directory": True,
                "gold_labels_present_here": False,
                "role_slots": list(contract.ROLE_SLOTS),
                "reviewer_or_adjudicator_must_not_receive_other_role_forms": True,
                "visible_anchor_examples_should_be_provided_separately": True,
                "forbidden_context": [
                    "candidate B/C hypotheses or selected-class ranks",
                    "trajectory features, metric scores, thresholds, alerts",
                    "Inception, DINO, FID or any embedding distance",
                    "gold labels or another role's responses",
                ],
            },
        )

    return contract.publish_artifact(output, identity=identity, builder=builder)


def validate_gold_lock(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest, _ = contract.validate_artifact(
        path, expected_kind="EVENT_RICH_HIDDEN_QUALIFICATION_GOLD_LOCK_V1"
    )
    identity = manifest["identity"]
    if identity.get("event_protocol_identity_sha256") != contract.EVENT_PROTOCOL_IDENTITY or identity.get("item_count") != 60:
        raise RuntimeError("hidden-gold identity binding changed")
    fields = (
        "qualification_id",
        "source_item_id",
        "class_id",
        "class_name",
        "image_sha256",
        "image_pixel_sha256",
        "severity",
        "components",
        "localization_reason",
        "gold_stratum",
    )
    rows = contract.read_csv_exact(path / "custody/gold_rows.csv", fields)
    contract.validate_unique_axis(rows, ("qualification_id",))
    if len(rows) != 60:
        raise RuntimeError("hidden-gold row count changed")
    return manifest, rows


def load_qualification_response(
    path: Path,
    slot: str,
    qualification_ids: Sequence[str],
) -> tuple[str, list[dict[str, Any]]]:
    rows = contract.read_csv_exact(path, contract.QUALIFICATION_RESPONSE_FIELDS)
    contract.validate_unique_axis(
        rows,
        ("qualification_id",),
        [(value,) for value in qualification_ids],
    )
    if {row["role_slot"] for row in rows} != {slot}:
        raise RuntimeError(f"qualification form role_slot must be exactly {slot}")
    token = one_token(rows, "role_token", f"qualification response {slot}")
    parsed = []
    for row in rows:
        contract.validate_attestation(row["independence_attestation"])
        parsed.append({"qualification_id": row["qualification_id"], **contract.validate_severity_row(row)})
    return token, parsed


def evaluate_qualification(args: argparse.Namespace) -> Path:
    gold_manifest, gold_rows = validate_gold_lock(args.gold_lock)
    gold_identity = gold_manifest["identity"]
    output = args.output.expanduser().absolute()
    if str(output) != gold_identity.get("authorized_evaluation_path"):
        raise RuntimeError(
            "qualification output path is not the unique pre-authorized path: "
            f"expected={gold_identity.get('authorized_evaluation_path')}"
        )
    form_kind = gold_identity["form_kind"]
    prior_identity = None
    if form_kind == "reserve":
        if args.prior_failed_evaluation is None:
            raise RuntimeError("reserve evaluation requires --prior-failed-evaluation")
        prior_manifest, _ = contract.validate_artifact(
            args.prior_failed_evaluation,
            expected_kind="EVENT_RICH_PANEL_QUALIFICATION_EVALUATION_V1",
        )
        prior = prior_manifest["identity"]
        if prior.get("panel_passed") is not False or prior.get("form_kind") != "qualification":
            raise RuntimeError("reserve is authorized only after a failed primary qualification")
        if prior.get("gold_identity_sha256") != gold_identity.get("disjoint_primary_gold_identity_sha256"):
            raise RuntimeError("reserve is not bound to this failed primary form")
        prior_identity = prior_manifest["identity_sha256"]
    elif args.prior_failed_evaluation is not None:
        raise RuntimeError("primary qualification cannot cite a previous attempt")
    qualification_ids = [row["qualification_id"] for row in gold_rows]
    response_paths = {
        slot: getattr(args, slot.replace("_", "_"))
        for slot in contract.ROLE_SLOTS
    }
    tokens: dict[str, str] = {}
    responses: dict[str, list[dict[str, Any]]] = {}
    for slot in contract.ROLE_SLOTS:
        token, rows = load_qualification_response(response_paths[slot], slot, qualification_ids)
        tokens[slot] = token
        responses[slot] = rows
    if len(set(tokens.values())) != len(tokens):
        raise RuntimeError("all five panel role tokens must be distinct")
    gold_binary = [int(row["severity"]) >= 2 for row in gold_rows]
    predictions = {
        slot: [int(row["severity"]) >= 2 for row in responses[slot]]
        for slot in contract.ROLE_SLOTS
    }
    metrics = contract.qualification_metrics(gold_binary, predictions)
    passed = bool(metrics["panel_passed"])
    identity = {
        "schema_version": 1,
        "artifact_kind": "EVENT_RICH_PANEL_QUALIFICATION_EVALUATION_V1",
        "status": "PASS_FORMAL_SCREEN_RELEASE_AUTHORIZED" if passed else "FAIL_STOP_FORM_CONSUMED_NO_SCREEN_RELEASE",
        "event_protocol_identity_sha256": contract.EVENT_PROTOCOL_IDENTITY,
        "gold_identity_sha256": gold_manifest["identity_sha256"],
        "form_kind": form_kind,
        "prior_failed_evaluation_identity_sha256": prior_identity,
        "panel_passed": passed,
        "form_consumed": True,
        "may_reuse_this_form_after_failure": False,
        "formal_screen_release_authorized": passed,
        "qualified_role_tokens": tokens if passed else None,
        "role_slots": list(contract.ROLE_SLOTS),
        "actual_personhood_or_organizational_independence_verified_by_software": False,
        "required_external_fact": "role tokens must map to five genuinely independent qualified reviewer/expert roles",
        "gold_or_other_role_responses_visible_to_panel": False,
        "candidate_scores_features_trajectories_embeddings_ranks_opened": False,
        **source_binding(),
    }

    def builder(root: Path) -> None:
        contract.write_json(root / "qualification_metrics.json", metrics)
        for slot in contract.ROLE_SLOTS:
            shutil.copyfile(response_paths[slot], root / f"{slot}_response.csv")
        contract.write_json(
            root / "release_decision.json",
            {
                "panel_passed": passed,
                "decision": "RELEASE_FORMAL_SCREEN" if passed else "STOP_AND_USE_DISJOINT_RESERVE_WITH_REPLACED_OR_RETRAINED_FAILED_ROLES",
                "primary_form_may_be_reused": False,
            },
        )

    return contract.publish_artifact(output, identity=identity, builder=builder)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    visible = sub.add_parser("lock-visible")
    visible.add_argument("--expert-1", type=Path, required=True)
    visible.add_argument("--expert-2", type=Path, required=True)
    visible.add_argument("--output", type=Path, required=True)
    visible.set_defaults(func=lock_visible)

    hidden = sub.add_parser("lock-hidden-gold")
    hidden.add_argument("--form-kind", choices=("qualification", "reserve"), required=True)
    hidden.add_argument("--visible-anchor-lock", type=Path, required=True)
    hidden.add_argument("--item-index", type=Path, required=True)
    hidden.add_argument("--curator-1", type=Path, required=True)
    hidden.add_argument("--curator-2", type=Path, required=True)
    hidden.add_argument("--resolver", type=Path, required=True)
    hidden.add_argument("--disjoint-from", type=Path)
    hidden.add_argument("--output", type=Path, required=True)
    hidden.set_defaults(func=lock_hidden_gold)

    qualify = sub.add_parser("evaluate-qualification")
    qualify.add_argument("--gold-lock", type=Path, required=True)
    for slot in contract.ROLE_SLOTS:
        qualify.add_argument(f"--{slot.replace('_', '-')}", dest=slot, type=Path, required=True)
    qualify.add_argument("--prior-failed-evaluation", type=Path)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.set_defaults(func=evaluate_qualification)
    return root


def main() -> None:
    args = parser().parse_args()
    path = args.func(args)
    print(json.dumps({"output": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()

