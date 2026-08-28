#!/usr/bin/env python3
"""Freeze the two-candidate, phenotype-aware DiT third-pool protocol.

This locker is deliberately score-blind.  It validates only immutable,
label-free product metadata, feature catalogs, protocol/source provenance, and
the sampling sources.  It never opens sample feature tables, visual labels,
review files, endpoint images, trajectory archives, or screening-result tables.

The resulting lock fixes seeds 250..849 (600 shared global-seed blocks, three
classes per block), exactly two non-combined co-primary candidates, the visual
phenotype definition, the event-count gate, and the Holm-adjusted confirmation
rules before any third-pool sampling begins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))

DEFAULT_PRIMARY_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_confirmation_v1/custom_label_free_v1"
)
DEFAULT_VISUAL_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_confirmation_v1/"
    "predxstart_visual_label_free_v1"
)
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_bad_good_third_pool_protocol_lock_v1"

PRIMARY_FEATURE = (
    "pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump"
)
VISUAL_FEATURE = "decoded_local_blur_severity__mean"
CLASSES = (207, 602, 795)
CALIBRATION_SEEDS = tuple(range(30, 50))
REFERENCE_PRODUCT_SEEDS = tuple(range(30, 130))
THIRD_POOL_SEEDS = tuple(range(250, 850))
THIRD_POOL_TRAJECTORIES = len(CLASSES) * len(THIRD_POOL_SEEDS)
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026082801

SOURCE_PATHS = {
    "primary_feature_extractor.py": ROOT
    / "experiments/analyze_dit_bad_good_custom_traces.py",
    "visual_feature_extractor.py": ROOT
    / "experiments/extract_dit_predxstart_visual_tracks.py",
    "trace_runner.py": ROOT / "experiments/trace_dit_imagenet256_custom_batch.py",
    "custom_sampler_helper.py": ROOT / "experiments/sample_dit_imagenet256_custom.py",
    "strict_reproduction_helper.py": ROOT / "experiments/reproduce_dit_imagenet256.py",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular file: {path}")
    return path.resolve()


def require_real_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real directory: {path}")
    return path.resolve()


def catalog_row(path: Path, feature: str) -> dict[str, str]:
    """Read one metadata row only; feature/sample score files are never opened."""

    matches: list[dict[str, str]] = []
    with require_regular(path, "feature catalog").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("feature") == feature:
                matches.append(dict(row))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one feature-catalog row for {feature}, got {len(matches)}"
        )
    return matches[0]


def validate_label_free_product(
    root: Path,
    *,
    feature: str,
    expected_experiment: str,
    expected_summary_status: str,
    supervision_field: str,
) -> dict[str, Any]:
    root = require_real_directory(root, "label-free feature product")
    manifest_path = require_regular(root / "manifest.json", "product manifest")
    completion_path = require_regular(root / "completion.json", "product completion")
    summary_path = require_regular(root / "summary.json", "product summary")
    catalog_path = require_regular(root / "feature_catalog.csv", "feature catalog")
    inventory_path = require_regular(root / "source_inventory.json", "source inventory")
    protocol_path = require_regular(root / "protocol_snapshot.json", "protocol snapshot")

    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    summary = load_json(summary_path)
    inventory = load_json(inventory_path)
    protocol = load_json(protocol_path)

    identity = manifest.get("identity_sha256")
    if (
        not isinstance(identity, str)
        or canonical_sha256(without_identity(manifest)) != identity
        or manifest.get("status") != "complete"
        or manifest.get("experiment") != expected_experiment
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != identity
        or completion.get("summary_file_sha256") != sha256_file(summary_path)
        or summary.get("status") != expected_summary_status
        or summary.get(supervision_field) is not False
        or tuple(summary.get("ordered_classes", ())) != CLASSES
        or tuple(summary.get("ordered_seeds", ())) != REFERENCE_PRODUCT_SEEDS
        or summary.get("sample_count") != len(CLASSES) * len(REFERENCE_PRODUCT_SEEDS)
    ):
        raise RuntimeError(f"label-free product contract failed: {root}")

    listed = manifest.get("files")
    if not isinstance(listed, list) or not all(isinstance(row, dict) for row in listed):
        raise RuntimeError(f"product manifest member list is malformed: {root}")
    by_name = {str(row.get("name")): row for row in listed}
    if len(by_name) != len(listed):
        raise RuntimeError(f"duplicate product manifest member: {root}")
    required_members = {
        "analysis_source.py",
        "feature_catalog.csv",
        "protocol_snapshot.json",
        "sample_features.csv",
        "source_inventory.json",
        "summary.json",
    }
    if not required_members.issubset(by_name):
        raise RuntimeError(f"product manifest lacks required members: {root}")
    for name, row in by_name.items():
        member = require_regular(root / name, f"product member {name}")
        if member.stat().st_size != row.get("bytes") or sha256_file(member) != row.get(
            "sha256"
        ):
            raise RuntimeError(f"product member changed: {member}")

    source = inventory.get("analysis_source")
    if (
        not isinstance(source, dict)
        or source.get("sha256") != by_name["analysis_source.py"].get("sha256")
        or manifest.get("analysis_source_sha256")
        != by_name["analysis_source.py"].get("sha256")
        or manifest.get("source_inventory_sha256") != sha256_file(inventory_path)
        or manifest.get("protocol_snapshot_sha256") != sha256_file(protocol_path)
    ):
        raise RuntimeError(f"analysis-source lineage failed: {root}")

    row = catalog_row(catalog_path, feature)
    if (
        row.get("latest_required_sampling_step") != "149"
        or row.get("latest_required_internal_timestep") != "100"
        or row.get("preterminal_actionable") != "True"
        or row.get("uses_realized_innovation") != "False"
    ):
        raise RuntimeError(f"candidate timing contract changed: {feature}")

    # Bind the complete product and exact feature implementation without opening
    # sample_features.csv, any labels, images, traces, or screening outputs.
    return {
        "path": str(root),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": identity,
        "completion_file_sha256": sha256_file(completion_path),
        "summary_file_sha256": sha256_file(summary_path),
        "source_inventory_file_sha256": sha256_file(inventory_path),
        "protocol_snapshot_file_sha256": sha256_file(protocol_path),
        "analysis_source_file_sha256": by_name["analysis_source.py"]["sha256"],
        "feature_catalog_file_sha256": by_name["feature_catalog.csv"]["sha256"],
        "sample_features_file_sha256": by_name["sample_features.csv"]["sha256"],
        "feature_catalog_record": row,
        "feature": feature,
        "experiment": expected_experiment,
        "label_free_supervision_verified": True,
    }


def source_snapshots(
    primary_lineage: Mapping[str, Any], visual_lineage: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, path in SOURCE_PATHS.items():
        source = require_regular(path, f"source {name}")
        records[name] = {
            "live_path_at_freeze": str(source),
            "sha256": sha256_file(source),
        }
    if (
        records["primary_feature_extractor.py"]["sha256"]
        != primary_lineage["analysis_source_file_sha256"]
        or records["visual_feature_extractor.py"]["sha256"]
        != visual_lineage["analysis_source_file_sha256"]
    ):
        raise RuntimeError("live feature source differs from the bound label-free product")
    return records


def build_protocol(
    primary_lineage: Mapping[str, Any],
    visual_lineage: Mapping[str, Any],
    snapshots: Mapping[str, Any],
) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_THIRD_POOL_SAMPLING_OR_VISUAL_REVIEW",
        "objective": (
            "Independently confirm two fixed, preterminal DiT trajectory signals on a "
            "new seed cohort while separating blur/soft-fusion failures from discrete "
            "structural failures."
        ),
        "third_pool": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "sampling_steps": 250,
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "classes_ordered": list(CLASSES),
            "global_seeds": list(THIRD_POOL_SEEDS),
            "global_seed_start_inclusive": THIRD_POOL_SEEDS[0],
            "global_seed_stop_inclusive": THIRD_POOL_SEEDS[-1],
            "global_seed_count": len(THIRD_POOL_SEEDS),
            "trajectories_per_class": len(THIRD_POOL_SEEDS),
            "trajectory_count": THIRD_POOL_TRAJECTORIES,
            "one_global_seed_block_contains_all_three_classes": True,
            "observation_only_generation": True,
            "disjoint_from_calibration_and_prior_confirmation_seeds_30_249": True,
        },
        "co_primary_family": {
            "family_size": 2,
            "candidate_ids": ["B_blur_mean", "C_c3_low_jump"],
            "combination_allowed": False,
            "post_hoc_candidate_choice_allowed": False,
            "both_candidates_reported_independently": True,
            "neither_candidate_requires_the_other_to_pass_for_its_own_conclusion": True,
        },
        "candidates": {
            "B_blur_mean": {
                "feature": VISUAL_FEATURE,
                "raw_orientation": "bad_high",
                "primary_endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                "guardrail_endpoint": "all_clear_bad_vs_clean_good",
                "latest_required_sampling_step": 149,
                "latest_required_internal_timestep": 100,
                "preterminal_actionable": True,
                "checkpoint_sampling_steps": [69, 79, 89, 99, 109, 119, 129, 139, 149],
                "formula": (
                    "mean over the nine fixed checkpoints of B_k, where pred_xstart_k "
                    "is decoded by the pinned FP32 SD-VAE; RGB is clipped to [0,1], "
                    "converted to grayscale, Gaussian-smoothed (sigma=0.7), and split "
                    "into a fixed 4x4 grid; the eight highest-variance tiles are active; "
                    "q_j=mean(Laplacian^2)/(mean(Sobel-gradient-magnitude^2)+1e-12); "
                    "B_k=-log(percentile_25(q_j over active tiles)+1e-12)."
                ),
                "claim_if_confirmed": (
                    "preterminal detector for final blur/soft-fusion failures in the "
                    "frozen three-class DiT setting; not a universal artifact detector"
                ),
                "intervention_authorization_if_gate_passes": (
                    "blur-specific intervention experiment only"
                ),
            },
            "C_c3_low_jump": {
                "feature": PRIMARY_FEATURE,
                "raw_orientation": "bad_low",
                "primary_endpoint": "all_clear_bad_vs_clean_good",
                "guardrail_endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                "latest_required_sampling_step": 149,
                "latest_required_internal_timestep": 100,
                "preterminal_actionable": True,
                "formula": (
                    "max(0, max_{k=100..148}(g_{k+1}-g_k)), where "
                    "g_k=alpha_bar[k]*(mean vertical-difference^2 + mean "
                    "horizontal-difference^2) of pred_xstart latent channel 3"
                ),
                "claim_if_confirmed": (
                    "support for the fixed channel-3 alpha-compensated gradient-energy "
                    "jump mechanism; not by itself authorization for blur intervention"
                ),
                "intervention_authorization_if_gate_passes": False,
            },
        },
        "phenotype_contract": {
            "frozen_before_third_pool_images_are_reviewed": True,
            "reviewer_candidate_hypothesis_blind": True,
            "reviewer_metric_score_threshold_alert_and_trajectory_blind": True,
            "three_independent_endpoint_only_reviewers": True,
            "severity_consensus": {
                "clear_bad": "at least two of three independent severity scores are 2 or 3",
                "clean_good": "at least two of three independent severity scores are 0",
                "mild_or_disputed": "neither majority; excluded from binary endpoints",
            },
            "adjudication": {
                "scope": "raw majority clear-bad only",
                "allowed_actions": ["retain_clear_bad", "downgrade_to_mild"],
                "promotion_allowed": False,
                "candidate_hypothesis_blind": True,
            },
            "review_components": {
                "blur_components": [
                    "global_blur",
                    "local_blur",
                    "soft_fusion_or_melting",
                ],
                "discrete_structure_components": [
                    "discrete_duplication_or_extra_part",
                    "detachment_or_floating_part",
                    "topology_or_attachment_error",
                    "limb_or_object_misalignment",
                ],
                "additional_components": ["texture_break", "other", "none"],
            },
            "component_consensus": (
                "For each broad component group, presence requires at least two of "
                "three reviewers to mark any member of that group."
            ),
            "blur_or_soft_fusion_positive": (
                "final retained clear-bad and blur-component consensus present; mixed "
                "blur-plus-discrete-structure failures are included and also reported separately"
            ),
            "structural_non_blur": (
                "final retained clear-bad, no blur-component consensus, and discrete-"
                "structure-component consensus present"
            ),
            "phenotype_disputed": (
                "final retained clear-bad without the required component consensus; "
                "retained in candidate C's all-bad endpoint but excluded from subtype-only cuts"
            ),
            "model_relative_quality_rule": (
                "Severity 2/3 is reserved for a clear defect materially below the normal "
                "quality of this frozen model/class pool; ordinary model limitations, "
                "slight softness, distant subjects, and merely imperfect details are not clear-bad."
            ),
        },
        "score_label_join_unlock_gate": {
            "labels_and_phenotypes_must_be_immutable_first": True,
            "minimum_blur_or_soft_fusion_clear_bad": 15,
            "minimum_total_clear_bad": 30,
            "logical_rule": "both event-count minima must hold",
            "if_gate_fails": (
                "publish aggregate label/phenotype counts only; do not open, compute, "
                "or publish any third-pool score-label join"
            ),
            "detector_performance_must_not_influence_expansion": True,
        },
        "confirmatory_statistics": {
            "primary_statistic": (
                "within-class pair-count-weighted tie-aware ROC AUC, using each "
                "candidate's frozen orientation and its own frozen endpoint"
            ),
            "randomization_test": {
                "unit": "global-seed block containing the ordered three class rows",
                "method": (
                    "apply one common random permutation of complete three-class visual-"
                    "label/phenotype blocks across global seeds, preserving every class's "
                    "observed endpoint counts and cross-class label dependence"
                ),
                "draws": PERMUTATION_DRAWS,
                "rng": f"numpy.default_rng(PCG64(seed={PERMUTATION_SEED}))",
                "p_value": "(1 + exceedances)/(1 + draws)",
                "alternative": "frozen-direction class-matched AUC greater than chance",
            },
            "multiple_testing": {
                "method": "Holm step-down across exactly the two co-primary raw permutation p-values",
                "family_size": 2,
                "familywise_alpha": 0.05,
                "strict_gate": "Holm-adjusted p < 0.05",
                "no_other_hypothesis_enters_family": True,
            },
            "candidate_gates": {
                "B_blur_mean": {
                    "class_matched_auc_at_least": 0.75,
                    "holm_adjusted_permutation_p_below": 0.05,
                    "alpha_0p10_true_positive_count_at_least": 3,
                    "alpha_0p10_TPR_strictly_greater_than_FPR": True,
                },
                "C_c3_low_jump": {
                    "class_matched_auc_at_least": 0.70,
                    "holm_adjusted_permutation_p_below": 0.05,
                },
            },
            "guardrails": [
                "B_blur_mean on all clear-bad versus clean-good",
                "C_c3_low_jump on blur/soft-fusion clear-bad versus clean-good",
                "per-class AUC and event counts",
                "mixed and structural-non-blur phenotype cuts",
            ],
            "guardrails_are_descriptive_not_candidate_replacements": True,
            "report_each_candidate_conclusion_even_if_the_other_fails": True,
        },
        "label_free_threshold_calibration": {
            "classes": list(CLASSES),
            "seeds": list(CALIBRATION_SEEDS),
            "calibration_count_per_class": len(CALIBRATION_SEEDS),
            "third_pool_seeds_excluded": True,
            "visual_labels_used": False,
            "candidate_selection_used_calibration_values": False,
            "alphas": [0.10, 0.05],
            "B_bad_high": {
                "alpha_0p10": "19th ascending order statistic; strict score > threshold",
                "alpha_0p05": "20th ascending order statistic; strict score > threshold",
            },
            "C_bad_low": {
                "alpha_0p10": "2nd ascending order statistic; strict score < threshold",
                "alpha_0p05": "1st ascending order statistic; strict score < threshold",
            },
            "guarantee_scope": (
                "Under within-class exchangeability, the strict marginal trigger bounds "
                "are 2/21 and 1/21. They are overall intervention budgets, not clean-good "
                "conditional false-positive-rate guarantees."
            ),
        },
        "input_lineage": {
            "primary_label_free_product": dict(primary_lineage),
            "visual_label_free_product": dict(visual_lineage),
        },
        "source_snapshots": dict(snapshots),
        "forbidden_changes_after_freeze": [
            "candidate feature, direction, checkpoint, reduction, or endpoint",
            "candidate combination, learned score, or post-hoc candidate choice",
            "seed range, class set, phenotype definition, event-count gate, or reviewer rubric",
            "permutation unit, p-value family, Holm correction, AUC gate, or operating-point gate",
            "calibration seeds, tail, order statistic, alpha, or strict comparison",
        ],
        "evidence_access_audit": {
            "sample_feature_tables_opened_by_protocol_locker": False,
            "visual_labels_or_reviews_opened": False,
            "endpoint_images_opened": False,
            "trajectory_archives_opened": False,
            "screening_result_tables_or_rows_opened": False,
            "feature_catalog_metadata_opened": True,
            "label_free_manifests_and_source_metadata_opened": True,
        },
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"lock artifact must not be a symlink: {path}")
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_protocol_lock(root: Path) -> dict[str, Any]:
    root = require_real_directory(root, "third-pool protocol lock")
    protocol_path = require_regular(root / "third_pool_protocol.json", "protocol")
    manifest_path = require_regular(root / "manifest.json", "lock manifest")
    completion_path = require_regular(root / "completion.json", "lock completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    protocol_identity = protocol.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    if (
        not isinstance(protocol_identity, str)
        or canonical_sha256(without_identity(protocol)) != protocol_identity
        or not isinstance(manifest_identity, str)
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != protocol_identity
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("protocol_identity_sha256") != protocol_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
    ):
        raise RuntimeError(f"third-pool protocol lock validation failed: {root}")
    pool = protocol.get("third_pool", {})
    family = protocol.get("co_primary_family", {})
    gate = protocol.get("score_label_join_unlock_gate", {})
    if (
        protocol.get("status") != "FROZEN_BEFORE_THIRD_POOL_SAMPLING_OR_VISUAL_REVIEW"
        or tuple(pool.get("classes_ordered", ())) != CLASSES
        or tuple(pool.get("global_seeds", ())) != THIRD_POOL_SEEDS
        or pool.get("trajectory_count") != THIRD_POOL_TRAJECTORIES
        or family.get("family_size") != 2
        or tuple(family.get("candidate_ids", ())) != ("B_blur_mean", "C_c3_low_jump")
        or family.get("combination_allowed") is not False
        or gate.get("minimum_blur_or_soft_fusion_clear_bad") != 15
        or gate.get("minimum_total_clear_bad") != 30
    ):
        raise RuntimeError("third-pool protocol scientific contract changed")
    return protocol


def publish(primary_root: Path, visual_root: Path, output: Path) -> Path:
    primary = validate_label_free_product(
        primary_root,
        feature=PRIMARY_FEATURE,
        expected_experiment="dit_bad_good_custom_trace_metric_discovery",
        expected_summary_status="DISCOVERY_ONLY_NOT_AN_INTERVENTION_TRIGGER",
        supervision_field="labels_joined",
    )
    visual = validate_label_free_product(
        visual_root,
        feature=VISUAL_FEATURE,
        expected_experiment="dit_predxstart_preterminal_visual_tracks_label_free",
        expected_summary_status="COMPLETE_LABEL_FREE_VISUAL_TRACK_EXTRACTION",
        supervision_field="labels_read_or_emitted",
    )
    snapshots = source_snapshots(primary, visual)
    protocol = build_protocol(primary, visual, snapshots)

    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite third-pool protocol lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "third_pool_protocol.json", protocol)
        shutil.copy2(Path(__file__).resolve(), staging / "locker_source.py")
        snapshot_root = staging / "source_snapshots"
        snapshot_root.mkdir()
        product_sources = {
            "primary_feature_extractor.py": Path(primary["path"]) / "analysis_source.py",
            "visual_feature_extractor.py": Path(visual["path"]) / "analysis_source.py",
        }
        for name, live_path in SOURCE_PATHS.items():
            source = product_sources.get(name, live_path)
            shutil.copy2(source, snapshot_root / name)
            if sha256_file(snapshot_root / name) != snapshots[name]["sha256"]:
                raise RuntimeError(f"source changed while freezing: {source}")
        members = artifact_records(staging)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "protocol_identity_sha256": protocol["identity_sha256"],
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "protocol_file_sha256": sha256_file(staging / "third_pool_protocol.json"),
                "protocol_identity_sha256": protocol["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
            },
        )
        validate_protocol_lock(staging)
        os.replace(staging, output)
        validate_protocol_lock(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    dummy = {
        "path": "/label-free",
        "analysis_source_file_sha256": "a" * 64,
        "feature": "dummy",
    }
    snapshots = {
        name: {"live_path_at_freeze": f"/{name}", "sha256": "b" * 64}
        for name in SOURCE_PATHS
    }
    protocol = build_protocol(dummy, dummy, snapshots)
    assert canonical_sha256(without_identity(protocol)) == protocol["identity_sha256"]
    assert protocol["third_pool"]["trajectory_count"] == 1800
    assert protocol["co_primary_family"]["family_size"] == 2
    assert protocol["co_primary_family"]["combination_allowed"] is False
    assert protocol["score_label_join_unlock_gate"] == {
        "labels_and_phenotypes_must_be_immutable_first": True,
        "minimum_blur_or_soft_fusion_clear_bad": 15,
        "minimum_total_clear_bad": 30,
        "logical_rule": "both event-count minima must hold",
        "if_gate_fails": (
            "publish aggregate label/phenotype counts only; do not open, compute, "
            "or publish any third-pool score-label join"
        ),
        "detector_performance_must_not_influence_expansion": True,
    }
    assert protocol["confirmatory_statistics"]["multiple_testing"]["family_size"] == 2
    print(
        "self-test passed: 600 seed blocks, 1800 trajectories, two non-combined "
        "co-primary candidates, phenotype/event gates, and Holm family"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.validate is not None:
        protocol = validate_protocol_lock(args.validate)
        print(
            json.dumps(
                {
                    "output": str(args.validate.expanduser().absolute()),
                    "protocol_identity_sha256": protocol["identity_sha256"],
                    "status": "valid",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = publish(args.primary_root, args.visual_root, args.output)
    protocol = validate_protocol_lock(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "protocol_identity_sha256": protocol["identity_sha256"],
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
