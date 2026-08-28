#!/usr/bin/env python3
"""Freeze the label-free discovery reference for prospective bad/good testing.

The lock deliberately reads only label-free feature products.  It fixes one
primary two-component score and two single-feature backups before any fresh
confirmation trajectory is generated or visually labelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
DEFAULT_PRIMARY = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_discovery/dit_targeted160_custom_label_free_v3"
)
DEFAULT_POSTERIOR = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_discovery/dit_targeted160_posterior_evidence_label_free_v1"
)
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v2"

FEATURE_A = (
    "learned_range_cond_minus_uncond_logstd_gap_tile4x4_concentration_guided3"
    "__q2_max_positive_jump"
)
FEATURE_B = (
    "pred_xstart_cond_uncond_disagreement_rms_channel4"
    "__q1_centered_cusum_range"
)
RAW_FEATURE_A = (
    "cfg_variance_raw_gap_tile4x4_concentration_guided3__q2_max_positive_jump"
)
OLD_FIXED_SCORE = "fixed_two_phase_predicted_clean_score_label_free_reference"
REFERENCE_CLASSES = (207, 602, 795)
REFERENCE_SEEDS = tuple(range(10, 30))
ROBUST_MAD_FACTOR = 1.4826
ROBUST_SCALE_FLOOR = 1e-6
DESCRIPTIVE_QUANTILES = (0.90, 0.95)
CONFORMAL_ALPHAS = (0.10, 0.05)

E_VALUE_CONTROLS = (
    "weak_conditional_cfg1_full_running_max_log_e_from_E0__full_maximum",
    "weak_conditional_cfg1_tile4x4_uniform_mixture_running_max_log_e_from_E0__full_maximum",
    "weak_unconditional_full_running_max_log_e_from_E0__full_maximum",
    "weak_unconditional_tile4x4_uniform_mixture_running_max_log_e_from_E0__full_maximum",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_product(root: Path, expected_status: str) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"feature product must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if manifest.get("status") != "complete" or completion.get("complete") is not True:
        raise RuntimeError(f"feature product is incomplete: {root}")
    if completion.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"manifest hash mismatch: {root}")
    if completion.get("manifest_identity_sha256") != manifest.get("identity_sha256"):
        raise RuntimeError(f"manifest identity mismatch: {root}")
    by_name = {item.get("name"): item for item in manifest.get("files", [])}
    if len(by_name) != len(manifest.get("files", [])):
        raise RuntimeError(f"duplicate manifest member: {root}")
    for name, item in by_name.items():
        path = root / str(name)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or indirect product member: {path}")
        if item.get("sha256") != sha256_file(path) or item.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"product member changed: {path}")
    summary = load_json(root / "summary.json")
    if summary.get("status") != expected_status:
        raise RuntimeError(
            f"unexpected product semantic status: {summary.get('status')!r} != {expected_status!r}"
        )
    if summary.get("labels_joined", False) is not False:
        raise RuntimeError("candidate freezing must use a label-free feature product")
    if summary.get("supervision_audit", {}).get("labels_read_or_emitted", False) is not False:
        raise RuntimeError("posterior product is not label-free")
    return manifest


def clean_record(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            result[str(key)] = None
        elif isinstance(value, (np.integer,)):
            result[str(key)] = int(value)
        elif isinstance(value, (np.floating,)):
            result[str(key)] = float(value)
        elif isinstance(value, (np.bool_,)):
            result[str(key)] = bool(value)
        else:
            result[str(key)] = value
    return result


def catalog_record(root: Path, feature: str) -> dict[str, Any]:
    catalog = pd.read_csv(root / "feature_catalog.csv")
    match = catalog.loc[catalog["feature"] == feature]
    if len(match) != 1:
        raise RuntimeError(f"expected exactly one catalog row for {feature}: {len(match)}")
    return clean_record(match.iloc[0])


def require_exact_sample_alignment(primary: pd.DataFrame, posterior: pd.DataFrame) -> None:
    keys = ["sample_index", "run_index", "global_seed", "class_slot", "class_id"]
    if len(primary) != 160 or len(posterior) != 160:
        raise RuntimeError("discovery products must each contain exactly 160 samples")
    if not np.array_equal(primary[keys].to_numpy(), posterior[keys].to_numpy()):
        raise RuntimeError("primary and posterior sample orders differ")
    observed_classes = tuple(sorted(int(value) for value in primary["class_id"].unique()))
    if observed_classes != (207, 340, 354, 366, 444, 602, 795, 981):
        raise RuntimeError(f"unexpected discovery classes: {observed_classes}")
    observed_seeds = tuple(sorted(int(value) for value in primary["global_seed"].unique()))
    if observed_seeds != REFERENCE_SEEDS:
        raise RuntimeError(f"unexpected discovery seeds: {observed_seeds}")


def build_lock(primary_root: Path, posterior_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    primary_manifest = validate_product(
        primary_root, "DISCOVERY_ONLY_NOT_AN_INTERVENTION_TRIGGER"
    )
    posterior_manifest = validate_product(
        posterior_root, "COMPLETE_LABEL_FREE_SUPPLEMENTARY_ANALYSIS"
    )
    primary_inventory = load_json(primary_root / "source_inventory.json")
    posterior_inventory = load_json(posterior_root / "source_inventory.json")
    primary_runs = primary_inventory.get("trace_runs")
    posterior_runs = posterior_inventory.get("trace_runs")
    if not isinstance(primary_runs, list) or not primary_runs:
        raise RuntimeError("primary discovery product lacks trace lineage")
    if not isinstance(posterior_runs, list) or len(posterior_runs) != len(primary_runs):
        raise RuntimeError("posterior discovery product lacks matching trace lineage")
    lineage_fields = (
        "cfg_epsilon_channels",
        "cfg_scale",
        "classes",
        "completion_sha256",
        "global_seed",
        "identity_sha256",
        "manifest_sha256",
        "source_snapshot_sha256",
        "trace_sha256",
    )
    if [
        {key: run.get(key) for key in lineage_fields} for run in primary_runs
    ] != [
        {key: run.get(key) for key in lineage_fields} for run in posterior_runs
    ]:
        raise RuntimeError("primary/posterior discovery products bind different trace cohorts")
    sampler_source_snapshot = primary_runs[0].get("source_snapshot_sha256")
    if (
        primary_runs[0].get("cfg_scale") != 4.0
        or primary_runs[0].get("cfg_epsilon_channels") != 3
        or not isinstance(sampler_source_snapshot, dict)
        or any(run.get("source_snapshot_sha256") != sampler_source_snapshot for run in primary_runs)
    ):
        raise RuntimeError("discovery sampler contract is not the expected frozen CFG=4 contract")
    primary = pd.read_csv(primary_root / "sample_features.csv")
    posterior = pd.read_csv(posterior_root / "sample_features.csv")
    require_exact_sample_alignment(primary, posterior)
    required_primary = {FEATURE_B, RAW_FEATURE_A, OLD_FIXED_SCORE}
    required_posterior = {FEATURE_A, *E_VALUE_CONTROLS}
    if missing := sorted(required_primary - set(primary.columns)):
        raise RuntimeError(f"primary product lacks frozen columns: {missing}")
    if missing := sorted(required_posterior - set(posterior.columns)):
        raise RuntimeError(f"posterior product lacks frozen columns: {missing}")

    keep = ["sample_index", "run_index", "global_seed", "class_slot", "class_id"]
    reference = primary[keep + [FEATURE_B, RAW_FEATURE_A, OLD_FIXED_SCORE]].copy()
    reference[FEATURE_A] = posterior[FEATURE_A].to_numpy(dtype=np.float64)
    for feature in E_VALUE_CONTROLS:
        reference[feature] = posterior[feature].to_numpy(dtype=np.float64)
    reference = reference.loc[reference["class_id"].isin(REFERENCE_CLASSES)].copy()
    if len(reference) != len(REFERENCE_CLASSES) * len(REFERENCE_SEEDS):
        raise RuntimeError("reference subset is not the expected 3 classes x 20 seeds")
    if not np.isfinite(reference.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("reference table contains non-finite numbers")

    class_reference: dict[str, Any] = {}
    score_parts: list[pd.DataFrame] = []
    for class_id in REFERENCE_CLASSES:
        part = reference.loc[reference["class_id"] == class_id].copy()
        if tuple(sorted(int(value) for value in part["global_seed"])) != REFERENCE_SEEDS:
            raise RuntimeError(f"class {class_id} does not contain the frozen reference seeds")
        oriented_a = -part[FEATURE_A].to_numpy(dtype=np.float64)
        oriented_b = part[FEATURE_B].to_numpy(dtype=np.float64)
        statistics: dict[str, dict[str, float]] = {}
        for short_name, values in (("A_low_is_bad", oriented_a), ("B_high_is_bad", oriented_b)):
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = float(max(ROBUST_MAD_FACTOR * mad, ROBUST_SCALE_FLOOR))
            statistics[short_name] = {"median": median, "mad": mad, "scale": scale}
        part["z_A_low_is_bad"] = (
            oriented_a - statistics["A_low_is_bad"]["median"]
        ) / statistics["A_low_is_bad"]["scale"]
        part["z_B_high_is_bad"] = (
            oriented_b - statistics["B_high_is_bad"]["median"]
        ) / statistics["B_high_is_bad"]["scale"]
        part["S_AND"] = np.minimum(part["z_A_low_is_bad"], part["z_B_high_is_bad"])
        descriptive_thresholds = {
            f"q{int(100 * quantile):02d}_linear": float(
                np.quantile(part["S_AND"], quantile, method="linear")
            )
            for quantile in DESCRIPTIVE_QUANTILES
        }
        sorted_scores = np.sort(part["S_AND"].to_numpy(dtype=np.float64))
        conformal_thresholds: dict[str, Any] = {}
        for alpha in CONFORMAL_ALPHAS:
            order = int(np.ceil((len(sorted_scores) + 1) * (1.0 - alpha)))
            order = min(max(order, 1), len(sorted_scores))
            conformal_thresholds[f"alpha_{alpha:.2f}".replace(".", "p")] = {
                "threshold": float(sorted_scores[order - 1]),
                "calibration_order_statistic_1_based": order,
                "strict_comparison": "new_score > threshold",
                "finite_sample_marginal_trigger_probability_upper_bound": float(
                    (len(sorted_scores) + 1 - order) / (len(sorted_scores) + 1)
                ),
            }
        class_reference[str(class_id)] = {
            "sample_count": int(len(part)),
            "statistics": statistics,
            "S_AND_descriptive_linear_quantiles_no_error_rate_guarantee": descriptive_thresholds,
            "S_AND_split_conformal_thresholds": conformal_thresholds,
        }
        score_parts.append(part)
    reference = pd.concat(score_parts, ignore_index=True).sort_values(
        ["global_seed", "class_slot"]
    )

    protocol: dict[str, Any] = {
        "schema_version": 2,
        "status": "FROZEN_BEFORE_FRESH_CONFIRMATION_GENERATION_AND_LABEL_JOIN",
        "objective": (
            "Prospectively test a preterminal, model-relative score for obvious blur, "
            "fusion, gross topology, and major limb/object misalignment failures."
        ),
        "discovery_role": "hypothesis_generation_only",
        "fresh_confirmation": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "cfg_scale": 4.0,
            "classes": list(REFERENCE_CLASSES),
            "seeds": {"start_inclusive": 30, "stop_inclusive": 129, "count": 100},
            "trajectory_count": 300,
            "visual_labels_must_be_locked_before_score_join": True,
            "metric_visibility_to_reviewers": "forbidden",
            "expansion_or_stopping_rule": (
                "The 300-path cohort is fixed. Any later expansion must use disjoint seeds "
                "and be decided only from locked clear-bad event count or the confidence width "
                "of the visual-label prevalence estimate, never detector precision or observed score performance."
            ),
        },
        "sampler_lineage_contract": {
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "source_snapshot_sha256": sampler_source_snapshot,
            "fresh_primary_and_posterior_trace_identity_lists_must_match_exactly": True,
            "checkpoint_and_vae_validation": (
                "enforced by the hash-frozen strict reproduction and custom trace helpers"
            ),
        },
        "primary_candidate": {
            "name": "S_AND",
            "formula": "min(z_A, z_B)",
            "z_A": "(-A - class_median_reference(-A)) / class_robust_scale_reference(-A)",
            "z_B": "(B - class_median_reference(B)) / class_robust_scale_reference(B)",
            "orientation": "higher_is_more_bad_like",
            "motivation": (
                "Require both abnormal failure to spatially localize conditional-vs-unconditional "
                "posterior log-standard-deviation disagreement in q2 and abnormal q1 branch "
                "disagreement regime change in the deliberately unguided fourth epsilon channel."
            ),
            "latest_required_sampling_step": 149,
            "latest_required_internal_timestep": 100,
            "preterminal_actionable": True,
            "selection_warning": "Post-hoc on discovery labels; only fresh confirmation is inferential.",
            "alert_budgets": {
                "primary": "strictly greater than frozen class-specific split-conformal alpha_0p10 threshold",
                "secondary": "strictly greater than frozen class-specific split-conformal alpha_0p05 threshold",
                "thresholds_use_no_visual_labels": True,
                "guarantee": (
                    "Under exchangeability with the class-specific 20-sample calibration set, "
                    "strict exceedance has marginal trigger probability at most 2/21 for "
                    "alpha_0p10 and 1/21 for alpha_0p05; ties make it conservative."
                ),
                "not_a_good_image_fpr": True,
                "descriptive_linear_quantiles": (
                    "also recorded for audit only; they carry no finite-sample error-rate guarantee"
                ),
            },
        },
        "single_feature_backups": {
            "A": {
                "feature": FEATURE_A,
                "orientation": "lower_is_more_bad_like",
                "catalog": catalog_record(posterior_root, FEATURE_A),
            },
            "B": {
                "feature": FEATURE_B,
                "orientation": "higher_is_more_bad_like",
                "catalog": catalog_record(primary_root, FEATURE_B),
            },
        },
        "negative_controls": {
            "old_fixed_predicted_clean_score": OLD_FIXED_SCORE,
            "exact_path_evidence_running_maxima": list(E_VALUE_CONTROLS),
            "exact_e_value_alert_log_thresholds": {
                "alpha_0p10": float(np.log(10.0)),
                "alpha_0p05": float(np.log(20.0)),
            },
        },
        "normalization": {
            "reference_classes": list(REFERENCE_CLASSES),
            "reference_seeds": list(REFERENCE_SEEDS),
            "reference_is_label_free": True,
            "median_absolute_deviation_factor": ROBUST_MAD_FACTOR,
            "scale_formula": "max(1.4826 * MAD, 1e-6)",
            "scale_floor": ROBUST_SCALE_FLOOR,
            "class_reference": class_reference,
        },
        "evaluation": {
            "positive": "2-of-3 blind reviewers rate clear bad (severity 2 or 3), after conservative adjudication",
            "negative": "2-of-3 blind reviewers rate ordinary/clean (severity 0)",
            "excluded": "mild, disputed, uncertain, or adjudication downgrade",
            "primary_continuous_statistic": {
                "name": "class-matched pair-weighted ROC AUC of S_AND",
                "formula": (
                    "sum_c sum_{b in bad_c,g in good_c} [I(S_b>S_g)+0.5 I(S_b=S_g)] "
                    "/ sum_c (n_bad_c*n_good_c), over classes containing both labels"
                ),
            },
            "secondary_continuous_statistic": {
                "name": "macro mean of within-class ROC AUC",
                "formula": "unweighted arithmetic mean over classes containing both labels",
            },
            "primary_fixed_operating_point": "frozen class-specific split-conformal alpha_0p10 alert",
            "primary_randomization_test": {
                "null": "S_AND is exchangeable with the binary visual label within each class",
                "unit": "sample within class among clear_bad and clean_good only",
                "constraint": "preserve each class's observed clear_bad and clean_good counts",
                "statistic": "class-matched pair-weighted ROC AUC; orientation is frozen higher-is-bad",
                "draws": 100000,
                "rng": "numpy.random.Generator(PCG64(seed=2026082701))",
                "p_value": "(1 + number(permuted_statistic >= observed_statistic)) / 100001",
                "sidedness": "one-sided",
            },
            "uncertainty_intervals": {
                "TPR_and_FPR": "two-sided 95% Clopper-Pearson exact binomial intervals, reported separately",
                "TPR_minus_FPR": (
                    "95% percentile interval from 100000 global-seed cluster bootstrap draws "
                    "using numpy PCG64(seed=2026082702); each draw resamples the 100 seed clusters "
                    "with replacement and retains all three class rows"
                ),
                "auc": (
                    "95% percentile interval from 100000 within-(class,label) stratified bootstrap "
                    "draws using numpy PCG64(seed=2026082703)"
                ),
            },
            "report": [
                "TPR and FPR with exact binomial intervals",
                "TPR minus FPR with global-seed cluster bootstrap interval",
                "per-class bad ranks and per-class AUC",
                "A and B separately to test conjunction mechanism",
            ],
            "time_to_signal": {
                "status": "EXPLORATORY_NOT_PART_OF_CONFIRMATION_GATE",
                "reason": (
                    "The frozen scalar normalizers apply to the full q1/q2 reductions; no "
                    "prefix-specific normalization or alert boundary was selected before confirmation."
                ),
                "allowed_claim": "latest full score is available at sampling step 149 / internal t=100",
            },
            "initial_go_gate": {
                "minimum_clear_bad_events_for_decision": 15,
                "S_AND_class_matched_auc_at_least": 0.75,
                "S_AND_stratified_permutation_one_sided_p_below": 0.05,
                "alpha_0p10_TPR_minus_FPR_point_above": 0.0,
                "no_class_with_two_or_more_bad_events_has_auc_below": 0.60,
            },
            "interpretation_if_fewer_than_15_bad_events": "pilot only; expand using disjoint seeds without changing formulas",
        },
        "source_products": {
            "primary_label_free": {
                "path": str(primary_root.resolve()),
                "manifest_identity_sha256": primary_manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(primary_root / "manifest.json"),
                "sample_features_sha256": sha256_file(primary_root / "sample_features.csv"),
                "label_free_reference_stats_sha256": sha256_file(
                    primary_root / "label_free_reference_stats.npz"
                ),
                "analysis_source_sha256": primary_manifest["analysis_source_sha256"],
            },
            "posterior_label_free": {
                "path": str(posterior_root.resolve()),
                "manifest_identity_sha256": posterior_manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(posterior_root / "manifest.json"),
                "sample_features_sha256": sha256_file(posterior_root / "sample_features.csv"),
                "analysis_source_sha256": posterior_manifest["analysis_source_sha256"],
            },
        },
        "claims_forbidden_before_confirmation_unseal": [
            "validated detector",
            "bad-case posterior probability",
            "authorization to guide, reject, or roll back",
            "cross-class or cross-model generalization",
        ],
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol, reference


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def publish(primary_root: Path, posterior_root: Path, output: Path) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite candidate lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        protocol, reference = build_lock(primary_root.resolve(), posterior_root.resolve())
        write_json(staging / "candidate_protocol.json", protocol)
        reference.to_csv(staging / "label_free_reference_rows.csv", index=False)
        shutil.copy2(Path(__file__).resolve(), staging / "locker_source.py")
        members = []
        for path in sorted(staging.iterdir()):
            members.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
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
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "protocol_file_sha256": sha256_file(staging / "candidate_protocol.json"),
            "protocol_identity_sha256": protocol["identity_sha256"],
        }
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--posterior-root", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = publish(args.primary_root, args.posterior_root, args.output)
    protocol = load_json(output / "candidate_protocol.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": protocol["status"],
                "protocol_identity_sha256": protocol["identity_sha256"],
                "confirmation_trajectory_count": protocol["fresh_confirmation"]["trajectory_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
