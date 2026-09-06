#!/usr/bin/env python3
"""Analyze the frozen global zero-anchor correction on all 100 heldout steps.

The primary certificate is H=(1+lambda)^2 E||T||^2-4 E||Y||^2.
For 0<lambda<1, H>=0 suffices for same-input single-step coupling and
marginal W2 nonincrease. Inactive steps pass only by checked exact identity.
The analysis never changes a coefficient, selects a time window, or uses FID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch

PROTOCOL = "raev2_global_proximal_real_coupling_audit_v1"
METHOD = "global_zero_anchor"
NUM_STEPS = 100
NUM_SAMPLES = 1000
CONFIDENCE_ALPHA = 0.05
FAMILY_SIZE = 100  # Fixed before inspecting C; never reduced to active-step count.
PRIMARY = "w2_nonincrease_certificate"
STRONGER = "w2_nonovershoot_certificate"
METRICS = (
    PRIMARY, STRONGER, "risk_gain", "J", "nonexpansive_slack",
    "risk_before", "risk_after", "applied_correction_mse", "h_target_mse",
    "target_second_moment", "predicted_second_moment", "residual_cross_moment",
    "diagonal_risk_after", "diagonal_risk_gain", "diagonal_J",
)
IDENTITIES = ("step_index", "sample_id", "source_row", "label")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def bool_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    raise ValueError(f"invalid active flag: {value!r}")


def confidence(values: np.ndarray, z: float) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {"mean": mean, "se": se, "pointwise_lower": mean - 1.96 * se,
            "pointwise_upper": mean + 1.96 * se,
            "simultaneous_lower": mean - z * se, "simultaneous_upper": mean + z * se}


def analyze(frame: pd.DataFrame, step_metadata: pd.DataFrame, *,
            expected_steps: int = NUM_STEPS, expected_samples: int = NUM_SAMPLES) -> tuple[pd.DataFrame, dict]:
    """Pure CPU statistical analysis; small expected sizes are for unit checks."""
    required = set(IDENTITIES) | set(METRICS)
    if not required.issubset(frame.columns):
        raise ValueError(f"missing per-image columns: {required - set(frame.columns)}")
    if not {"step_index", "effective_lambda", "active"}.issubset(step_metadata.columns):
        raise ValueError("step metadata must contain effective_lambda and active")
    if set(frame.step_index.unique()) != set(range(expected_steps)):
        raise ValueError("all expected time steps must be retained")
    if (step_metadata.step_index.duplicated().any()
            or set(step_metadata.step_index) != set(range(expected_steps))):
        raise ValueError("step metadata is incomplete or duplicated")
    if frame.duplicated(["step_index", "sample_id"]).any():
        raise ValueError("duplicate time/sample rows")
    if not np.isfinite(frame[list(required)].to_numpy(dtype=np.float64)).all():
        raise ValueError("nonfinite identity or metric")
    for key in IDENTITIES:
        if not np.equal(frame[key], np.floor(frame[key])).all():
            raise ValueError(f"noninteger identity: {key}")
    if (frame.groupby("sample_id")[["source_row", "label"]].nunique() != 1).any().any():
        raise ValueError("sample identities change across time")
    for _, group in frame.groupby("step_index"):
        ordered = group.sort_values("sample_id")
        if len(group) != expected_samples or not np.array_equal(ordered.sample_id, np.arange(expected_samples)):
            raise ValueError("each step must use the same complete ordered sample bank")
        if not np.array_equal(ordered.label, np.arange(expected_samples)):
            raise ValueError("heldout bank must have one image per class in label order")
        if ordered.source_row.nunique() != expected_samples:
            raise ValueError("heldout rows must be unique")
    nonnegative = ("risk_before", "risk_after", "applied_correction_mse", "h_target_mse",
                   "target_second_moment", "predicted_second_moment", "diagonal_risk_after")
    if (frame[list(nonnegative)] < 0).any().any():
        raise ValueError("negative squared norm or risk")

    z = NormalDist().inv_cdf(1 - CONFIDENCE_ALPHA / (2 * FAMILY_SIZE))
    rows = []
    metadata = step_metadata.set_index("step_index")
    max_direct_formula_difference = {PRIMARY: 0.0, STRONGER: 0.0}
    for index, group in frame.groupby("step_index", sort=True):
        item = metadata.loc[index]
        lam = float(item.effective_lambda)
        active = bool_value(item.active)
        if not math.isfinite(lam) or not 0 < lam <= 1 or active != (lam < 1):
            raise ValueError(f"invalid effective scalar map at step {index}")
        second_difference = 2 * group.residual_cross_moment.to_numpy() + group.risk_before.to_numpy()
        b = group.target_second_moment.to_numpy()
        a = group.predicted_second_moment.to_numpy()
        factors = {PRIMARY: (1 + lam) ** 2, STRONGER: lam ** 2}
        subtract = {PRIMARY: 4.0, STRONGER: 1.0}
        for key in (PRIMARY, STRONGER):
            stable = (factors[key] - subtract[key]) * b + factors[key] * second_difference
            saved = group[key].to_numpy()
            # This checks formula/CSV consistency, not a statistical tolerance.
            roundoff = 32 * np.finfo(np.float64).eps * (1 + np.abs(b) + np.abs(second_difference))
            if np.any(np.abs(stable - saved) > roundoff):
                raise ValueError(f"saved {key} does not match the frozen lambda at step {index}")
            direct = factors[key] * a - subtract[key] * b
            max_direct_formula_difference[key] = max(max_direct_formula_difference[key],
                                                     float(np.max(np.abs(direct - saved))))
        identity = bool((group.risk_gain == 0).all()
                        and (group.applied_correction_mse == 0).all()
                        and (group.risk_before == group.risk_after).all())
        if not active and not identity:
            raise ValueError(f"inactive step {index} is not the exact applied identity")
        row = {"step_index": int(index), "effective_lambda": lam, "active": active,
               "samples": len(group), "identity_verified": bool(not active and identity)}
        for key in ("time", "next_time", "slope_mean", "slope_max", "train_J"):
            if key in item:
                row[key] = float(item[key])
        for key in METRICS:
            row.update({f"{key}_{name}": value for name, value in confidence(group[key], z).items()})
        row["primary_certificate_pass"] = bool(not active or row[PRIMARY + "_simultaneous_lower"] > 0)
        row["stronger_certificate_pass"] = bool(not active or row[STRONGER + "_simultaneous_lower"] > 0)
        row["admission_basis"] = "positive_primary_simultaneous_lower" if active and row["primary_certificate_pass"] else (
            "exact_applied_identity" if not active else "primary_certificate_unconfirmed")
        rows.append(row)
    steps = pd.DataFrame(rows)
    active = steps[steps.active]
    primary_pass = bool(steps.primary_certificate_pass.all())
    identity_pass = bool(steps.loc[~steps.active, "identity_verified"].all())
    eligible = bool(primary_pass and identity_pass and len(active) > 0)
    summary = {
        "protocol": "raev2_global_proximal_heldout_analysis_v1", "method": METHOD,
        "complete": True, "steps": expected_steps, "samples": expected_samples,
        "all_official_steps_retained": True, "active_step_count": len(active),
        "inactive_step_count": int((~steps.active).sum()), "inactive_identity_checks_pass": identity_pass,
        "primary_certificate": {"name": PRIMARY, "formula": "(1+lambda)^2 E||T||^2 - 4 E||Y||^2",
            "active_positive_simultaneous_lower_count": int((active[PRIMARY + "_simultaneous_lower"] > 0).sum()),
            "active_negative_simultaneous_upper_count": int((active[PRIMARY + "_simultaneous_upper"] < 0).sum()),
            "unconfirmed_active_steps": active.loc[~active.primary_certificate_pass, "step_index"].astype(int).tolist(),
            "all_steps_pass": primary_pass},
        "stronger_secondary_certificate": {"name": STRONGER, "formula": "lambda^2 E||T||^2 - E||Y||^2",
            "active_positive_simultaneous_lower_count": int((active[STRONGER + "_simultaneous_lower"] > 0).sum()),
            "used_for_admission_or_method_selection": False},
        "eligible_for_fixed_1k_test": eligible,
        "admission_status": "eligible_by_approximate_heldout_certificate" if eligible else (
            "identity_only_no_active_candidate" if len(active) == 0 else "certificate_not_established"),
        "statistical_protocol": {
            "alpha": CONFIDENCE_ALPHA, "bonferroni_family_size": FAMILY_SIZE, "two_sided_z": z,
            "primary_family": "all 100 primary-certificate intervals, including analytically inactive steps",
            "secondary_intervals": "each metric has its own 100-step family; no joint coverage across metric families is claimed",
            "units": "independent heldout images, one image per each fixed class; all 100 times remain paired",
            "se": "pooled sample standard deviation / sqrt(number of images)",
            "class_balance_limit": "For fixed classes, pooled SE includes between-class mean heterogeneity and is conservative in expectation; within-class variance is not separately identifiable with one image per class.",
            "fit_uncertainty": "inference is conditional on the independently fitted and frozen bank-A coefficients",
            "coverage_limit": "Bonferroni normal approximation, requiring appropriate moment/tail and sampling conditions; not a distribution-free finite-sample guarantee",
        },
        "numerical_checks": {"stable_certificate_formula_matches": True,
            "maximum_absolute_stable_vs_direct_fp32_moment_discrepancy": max_direct_formula_difference,
            "inactive_criterion": "effective_lambda==1; each image has exactly zero actual gain/correction and identical before/after risk; no positive significance requirement",
            "arithmetic_limit": "certificates use recorded stable FP64 combinations of FP32 products; theorem refers to the exact scalar map and reported inference does not replace a numerical error bound"},
        "coupling_diagnostics": {"active_D_positive_simultaneous_lower_count": int((active.risk_gain_simultaneous_lower > 0).sum()),
            "active_D_negative_simultaneous_upper_count": int((active.risk_gain_simultaneous_upper < 0).sum()),
            "active_J_positive_simultaneous_lower_count": int((active.J_simultaneous_lower > 0).sum()),
            "minimum_nonexpansive_slack": float(frame.nonexpansive_slack.min())},
        "original_diagonal_secondary": {"retained": True, "used_for_selection": False,
            "purpose": "replication of previously failed diagonal calibration, never revival based on C",
            "negative_D_simultaneous_upper_count": int((steps.diagonal_risk_gain_simultaneous_upper < 0).sum())},
        "coefficients_changed": False, "time_window_selected": False, "fid_evaluated": False,
        "claim_limit": "Finite-estimate heldout evidence for the same-input single-step global scalar map only; no rollout, endpoint W2, FID or 5% improvement claim.",
        "negative_certificate_interpretation": "Failure to certify the sufficient condition does not establish actual marginal W2 degradation.",
    }
    return steps, summary


def discover_inputs(root: Path) -> list[Path]:
    if (root / "heldout_per_image.csv").exists():
        return [root]
    shards = sorted(path for path in root.glob("shard_*_of_*") if path.is_dir())
    if shards:
        return shards
    if (root / "merged/heldout_per_image.csv").exists():
        return [root / "merged"]
    raise FileNotFoundError(f"no audit shards or merged data under {root}")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames, step_frames, requests, summaries, records = [], [], [], [], []
    used_steps = set()
    for directory in discover_inputs(root):
        required = [directory / name for name in ("heldout_per_image.csv", "calibration.pt", "summary.json")]
        if not all(path.exists() for path in required):
            raise ValueError(f"incomplete audit: {directory}")
        summary = json.loads((directory / "summary.json").read_text())
        payload = torch.load(directory / "calibration.pt", map_location="cpu", weights_only=True)
        local_requests = [payload["request"]] if "request" in payload else payload.get("source_requests", [])
        if not local_requests or any(r.get("protocol") != PROTOCOL or r.get("fit_structure") != METHOD for r in local_requests):
            raise ValueError("calibration identity is not the frozen global zero-anchor method")
        if "request" in payload:
            if not summary.get("complete"):
                raise ValueError(f"incomplete shard: {directory}")
            disk_request = json.loads((directory / "request.json").read_text())
            if disk_request != payload["request"]:
                raise ValueError("request file differs from the calibrated payload")
        else:
            for source in payload.get("source_shards", []):
                source_dir = Path(source["path"])
                source_summary = json.loads((source_dir / "summary.json").read_text())
                if not source_summary.get("complete"):
                    raise ValueError("merged payload contains an incomplete source shard")
                summaries.append(source_summary)
        calibration_hash = sha256(directory / "calibration.pt")
        if summary.get("calibration_sha256") != calibration_hash:
            raise ValueError("calibration SHA does not match its recorded summary")
        requests.extend(local_requests)
        if "heldout_noise_sha256" in summary:
            summaries.append(summary)
        frame = pd.read_csv(directory / "heldout_per_image.csv", float_precision="round_trip")
        applied = payload.get("applied_fp32_steps", {})
        if set(map(int, applied)) != set(map(int, payload["steps"])):
            raise ValueError("applied coefficients do not cover the fitted time steps")
        extracted = []
        for raw_index, fit in applied.items():
            index = int(raw_index)
            if index in used_steps:
                raise ValueError(f"duplicate calibration step {index}")
            used_steps.add(index)
            slope = fit["slope"]
            scalar = slope.reshape(-1)[0]
            if (slope.dtype != torch.float32 or not torch.isfinite(slope).all()
                    or bool(scalar < 0) or not torch.equal(slope, scalar.expand_as(slope))
                    or bool((fit["center"] != 0).any()) or bool((fit["offset"] != 0).any())):
                raise ValueError("applied calibration is not a finite global scalar with zero anchor")
            if int(fit["count"]) != NUM_SAMPLES:
                raise ValueError("unexpected number of calibration images")
            lam = 1.0 / float((1 + scalar).item())
            extracted.append({"step_index": index, "effective_lambda": lam, "active": lam < 1.0})
        extracted = pd.DataFrame(extracted).sort_values("step_index")
        if (directory / "step_metrics.csv").exists():
            steps = pd.read_csv(directory / "step_metrics.csv", float_precision="round_trip").sort_values("step_index")
            if not np.array_equal(steps.step_index.to_numpy(), extracted.step_index.to_numpy()):
                raise ValueError("step CSV does not match applied calibration")
            if not np.array_equal(steps.effective_lambda.to_numpy(), extracted.effective_lambda.to_numpy()):
                raise ValueError("effective lambda differs from the applied FP32 denominator")
            if [bool_value(value) for value in steps.active] != extracted.active.tolist():
                raise ValueError("step activity differs from the applied calibration")
        else:
            steps = extracted
            grid = local_requests[0]["time_grid"]
            steps["time"] = [grid[i] for i in steps.step_index]
            steps["next_time"] = [grid[i + 1] for i in steps.step_index]
        frames.append(frame)
        step_frames.append(steps)
        record_paths = required + ([directory / "step_metrics.csv"] if (directory / "step_metrics.csv").exists() else [])
        if (directory / "request.json").exists():
            record_paths.append(directory / "request.json")
        records.append({"directory": str(directory), "artifacts": [artifact(path) for path in record_paths]})
        del payload
    if used_steps != set(range(NUM_STEPS)):
        raise ValueError("the complete official 100-step calibration is required")
    compare_keys = ("checkpoint_sha256", "config_sha256", "normalization_stats_sha256", "source_sha256",
                    "train_bank_metadata", "heldout_bank_metadata", "parent_calibration_sha256", "time_grid",
                    "noise_seed_train", "noise_seed_heldout", "noise_schema", "precision", "batch_size",
                    "forward_layout", "tf32", "ig_scale", "ig_interval", "fit_structure")
    for key in compare_keys:
        if any(key not in r for r in requests) or any(r[key] != requests[0][key] for r in requests):
            raise ValueError(f"audit sources disagree or omit {key}")
    for key in ("heldout_noise_sha256", "train_noise_sha256"):
        if not summaries or any(key not in s for s in summaries) or len({s[key] for s in summaries}) != 1:
            raise ValueError(f"audit sources disagree or omit {key}")
    first = requests[0]
    if (first["ig_scale"] != 1.78 or first["ig_interval"] != [0.1, 1.0]
            or first.get("num_steps") != NUM_STEPS or first.get("correction_multiplier") != 1.0
            or first.get("correction_time_selection") != "none"):
        raise ValueError("guidance or correction protocol differs from the frozen official setup")
    provenance = {"method_identity_checked": True, "method": METHOD,
        "calibration_hashes_match_recorded_summaries": True, "all_recorded_source_identities_match": True,
        "inputs": records, "frozen_source_identity": {key: first[key] for key in compare_keys},
        "heldout_noise_sha256": summaries[0]["heldout_noise_sha256"],
        "train_noise_sha256": summaries[0]["train_noise_sha256"],
        "source_hash_scope": "input CSV/request/calibration artifacts were hashed; recorded checkpoint/config/source identities compared across all shards, without rehashing model weights"}
    return pd.concat(frames, ignore_index=True), pd.concat(step_frames, ignore_index=True), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    frame, metadata, provenance = load_inputs(args.audit_root.expanduser().resolve())
    steps, summary = analyze(frame, metadata)
    out.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["step_index", "sample_id"]).to_csv(out / "heldout_per_image.csv", index=False)
    metadata.sort_values("step_index").to_csv(out / "input_step_metrics.csv", index=False)
    steps.to_csv(out / "step_certificates.csv", index=False)
    summary["provenance"] = provenance
    summary["analyzer"] = artifact(Path(__file__))
    summary["output_artifacts"] = [artifact(out / name) for name in
                                   ("heldout_per_image.csv", "input_step_metrics.csv", "step_certificates.csv")]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("provenance", "output_artifacts")}, indent=2))


if __name__ == "__main__":
    main()
