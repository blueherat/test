#!/usr/bin/env python3
"""Merge complete time shards and assess a fixed proximal calibration.

No coefficient, sample, time interval, or negative result is discarded.
FID is deliberately absent: these are independent teacher-coupling checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    return {"mean": mean, "se": se, "lower": mean-1.96*se, "upper": mean+1.96*se}


def analyze(frame: pd.DataFrame, expected_steps: int) -> tuple[pd.DataFrame, dict]:
    required = {"step_index", "sample_id", "source_row", "label", "risk_before", "risk_after", "J", "risk_gain", "nonexpansive_slack"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {required-set(frame.columns)}")
    if set(frame.step_index.unique()) != set(range(expected_steps)):
        raise ValueError("incomplete or extra time steps; all official steps required")
    if frame.duplicated(["step_index", "sample_id"]).any():
        raise ValueError("duplicate time/sample rows")
    sizes = frame.groupby("step_index").size()
    if sizes.nunique() != 1 or sizes.iloc[0] < 2:
        raise ValueError("all steps must contain the same heldout sample bank")
    banks = [set(group.sample_id) for _, group in frame.groupby("step_index")]
    if any(bank != banks[0] for bank in banks):
        raise ValueError("heldout identities differ across time")
    if (frame.groupby("sample_id")[["source_row", "label"]].nunique() != 1).any().any():
        raise ValueError("heldout source rows or labels change across time")
    for key in required-{"sample_id", "step_index", "source_row", "label"}:
        if not np.isfinite(frame[key]).all():
            raise ValueError(f"nonfinite {key}")
    before = frame.pivot(index="sample_id", columns="step_index", values="risk_before").to_numpy()
    after = frame.pivot(index="sample_id", columns="step_index", values="risk_after").to_numpy()
    if np.min(before) < 0 or np.min(after) < 0:
        raise ValueError("negative squared risk")
    mb, ma = before.mean(0), after.mean(0)
    sb, sa = np.sqrt(mb), np.sqrt(ma)
    source_before, source_after = float(sb.sum()), float(sa.sum())
    # Delta-method SE with whole images as units: all times share X and epsilon.
    denom_before, denom_after = np.where(sb > 0, 2*sb, np.inf), np.where(sa > 0, 2*sa, np.inf)
    influence = ((before-mb)/denom_before-(after-ma)/denom_after).sum(1)
    gain = source_before-source_after
    se = float(influence.std(ddof=1)/np.sqrt(len(influence)))
    step_rows = []
    for index, group in frame.groupby("step_index", sort=True):
        row = {"step_index": int(index), "risk_before": float(group.risk_before.mean()),
               "risk_after": float(group.risk_after.mean()), "samples": len(group)}
        for key in ("risk_gain", "J", "nonexpansive_slack"):
            row.update({f"{key}_{name}": value for name, value in ci95(group[key]).items()})
        step_rows.append(row)
    step_frame = pd.DataFrame(step_rows)
    simultaneous_z = NormalDist().inv_cdf(1 - 0.05 / (2 * expected_steps))
    step_frame["risk_gain_simultaneous_lower"] = step_frame.risk_gain_mean - simultaneous_z * step_frame.risk_gain_se
    step_frame["risk_gain_simultaneous_upper"] = step_frame.risk_gain_mean + simultaneous_z * step_frame.risk_gain_se
    summary = {
        "protocol": "raev2_proximal_calibration_heldout_v1",
        "all_official_steps_retained": True, "steps": expected_steps, "samples": int(sizes.iloc[0]),
        "step_risk_gain_positive_count": int((step_frame.risk_gain_mean > 0).sum()),
        "step_J_positive_count": int((step_frame.J_mean > 0).sum()),
        "step_gain_ci_positive_count": int((step_frame.risk_gain_lower > 0).sum()),
        "step_gain_ci_negative_count": int((step_frame.risk_gain_upper < 0).sum()),
        "step_gain_bonferroni_positive_count": int((step_frame.risk_gain_simultaneous_lower > 0).sum()),
        "step_gain_bonferroni_negative_count": int((step_frame.risk_gain_simultaneous_upper < 0).sum()),
        "simultaneous_intervals": "95% Bonferroni normal approximation across all steps; not a distribution-free finite-sample guarantee",
        "minimum_nonexpansive_slack": float(frame.nonexpansive_slack.min()),
        "unweighted_local_rms_source_sum": {
            "before": source_before, "after": source_after, "gain": gain,
            "gain_relative": gain/source_before if source_before else 0.,
            "gain_se": se, "gain_ci95": [gain-1.96*se, gain+1.96*se],
            "interpretation": "sum_k sqrt(E coupling step error); not the full W2 bound without baseline map Lipschitz factors",
        },
        "statistical_units": "independent images, one per class, with all times kept together; pooled between-image SE and asymptotic delta-method CI; within-class variance not separately identifiable",
        "all_step_mean_risk_nonincreasing": bool((step_frame.risk_gain_mean >= 0).all()),
        "no_fid_evaluated": True,
        "claim_limit": "Finite-sample teacher-coupling evidence, not proof of endpoint FID improvement or population per-step positivity.",
    }
    return step_frame, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-steps", type=int, default=100)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    frames, requests, summaries, calibrations, provenance = [], [], [], {}, []
    applied = {}
    for shard in args.shard:
        shard = shard.resolve()
        request = json.loads((shard/"request.json").read_text())
        run_summary = json.loads((shard/"summary.json").read_text())
        if not run_summary.get("complete"):
            raise ValueError(f"incomplete shard: {shard}")
        summaries.append(run_summary)
        requests.append(request)
        frames.append(pd.read_csv(shard/"heldout_per_image.csv"))
        payload = torch.load(shard/"calibration.pt", map_location="cpu", weights_only=True)
        for raw_index, fit in payload["steps"].items():
            index = int(raw_index)
            if index in calibrations:
                raise ValueError(f"duplicate calibration step {index}")
            if not bool(torch.isfinite(fit["slope"]).all()) or bool((fit["slope"] < 0).any()):
                raise ValueError("calibration must have finite nonnegative slopes")
            calibrations[index] = fit
        for raw_index, fit in payload.get("applied_fp32_steps", {}).items():
            applied[int(raw_index)] = fit
        provenance.append({"path": str(shard), "request_sha256": sha256(shard/"request.json"),
                           "calibration_sha256": sha256(shard/"calibration.pt"),
                           "heldout_csv_sha256": sha256(shard/"heldout_per_image.csv")})
    # Only shard membership and output paths may vary; source/bank identities
    # are additionally retained for explicit review instead of guessed aliases.
    for key in ("checkpoint_sha256", "config_sha256", "time_grid", "source_sha256",
                "train_bank_metadata", "heldout_bank_metadata", "normalization_stats_sha256",
                "noise_seed_train", "noise_seed_heldout", "noise_schema", "precision",
                "batch_size", "forward_layout", "tf32", "ig_scale", "ig_interval"):
        values = [request[key] for request in requests]
        if any(value != values[0] for value in values):
            raise ValueError(f"shards differ in {key}")
    for key in ("train_noise_sha256", "heldout_noise_sha256"):
        values = [summary[key] for summary in summaries]
        if any(value != values[0] for value in values):
            raise ValueError(f"shards differ in {key}")
    if set(calibrations) != set(range(args.expected_steps)):
        raise ValueError("not all calibration steps are available")
    if set(applied) != set(calibrations):
        raise ValueError("applied FP32 coefficients are incomplete")
    frame = pd.concat(frames, ignore_index=True).sort_values(["step_index", "sample_id"])
    step_frame, summary = analyze(frame, args.expected_steps)
    summary["shards"] = provenance
    frame.to_csv(out/"heldout_per_image.csv", index=False)
    step_frame.to_csv(out/"heldout_summary_by_step.csv", index=False)
    torch.save({"steps": calibrations, "applied_fp32_steps": applied,
                "source_requests": requests, "source_shards": provenance}, out/"calibration.pt")
    summary["calibration_sha256"] = sha256(out/"calibration.pt")
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
