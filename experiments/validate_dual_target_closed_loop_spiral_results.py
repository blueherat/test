#!/usr/bin/env python3
"""Validate completeness, uniqueness, and finite metrics for the formal toy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_SPECS = {
    "endpoint_metrics.csv": (15, ["condition"], ["swd_2d", "swd_fullD"]),
    "teacher_metrics.csv": (
        135,
        ["condition", "time"],
        ["paired_velocity_mse", "bayes_velocity_mse"],
    ),
    "rollout_metrics.csv": (
        126,
        ["condition", "time"],
        ["rollout_bayes_velocity_mse", "state_swd_2d"],
    ),
    "branch_pair_metrics.csv": (
        54,
        ["model", "time"],
        ["first_bayes_mse", "second_bayes_mse"],
    ),
    "cross_gate_endpoint_metrics.csv": (
        12,
        ["condition"],
        ["swd_2d", "swd_fullD"],
    ),
    "cross_gate_teacher_metrics.csv": (
        108,
        ["condition", "time"],
        ["paired_velocity_mse", "bayes_velocity_mse"],
    ),
    "cross_gate_rollout_metrics.csv": (
        108,
        ["condition", "time"],
        ["rollout_bayes_velocity_mse", "state_swd_2d"],
    ),
}


def parse_int_list(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def validate_table(path: Path, spec: tuple[int, list[str], list[str]]) -> list[str]:
    expected_rows, keys, required_metrics = spec
    frame = pd.read_csv(path)
    errors: list[str] = []
    if len(frame) != expected_rows:
        errors.append(f"{path.name}: expected {expected_rows} rows, found {len(frame)}")
    missing = sorted(set(keys + required_metrics) - set(frame.columns))
    if missing:
        errors.append(f"{path.name}: missing columns {missing}")
        return errors
    if frame.duplicated(keys).any():
        errors.append(f"{path.name}: duplicate keys for {keys}")
    values = frame[required_metrics].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append(f"{path.name}: required metrics contain NaN/Inf")
    return errors


def validate_results(root: Path, seeds: list[int], dims: list[int]) -> dict:
    errors: list[str] = []
    settings: list[dict] = []
    for seed in seeds:
        for dimension in dims:
            setting = root / f"seed{seed}" / f"D{dimension}_H128"
            setting_errors: list[str] = []
            if not (setting / "complete.json").is_file():
                setting_errors.append("missing complete.json")
            evaluation_config = setting / "evaluation_config.json"
            if not evaluation_config.is_file():
                setting_errors.append("missing evaluation_config.json")
            else:
                payload = json.loads(evaluation_config.read_text(encoding="utf-8"))
                if abs(float(payload["intrinsic_jitter_std_after_v4_scale"]) - 0.024) > 1e-12:
                    setting_errors.append("incorrect scaled intrinsic jitter")
            for filename, spec in TABLE_SPECS.items():
                path = setting / filename
                if not path.is_file():
                    setting_errors.append(f"missing {filename}")
                else:
                    setting_errors.extend(validate_table(path, spec))
            settings.append(
                {
                    "seed": seed,
                    "ambient_dim": dimension,
                    "status": "passed" if not setting_errors else "failed",
                    "errors": setting_errors,
                }
            )
            errors.extend(f"seed={seed}, D={dimension}: {error}" for error in setting_errors)

    endpoint_summary_path = root / "aggregate" / "endpoint_seed_summary.csv"
    if not endpoint_summary_path.is_file():
        errors.append("missing aggregate/endpoint_seed_summary.csv")
    else:
        endpoint_summary = pd.read_csv(endpoint_summary_path)
        if set(endpoint_summary["seeds"].astype(int)) != {len(seeds)}:
            errors.append("endpoint summary does not contain every expected seed")

    solver_path = root / "aggregate" / "solver_step_convergence.csv"
    if not solver_path.is_file():
        errors.append("missing aggregate/solver_step_convergence.csv")
    else:
        solver = pd.read_csv(solver_path)
        expected = len(dims) * 5 * 2
        if len(solver) != expected:
            errors.append(f"solver convergence expected {expected} rows, found {len(solver)}")
        if not np.isfinite(solver[["swd_2d", "swd_fullD"]].to_numpy()).all():
            errors.append("solver convergence contains non-finite SWD")

    return {
        "status": "passed" if not errors else "failed",
        "expected_seeds": seeds,
        "expected_dimensions": dims,
        "settings": settings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20260831,20260901,20260902"))
    parser.add_argument("--dims", type=parse_int_list, default=parse_int_list("2,512"))
    args = parser.parse_args()

    report = validate_results(args.root, args.seeds, args.dims)
    output = args.root / "aggregate" / "validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
