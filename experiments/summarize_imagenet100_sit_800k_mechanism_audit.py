#!/usr/bin/env python3
"""Validate and summarize the paired 800K SiT field-mechanism audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from experiments.summarize_imagenet100_sit_400k_mechanism_audit import (
        load_json,
        normalize_row,
        one_row,
        verify_pairing,
        write_csv,
    )
except ModuleNotFoundError:
    from summarize_imagenet100_sit_400k_mechanism_audit import (
        load_json,
        normalize_row,
        one_row,
        verify_pairing,
        write_csv,
    )


BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_PAIR = BASE / "fid5k_static_pair_v_to_jit_x_step800000_seed0"
DEFAULT_AUDIT = BASE / "fid5k_step800k_floor_audit_seed0"
MECHANISM_COUNTS = {
    "floor_only": 4,
    "floor_residual": 1,
    "pre_floor_pair": 1,
    "post_floor_pair": 1,
    "parallel_pair": 1,
    "orthogonal_pair": 1,
    "x_inference_floor_0p02": 1,
    "x_inference_floor_0p01": 1,
    "x_inference_floor_0p005": 1,
    "x_inference_floor_0p001": 1,
}
DEFAULT_WEAK_STEPS = (400, 500, 600, 700)


def load_rows(
    root: Path,
    weak_steps: tuple[int, ...],
    *,
    allow_incomplete: bool,
) -> tuple[list[dict], list[str]]:
    expected = dict(MECHANISM_COUNTS)
    expected.update({f"same_target_v{step}": 5 for step in weak_steps})
    rows: list[dict] = []
    missing: list[str] = []
    for series, count in expected.items():
        path = root / series / "field_control_fid5k.json"
        if not path.is_file():
            missing.append(series)
            continue
        series_rows = [
            normalize_row(row, series=series)
            for row in load_json(path)["rows"]
        ]
        if len(series_rows) != count:
            raise ValueError(f"{series}: expected {count} rows, found {len(series_rows)}")
        rows.extend(series_rows)
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"incomplete 800K audit; missing: {', '.join(missing)}")
    return rows, missing


def plot_summary(
    path: Path,
    rows: list[dict],
    full_rows: list[dict],
    baseline_fid: float,
    weak_steps: tuple[int, ...],
) -> None:
    by_series = {
        series: sorted(
            (row for row in rows if row["series"] == series),
            key=lambda row: -float(row["scale"]),
        )
        for series in {row["series"] for row in rows}
    }
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    x_curve = sorted(
        (row for row in full_rows if -1.0 <= float(row["scale"]) <= 0.0),
        key=lambda row: -float(row["scale"]),
    )
    gamma = [-float(row["scale"]) for row in x_curve]
    axes[0].plot(gamma, [float(row["fid"]) for row in x_curve], "o-", label="v800 vs x800")
    if "floor_only" in by_series:
        floor = by_series["floor_only"]
        axes[0].plot(
            [-float(row["scale"]) for row in floor],
            [float(row["fid"]) for row in floor],
            "o-",
            label="deterministic floor only",
        )
    axes[0].axhline(baseline_fid, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Floor control", xlabel="guidance gamma", ylabel="FID-5K")
    axes[0].legend()

    series_order = (
        "x800_full",
        "floor_only",
        "floor_residual",
        "pre_floor_pair",
        "post_floor_pair",
        "parallel_pair",
        "orthogonal_pair",
    )
    mechanism = {"x800_full": one_row(full_rows, scale=-1.0)}
    for series in series_order[1:]:
        if series in by_series:
            mechanism[series] = one_row(by_series[series], scale=-1.0)
    names = [name for name in series_order if name in mechanism]
    gains = [baseline_fid - float(mechanism[name]["fid"]) for name in names]
    axes[1].bar(range(len(names)), gains)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(range(len(names)), names, rotation=35, ha="right")
    axes[1].set(title="Mechanism at gamma=1", ylabel="FID gain over v800")

    axes[2].plot(gamma, [float(row["fid"]) for row in x_curve], "o-", label="x800 weak")
    for step in weak_steps:
        series = f"same_target_v{step}"
        if series not in by_series:
            continue
        selected = [row for row in by_series[series] if float(row["scale"]) <= 0]
        axes[2].plot(
            [-float(row["scale"]) for row in selected],
            [float(row["fid"]) for row in selected],
            "o-",
            label=series,
        )
    axes[2].axhline(baseline_fid, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Prediction target vs same-target AG", xlabel="guidance gamma", ylabel="FID-5K")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sweep-root", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--weak-steps", nargs="+", type=int, default=list(DEFAULT_WEAK_STEPS))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    weak_steps = tuple(args.weak_steps)

    full_payload = load_json(args.full_sweep_root / "static_pair_v_to_jit_x_fid5k.json")
    full_rows = [normalize_row(row, series="x800_full_pair") for row in full_payload["rows"]]
    baseline = one_row(full_rows, scale=0.0)
    x_endpoint = one_row(full_rows, scale=1.0)
    x_guided = one_row(full_rows, scale=-1.0)
    verify_pairing(full_rows, reference_row=baseline)
    rows, missing = load_rows(
        args.audit_root,
        weak_steps,
        allow_incomplete=args.allow_incomplete,
    )
    verify_pairing(rows, reference_row=baseline)

    baseline_fid = float(baseline["fid"])
    for row in rows:
        row["guidance_gamma"] = -float(row["scale"])
        row["fid_gain_vs_v800"] = baseline_fid - float(row["fid"])
    endpoints = [
        one_row([row for row in rows if row["series"] == f"same_target_v{step}"], scale=1.0)
        for step in weak_steps
        if any(row["series"] == f"same_target_v{step}" for row in rows)
    ]
    matched = min(
        endpoints,
        key=lambda row: abs(float(row["fid"]) - float(x_endpoint["fid"])),
        default=None,
    )
    args.audit_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.audit_root / "combined_mechanism_audit.csv", rows)
    payload = {
        "protocol": "imagenet100_sit_800k_mechanism_audit_v1",
        "audit_complete": not missing,
        "pairing_verified": bool(rows) and not missing,
        "missing_series": missing,
        "v800_fid": baseline_fid,
        "x800_endpoint_fid": float(x_endpoint["fid"]),
        "x800_guided_scale_minus_1_fid": float(x_guided["fid"]),
        "x800_guidance_gain": baseline_fid - float(x_guided["fid"]),
        "matched_same_target_weak": matched,
        "rows": rows,
    }
    (args.audit_root / "combined_mechanism_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        plot_summary(
            args.audit_root / "combined_mechanism_audit.png",
            rows,
            full_rows,
            baseline_fid,
            weak_steps,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
