#!/usr/bin/env python3
"""Validate and summarize the paired 400K field-mechanism audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_BASE = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
EXPECTED_AUDIT_COUNTS = {
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
    "same_target_v240": 5,
    "same_target_v270": 5,
    "same_target_v300": 5,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def one_row(rows: list[dict], *, scale: float) -> dict:
    matches = [row for row in rows if math.isclose(float(row["scale"]), scale)]
    if len(matches) != 1:
        raise ValueError(f"expected one scale={scale:g} row, found {len(matches)}")
    return matches[0]


def normalize_row(row: dict, *, series: str) -> dict:
    normalized = dict(row)
    normalized["series"] = series
    for key in (
        "scale",
        "fid",
        "sfid",
        "inception_score",
        "sampling_elapsed_seconds_max_rank",
    ):
        normalized[key] = float(normalized.get(key, 0.0))
    for key in (
        "checkpoint_step",
        "other_checkpoint_step",
        "num_samples",
        "total_nfe",
        "total_model_forwards",
    ):
        normalized[key] = int(float(normalized.get(key, 0)))
    normalized.setdefault("control_mode", "full_pair")
    return normalized


def load_audit_rows(audit_root: Path, *, allow_incomplete: bool) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for series, expected_count in EXPECTED_AUDIT_COUNTS.items():
        summary_path = audit_root / series / "field_control_fid5k.json"
        if not summary_path.is_file():
            missing.append(series)
            continue
        payload = load_json(summary_path)
        series_rows = [normalize_row(row, series=series) for row in payload["rows"]]
        if len(series_rows) != expected_count:
            raise ValueError(
                f"{series}: expected {expected_count} rows, found {len(series_rows)}"
            )
        rows.extend(series_rows)
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"incomplete 400K audit; missing: {', '.join(missing)}")
    return rows, missing


def verify_pairing(rows: list[dict], *, reference_row: dict) -> None:
    expected_noise = reference_row["noise_fingerprint"]
    expected_labels = reference_row["label_fingerprint"]
    for row in rows:
        if int(row["num_samples"]) != 5_000:
            raise ValueError(f"{row['series']} is not a 5K result")
        if row["noise_fingerprint"] != expected_noise:
            raise ValueError(f"{row['series']} has a different noise fingerprint")
        if row["label_fingerprint"] != expected_labels:
            raise ValueError(f"{row['series']} has a different label fingerprint")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) or ["series"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict], full_rows: list[dict], baseline_fid: float) -> None:
    import matplotlib.pyplot as plt

    by_series = {
        series: sorted(
            (row for row in rows if row["series"] == series),
            key=lambda row: -float(row["scale"]),
        )
        for series in {row["series"] for row in rows}
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    x_curve = sorted(
        (row for row in full_rows if -1.0 <= float(row["scale"]) <= 0.0),
        key=lambda row: -float(row["scale"]),
    )
    axes[0].plot(
        [-float(row["scale"]) for row in x_curve],
        [float(row["fid"]) for row in x_curve],
        marker="o",
        label="v400 vs x400",
    )
    if "floor_only" in by_series:
        floor_rows = by_series["floor_only"]
        axes[0].plot(
            [-float(row["scale"]) for row in floor_rows],
            [float(row["fid"]) for row in floor_rows],
            marker="o",
            label="deterministic floor only",
        )
    axes[0].axhline(baseline_fid, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Floor control", xlabel="guidance gamma = -scale", ylabel="FID-5K")
    axes[0].legend()

    mechanism_series = [
        "x400 full",
        "floor_only",
        "floor_residual",
        "pre_floor_pair",
        "post_floor_pair",
        "parallel_pair",
        "orthogonal_pair",
    ]
    mechanism_rows = {"x400 full": one_row(full_rows, scale=-1.0)}
    for series in mechanism_series[1:]:
        if series in by_series:
            mechanism_rows[series] = one_row(by_series[series], scale=-1.0)
    names = [name for name in mechanism_series if name in mechanism_rows]
    gains = [baseline_fid - float(mechanism_rows[name]["fid"]) for name in names]
    axes[1].bar(range(len(names)), gains)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(range(len(names)), names, rotation=35, ha="right")
    axes[1].set(title="Mechanism at gamma=1", ylabel="FID gain over v400")

    axes[2].plot(
        [-float(row["scale"]) for row in x_curve],
        [float(row["fid"]) for row in x_curve],
        marker="o",
        label="x400 weak",
    )
    for series in ("same_target_v240", "same_target_v270", "same_target_v300"):
        if series not in by_series:
            continue
        guidance_rows = [row for row in by_series[series] if float(row["scale"]) <= 0]
        axes[2].plot(
            [-float(row["scale"]) for row in guidance_rows],
            [float(row["fid"]) for row in guidance_rows],
            marker="o",
            label=series,
        )
    axes[2].axhline(baseline_fid, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Prediction target vs same-target AG", xlabel="guidance gamma", ylabel="FID-5K")
    axes[2].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-sweep-root",
        type=Path,
        default=DEFAULT_BASE / "fid5k_static_pair_v_to_jit_x_step400000_seed0",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=DEFAULT_BASE / "fid5k_step400k_floor_audit_seed0",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    full_payload = load_json(args.full_sweep_root / "static_pair_v_to_jit_x_fid5k.json")
    full_rows = [normalize_row(row, series="x400_full_pair") for row in full_payload["rows"]]
    baseline = one_row(full_rows, scale=0.0)
    x_endpoint = one_row(full_rows, scale=1.0)
    x_guided = one_row(full_rows, scale=-1.0)
    verify_pairing(full_rows, reference_row=baseline)
    audit_rows, missing = load_audit_rows(
        args.audit_root,
        allow_incomplete=args.allow_incomplete,
    )
    verify_pairing(audit_rows, reference_row=baseline)

    baseline_fid = float(baseline["fid"])
    for row in audit_rows:
        row["guidance_gamma"] = -float(row["scale"])
        row["fid_gain_vs_v400"] = baseline_fid - float(row["fid"])

    weak_endpoints = [
        one_row(
            [row for row in audit_rows if row["series"] == series],
            scale=1.0,
        )
        for series in ("same_target_v240", "same_target_v270", "same_target_v300")
        if any(row["series"] == series for row in audit_rows)
    ]
    matched_weak = min(
        weak_endpoints,
        key=lambda row: abs(float(row["fid"]) - float(x_endpoint["fid"])),
        default=None,
    )

    args.audit_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.audit_root / "combined_mechanism_audit.csv", audit_rows)
    summary = {
        "protocol": "imagenet100_sit_400k_mechanism_audit_v1",
        "audit_complete": not missing,
        "pairing_verified": bool(audit_rows) and not missing,
        "available_rows_pairing_verified": True,
        "missing_series": missing,
        "v400_fid": baseline_fid,
        "x400_endpoint_fid": float(x_endpoint["fid"]),
        "x400_guided_scale_minus_1_fid": float(x_guided["fid"]),
        "x400_guidance_gain": baseline_fid - float(x_guided["fid"]),
        "matched_same_target_weak": matched_weak,
        "rows": audit_rows,
    }
    (args.audit_root / "combined_mechanism_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if audit_rows:
        plot_summary(
            args.audit_root / "combined_mechanism_audit.png",
            audit_rows,
            full_rows,
            baseline_fid,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
