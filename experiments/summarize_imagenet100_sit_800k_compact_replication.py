#!/usr/bin/env python3
"""Select the matched weak model and summarize the compact SiT 800K replication."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def one_scale(payload: dict, scale: float) -> dict:
    matches = [
        row
        for row in payload.get("rows", [])
        if math.isclose(float(row["scale"]), scale, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one scale={scale:g} row, found {len(matches)}")
    return matches[0]


def _candidate(value: str) -> tuple[int, Path]:
    step_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be STEP=SUMMARY_JSON")
    return int(step_text), Path(path_text).expanduser().resolve()


def normalized_fingerprint(row: dict) -> tuple[str, str]:
    return str(row["noise_fingerprint"]), str(row["label_fingerprint"])


def select_matched_weak(args: argparse.Namespace) -> dict:
    x_payload = load_json(args.x_endpoint_summary)
    x_row = one_scale(x_payload, 1.0)
    if int(x_row["checkpoint_step"]) != 800_000:
        raise ValueError("x endpoint anchor is not v800")
    if int(x_row["other_checkpoint_step"]) != 800_000:
        raise ValueError("x endpoint is not x800")
    candidates = []
    for requested_step, path in args.v_candidate:
        payload = load_json(path)
        row = one_scale(payload, 1.0)
        actual_step = int(row["other_checkpoint_step"])
        if actual_step != requested_step * 1000:
            raise ValueError(
                f"candidate v{requested_step} reports checkpoint step {actual_step}"
            )
        if int(row["checkpoint_step"]) != 800_000:
            raise ValueError(f"candidate v{requested_step} anchor is not v800")
        if normalized_fingerprint(row) != normalized_fingerprint(x_row):
            raise ValueError(f"candidate v{requested_step} is not paired with x800 endpoint")
        checkpoint = Path(payload["other"]["checkpoint"]).resolve()
        candidates.append(
            {
                "label": f"v{requested_step}",
                "checkpoint_step": actual_step,
                "checkpoint": str(checkpoint),
                "fid": float(row["fid"]),
                "absolute_fid_gap_to_x800": abs(float(row["fid"]) - float(x_row["fid"])),
                "summary": str(path),
            }
        )
    if not candidates:
        raise ValueError("at least one same-target weak candidate is required")
    candidates.sort(key=lambda row: (row["absolute_fid_gap_to_x800"], row["checkpoint_step"]))
    matched = candidates[0]
    if not Path(matched["checkpoint"]).is_file():
        raise FileNotFoundError(matched["checkpoint"])
    payload = {
        "protocol": "imagenet100_sit_800k_quality_match_v1",
        "selection_rule": "minimum absolute ADM FID-5K gap to the standard x800 endpoint",
        "selection_uses_only_global_seed": 0,
        "x800_endpoint_fid": float(x_row["fid"]),
        "x800_endpoint_sfid": float(x_row["sfid"]),
        "x800_endpoint_inception_score": float(x_row["inception_score"]),
        "noise_fingerprint": x_row["noise_fingerprint"],
        "label_fingerprint": x_row["label_fingerprint"],
        "candidates": candidates,
        "matched": matched,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    return payload


def _frozen_row(path: Path, condition: str) -> dict:
    payload = load_json(path)
    return {
        "condition": condition,
        "fid": float(payload["fid"]),
        "sfid": float(payload["sfid"]),
        "inception_score": float(payload["inception_score"]),
        "noise_fingerprint": str(payload["noise_fingerprint"]),
        "label_fingerprint": str(payload["label_fingerprint"]),
        "anchor_step": int(payload["anchor"]["checkpoint_step"]),
        "other_step": int(payload["other"]["checkpoint_step"]),
        "total_nfe": int(payload["total_nfe"]),
        "sampling_peak_memory_mib": int(payload["sampling_peak_memory_mib"]),
        "fid_peak_memory_mib": int(payload["fid_peak_memory_mib"]),
        "source": str(path),
    }


def _static_row(row: dict, condition: str, source: Path) -> dict:
    return {
        "condition": condition,
        "fid": float(row["fid"]),
        "sfid": float(row["sfid"]),
        "inception_score": float(row["inception_score"]),
        "noise_fingerprint": str(row["noise_fingerprint"]),
        "label_fingerprint": str(row["label_fingerprint"]),
        "anchor_step": int(row["checkpoint_step"]),
        "other_step": int(row["other_checkpoint_step"]),
        "total_nfe": int(row["total_nfe"]),
        "sampling_peak_memory_mib": int(row["sampling_peak_memory_mib"]),
        "fid_peak_memory_mib": int(row["fid_peak_memory_mib"]),
        "source": str(source),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict]) -> None:
    order = ("baseline", "x_closed", "x_frozen", "vweak_closed", "vweak_frozen")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for seed in (0, 1):
        selected = {row["condition"]: row for row in rows if row["sample_seed"] == seed}
        axes[0].plot(
            range(len(order)),
            [selected[name]["fid"] for name in order],
            "o-",
            label=f"sample seed {seed}",
        )
    axes[0].set_xticks(range(len(order)), order, rotation=25, ha="right")
    axes[0].set_ylabel("ADM FID-5K (lower is better)")
    axes[0].set_title("SiT 800K compact replication")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    labels = []
    values = []
    for seed in (0, 1):
        selected = {row["condition"]: row for row in rows if row["sample_seed"] == seed}
        for prefix in ("x", "vweak"):
            closed_gain = selected["baseline"]["fid"] - selected[f"{prefix}_closed"]["fid"]
            frozen_gain = selected["baseline"]["fid"] - selected[f"{prefix}_frozen"]["fid"]
            labels.append(f"s{seed} {prefix}\nclosed")
            values.append(closed_gain)
            labels.append(f"s{seed} {prefix}\nfrozen")
            values.append(frozen_gain)
    axes[1].bar(range(len(labels)), values)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[1].set_ylabel("FID gain over paired baseline")
    axes[1].set_title("Closed-loop and frozen-gap gains")
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_final(args: argparse.Namespace) -> dict:
    selection = load_json(args.selection)
    matched_step = int(selection["matched"]["checkpoint_step"])
    rows: list[dict] = []
    seed_metrics = []
    for seed in (0, 1):
        seed_root = args.root / f"seed{seed}"
        x_path = seed_root / "x_pair/field_control_fid5k.json"
        v_path = seed_root / "vweak_closed/field_control_fid5k.json"
        x_payload = load_json(x_path)
        v_payload = load_json(v_path)
        seed_rows = [
            _static_row(one_scale(x_payload, 0.0), "baseline", x_path),
            _static_row(one_scale(x_payload, -1.0), "x_closed", x_path),
            _frozen_row(
                seed_root / "x_frozen/frozen_guidance_fid5k.json",
                "x_frozen",
            ),
            _static_row(one_scale(v_payload, -1.0), "vweak_closed", v_path),
            _frozen_row(
                seed_root / "vweak_frozen/frozen_guidance_fid5k.json",
                "vweak_frozen",
            ),
        ]
        fingerprints = {
            (row["noise_fingerprint"], row["label_fingerprint"])
            for row in seed_rows
        }
        if len(fingerprints) != 1:
            raise ValueError(f"sample seed {seed} conditions are not noise/label paired")
        for row in seed_rows:
            if row["anchor_step"] != 800_000:
                raise ValueError(f"{row['condition']} seed {seed} does not use v800")
            expected_other = 800_000 if row["condition"].startswith("x_") else matched_step
            if row["condition"] == "baseline":
                expected_other = 800_000
            if row["other_step"] != expected_other:
                raise ValueError(
                    f"{row['condition']} seed {seed} uses other step {row['other_step']}"
                )
            row["sample_seed"] = seed
            rows.append(row)
        by_condition = {row["condition"]: row for row in seed_rows}
        metrics = {"sample_seed": seed, "conditions": by_condition}
        for prefix in ("x", "vweak"):
            baseline_fid = by_condition["baseline"]["fid"]
            closed_gain = baseline_fid - by_condition[f"{prefix}_closed"]["fid"]
            frozen_gain = baseline_fid - by_condition[f"{prefix}_frozen"]["fid"]
            metrics[f"{prefix}_closed_gain"] = closed_gain
            metrics[f"{prefix}_frozen_gain"] = frozen_gain
            metrics[f"{prefix}_frozen_retention"] = (
                frozen_gain / closed_gain if closed_gain > 0 else float("nan")
            )
        seed_metrics.append(metrics)

    for row in rows:
        baseline = next(
            item for item in rows
            if item["sample_seed"] == row["sample_seed"] and item["condition"] == "baseline"
        )
        row["fid_gain_vs_paired_baseline"] = baseline["fid"] - row["fid"]
    retention_values = [
        metrics[f"{prefix}_frozen_retention"]
        for metrics in seed_metrics
        for prefix in ("x", "vweak")
    ]
    closed_gains = [
        metrics[f"{prefix}_closed_gain"]
        for metrics in seed_metrics
        for prefix in ("x", "vweak")
    ]
    passed = all(gain > 0 for gain in closed_gains) and all(
        math.isfinite(value) and value >= 0.60 for value in retention_values
    )
    output_json = args.root / "compact_replication_summary.json"
    output_csv = args.root / "compact_replication_rows.csv"
    output_figure = args.root / "compact_replication.png"
    payload = {
        "protocol": "imagenet100_sit_800k_compact_replication_v1",
        "comparison_is_paired_within_each_sample_seed": True,
        "sample_seeds": [0, 1],
        "quality_match": selection,
        "acceptance": {
            "rule": (
                "for both directions and both sample seeds: closed-loop FID gain > 0 "
                "and frozen-gap retains at least 60% of that gain"
            ),
            "passed": passed,
        },
        "seed_metrics": seed_metrics,
        "rows": rows,
        "csv": str(output_csv),
        "figure": str(output_figure),
    }
    _write_csv(output_csv, rows)
    _plot(output_figure, rows)
    atomic_json(output_json, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--x-endpoint-summary", type=Path, required=True)
    select.add_argument("--v-candidate", type=_candidate, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        args.x_endpoint_summary = args.x_endpoint_summary.expanduser().resolve()
        args.output = args.output.expanduser().resolve()
        payload = select_matched_weak(args)
    else:
        args.root = args.root.expanduser().resolve()
        args.selection = args.selection.expanduser().resolve()
        payload = summarize_final(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
