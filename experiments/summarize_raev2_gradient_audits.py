"""Summarize paired RAEv2 Flow/LPL parameter-gradient probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("audit must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("audit name cannot be empty")
    return name, Path(raw_path).expanduser()


def recommended_lpl_weight(
    *,
    flow_gradient_norm: float,
    probe_gradient_norm: float,
    probe_weight: float,
    target_ratio: float,
) -> float:
    if flow_gradient_norm <= 0 or probe_gradient_norm <= 0:
        raise ValueError("gradient norms must be positive")
    if probe_weight <= 0 or target_ratio <= 0:
        raise ValueError("probe weight and target ratio must be positive")
    unweighted_lpl_gradient_norm = probe_gradient_norm / probe_weight
    return target_ratio * flow_gradient_norm / unweighted_lpl_gradient_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-audit", type=Path, required=True)
    parser.add_argument("--lpl-audit", action="append", type=parse_named_path, required=True)
    parser.add_argument("--target-ratio", type=float, default=0.20)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow = json.loads(args.flow_audit.read_text(encoding="utf-8"))
    if flow["component"] != "flow":
        raise ValueError("--flow-audit does not contain a Flow gradient probe")
    if args.target_ratio <= 0:
        raise ValueError("--target-ratio must be positive")

    rows = []
    weights = {}
    for name, path in args.lpl_audit:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit["component"] != "lpl":
            raise ValueError(f"{path} does not contain an LPL gradient probe")
        if audit["global_samples"] != flow["global_samples"]:
            raise ValueError("Flow and LPL probes used different sample counts")
        if audit["data_indices_sha256"] != flow["data_indices_sha256"]:
            raise ValueError("Flow and LPL probes used different data indices")
        if abs(float(audit["flow_loss"]) - float(flow["flow_loss"])) > 1e-6:
            raise ValueError("Flow and LPL probes did not reproduce the same noisy batch")
        weight = recommended_lpl_weight(
            flow_gradient_norm=float(flow["parameter_gradient_norm"]),
            probe_gradient_norm=float(audit["parameter_gradient_norm"]),
            probe_weight=float(audit["lpl_weight"]),
            target_ratio=float(args.target_ratio),
        )
        weights[name] = weight
        rows.append(
            {
                "name": name,
                "lpl_target": audit["lpl_target"],
                "global_samples": audit["global_samples"],
                "flow_loss": audit["flow_loss"],
                "lpl_loss": audit["lpl_loss"],
                "flow_gradient_norm": flow["parameter_gradient_norm"],
                "probe_weight": audit["lpl_weight"],
                "probe_gradient_norm": audit["parameter_gradient_norm"],
                "unweighted_lpl_gradient_norm": (
                    float(audit["parameter_gradient_norm"])
                    / float(audit["lpl_weight"])
                ),
                "target_weighted_gradient_ratio": args.target_ratio,
                "recommended_lpl_weight": weight,
                "mean_active_guidance_scale": audit[
                    "mean_active_guidance_scale"
                ],
            }
        )
    frame = pd.DataFrame(rows).sort_values("name")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "format_version": 1,
                "target_weighted_lpl_over_flow_gradient": args.target_ratio,
                "flow_audit": flow,
                "recommended_lpl_weights": weights,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    frame.to_csv(args.output_csv, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
