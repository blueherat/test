"""Resume-safe evaluation and plotting for the RAEv2 Flow checkpoint sweep."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_raev2_long_pipeline import (  # noqa: E402
    append_event,
    atomic_json,
    evaluate_branch,
    merge_evaluations,
    verify_same_noise_protocol,
    write_flow_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--python", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    sample_root = pipeline_root / f"samples_n{args.sample_count}_seed0"
    logs = pipeline_root / "logs"
    event_log = pipeline_root / "events.jsonl"
    status_path = pipeline_root / "flow_evaluation_status.json"
    names = ["official", *(f"flow_s{step:04d}" for step in range(10, 151, 10))]
    directories = {name: sample_root / name for name in names}
    missing = [
        str(directory / "sampling_summary.json")
        for directory in directories.values()
        if not (directory / "sampling_summary.json").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Flow samples are incomplete: {missing}")

    plan = {
        "state": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "sample_count": args.sample_count,
        "branches": names,
    }
    atomic_json(status_path, plan)
    append_event(event_log, "flow_evaluation_start", branches=names)
    try:
        fingerprints = verify_same_noise_protocol(directories)
        atomic_json(pipeline_root / "same_noise_flow_audit.json", fingerprints)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        evaluations = {}
        for name, directory in directories.items():
            evaluations[name] = evaluate_branch(
                name,
                directory / "samples.npz",
                python=args.python.expanduser().resolve(),
                pipeline_root=pipeline_root,
                logs=logs,
                event_log=event_log,
                env=environment,
            )
        metrics_csv = pipeline_root / f"metrics_flow_n{args.sample_count}.csv"
        merge_evaluations(evaluations, metrics_csv)
        curve_csv = pipeline_root / f"curve_flow_n{args.sample_count}.csv"
        curve_png = pipeline_root / f"curve_flow_n{args.sample_count}.png"
        write_flow_curve(
            metrics_csv,
            output_csv=curve_csv,
            output_png=curve_png,
        )
        result = {
            **plan,
            "state": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "metrics_csv": str(metrics_csv),
            "curve_csv": str(curve_csv),
            "curve_png": str(curve_png),
        }
        atomic_json(status_path, result)
        append_event(event_log, "flow_evaluation_complete", **result)
        print(json.dumps(result, indent=2))
    except Exception as error:
        atomic_json(
            status_path,
            {**plan, "state": "failed", "error": repr(error)},
        )
        append_event(event_log, "flow_evaluation_failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
