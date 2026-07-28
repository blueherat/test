"""Evaluate a paired Flow/LPL checkpoint trajectory with fixed sampling noise."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.summarize_rae_lpl_authenticity import (
    FID,
    IS,
    KID,
    read_json,
    sampling_provenance_valid,
)


DEFAULT_ENDPOINTS = tuple(range(500, 5001, 500))


def parse_endpoints(value: str) -> tuple[int, ...]:
    endpoints = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not endpoints or any(endpoint <= 0 for endpoint in endpoints):
        raise argparse.ArgumentTypeError("endpoints must be positive integers")
    if tuple(sorted(set(endpoints))) != endpoints:
        raise argparse.ArgumentTypeError("endpoints must be unique and increasing")
    return endpoints


def evaluation_rows(path: Path) -> dict[str, dict]:
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else [payload]
    indexed = {str(row["branch"]): row for row in rows}
    if set(indexed) != {"flow", "lpl"}:
        raise ValueError(f"{path} must contain exactly flow and lpl rows")
    for branch, row in indexed.items():
        if not sampling_provenance_valid(row):
            raise ValueError(f"{path} has invalid sampling provenance for {branch}")
    flow = indexed["flow"]
    lpl = indexed["lpl"]
    paired_fields = (
        "endpoint",
        "sample_count",
        "sampling_seed",
        "sampling_steps",
        "sampling_processes",
        "per_process_batch",
        "label_sampler_version",
        "reference_sha256",
    )
    mismatches = [
        field for field in paired_fields if flow.get(field) != lpl.get(field)
    ]
    if mismatches:
        raise ValueError(f"{path} has unpaired fields: {mismatches}")
    return indexed


def summarize_curve(
    evaluation_paths: list[Path],
    *,
    official_evaluation: Path | None = None,
) -> dict:
    official = None
    if official_evaluation is not None:
        payload = read_json(official_evaluation)
        rows = payload if isinstance(payload, list) else [payload]
        if len(rows) != 1 or not sampling_provenance_valid(rows[0]):
            raise ValueError("official evaluation must contain one provenance-valid row")
        official = rows[0]

    rows = []
    seen_endpoints = set()
    for path in evaluation_paths:
        indexed = evaluation_rows(path)
        flow = indexed["flow"]
        lpl = indexed["lpl"]
        endpoint = int(flow["endpoint"])
        if endpoint in seen_endpoints:
            raise ValueError(f"duplicate endpoint {endpoint}")
        seen_endpoints.add(endpoint)
        row = {
            "endpoint": endpoint,
            "sample_count": int(flow["sample_count"]),
            "sampling_seed": int(flow["sampling_seed"]),
            "flow_fid": float(flow[FID]),
            "lpl_fid": float(lpl[FID]),
            "lpl_minus_flow_fid": float(lpl[FID]) - float(flow[FID]),
            "flow_kid": float(flow[KID]),
            "lpl_kid": float(lpl[KID]),
            "lpl_minus_flow_kid": float(lpl[KID]) - float(flow[KID]),
            "flow_is": float(flow[IS]),
            "lpl_is": float(lpl[IS]),
            "lpl_minus_flow_is": float(lpl[IS]) - float(flow[IS]),
            "flow_sample_npz_sha256": flow["sample_npz_sha256"],
            "lpl_sample_npz_sha256": lpl["sample_npz_sha256"],
        }
        if official is not None:
            row.update(
                {
                    "official_fid": float(official[FID]),
                    "lpl_minus_official_fid": float(lpl[FID]) - float(official[FID]),
                    "official_kid": float(official[KID]),
                    "lpl_minus_official_kid": float(lpl[KID]) - float(official[KID]),
                    "official_is": float(official[IS]),
                    "lpl_minus_official_is": float(lpl[IS]) - float(official[IS]),
                }
            )
        rows.append(row)

    rows.sort(key=lambda row: row["endpoint"])
    if official is not None and rows:
        first_evaluation = evaluation_rows(evaluation_paths[0])["flow"]
        paired_fields = (
            "sample_count",
            "sampling_seed",
            "sampling_steps",
            "reference_sha256",
        )
        mismatches = [
            field
            for field in paired_fields
            if official.get(field) != first_evaluation.get(field)
        ]
        if mismatches:
            raise ValueError(
                "official evaluation does not match checkpoint curve: "
                f"{mismatches}"
            )
    return {
        "protocol": "rae-lpl-checkpoint-curve-v1",
        "paired_fixed_sampling": True,
        "official_evaluation": (
            str(official_evaluation.expanduser().resolve())
            if official_evaluation is not None
            else None
        ),
        "rows": rows,
    }


def plot_curve(rows: list[dict], output: Path) -> None:
    if not rows:
        raise ValueError("cannot plot an empty checkpoint curve")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    endpoints = [row["endpoint"] for row in rows]
    panels = (
        ("fid", "FID (lower is better)"),
        ("kid", "KID (lower is better)"),
        ("is", "Inception Score (higher is better)"),
    )
    colors = {"flow": "#2563EB", "lpl": "#EA580C"}
    markers = {"flow": "o", "lpl": "s"}
    styles = {"flow": "--", "lpl": "-"}
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, panels):
        for branch in ("flow", "lpl"):
            axis.plot(
                endpoints,
                [row[f"{branch}_{metric}"] for row in rows],
                label="Flow" if branch == "flow" else "LPL",
                color=colors[branch],
                marker=markers[branch],
                linestyle=styles[branch],
                linewidth=2.2,
                markersize=6,
            )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Post-training updates")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=9)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "RAE-LPL checkpoint trajectory (fixed paired sampling)",
        fontsize=15,
        y=1.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def evaluation_command(
    *,
    flow_branch: Path,
    lpl_branch: Path,
    endpoint: int,
    sample_count: int,
    sampling_seed: int,
    steps: int,
    devices: str,
    reference: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments/evaluate_rae_strict_lpl_generation.py"),
        "--branch",
        f"flow={flow_branch}",
        "--branch",
        f"lpl={lpl_branch}",
        "--reference",
        str(reference),
        "--endpoint",
        str(endpoint),
        "--sample-count",
        str(sample_count),
        "--steps",
        str(steps),
        "--devices",
        devices,
        "--sampling-seed",
        str(sampling_seed),
        "--output",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-branch", type=Path, required=True)
    parser.add_argument("--lpl-branch", type=Path, required=True)
    parser.add_argument(
        "--endpoints",
        type=parse_endpoints,
        default=DEFAULT_ENDPOINTS,
        help="Comma-separated increasing checkpoint steps.",
    )
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--sampling-seed", type=int, default=20260715)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"),
    )
    parser.add_argument("--official-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if args.sample_count <= 0 or args.sample_count % 1000 != 0:
        raise ValueError("sample count must be a positive multiple of 1000")
    paths = (
        args.flow_branch,
        args.lpl_branch,
        args.reference,
    )
    for path in paths:
        if not path.expanduser().exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for endpoint in args.endpoints:
        output = output_dir / (
            f"eval_pair_s{endpoint}_n{args.sample_count}"
            f"_seed{args.sampling_seed}.json"
        )
        evaluations.append(output)
        command = evaluation_command(
            flow_branch=args.flow_branch.expanduser().resolve(),
            lpl_branch=args.lpl_branch.expanduser().resolve(),
            endpoint=endpoint,
            sample_count=args.sample_count,
            sampling_seed=args.sampling_seed,
            steps=args.steps,
            devices=args.devices,
            reference=args.reference.expanduser().resolve(),
            output=output,
        )
        print(json.dumps({"endpoint": endpoint, "command": command}, indent=2))
        if args.print_only:
            continue
        if args.skip_existing and output.exists():
            evaluation_rows(output)
            continue
        subprocess.run(command, cwd=ROOT, check=True)

    if args.print_only:
        return
    summary = summarize_curve(
        evaluations,
        official_evaluation=args.official_evaluation,
    )
    summary_path = output_dir / (
        f"checkpoint_curve_n{args.sample_count}_seed{args.sampling_seed}.json"
    )
    csv_path = summary_path.with_suffix(".csv")
    plot_path = summary_path.with_suffix(".png")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(summary["rows"]).to_csv(csv_path, index=False)
    plot_curve(summary["rows"], plot_path)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "csv": str(csv_path),
                "plot": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
