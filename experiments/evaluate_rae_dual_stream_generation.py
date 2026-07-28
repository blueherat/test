"""Evaluate full RAE dual-stream controls and apply the registered gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_rae_layerwise_path_generation import (
    DEFAULT_REFERENCE,
    NumpyRGBDataset,
    fidelity_metrics,
)


RESULTS = Path.home() / "data/eqvae/experiments/rae_dual_stream_full"
BASELINE_TABLE = (
    Path.home()
    / "data/eqvae/experiments/rae_layerwise_path_train/"
    "layerwise_path_generation_metrics.csv"
)
CONDITIONS = ("semantic_only", "paired_detail", "shuffled_detail")


def build_gate(table: pd.DataFrame) -> dict[str, object]:
    indexed = table.set_index("condition")
    fid = indexed["frechet_inception_distance"]
    kid = indexed["kernel_inception_distance_mean"]
    fid_improvement = float((fid["static"] - fid["paired_detail"]) / fid["static"])
    gate = {
        "seed": 3407,
        "static_fid": float(fid["static"]),
        "semantic_only_fid": float(fid["semantic_only"]),
        "paired_detail_fid": float(fid["paired_detail"]),
        "shuffled_detail_fid": float(fid["shuffled_detail"]),
        "static_kid": float(kid["static"]),
        "semantic_only_kid": float(kid["semantic_only"]),
        "paired_detail_kid": float(kid["paired_detail"]),
        "shuffled_detail_kid": float(kid["shuffled_detail"]),
        "paired_fid_improvement_vs_static": fid_improvement,
        "paired_beats_semantic_only": bool(fid["paired_detail"] < fid["semantic_only"]),
        "paired_beats_shuffled_detail": bool(
            fid["paired_detail"] < fid["shuffled_detail"]
        ),
        "gate_pass": bool(
            fid_improvement >= 0.05
            and fid["paired_detail"] < fid["semantic_only"]
            and fid["paired_detail"] < fid["shuffled_detail"]
        ),
    }
    gate["decision"] = (
        "continue_dual_stream" if gate["gate_pass"] else "stop_dual_stream"
    )
    return gate


def evaluate(
    sample_dir: Path,
    reference_path: Path,
    baseline_table: Path,
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    reference = NumpyRGBDataset(reference_path)
    rows = []
    for condition in CONDITIONS:
        samples = NumpyRGBDataset(sample_dir / f"{condition}.npz")
        metrics = fidelity_metrics(samples, reference, batch_size=batch_size)
        rows.append({"condition": condition, "seed": 3407, **metrics})
    baseline = pd.read_csv(baseline_table)
    static = baseline[
        (baseline["seed"] == 3407)
        & (baseline["path_mode"] == "static")
        & (baseline["subspace_kind"] == "middle_guided")
    ]
    if len(static) != 1:
        raise ValueError("expected exactly one seed-3407 static baseline row")
    baseline_row = {"condition": "static", **static.iloc[0].to_dict()}
    rows.append(baseline_row)
    table = pd.DataFrame(rows)
    return table, build_gate(table)


def render_report(gate: dict[str, object]) -> str:
    return (
        "# RAE 双流完整生成门控\n\n"
        f"- Static FID：{gate['static_fid']:.4f}\n"
        f"- Semantic-only FID：{gate['semantic_only_fid']:.4f}\n"
        f"- Paired-detail FID：{gate['paired_detail_fid']:.4f}\n"
        f"- Shuffled-detail FID：{gate['shuffled_detail_fid']:.4f}\n"
        f"- Paired 相对 static 改善：{100 * gate['paired_fid_improvement_vs_static']:.2f}%\n"
        f"- 完整门控：{'通过' if gate['gate_pass'] else '未通过'}\n"
        f"- 决策：`{gate['decision']}`\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline-table", type=Path, default=BASELINE_TABLE)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    sample_dir = args.sample_dir or args.results / "generation_seed3407"
    table, gate = evaluate(
        sample_dir,
        args.reference,
        args.baseline_table,
        batch_size=int(args.batch_size),
    )
    table.to_csv(args.results / "generation_metrics.csv", index=False)
    (args.results / "generation_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = render_report(gate)
    (args.results / "generation_gate_zh.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
