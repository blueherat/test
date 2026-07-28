"""Apply the preregistered layerwise-path generation gate to completed seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS = Path.home() / "data/eqvae/experiments/rae_layerwise_path_train"
EXPECTED_CONDITIONS = ("static", "annealed", "reverse", "random")


def condition_name(row: pd.Series) -> str:
    if row["subspace_kind"] == "random_energy_matched":
        return "random"
    return str(row["path_mode"])


def metric_columns(table: pd.DataFrame) -> tuple[str, str]:
    fid = "frechet_inception_distance"
    kid = "kernel_inception_distance_mean"
    missing = [name for name in (fid, kid) if name not in table.columns]
    if missing:
        raise KeyError(f"missing generation metrics: {missing}")
    return fid, kid


def analyze(table: pd.DataFrame) -> dict[str, object]:
    table = table.copy()
    table["condition"] = table.apply(condition_name, axis=1)
    fid_column, kid_column = metric_columns(table)
    seed_tables: dict[str, pd.DataFrame] = {}
    summaries = []
    for metric_name, column in (("fid", fid_column), ("kid", kid_column)):
        pivot = table.pivot(index="seed", columns="condition", values=column).sort_index()
        seed_tables[metric_name] = pivot
        complete = pivot.dropna(subset=list(EXPECTED_CONDITIONS))
        for seed, row in complete.iterrows():
            summaries.append(
                {
                    "metric": metric_name,
                    "seed": int(seed),
                    "static": float(row["static"]),
                    "annealed": float(row["annealed"]),
                    "reverse": float(row["reverse"]),
                    "random": float(row["random"]),
                    "annealed_improvement": float(
                        (row["static"] - row["annealed"]) / row["static"]
                    ),
                    "annealed_beats_static": bool(row["annealed"] < row["static"]),
                    "reverse_worse_than_annealed": bool(row["reverse"] > row["annealed"]),
                    "random_worse_than_annealed": bool(row["random"] > row["annealed"]),
                }
            )
    summary = pd.DataFrame(summaries)
    gate_rows = []
    for metric_name in ("fid", "kid"):
        rows = summary[summary["metric"] == metric_name]
        seed_count = len(rows)
        gate_rows.append(
            {
                "metric": metric_name,
                "complete_seeds": seed_count,
                "mean_annealed_improvement": (
                    float(rows["annealed_improvement"].mean()) if seed_count else None
                ),
                "annealed_direction_count": (
                    int(rows["annealed_beats_static"].sum()) if seed_count else 0
                ),
                "reverse_control_count": (
                    int(rows["reverse_worse_than_annealed"].sum()) if seed_count else 0
                ),
                "random_control_count": (
                    int(rows["random_worse_than_annealed"].sum()) if seed_count else 0
                ),
                "direction_failure_count": (
                    int((~rows["annealed_beats_static"]).sum()) if seed_count else 0
                ),
                "futility_stop": bool(
                    seed_count > 0 and (~rows["annealed_beats_static"]).any()
                ),
                "gate_pass": bool(
                    seed_count == 3
                    and rows["annealed_improvement"].mean() >= 0.05
                    and rows["annealed_beats_static"].all()
                    and rows["reverse_worse_than_annealed"].all()
                    and rows["random_worse_than_annealed"].all()
                ),
            }
        )
    futility_stop = bool(any(row["futility_stop"] for row in gate_rows))
    complete = all(row["complete_seeds"] == 3 for row in gate_rows)
    return {
        "seed_rows": summary.to_dict(orient="records"),
        "gate_rows": gate_rows,
        "gate_pass_both_metrics": bool(all(row["gate_pass"] for row in gate_rows)),
        "futility_stop": futility_stop,
        "status": "stopped_futility" if futility_stop else ("complete" if complete else "incomplete"),
    }


def render_chinese_report(result: dict[str, object]) -> str:
    lines = ["# RAE Layerwise Path 生成门控", ""]
    lines.append(f"状态：`{result['status']}`")
    lines.append("")
    for gate in result["gate_rows"]:
        improvement = gate["mean_annealed_improvement"]
        improvement_text = "NA" if improvement is None else f"{100.0 * improvement:.2f}%"
        lines.extend(
            [
                f"## {str(gate['metric']).upper()}",
                "",
                f"- 完整种子：{gate['complete_seeds']}/3",
                f"- Annealed 相对 static 平均改善：{improvement_text}",
                f"- 同方向种子：{gate['annealed_direction_count']}/3",
                f"- Reverse 更差：{gate['reverse_control_count']}/3",
                f"- 等能量 random 更差：{gate['random_control_count']}/3",
                f"- Annealed 反方向：{gate['direction_failure_count']} 个 seed",
                f"- Futility stop：{'触发' if gate['futility_stop'] else '未触发'}",
                f"- 预注册门控：{'通过' if gate['gate_pass'] else '未通过'}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    input_path = args.results / "layerwise_path_generation_metrics.csv"
    result = analyze(pd.read_csv(input_path))
    json_path = args.results / "layerwise_path_gate.json"
    report_path = args.results / "layerwise_path_gate_zh.md"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_chinese_report(result), encoding="utf-8")
    print(render_chinese_report(result))
    print(json_path)


if __name__ == "__main__":
    main()
