#!/usr/bin/env python3
"""Package the completed official SiT-S/2 replication into compact Git data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "official_sit_s2_recent_replication_v1"
)
DEFAULT_SOURCE_METADATA = Path(
    "/home/zhoushunyu/data/eqvae/models/SiT-official/"
    "SiT-S-2-256-imagenet100-subset.pt.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "docs/data/imagenet100_official_sit_s2_recent_replication"
)
DEFAULT_REPORT = REPO_ROOT / "docs/IMAGENET100_OFFICIAL_SIT_S2_REPLICATION_ZH.md"


@dataclass(frozen=True)
class ExperimentSpec:
    slug: str
    title: str
    summary: str
    run: str | None
    baseline_condition: str
    auxiliary_condition: str | None


SPECS = (
    ExperimentSpec(
        "final_linear_x",
        "末层线性 x 头",
        "fid/frozen-final-linear-x/frozen_v_clean_head_fid1k.json",
        "runs/frozen-final-linear-x",
        "velocity",
        "clean",
    ),
    ExperimentSpec(
        "depth8_v",
        "第 8 层 v 头",
        "fid/frozen-depth8-v/frozen_internal_velocity_head_fid1k.json",
        "runs/frozen-depth8-v",
        "full",
        "internal",
    ),
    ExperimentSpec(
        "depth8_x",
        "第 8 层 x 头",
        "fid/frozen-depth8-x/frozen_internal_clean_head_fid1k.json",
        "runs/frozen-depth8-x",
        "full",
        "internal",
    ),
    ExperimentSpec(
        "depth8_epsilon",
        "第 8 层 epsilon 头",
        "fid/frozen-depth8-epsilon/frozen_internal_epsilon_head_fid1k.json",
        "runs/frozen-depth8-epsilon",
        "full",
        "internal",
    ),
    ExperimentSpec(
        "hidden_state_mixing",
        "第 8/12 层 hidden/output 内外插",
        "fid/hidden-state-mixing/hidden_state_extrapolation_fid1k.json",
        None,
        "final",
        "internal_depth8",
    ),
    ExperimentSpec(
        "depth12_x_full",
        "第 12 层完整 x 头",
        "fid/frozen-depth12-x-full/frozen_internal_clean_head_fid1k.json",
        "runs/frozen-depth12-x-full",
        "full",
        "internal",
    ),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_pipeline(input_root: Path) -> dict[str, Any]:
    state = read_json(input_root / "pipeline_state.json")
    if state.get("status") != "complete":
        raise RuntimeError(f"replication pipeline is not complete: {state.get('status')!r}")
    incomplete = {
        name: stage.get("status")
        for name, stage in state.get("stages", {}).items()
        if stage.get("status") != "complete"
    }
    if incomplete:
        raise RuntimeError(f"pipeline contains incomplete stages: {incomplete}")
    expected = {
        "prepare_official_subset",
        "train_final_linear_x",
        "fid_final_linear_x",
        "train_depth8_v",
        "fid_depth8_v",
        "audit_hidden_gap",
        "train_depth8_x",
        "fid_depth8_x",
        "train_depth8_epsilon",
        "fid_depth8_epsilon",
        "fid_hidden_state_mixing",
        "train_depth12_x_full",
        "fid_depth12_x_full",
    }
    if set(state.get("stages", {})) != expected:
        raise RuntimeError("completed pipeline stage set differs from registered scope")
    return state


def portable_fid_rows(
    spec: ExperimentSpec,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if summary.get("comparison_is_paired") is not True:
        raise ValueError(f"{spec.slug} is not marked as a paired comparison")
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{spec.slug} has no FID rows")
    fingerprints = {
        (row.get("noise_fingerprint"), row.get("label_fingerprint")) for row in rows
    }
    if len(fingerprints) != 1:
        raise ValueError(f"{spec.slug} does not share noise and labels")
    portable: list[dict[str, Any]] = []
    for row in rows:
        values = {
            "experiment": spec.slug,
            "condition": row["condition"],
            "mode": row["mode"],
            "space": row.get("extrapolation_space", "output"),
            "gamma": float(row.get("gamma", 0.0)),
            "alpha": float(row.get("alpha", 0.0)),
            "num_samples": int(row["num_samples"]),
            "fid": float(row["fid"]),
            "sfid": float(row["sfid"]),
            "inception_score": float(row["inception_score"]),
            "total_nfe": int(row["total_nfe"]),
            "sampling_peak_memory_mib": int(row["sampling_peak_memory_mib"]),
            "fid_peak_memory_mib": int(row["fid_peak_memory_mib"]),
            "sample_sha256": row["sample_sha256"],
            "noise_fingerprint": row["noise_fingerprint"],
            "label_fingerprint": row["label_fingerprint"],
        }
        if not all(
            math.isfinite(float(values[key]))
            for key in ("fid", "sfid", "inception_score")
        ):
            raise ValueError(f"{spec.slug} contains a non-finite metric")
        portable.append(values)
    return portable


def training_summary(spec: ExperimentSpec, input_root: Path) -> dict[str, Any]:
    if spec.run is None:
        raise ValueError(f"{spec.slug} has no training run")
    run_root = input_root / spec.run
    run_config = read_json(run_root / "run_config.json")
    metrics = [
        json.loads(line)
        for line in (run_root / "train_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    train_rows = [row for row in metrics if any(key.startswith("train_") for key in row)]
    validation_rows = [row for row in metrics if "ema_validation" in row]
    if not train_rows or not validation_rows:
        raise ValueError(f"{spec.slug} has incomplete training metrics")
    train_key = next(key for key in train_rows[-1] if key.startswith("train_"))
    ema = validation_rows[-1]["ema_validation"]
    config = run_config["config"]
    return {
        "experiment": spec.slug,
        "prediction_target": config.get("prediction_target", "clean"),
        "internal_depth": config.get("internal_depth", 12),
        "trainable_parameters": int(run_config["trainable_parameter_count"]),
        "max_steps": int(config["max_steps"]),
        "global_batch_size": int(config["global_batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "loss_name": train_key,
        "first_train_loss": float(train_rows[0][train_key]),
        "final_train_loss": float(train_rows[-1][train_key]),
        "final_validation_step": int(validation_rows[-1]["step"]),
        "ema_native_mse": ema.get("internal_native_mse", ema.get("clean_mse")),
        "ema_velocity_mse": ema.get(
            "internal_velocity_mse", ema.get("clean_derived_velocity_mse")
        ),
        "frozen_velocity_mse": ema.get("frozen_velocity_mse"),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "source_step": int(config["source_step"]),
        "protocol": run_config["protocol"],
    }


def condition(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [row for row in rows if row["condition"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one condition {name!r}, found {len(matches)}")
    return matches[0]


def best(rows: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    matches = [row for row in rows if row["mode"] == mode]
    return min(matches, key=lambda row: row["fid"]) if matches else None


def plot_sweeps(
    rows_by_spec: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, spec in zip(axes.flat, SPECS):
        rows = rows_by_spec[spec.slug]
        baseline = condition(rows, spec.baseline_condition)
        axis.axhline(baseline["fid"], color="black", linestyle="--", label="baseline")
        for mode, marker, linestyle in (
            ("extrapolation", "o", "-"),
            ("interpolation", "s", ":"),
        ):
            mode_rows = [row for row in rows if row["mode"] == mode]
            spaces = sorted({row["space"] for row in mode_rows})
            for space in spaces:
                selected = [row for row in mode_rows if row["space"] == space]
                key = "gamma" if mode == "extrapolation" else "alpha"
                selected.sort(key=lambda row: row[key])
                if selected:
                    axis.plot(
                        [row[key] for row in selected],
                        [row["fid"] for row in selected],
                        marker=marker,
                        linestyle=linestyle,
                        label=f"{mode}-{space}",
                    )
        axis.set_xscale("log")
        axis.set_title(spec.slug)
        axis.set_xlabel("coefficient")
        axis.set_ylabel("FID-1K")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def preview_grid(
    input_root: Path,
    rows_by_spec: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    width, image_width, image_height, header = 1_170, 480, 240, 34
    canvas = Image.new(
        "RGB", (width, len(SPECS) * (image_height + header)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, spec in enumerate(SPECS):
        rows = rows_by_spec[spec.slug]
        baseline = condition(rows, spec.baseline_condition)
        treatment = best(rows, "extrapolation") or best(rows, "interpolation")
        if treatment is None:
            treatment = baseline
        summary_root = (input_root / spec.summary).parent
        y = row_index * (image_height + header)
        draw.text((5, y + 10), spec.slug, fill="black")
        for column, (label, selected) in enumerate(
            (("baseline", baseline), ("best treatment", treatment))
        ):
            x = 190 + column * 490
            draw.text(
                (x, y + 10),
                f"{label}: {selected['condition']} FID={selected['fid']:.3f}",
                fill="black",
            )
            preview = summary_root / selected["condition"] / "preview_rank_00.png"
            if not preview.is_file():
                raise FileNotFoundError(preview)
            image = Image.open(preview).convert("RGB")
            image.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y + header))
    canvas.save(output, optimize=True)


def markdown_report(
    *,
    state: dict[str, Any],
    scope: dict[str, Any],
    source: dict[str, Any],
    rows_by_spec: dict[str, list[dict[str, Any]]],
    training: list[dict[str, Any]],
    data_root: Path,
) -> str:
    sample_counts = {
        row["num_samples"] for rows in rows_by_spec.values() for row in rows
    }
    if len(sample_counts) != 1:
        raise ValueError(f"experiments use different sample counts: {sample_counts}")
    sample_count = next(iter(sample_counts))
    lines = [
        "# 官方 SiT-S/2 近期冻结头实验复现",
        "",
        "本报告只整理实验配置与结果，不作机制解释。大权重、生成样本和运行日志保留在本机 `/data`，不进入 Git。",
        "",
        "## 官方来源与适配",
        "",
        f"- 官方仓库：`{source['pretrained_source']['repository']}`",
        f"- 官方文件：`{source['pretrained_source']['filename']}`",
        f"- 原始权重 SHA256：`{source['raw_sha256']}`",
        f"- ImageNet-100 子集权重 SHA256：`{source['output_sha256']}`",
        f"- 类别与 unconditional 输出等价审计：`{source['equivalence_audit']['passed']}`，最大绝对误差分别为 "
        f"`{source['equivalence_audit']['class_output_max_abs']}` 和 "
        f"`{source['equivalence_audit']['unconditional_output_max_abs']}`。",
        "- 官方最终权重没有发布训练 step；实验记录中的 `source_step=0` 仅是带来源校验的哨兵值。",
        "",
        "## 复现范围",
        "",
        f"- 起始提交：`{scope['start_commit']}`",
        f"- 代码提交范围终点：`{scope['through_commit']}`",
        f"- 正式流程开始：`{state['started_at']}`",
        f"- 正式流程结束：`{state['finished_at']}`",
        "- 所有 FID 横向条件使用相同初始噪声、类别、VAE、ODE 和 ADM reference。",
        f"- 指标为单 seed、{sample_count:,} 张样本的配对 FID，不等同于正式 FID-50K。",
        "- 官方只发布一个最终 SiT-S/2 state dict，因此同一训练轨迹的 EMA 权重外推不可识别，未构造替代实验。",
        "",
        "## 训练结果",
        "",
        "| 实验 | 目标 | 深度 | 可训练参数 | step | 首个 loss | 最终 loss | EMA native MSE | EMA velocity MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training:
        lines.append(
            f"| {row['experiment']} | {row['prediction_target']} | {row['internal_depth']} | "
            f"{row['trainable_parameters']} | {row['max_steps']} | "
            f"{row['first_train_loss']:.6f} | {row['final_train_loss']:.6f} | "
            f"{float(row['ema_native_mse']):.6f} | {float(row['ema_velocity_mse']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"## 配对 FID-{sample_count}",
            "",
            "| 实验 | baseline | auxiliary | 最佳外推 | 相对 baseline | 最佳内插 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for spec in SPECS:
        rows = rows_by_spec[spec.slug]
        baseline = condition(rows, spec.baseline_condition)
        auxiliary = (
            condition(rows, spec.auxiliary_condition)
            if spec.auxiliary_condition is not None
            else None
        )
        extrapolation = best(rows, "extrapolation")
        interpolation = best(rows, "interpolation")
        auxiliary_text = f"{auxiliary['fid']:.4f}" if auxiliary else "-"
        extrapolation_text = (
            f"{extrapolation['condition']}: {extrapolation['fid']:.4f}"
            if extrapolation
            else "-"
        )
        delta_text = (
            f"{extrapolation['fid'] - baseline['fid']:+.4f}"
            if extrapolation
            else "-"
        )
        interpolation_text = (
            f"{interpolation['condition']}: {interpolation['fid']:.4f}"
            if interpolation
            else "-"
        )
        lines.append(
            f"| {spec.slug} | {baseline['fid']:.4f} | {auxiliary_text} | "
            f"{extrapolation_text} | {delta_text} | {interpolation_text} |"
        )
    try:
        relative = data_root.relative_to(REPO_ROOT)
    except ValueError:
        relative = data_root
    lines.extend(
        [
            "",
            "## 文件",
            "",
            f"- 全部逐条件指标：`{relative}/fid1k_all.csv`",
            f"- 训练摘要：`{relative}/training_summary.csv`",
            f"- 结构化结果：`{relative}/results.json`",
            f"- FID 曲线：`{relative}/fid1k_sweeps.png`",
            f"- 配对预览：`{relative}/preview_comparison.png`",
            f"- hidden gap 审计：`{relative}/hidden_state_gap_audit.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def package(args: argparse.Namespace) -> None:
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    report = args.report.expanduser().resolve()
    source_path = args.source_metadata.expanduser().resolve()
    state = validate_pipeline(input_root)
    scope = read_json(input_root / "replication_scope.json")
    source = read_json(source_path)

    temporary = output_root.with_name(f".{output_root.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    raw_root = temporary / "raw"
    raw_root.mkdir()

    all_rows: list[dict[str, Any]] = []
    rows_by_spec: dict[str, list[dict[str, Any]]] = {}
    training: list[dict[str, Any]] = []
    for spec in SPECS:
        source_summary = input_root / spec.summary
        summary = read_json(source_summary)
        rows = portable_fid_rows(spec, summary)
        rows_by_spec[spec.slug] = rows
        all_rows.extend(rows)
        shutil.copy2(source_summary, raw_root / f"{spec.slug}.json")
        csv_path = Path(str(summary["csv"])).resolve()
        if not csv_path.is_relative_to(input_root):
            raise ValueError(f"summary CSV escapes input root: {csv_path}")
        shutil.copy2(csv_path, raw_root / f"{spec.slug}.csv")
        if spec.run is not None:
            training.append(training_summary(spec, input_root))

    write_csv(temporary / "fid1k_all.csv", all_rows)
    write_csv(temporary / "training_summary.csv", training)
    shutil.copy2(input_root / "replication_scope.json", temporary)
    shutil.copy2(source_path, temporary / "official_checkpoint_metadata.json")
    audit_root = input_root / "audit/hidden-gap"
    shutil.copy2(audit_root / "hidden_state_gap_audit.csv", temporary)
    shutil.copy2(audit_root / "hidden_state_gap_audit.json", raw_root)

    results = {
        "format": "eqvae_official_sit_s2_recent_replication_portable_v1",
        "scope": scope,
        "pipeline": {
            "started_at": state["started_at"],
            "finished_at": state["finished_at"],
            "profile": state["profile"],
            "stage_status": {
                name: stage["status"] for name, stage in state["stages"].items()
            },
        },
        "official_checkpoint": source,
        "training": training,
        "fid1k_rows": all_rows,
    }
    write_json(temporary / "results.json", results)
    plot_sweeps(rows_by_spec, temporary / "fid1k_sweeps.png")
    preview_grid(input_root, rows_by_spec, temporary / "preview_comparison.png")
    (temporary / "README.md").write_text(
        "# Official SiT-S/2 replication data\n\n"
        "Compact metrics and figures for the post-3901741 experiments. "
        "Checkpoints, generated sample NPZ files, and logs remain on `/data`.\n",
        encoding="utf-8",
    )

    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        markdown_report(
            state=state,
            scope=scope,
            source=source,
            rows_by_spec=rows_by_spec,
            training=training,
            data_root=output_root,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "packaged",
                "output_root": str(output_root),
                "report": str(report),
                "fid_rows": len(all_rows),
                "training_runs": len(training),
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


if __name__ == "__main__":
    package(build_parser().parse_args())
