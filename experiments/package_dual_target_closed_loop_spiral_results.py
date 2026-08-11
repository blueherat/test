#!/usr/bin/env python3
"""Package compact spiral-toy evidence and lossless figures for Git.

Model checkpoints and raw sample tensors are deliberately excluded.  CSV/JSON
evidence is copied with its relative provenance, while every PNG is embedded
losslessly into one multi-page PDF with a page-to-source manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


SETTING_FILES = (
    "config.json",
    "evaluation_config.json",
    "complete.json",
    "train_history.csv",
    "endpoint_metrics.csv",
    "teacher_metrics.csv",
    "rollout_metrics.csv",
    "gradient_audit.csv",
    "branch_pair_metrics.csv",
    "cross_gate_endpoint_metrics.csv",
    "cross_gate_teacher_metrics.csv",
    "cross_gate_rollout_metrics.csv",
)

FIGURE_DESCRIPTIONS = {
    "endpoint_ambient_swd.png": "跨 seed 的闭环终点 full-D SWD 总览",
    "teacher_vs_rollout_bayes_error.png": "teacher-forced 与真实 rollout 的 Bayes 场误差",
    "spiral_endpoint_geometry_atlas.png": "连续螺旋的六类终点几何指标总览",
    "spiral_common_head_gate_controls.png": "固定 D0 双头后只替换 gate 的公平对照",
    "spiral_common_head_teacher_vs_rollout.png": "固定 D0 双头的 teacher/rollout Bayes 场误差",
    "endpoint_scatter.png": "参考螺旋与各方法闭环终点散点图",
    "mechanism_summary.png": "单 setting 的终点、teacher 与 rollout 机制摘要",
    "solver_step_convergence.png": "Heun 采样步数的数值收敛检查",
}


def collect_data_files(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for setting in sorted(source_root.glob("seed*/D*_H*")):
        for filename in SETTING_FILES:
            path = setting / filename
            if path.is_file():
                paths.append(path)
    aggregate = source_root / "aggregate"
    if aggregate.is_dir():
        paths.extend(sorted(aggregate.glob("*.csv")))
        paths.extend(sorted(aggregate.glob("*.json")))
    return sorted(set(paths), key=lambda path: path.relative_to(source_root).as_posix())


def collect_figure_files(source_root: Path) -> list[Path]:
    aggregate = sorted((source_root / "aggregate").glob("*.png"))
    settings: list[Path] = []
    for setting in sorted(source_root.glob("seed*/D*_H*")):
        for filename in ("endpoint_scatter.png", "mechanism_summary.png"):
            path = setting / filename
            if path.is_file():
                settings.append(path)
    return aggregate + settings


def describe_figure(path: Path, source_root: Path) -> str:
    description = FIGURE_DESCRIPTIONS.get(path.name, path.stem)
    relative = path.relative_to(source_root)
    if relative.parts and relative.parts[0].startswith("seed"):
        seed = relative.parts[0].removeprefix("seed")
        setting = relative.parts[1]
        return f"seed={seed}, {setting}: {description}"
    return description


def package_results(source_root: Path, destination_root: Path) -> dict:
    source_root = source_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    data_root = destination_root / "data"

    data_files = collect_data_files(source_root)
    figure_files = collect_figure_files(source_root)
    if not data_files:
        raise RuntimeError(f"no result tables found under {source_root}")
    if not figure_files:
        raise RuntimeError(f"no PNG figures found under {source_root}")
    if any(path.name == "checkpoint.pt" for path in data_files):
        raise AssertionError("checkpoint must never enter the compact package")

    for source in data_files:
        destination = data_root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    try:
        import img2pdf
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise RuntimeError("install img2pdf before packaging figures") from exc

    pdf_path = destination_root / "all_experiment_figures.pdf"
    pdf_path.write_bytes(img2pdf.convert(*[str(path) for path in figure_files]))

    manifest_path = destination_root / "figures_manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("page", "source_path", "description"))
        for page, path in enumerate(figure_files, 1):
            writer.writerow(
                (
                    page,
                    path.relative_to(source_root).as_posix(),
                    describe_figure(path, source_root),
                )
            )

    payload = {
        "source_root": str(source_root),
        "data_file_count": len(data_files),
        "figure_page_count": len(figure_files),
        "pdf": pdf_path.name,
        "figure_manifest": manifest_path.name,
        "excluded": ["checkpoint.pt", "raw sample tensors"],
    }
    (destination_root / "package_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package_results(args.source_root, args.destination_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
