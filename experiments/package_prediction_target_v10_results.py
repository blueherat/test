#!/usr/bin/env python3
"""Archive the reviewable v4/v10 prediction-target results for Git."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import img2pdf


V10_SEEDS = (20260807, 20260808, 20260809)
EXPECTED_CONDITIONS = 187
ATLAS_DESCRIPTIONS = (
    "baseline：参考分布与 eps、v、x 三种预测目标的端点分布。",
    "x-eps 路径：从 eps 到 x 以及越过 x 的原始外推结果。",
    "x-v 路径：从 v 到 x 以及越过 x 的原始外推结果。",
    "v4 回归：固定绝对 RMS 动作强度的正负方向对照。",
    "分布分解：沿螺旋覆盖率与有符号 ridge 法向分布。",
    "配对位移：同一初始噪声下从 x 端点到外推端点的位移场。",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"pdfinfo did not report a page count for {path}")


def describe_v4(path: Path) -> str:
    if path.name == "swd_manifold_tradeoff.png":
        return "v4 汇总：生成 SWD 与流形偏离之间的权衡。"
    if path.name == "extrapolation_phase_diagram.png":
        return "v4 汇总：不同预测目标外推条件的相图。"
    if path.name == "generation_scatter.png":
        return "v4 单配置：参考分布、各预测目标及外推结果的端点散点图。"
    return "v4 预测目标外推实验图。"


def find_v4_pngs(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root in sorted(repo_root.glob("prediction_target_toy_v4_*")):
        if root.is_dir():
            paths.extend(sorted(root.rglob("*.png")))
    if not paths:
        raise FileNotFoundError("no archived prediction_target_toy_v4_* PNG files found")
    return paths


def setting_dir(v10_root: Path, seed: int) -> Path:
    path = (
        v10_root
        / f"worker_seed{seed}"
        / f"seed{seed}"
        / "D512"
        / "curv1"
        / "H1024"
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def validate_v10_results(v10_root: Path) -> None:
    complete = v10_root / "COMPLETE"
    if not complete.is_file():
        raise FileNotFoundError(f"v10 run is not complete: {complete}")

    combined = v10_root / "generation_metrics_v10_all_seeds.csv"
    with combined.open(newline="", encoding="utf-8") as handle:
        combined_rows = list(csv.DictReader(handle))
    expected_total = EXPECTED_CONDITIONS * len(V10_SEEDS)
    if len(combined_rows) != expected_total:
        raise ValueError(
            f"combined v10 metrics contain {len(combined_rows)} rows; "
            f"expected {expected_total}"
        )
    seeds = {int(float(row["seed"])) for row in combined_rows}
    if seeds != set(V10_SEEDS):
        raise ValueError(f"unexpected independent seeds: {sorted(seeds)}")
    if {row["training_source"] for row in combined_rows} != {"reused_v4_checkpoint"}:
        raise ValueError("v10 archive must contain only reused v4 checkpoints")

    signatures_by_seed: list[set[tuple[str, ...]]] = []
    signature_fields = ("condition", "kind", "strength", "t_low", "t_high")
    for seed in V10_SEEDS:
        rows = [row for row in combined_rows if int(float(row["seed"])) == seed]
        signatures = {
            tuple(row.get(field, "") for field in signature_fields) for row in rows
        }
        if len(rows) != EXPECTED_CONDITIONS or len(signatures) != EXPECTED_CONDITIONS:
            raise ValueError(
                f"seed {seed} has {len(rows)} rows and {len(signatures)} unique conditions"
            )
        signatures_by_seed.append(signatures)
    if any(signatures != signatures_by_seed[0] for signatures in signatures_by_seed[1:]):
        raise ValueError("v10 condition sets differ across seeds")

    replay = v10_root / "replay_seed20260807" / "generation_metrics_v10_all.csv"
    worker = v10_root / "worker_seed20260807" / "generation_metrics_v10_all.csv"
    if not replay.is_file() or sha256(replay) != sha256(worker):
        raise ValueError("seed-20260807 deterministic replay does not match its worker")


def copy_review_data(v10_root: Path, archive_root: Path) -> list[Path]:
    copied: list[Path] = []
    archive_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "generation_metrics_v10_all_seeds.csv",
        "generation_metrics_v10_seed_summary.csv",
        "aggregate_manifest_v10.json",
    ):
        source = v10_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = archive_root / name
        shutil.copy2(source, target)
        copied.append(target)

    for seed in V10_SEEDS:
        source_setting = setting_dir(v10_root, seed)
        target_seed = archive_root / f"seed{seed}"
        target_seed.mkdir(parents=True, exist_ok=True)
        worker_manifest = v10_root / f"worker_seed{seed}" / "manifest_v10.json"
        sources = (
            worker_manifest,
            source_setting / "generation_metrics_v10.csv",
            source_setting / "trajectory_mechanism_v10.csv",
            source_setting / "setting_manifest_v10.json",
            source_setting / "embedding_observability.json",
        )
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            target = target_seed / source.name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def build_combined_pdf(
    *,
    repo_root: Path,
    v10_root: Path,
    output_pdf: Path,
    manifest_path: Path,
) -> int:
    v4_pngs = find_v4_pngs(repo_root)
    manifest_rows: list[dict[str, object]] = []
    page = 1

    with tempfile.TemporaryDirectory(prefix="prediction_target_pdf_") as temp_name:
        temp = Path(temp_name)
        inputs: list[Path] = []

        v4_pdf = temp / "v4_pngs.pdf"
        v4_pdf.write_bytes(img2pdf.convert(*[str(path) for path in v4_pngs]))
        inputs.append(v4_pdf)
        for path in v4_pngs:
            manifest_rows.append(
                {
                    "page": page,
                    "experiment": "v4",
                    "seed": "",
                    "source": path.relative_to(repo_root).as_posix(),
                    "description": describe_v4(path),
                }
            )
            page += 1

        for seed in V10_SEEDS:
            source_setting = setting_dir(v10_root, seed)
            atlas = source_setting / "visual_paradox_atlas_v10.pdf"
            precision = source_setting / "precision_coverage_v10.png"
            if pdf_pages(atlas) != len(ATLAS_DESCRIPTIONS):
                raise ValueError(f"unexpected v10 atlas page count: {atlas}")
            inputs.append(atlas)
            source_prefix = (
                f"local-run/worker_seed{seed}/seed{seed}/D512/curv1/H1024"
            )
            for atlas_page, description in enumerate(ATLAS_DESCRIPTIONS, 1):
                manifest_rows.append(
                    {
                        "page": page,
                        "experiment": "v10_full_mechanism",
                        "seed": seed,
                        "source": f"{source_prefix}/visual_paradox_atlas_v10.pdf#page={atlas_page}",
                        "description": description,
                    }
                )
                page += 1

            precision_pdf = temp / f"precision_seed{seed}.pdf"
            precision_pdf.write_bytes(img2pdf.convert(str(precision)))
            inputs.append(precision_pdf)
            manifest_rows.append(
                {
                    "page": page,
                    "experiment": "v10_full_mechanism",
                    "seed": seed,
                    "source": f"{source_prefix}/precision_coverage_v10.png",
                    "description": "完整 187 条件的 coverage W1 与 conditional ridge W1 对照图。",
                }
            )
            page += 1

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.unlink(missing_ok=True)
        subprocess.run(
            ["pdfunite", *[str(path) for path in inputs], str(output_pdf)],
            check=True,
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("page", "experiment", "seed", "source", "description"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    expected_pages = len(manifest_rows)
    actual_pages = pdf_pages(output_pdf)
    if actual_pages != expected_pages:
        raise ValueError(f"combined PDF has {actual_pages} pages, expected {expected_pages}")
    return actual_pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--v10-root",
        type=Path,
        default=Path(
            "/home/zhoushunyu/data/eqvae/experiments/"
            "prediction_target_toy_v10_final_full_mechanism"
        ),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("prediction_target_toy_v10_final_full_mechanism"),
    )
    parser.add_argument("--output-pdf", type=Path, default=Path("all_experiment_figures.pdf"))
    parser.add_argument("--manifest", type=Path, default=Path("figures_manifest.tsv"))
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    v10_root = args.v10_root.expanduser().resolve()
    archive_root = (repo_root / args.archive_root).resolve()
    output_pdf = (repo_root / args.output_pdf).resolve()
    manifest = (repo_root / args.manifest).resolve()

    validate_v10_results(v10_root)
    copied = copy_review_data(v10_root, archive_root)
    pages = build_combined_pdf(
        repo_root=repo_root,
        v10_root=v10_root,
        output_pdf=output_pdf,
        manifest_path=manifest,
    )
    archive_manifest = {
        "format": "prediction_target_v10_git_archive_v1",
        "v10_source": str(v10_root),
        "training_source": "reused_v4_checkpoint",
        "independent_seeds": list(V10_SEEDS),
        "conditions_per_seed": EXPECTED_CONDITIONS,
        "excluded": ["model checkpoints", "per-condition NPZ samples", "partial CSVs", "deterministic replay"],
        "combined_pdf": output_pdf.relative_to(repo_root).as_posix(),
        "combined_pdf_pages": pages,
        "combined_pdf_sha256": sha256(output_pdf),
        "figure_manifest": manifest.relative_to(repo_root).as_posix(),
        "files": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(copied)
        ],
    }
    manifest_out = archive_root / "archive_manifest.json"
    manifest_out.write_text(
        json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive_root": str(archive_root),
                "copied_files": len(copied),
                "pdf": str(output_pdf),
                "pdf_pages": pages,
                "pdf_bytes": output_pdf.stat().st_size,
                "figure_manifest": str(manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
