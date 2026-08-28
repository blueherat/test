#!/usr/bin/env python3
"""Summarize and render matched ADM64 all-step local path-evidence runs.

The input is a completed ``observe_adm64_local_path_evidence.py`` directory.
This analysis never changes a sampler output.  It renders matched-seed
trajectories and fixed-timestep candidate timelines, and it keeps exploratory
candidate flags separate from formal model-relative bad-case labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


TIMELINE_TIMESTEPS = (249, 225, 201, 177, 153, 129, 105, 81, 65, 49, 25, 1)
CONTROL_COLORS = ("#4C78A8", "#707070")
CANDIDATE_COLOR = "#E17C05"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_candidates(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_constraints", {}).get("path_evidence_seen") is not False:
        raise RuntimeError("preselection must explicitly say path_evidence_seen=false")
    if payload.get("selection_constraints", {}).get("formal_endpoint") is not False:
        raise RuntimeError("this summarizer accepts exploratory preselection only")
    result: dict[int, int] = {}
    for block in payload.get("matched_seed_blocks", []):
        seed = int(block["seed"])
        class_id = int(block["candidate_class_id"])
        if seed in result:
            raise RuntimeError(f"duplicate candidate seed: {seed}")
        result[seed] = class_id
    if not result:
        raise RuntimeError("preselection contains no matched seed blocks")
    return result


def load_run(run_dir: Path, candidates: dict[int, int]) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, np.ndarray]]]:
    completion_path = run_dir / "completion.json"
    manifest_path = run_dir / "manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("local-Q run is incomplete")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise RuntimeError("completion marker is not complete")

    rows: list[dict[str, Any]] = []
    traces: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for signal_path in sorted((run_dir / "signals").glob("class_*/*.json")):
        signal = json.loads(signal_path.read_text(encoding="utf-8"))
        class_id = int(signal["class_id"])
        seed = int(signal["seed"])
        pair = (class_id, seed)
        trace_record = signal["trace"]
        trace_path = run_dir / trace_record["relative_path"]
        if sha256_file(trace_path) != trace_record["sha256"]:
            raise RuntimeError(f"trace hash mismatch: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        required = {
            "internal_timestep", "effective_nonidentity", "cumulative_log_e",
            "L_scalar", "tile_bounds_yxyx", "selected_energy_fraction",
            "pred_xstart", "L_map", "K_scalar",
        }
        if not required.issubset(trace):
            raise RuntimeError(f"trace is missing required arrays: {pair}")
        if trace["internal_timestep"].tolist() != list(range(249, 0, -1)):
            raise RuntimeError(f"unexpected all-step time order: {pair}")
        effective = trace["effective_nonidentity"].astype(bool)
        if not effective.any():
            raise RuntimeError(f"no effective local-Q steps: {pair}")
        cumulative = trace["cumulative_log_e"].astype(np.float64)
        max_from_e0 = max(0.0, float(cumulative.max()))
        effective_indices = np.flatnonzero(effective)
        effective_max_index = int(effective_indices[np.argmax(cumulative[effective])])
        top_increment_index = int(effective_indices[np.argmax(trace["L_scalar"][effective])])
        is_candidate = candidates.get(seed) == class_id
        row = {
            "class_id": class_id,
            "seed": seed,
            "exploratory_candidate": is_candidate,
            "formal_relative_bad": "not_evaluated",
            "total_K_budget": float(signal["summary"]["total_K_budget"]),
            "total_applied_K": float(trace["K_scalar"].sum(dtype=np.float64)),
            "final_log_e": float(cumulative[-1]),
            "max_log_e_from_E0": max_from_e0,
            "effective_max_log_e": float(cumulative[effective_max_index]),
            "effective_max_internal_t": int(trace["internal_timestep"][effective_max_index]),
            "effective_max_reverse_index": effective_max_index,
            "top_increment": float(trace["L_scalar"][top_increment_index]),
            "top_increment_internal_t": int(trace["internal_timestep"][top_increment_index]),
            "top_increment_tile_yxyx": [
                int(value) for value in trace["tile_bounds_yxyx"][top_increment_index]
            ],
            "median_selected_energy_fraction": float(
                np.median(trace["selected_energy_fraction"][effective])
            ),
            "crossed_alpha_0p05": bool(max_from_e0 >= -math.log(0.05)),
            "crossed_alpha_0p10": bool(max_from_e0 >= -math.log(0.10)),
            "crossed_alpha_0p20": bool(max_from_e0 >= -math.log(0.20)),
        }
        rows.append(row)
        traces[pair] = trace

    actual_seeds = {int(row["seed"]) for row in rows}
    if actual_seeds != set(candidates):
        raise RuntimeError(
            f"run/preselection seed mismatch: {sorted(actual_seeds)} != {sorted(candidates)}"
        )
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed, block in by_seed.items():
        if len(block) != 3 or sum(bool(row["exploratory_candidate"]) for row in block) != 1:
            raise RuntimeError(f"seed {seed} is not one candidate plus two controls")
    return rows, traces


def matched_diagnostics(rows: list[dict[str, Any]], traces: dict[tuple[int, int], dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    diagnostics = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        block = [row for row in rows if int(row["seed"]) == seed]
        candidate = next(row for row in block if row["exploratory_candidate"])
        controls = [row for row in block if not row["exploratory_candidate"]]
        curves = [
            traces[(int(row["class_id"]), seed)]["cumulative_log_e"].astype(np.float64)
            for row in block
        ]
        correlations = []
        for left in range(3):
            for right in range(left + 1, 3):
                correlations.append(float(np.corrcoef(curves[left], curves[right])[0, 1]))
        control_max = [float(row["max_log_e_from_E0"]) for row in controls]
        candidate_max = float(candidate["max_log_e_from_E0"])
        diagnostics.append(
            {
                "seed": seed,
                "candidate_class_id": int(candidate["class_id"]),
                "candidate_max_log_e": candidate_max,
                "control_max_log_e": control_max,
                "candidate_minus_control_mean_max_log_e": candidate_max - float(np.mean(control_max)),
                "candidate_rank_among_three_by_max_log_e": 1
                + sum(float(row["max_log_e_from_E0"]) > candidate_max for row in controls),
                "within_seed_curve_correlations": correlations,
                "within_seed_min_curve_correlation": min(correlations),
                "candidate_enriched_in_this_block": bool(
                    candidate_max > max(control_max)
                ),
            }
        )
    return diagnostics


def render_curves(rows: list[dict[str, Any]], traces: dict[tuple[int, int], dict[str, np.ndarray]], output: Path) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    figure, axes = plt.subplots(len(seeds), 1, figsize=(10, 3.2 * len(seeds)), sharex=True)
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        block = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: (not bool(row["exploratory_candidate"]), int(row["class_id"])),
        )
        control_index = 0
        for row in block:
            pair = (int(row["class_id"]), seed)
            cumulative = traces[pair]["cumulative_log_e"].astype(np.float64)
            reverse_index = np.arange(cumulative.size)
            if row["exploratory_candidate"]:
                color, style, width = CANDIDATE_COLOR, "-", 2.5
                label = f"class {pair[0]} candidate"
            else:
                color = CONTROL_COLORS[control_index]
                style = ("--", ":")[control_index]
                width = 1.6
                label = f"class {pair[0]} control"
                control_index += 1
            axis.plot(reverse_index, cumulative, color=color, linestyle=style, linewidth=width, label=label)
        axis.axhline(0.0, color="#303030", linewidth=0.8)
        axis.axhline(-math.log(0.05), color="#303030", linewidth=1.0, linestyle="-.", label="log 20 threshold")
        axis.set_title(f"Shared innovation seed {seed}", loc="left", fontsize=11)
        axis.set_ylabel("cumulative log E")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    axes[-1].set_xlabel("reverse transition index (0 = noisiest, 248 = t=1)")
    figure.suptitle(
        "ADM64 localized operational path evidence — matched innovation blocks",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.005,
        "Orange is the visually preselected discovery candidate; labels are not formal bad-case endpoints.",
        ha="center",
        fontsize=9,
        color="#404040",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_candidate_timeline(row: dict[str, Any], trace: dict[str, np.ndarray], output: Path) -> None:
    timesteps = trace["internal_timestep"].astype(int)
    indices = [int(np.flatnonzero(timesteps == timestep)[0]) for timestep in TIMELINE_TIMESTEPS]
    selected_maps = trace["L_map"][indices].astype(np.float64)
    nonzero = np.abs(selected_maps[selected_maps != 0])
    limit = float(np.quantile(nonzero, 0.995)) if nonzero.size else 1.0
    limit = max(limit, 1e-12)
    figure, axes = plt.subplots(2, len(indices), figsize=(2.0 * len(indices), 4.5))
    for column, (index, timestep) in enumerate(zip(indices, TIMELINE_TIMESTEPS)):
        image = np.clip((trace["pred_xstart"][index].transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
        axes[0, column].imshow(image)
        y0, x0, y1, x1 = [int(value) for value in trace["tile_bounds_yxyx"][index]]
        axes[0, column].add_patch(
            patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=CANDIDATE_COLOR, linewidth=1.5)
        )
        axes[0, column].set_title(f"t={timestep}\nlogE={trace['cumulative_log_e'][index]:.2f}", fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].imshow(trace["L_map"][index], cmap="PuOr_r", vmin=-limit, vmax=limit)
        axes[1, column].add_patch(
            patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#202020", linewidth=1.0)
        )
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("predicted x0", fontsize=9)
    axes[1, 0].set_ylabel("local L map", fontsize=9)
    figure.suptitle(
        f"Candidate class {row['class_id']} / seed {row['seed']} — fixed-timestep audit",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Orange/black boxes are selected predictably before the transition innovation; PuOr scale is shared within this figure.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preselection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    preselection = args.preselection.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty analysis directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(preselection)
    rows, traces = load_run(run_dir, candidates)
    diagnostics = matched_diagnostics(rows, traces)
    csv_path = output_dir / "path_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_curves(rows, traces, output_dir / "matched_log_e_curves.png")
    timeline_files = []
    for row in rows:
        if not row["exploratory_candidate"]:
            continue
        filename = f"candidate_class_{int(row['class_id']):04d}_seed_{int(row['seed']):019d}_timeline.png"
        render_candidate_timeline(
            row,
            traces[(int(row["class_id"]), int(row["seed"]))],
            output_dir / filename,
        )
        timeline_files.append(filename)
    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "run_completion_sha256": sha256_file(run_dir / "completion.json"),
        "preselection": str(preselection),
        "preselection_sha256": sha256_file(preselection),
        "formal_bad_endpoint_used": False,
        "candidate_count": sum(bool(row["exploratory_candidate"]) for row in rows),
        "path_count": len(rows),
        "matched_diagnostics": diagnostics,
        "candidate_enriched_block_count": sum(
            bool(record["candidate_enriched_in_this_block"]) for record in diagnostics
        ),
        "outputs": {
            "path_summary_csv": "path_summary.csv",
            "matched_curves": "matched_log_e_curves.png",
            "candidate_timelines": timeline_files,
        },
        "interpretation_guard": (
            "This matched nine-path discovery diagnostic cannot estimate TPR, FPR, "
            "artifact prevalence, or image-quality improvement."
        ),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
