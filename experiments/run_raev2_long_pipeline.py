"""Resume-safe RAEv2 Flow/LPL training, sampling, evaluation, and plotting.

The pipeline is intentionally sequential: all four GPUs train Flow, then LPL,
then sample every comparable checkpoint.  Every stage discovers completed
artifacts before doing work, so relaunching the same command resumes instead of
starting over.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_training_core import file_sha256  # noqa: E402


DEFAULT_DATA_ROOT = Path("/home/zhoushunyu/data/eqvae")
CHECKPOINT_RE = re.compile(r"branch-(\d+)-global-(\d+)\.pt$")


@dataclass(frozen=True)
class BranchSpec:
    name: str
    objective: str
    experiment_name: str
    seed_checkpoint_dirs: tuple[Path, ...]


def checkpoint_branch_step(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    if match is None:
        raise ValueError(f"not a branch checkpoint name: {path}")
    return int(match.group(1))


def checkpoint_global_step(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    if match is None:
        raise ValueError(f"not a branch checkpoint name: {path}")
    return int(match.group(2))


def checkpoint_steps(target_step: int, every: int) -> tuple[int, ...]:
    if target_step <= 0 or every <= 0 or target_step % every:
        raise ValueError("target_step must be a positive multiple of every")
    return tuple(range(every, target_step + 1, every))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_event(path: Path, event: str, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now().astimezone().isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_checkpoint(path: Path, *, objective: str | None = None) -> dict:
    import torch

    checkpoint = torch.load(
        path.expanduser(),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    required = {"step", "epoch", "model", "ema", "optimizer", "scheduler", "raev2_lpl"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"{path} is missing checkpoint keys: {sorted(missing)}")
    metadata = checkpoint["raev2_lpl"]
    branch_update = int(metadata["branch_update"])
    if branch_update != checkpoint_branch_step(path):
        raise ValueError(f"{path} filename and metadata branch steps disagree")
    if int(checkpoint["step"]) != checkpoint_global_step(path):
        raise ValueError(f"{path} filename and checkpoint global steps disagree")
    if objective is not None and metadata["objective"] != objective:
        raise ValueError(
            f"{path} objective is {metadata['objective']!r}, expected {objective!r}"
        )
    optimizer = checkpoint["optimizer"]
    learning_rates = [
        float(group["lr"])
        for name in ("muon", "adamw")
        for group in optimizer[name]["param_groups"]
    ]
    if learning_rates != [2e-5, 2e-5]:
        raise ValueError(f"{path} has unexpected optimizer LRs: {learning_rates}")
    if int(checkpoint["scheduler"]["last_epoch"]) != int(checkpoint["step"]):
        raise ValueError(f"{path} scheduler step does not match checkpoint step")
    result = {
        "path": str(path.resolve()),
        "branch_update": branch_update,
        "global_step": int(checkpoint["step"]),
        "objective": metadata["objective"],
        "learning_rates": learning_rates,
    }
    del checkpoint
    return result


def valid_checkpoints(
    directories: Iterable[Path],
    *,
    objective: str,
    event_log: Path,
) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for directory in directories:
        for path in sorted((directory / "checkpoints").glob("branch-*-global-*.pt")):
            try:
                metadata = validate_checkpoint(path, objective=objective)
            except Exception as error:
                append_event(
                    event_log,
                    "checkpoint_rejected",
                    checkpoint=str(path),
                    error=repr(error),
                )
                continue
            step = int(metadata["branch_update"])
            current = found.get(step)
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
                found[step] = path
    return found


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    event_log: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_event(event_log, "command_start", command=command, log=str(log_path))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
        handle.write(f"### {datetime.now().astimezone().isoformat()}\n")
        handle.write(" ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    append_event(
        event_log,
        "command_end",
        command=command,
        returncode=int(completed.returncode),
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )


def torchrun_prefix(python: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
    ]


def train_branch(
    spec: BranchSpec,
    *,
    target_step: int,
    checkpoint_every: int,
    python: Path,
    config: Path,
    data_path: Path,
    index_map: Path,
    source_checkpoint: Path,
    results_root: Path,
    dino_repo: Path,
    lpl_weight: float,
    min_free_gib: float,
    logs: Path,
    event_log: Path,
    env: dict[str, str],
) -> Path:
    experiment_dir = results_root / spec.experiment_name
    candidates = valid_checkpoints(
        (*spec.seed_checkpoint_dirs, experiment_dir),
        objective=spec.objective,
        event_log=event_log,
    )
    completed = candidates.get(target_step)
    if completed is not None:
        append_event(
            event_log,
            "training_skipped",
            branch=spec.name,
            checkpoint=str(completed),
        )
        return completed
    if not candidates:
        raise FileNotFoundError(f"no resumable checkpoint found for {spec.name}")
    resume_step = max(step for step in candidates if step < target_step)
    resume = candidates[resume_step]
    command = [
        *torchrun_prefix(python),
        "experiments/train_raev2_strict_lpl.py",
        "--config",
        str(config),
        "--data-path",
        str(data_path),
        "--index-map",
        str(index_map),
        "--results-dir",
        str(results_root),
        "--experiment-name",
        spec.experiment_name,
        "--source-checkpoint",
        str(source_checkpoint),
        "--resume",
        str(resume),
        "--objective",
        spec.objective,
        "--max-updates",
        str(target_step),
        "--save-every",
        str(checkpoint_every),
        "--num-workers",
        "4",
        "--min-free-gib",
        str(float(min_free_gib)),
        "--dino-repo-dir",
        str(dino_repo),
    ]
    if spec.objective == "lpl":
        command.extend(
            [
                "--lpl-weight",
                repr(float(lpl_weight)),
                "--lpl-noise-threshold",
                "3.0",
                "--lpl-max-samples-per-rank",
                "1",
            ]
        )
    run_logged(
        command,
        log_path=logs / f"train_{spec.name}_to_{target_step}.log",
        event_log=event_log,
        env=env,
    )
    refreshed = valid_checkpoints(
        (*spec.seed_checkpoint_dirs, experiment_dir),
        objective=spec.objective,
        event_log=event_log,
    )
    if target_step not in refreshed:
        raise RuntimeError(f"{spec.name} training ended without step {target_step}")
    return refreshed[target_step]


def sampling_complete(directory: Path, *, samples: int, checkpoint: Path) -> bool:
    summary_path = directory / "sampling_summary.json"
    archive = directory / "samples.npz"
    if not summary_path.exists() or not archive.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(summary.get("samples", -1)) == int(samples)
        and summary.get("state_key") == "ema"
        and Path(summary.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and Path(summary.get("archive", "")).resolve() == archive.resolve()
        and summary.get("archive_sha256") == file_sha256(archive)
        and archive.stat().st_size > 0
    )


def verify_same_noise_protocol(
    sample_directories: Mapping[str, Path],
    *,
    world_size: int = 4,
) -> dict[str, dict[str, str]]:
    """Verify that every branch used identical RNG states, noise, and labels."""

    invariant_keys = (
        "protocol",
        "world_size",
        "sampling_seed",
        "sample_count",
        "per_rank_batch",
        "sampler_steps",
        "guidance_cfg_scale",
        "guidance_ig_scale",
        "guidance_ig_t_min",
        "initial_generator_sha256",
        "first_noise_sha256",
        "first_label_sha256",
        "first_labels",
        "final_generator_sha256",
    )
    reference: dict[int, dict] = {}
    fingerprints: dict[str, dict[str, str]] = {}
    for branch, directory in sample_directories.items():
        branch_fingerprints = {}
        for rank in range(int(world_size)):
            path = directory / f"sampling_audit_rank{rank}.json"
            audit = json.loads(path.read_text(encoding="utf-8"))
            if audit.get("protocol") != "raev2_same_noise_v1":
                raise ValueError(f"{path} has an unsupported sampling protocol")
            comparable = {key: audit.get(key) for key in invariant_keys}
            if rank not in reference:
                reference[rank] = comparable
            elif comparable != reference[rank]:
                changed = [
                    key
                    for key in invariant_keys
                    if comparable[key] != reference[rank][key]
                ]
                raise ValueError(
                    f"{branch} rank {rank} is not same-noise comparable; "
                    f"different fields: {changed}"
                )
            branch_fingerprints[f"rank{rank}_noise"] = str(
                audit["first_noise_sha256"]
            )
            branch_fingerprints[f"rank{rank}_labels"] = str(
                audit["first_label_sha256"]
            )
        fingerprints[branch] = branch_fingerprints
    return fingerprints


def sample_branch(
    name: str,
    checkpoint: Path,
    *,
    python: Path,
    config: Path,
    samples_root: Path,
    sample_count: int,
    per_rank_batch: int,
    dino_repo: Path,
    logs: Path,
    event_log: Path,
    env: dict[str, str],
) -> Path:
    output_dir = samples_root / name
    if sampling_complete(output_dir, samples=sample_count, checkpoint=checkpoint):
        append_event(event_log, "sampling_skipped", branch=name)
        return output_dir / "samples.npz"
    command = [
        *torchrun_prefix(python),
        "experiments/sample_raev2_threeway.py",
        "--config",
        str(config),
        "--branch",
        f"{name}={checkpoint}",
        "--results-dir",
        str(samples_root),
        "--sample-count",
        str(sample_count),
        "--per-rank-batch",
        str(per_rank_batch),
        "--sampling-seed",
        "0",
        "--precision",
        "bf16",
        "--state-key",
        "ema",
        "--dino-repo-dir",
        str(dino_repo),
    ]
    run_logged(
        command,
        log_path=logs / f"sample_{name}.log",
        event_log=event_log,
        env=env,
    )
    if not sampling_complete(output_dir, samples=sample_count, checkpoint=checkpoint):
        raise RuntimeError(f"sampling for {name} did not produce a valid summary")
    return output_dir / "samples.npz"


def write_curve(
    metrics_csv: Path,
    *,
    output_csv: Path,
    output_png: Path,
) -> None:
    frame = pd.read_csv(metrics_csv)
    rows = []
    for row in frame.to_dict(orient="records"):
        name = str(row["branch"])
        if name == "official":
            objective, branch_update = "official", 0
        else:
            match = re.fullmatch(r"(flow|lpl)_s(\d+)", name)
            if match is None:
                raise ValueError(f"unexpected branch name in metrics: {name}")
            objective, branch_update = match.group(1), int(match.group(2))
        rows.append({**row, "objective": objective, "branch_update": branch_update})
    result = pd.DataFrame(rows).sort_values(["objective", "branch_update"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("frechet_inception_distance", "FID", "lower"),
        ("kernel_inception_distance_mean", "KID", "lower"),
        ("inception_score_mean", "Inception Score", "higher"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    official = result[result["objective"] == "official"].iloc[0]
    colors = {"flow": "#2563eb", "lpl": "#dc2626"}
    for axis, (column, title, direction) in zip(axes, metrics, strict=True):
        for objective in ("flow", "lpl"):
            subset = result[result["objective"] == objective].sort_values(
                "branch_update"
            )
            axis.plot(
                subset["branch_update"],
                subset[column],
                marker="o",
                linewidth=2,
                markersize=4,
                label=objective.upper(),
                color=colors[objective],
            )
        axis.axhline(
            float(official[column]),
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            label="Official EMA",
        )
        axis.set_title(f"{title} ({direction} is better)")
        axis.set_xlabel("Continuation optimizer steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "RAEv2 DINOv3-L-K7: Flow vs decoder-feature LPL, 5k samples",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_flow_curve(
    metrics_csv: Path,
    *,
    output_csv: Path,
    output_png: Path,
) -> None:
    """Write a Flow-only continuation curve against the official EMA."""

    frame = pd.read_csv(metrics_csv)
    rows = []
    for row in frame.to_dict(orient="records"):
        name = str(row["branch"])
        if name == "official":
            branch_update = 0
        else:
            match = re.fullmatch(r"flow_s(\d+)", name)
            if match is None:
                raise ValueError(f"unexpected Flow-only branch name: {name}")
            branch_update = int(match.group(1))
        rows.append({**row, "branch_update": branch_update})
    result = pd.DataFrame(rows).sort_values("branch_update")
    if result["branch_update"].duplicated().any():
        raise ValueError("Flow-only metrics contain duplicate continuation steps")
    official_rows = result[result["branch"] == "official"]
    if len(official_rows) != 1:
        raise ValueError("Flow-only metrics require exactly one official row")
    flow = result[result["branch"] != "official"]
    if flow.empty:
        raise ValueError("Flow-only metrics contain no continuation checkpoints")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("frechet_inception_distance", "FID", "lower"),
        ("kernel_inception_distance_mean", "KID", "lower"),
        ("inception_score_mean", "Inception Score", "higher"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    official = official_rows.iloc[0]
    for axis, (column, title, direction) in zip(axes, metrics, strict=True):
        axis.plot(
            flow["branch_update"],
            flow[column],
            marker="o",
            linewidth=2,
            markersize=5,
            label="Flow continuation",
            color="#2563eb",
        )
        axis.axhline(
            float(official[column]),
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            label="Official EMA",
        )
        axis.set_title(f"{title} ({direction} is better)")
        axis.set_xlabel("Continuation optimizer steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "RAEv2 DINOv3-L-K7: Flow continuation, 5k samples",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_lpl_curve(
    metrics_csv: Path,
    *,
    output_csv: Path,
    output_png: Path,
) -> None:
    """Write an LPL-only continuation curve against the official EMA."""

    frame = pd.read_csv(metrics_csv)
    rows = []
    for row in frame.to_dict(orient="records"):
        name = str(row["branch"])
        if name == "official":
            branch_update = 0
        else:
            match = re.fullmatch(r"lpl_s(\d+)", name)
            if match is None:
                raise ValueError(f"unexpected LPL-only branch name: {name}")
            branch_update = int(match.group(1))
        rows.append({**row, "branch_update": branch_update})
    result = pd.DataFrame(rows).sort_values("branch_update")
    if result["branch_update"].duplicated().any():
        raise ValueError("LPL-only metrics contain duplicate continuation steps")
    official_rows = result[result["branch"] == "official"]
    if len(official_rows) != 1:
        raise ValueError("LPL-only metrics require exactly one official row")
    lpl = result[result["branch"] != "official"]
    if lpl.empty:
        raise ValueError("LPL-only metrics contain no continuation checkpoints")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("frechet_inception_distance", "FID", "lower"),
        ("kernel_inception_distance_mean", "KID", "lower"),
        ("inception_score_mean", "Inception Score", "higher"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    official = official_rows.iloc[0]
    for axis, (column, title, direction) in zip(axes, metrics, strict=True):
        axis.plot(
            lpl["branch_update"],
            lpl[column],
            marker="o",
            linewidth=2,
            markersize=5,
            label="LPL continuation",
            color="#dc2626",
        )
        axis.axhline(
            float(official[column]),
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            label="Official EMA",
        )
        axis.set_title(f"{title} ({direction} is better)")
        axis.set_xlabel("Continuation optimizer steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "RAEv2 DINOv3-L-K7: LPL continuation, 5k samples",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluation_complete(
    metrics_csv: Path,
    *,
    branch: str,
    sample_directory: Path,
) -> bool:
    if not metrics_csv.exists():
        return False
    summary_path = sample_directory / "sampling_summary.json"
    if not summary_path.exists():
        return False
    try:
        frame = pd.read_csv(metrics_csv)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if len(frame) != 1 or str(frame.iloc[0].get("branch")) != branch:
        return False
    return (
        str(frame.iloc[0].get("sample_sha256")) == summary.get("archive_sha256")
        and int(frame.iloc[0].get("sample_count", -1))
        == int(summary.get("samples", -2))
    )


def evaluate_branch(
    name: str,
    archive: Path,
    *,
    python: Path,
    pipeline_root: Path,
    logs: Path,
    event_log: Path,
    env: dict[str, str],
) -> Path:
    output = pipeline_root / "metrics" / f"{name}.csv"
    if evaluation_complete(output, branch=name, sample_directory=archive.parent):
        append_event(event_log, "evaluation_skipped", branch=name)
        return output
    command = [
        str(python),
        "experiments/evaluate_raev2_samples.py",
        "--branch",
        f"{name}={archive}",
        "--reference",
        "/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz",
        "--output",
        str(output),
        "--batch-size",
        "64",
        "--seed",
        "0",
    ]
    run_logged(
        command,
        log_path=logs / f"evaluate_{name}.log",
        event_log=event_log,
        env=env,
    )
    if not evaluation_complete(
        output, branch=name, sample_directory=archive.parent
    ):
        raise RuntimeError(f"evaluation for {name} did not produce valid metrics")
    return output


def merge_evaluations(inputs: Mapping[str, Path], output: Path) -> None:
    frames = []
    for branch, path in inputs.items():
        frame = pd.read_csv(path)
        if len(frame) != 1 or str(frame.iloc[0]["branch"]) != branch:
            raise ValueError(f"invalid per-branch metrics in {path}")
        frames.append(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps(combined.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-step", type=int, default=150)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--per-rank-batch", type=int, default=32)
    parser.add_argument("--lpl-weight", type=float, default=0.0007801168201348963)
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_free_gib <= 0:
        raise ValueError("--min-free-gib must be positive")
    steps = checkpoint_steps(args.target_step, args.checkpoint_every)
    data_root = args.data_root.expanduser().resolve()
    experiment_root = data_root / "experiments" / "raev2_lpl_pilot"
    pipeline_root = experiment_root / f"long_flow_lpl_s{args.target_step}"
    logs = pipeline_root / "logs"
    event_log = pipeline_root / "events.jsonl"
    status_path = pipeline_root / "status.json"
    python = data_root / "envs" / "raev2" / "bin" / "python"
    config = ROOT / "experiments" / "configs" / "raev2_strict_lpl_dinov3l_k7.yaml"
    source = data_root / "models" / "RAEv2" / "stage2" / "imagenet" / "dinov3l-k7" / "checkpoint.pt"
    index_map = data_root / "datasets" / "raev2_imagenet_train_lexicographic_indices.npy"
    dino_repo = data_root / "models" / "RAEv2" / "dinov3_repo"
    data_path = Path("/data/shared/imagenet-1k/data")
    samples_root = pipeline_root / f"samples_n{args.sample_count}_seed0"

    flow = BranchSpec(
        name="flow",
        objective="flow",
        experiment_name=f"flow_official_{args.target_step}_strict_from30",
        seed_checkpoint_dirs=(
            experiment_root / "flow_official_10_strict",
            experiment_root / "flow_official_30_strict_from10",
        ),
    )
    lpl = BranchSpec(
        name="lpl",
        objective="lpl",
        experiment_name=f"lpl_official_{args.target_step}_strict_from5",
        seed_checkpoint_dirs=(experiment_root / "lpl_official_30_strict",),
    )

    required = (python, config, source, index_map, dino_repo, data_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required paths do not exist: {missing}")
    free_bytes = shutil.disk_usage(data_root).free
    if free_bytes < 500 * 1024**3:
        raise RuntimeError(
            f"long pipeline requires at least 500 GiB free under {data_root}; "
            f"found {free_bytes / 1024**3:.1f} GiB"
        )

    plan = {
        "target_step": args.target_step,
        "checkpoint_every": args.checkpoint_every,
        "steps": steps,
        "sample_count": args.sample_count,
        "per_rank_batch": args.per_rank_batch,
        "lpl_weight": args.lpl_weight,
        "min_free_gib": args.min_free_gib,
        "pipeline_root": str(pipeline_root),
        "estimated_checkpoint_count": 2 * len(steps),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    pipeline_root.mkdir(parents=True, exist_ok=True)
    atomic_json(status_path, {**plan, "state": "running"})
    append_event(event_log, "pipeline_start", **plan)

    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    try:
        train_branch(
            flow,
            target_step=args.target_step,
            checkpoint_every=args.checkpoint_every,
            python=python,
            config=config,
            data_path=data_path,
            index_map=index_map,
            source_checkpoint=source,
            results_root=experiment_root,
            dino_repo=dino_repo,
            lpl_weight=args.lpl_weight,
            min_free_gib=args.min_free_gib,
            logs=logs,
            event_log=event_log,
            env=env,
        )
        train_branch(
            lpl,
            target_step=args.target_step,
            checkpoint_every=args.checkpoint_every,
            python=python,
            config=config,
            data_path=data_path,
            index_map=index_map,
            source_checkpoint=source,
            results_root=experiment_root,
            dino_repo=dino_repo,
            lpl_weight=args.lpl_weight,
            min_free_gib=args.min_free_gib,
            logs=logs,
            event_log=event_log,
            env=env,
        )

        flow_checkpoints = valid_checkpoints(
            (*flow.seed_checkpoint_dirs, experiment_root / flow.experiment_name),
            objective="flow",
            event_log=event_log,
        )
        lpl_checkpoints = valid_checkpoints(
            (*lpl.seed_checkpoint_dirs, experiment_root / lpl.experiment_name),
            objective="lpl",
            event_log=event_log,
        )
        missing_flow = [step for step in steps if step not in flow_checkpoints]
        missing_lpl = [step for step in steps if step not in lpl_checkpoints]
        if missing_flow or missing_lpl:
            raise RuntimeError(
                f"missing curve checkpoints: flow={missing_flow}, lpl={missing_lpl}"
            )

        samples: dict[str, Path] = {}
        samples["official"] = sample_branch(
            "official",
            source,
            python=python,
            config=config,
            samples_root=samples_root,
            sample_count=args.sample_count,
            per_rank_batch=args.per_rank_batch,
            dino_repo=dino_repo,
            logs=logs,
            event_log=event_log,
            env=env,
        )
        for step in steps:
            for objective, checkpoints in (
                ("flow", flow_checkpoints),
                ("lpl", lpl_checkpoints),
            ):
                name = f"{objective}_s{step:04d}"
                samples[name] = sample_branch(
                    name,
                    checkpoints[step],
                    python=python,
                    config=config,
                    samples_root=samples_root,
                    sample_count=args.sample_count,
                    per_rank_batch=args.per_rank_batch,
                    dino_repo=dino_repo,
                    logs=logs,
                    event_log=event_log,
                    env=env,
                )

        same_noise_fingerprints = verify_same_noise_protocol(
            {name: archive.parent for name, archive in samples.items()}
        )
        atomic_json(
            pipeline_root / "same_noise_audit.json",
            same_noise_fingerprints,
        )
        append_event(event_log, "same_noise_audit_passed")

        evaluation_env = dict(env)
        evaluation_env["CUDA_VISIBLE_DEVICES"] = "0"
        evaluation_files = {
            name: evaluate_branch(
                name,
                archive,
                python=python,
                pipeline_root=pipeline_root,
                logs=logs,
                event_log=event_log,
                env=evaluation_env,
            )
            for name, archive in samples.items()
        }
        metrics_csv = pipeline_root / f"metrics_n{args.sample_count}.csv"
        merge_evaluations(evaluation_files, metrics_csv)
        curve_csv = pipeline_root / f"curve_n{args.sample_count}.csv"
        curve_png = pipeline_root / f"curve_n{args.sample_count}.png"
        write_curve(metrics_csv, output_csv=curve_csv, output_png=curve_png)
        component_audit_dir = pipeline_root / "lpl_component_audit_phase0"
        atomic_json(
            status_path,
            {
                **plan,
                "state": "main_complete_phase2_running",
                "metrics_csv": str(metrics_csv),
                "curve_csv": str(curve_csv),
                "curve_png": str(curve_png),
            },
        )
        component_outputs = (
            component_audit_dir / "component_audit_summary.csv",
            component_audit_dir / "component_audit_layers.csv",
            component_audit_dir / "component_audit.png",
            component_audit_dir / "manifest.json",
        )
        if not all(path.exists() and path.stat().st_size > 0 for path in component_outputs):
            component_command = [
                str(python),
                "experiments/run_raev2_lpl_component_audit.py",
                "--config",
                str(config),
                "--data-path",
                str(data_path),
                "--index-map",
                str(index_map),
                "--checkpoint",
                f"official={source}",
                "--checkpoint",
                f"flow_s{args.target_step:04d}={flow_checkpoints[args.target_step]}",
                "--checkpoint",
                f"lpl_s{args.target_step:04d}={lpl_checkpoints[args.target_step]}",
                "--output-dir",
                str(component_audit_dir),
                "--samples",
                "4",
                "--state-key",
                "model",
                "--precision",
                "bf16",
                "--dino-repo-dir",
                str(dino_repo),
            ]
            run_logged(
                component_command,
                log_path=logs / "lpl_component_audit_phase0.log",
                event_log=event_log,
                env=evaluation_env,
            )
        if not all(path.exists() and path.stat().st_size > 0 for path in component_outputs):
            raise RuntimeError("LPL component audit ended with incomplete outputs")
        final = {
            **plan,
            "state": "complete",
            "metrics_csv": str(metrics_csv),
            "curve_csv": str(curve_csv),
            "curve_png": str(curve_png),
            "component_audit_dir": str(component_audit_dir),
        }
        atomic_json(status_path, final)
        append_event(event_log, "pipeline_complete", **final)
    except Exception as error:
        atomic_json(status_path, {**plan, "state": "failed", "error": repr(error)})
        append_event(event_log, "pipeline_failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
