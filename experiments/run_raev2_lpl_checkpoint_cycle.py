"""Resume-safe RAEv2 LPL train, sample, and evaluation cycle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    run_logged,
    sample_branch,
    torchrun_prefix,
    validate_checkpoint,
    verify_same_noise_protocol,
    write_lpl_curve,
)


DEFAULT_DATA_ROOT = Path("/home/zhoushunyu/data/eqvae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-step", type=int, default=800)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--per-rank-batch", type=int, default=16)
    parser.add_argument("--min-free-gib", type=float, default=0.5)
    parser.add_argument("--compile-stage2", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--packed-data-path",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint_path(experiment_dir: Path, step: int, source_step: int) -> Path:
    return (
        experiment_dir
        / "checkpoints"
        / f"branch-{step:07d}-global-{source_step + step:07d}.pt"
    )


def training_command(
    *,
    python: Path,
    config: Path,
    data_path: Path,
    packed_data_path: Path,
    index_map: Path,
    results_root: Path,
    experiment_name: str,
    source_checkpoint: Path,
    resume: Path,
    target_step: int,
    checkpoint_every: int,
    min_free_gib: float,
    dino_repo: Path,
    compile_stage2: bool,
) -> list[str]:
    command = [
        *torchrun_prefix(python),
        "experiments/train_raev2_strict_lpl.py",
        "--config",
        str(config),
        "--data-path",
        str(data_path),
        "--packed-data-path",
        str(packed_data_path),
        "--index-map",
        str(index_map),
        "--results-dir",
        str(results_root),
        "--experiment-name",
        experiment_name,
        "--source-checkpoint",
        str(source_checkpoint),
        "--resume",
        str(resume),
        "--objective",
        "lpl",
        "--max-updates",
        str(target_step),
        "--save-every",
        str(checkpoint_every),
        "--precision",
        "bf16",
        "--ema-device",
        "cpu",
        "--num-workers",
        "4",
        "--min-free-gib",
        str(float(min_free_gib)),
        "--dino-repo-dir",
        str(dino_repo),
        "--lpl-weight",
        "0.0007801168201348963",
        "--lpl-noise-threshold",
        "3.0",
        "--lpl-max-samples-per-rank",
        "1",
    ]
    if compile_stage2:
        command.append("--compile-stage2")
    return command


def main() -> None:
    args = parse_args()
    if args.target_step <= 0:
        raise ValueError("--target-step must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.target_step % args.checkpoint_every:
        raise ValueError("--target-step must be divisible by --checkpoint-every")
    if args.sample_count <= 0 or args.sample_count % 1000:
        raise ValueError("--sample-count must be a positive multiple of 1000")
    if args.min_free_gib <= 0:
        raise ValueError("--min-free-gib must be positive")

    data_root = args.data_root.expanduser().resolve()
    results_root = data_root / "experiments" / "raev2_lpl_pilot"
    experiment_name = f"lpl_official_{args.target_step}_strict_from10"
    experiment_dir = results_root / experiment_name
    logs = experiment_dir / "cycle_logs"
    event_log = experiment_dir / "cycle_events.jsonl"
    status_path = experiment_dir / "cycle_status.json"
    sample_root = experiment_dir / f"samples_n{args.sample_count}_seed0"
    python = data_root / "envs" / "raev2" / "bin" / "python"
    config = ROOT / "experiments" / "configs" / "raev2_strict_lpl_dinov3l_k7.yaml"
    source_checkpoint = (
        data_root
        / "models"
        / "RAEv2"
        / "stage2"
        / "imagenet"
        / "dinov3l-k7"
        / "checkpoint.pt"
    )
    initial_checkpoint = (
        results_root
        / "lpl_official_150_strict_from5"
        / "checkpoints"
        / "branch-0000010-global-0100090.pt"
    )
    index_map = data_root / "datasets" / "raev2_imagenet_train_lexicographic_indices.npy"
    dino_repo = data_root / "models" / "RAEv2" / "dinov3_repo"
    data_path = Path("/data/shared/imagenet-1k/data")
    packed_data_path = args.packed_data_path.expanduser().resolve()
    official_root = results_root / "long_flow_lpl_s150"
    official_sample_dir = (
        official_root / f"samples_n{args.sample_count}_seed0" / "official"
    )
    official_metrics = official_root / "metrics" / "official.csv"
    required = (
        python,
        config,
        source_checkpoint,
        initial_checkpoint,
        index_map,
        dino_repo,
        data_path,
        packed_data_path / "manifest.json",
        official_sample_dir / "sampling_summary.json",
        official_metrics,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required paths do not exist: {missing}")

    initial = validate_checkpoint(initial_checkpoint, objective="lpl")
    start_step = int(initial["branch_update"])
    source_step = int(initial["global_step"]) - start_step
    targets = list(
        range(
            args.checkpoint_every,
            args.target_step + 1,
            args.checkpoint_every,
        )
    )
    targets = [step for step in targets if step > start_step]
    estimated_checkpoint_bytes = len(targets) * initial_checkpoint.stat().st_size
    free_bytes = shutil.disk_usage(data_root).free
    if free_bytes < estimated_checkpoint_bytes + 100 * 1024**3:
        raise RuntimeError(
            "insufficient disk headroom for checkpoints and samples: "
            f"need at least {(estimated_checkpoint_bytes + 100 * 1024**3) / 1024**3:.1f} GiB, "
            f"found {free_bytes / 1024**3:.1f} GiB"
        )

    plan = {
        "state": "planned",
        "target_step": args.target_step,
        "checkpoint_every": args.checkpoint_every,
        "targets": targets,
        "sample_count": args.sample_count,
        "sampling_seed": 0,
        "per_rank_sampling_batch": args.per_rank_batch,
        "training_global_batch": 1024,
        "training_micro_batch_per_rank": 1,
        "training_grad_accum_steps": 256,
        "training_data_workers_per_rank": 4,
        "training_data_backend": "packed_random_access",
        "compile_stage2": args.compile_stage2,
        "packed_data_path": str(packed_data_path),
        "min_free_gib": args.min_free_gib,
        "initial_checkpoint": str(initial_checkpoint),
        "experiment_dir": str(experiment_dir),
        "estimated_checkpoint_gib": estimated_checkpoint_bytes / 1024**3,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    experiment_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    evaluation_files: dict[str, Path] = {"official": official_metrics}
    current_checkpoint = initial_checkpoint
    current_step = start_step
    started_at = datetime.now().astimezone().isoformat()
    atomic_json(
        status_path,
        {**plan, "state": "running", "started_at": started_at},
    )
    append_event(event_log, "cycle_start", **plan)

    try:
        for target in targets:
            name = f"lpl_s{target:04d}"
            target_checkpoint = checkpoint_path(experiment_dir, target, source_step)
            if target_checkpoint.exists():
                metadata = validate_checkpoint(target_checkpoint, objective="lpl")
                if int(metadata["branch_update"]) != target:
                    raise ValueError(f"invalid checkpoint step in {target_checkpoint}")
                append_event(
                    event_log,
                    "training_skipped",
                    branch=name,
                    checkpoint=str(target_checkpoint),
                )
            else:
                atomic_json(
                    status_path,
                    {
                        **plan,
                        "state": "training",
                        "started_at": started_at,
                        "current_target": target,
                        "resume_checkpoint": str(current_checkpoint),
                    },
                )
                command = training_command(
                    python=python,
                    config=config,
                    data_path=data_path,
                    packed_data_path=packed_data_path,
                    index_map=index_map,
                    results_root=results_root,
                    experiment_name=experiment_name,
                    source_checkpoint=source_checkpoint,
                    resume=current_checkpoint,
                    target_step=target,
                    checkpoint_every=args.checkpoint_every,
                    min_free_gib=args.min_free_gib,
                    dino_repo=dino_repo,
                    compile_stage2=args.compile_stage2,
                )
                run_logged(
                    command,
                    log_path=logs / f"train_to_{target:04d}.log",
                    event_log=event_log,
                    env=environment,
                )
                metadata = validate_checkpoint(target_checkpoint, objective="lpl")
                if int(metadata["branch_update"]) != target:
                    raise ValueError(f"invalid checkpoint step in {target_checkpoint}")
                audit_dir = experiment_dir / "training_audits"
                audit_dir.mkdir(parents=True, exist_ok=True)
                for filename in ("manifest.json", "first_batch_audit.json"):
                    source = experiment_dir / filename
                    if not source.exists():
                        raise FileNotFoundError(
                            f"training did not produce required audit: {source}"
                        )
                    shutil.copy2(
                        source,
                        audit_dir / f"train_to_{target:04d}_{filename}",
                    )

            current_checkpoint = target_checkpoint
            current_step = target
            atomic_json(
                status_path,
                {
                    **plan,
                    "state": "sampling",
                    "started_at": started_at,
                    "current_target": target,
                    "checkpoint": str(target_checkpoint),
                },
            )
            archive = sample_branch(
                name,
                target_checkpoint,
                python=python,
                config=config,
                samples_root=sample_root,
                sample_count=args.sample_count,
                per_rank_batch=args.per_rank_batch,
                dino_repo=dino_repo,
                logs=logs,
                event_log=event_log,
                env=environment,
            )
            fingerprints = verify_same_noise_protocol(
                {
                    "official": official_sample_dir,
                    name: archive.parent,
                }
            )
            atomic_json(
                experiment_dir / "sampling_audits" / f"{name}.json",
                fingerprints,
            )

            atomic_json(
                status_path,
                {
                    **plan,
                    "state": "evaluating",
                    "started_at": started_at,
                    "current_target": target,
                    "sample_archive": str(archive),
                },
            )
            evaluation_environment = dict(environment)
            evaluation_environment["CUDA_VISIBLE_DEVICES"] = "0"
            evaluation_files[name] = evaluate_branch(
                name,
                archive,
                python=python,
                pipeline_root=experiment_dir,
                logs=logs,
                event_log=event_log,
                env=evaluation_environment,
            )
            metrics_csv = experiment_dir / f"metrics_lpl_n{args.sample_count}.csv"
            merge_evaluations(evaluation_files, metrics_csv)
            curve_csv = experiment_dir / f"curve_lpl_n{args.sample_count}.csv"
            curve_png = experiment_dir / f"curve_lpl_n{args.sample_count}.png"
            write_lpl_curve(
                metrics_csv,
                output_csv=curve_csv,
                output_png=curve_png,
            )
            append_event(
                event_log,
                "checkpoint_cycle_complete",
                branch=name,
                checkpoint=str(target_checkpoint),
                sample_archive=str(archive),
                evaluation=str(evaluation_files[name]),
            )

        result = {
            **plan,
            "state": "complete",
            "started_at": started_at,
            "completed_at": datetime.now().astimezone().isoformat(),
            "last_completed_step": current_step,
            "last_checkpoint": str(current_checkpoint),
            "metrics_csv": str(experiment_dir / f"metrics_lpl_n{args.sample_count}.csv"),
            "curve_png": str(experiment_dir / f"curve_lpl_n{args.sample_count}.png"),
        }
        atomic_json(status_path, result)
        append_event(event_log, "cycle_complete", **result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as error:
        failed = {
            **plan,
            "state": "failed",
            "started_at": started_at,
            "failed_at": datetime.now().astimezone().isoformat(),
            "last_completed_step": current_step,
            "last_checkpoint": str(current_checkpoint),
            "error": repr(error),
        }
        atomic_json(status_path, failed)
        append_event(event_log, "cycle_failed", **failed)
        raise


if __name__ == "__main__":
    main()
