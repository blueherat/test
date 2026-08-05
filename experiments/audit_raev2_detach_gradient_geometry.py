#!/usr/bin/env python3
"""
Audit the parameter-gradient geometry of Flow and prediction-detach LPL in RAEv2.

The audit launches four gradient-only DDP jobs on exactly the same checkpoint,
data stream, RNG seed, and microbatches:

    g_F              : Flow gradient
    g_L              : weighted LPL gradient, including --lpl-weight
    g_+ = g_F + g_L  : total gradient
    g_- = g_F - g_L  : total gradient with negative LPL weight

It then computes

    ||g_L|| / ||g_F||
    cos(g_F, g_L)
    ||g_L_parallel|| / ||g_F||
    ||g_L_perp|| / ||g_F||

using the polarization identity

    <g_F, g_L> = (||g_+||^2 - ||g_-||^2) / 4.

No optimizer step is taken and no checkpoint is written. The script makes a
temporary copy of train_raev2_strict_lpl.py with a "total" audit component;
the original training script is not modified.

By default it waits until the current RAEv2 sampling/evaluation launcher has
finished and all selected GPUs have enough free memory.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_WAIT_PATTERNS = (
    "sample_raev2_threeway.py",
    "run_raev2_ig_scale_sweep.py",
    "compare_flow50_detach50_5k.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo", type=Path, default=Path("/home/zhoushunyu/eqvae"))
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "RAEv2 data root. Auto-detects /data/users/zhoushunyu/eqvae "
            "then /home/zhoushunyu/data/eqvae."
        ),
    )
    parser.add_argument("--python", type=Path, help="RAEv2 environment Python.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Model checkpoint at which both gradients are measured.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable output label; defaults to checkpoint stem.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root directory for the four audits and summary.",
    )
    parser.add_argument(
        "--lpl-weight",
        type=float,
        default=2.9384045033942286e-5,
        help="The actual prediction-detach LPL coefficient used in training.",
    )
    parser.add_argument("--lpl-target", default="full_base")
    parser.add_argument("--lpl-variant", default="prediction_detach")
    parser.add_argument("--lpl-guidance-scale", type=float, default=1.78)
    parser.add_argument("--lpl-noise-threshold", type=float, default=3.0)
    parser.add_argument("--lpl-max-samples-per-rank", type=int, default=1)
    parser.add_argument(
        "--microbatches",
        type=int,
        default=64,
        help="Gradient-accumulation microbatches per rank; global samples = 4*N.",
    )
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Physical GPU indices passed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--min-start-free-gib",
        type=float,
        default=22.0,
        help="Wait until every selected physical GPU has at least this much free memory.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=60,
        help="Polling interval while sampling/GPU memory is busy.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Launch immediately instead of waiting for current sampling and free GPUs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rerun already completed component audits.",
    )
    parser.add_argument(
        "--keep-patched-trainer",
        action="store_true",
        help="Keep the temporary patched training file for inspection.",
    )
    return parser.parse_args()


def choose_data_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = (
        Path("/data/users/zhoushunyu/eqvae"),
        Path("/home/zhoushunyu/data/eqvae"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not auto-detect data root. Pass --data-root explicitly."
    )


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def patch_trainer(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")

    old_choices = 'choices=("flow", "lpl"),\n        help="Accumulate one paired gradient probe without taking an optimizer step.",'
    new_choices = 'choices=("flow", "lpl", "total"),\n        help="Accumulate one paired gradient probe without taking an optimizer step.",'
    if old_choices not in text:
        raise RuntimeError(
            "Could not find the expected gradient-audit choices in "
            f"{source}. The training script may have changed."
        )
    text = text.replace(old_choices, new_choices, 1)

    old_validation = (
        '    if args.gradient_audit_component == "flow" and args.objective != "flow":\n'
        '        raise ValueError("a Flow gradient audit requires --objective flow")\n'
    )
    new_validation = old_validation + (
        '    if args.gradient_audit_component == "total" and args.objective != "lpl":\n'
        '        raise ValueError("a total gradient audit requires --objective lpl")\n'
    )
    if old_validation not in text:
        raise RuntimeError(
            "Could not find the expected gradient-audit validation block in "
            f"{source}. The training script may have changed."
        )
    text = text.replace(old_validation, new_validation, 1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def current_user_processes() -> list[tuple[int, str]]:
    user = getpass.getuser()
    result = subprocess.run(
        ["ps", "-u", user, "-o", "pid=,args="],
        check=True,
        text=True,
        capture_output=True,
    )
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        rows.append((pid, command.strip()))
    return rows


def matching_busy_processes() -> list[tuple[int, str]]:
    own_pid = os.getpid()
    parent_pid = os.getppid()
    matches: list[tuple[int, str]] = []
    for pid, command in current_user_processes():
        if pid in (own_pid, parent_pid):
            continue
        if any(pattern in command for pattern in DEFAULT_WAIT_PATTERNS):
            matches.append((pid, command))
    return matches


def gpu_free_gib(physical_gpu_ids: list[int]) -> dict[int, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    all_values: dict[int, float] = {}
    for line in result.stdout.splitlines():
        index_text, memory_text = [piece.strip() for piece in line.split(",", 1)]
        all_values[int(index_text)] = float(memory_text) / 1024.0
    missing = [idx for idx in physical_gpu_ids if idx not in all_values]
    if missing:
        raise RuntimeError(f"nvidia-smi did not report GPUs: {missing}")
    return {idx: all_values[idx] for idx in physical_gpu_ids}


def wait_until_available(
    gpu_ids: list[int],
    *,
    min_free_gib: float,
    poll_seconds: int,
) -> None:
    print(
        "Waiting for the current sampling job to finish and for GPU memory to be free.",
        flush=True,
    )
    while True:
        busy = matching_busy_processes()
        free = gpu_free_gib(gpu_ids)
        memory_ready = all(value >= min_free_gib for value in free.values())
        if not busy and memory_ready:
            print(
                "Resources are ready: "
                + ", ".join(f"GPU {idx}: {value:.2f} GiB free" for idx, value in free.items()),
                flush=True,
            )
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        reasons: list[str] = []
        if busy:
            short = "; ".join(f"pid={pid} {cmd[:120]}" for pid, cmd in busy[:4])
            reasons.append(f"matching job still running: {short}")
        if not memory_ready:
            reasons.append(
                "free memory: "
                + ", ".join(f"GPU {idx}={value:.2f} GiB" for idx, value in free.items())
            )
        print(f"[{timestamp}] waiting: " + " | ".join(reasons), flush=True)
        time.sleep(poll_seconds)


def remove_if_requested(directory: Path, force: bool) -> None:
    if force and directory.exists():
        shutil.rmtree(directory)


def run_component(
    *,
    name: str,
    component: str,
    objective: str,
    lpl_weight: float,
    patched_trainer: Path,
    repo: Path,
    python: Path,
    checkpoint: Path,
    output_dir: Path,
    config: Path,
    data_path: Path,
    packed_data_path: Path,
    index_map: Path,
    dino_ckpt_dir: Path,
    dino_repo_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    experiment_name = name
    experiment_dir = output_dir / experiment_name
    result_file = experiment_dir / "gradient_audit.json"

    remove_if_requested(experiment_dir, args.force)
    if result_file.is_file():
        print(f"[reuse] {name}: {result_file}", flush=True)
        return json.loads(result_file.read_text(encoding="utf-8"))

    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(patched_trainer.relative_to(repo)),
        "--config",
        str(config),
        "--data-path",
        str(data_path),
        "--packed-data-path",
        str(packed_data_path),
        "--index-map",
        str(index_map),
        "--results-dir",
        str(output_dir),
        "--experiment-name",
        experiment_name,
        "--source-checkpoint",
        str(checkpoint),
        "--objective",
        objective,
        "--lpl-target",
        args.lpl_target,
        "--lpl-variant",
        args.lpl_variant,
        "--lpl-gradient-mode",
        "direct",
        "--lpl-guidance-scale",
        str(args.lpl_guidance_scale),
        "--lpl-multiscale-scales",
        "1.0,1.39,1.78",
        f"--lpl-weight={float(lpl_weight)!r}",
        "--lpl-noise-threshold",
        str(args.lpl_noise_threshold),
        "--lpl-max-samples-per-rank",
        str(args.lpl_max_samples_per_rank),
        "--max-updates",
        "1",
        "--save-every",
        "1",
        "--skip-checkpoint-save",
        "--precision",
        "bf16",
        "--ema-device",
        "cpu",
        "--global-seed",
        str(args.global_seed),
        "--num-workers",
        str(args.num_workers),
        "--min-free-gib",
        "0.5",
        "--gradient-audit-component",
        component,
        "--gradient-audit-microbatches",
        str(args.microbatches),
        "--gradient-audit-skip-optimizer-state",
        "--dino-ckpt-dir",
        str(dino_ckpt_dir),
        "--dino-repo-dir",
        str(dino_repo_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(f"\n[run] {name}", flush=True)
    print(" ".join(shlex.quote(item) for item in command), flush=True)
    subprocess.run(command, cwd=repo, env=env, check=True)

    if not result_file.is_file():
        raise RuntimeError(f"Audit completed but did not create {result_file}")
    return json.loads(result_file.read_text(encoding="utf-8"))


def load_first_batch(output_dir: Path, component_name: str) -> Any:
    path = output_dir / component_name / "first_batch_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing first-batch audit: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_same_stream(
    output_dir: Path,
    audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    index_hashes = {
        name: payload.get("data_indices_sha256") for name, payload in audits.items()
    }
    if len(set(index_hashes.values())) != 1:
        raise RuntimeError(f"Data-index hashes differ across audits: {index_hashes}")

    first_batches = {
        name: load_first_batch(output_dir, name) for name in audits
    }
    reference_name = next(iter(first_batches))
    reference = first_batches[reference_name]
    mismatched = [
        name for name, value in first_batches.items() if value != reference
    ]
    if mismatched:
        raise RuntimeError(
            "First-batch hashes differ across audits. "
            f"Reference={reference_name}, mismatched={mismatched}"
        )

    global_samples = {
        name: int(payload["global_samples"]) for name, payload in audits.items()
    }
    if len(set(global_samples.values())) != 1:
        raise RuntimeError(f"Global sample counts differ: {global_samples}")

    return {
        "data_indices_sha256": next(iter(index_hashes.values())),
        "first_batch_exact_match": True,
        "global_samples": next(iter(global_samples.values())),
    }


def gradient_geometry(
    *,
    flow_norm: float,
    lpl_norm: float,
    plus_norm: float,
    minus_norm: float,
    lpl_weight: float,
) -> dict[str, float]:
    if flow_norm <= 0 or lpl_norm <= 0:
        raise ValueError(
            f"Gradient norms must be positive; flow={flow_norm}, lpl={lpl_norm}"
        )

    # The + / - polarization identity is numerically more stable than
    # subtracting ||g_F||^2 and ||g_L||^2 from one total norm.
    dot_pm = (plus_norm**2 - minus_norm**2) / 4.0
    dot_plus = (plus_norm**2 - flow_norm**2 - lpl_norm**2) / 2.0
    dot_minus = (flow_norm**2 + lpl_norm**2 - minus_norm**2) / 2.0

    cosine_raw = dot_pm / (flow_norm * lpl_norm)
    cosine_clamped = max(-1.0, min(1.0, cosine_raw))
    angle_degrees = math.degrees(math.acos(cosine_clamped))

    ratio = lpl_norm / flow_norm
    signed_parallel_ratio = dot_pm / (flow_norm**2)
    perpendicular_sq = max(
        lpl_norm**2 - (dot_pm / flow_norm) ** 2,
        0.0,
    )
    perpendicular_ratio = math.sqrt(perpendicular_sq) / flow_norm
    total_plus_ratio = plus_norm / flow_norm
    total_minus_ratio = minus_norm / flow_norm

    unweighted_lpl_norm = lpl_norm / abs(lpl_weight)
    weight_for_20_percent = abs(lpl_weight) * 0.20 / ratio
    weight_for_10_percent = abs(lpl_weight) * 0.10 / ratio

    denominator = max(abs(dot_pm), 1e-30)
    polarization_relative_disagreement = max(
        abs(dot_plus - dot_pm),
        abs(dot_minus - dot_pm),
    ) / denominator

    return {
        "flow_gradient_norm": flow_norm,
        "weighted_lpl_gradient_norm": lpl_norm,
        "unweighted_lpl_gradient_norm": unweighted_lpl_norm,
        "total_plus_gradient_norm": plus_norm,
        "total_minus_gradient_norm": minus_norm,
        "weighted_lpl_to_flow_norm_ratio": ratio,
        "gradient_dot_product": dot_pm,
        "gradient_cosine_raw": cosine_raw,
        "gradient_cosine_clamped": cosine_clamped,
        "gradient_angle_degrees": angle_degrees,
        "signed_parallel_lpl_to_flow_ratio": signed_parallel_ratio,
        "orthogonal_lpl_to_flow_ratio": perpendicular_ratio,
        "total_plus_to_flow_norm_ratio": total_plus_ratio,
        "total_minus_to_flow_norm_ratio": total_minus_ratio,
        "weight_for_target_ratio_0p20": weight_for_20_percent,
        "weight_for_target_ratio_0p10": weight_for_10_percent,
        "dot_from_plus_only": dot_plus,
        "dot_from_minus_only": dot_minus,
        "polarization_relative_disagreement": polarization_relative_disagreement,
    }


def main() -> None:
    args = parse_args()
    repo = require_dir(args.repo, "repository")
    data_root = choose_data_root(args.data_root)

    python = require_file(
        args.python or data_root / "envs/raev2/bin/python",
        "RAEv2 Python",
    )
    config = require_file(
        repo / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml",
        "RAEv2 config",
    )
    source_trainer = require_file(
        repo / "experiments/train_raev2_strict_lpl.py",
        "strict LPL trainer",
    )
    checkpoint = require_file(
        args.checkpoint
        or data_root / "models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt",
        "checkpoint",
    )
    data_path = require_dir(Path("/data/shared/imagenet-1k/data"), "ImageNet data")
    packed_data_path = require_dir(
        Path("/data/shared/imagenet-1k/random_access_v1"),
        "packed ImageNet data",
    )
    index_map = require_file(
        data_root / "datasets/raev2_imagenet_train_lexicographic_indices.npy",
        "ImageNet index map",
    )
    dino_ckpt_dir = require_dir(
        data_root / "models/RAEv2/encoders/dinov3",
        "DINOv3 checkpoint directory",
    )
    dino_repo_dir = require_dir(
        data_root / "models/RAEv2/dinov3_repo",
        "DINOv3 repository",
    )

    label = args.label or checkpoint.stem
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    output_root = (
        args.output_root
        or data_root / "experiments/raev2_detach_gradient_geometry"
    )
    output_dir = (
        output_root.expanduser().resolve()
        / f"{safe_label}_mb{args.microbatches}_w{args.lpl_weight:.8g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    patched_trainer = (
        repo / "experiments/_tmp_train_raev2_strict_lpl_total_audit.py"
    )
    patch_trainer(source_trainer, patched_trainer)

    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise ValueError(
            "This strict audit preserves the existing four-rank protocol; "
            f"got GPU list {gpu_ids}"
        )

    try:
        if not args.no_wait:
            wait_until_available(
                gpu_ids,
                min_free_gib=args.min_start_free_gib,
                poll_seconds=args.wait_seconds,
            )

        common = dict(
            patched_trainer=patched_trainer,
            repo=repo,
            python=python,
            checkpoint=checkpoint,
            output_dir=output_dir,
            config=config,
            data_path=data_path,
            packed_data_path=packed_data_path,
            index_map=index_map,
            dino_ckpt_dir=dino_ckpt_dir,
            dino_repo_dir=dino_repo_dir,
            args=args,
        )

        audits = {
            "flow": run_component(
                name="flow",
                component="flow",
                objective="flow",
                lpl_weight=args.lpl_weight,
                **common,
            ),
            "lpl": run_component(
                name="lpl",
                component="lpl",
                objective="lpl",
                lpl_weight=args.lpl_weight,
                **common,
            ),
            "total_plus": run_component(
                name="total_plus",
                component="total",
                objective="lpl",
                lpl_weight=args.lpl_weight,
                **common,
            ),
            "total_minus": run_component(
                name="total_minus",
                component="total",
                objective="lpl",
                lpl_weight=-args.lpl_weight,
                **common,
            ),
        }

        stream_audit = verify_same_stream(output_dir, audits)
        geometry = gradient_geometry(
            flow_norm=float(audits["flow"]["parameter_gradient_norm"]),
            lpl_norm=float(audits["lpl"]["parameter_gradient_norm"]),
            plus_norm=float(audits["total_plus"]["parameter_gradient_norm"]),
            minus_norm=float(audits["total_minus"]["parameter_gradient_norm"]),
            lpl_weight=args.lpl_weight,
        )

        summary = {
            "format_version": 1,
            "checkpoint": str(checkpoint),
            "label": label,
            "lpl_weight": args.lpl_weight,
            "lpl_target": args.lpl_target,
            "lpl_variant": args.lpl_variant,
            "lpl_guidance_scale": args.lpl_guidance_scale,
            "lpl_noise_threshold": args.lpl_noise_threshold,
            "lpl_max_samples_per_rank": args.lpl_max_samples_per_rank,
            "per_rank_microbatches": args.microbatches,
            "stream_audit": stream_audit,
            "component_audits": audits,
            "geometry": geometry,
        }
        summary_path = output_dir / "gradient_geometry.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        markdown = f"""# RAEv2 Flow vs prediction-detach LPL gradient geometry

- checkpoint: `{checkpoint}`
- global samples: `{stream_audit["global_samples"]}`
- weighted LPL coefficient: `{args.lpl_weight:.12g}`
- exact same stream: `{stream_audit["first_batch_exact_match"]}`

| quantity | value |
|---|---:|
| `||g_flow||` | {geometry["flow_gradient_norm"]:.8g} |
| `||lambda g_lpl||` | {geometry["weighted_lpl_gradient_norm"]:.8g} |
| weighted LPL / Flow | {geometry["weighted_lpl_to_flow_norm_ratio"]:.6f} |
| cosine | {geometry["gradient_cosine_raw"]:.6f} |
| angle | {geometry["gradient_angle_degrees"]:.3f} deg |
| signed parallel / Flow | {geometry["signed_parallel_lpl_to_flow_ratio"]:.6f} |
| orthogonal / Flow | {geometry["orthogonal_lpl_to_flow_ratio"]:.6f} |
| `||g_flow + lambda g_lpl|| / ||g_flow||` | {geometry["total_plus_to_flow_norm_ratio"]:.6f} |
| coefficient giving 20% norm | {geometry["weight_for_target_ratio_0p20"]:.12g} |
| polarization disagreement | {geometry["polarization_relative_disagreement"]:.3e} |

Interpretation:

- `weighted LPL / Flow = 0.20` means the auxiliary parameter-gradient norm is 20% of Flow.
- `cosine > 0` is locally cooperative; `cosine < 0` is locally conflicting.
- `orthogonal / Flow` measures the genuinely new update direction introduced by LPL.
- A large polarization disagreement indicates that separate launches were not numerically
  identical enough for a reliable cosine estimate; the norm ratio remains usable.
"""
        markdown_path = output_dir / "gradient_geometry.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        print("\n" + markdown, flush=True)
        print(f"JSON: {summary_path}", flush=True)
        print(f"Markdown: {markdown_path}", flush=True)

    finally:
        if patched_trainer.exists() and not args.keep_patched_trainer:
            patched_trainer.unlink()


if __name__ == "__main__":
    main()