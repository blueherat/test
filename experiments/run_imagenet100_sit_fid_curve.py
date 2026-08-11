#!/usr/bin/env python3
"""Run and validate an unguided 5k FID curve over SiT checkpoints.

This is orchestration only. Sampling remains in
``sample_imagenet100_sit_fid.py`` and FID remains in ``compute_adm_fid.py``.
Every checkpoint uses the same EMA weights, global seed, official SiT Dopri5
sampler, SD-VAE decoder, ImageNet-100 validation reference, and ADM evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import torch

try:
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
except ModuleNotFoundError:
    from train_imagenet100_sit_flow import atomic_json_dump, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_seed0"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_lowmem_v1"
)
DEFAULT_REFERENCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k/"
    "sit-s-2_step100000_seed0/reference_imagenet100_validation_n5000.npz"
)
DEFAULT_ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")


def parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not steps or any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("steps must be a non-empty list of positive integers")
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("steps must not contain duplicates")
    return sorted(steps)


def parse_gpu_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers") from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("at least one non-negative GPU index is required")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU indices must not contain duplicates")
    return indices


def parse_nvidia_memory_mib(output: str) -> dict[int, int]:
    usage: dict[int, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise ValueError(f"unexpected nvidia-smi memory row: {line!r}")
        usage[int(fields[0])] = int(fields[1])
    if not usage:
        raise ValueError("nvidia-smi returned no GPU memory rows")
    return usage


def query_gpu_memory_mib() -> dict[int, int]:
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_memory_mib(result.stdout)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def absolute_without_resolving_symlinks(path: Path) -> Path:
    """Make a path absolute while preserving virtual-environment entry symlinks."""

    return path.expanduser().absolute()


def checkpoint_metadata(path: Path, expected_step: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    step = int(checkpoint.get("step", -1))
    protocol = checkpoint.get("protocol")
    if step != expected_step:
        raise ValueError(f"checkpoint step mismatch: path={path}, payload={step}")
    if protocol != "imagenet100_sit_linear_flow_v1":
        raise ValueError(f"unexpected checkpoint protocol in {path}: {protocol!r}")
    config = checkpoint.get("config", {})
    metadata = {
        "step": step,
        "protocol": protocol,
        "model_name": config.get("model_name"),
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
    }
    del checkpoint
    return metadata


def valid_resource_audit(
    path: Path,
    *,
    gpu_indices: Sequence[int],
    memory_ceiling_mib: int,
) -> bool:
    if not path.is_file():
        return False
    audit = load_json(path)
    if audit.get("violation") is not None or int(audit.get("return_code", -1)) != 0:
        return False
    if int(audit.get("memory_ceiling_mib", -1)) != memory_ceiling_mib:
        return False
    monitored = [int(index) for index in audit.get("monitored_gpu_indices", [])]
    if monitored != list(gpu_indices):
        return False
    peaks = {int(index): int(value) for index, value in audit.get("peak_memory_mib", {}).items()}
    return all(peaks.get(index, memory_ceiling_mib) < memory_ceiling_mib for index in gpu_indices)


def valid_sampling_artifact(
    output_dir: Path,
    *,
    checkpoint: dict[str, object],
    num_samples: int,
    global_seed: int,
    world_size: int,
    per_rank_batch_size: int,
    vae_decode_batch_size: int,
    cuda_allocator_limit_gib: float,
    gpu_indices: Sequence[int],
    memory_ceiling_mib: int,
) -> bool:
    manifest_path = output_dir / "sampling_manifest.json"
    sample_path = output_dir / f"samples_unguided_n{num_samples}.npz"
    if not manifest_path.is_file() or not sample_path.is_file():
        return False
    if not valid_resource_audit(
        output_dir / "sampling_resource_audit.json",
        gpu_indices=gpu_indices,
        memory_ceiling_mib=memory_ceiling_mib,
    ):
        return False
    manifest = load_json(manifest_path)
    expected = {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_step": checkpoint["step"],
        "weights": "ema",
        "requested_samples": num_samples,
        "global_seed": global_seed,
        "world_size": world_size,
        "per_rank_batch_size": per_rank_batch_size,
        "vae_decode_batch_size": vae_decode_batch_size,
        "cuda_allocator_limit_gib": cuda_allocator_limit_gib,
        "cfg_scale": 1.0,
        "guidance": False,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"existing sampling artifact has incompatible semantics: "
            f"{manifest_path}: {mismatches}"
        )
    if Path(manifest["samples"]).resolve() != sample_path.resolve():
        raise ValueError(f"sampling manifest points to a different sample file: {manifest_path}")
    return True


def valid_fid_artifact(
    output_dir: Path,
    *,
    reference: Path,
    num_samples: int,
    fid_batch_size: int,
    fid_gpu_memory_fraction: float,
    gpu_indices: Sequence[int],
    memory_ceiling_mib: int,
) -> bool:
    result_path = output_dir / "fid5k_adm_results.json"
    sample_path = output_dir / f"samples_unguided_n{num_samples}.npz"
    if not result_path.is_file():
        return False
    if not valid_resource_audit(
        output_dir / "fid_resource_audit.json",
        gpu_indices=gpu_indices,
        memory_ceiling_mib=memory_ceiling_mib,
    ):
        return False
    result = load_json(result_path)
    if Path(result.get("reference", "")).resolve() != reference.resolve():
        raise ValueError(f"FID result uses a different reference: {result_path}")
    if Path(result.get("samples", "")).resolve() != sample_path.resolve():
        raise ValueError(f"FID result uses a different sample file: {result_path}")
    expected = {
        "batch_size": fid_batch_size,
        "gpu_memory_fraction": fid_gpu_memory_fraction,
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"existing FID artifact has incompatible resource protocol: "
            f"{result_path}: {mismatches}"
        )
    for key in ("fid", "sfid", "inception_score"):
        if not math.isfinite(float(result.get(key, float("nan")))):
            raise ValueError(f"non-finite {key} in {result_path}")
    return True


def _stream_process_output(
    process: subprocess.Popen[str],
    log,
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    env: dict[str, str],
    monitored_gpu_indices: Sequence[int],
    memory_ceiling_mib: int,
    memory_poll_interval: float,
    resource_audit_path: Path,
) -> dict[str, object]:
    """Run a subprocess while enforcing and recording a total GPU-memory ceiling."""

    if memory_ceiling_mib <= 0 or memory_poll_interval <= 0:
        raise ValueError("GPU memory ceiling and polling interval must be positive")
    monitored = list(monitored_gpu_indices)
    if not monitored:
        raise ValueError("at least one GPU must be monitored")
    initial = query_gpu_memory_mib()
    missing = sorted(set(monitored) - set(initial))
    if missing:
        raise ValueError(f"nvidia-smi did not report GPUs: {missing}")
    initial_selected = {index: initial[index] for index in monitored}
    initial_over_limit = {
        index: used
        for index, used in initial_selected.items()
        if used >= memory_ceiling_mib
    }
    if initial_over_limit:
        raise RuntimeError(
            f"refusing to launch above {memory_ceiling_mib} MiB: {initial_over_limit}"
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[launch] {' '.join(command)}", flush=True)
    peak = initial_selected.copy()
    samples = 1
    violation: dict[str, object] | None = None
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        output_thread = threading.Thread(
            target=_stream_process_output,
            args=(process, log),
            daemon=True,
        )
        output_thread.start()
        while process.poll() is None:
            time.sleep(memory_poll_interval)
            try:
                current = query_gpu_memory_mib()
            except Exception as error:
                violation = {
                    "reason": "memory_monitor_failed",
                    "error": repr(error),
                }
                _terminate_process_group(process)
                break
            samples += 1
            selected = {index: current[index] for index in monitored}
            peak = {
                index: max(peak[index], selected[index])
                for index in monitored
            }
            over_limit = {
                index: used
                for index, used in selected.items()
                if used >= memory_ceiling_mib
            }
            if over_limit:
                violation = {
                    "reason": "gpu_memory_ceiling_reached",
                    "observed_mib": over_limit,
                }
                _terminate_process_group(process)
                break
        return_code = process.wait()
        output_thread.join(timeout=10)

    audit: dict[str, object] = {
        "command": list(command),
        "monitored_gpu_indices": monitored,
        "memory_ceiling_mib": memory_ceiling_mib,
        "memory_poll_interval_seconds": memory_poll_interval,
        "initial_memory_mib": initial_selected,
        "peak_memory_mib": peak,
        "monitor_samples": samples,
        "elapsed_seconds": time.time() - started,
        "return_code": return_code,
        "violation": violation,
    }
    atomic_json_dump(audit, resource_audit_path)
    if violation is not None:
        raise RuntimeError(f"GPU memory safety guard stopped the command: {violation}")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return audit


def fid_environment(
    base_env: dict[str, str],
    *,
    cuda_visible_devices: str,
) -> dict[str, str]:
    """Keep TensorFlow FID evaluation on one GPU with on-demand allocation."""
    env = base_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return env


def evaluate_checkpoint(
    *,
    step: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, object]:
    checkpoint_path = args.run_dir / "checkpoints" / f"step_{step:08d}.pt"
    checkpoint = checkpoint_metadata(checkpoint_path, step)
    output_dir = args.output_root / f"sit-s-2_step{step:06d}_seed0"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"samples_unguided_n{args.num_samples}.npz"

    if not valid_sampling_artifact(
        output_dir,
        checkpoint=checkpoint,
        num_samples=args.num_samples,
        global_seed=args.global_seed,
        world_size=len(args.sampling_gpu_indices),
        per_rank_batch_size=args.per_rank_batch_size,
        vae_decode_batch_size=args.vae_decode_batch_size,
        cuda_allocator_limit_gib=args.cuda_allocator_limit_gib,
        gpu_indices=args.sampling_gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
        sampling_audit = run_logged(
            (
                args.torchrun,
                "--standalone",
                f"--nproc_per_node={len(args.sampling_gpu_indices)}",
                str(REPO_ROOT / "experiments/sample_imagenet100_sit_fid.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--weights",
                "ema",
                "--num-samples",
                str(args.num_samples),
                "--per-rank-batch-size",
                str(args.per_rank_batch_size),
                "--vae-decode-batch-size",
                str(args.vae_decode_batch_size),
                "--cuda-allocator-limit-gib",
                str(args.cuda_allocator_limit_gib),
                "--global-seed",
                str(args.global_seed),
                "--precision",
                "fp32",
                "--allow-tf32",
            ),
            output_dir / "sampling.log",
            env=env,
            monitored_gpu_indices=args.sampling_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "sampling_resource_audit.json",
        )
        if not valid_sampling_artifact(
            output_dir,
            checkpoint=checkpoint,
            num_samples=args.num_samples,
            global_seed=args.global_seed,
            world_size=len(args.sampling_gpu_indices),
            per_rank_batch_size=args.per_rank_batch_size,
            vae_decode_batch_size=args.vae_decode_batch_size,
            cuda_allocator_limit_gib=args.cuda_allocator_limit_gib,
            gpu_indices=args.sampling_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
        ):
            raise RuntimeError(f"sampler completed without a valid manifest: {output_dir}")
    else:
        print(f"[reuse] valid samples for step {step}", flush=True)
        sampling_audit = load_json(output_dir / "sampling_resource_audit.json")

    if not valid_fid_artifact(
        output_dir,
        reference=args.reference,
        num_samples=args.num_samples,
        fid_batch_size=args.fid_batch_size,
        fid_gpu_memory_fraction=args.fid_gpu_memory_fraction,
        gpu_indices=args.fid_gpu_indices,
        memory_ceiling_mib=args.gpu_memory_ceiling_mib,
    ):
        fid_audit = run_logged(
            (
                str(args.adm_python),
                str(REPO_ROOT / "experiments/compute_adm_fid.py"),
                "--reference",
                str(args.reference),
                "--samples",
                str(sample_path),
                "--batch-size",
                str(args.fid_batch_size),
                "--gpu-memory-fraction",
                str(args.fid_gpu_memory_fraction),
                "--output",
                str(output_dir / "fid5k_adm_results.json"),
            ),
            output_dir / "evaluation.log",
            env=fid_environment(
                env,
                cuda_visible_devices=args.fid_cuda_visible_devices,
            ),
            monitored_gpu_indices=args.fid_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
            memory_poll_interval=args.memory_poll_interval,
            resource_audit_path=output_dir / "fid_resource_audit.json",
        )
        if not valid_fid_artifact(
            output_dir,
            reference=args.reference,
            num_samples=args.num_samples,
            fid_batch_size=args.fid_batch_size,
            fid_gpu_memory_fraction=args.fid_gpu_memory_fraction,
            gpu_indices=args.fid_gpu_indices,
            memory_ceiling_mib=args.gpu_memory_ceiling_mib,
        ):
            raise RuntimeError(f"ADM evaluator completed without a valid result: {output_dir}")
    else:
        print(f"[reuse] valid FID for step {step}", flush=True)
        fid_audit = load_json(output_dir / "fid_resource_audit.json")

    result = load_json(output_dir / "fid5k_adm_results.json")
    return {
        **checkpoint,
        "output_dir": str(output_dir),
        "reference": str(args.reference.resolve()),
        "samples": str(sample_path.resolve()),
        "num_samples": args.num_samples,
        "weights": "ema",
        "cfg_scale": 1.0,
        "guidance": False,
        "global_seed": args.global_seed,
        "sampling_peak_memory_mib": max(
            int(value) for value in sampling_audit["peak_memory_mib"].values()
        ),
        "fid_peak_memory_mib": max(
            int(value) for value in fid_audit["peak_memory_mib"].values()
        ),
        "fid": float(result["fid"]),
        "sfid": float(result["sfid"]),
        "inception_score": float(result["inception_score"]),
    }


def linear_slope(rows: Sequence[dict[str, object]]) -> float:
    if len(rows) < 2:
        return float("nan")
    x = [float(row["step"]) for row in rows]
    y = [float(row["fid"]) for row in rows]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator


def save_summary(rows: list[dict[str, object]], output_root: Path) -> dict[str, object]:
    rows = sorted(rows, key=lambda row: int(row["step"]))
    csv_path = output_root / "sit-s-2_unguided_fid5k_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tail = rows[-min(3, len(rows)) :]
    tail_slope = linear_slope(tail)
    last_delta = (
        float(rows[-1]["fid"]) - float(rows[-2]["fid"])
        if len(rows) >= 2
        else float("nan")
    )
    improving_tail = bool(
        len(rows) >= 2
        and last_delta < 0.0
        and (len(tail) < 3 or tail_slope < 0.0)
    )
    summary = {
        "protocol": "imagenet100_sit_unguided_fid5k_curve_v1",
        "steps": [int(row["step"]) for row in rows],
        "fids": [float(row["fid"]) for row in rows],
        "latest_delta": last_delta,
        "tail_linear_slope_fid_per_step": tail_slope,
        "improving_tail": improving_tail,
        "decision_rule": "latest FID decreases and the last-three-point linear slope is negative",
        "csv": str(csv_path),
        "rows": rows,
    }
    atomic_json_dump(summary, output_root / "sit-s-2_unguided_fid5k_curve.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=parse_steps, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument("--torchrun", default=shutil.which("torchrun") or "torchrun")
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--per-rank-batch-size", type=int, default=64)
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=7.5)
    parser.add_argument("--sampling-cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.30)
    parser.add_argument("--fid-cuda-visible-devices", default="0")
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=9 * 1024)
    parser.add_argument("--memory-poll-interval", type=float, default=0.25)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--require-improving-tail", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    # Do not call ``resolve`` here. The ADM environment's ``bin/python`` is a
    # symlink, and Python uses that entry path plus the adjacent pyvenv.cfg to
    # activate the environment containing TensorFlow.
    args.adm_python = absolute_without_resolving_symlinks(args.adm_python)
    args.sampling_gpu_indices = parse_gpu_indices(args.sampling_cuda_visible_devices)
    args.fid_gpu_indices = parse_gpu_indices(args.fid_cuda_visible_devices)
    if len(args.fid_gpu_indices) != 1:
        raise ValueError("ADM FID must be restricted to exactly one visible GPU")
    if not args.reference.is_file():
        raise FileNotFoundError(f"missing fixed ImageNet-100 reference: {args.reference}")
    if not args.adm_python.is_file():
        raise FileNotFoundError(f"missing ADM evaluator environment: {args.adm_python}")
    if min(
        args.num_samples,
        args.per_rank_batch_size,
        args.vae_decode_batch_size,
        args.fid_batch_size,
        args.cuda_allocator_limit_gib,
        args.gpu_memory_ceiling_mib,
        args.memory_poll_interval,
    ) <= 0:
        raise ValueError("sample and batch sizes must be positive")
    if not 0.0 < args.fid_gpu_memory_fraction < 1.0:
        raise ValueError("FID GPU memory fraction must be between 0 and 1")
    if args.cuda_allocator_limit_gib * 1024 >= args.gpu_memory_ceiling_mib:
        raise ValueError("PyTorch allocator limit must leave headroom below the GPU ceiling")
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.sampling_cuda_visible_devices
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    rows = [evaluate_checkpoint(step=step, args=args, env=env) for step in args.steps]
    summary = save_summary(rows, args.output_root)
    print(json.dumps(summary, indent=2), flush=True)
    if args.require_improving_tail and not summary["improving_tail"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
