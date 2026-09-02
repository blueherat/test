#!/usr/bin/env python3
"""Paired solver audit for the best ImageNet-100 depth-4 Internal Guidance.

The learned continuous field is kept fixed:

    G = S + gamma(t) * (S - W4),

with gamma 0.6 on [0, .25), 0.7 on [.25, .5), and zero afterwards.
Only the numerical solver changes.  Implicit methods predict an endpoint and
then apply relaxed Picard corrections in which the vector field is evaluated
at the candidate future state.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.implicit_fixed_point_solvers import integrate_fixed_grid  # noqa: E402
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
    load_repo_modules,
    parse_gpus,
    read_json,
    runtime_paths,
)


EXPECTED_NOISE = "ab8419c7fdfd5b15dacbf4d37a3d567158e4332f25fd94580d3df73bac87e2c2"
EXPECTED_LABEL = "7c3ae6894e7ebab5c9b6524606f03b6a56b38dccbe472ff40edde26e48654fe6"
HISTORICAL_DOPRI5_FID = 64.85087470050013
METHODS = {
    "dopri5",
    "euler",
    "heun",
    "backward_euler",
    "implicit_midpoint",
    "implicit_trapezoid",
}


def _tag(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"
    return text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Condition:
    method: str
    steps: int = 0
    corrections: int = 1
    relaxation: float = 1.0

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unsupported method: {self.method}")
        if self.method == "dopri5":
            if self.steps != 0 or self.corrections != 1 or self.relaxation != 1.0:
                raise ValueError("dopri5 does not accept fixed-step settings")
            return
        if self.steps <= 0:
            raise ValueError("fixed-step methods require positive steps")
        if self.corrections <= 0:
            raise ValueError("corrections must be positive")
        if not math.isfinite(self.relaxation) or not 0 < self.relaxation <= 1:
            raise ValueError("relaxation must lie in (0, 1]")

    @property
    def name(self) -> str:
        self.validate()
        if self.method == "dopri5":
            return "depth4_ig_dopri5"
        suffix = f"_n{self.steps}"
        if self.method.startswith("implicit") or self.method == "backward_euler":
            suffix += f"_k{self.corrections}_r{_tag(self.relaxation)}"
        return f"depth4_ig_{self.method}{suffix}"

    @property
    def nominal_nfe(self) -> int | None:
        if self.method == "dopri5":
            return None
        if self.method == "euler":
            return self.steps
        if self.method == "heun":
            return 2 * self.steps
        return (1 + self.corrections) * self.steps

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": "eqvae_implicit_ig_solver_condition_v1",
            "name": self.name,
            "method": self.method,
            "steps": self.steps,
            "corrections": self.corrections,
            "relaxation": self.relaxation,
            "nominal_nfe_per_batch": self.nominal_nfe,
            "field": "S + gamma(t)*(S-W4)",
            "gamma_segments": [[0.0, 0.25, 0.6], [0.25, 0.5, 0.7], [0.5, 1.0, 0.0]],
        }


def condition_from_payload(payload: dict[str, Any]) -> Condition:
    condition = Condition(
        method=str(payload["method"]),
        steps=int(payload["steps"]),
        corrections=int(payload["corrections"]),
        relaxation=float(payload["relaxation"]),
    )
    if payload != condition.payload():
        raise ValueError("non-canonical condition payload")
    return condition


def parse_condition(text: str) -> Condition:
    fields = text.split(":")
    method = fields[0]
    if method == "dopri5":
        if len(fields) != 1:
            raise argparse.ArgumentTypeError("dopri5 takes no suffix")
        return Condition("dopri5")
    if method not in METHODS:
        raise argparse.ArgumentTypeError(f"unsupported method: {method}")
    try:
        steps = int(fields[1])
        corrections = int(fields[2]) if len(fields) >= 3 else 1
        relaxation = float(fields[3]) if len(fields) >= 4 else 1.0
    except (IndexError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "condition syntax is method:steps[:corrections[:relaxation]]"
        ) from error
    if len(fields) > 4:
        raise argparse.ArgumentTypeError("too many condition fields")
    condition = Condition(method, steps, corrections, relaxation)
    try:
        condition.validate()
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return condition


def gamma_at(time_value: float) -> float:
    if time_value < 0.25:
        return 0.6
    if time_value < 0.5:
        return 0.7
    return 0.0


def segment_step_counts(total: int) -> tuple[int, int, int]:
    if total < 4:
        raise ValueError("at least four fixed steps are required")
    first = max(1, int(round(0.25 * total)))
    second = max(1, int(round(0.25 * total)))
    late = total - first - second
    if late <= 0:
        raise ValueError("step allocation left no late segment")
    return first, second, late


def reusable(path: Path, condition: Condition, args: argparse.Namespace) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        manifest = result["sampling_manifest"]
        metrics = result.get("metrics")
        return (
            result["condition"] == condition.payload()
            and int(manifest["sampling"]["num_samples"]) == args.num_samples
            and int(manifest["sampling"]["batch_size"]) == args.batch_size
            and int(manifest["sampling"]["seed"]) == args.seed
            and bool(manifest["noise_sha256"])
            and bool(manifest["label_sha256"])
            and (
                args.skip_fid
                or isinstance(metrics, dict)
                and all(math.isfinite(float(metrics[key])) for key in ("fid", "sfid", "inception_score"))
            )
        )
    except Exception:
        return False


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    from diffusers.models import AutoencoderKL
    from torchdiffeq import odeint
    from torchvision.utils import save_image

    repo = Path(args.repo).resolve()
    data = Path(args.data).resolve()
    paths = runtime_paths(repo, data, Path(args.adm_python))
    condition = condition_from_payload(read_json(Path(args.condition_json)))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if reusable(result_path, condition, args):
        print(json.dumps({"event": "reuse", "condition": condition.name}), flush=True)
        return

    modules = load_repo_modules(repo)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    allocator = modules["configure_cuda_allocator"](
        device, limit_gib=args.cuda_allocator_limit_gib
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    sit_module, source_metadata = modules["load_official_sit_module"](
        Path(modules["DEFAULT_OFFICIAL_SIT_REPO"]).expanduser().resolve(),
        verify_source=True,
    )
    strong, semantics, strong_metadata = modules["load_sit_field_model"](
        checkpoint_path=paths["strong"],
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("implicit IG audit requires the native-v v800 source")
    head = modules["load_internal_head_for_source"](
        checkpoint_path=paths["depth4"],
        name="depth4_v",
        head_weights="ema",
        model=strong,
        sit_module=sit_module,
        source_checkpoint_path=paths["strong"],
        source_metadata=source_metadata,
        device=device,
    )
    heads = {"depth4_v": head}
    vae = (
        AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True)
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    class Field:
        def __init__(self, labels: Any, gamma: float | None = None):
            self.labels = labels
            self.gamma = gamma
            self.nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            times = time.expand(len(latent))
            full, trained, _ = modules["evaluate_source_with_heads"](
                strong, latent, times, self.labels, heads=heads
            )
            gamma = gamma_at(float(time.detach().float().item())) if self.gamma is None else self.gamma
            if gamma == 0.0:
                return full
            return full + gamma * (full - trained["depth4_v"])

    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    total_nfe = 0
    update_means: list[float] = []
    update_maxima: list[float] = []
    cursor = 0
    preview = None

    with torch.inference_mode():
        while cursor < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - cursor)
            batch_index = cursor // args.batch_size
            generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
            noise = torch.randn(
                current_batch, *modules["LATENT_SHAPE"], generator=generator, device=device
            )
            labels = torch.randint(
                0, modules["NUM_CLASSES"], (current_batch,), generator=generator, device=device
            )
            if condition.method == "dopri5":
                field = Field(labels)
                endpoint = odeint(
                    field,
                    noise.float(),
                    torch.tensor([0.0, 1.0], device=device),
                    method="dopri5",
                    atol=args.atol,
                    rtol=args.rtol,
                )[-1]
                batch_nfe = field.nfe
            else:
                state = noise.float()
                batch_nfe = 0
                counts = segment_step_counts(condition.steps)
                for (start, end, gamma), count in zip(
                    ((0.0, 0.25, 0.6), (0.25, 0.5, 0.7), (0.5, 1.0, 0.0)),
                    counts,
                ):
                    field = Field(labels, gamma)
                    result = integrate_fixed_grid(
                        field,
                        state,
                        torch.linspace(start, end, count + 1, device=device),
                        method=condition.method,
                        corrections=condition.corrections,
                        relaxation=condition.relaxation,
                    )
                    if result.nfe != field.nfe:
                        raise AssertionError("solver and field NFE accounting disagree")
                    state = result.endpoint
                    batch_nfe += result.nfe
                    update_means.append(result.mean_last_update_rms)
                    update_maxima.append(result.max_last_update_rms)
                endpoint = state
            if not torch.isfinite(endpoint).all():
                raise FloatingPointError(condition.name)
            decoded = modules["decode_latents_in_chunks"](
                vae,
                endpoint,
                scaling_factor=modules["SD_VAE_SCALING_FACTOR"],
                chunk_size=args.vae_decode_batch_size,
            )
            stop = cursor + current_batch
            images[cursor:stop] = modules["official_pixel_quantization"](decoded)
            labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
            noise_hash.update(noise.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().contiguous().numpy().tobytes())
            if preview is None:
                preview = decoded[: min(16, len(decoded))].cpu()
            total_nfe += batch_nfe
            cursor = stop
            if cursor == current_batch or cursor == args.num_samples or cursor % 256 == 0:
                print(
                    json.dumps(
                        {
                            "condition": condition.name,
                            "generated": cursor,
                            "total": args.num_samples,
                            "last_batch_nfe": batch_nfe,
                        }
                    ),
                    flush=True,
                )

    sample_path = output / f"samples_n{args.num_samples}.npz"
    label_path = output / f"labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    assert preview is not None
    save_image(preview, output / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))
    manifest = {
        "format": "eqvae_implicit_ig_solver_samples_v1",
        "condition": condition.payload(),
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "strong": strong_metadata,
        "head": {
            "depth": head.depth,
            "prediction_target": head.prediction_target,
            "checkpoint": head.checkpoint,
            "checkpoint_sha256": head.checkpoint_sha256,
        },
        "noise_sha256": noise_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "total_nfe": total_nfe,
        "picard_last_update_rms": {
            "mean": float(sum(update_means) / len(update_means)) if update_means else 0.0,
            "max": max(update_maxima, default=0.0),
        },
        "samples": str(sample_path),
        "labels": str(label_path),
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    modules["atomic_json_dump"](manifest, output / "sampling_manifest.json")

    metrics = None
    if not args.skip_fid:
        del vae, strong, heads, head
        gc.collect()
        torch.cuda.empty_cache()
        metric_path = output / "adm_metrics.json"
        environment = os.environ.copy()
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        subprocess.run(
            [
                str(paths["adm_python"]),
                str(paths["compute_fid"]),
                "--reference",
                str(paths["reference"]),
                "--samples",
                str(sample_path),
                "--batch-size",
                str(args.fid_batch_size),
                "--gpu-memory-fraction",
                str(args.fid_gpu_memory_fraction),
                "--output",
                str(metric_path),
            ],
            cwd=repo,
            env=environment,
            check=True,
        )
        metrics = read_json(metric_path)
    result = {
        "format": "eqvae_implicit_ig_solver_result_v1",
        "condition": condition.payload(),
        "sampling_manifest": manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
    }
    modules["atomic_json_dump"](result, result_path)
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "event": "complete",
                "condition": condition.name,
                "fid": None if metrics is None else metrics["fid"],
            }
        ),
        flush=True,
    )


def run_one(
    condition: Condition,
    gpu: int,
    root: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = root / condition.name
    output.mkdir(parents=True, exist_ok=True)
    condition_path = output / "condition.json"
    atomic_json(condition_path, condition.payload())
    result_path = output / "condition_result.json"
    if reusable(result_path, condition, args):
        return read_json(result_path)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--repo", str(repo),
        "--data", str(data),
        "--adm-python", str(adm_python),
        "--condition-json", str(condition_path),
        "--output-dir", str(output),
        "--num-samples", str(args.num_samples),
        "--batch-size", str(args.batch_size),
        "--vae-decode-batch-size", str(args.vae_decode_batch_size),
        "--seed", str(args.seed),
        "--atol", str(args.atol),
        "--rtol", str(args.rtol),
        "--cuda-allocator-limit-gib", str(args.cuda_allocator_limit_gib),
        "--fid-batch-size", str(args.fid_batch_size),
        "--fid-gpu-memory-fraction", str(args.fid_gpu_memory_fraction),
    ]
    if args.keep_samples:
        command.append("--keep-samples")
    if args.skip_fid:
        command.append("--skip-fid")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
        raise RuntimeError(f"{condition.name} failed on GPU {gpu}\n{tail}")
    result = read_json(result_path)
    metrics = result.get("metrics")
    suffix = "" if metrics is None else f" FID={float(metrics['fid']):.4f}"
    print(f"[GPU {gpu}] {condition.name}:{suffix}", flush=True)
    return result


def run_parallel(
    conditions: tuple[Condition, ...],
    gpus: tuple[int, ...],
    root: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lanes: list[list[Condition]] = [[] for _ in gpus]
    for index, condition in enumerate(conditions):
        lanes[index % len(gpus)].append(condition)

    def lane(gpu: int, items: list[Condition]) -> list[dict[str, Any]]:
        return [run_one(item, gpu, root, repo, data, adm_python, args) for item in items]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(lane, gpu, items) for gpu, items in zip(gpus, lanes) if items]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def write_summary(root: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = []
    for result in results:
        condition = result["condition"]
        manifest = result["sampling_manifest"]
        metrics = result.get("metrics")
        rows.append(
            {
                "condition": condition["name"],
                "method": condition["method"],
                "steps": condition["steps"],
                "corrections": condition["corrections"],
                "relaxation": condition["relaxation"],
                "nominal_nfe_per_batch": condition["nominal_nfe_per_batch"],
                "measured_total_nfe": manifest["total_nfe"],
                "picard_last_update_rms_mean": manifest["picard_last_update_rms"]["mean"],
                "picard_last_update_rms_max": manifest["picard_last_update_rms"]["max"],
                "fid": None if metrics is None else float(metrics["fid"]),
                "sfid": None if metrics is None else float(metrics["sfid"]),
                "inception_score": None if metrics is None else float(metrics["inception_score"]),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
            }
        )
    rows.sort(key=lambda row: row["condition"])
    if len({row["noise_sha256"] for row in rows}) != 1 or len({row["label_sha256"] for row in rows}) != 1:
        raise RuntimeError("conditions did not use paired noise and labels")
    if args.num_samples == 1000 and args.batch_size == 8 and args.seed == 0:
        if rows[0]["noise_sha256"] != EXPECTED_NOISE or rows[0]["label_sha256"] != EXPECTED_LABEL:
            raise RuntimeError("paired bank differs from the historical FID-1K protocol")
    measured = [row for row in rows if row["fid"] is not None]
    anchor = next((row for row in rows if row["method"] == "dopri5"), None)
    if anchor is not None and anchor["fid"] is not None:
        if args.num_samples == 1000 and abs(float(anchor["fid"]) - HISTORICAL_DOPRI5_FID) > 0.15:
            raise RuntimeError("Dopri5 historical anchor failed to reproduce")
        for row in rows:
            row["fid_delta_vs_dopri5"] = None if row["fid"] is None else row["fid"] - anchor["fid"]
    else:
        for row in rows:
            row["fid_delta_vs_dopri5"] = None
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "all_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(measured, key=lambda row: row["fid"]) if measured else None
    atomic_json(
        summary_dir / "summary.json",
        {
            "format": "eqvae_implicit_ig_solver_summary_v1",
            "question": "Does future-state fixed-point integration improve depth4 IG at finite NFE?",
            "protocol": {
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "atol": args.atol,
                "rtol": args.rtol,
            },
            "best": best,
            "dopri5_anchor": anchor,
            "rows": str(summary_dir / "all_conditions.csv"),
        },
    )
    if best is not None:
        print(json.dumps({"event": "summary", "best": best}, indent=2), flush=True)


def sweep(args: argparse.Namespace) -> None:
    repo = detect_repo()
    data = detect_data()
    adm_python = detect_adm_python()
    runtime_paths(repo, data, adm_python)
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else data / "implicit_future_state_ig_solver_v1"
    )
    root.mkdir(parents=True, exist_ok=True)
    conditions = tuple(dict.fromkeys(args.conditions))
    atomic_json(
        root / "request.json",
        {
            "format": "eqvae_implicit_ig_solver_request_v1",
            "conditions": [condition.payload() for condition in conditions],
            "gpus": list(args.gpus),
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    )
    print(json.dumps({"conditions": [item.name for item in conditions], "gpus": args.gpus}, indent=2))
    if args.dry_run:
        return
    results = run_parallel(conditions, args.gpus, root, repo, data, adm_python, args)
    write_summary(root, results, args)


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep_parser = subparsers.add_parser("sweep")
    add_shared(sweep_parser)
    sweep_parser.add_argument("--gpus", type=parse_gpus, default=(1, 2, 3))
    sweep_parser.add_argument("--output-root", type=Path)
    sweep_parser.add_argument("--dry-run", action="store_true")
    sweep_parser.add_argument(
        "--conditions",
        type=parse_condition,
        nargs="+",
        default=(
            Condition("dopri5"),
            Condition("euler", 64),
            Condition("heun", 32),
            Condition("backward_euler", 32, 1),
            Condition("backward_euler", 16, 3),
            Condition("implicit_midpoint", 32, 1),
            Condition("implicit_midpoint", 16, 3),
            Condition("implicit_trapezoid", 16, 3),
        ),
    )
    worker_parser = subparsers.add_parser("worker")
    add_shared(worker_parser)
    worker_parser.add_argument("--repo", type=Path, required=True)
    worker_parser.add_argument("--data", type=Path, required=True)
    worker_parser.add_argument("--adm-python", type=Path, required=True)
    worker_parser.add_argument("--condition-json", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts and batch sizes must be positive")
    if args.command == "worker":
        worker(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
