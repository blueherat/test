#!/usr/bin/env python3
"""Paired FID-1K sweep for two internal-head amplitudes before t=0.5.

The strong model is the SiT-v 800K EMA checkpoint.  The first segment always
uses the post-trained depth-4 v head; the second uses depth 4 by default and
can instead use depth 10 with ``--second-depth 10``.  The sampled field is

    [0, first_end):       S + gamma_first  * (S - W4)
    [first_end, end):     S + gamma_second * (S - W{4 or 10})
    [end, 1]:             S

The default boundaries are 0.25 and 0.5.  Therefore (0.6, 0.6) is exactly the
previous best early-gamma=0.6, late-gamma=0 condition, while unequal gammas
test whether the useful early interval has a non-constant amplitude profile.

The sweep uses identical noise, labels, model weights, sampler, and ADM metric
for every condition.  It runs a common coarse grid, verifies the historical
(0.6, 0.6) anchor, and then evaluates a local grid around the coarse optimum.
Generated sample arrays are deleted after metric computation by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_NOISE = "ab8419c7fdfd5b15dacbf4d37a3d567158e4332f25fd94580d3df73bac87e2c2"
EXPECTED_LABEL = "7c3ae6894e7ebab5c9b6524606f03b6a56b38dccbe472ff40edde26e48654fe6"
HISTORICAL_CONSTANT_G06_FID = 64.99576588867058
HISTORICAL_FIRST_ONLY_G06_FID = 72.89829712444464
ANCHOR_TOL = 0.15
DEFAULT_COARSE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def detect_repo() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, here.parent, Path(__file__).resolve().parent.parent):
        if (candidate / "experiments/compute_adm_fid.py").is_file():
            return candidate
    raise FileNotFoundError("Cannot find eqvae repository")


def detect_data() -> Path:
    marker = Path("runs/sit-s-2_seed0/checkpoints/step_00800000.pt")
    for candidate in (
        Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow"),
        Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow"),
    ):
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError("Cannot find ImageNet-100 SiT data root")


def detect_adm_python() -> Path:
    for candidate in (
        Path("/data/shared/envs/adm-fid/bin/python"),
        Path("/home/zhoushunyu/data/shared/envs/adm-fid/bin/python"),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Cannot find ADM-FID environment")


def runtime_paths(repo: Path, data: Path, adm_python: Path) -> dict[str, Path]:
    paths = {
        "strong": data / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt",
        "depth4": (
            data
            / "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
        ),
        "depth10": (
            data
            / "multiscale_guidance_study_v1/runs/depth10_v/checkpoints/step_00050000.pt"
        ),
        "reference": (
            data / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
        ),
        "compute_fid": repo / "experiments/compute_adm_fid.py",
        "adm_python": adm_python,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))
    return paths


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_gpus(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated GPU indices") from exc
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("GPU indices must be unique and non-negative")
    return values


def parse_gammas(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated gamma values") from exc
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise argparse.ArgumentTypeError("Gammas must be finite and non-negative")
    return tuple(sorted(set(round(value, 8) for value in values)))


def gamma_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    return text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Condition:
    gamma_first: float
    gamma_second: float
    first_end: float = 0.25
    guidance_end: float = 0.5
    second_depth: int = 4

    @property
    def name(self) -> str:
        prefix = "depth4" if self.second_depth == 4 else "depth4_to_depth10"
        return f"{prefix}_g1{gamma_tag(self.gamma_first)}_g2{gamma_tag(self.gamma_second)}"

    def gamma_at(self, time: float) -> float:
        if time < self.first_end:
            return self.gamma_first
        if time < self.guidance_end:
            return self.gamma_second
        return 0.0

    def depth_at(self, time: float) -> int | None:
        if time < self.first_end:
            return 4
        if time < self.guidance_end:
            return self.second_depth
        return None

    def payload(self) -> dict[str, Any]:
        if not 0 < self.first_end < self.guidance_end < 1:
            raise ValueError("Expected 0 < first_end < guidance_end < 1")
        if min(self.gamma_first, self.gamma_second) < 0:
            raise ValueError("Gammas must be non-negative")
        if self.second_depth not in (4, 10):
            raise ValueError("second_depth must be 4 or 10")
        payload = {
            "format": "eqvae_internal_early_two_segment_condition_v1",
            "name": self.name,
            "weak_depth": 4,
            "gamma_first": float(self.gamma_first),
            "gamma_second": float(self.gamma_second),
            "gamma_late": 0.0,
            "first_interval": [0.0, float(self.first_end)],
            "second_interval": [float(self.first_end), float(self.guidance_end)],
            "late_interval": [float(self.guidance_end), 1.0],
            "formula": "S + gamma_segment*(S-W4); gamma_late=0",
        }
        # Preserve the exact v1 payload for the already completed d4->d4 run.
        if self.second_depth == 10:
            payload.update(
                {
                    "format": "eqvae_internal_early_two_segment_condition_v2",
                    "first_weak_depth": 4,
                    "second_weak_depth": 10,
                    "formula": (
                        "S + gamma_first*(S-W4) for first interval; "
                        "S + gamma_second*(S-W10) for second interval; gamma_late=0"
                    ),
                }
            )
            payload.pop("weak_depth")
        return payload


def condition_from_payload(payload: dict[str, Any]) -> Condition:
    first = payload["first_interval"]
    second = payload["second_interval"]
    condition = Condition(
        gamma_first=float(payload["gamma_first"]),
        gamma_second=float(payload["gamma_second"]),
        first_end=float(first[1]),
        guidance_end=float(second[1]),
        second_depth=int(payload.get("second_weak_depth", payload.get("weak_depth", 4))),
    )
    if payload != condition.payload():
        raise ValueError("Non-canonical condition payload")
    return condition


def load_repo_modules(repo: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo))
    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_source_with_heads,
        load_internal_head_for_source,
        load_sit_field_model,
    )
    from experiments.sample_imagenet100_sit_fid import (
        configure_cuda_allocator,
        decode_latents_in_chunks,
        official_pixel_quantization,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        atomic_json_dump,
        load_official_sit_module,
    )

    return locals()


def reusable(
    path: Path, condition: Condition, num_samples: int, batch_size: int, seed: int
) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        manifest = result["sampling_manifest"]
        sampling = manifest["sampling"]
        metrics = result["metrics"]
        return (
            result["condition"] == condition.payload()
            and int(sampling["num_samples"]) == num_samples
            and int(sampling["batch_size"]) == batch_size
            and int(sampling["seed"]) == seed
            and bool(manifest["noise_sha256"])
            and bool(manifest["label_sha256"])
            and all(
                isinstance(metrics.get(key), (int, float))
                and math.isfinite(float(metrics[key]))
                for key in ("fid", "sfid", "inception_score")
            )
        )
    except Exception:
        return False


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from diffusers.models import AutoencoderKL
    from torchdiffeq import odeint
    from torchvision.utils import save_image

    repo = Path(args.repo).resolve()
    data = Path(args.data).resolve()
    # Keep the dedicated ADM-FID launcher path intact.  It may be a wrapper or
    # symlink whose execution environment differs from its resolved target.
    paths = runtime_paths(repo, data, Path(args.adm_python))
    condition = condition_from_payload(read_json(Path(args.condition_json)))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if reusable(
        result_path, condition, args.num_samples, args.batch_size, args.seed
    ):
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
        raise ValueError("Strong checkpoint must predict native velocity")

    heads = {}
    required_depths = (4,) if condition.second_depth == 4 else (4, 10)
    for depth in required_depths:
        name = f"depth{depth}_v"
        heads[name] = modules["load_internal_head_for_source"](
            checkpoint_path=paths[f"depth{depth}"],
            name=name,
            head_weights="ema",
            model=strong,
            sit_module=sit_module,
            source_checkpoint_path=paths["strong"],
            source_metadata=source_metadata,
            device=device,
        )
    vae = (
        AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    class Field:
        def __init__(self, labels: Any):
            self.labels = labels
            self.nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            times = time.expand(len(latent))
            full, trained, _ = modules["evaluate_source_with_heads"](
                strong, latent, times, self.labels, heads=heads
            )
            gamma = condition.gamma_at(float(time.detach().float().item()))
            if gamma == 0.0:
                return full
            depth = condition.depth_at(float(time.detach().float().item()))
            if depth is None:
                raise AssertionError("Nonzero gamma outside the guidance interval")
            return full + gamma * (full - trained[f"depth{depth}_v"])

    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    cursor = 0
    total_nfe = 0
    preview = None

    with torch.inference_mode():
        while cursor < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - cursor)
            batch_index = cursor // args.batch_size
            generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
            noise = torch.randn(
                current_batch,
                *modules["LATENT_SHAPE"],
                generator=generator,
                device=device,
            )
            labels = torch.randint(
                0,
                modules["NUM_CLASSES"],
                (current_batch,),
                generator=generator,
                device=device,
            )
            field = Field(labels)
            endpoint = odeint(
                field,
                noise.float(),
                torch.tensor([0.0, 1.0], device=device),
                method="dopri5",
                atol=args.atol,
                rtol=args.rtol,
            )[-1]
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
            total_nfe += field.nfe
            cursor = stop
            if cursor == current_batch or cursor == args.num_samples or cursor % 256 == 0:
                print(
                    json.dumps(
                        {
                            "condition": condition.name,
                            "generated": cursor,
                            "total": args.num_samples,
                            "last_batch_nfe": field.nfe,
                        }
                    ),
                    flush=True,
                )

    sample_path = output / f"samples_n{args.num_samples}.npz"
    label_path = output / f"labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    save_image(preview, output / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))

    manifest = {
        "format": "eqvae_internal_early_two_segment_samples_v1",
        "condition": condition.payload(),
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "integrator": "dopri5",
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "strong": strong_metadata,
        "heads": {
            name: {
                "depth": head.depth,
                "prediction_target": head.prediction_target,
                "checkpoint": head.checkpoint,
                "checkpoint_sha256": head.checkpoint_sha256,
            }
            for name, head in heads.items()
        },
        "noise_sha256": noise_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "total_nfe": total_nfe,
        "samples": str(sample_path),
        "labels": str(label_path),
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    modules["atomic_json_dump"](manifest, output / "sampling_manifest.json")

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
        "format": "eqvae_internal_early_two_segment_result_v1",
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
            {"event": "complete", "condition": condition.name, "fid": metrics["fid"]}
        ),
        flush=True,
    )


def run_one(
    *,
    script: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    gpu: int,
    root: Path,
    phase: str,
    condition: Condition,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = root / phase / condition.name
    output.mkdir(parents=True, exist_ok=True)
    condition_path = output / "condition.json"
    atomic_json(condition_path, condition.payload())
    result_path = output / "condition_result.json"
    if reusable(
        result_path, condition, args.num_samples, args.batch_size, args.seed
    ):
        result = read_json(result_path)
        print(
            f"[reuse] {condition.name}: FID={float(result['metrics']['fid']):.4f}",
            flush=True,
        )
        return result

    command = [
        sys.executable,
        str(script),
        "worker",
        "--repo",
        str(repo),
        "--data",
        str(data),
        "--adm-python",
        str(adm_python),
        "--condition-json",
        str(condition_path),
        "--output-dir",
        str(output),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--vae-decode-batch-size",
        "2",
        "--seed",
        str(args.seed),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--fid-batch-size",
        str(args.fid_batch_size),
        "--fid-gpu-memory-fraction",
        str(args.fid_gpu_memory_fraction),
    ]
    if args.keep_samples:
        command.append("--keep-samples")
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
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-50:])
        raise RuntimeError(f"{condition.name} failed on GPU {gpu}\n{tail}")
    result = read_json(result_path)
    print(
        f"[GPU {gpu}] {condition.name}: FID={float(result['metrics']['fid']):.4f}",
        flush=True,
    )
    return result


def run_jobs(
    jobs: list[tuple[str, Condition]],
    *,
    gpus: tuple[int, ...],
    script: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lanes: list[list[tuple[str, Condition]]] = [[] for _ in gpus]
    for index, job in enumerate(jobs):
        lanes[index % len(gpus)].append(job)

    def lane(gpu: int, items: list[tuple[str, Condition]]) -> list[dict[str, Any]]:
        return [
            run_one(
                script=script,
                repo=repo,
                data=data,
                adm_python=adm_python,
                gpu=gpu,
                root=root,
                phase=phase,
                condition=condition,
                args=args,
            )
            for phase, condition in items
        ]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(lane, gpu, items)
            for gpu, items in zip(gpus, lanes)
            if items
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def rows_from_phase(root: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted((root / phase).glob("*/condition_result.json")):
        result = read_json(result_path)
        condition = result["condition"]
        manifest = result["sampling_manifest"]
        metrics = result["metrics"]
        rows.append(
            {
                "phase": phase,
                "gamma_first": float(condition["gamma_first"]),
                "gamma_second": float(condition["gamma_second"]),
                "first_depth": int(condition.get("first_weak_depth", 4)),
                "second_depth": int(
                    condition.get("second_weak_depth", condition.get("weak_depth", 4))
                ),
                "first_end": float(condition["first_interval"][1]),
                "guidance_end": float(condition["second_interval"][1]),
                "fid": float(metrics["fid"]),
                "sfid": float(metrics["sfid"]),
                "inception_score": float(metrics["inception_score"]),
                "total_nfe": int(manifest["total_nfe"]),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
            }
        )
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"coarse": 0, "controls": 1, "refine": 2}
    selected: dict[tuple[float, float], dict[str, Any]] = {}
    for row in rows:
        key = (round(row["gamma_first"], 8), round(row["gamma_second"], 8))
        previous = selected.get(key)
        if previous is None or priority[row["phase"]] >= priority[previous["phase"]]:
            selected[key] = row
    return list(selected.values())


def verify_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not rows:
        raise RuntimeError("No completed conditions")
    noise_hashes = {row["noise_sha256"] for row in rows}
    label_hashes = {row["label_sha256"] for row in rows}
    if len(noise_hashes) != 1 or len(label_hashes) != 1:
        raise RuntimeError(
            f"Sweep is not paired: noise={len(noise_hashes)}, labels={len(label_hashes)}"
        )
    if args.num_samples == 1000 and args.batch_size == 8 and args.seed == 0:
        if next(iter(noise_hashes)) != EXPECTED_NOISE:
            raise RuntimeError("Noise hash does not match the historical paired set")
        if next(iter(label_hashes)) != EXPECTED_LABEL:
            raise RuntimeError("Label hash does not match the historical paired set")
        anchor_pair = (0.6, 0.6) if args.second_depth == 4 else (0.6, 0.0)
        expected_fid = (
            HISTORICAL_CONSTANT_G06_FID
            if args.second_depth == 4
            else HISTORICAL_FIRST_ONLY_G06_FID
        )
        anchor = next(
            (
                row
                for row in rows
                if round(row["gamma_first"], 8) == anchor_pair[0]
                and round(row["gamma_second"], 8) == anchor_pair[1]
            ),
            None,
        )
        if anchor is None:
            raise RuntimeError(f"The historical {anchor_pair} anchor is missing")
        delta = abs(anchor["fid"] - expected_fid)
        if delta > ANCHOR_TOL:
            raise RuntimeError(
                "Historical anchor reproduction failed: "
                f"new={anchor['fid']:.6f}, old={expected_fid:.6f}, "
                f"delta={delta:.6f}, tolerance={ANCHOR_TOL}"
            )


def local_pairs(
    center_first: float,
    center_second: float,
    *,
    radius: float,
    step: float,
) -> set[tuple[float, float]]:
    count = int(round(radius / step))
    if step <= 0 or radius < 0 or abs(count * step - radius) > 1e-8:
        raise ValueError("Refine radius must be a non-negative multiple of step")
    offsets = tuple(index * step for index in range(-count, count + 1))
    first = {round(max(0.0, center_first + offset), 8) for offset in offsets}
    second = {round(max(0.0, center_second + offset), 8) for offset in offsets}
    return set(itertools.product(first, second))


def write_summary(root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = deduplicate(rows)
    verify_rows(rows, args)
    rows.sort(key=lambda row: (row["gamma_first"], row["gamma_second"]))
    best = min(rows, key=lambda row: row["fid"])
    anchor_pair = (0.6, 0.6) if args.second_depth == 4 else (0.6, 0.0)
    anchor = next(
        row
        for row in rows
        if round(row["gamma_first"], 8) == anchor_pair[0]
        and round(row["gamma_second"], 8) == anchor_pair[1]
    )
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_path = summary_dir / "all_conditions.csv"
    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_path = summary_dir / "best_condition.json"
    atomic_json(best_path, best)
    comparison: dict[str, Any] | None = None
    if (
        args.second_depth == 10
        and args.num_samples == 1000
        and args.batch_size == 8
        and args.seed == 0
    ):
        depth4_path = (
            root.parent
            / "internal_head_early_two_segment_gamma_sweep_v1/summary/all_conditions.csv"
        )
        if not depth4_path.is_file():
            raise FileNotFoundError(f"Missing completed depth4 sweep: {depth4_path}")
        with depth4_path.open(newline="", encoding="utf-8") as handle:
            depth4_rows = list(csv.DictReader(handle))
        depth4_map = {
            (round(float(row["gamma_first"]), 8), round(float(row["gamma_second"]), 8)): row
            for row in depth4_rows
        }
        depth10_map = {
            (round(row["gamma_first"], 8), round(row["gamma_second"], 8)): row
            for row in rows
        }
        paired_rows = []
        for pair in sorted(set(depth4_map) & set(depth10_map)):
            depth4_row = depth4_map[pair]
            depth10_row = depth10_map[pair]
            paired_rows.append(
                {
                    "gamma_first": pair[0],
                    "gamma_second": pair[1],
                    "fid_depth4_second": float(depth4_row["fid"]),
                    "fid_depth10_second": float(depth10_row["fid"]),
                    "depth10_identity_benefit_fid": (
                        float(depth4_row["fid"]) - float(depth10_row["fid"])
                    ),
                    "sfid_depth4_second": float(depth4_row["sfid"]),
                    "sfid_depth10_second": float(depth10_row["sfid"]),
                    "is_depth4_second": float(depth4_row["inception_score"]),
                    "is_depth10_second": float(depth10_row["inception_score"]),
                }
            )
        paired_path = summary_dir / "paired_second_depth_delta.csv"
        with paired_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
        best_depth4 = min(depth4_rows, key=lambda row: float(row["fid"]))
        depth4_best_pair = (
            round(float(best_depth4["gamma_first"]), 8),
            round(float(best_depth4["gamma_second"]), 8),
        )
        depth10_at_depth4_best = depth10_map.get(depth4_best_pair)
        comparison = {
            "best_depth4_second": {
                "gamma_first": depth4_best_pair[0],
                "gamma_second": depth4_best_pair[1],
                "fid": float(best_depth4["fid"]),
            },
            "best_depth10_second": best,
            "optimized_depth10_benefit_fid": float(best_depth4["fid"]) - best["fid"],
            "depth10_at_depth4_optimum": depth10_at_depth4_best,
            "same_pair_depth10_benefit_at_depth4_optimum": (
                float(best_depth4["fid"]) - float(depth10_at_depth4_best["fid"])
                if depth10_at_depth4_best is not None
                else None
            ),
            "max_same_pair_depth10_benefit": max(
                paired_rows, key=lambda row: row["depth10_identity_benefit_fid"]
            ),
            "min_same_pair_depth10_benefit": min(
                paired_rows, key=lambda row: row["depth10_identity_benefit_fid"]
            ),
            "paired_conditions": len(paired_rows),
            "paired_file": str(paired_path),
        }

    summary = {
        "format": "eqvae_internal_early_two_segment_summary_v1",
        "scientific_question": (
            "With guidance disabled after t=0.5, does replacing the second "
            f"early segment by depth{args.second_depth} improve the optimized schedule?"
        ),
        "best": best,
        "historical_anchor": anchor,
        "best_improvement_over_anchor_fid": anchor["fid"] - best["fid"],
        "best_is_nonconstant": (
            round(best["gamma_first"], 8) != round(best["gamma_second"], 8)
        ),
        "protocol": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "first_end": args.first_end,
            "guidance_end": args.guidance_end,
            "gamma_late": 0.0,
            "first_depth": 4,
            "second_depth": args.second_depth,
            "coarse_gammas": list(args.coarse_gammas),
            "refine": args.refine,
            "refine_radius": args.refine_radius,
            "refine_step": args.refine_step,
        },
        "pairing": {
            "noise_sha256": rows[0]["noise_sha256"],
            "label_sha256": rows[0]["label_sha256"],
            "verified": True,
        },
        "files": {"all_conditions": str(all_path), "best_condition": str(best_path)},
        "depth4_comparison": comparison,
    }
    atomic_json(summary_dir / "summary.json", summary)
    print("\n=== FINAL ===")
    print(
        f"best: gamma_first={best['gamma_first']:.3f}, "
        f"gamma_second={best['gamma_second']:.3f}, FID={best['fid']:.4f}"
    )
    print(
        f"historical anchor {anchor_pair}: FID={anchor['fid']:.4f}; "
        f"improvement={anchor['fid'] - best['fid']:+.4f}"
    )
    if comparison is not None:
        print(
            "optimized depth10 benefit over optimized depth4: "
            f"{comparison['optimized_depth10_benefit_fid']:+.4f} FID"
        )
    print(f"summary: {summary_dir / 'summary.json'}")


def sweep(args: argparse.Namespace) -> None:
    repo = detect_repo()
    data = detect_data()
    adm_python = detect_adm_python()
    runtime_paths(repo, data, adm_python)
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else data
        / (
            "internal_head_early_two_segment_gamma_sweep_v1"
            if args.second_depth == 4
            else "internal_head_early_two_segment_depth10_gamma_sweep_v1"
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    pairs = list(itertools.product(args.coarse_gammas, repeat=2))
    jobs = [
        (
            "coarse",
            Condition(
                gamma_first=gamma_first,
                gamma_second=gamma_second,
                first_end=args.first_end,
                guidance_end=args.guidance_end,
                second_depth=args.second_depth,
            ),
        )
        for gamma_first, gamma_second in pairs
    ]
    atomic_json(
        root / "request.json",
        {
            "format": "eqvae_internal_early_two_segment_request_v1",
            "repo": str(repo),
            "data": str(data),
            "gpus": list(args.gpus),
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "first_end": args.first_end,
            "guidance_end": args.guidance_end,
            "first_depth": 4,
            "second_depth": args.second_depth,
            "gamma_late": 0.0,
            "coarse_gammas": list(args.coarse_gammas),
            "coarse_conditions": len(jobs),
            "refine": args.refine,
            "refine_radius": args.refine_radius,
            "refine_step": args.refine_step,
        },
    )
    print("=== EARLY TWO-SEGMENT GAMMA SWEEP ===")
    print(f"GPUs: {args.gpus}")
    print(
        f"intervals: [0,{args.first_end}) depth4, "
        f"[{args.first_end},{args.guidance_end}) depth{args.second_depth}, late=0"
    )
    print(f"coarse grid: {args.coarse_gammas} x {args.coarse_gammas}")
    print(f"coarse conditions: {len(jobs)}")
    print(f"output: {root}")
    if args.dry_run:
        for phase, condition in jobs:
            print(phase, condition.payload())
        return

    # Run the historical anchor first so a semantic/RNG regression fails early.
    anchor_pair = (0.6, 0.6) if args.second_depth == 4 else (0.6, 0.0)
    anchor = Condition(
        anchor_pair[0],
        anchor_pair[1],
        args.first_end,
        args.guidance_end,
        args.second_depth,
    )
    run_jobs(
        [("coarse", anchor)],
        gpus=(args.gpus[0],),
        script=script,
        repo=repo,
        data=data,
        adm_python=adm_python,
        root=root,
        args=args,
    )
    verify_rows(rows_from_phase(root, "coarse"), args)
    remaining = [job for job in jobs if job[1] != anchor]
    run_jobs(
        remaining,
        gpus=args.gpus,
        script=script,
        repo=repo,
        data=data,
        adm_python=adm_python,
        root=root,
        args=args,
    )
    coarse_rows = rows_from_phase(root, "coarse")
    verify_rows(coarse_rows, args)
    coarse_best = min(coarse_rows, key=lambda row: row["fid"])
    print(
        f"coarse best: ({coarse_best['gamma_first']:.3f}, "
        f"{coarse_best['gamma_second']:.3f}), FID={coarse_best['fid']:.4f}"
    )

    # Always evaluate the completed d4 schedule's optimum so the second-head
    # identity has a direct same-pair comparison even if the d10 refinement
    # center moves elsewhere.
    control_jobs = []
    if args.second_depth == 10:
        control_jobs = [
            (
                "controls",
                Condition(0.6, 0.7, args.first_end, args.guidance_end, 10),
            )
        ]
        run_jobs(
            control_jobs,
            gpus=(args.gpus[0],),
            script=script,
            repo=repo,
            data=data,
            adm_python=adm_python,
            root=root,
            args=args,
        )

    if args.refine:
        refine_pairs = local_pairs(
            coarse_best["gamma_first"],
            coarse_best["gamma_second"],
            radius=args.refine_radius,
            step=args.refine_step,
        )
        coarse_pairs = {
            (round(first, 8), round(second, 8)) for first, second in pairs
        }
        refine_pairs -= coarse_pairs
        if args.second_depth == 10:
            refine_pairs.discard((0.6, 0.7))
        refine_jobs = [
            (
                "refine",
                Condition(
                    first,
                    second,
                    args.first_end,
                    args.guidance_end,
                    args.second_depth,
                ),
            )
            for first, second in sorted(refine_pairs)
        ]
        print(f"refine conditions: {len(refine_jobs)}")
        run_jobs(
            refine_jobs,
            gpus=args.gpus,
            script=script,
            repo=repo,
            data=data,
            adm_python=adm_python,
            root=root,
            args=args,
        )

    rows = (
        rows_from_phase(root, "coarse")
        + rows_from_phase(root, "controls")
        + rows_from_phase(root, "refine")
    )
    write_summary(root, rows, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    sweep_parser.add_argument("--coarse-gammas", type=parse_gammas, default=DEFAULT_COARSE)
    sweep_parser.add_argument("--first-end", type=float, default=0.25)
    sweep_parser.add_argument("--guidance-end", type=float, default=0.5)
    sweep_parser.add_argument("--second-depth", type=int, choices=(4, 10), default=4)
    sweep_parser.add_argument("--num-samples", type=int, default=1000)
    sweep_parser.add_argument("--batch-size", type=int, default=8)
    sweep_parser.add_argument("--seed", type=int, default=0)
    sweep_parser.add_argument("--atol", type=float, default=1e-6)
    sweep_parser.add_argument("--rtol", type=float, default=1e-3)
    sweep_parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    sweep_parser.add_argument("--fid-batch-size", type=int, default=8)
    sweep_parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    sweep_parser.add_argument("--output-root", type=Path)
    sweep_parser.add_argument(
        "--refine", action=argparse.BooleanOptionalAction, default=True
    )
    sweep_parser.add_argument("--refine-radius", type=float, default=0.1)
    sweep_parser.add_argument("--refine-step", type=float, default=0.05)
    sweep_parser.add_argument("--keep-samples", action="store_true")
    sweep_parser.add_argument("--dry-run", action="store_true")

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--repo", required=True)
    worker_parser.add_argument("--data", required=True)
    worker_parser.add_argument("--adm-python", required=True)
    worker_parser.add_argument("--condition-json", required=True)
    worker_parser.add_argument("--output-dir", required=True)
    worker_parser.add_argument("--num-samples", type=int, required=True)
    worker_parser.add_argument("--batch-size", type=int, required=True)
    worker_parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    worker_parser.add_argument("--seed", type=int, required=True)
    worker_parser.add_argument("--atol", type=float, default=1e-6)
    worker_parser.add_argument("--rtol", type=float, default=1e-3)
    worker_parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    worker_parser.add_argument("--fid-batch-size", type=int, default=8)
    worker_parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    worker_parser.add_argument("--keep-samples", action="store_true")
    return parser


def validate_sweep(args: argparse.Namespace) -> None:
    if not 0 < args.first_end < args.guidance_end < 1:
        raise ValueError("Expected 0 < first_end < guidance_end < 1")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("Sample count and batch size must be positive")
    if args.cuda_allocator_limit_gib <= 0 or args.fid_batch_size <= 0:
        raise ValueError("Memory limit and FID batch size must be positive")
    if args.refine_step <= 0 or args.refine_radius < 0:
        raise ValueError("Invalid refine step/radius")
    ratio = args.refine_radius / args.refine_step
    if abs(ratio - round(ratio)) > 1e-8:
        raise ValueError("Refine radius must be an integer multiple of step")
    if args.num_samples == 1000:
        required = {0.6} if args.second_depth == 4 else {0.0, 0.6}
        if not required.issubset(args.coarse_gammas):
            raise ValueError(
                f"Default 1K protocol requires gammas {sorted(required)} for anchor verification"
            )


def self_test() -> None:
    condition = Condition(0.3, 0.7, 0.25, 0.5)
    assert condition.gamma_at(0.0) == 0.3
    assert condition.gamma_at(0.249999) == 0.3
    assert condition.gamma_at(0.25) == 0.7
    assert condition.gamma_at(0.499999) == 0.7
    assert condition.gamma_at(0.5) == 0.0
    assert condition.gamma_at(1.0) == 0.0
    assert condition_from_payload(condition.payload()) == condition
    mixed = Condition(0.3, 0.7, 0.25, 0.5, 10)
    assert mixed.depth_at(0.0) == 4
    assert mixed.depth_at(0.249999) == 4
    assert mixed.depth_at(0.25) == 10
    assert mixed.depth_at(0.499999) == 10
    assert mixed.depth_at(0.5) is None
    assert condition_from_payload(mixed.payload()) == mixed
    assert len(local_pairs(0.6, 0.6, radius=0.1, step=0.05)) == 25
    print("self-test: OK")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "self-test":
        self_test()
        return
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "sweep":
        validate_sweep(args)
        sweep(args)
    elif args.command == "worker":
        worker(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
