#!/usr/bin/env python3
"""Paired FID-1K gamma sweep for frozen SiT velocity heads and depth schedules.

This runner reuses the existing atomic evaluator:
    experiments/evaluate_imagenet100_sit_multiscale_condition.py

It does not train heads and does not rebuild the latent atlas.

Default experiment family
-------------------------
Static native velocity-head guidance:
    depth4_v, depth6_v, depth8_v, depth10_v, depth12_v

Depth schedules are all combinations of {4, 6, 8, 10} of lengths 2, 3, and 4.
For every combination, both forward (shallow->deep) and reverse (deep->shallow)
orders are evaluated, with both native and RMS-matched gap amplitudes.

Explicitly excluded by design:
    depth8_x, depth8_epsilon, depth12_x, external_v500 gamma sweep

Gamma grid:
    union(0:0.1:1, 0.2:0.05:0.6)
which is
    0, .1, .2, .25, .3, .35, .4, .45, .5, .55, .6, .7, .8, .9, 1.0

Gamma=0 is physically evaluated only once as the common v800 baseline and is
then copied logically into every curve during summary generation.

The existing repository implementation originally supports exactly three
schedule depths and does not pass arbitrary condition depths into schedule_depth.
Use --install-schedule-patch --patch-only once before the sweep. The patch is
minimal, backed up, syntax-checked, and preserves the original 3-depth behavior.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import itertools
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def detect_repo_root() -> Path:
    here = Path(__file__).resolve()
    for root in (here.parent, here.parent.parent):
        if (root / "experiments/evaluate_imagenet100_sit_multiscale_condition.py").is_file():
            return root
    # Intended installation is <repo>/experiments/<this-file>.
    return here.parent.parent


REPO_ROOT = detect_repo_root()
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EVALUATOR = EXPERIMENTS_DIR / "evaluate_imagenet100_sit_multiscale_condition.py"
GUIDANCE_CORE = EXPERIMENTS_DIR / "imagenet100_sit_multiscale_guidance.py"
SAMPLER = EXPERIMENTS_DIR / "sample_imagenet100_sit_multiscale_guidance.py"

# Union of 0..1 step 0.1 and 0.2..0.6 step 0.05, written explicitly to avoid
# floating-point range surprises.
DEFAULT_GAMMAS = (
    0.00,
    0.10,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)
DEFAULT_SCHEDULE_POOL = (4, 6, 8, 10)
DEFAULT_SCHEDULE_LENGTHS = (2, 3, 4)
STATIC_DEPTHS = (4, 6, 8, 10, 12)

DATA_ROOT_CANDIDATES = (
    Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow"),
    Path("/data/users/zhoushunyu/eqvae/imagenet_sit_flow"),
)


def _real_data_roots() -> tuple[Path, ...]:
    return DATA_ROOT_CANDIDATES


def pick_existing(relative: str) -> Path:
    for root in _real_data_roots():
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow") / relative


def detect_output_data_root() -> Path:
    strong_rel = "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
    for root in _real_data_roots():
        if (root / strong_rel).is_file():
            return root
    return Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")


DATA_ROOT = detect_output_data_root()
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "internal_head_gamma_schedule_sweep_v4"
DEFAULT_STRONG = pick_existing("runs/sit-s-2_seed0/checkpoints/step_00800000.pt")
# Existing evaluator requires this path even when no condition uses external_v500.
DEFAULT_EXTERNAL_WEAK = pick_existing("runs/sit-s-2_seed0/checkpoints/step_00500000.pt")
DEFAULT_ATLAS = pick_existing("multiscale_guidance_study_v1/atlas/atlas_summary.json")
DEFAULT_REFERENCE = pick_existing(
    "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
)
DEFAULT_ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")

DEFAULT_HEADS = {
    "depth4_v": pick_existing(
        "multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt"
    ),
    "depth6_v": pick_existing(
        "multiscale_guidance_study_v1/runs/depth6_v/checkpoints/step_00050000.pt"
    ),
    "depth8_v": pick_existing(
        "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
        "checkpoints/step_00050000.pt"
    ),
    "depth10_v": pick_existing(
        "multiscale_guidance_study_v1/runs/depth10_v/checkpoints/step_00050000.pt"
    ),
    "depth12_v": pick_existing(
        "multiscale_guidance_study_v1/runs/depth12_v/checkpoints/step_00050000.pt"
    ),
}

EXCLUDED_HEADS = {"depth8_x", "depth8_epsilon", "depth12_x"}
PATCH_MARKER_CORE = "gamma_schedule_sweep_v4_generalized_schedule_depth"
PATCH_MARKER_SAMPLER = "gamma_schedule_sweep_v4_condition_depths"


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    required_heads: tuple[str, ...]
    condition_extra: dict[str, Any]
    realized_depth_order: tuple[int, ...] = ()
    rms_matched: bool = False


@dataclass(frozen=True)
class Job:
    name: str
    experiment: str
    gamma: float
    condition_path: Path
    output_dir: Path
    required_heads: tuple[str, ...]


def gamma_tag(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".") or "0"
    return text.replace("-", "m").replace(".", "p")


def parse_gpu_list(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU list must be comma-separated integers") from exc
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("GPU list must contain unique non-negative IDs")
    return gpus


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result) or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return result


def parse_head(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--head must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--head must use non-empty NAME=PATH")
    return name, Path(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def valid_result(path: Path, expected_samples: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        metrics = result["metrics"]
        manifest = result["sampling_manifest"]
        if not isinstance(metrics, dict) or not isinstance(manifest, dict):
            return False
        if not all(
            isinstance(metrics.get(key), (int, float))
            for key in ("fid", "sfid", "inception_score")
        ):
            return False
        if int(manifest["sampling"]["num_samples"]) != expected_samples:
            return False
        return bool(manifest.get("noise_sha256") and manifest.get("label_sha256"))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _backup_once(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".pre_gamma_schedule_sweep_v4.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    return backup


def _compile_python_files(paths: Iterable[Path]) -> None:
    subprocess.run(
        [sys.executable, "-m", "py_compile", *[str(path) for path in paths]],
        cwd=REPO_ROOT,
        check=True,
    )


def _function_block(text: str, function_name: str, next_function_name: str) -> str:
    pattern = re.compile(
        rf"def {re.escape(function_name)}\(.*?(?=\ndef {re.escape(next_function_name)}\()",
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    return "" if match is None else match.group(0)


def _depth_schedule_branch(text: str) -> str:
    # Restrict detection to the actual ConditionField depth_schedule branch.
    # The unpatched sampler already contains unrelated `depths=` occurrences in
    # raw_compute_schedule, so a file-wide substring check would be unsafe.
    pattern = re.compile(
        r'elif kind == "depth_schedule":(?P<body>.*?)(?=\n\s*else:)',
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    return "" if match is None else match.group("body")


def schedule_patch_status() -> dict[str, bool]:
    if not GUIDANCE_CORE.is_file() or not SAMPLER.is_file():
        return {"core": False, "sampler": False}
    core = GUIDANCE_CORE.read_text(encoding="utf-8")
    sampler = SAMPLER.read_text(encoding="utf-8")

    core_block = _function_block(core, "schedule_depth", "finite_number")
    core_ok = bool(core_block) and (
        PATCH_MARKER_CORE in core_block
        or ("depths: Sequence[int]" in core_block and "torch.bucketize" in core_block)
    )

    schedule_branch = _depth_schedule_branch(sampler)
    sampler_ok = bool(schedule_branch) and (
        PATCH_MARKER_SAMPLER in schedule_branch
        or (
            "selected = schedule_depth(" in schedule_branch
            and "depths=" in schedule_branch
            and 'self.condition.get("depths", (4, 8, 10))' in schedule_branch
        )
    )
    return {"core": core_ok, "sampler": sampler_ok}


def install_schedule_patch() -> None:
    for path in (GUIDANCE_CORE, SAMPLER):
        if not path.is_file():
            raise FileNotFoundError(path)

    status = schedule_patch_status()
    if status["core"] and status["sampler"]:
        print("schedule patch already installed", flush=True)
        _compile_python_files((GUIDANCE_CORE, SAMPLER))
        return

    # --- Patch schedule_depth in the pure guidance core. ---
    if not status["core"]:
        _backup_once(GUIDANCE_CORE)
        text = GUIDANCE_CORE.read_text(encoding="utf-8")
        pattern = re.compile(
            r"def schedule_depth\(\n.*?\n\s*return values\[indices\]\n",
            flags=re.DOTALL,
        )
        match = pattern.search(text)
        if match is None:
            raise RuntimeError(
                "could not locate current schedule_depth() implementation; "
                "refusing to patch an unknown source layout"
            )
        replacement = f'''def schedule_depth(\n    time_value: torch.Tensor,\n    *,\n    order: str,\n    depths: Sequence[int] = (4, 8, 10),\n) -> torch.Tensor:\n    # {PATCH_MARKER_CORE}\n    selected = tuple(int(depth) for depth in depths)\n    if len(selected) < 2 or any(depth < 1 for depth in selected):\n        raise ValueError("at least two positive schedule depths are required")\n    if len(set(selected)) != len(selected):\n        raise ValueError("schedule depths must be unique")\n    if order == "coarse_to_fine":\n        pass\n    elif order == "fine_to_coarse":\n        selected = tuple(reversed(selected))\n    else:\n        raise ValueError(f"unsupported depth order: {{order!r}}")\n\n    # Preserve the original implementation exactly for the historical 3-stage\n    # schedule, including boundary/tie semantics.\n    if len(selected) == 3:\n        weights = time_partition_weights(time_value)\n        matrix = torch.stack([weights[name] for name in TIME_NAMES], dim=1)\n        indices = matrix.argmax(dim=1)\n    else:\n        # General N-stage schedule: equal time partitions. At an exact boundary\n        # enter the next stage, matching the effective float32 behavior of the\n        # historical 3-stage implementation.\n        boundaries = torch.arange(1, len(selected), device=time_value.device, dtype=torch.float32)\n        boundaries = boundaries / float(len(selected))\n        indices = torch.bucketize(time_value.float(), boundaries, right=True)\n    values = torch.as_tensor(selected, device=time_value.device)\n    return values[indices]\n'''
        text = text[: match.start()] + replacement + text[match.end() :]
        GUIDANCE_CORE.write_text(text, encoding="utf-8")

    # --- Patch sampler so condition["depths"] is actually passed through. ---
    status = schedule_patch_status()
    if not status["sampler"]:
        _backup_once(SAMPLER)
        text = SAMPLER.read_text(encoding="utf-8")
        old = '''                selected = schedule_depth(\n                    times,\n                    order=str(self.condition["order"]),\n                )'''
        if old not in text:
            raise RuntimeError(
                "could not locate sampler depth_schedule call; refusing to patch unknown source"
            )
        new = f'''                selected = schedule_depth(\n                    times,\n                    order=str(self.condition["order"]),\n                    # {PATCH_MARKER_SAMPLER}\n                    depths=tuple(\n                        int(value)\n                        for value in self.condition.get("depths", (4, 8, 10))\n                    ),\n                )'''
        text = text.replace(old, new, 1)
        SAMPLER.write_text(text, encoding="utf-8")

    _compile_python_files((GUIDANCE_CORE, SAMPLER))
    final = schedule_patch_status()
    if not all(final.values()):
        raise RuntimeError(f"schedule patch validation failed: {final}")
    print(
        json.dumps(
            {
                "event": "schedule_patch_installed",
                "core": str(GUIDANCE_CORE),
                "sampler": str(SAMPLER),
                "core_backup": str(
                    GUIDANCE_CORE.with_suffix(
                        GUIDANCE_CORE.suffix + ".pre_gamma_schedule_sweep_v4.bak"
                    )
                ),
                "sampler_backup": str(
                    SAMPLER.with_suffix(
                        SAMPLER.suffix + ".pre_gamma_schedule_sweep_v4.bak"
                    )
                ),
            },
            indent=2,
        ),
        flush=True,
    )


def verify_schedule_runtime_support() -> None:
    status = schedule_patch_status()
    if not all(status.values()):
        raise RuntimeError(
            "repository does not yet support arbitrary-length depth schedules. Run:\n"
            f"  {sys.executable} {Path(__file__).name} --install-schedule-patch --patch-only"
        )

    code = r'''
import torch
from experiments.imagenet100_sit_multiscale_guidance import schedule_depth

def values(depths, times, order="coarse_to_fine"):
    t = torch.tensor(times, dtype=torch.float32)
    return schedule_depth(t, order=order, depths=depths).tolist()

assert values((4, 8), [0.0, 0.49, 0.5, 0.51, 1.0]) == [4, 4, 8, 8, 8]
# This is the original 3-stage implementation, preserved exactly. In float32,
# querying Python's 1/3 and 2/3 lands just into the next stage.
assert values((4, 6, 8), [0.0, 0.32, 1/3, 0.34, 0.66, 2/3, 0.67, 1.0]) == [4, 4, 6, 6, 6, 8, 8, 8]
assert values((4, 6, 8, 10), [0.0, 0.25, 0.2501, 0.5, 0.5001, 0.75, 0.7501, 1.0]) == [4, 6, 6, 8, 8, 10, 10, 10]
assert values((4, 8), [0.0, 0.49, 0.5, 0.51, 1.0], "fine_to_coarse") == [8, 8, 4, 4, 4]
'''
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def build_experiments(
    schedule_pool: tuple[int, ...], schedule_lengths: tuple[int, ...]
) -> tuple[Experiment, ...]:
    if tuple(sorted(schedule_pool)) != schedule_pool:
        raise ValueError("--schedule-pool must be strictly increasing")
    if any(length < 2 or length > len(schedule_pool) for length in schedule_lengths):
        raise ValueError("schedule lengths must lie in [2, len(schedule_pool)]")

    experiments: list[Experiment] = []

    # Static curves use the native gap only. They answer each head's own optimal gamma.
    for depth in STATIC_DEPTHS:
        head = f"depth{depth}_v"
        experiments.append(
            Experiment(
                name=head,
                kind="static",
                required_heads=(head,),
                condition_extra={"kind": "full_gap", "provider": head},
                realized_depth_order=(depth,),
                rms_matched=False,
            )
        )

    # All requested depth combinations, forward/reverse, native/RMS.
    for length in sorted(set(schedule_lengths)):
        for combo in itertools.combinations(schedule_pool, length):
            required = tuple(f"depth{depth}_v" for depth in combo)
            for rms_matched in (False, True):
                amp = "rms" if rms_matched else "native"
                for reverse in (False, True):
                    realized = tuple(reversed(combo)) if reverse else tuple(combo)
                    order = "fine_to_coarse" if reverse else "coarse_to_fine"
                    path_tag = "_".join(str(depth) for depth in realized)
                    experiments.append(
                        Experiment(
                            name=f"schedule_{path_tag}_{amp}",
                            kind="schedule",
                            required_heads=required,
                            condition_extra={
                                "kind": "depth_schedule",
                                "order": order,
                                # Always store the canonical ascending tuple; order controls reversal.
                                "depths": list(combo),
                                "rms_matched": rms_matched,
                            },
                            realized_depth_order=realized,
                            rms_matched=rms_matched,
                        )
                    )

    names = [experiment.name for experiment in experiments]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate experiment names")
    return tuple(experiments)


def condition_payload(name: str, gamma: float, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "eqvae_imagenet100_sit_multiscale_condition_v1",
        "name": name,
        "evaluation_group": "internal_head_gamma_schedule_sweep_v4",
        "hypothesis_id": "internal_head_gamma_schedule_sweep_v4",
        "gamma": float(gamma),
        **extra,
    }


def build_jobs(
    args: argparse.Namespace, experiments: tuple[Experiment, ...]
) -> tuple[Job, list[Job]]:
    conditions_dir = args.output_root / "conditions"
    evaluations_dir = args.output_root / "evaluations"
    conditions_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    baseline_name = "gamma_schedule_sweep_v4_baseline_v800"
    baseline_path = conditions_dir / f"{baseline_name}.json"
    atomic_json(baseline_path, condition_payload(baseline_name, 0.0, {"kind": "baseline"}))
    baseline = Job(
        name=baseline_name,
        experiment="baseline",
        gamma=0.0,
        condition_path=baseline_path,
        output_dir=evaluations_dir / baseline_name,
        required_heads=(),
    )

    jobs: list[Job] = []
    # Interleave gamma first so partial runs already compare the whole experiment family.
    for gamma in args.gammas:
        if abs(gamma) < 1e-12:
            continue
        for experiment in experiments:
            name = f"gamma_schedule_sweep_v4_{experiment.name}_g{gamma_tag(gamma)}"
            path = conditions_dir / f"{name}.json"
            atomic_json(path, condition_payload(name, gamma, experiment.condition_extra))
            jobs.append(
                Job(
                    name=name,
                    experiment=experiment.name,
                    gamma=gamma,
                    condition_path=path,
                    output_dir=evaluations_dir / name,
                    required_heads=experiment.required_heads,
                )
            )
    return baseline, jobs


def evaluator_command(args: argparse.Namespace, job: Job) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--condition-json",
        str(job.condition_path),
        "--atlas-summary",
        str(args.atlas_summary),
        "--output-dir",
        str(job.output_dir),
        "--strong-checkpoint",
        str(args.strong_checkpoint),
        "--external-weak-checkpoint",
        str(args.external_weak_checkpoint),
        "--reference",
        str(args.reference),
        "--adm-python",
        str(args.adm_python),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.sample_batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--seed",
        str(args.seed),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--fid-batch-size",
        str(args.fid_batch_size),
        "--fid-gpu-memory-fraction",
        str(args.fid_gpu_memory_fraction),
        "--device",
        "cuda:0",
    ]
    for head in job.required_heads:
        command.extend(("--head", f"{head}={args.heads[head]}"))
    return command


def run_job(args: argparse.Namespace, job: Job, physical_gpu: int) -> dict[str, Any]:
    result_path = job.output_dir / "condition_result.json"
    if valid_result(result_path, args.num_samples):
        return {
            "status": "complete",
            "reused": True,
            "gpu": physical_gpu,
            "result": str(result_path),
        }

    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "logs" / f"{job.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = evaluator_command(args, job)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        "/home/zhoushunyu/data/eqvae/torchinductor_cache",
    )
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        code = process.wait()
    if code != 0:
        return {
            "status": "failed",
            "reused": False,
            "gpu": physical_gpu,
            "return_code": code,
            "log": str(log_path),
            "elapsed_seconds": time.time() - started,
        }
    if not valid_result(result_path, args.num_samples):
        return {
            "status": "failed",
            "reused": False,
            "gpu": physical_gpu,
            "return_code": 0,
            "error": "evaluator exited 0 but condition_result.json is invalid",
            "log": str(log_path),
            "elapsed_seconds": time.time() - started,
        }
    return {
        "status": "complete",
        "reused": False,
        "gpu": physical_gpu,
        "result": str(result_path),
        "elapsed_seconds": time.time() - started,
    }


def flatten(job: Job, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    manifest = result["sampling_manifest"]
    sampling = manifest["sampling"]
    return {
        "experiment": job.experiment,
        "gamma": job.gamma,
        "fid": metrics["fid"],
        "sfid": metrics["sfid"],
        "inception_score": metrics["inception_score"],
        "num_samples": sampling["num_samples"],
        "integrator": sampling["integrator"],
        "noise_sha256": manifest["noise_sha256"],
        "label_sha256": manifest["label_sha256"],
        "total_nfe": manifest["total_nfe"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "condition_name": job.name,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    args: argparse.Namespace,
    baseline: Job,
    jobs: list[Job],
    experiments: tuple[Experiment, ...],
) -> None:
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    baseline_result_path = baseline.output_dir / "condition_result.json"
    if not valid_result(baseline_result_path, args.num_samples):
        raise RuntimeError("baseline is incomplete; cannot summarize")
    baseline_result = read_json(baseline_result_path)
    baseline_row = flatten(baseline, baseline_result)
    baseline_fid = float(baseline_row["fid"])

    rows: list[dict[str, Any]] = []
    experiment_names = [experiment.name for experiment in experiments]
    for experiment_name in experiment_names:
        row = dict(baseline_row)
        row["experiment"] = experiment_name
        row["condition_name"] = baseline.name
        row["fid_improvement_vs_baseline"] = 0.0
        rows.append(row)

    failures: list[dict[str, Any]] = []
    for job in jobs:
        path = job.output_dir / "condition_result.json"
        if not valid_result(path, args.num_samples):
            failures.append(
                {
                    "experiment": job.experiment,
                    "gamma": job.gamma,
                    "condition_name": job.name,
                    "log": str(args.output_root / "logs" / f"{job.name}.log"),
                }
            )
            continue
        row = flatten(job, read_json(path))
        row["fid_improvement_vs_baseline"] = baseline_fid - float(row["fid"])
        rows.append(row)

    # Strict pairing audit: every successful real run must use identical noise/labels.
    actual_pairs = {
        (str(baseline_row["noise_sha256"]), str(baseline_row["label_sha256"]))
    }
    for row in rows:
        actual_pairs.add((str(row["noise_sha256"]), str(row["label_sha256"])))
    if len(actual_pairs) != 1:
        raise RuntimeError(f"noise/label pairing audit failed: {actual_pairs}")

    rows.sort(key=lambda row: (str(row["experiment"]), float(row["gamma"])))
    metrics_path = summary_dir / "gamma_sweep_metrics.csv"
    write_csv(metrics_path, rows)

    best_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        curve = [row for row in rows if row["experiment"] == experiment.name]
        best = min(curve, key=lambda row: float(row["fid"]))
        best_rows.append(
            {
                "experiment": experiment.name,
                "kind": experiment.kind,
                "depth_order": "->".join(str(value) for value in experiment.realized_depth_order),
                "rms_matched": experiment.rms_matched,
                "best_gamma": best["gamma"],
                "best_fid": best["fid"],
                "fid_improvement_vs_baseline": best["fid_improvement_vs_baseline"],
                "sfid": best["sfid"],
                "inception_score": best["inception_score"],
                "condition_name": best["condition_name"],
            }
        )
    best_rows.sort(key=lambda row: float(row["best_fid"]))
    best_path = summary_dir / "best_by_experiment.csv"
    write_csv(best_path, best_rows)
    best_schedules_path = summary_dir / "best_schedules.csv"
    write_csv(
        best_schedules_path,
        [row for row in best_rows if row["kind"] == "schedule"],
    )

    # Wide FID table for plotting curves.
    all_gammas = sorted({float(row["gamma"]) for row in rows})
    by_key = {(str(row["experiment"]), float(row["gamma"])): row for row in rows}
    pivot_path = summary_dir / "fid_by_gamma.csv"
    with pivot_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gamma", *experiment_names])
        writer.writeheader()
        for gamma in all_gammas:
            record: dict[str, Any] = {"gamma": gamma}
            for experiment_name in experiment_names:
                row = by_key.get((experiment_name, gamma))
                record[experiment_name] = "" if row is None else row["fid"]
            writer.writerow(record)

    failures_path = summary_dir / "failures.json"
    atomic_json(failures_path, failures)
    atomic_json(
        summary_dir / "summary.json",
        {
            "format": "eqvae_imagenet100_sit_internal_head_gamma_schedule_sweep_summary_v4",
            "baseline_fid": baseline_fid,
            "baseline_condition": baseline.name,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "gammas": list(args.gammas),
            "experiment_count": len(experiments),
            "static_count": sum(experiment.kind == "static" for experiment in experiments),
            "schedule_count": sum(experiment.kind == "schedule" for experiment in experiments),
            "completed_curve_rows": len(rows),
            "failed_conditions": len(failures),
            "paired_noise_sha256": baseline_row["noise_sha256"],
            "paired_label_sha256": baseline_row["label_sha256"],
            "files": {
                "metrics": str(metrics_path),
                "best": str(best_path),
                "best_schedules": str(best_schedules_path),
                "fid_pivot": str(pivot_path),
                "failures": str(failures_path),
            },
        },
    )


def validate_inputs(args: argparse.Namespace, experiments: tuple[Experiment, ...]) -> None:
    required_files = [
        EVALUATOR,
        GUIDANCE_CORE,
        SAMPLER,
        args.strong_checkpoint,
        args.external_weak_checkpoint,
        args.atlas_summary,
        args.reference,
        args.adm_python,
    ]
    required_head_names = sorted(
        {head for experiment in experiments for head in experiment.required_heads}
    )
    missing_head_keys = sorted(set(required_head_names) - set(args.heads))
    if missing_head_keys:
        raise ValueError(f"missing head mappings: {missing_head_keys}")
    required_files.extend(args.heads[name] for name in required_head_names)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n  " + "\n  ".join(missing))

    forbidden = sorted(EXCLUDED_HEADS & set(args.heads))
    if forbidden:
        raise ValueError(f"excluded non-velocity heads are present: {forbidden}")

    atlas = read_json(args.atlas_summary)
    calibration = atlas.get("rms_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("atlas lacks rms_calibration")
    schedule_head_names = {
        head
        for experiment in experiments
        if experiment.kind == "schedule" and experiment.rms_matched
        for head in experiment.required_heads
    }
    missing_calibration = sorted(schedule_head_names - set(calibration))
    if missing_calibration:
        raise ValueError(
            "atlas lacks RMS calibration for schedule heads: " + ", ".join(missing_calibration)
        )


def build_protocol(args: argparse.Namespace, experiments: tuple[Experiment, ...]) -> dict[str, Any]:
    return {
        "format": "eqvae_imagenet100_sit_internal_head_gamma_schedule_sweep_protocol_v4",
        "strong_checkpoint": str(args.strong_checkpoint),
        "atlas_summary": str(args.atlas_summary),
        "reference": str(args.reference),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "gpus": list(args.gpus),
        "gammas": list(args.gammas),
        "gamma_definition": "union(0:0.1:1, 0.2:0.05:0.6)",
        "heads": {name: str(path) for name, path in sorted(args.heads.items())},
        "schedule_pool": list(args.schedule_pool),
        "schedule_lengths": list(args.schedule_lengths),
        "excluded": [
            "depth8_x",
            "depth8_epsilon",
            "depth12_x",
            "external_v500 gamma sweep",
        ],
        "rms_definition": (
            "time-dependent scale_to_depth8_v from the existing atlas; "
            "RMS-matched schedules normalize each selected depth gap to depth8_v RMS"
        ),
        "experiments": [
            {
                "name": experiment.name,
                "kind": experiment.kind,
                "required_heads": list(experiment.required_heads),
                "realized_depth_order": list(experiment.realized_depth_order),
                "rms_matched": experiment.rms_matched,
                "condition": experiment.condition_extra,
            }
            for experiment in experiments
        ],
        "pairing_rule": (
            "same seed + same batch size; summary hard-fails unless all successful runs "
            "share one noise/label fingerprint"
        ),
    }


def run(args: argparse.Namespace) -> None:
    if args.patch_only and not args.install_schedule_patch:
        raise ValueError("--patch-only requires --install-schedule-patch")
    if args.install_schedule_patch:
        install_schedule_patch()
        verify_schedule_runtime_support()
        if args.patch_only:
            return

    args.output_root = args.output_root.expanduser().resolve()
    args.strong_checkpoint = args.strong_checkpoint.expanduser().resolve()
    args.external_weak_checkpoint = args.external_weak_checkpoint.expanduser().resolve()
    args.atlas_summary = args.atlas_summary.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.adm_python = args.adm_python.expanduser().absolute()

    # Velocity-only defaults. Overrides are allowed only for these names.
    heads = {name: path.expanduser().resolve() for name, path in DEFAULT_HEADS.items()}
    for name, path in args.head:
        if name in EXCLUDED_HEADS:
            raise ValueError(f"{name} is intentionally excluded from this sweep")
        if name not in DEFAULT_HEADS:
            raise ValueError(
                f"unknown head {name!r}; allowed velocity heads: {sorted(DEFAULT_HEADS)}"
            )
        heads[name] = path.expanduser().resolve()
    args.heads = heads

    experiments = build_experiments(args.schedule_pool, args.schedule_lengths)
    validate_inputs(args, experiments)
    if any(len(experiment.realized_depth_order) != 3 for experiment in experiments if experiment.kind == "schedule"):
        verify_schedule_runtime_support()

    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / "pipeline.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another sweep process owns {lock_path}") from exc

    protocol = build_protocol(args, experiments)
    protocol_path = args.output_root / "sweep_protocol.json"
    if protocol_path.is_file() and read_json(protocol_path) != protocol:
        raise ValueError(
            f"existing protocol differs: {protocol_path}; use a new --output-root"
        )
    atomic_json(protocol_path, protocol)

    baseline, jobs = build_jobs(args, experiments)
    static_count = sum(experiment.kind == "static" for experiment in experiments)
    schedule_count = sum(experiment.kind == "schedule" for experiment in experiments)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "static_curves": static_count,
                "schedule_curves": schedule_count,
                "total_experiment_curves": len(experiments),
                "baseline_runs": 1,
                "guided_conditions": len(jobs),
                "total_real_evaluations": len(jobs) + 1,
                "gammas": list(args.gammas),
                "gpus": list(args.gpus),
                "schedule_pool": list(args.schedule_pool),
                "schedule_lengths": list(args.schedule_lengths),
            },
            indent=2,
        ),
        flush=True,
    )
    print("\nExperiment curves:")
    for experiment in experiments:
        print(
            f"  {experiment.name:34s}  order={experiment.realized_depth_order} "
            f"rms={experiment.rms_matched}",
            flush=True,
        )

    if args.dry_run:
        print("\n[baseline command]")
        print(shlex.join(evaluator_command(args, baseline)))
        print("\n[first guided command]")
        print(shlex.join(evaluator_command(args, jobs[0])))
        print("\n[last guided command]")
        print(shlex.join(evaluator_command(args, jobs[-1])))
        return

    state_path = args.output_root / "pipeline_state.json"
    state_lock = threading.Lock()
    stop = threading.Event()
    state: dict[str, Any] = {
        "format": "eqvae_imagenet100_sit_internal_head_gamma_schedule_sweep_state_v4",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "jobs": {},
    }
    if state_path.is_file():
        old = read_json(state_path)
        if old.get("format") == state["format"]:
            state = old
            state["status"] = "running"
    atomic_json(state_path, state)

    # Common baseline first: catches broken shared setup before consuming both GPUs.
    print(f"[baseline] GPU {args.gpus[0]}: {baseline.name}", flush=True)
    baseline_status = run_job(args, baseline, args.gpus[0])
    state["jobs"][baseline.name] = baseline_status
    atomic_json(state_path, state)
    if baseline_status["status"] != "complete":
        state["status"] = "failed"
        atomic_json(state_path, state)
        raise RuntimeError(f"baseline failed: {baseline_status}")

    work: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        work.put(job)

    any_failed = threading.Event()

    def worker(gpu: int) -> None:
        while not stop.is_set():
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            try:
                print(
                    f"[GPU {gpu}] {job.experiment} gamma={job.gamma:g} ({job.name})",
                    flush=True,
                )
                status = run_job(args, job, gpu)
                with state_lock:
                    state["jobs"][job.name] = status
                    atomic_json(state_path, state)
                if status["status"] != "complete":
                    any_failed.set()
                    print(
                        f"[FAILED GPU {gpu}] {job.name}; see {status.get('log', '')}",
                        flush=True,
                    )
                    if args.fail_fast:
                        stop.set()
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False, name=f"gpu-{gpu}")
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    write_summary(args, baseline, jobs, experiments)
    state["status"] = "complete_with_failures" if any_failed.is_set() else "complete"
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "summary": str(args.output_root / "summary/summary.json"),
                "best": str(args.output_root / "summary/best_by_experiment.csv"),
                "best_schedules": str(args.output_root / "summary/best_schedules.csv"),
                "fid_pivot": str(args.output_root / "summary/fid_by_gamma.csv"),
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpus", type=parse_gpu_list, default=parse_gpu_list("2,3"))
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--external-weak-checkpoint", type=Path, default=DEFAULT_EXTERNAL_WEAK)
    parser.add_argument("--atlas-summary", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=DEFAULT_ADM_PYTHON)
    parser.add_argument(
        "--head",
        action="append",
        type=parse_head,
        default=[],
        metavar="NAME=PATH",
        help="override one velocity-head checkpoint; x/epsilon heads are rejected",
    )
    parser.add_argument(
        "--schedule-pool",
        type=parse_int_list,
        default=DEFAULT_SCHEDULE_POOL,
        help="depth pool used to enumerate schedules; default: 4,6,8,10",
    )
    parser.add_argument(
        "--schedule-lengths",
        type=parse_int_list,
        default=DEFAULT_SCHEDULE_LENGTHS,
        help="schedule lengths to enumerate; default: 2,3,4",
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=8.0)
    parser.add_argument("--fid-batch-size", type=int, default=8)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--install-schedule-patch", action="store_true")
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    args.gammas = DEFAULT_GAMMAS
    return args


if __name__ == "__main__":
    run(build_parser())