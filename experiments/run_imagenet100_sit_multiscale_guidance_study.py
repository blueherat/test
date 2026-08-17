#!/usr/bin/env python3
"""Run the resumable ImageNet-100 multiscale-guidance mechanism study."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch

try:
    from experiments.evaluate_imagenet100_sit_multiscale_condition import valid_result
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        atomic_json_dump,
        git_value,
    )
    from experiments.train_imagenet100_sit_frozen_internal_v_head import PROTOCOL
except ModuleNotFoundError:
    from evaluate_imagenet100_sit_multiscale_condition import valid_result
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR,
        DEFAULT_OFFICIAL_SIT_REPO,
        atomic_json_dump,
        git_value,
    )
    from train_imagenet100_sit_frozen_internal_v_head import PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "multiscale_guidance_study_v1"
DEFAULT_SMOKE_ROOT = DEFAULT_DATA_ROOT / "multiscale_guidance_study_smoke_v1"
DEFAULT_V800 = DEFAULT_DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_V500 = DEFAULT_DATA_ROOT / "runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
DEFAULT_DEPTH8_V = (
    DEFAULT_DATA_ROOT
    / "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH8_X = (
    DEFAULT_DATA_ROOT
    / "runs/sit-s-2_v800-ema_frozen-internal-x-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH8_EPS = (
    DEFAULT_DATA_ROOT
    / "runs/sit-s-2_v800-ema_frozen-internal-eps-depth8_seed0/"
    "checkpoints/step_00050000.pt"
)
DEFAULT_DEPTH12_X = (
    DEFAULT_DATA_ROOT
    / "runs/sit-s-2_v800-ema_frozen-final-x-fullhead-depth12_seed0/"
    "checkpoints/step_00050000.pt"
)
REFERENCE = (
    DEFAULT_DATA_ROOT
    / "adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
)
ADM_PYTHON = Path("/data/shared/envs/adm-fid/bin/python")


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    artifact: Path
    validator: Callable[[Path], bool]
    environment: dict[str, str]


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    payload: dict[str, object]
    num_samples: int


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def valid_head_checkpoint(
    path: Path,
    *,
    expected_step: int,
    expected_depth: int,
    source_checkpoint: Path,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        config = payload["config"]
        return bool(
            payload.get("protocol") == PROTOCOL
            and int(payload.get("step", -1)) == expected_step
            and int(config.get("internal_depth", -1)) == expected_depth
            and config.get("prediction_target") == "velocity"
            and config.get("source_state_key") == "ema"
            and Path(config.get("source_checkpoint", "")).resolve()
            == source_checkpoint.resolve()
            and "internal_head_ema" in payload
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def valid_atlas(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        expected = {
            "depth4_v",
            "depth6_v",
            "depth8_v",
            "depth10_v",
            "depth12_v",
            "depth8_x",
            "depth8_epsilon",
            "depth12_x",
        }
        rms = set(payload["rms_calibration"])
        actions = set(payload["action_calibration"])
        return bool(
            payload.get("format") == "eqvae_imagenet100_sit_multiscale_atlas_v1"
            and expected.issubset(set(payload["heads"]))
            and expected.issubset(rms)
            and {"depth8_v", "depth12_x", "external_v500"}.issubset(actions)
            and set(payload["delay_fit"]["fitted_lag_time"]) == {"low", "mid", "high"}
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def condition_result_validator(expected_samples: int) -> Callable[[Path], bool]:
    return lambda path: valid_result(path, expected_samples=expected_samples)


def stage_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        "/home/zhoushunyu/data/eqvae/torchinductor_cache",
    )
    return environment


def parse_gpu_list(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU list must contain integers") from error
    if not gpus or any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("GPU list must contain unique non-negative IDs")
    return gpus


def add_condition(
    conditions: list[ConditionSpec],
    *,
    group: str,
    name: str,
    hypothesis_id: str,
    num_samples: int,
    **payload: object,
) -> None:
    full_name = f"{group}_{name}"
    conditions.append(
        ConditionSpec(
            name=full_name,
            num_samples=num_samples,
            payload={
                "format": "eqvae_imagenet100_sit_multiscale_condition_v1",
                "name": full_name,
                "evaluation_group": group,
                "hypothesis_id": hypothesis_id,
                **payload,
            },
        )
    )


def build_smoke_conditions(num_samples: int, euler_steps: int) -> list[ConditionSpec]:
    conditions: list[ConditionSpec] = []
    add_condition(
        conditions,
        group="smoke",
        name="baseline_adaptive",
        hypothesis_id="plumbing",
        num_samples=num_samples,
        kind="baseline",
    )
    add_condition(
        conditions,
        group="smoke",
        name="causal_depth8_early_low_equal",
        hypothesis_id="causal_map",
        num_samples=num_samples,
        kind="band_time",
        provider="depth8_v",
        gamma=0.4,
        interval="early",
        band="low",
        amplitude="equal_action",
    )
    for order in ("coarse_to_fine", "fine_to_coarse"):
        add_condition(
            conditions,
            group="smoke",
            name=f"depth_schedule_{order}",
            hypothesis_id="idea1_time_varying_depth",
            num_samples=num_samples,
            kind="depth_schedule",
            gamma=0.4,
            order=order,
            depths=[4, 8, 10],
            rms_matched=True,
        )
    for reverse in (False, True):
        add_condition(
            conditions,
            group="smoke",
            name="spectral_router" if not reverse else "spectral_anti_router",
            hypothesis_id="idea2_spectral_routing",
            num_samples=num_samples,
            kind="spectral_router",
            gamma=0.4,
            reverse=reverse,
            rms_matched=True,
        )
    add_condition(
        conditions,
        group="smoke",
        name="euler_baseline",
        hypothesis_id="idea3_spectral_delay",
        num_samples=num_samples,
        kind="euler_baseline",
        steps=euler_steps,
    )
    add_condition(
        conditions,
        group="smoke",
        name="spectral_delay",
        hypothesis_id="idea3_spectral_delay",
        num_samples=num_samples,
        kind="spectral_delay",
        steps=euler_steps,
        gamma=0.4,
        rms_matched=True,
    )
    for order in ("coarse_to_fine", "fine_to_coarse"):
        add_condition(
            conditions,
            group="smoke",
            name=f"raw_compute_{order}",
            hypothesis_id="idea4_unresolved_computation",
            num_samples=num_samples,
            kind="raw_compute_schedule",
            gamma=0.4,
            order=order,
            depths=[4, 8, 10],
            rms_matched=True,
        )
    return conditions


def build_full_conditions(
    *, screen_samples: int, confirm_samples: int, euler_steps: int
) -> list[ConditionSpec]:
    if screen_samples < 1:
        raise ValueError("screen_samples must be positive")
    if confirm_samples < 0:
        raise ValueError("confirm_samples cannot be negative")
    conditions: list[ConditionSpec] = []
    provider_gamma = {"depth8_v": 0.4, "depth12_x": 0.08, "external_v500": 3.0}
    add_condition(
        conditions,
        group="screen",
        name="baseline_adaptive",
        hypothesis_id="baseline",
        num_samples=screen_samples,
        kind="baseline",
    )
    for provider, gamma in provider_gamma.items():
        add_condition(
            conditions,
            group="screen",
            name=f"full_{provider}",
            hypothesis_id="successful_failed_control",
            num_samples=screen_samples,
            kind="full_gap",
            provider=provider,
            gamma=gamma,
        )
        for amplitude in ("native", "equal_action"):
            for interval in ("early", "mid", "late"):
                for band in ("low", "mid", "high"):
                    add_condition(
                        conditions,
                        group="screen",
                        name=f"map_{provider}_{amplitude}_{interval}_{band}",
                        hypothesis_id="causal_map",
                        num_samples=screen_samples,
                        kind="band_time",
                        provider=provider,
                        gamma=gamma,
                        interval=interval,
                        band=band,
                        amplitude=amplitude,
                    )
            for order in ("coarse_to_fine", "fine_to_coarse"):
                add_condition(
                    conditions,
                    group="screen",
                    name=f"order_{provider}_{amplitude}_{order}",
                    hypothesis_id="causal_order",
                    num_samples=screen_samples,
                    kind="ordered_bands",
                    provider=provider,
                    gamma=gamma,
                    order=order,
                    amplitude=amplitude,
                )

    for depth in (4, 6, 8, 10, 12):
        add_condition(
            conditions,
            group="screen",
            name=f"static_depth{depth}",
            hypothesis_id="idea1_time_varying_depth",
            num_samples=screen_samples,
            kind="static_depth",
            gamma=0.4,
            depth=depth,
            rms_matched=False,
        )
    for rms_matched in (False, True):
        suffix = "rms" if rms_matched else "native"
        for order in ("coarse_to_fine", "fine_to_coarse"):
            add_condition(
                conditions,
                group="screen",
                name=f"depth_schedule_{suffix}_{order}",
                hypothesis_id="idea1_time_varying_depth",
                num_samples=screen_samples,
                kind="depth_schedule",
                gamma=0.4,
                order=order,
                depths=[4, 8, 10],
                rms_matched=rms_matched,
            )
        for reverse in (False, True):
            add_condition(
                conditions,
                group="screen",
                name=("spectral_router" if not reverse else "spectral_anti_router")
                + f"_{suffix}",
                hypothesis_id="idea2_spectral_routing",
                num_samples=screen_samples,
                kind="spectral_router",
                gamma=0.4,
                reverse=reverse,
                rms_matched=rms_matched,
            )

    add_condition(
        conditions,
        group="screen",
        name="euler_baseline",
        hypothesis_id="idea3_spectral_delay",
        num_samples=screen_samples,
        kind="euler_baseline",
        steps=euler_steps,
    )
    add_condition(
        conditions,
        group="screen",
        name="euler_depth8_gamma0p4",
        hypothesis_id="idea3_spectral_delay",
        num_samples=screen_samples,
        kind="euler_depth8",
        steps=euler_steps,
        gamma=0.4,
    )
    for rms_matched in (False, True):
        suffix = "rms" if rms_matched else "native"
        for gamma in (0.1, 0.2, 0.4, 0.8):
            gamma_name = str(gamma).replace(".", "p")
            add_condition(
                conditions,
                group="screen",
                name=f"spectral_delay_{suffix}_gamma{gamma_name}",
                hypothesis_id="idea3_spectral_delay",
                num_samples=screen_samples,
                kind="spectral_delay",
                steps=euler_steps,
                gamma=gamma,
                rms_matched=rms_matched,
            )

    raw_definitions = (
        ("static_h8", "coarse_to_fine", [8, 8, 8]),
        ("coarse_to_fine", "coarse_to_fine", [4, 8, 10]),
        ("fine_to_coarse", "fine_to_coarse", [4, 8, 10]),
    )
    for rms_matched in (False, True):
        suffix = "rms" if rms_matched else "native"
        for label, order, depths in raw_definitions:
            add_condition(
                conditions,
                group="screen",
                name=f"raw_compute_{suffix}_{label}",
                hypothesis_id="idea4_unresolved_computation",
                num_samples=screen_samples,
                kind="raw_compute_schedule",
                gamma=0.4,
                order=order,
                depths=depths,
                rms_matched=rms_matched,
            )

    confirmations = (
        ("baseline_adaptive", "baseline", {"kind": "baseline"}),
        (
            "full_depth8_v",
            "successful_failed_control",
            {"kind": "full_gap", "provider": "depth8_v", "gamma": 0.4},
        ),
        (
            "full_external_v500",
            "successful_failed_control",
            {"kind": "full_gap", "provider": "external_v500", "gamma": 3.0},
        ),
        (
            "depth_schedule_rms_coarse_to_fine",
            "idea1_time_varying_depth",
            {
                "kind": "depth_schedule",
                "gamma": 0.4,
                "order": "coarse_to_fine",
                "depths": [4, 8, 10],
                "rms_matched": True,
            },
        ),
        (
            "depth_schedule_rms_fine_to_coarse",
            "idea1_time_varying_depth",
            {
                "kind": "depth_schedule",
                "gamma": 0.4,
                "order": "fine_to_coarse",
                "depths": [4, 8, 10],
                "rms_matched": True,
            },
        ),
        (
            "spectral_router_rms",
            "idea2_spectral_routing",
            {"kind": "spectral_router", "gamma": 0.4, "reverse": False, "rms_matched": True},
        ),
        (
            "spectral_anti_router_rms",
            "idea2_spectral_routing",
            {"kind": "spectral_router", "gamma": 0.4, "reverse": True, "rms_matched": True},
        ),
        (
            "euler_baseline",
            "idea3_spectral_delay",
            {"kind": "euler_baseline", "steps": euler_steps},
        ),
        (
            "euler_depth8_gamma0p4",
            "idea3_spectral_delay",
            {"kind": "euler_depth8", "steps": euler_steps, "gamma": 0.4},
        ),
        (
            "spectral_delay_rms_gamma0p4",
            "idea3_spectral_delay",
            {
                "kind": "spectral_delay",
                "steps": euler_steps,
                "gamma": 0.4,
                "rms_matched": True,
            },
        ),
        (
            "raw_compute_rms_coarse_to_fine",
            "idea4_unresolved_computation",
            {
                "kind": "raw_compute_schedule",
                "gamma": 0.4,
                "order": "coarse_to_fine",
                "depths": [4, 8, 10],
                "rms_matched": True,
            },
        ),
        (
            "raw_compute_rms_fine_to_coarse",
            "idea4_unresolved_computation",
            {
                "kind": "raw_compute_schedule",
                "gamma": 0.4,
                "order": "fine_to_coarse",
                "depths": [4, 8, 10],
                "rms_matched": True,
            },
        ),
    )
    if confirm_samples:
        for name, hypothesis_id, payload in confirmations:
            add_condition(
                conditions,
                group="confirm",
                name=name,
                hypothesis_id=hypothesis_id,
                num_samples=confirm_samples,
                **payload,
            )
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate condition name")
    return conditions


def write_protocol(
    *, args: argparse.Namespace, conditions: list[ConditionSpec], head_paths: dict[str, Path]
) -> None:
    payload = {
        "format": "eqvae_imagenet100_sit_multiscale_study_protocol_v1",
        "profile": args.profile,
        "git_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "gpu": args.gpu,
        "evaluation_gpus": list(args.evaluation_gpus or (args.gpu,)),
        "strong_checkpoint": str(args.strong_checkpoint),
        "external_weak_checkpoint": str(args.external_weak_checkpoint),
        "heads": {name: str(path) for name, path in head_paths.items()},
        "conceptual_corrections": [
            "latent FFT bands describe SD-VAE latent spatial scale, not decoded pixel texture",
            "causal cells are finite closed-loop interventions and do not add linearly",
            "native-amplitude and equal-action maps are both required",
            "routing and remaining-computation rules are fixed before FID evaluation",
            "raw remaining-computation is a proxy confounded by final-layer representation mismatch",
            "FID-1K is a screen; named method candidates use paired FID-5K confirmation",
        ],
        "conditions": [
            {"num_samples": condition.num_samples, **condition.payload}
            for condition in conditions
        ],
    }
    path = args.output_root / "study_protocol.json"
    if path.is_file() and read_json(path) != payload:
        raise ValueError(
            f"existing protocol differs from this run: {path}; use a new output root"
        )
    atomic_json_dump(payload, path)
    condition_dir = args.output_root / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        atomic_json_dump(condition.payload, condition_dir / f"{condition.name}.json")


def build_stages(
    args: argparse.Namespace,
) -> tuple[list[Stage], list[ConditionSpec], dict[str, Path]]:
    smoke = args.profile == "smoke"
    head_steps = 2 if smoke else 50_000
    head_batch = 16 if smoke else 256
    output = args.output_root
    environment = stage_environment(args.gpu)
    python = sys.executable
    checkpoint_name = f"step_{head_steps:08d}.pt"
    head_paths: dict[str, Path] = {"depth8_v": args.depth8_v_checkpoint}
    stages: list[Stage] = []

    for depth in (4, 6, 10, 12):
        run_dir = output / "runs" / f"depth{depth}_v"
        checkpoint = run_dir / "checkpoints" / checkpoint_name
        head_paths[f"depth{depth}_v"] = checkpoint
        command = (
            python,
            str(REPO_ROOT / "experiments/train_imagenet100_sit_frozen_internal_v_head.py"),
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            str(run_dir),
            "--official-sit-repo",
            str(args.official_sit_repo),
            "--source-checkpoint",
            str(args.strong_checkpoint),
            "--source-state-key",
            "ema",
            "--internal-depth",
            str(depth),
            "--prediction-target",
            "velocity",
            "--global-batch-size",
            str(head_batch),
            "--max-steps",
            str(head_steps),
            "--learning-rate",
            "1e-4",
            "--precision",
            "bf16",
            "--num-workers",
            "0" if smoke else "4",
            "--prefetch-factor",
            "2" if smoke else "4",
            "--log-every",
            "1" if smoke else "100",
            "--validation-every",
            str(head_steps if smoke else 10_000),
            "--validation-batches",
            "1" if smoke else "8",
            "--save-every",
            str(head_steps),
            "--seed",
            "0",
            "--device",
            "cuda:0",
            "--resume",
            "auto",
            "--no-compile" if smoke else "--compile",
        )
        stages.append(
            Stage(
                name=f"train_depth{depth}_v",
                command=command,
                artifact=checkpoint,
                validator=lambda path, step=head_steps, depth=depth: valid_head_checkpoint(
                    path,
                    expected_step=step,
                    expected_depth=depth,
                    source_checkpoint=args.strong_checkpoint,
                ),
                environment=environment,
            )
        )

    head_paths.update(
        {
            "depth8_x": args.depth8_x_checkpoint,
            "depth8_epsilon": args.depth8_epsilon_checkpoint,
            "depth12_x": args.depth12_x_checkpoint,
        }
    )
    atlas_dir = output / "atlas"
    atlas_path = atlas_dir / "atlas_summary.json"
    atlas_command = [
        python,
        str(REPO_ROOT / "experiments/analyze_imagenet100_sit_multiscale_guidance.py"),
        "--output-dir",
        str(atlas_dir),
        "--official-sit-repo",
        str(args.official_sit_repo),
        "--strong-checkpoint",
        str(args.strong_checkpoint),
        "--external-weak-checkpoint",
        str(args.external_weak_checkpoint),
        "--depth8-v-checkpoint",
        str(args.depth8_v_checkpoint),
        "--depth8-x-checkpoint",
        str(args.depth8_x_checkpoint),
        "--depth8-epsilon-checkpoint",
        str(args.depth8_epsilon_checkpoint),
        "--depth12-x-checkpoint",
        str(args.depth12_x_checkpoint),
        "--samples",
        "2" if smoke else "64",
        "--time-points",
        "3" if smoke else "49",
        "--max-delay-steps",
        "2" if smoke else "8",
        "--device",
        "cuda:0",
    ]
    for name in ("depth4_v", "depth6_v", "depth10_v", "depth12_v"):
        atlas_command.extend(("--head", f"{name}={head_paths[name]}"))
    stages.append(
        Stage(
            name="build_latent_atlas",
            command=tuple(atlas_command),
            artifact=atlas_path,
            validator=valid_atlas,
            environment=environment,
        )
    )

    if smoke:
        conditions = build_smoke_conditions(num_samples=32, euler_steps=8)
    else:
        conditions = build_full_conditions(
            screen_samples=args.screen_samples,
            confirm_samples=args.confirm_samples,
            euler_steps=args.euler_steps,
        )
    write_protocol(args=args, conditions=conditions, head_paths=head_paths)
    condition_dir = output / "conditions"
    head_arguments: list[str] = []
    for name, path in sorted(head_paths.items()):
        head_arguments.extend(("--head", f"{name}={path}"))
    for condition in conditions:
        condition_output = output / "evaluations" / condition.name
        result_path = condition_output / "condition_result.json"
        command = (
            python,
            str(REPO_ROOT / "experiments/evaluate_imagenet100_sit_multiscale_condition.py"),
            "--condition-json",
            str(condition_dir / f"{condition.name}.json"),
            "--atlas-summary",
            str(atlas_path),
            "--output-dir",
            str(condition_output),
            "--strong-checkpoint",
            str(args.strong_checkpoint),
            "--external-weak-checkpoint",
            str(args.external_weak_checkpoint),
            *head_arguments,
            "--reference",
            str(args.reference),
            "--adm-python",
            str(args.adm_python),
            "--num-samples",
            str(condition.num_samples),
            "--batch-size",
            "4" if smoke else str(args.sample_batch_size),
            "--vae-decode-batch-size",
            "2",
            "--seed",
            "0",
            "--cuda-allocator-limit-gib",
            str(args.cuda_allocator_limit_gib),
            "--fid-batch-size",
            "8",
            "--fid-gpu-memory-fraction",
            str(args.fid_gpu_memory_fraction),
            "--device",
            "cuda:0",
        )
        stages.append(
            Stage(
                name=f"evaluate_{condition.name}",
                command=command,
                artifact=result_path,
                validator=condition_result_validator(condition.num_samples),
                environment=environment,
            )
        )
    return stages, conditions, head_paths


def run_stage(stage: Stage, log_path: Path, *, mirror_output: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(stage.command)
    print(json.dumps({"event": "stage_start", "stage": stage.name}), flush=True)
    print(command_text, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {command_text}\n")
        log.flush()
        process = subprocess.Popen(
            stage.command,
            cwd=REPO_ROOT,
            env=stage.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if mirror_output:
                print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, stage.command)
    if not stage.validator(stage.artifact):
        raise RuntimeError(
            f"stage {stage.name} exited successfully but artifact is invalid: {stage.artifact}"
        )


def flatten_result(result: dict[str, object]) -> dict[str, object]:
    condition = result["condition"]
    metrics = result["metrics"]
    manifest = result["sampling_manifest"]
    if not isinstance(condition, dict) or not isinstance(metrics, dict) or not isinstance(manifest, dict):
        raise ValueError("malformed condition result")
    sampling = manifest["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("malformed sampling metadata")
    return {
        "name": condition["name"],
        "group": condition["evaluation_group"],
        "hypothesis_id": condition["hypothesis_id"],
        "kind": condition["kind"],
        "provider": condition.get("provider", ""),
        "amplitude": condition.get("amplitude", ""),
        "interval": condition.get("interval", ""),
        "band": condition.get("band", ""),
        "order": condition.get("order", ""),
        "gamma": condition.get("gamma", 0.0),
        "rms_matched": condition.get("rms_matched", ""),
        "reverse": condition.get("reverse", ""),
        "depth": condition.get("depth", ""),
        "depths": json.dumps(condition.get("depths", [])),
        "num_samples": sampling["num_samples"],
        "integrator": sampling["integrator"],
        "fid": metrics["fid"],
        "sfid": metrics["sfid"],
        "inception_score": metrics["inception_score"],
        "noise_sha256": manifest["noise_sha256"],
        "label_sha256": manifest["label_sha256"],
        "total_nfe": manifest["total_nfe"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }


def write_summary(args: argparse.Namespace, conditions: list[ConditionSpec]) -> None:
    rows: list[dict[str, object]] = []
    for condition in conditions:
        result_path = args.output_root / "evaluations" / condition.name / "condition_result.json"
        if not valid_result(result_path, expected_samples=condition.num_samples):
            raise RuntimeError(f"cannot summarize incomplete result: {result_path}")
        rows.append(flatten_result(read_json(result_path)))
    fingerprints: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for row in rows:
        key = (str(row["group"]), int(row["num_samples"]))
        fingerprints.setdefault(key, set()).add(
            (str(row["noise_sha256"]), str(row["label_sha256"]))
        )
    mismatched = {str(key): values for key, values in fingerprints.items() if len(values) != 1}
    if mismatched:
        raise RuntimeError(f"paired random-input audit failed: {mismatched}")

    baselines: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["kind"] == "baseline":
            baselines[(str(row["group"]), "dopri5")] = float(row["fid"])
        if row["kind"] == "euler_baseline":
            baselines[(str(row["group"]), "fixed_euler")] = float(row["fid"])
    for row in rows:
        key = (str(row["group"]), str(row["integrator"]))
        row["fid_improvement_vs_matched_baseline"] = (
            baselines[key] - float(row["fid"]) if key in baselines else ""
        )
    rows.sort(key=lambda row: (str(row["group"]), float(row["fid"])))
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "condition_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "format": "eqvae_imagenet100_sit_multiscale_study_summary_v1",
        "profile": args.profile,
        "condition_count": len(rows),
        "paired_fingerprint_groups": {
            str(key): list(next(iter(values))) for key, values in fingerprints.items()
        },
        "rows": rows,
        "files": {"csv": str(csv_path)},
    }
    atomic_json_dump(payload, summary_dir / "study_summary.json")


def update_state(path: Path, payload: dict[str, object]) -> None:
    atomic_json_dump(payload, path)


def execute_stage(
    stage: Stage,
    *,
    state: dict[str, object],
    state_path: Path,
    output_root: Path,
    state_lock: threading.Lock | None = None,
    mirror_output: bool = True,
) -> None:
    lock = state_lock if state_lock is not None else nullcontext()
    if stage.validator(stage.artifact):
        with lock:
            stage_states = state.setdefault("stages", {})
            assert isinstance(stage_states, dict)
            stage_states[stage.name] = {
                "status": "complete",
                "reused": True,
                "artifact": str(stage.artifact),
            }
            update_state(state_path, state)
        print(json.dumps({"event": "stage_reuse", "stage": stage.name}), flush=True)
        return

    with lock:
        stage_states = state.setdefault("stages", {})
        assert isinstance(stage_states, dict)
        stage_states[stage.name] = {
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "artifact": str(stage.artifact),
            "command": list(stage.command),
            "physical_gpu": stage.environment.get("CUDA_VISIBLE_DEVICES"),
        }
        update_state(state_path, state)
    try:
        run_stage(
            stage,
            output_root / "logs" / f"{stage.name}.log",
            mirror_output=mirror_output,
        )
    except Exception as error:
        with lock:
            stage_states = state.setdefault("stages", {})
            assert isinstance(stage_states, dict)
            current = stage_states.get(stage.name, {})
            assert isinstance(current, dict)
            stage_states[stage.name] = {
                **current,
                "status": "failed",
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "error": repr(error),
            }
            update_state(state_path, state)
        raise
    with lock:
        stage_states = state.setdefault("stages", {})
        assert isinstance(stage_states, dict)
        current = stage_states.get(stage.name, {})
        assert isinstance(current, dict)
        stage_states[stage.name] = {
            **current,
            "status": "complete",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        update_state(state_path, state)
    print(json.dumps({"event": "stage_complete", "stage": stage.name}), flush=True)


def run_parallel_evaluations(
    stages: list[Stage],
    *,
    physical_gpus: tuple[int, ...],
    state: dict[str, object],
    state_path: Path,
    output_root: Path,
) -> None:
    if not stages:
        return
    if len(physical_gpus) == 1:
        for stage in stages:
            execute_stage(
                replace(stage, environment=stage_environment(physical_gpus[0])),
                state=state,
                state_path=state_path,
                output_root=output_root,
            )
        return

    state_lock = threading.Lock()
    stop = threading.Event()
    shards = [stages[index:: len(physical_gpus)] for index in range(len(physical_gpus))]

    def worker(gpu: int, shard: list[Stage]) -> None:
        for original in shard:
            if stop.is_set():
                return
            stage = replace(original, environment=stage_environment(gpu))
            try:
                execute_stage(
                    stage,
                    state=state,
                    state_path=state_path,
                    output_root=output_root,
                    state_lock=state_lock,
                    mirror_output=False,
                )
            except Exception:
                stop.set()
                raise

    with ThreadPoolExecutor(max_workers=len(physical_gpus)) as executor:
        futures = [
            executor.submit(worker, gpu, shard)
            for gpu, shard in zip(physical_gpus, shards)
        ]
        for future in futures:
            future.result()


def run(args: argparse.Namespace) -> None:
    for name in (
        "output_root",
        "cache_dir",
        "official_sit_repo",
        "strong_checkpoint",
        "external_weak_checkpoint",
        "depth8_v_checkpoint",
        "depth8_x_checkpoint",
        "depth8_epsilon_checkpoint",
        "depth12_x_checkpoint",
        "reference",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.adm_python = args.adm_python.expanduser().absolute()
    required = (
        args.cache_dir / "manifest.json",
        args.strong_checkpoint,
        args.external_weak_checkpoint,
        args.depth8_v_checkpoint,
        args.depth8_x_checkpoint,
        args.depth8_epsilon_checkpoint,
        args.depth12_x_checkpoint,
        args.reference,
        args.adm_python,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n  " + "\n  ".join(missing))
    if not (args.official_sit_repo / "models.py").is_file():
        raise FileNotFoundError(args.official_sit_repo / "models.py")
    if shutil.disk_usage(args.output_root.parent).free < args.minimum_free_gib * 2**30:
        raise RuntimeError("insufficient free disk for temporary sample arrays")
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / "pipeline.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another process owns {lock_path}") from error

    stages, conditions, _ = build_stages(args)
    if args.dry_run:
        for stage in stages:
            print(f"{stage.name}: {shlex.join(stage.command)}")
        return
    state_path = args.output_root / "pipeline_state.json"
    state: dict[str, object] = {
        "format": "eqvae_imagenet100_sit_multiscale_pipeline_state_v1",
        "profile": args.profile,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stages": {},
    }
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("profile") != args.profile:
            raise ValueError("existing state uses another profile")
    state["status"] = "running"
    update_state(state_path, state)
    prerequisite_stages = [
        stage for stage in stages if not stage.name.startswith("evaluate_")
    ]
    evaluation_stages = [
        stage for stage in stages if stage.name.startswith("evaluate_")
    ]
    try:
        for stage in prerequisite_stages:
            execute_stage(
                stage,
                state=state,
                state_path=state_path,
                output_root=args.output_root,
            )
        run_parallel_evaluations(
            evaluation_stages,
            physical_gpus=args.evaluation_gpus or (args.gpu,),
            state=state,
            state_path=state_path,
            output_root=args.output_root,
        )
    except Exception:
        state["status"] = "failed"
        state["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        update_state(state_path, state)
        raise
    write_summary(args, conditions)
    state["status"] = "complete"
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    update_state(state_path, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument(
        "--evaluation-gpus",
        type=parse_gpu_list,
        default=None,
        help="comma-separated physical GPUs for parallel evaluation workers",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--strong-checkpoint", type=Path, default=DEFAULT_V800)
    parser.add_argument("--external-weak-checkpoint", type=Path, default=DEFAULT_V500)
    parser.add_argument("--depth8-v-checkpoint", type=Path, default=DEFAULT_DEPTH8_V)
    parser.add_argument("--depth8-x-checkpoint", type=Path, default=DEFAULT_DEPTH8_X)
    parser.add_argument("--depth8-epsilon-checkpoint", type=Path, default=DEFAULT_DEPTH8_EPS)
    parser.add_argument("--depth12-x-checkpoint", type=Path, default=DEFAULT_DEPTH12_X)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--adm-python", type=Path, default=ADM_PYTHON)
    parser.add_argument("--screen-samples", type=int, default=1_000)
    parser.add_argument("--confirm-samples", type=int, default=5_000)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--euler-steps", type=int, default=100)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=8.0)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = DEFAULT_SMOKE_ROOT if args.profile == "smoke" else DEFAULT_OUTPUT_ROOT
    return args


if __name__ == "__main__":
    run(build_parser())
