#!/usr/bin/env python3
"""Re-run the post-3901741 SiT experiments on official SiT-S/2 weights.

The pipeline is resumable and waits for the official download to pass both a
byte-size and SHA256 check. Every subprocess has its own log and a stage is
reused only when its structured completion artifact validates.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

try:
    from experiments import train_imagenet100_sit_flow as base
    from experiments.official_imagenet100_sit_s2 import (
        DEFAULT_RAW_CHECKPOINT,
        DEFAULT_SUBSET_CHECKPOINT,
        HF_FILENAME,
        HF_REPOSITORY,
        HF_REVISION,
        RAW_CHECKPOINT_BYTES,
        RAW_CHECKPOINT_SHA256,
        SUBSET_CHECKPOINT_FORMAT,
    )
except ModuleNotFoundError:
    import train_imagenet100_sit_flow as base
    from official_imagenet100_sit_s2 import (
        DEFAULT_RAW_CHECKPOINT,
        DEFAULT_SUBSET_CHECKPOINT,
        HF_FILENAME,
        HF_REPOSITORY,
        HF_REVISION,
        RAW_CHECKPOINT_BYTES,
        RAW_CHECKPOINT_SHA256,
        SUBSET_CHECKPOINT_FORMAT,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
START_COMMIT = "3901741617bdae33e31c9dca04c89b3d7bec8696"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "official_sit_s2_recent_replication_v1"
)
REFERENCE = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/"
    "imagenet100_validation_n5000_adm_stats.npz"
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    artifact: Path
    validator: Callable[[Path], bool]
    environment: dict[str, str]


def parse_gpu_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("GPU list must contain unique comma-separated indices")
    return values


def valid_json(path: Path, *, rows: int | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if rows is not None:
        values = payload.get("rows")
        if not isinstance(values, list) or len(values) != rows:
            return False
        fingerprints = {
            (row.get("noise_fingerprint"), row.get("label_fingerprint"))
            for row in values
        }
        if len(fingerprints) != 1:
            return False
    return True


def valid_prepared_checkpoint(path: Path) -> bool:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not valid_json(metadata_path):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("format") == SUBSET_CHECKPOINT_FORMAT
        and metadata.get("raw_sha256") == RAW_CHECKPOINT_SHA256
        and metadata.get("output_sha256") == base.sha256_file(path)
        and metadata.get("equivalence_audit", {}).get("passed") is True
    )


def checkpoint_validator(
    expected_step: int,
    *,
    protocol: str,
    prediction_target: str | None = None,
    internal_depth: int | None = None,
    source_checkpoint: Path,
) -> Callable[[Path], bool]:
    def validate(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (OSError, RuntimeError, ValueError):
            return False
        config = payload.get("config", {})
        if not isinstance(config, dict):
            return False
        valid = (
            int(payload.get("step", -1)) == expected_step
            and payload.get("protocol") == protocol
            and Path(str(config.get("source_checkpoint", ""))).resolve()
            == source_checkpoint.resolve()
        )
        if prediction_target is not None:
            valid = valid and config.get("prediction_target") == prediction_target
        if internal_depth is not None:
            valid = valid and int(config.get("internal_depth", -1)) == internal_depth
        return valid

    return validate


def json_rows_validator(expected_rows: int) -> Callable[[Path], bool]:
    return lambda path: valid_json(path, rows=expected_rows)


def environment_for_training(gpus: list[int]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpus)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        "/home/zhoushunyu/data/eqvae/torchinductor_cache",
    )
    return environment


def common_train_arguments(
    *,
    source_checkpoint: Path,
    output_dir: Path,
    max_steps: int,
    global_batch_size: int,
    validation_every: int,
    validation_batches: int,
    save_every: int,
    compile_model: bool,
) -> list[str]:
    arguments = [
        "--source-checkpoint",
        str(source_checkpoint),
        "--source-state-key",
        "ema",
        "--output-dir",
        str(output_dir),
        "--global-batch-size",
        str(global_batch_size),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        "1e-4",
        "--seed",
        "0",
        "--precision",
        "bf16",
        "--allow-tf32",
        "--num-workers",
        "4" if max_steps > 2 else "0",
        "--prefetch-factor",
        "4",
        "--log-every",
        "50" if max_steps > 2 else "1",
        "--validation-every",
        str(validation_every),
        "--validation-batches",
        str(validation_batches),
        "--save-every",
        str(save_every),
        "--resume",
        "auto",
    ]
    arguments.append("--compile" if compile_model else "--no-compile")
    if compile_model:
        arguments.extend(("--compile-mode", "default"))
    return arguments


def fid_arguments(
    *,
    checkpoint: Path,
    output_root: Path,
    num_samples: int,
    gpus: list[int],
    gpu_memory_ceiling_mib: int,
) -> list[str]:
    visible = ",".join(str(index) for index in gpus)
    return [
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(output_root),
        "--reference",
        str(REFERENCE),
        "--num-samples",
        str(num_samples),
        "--per-rank-batch-size",
        "8" if num_samples >= 1_000 else "2",
        "--vae-decode-batch-size",
        "2",
        "--cuda-allocator-limit-gib",
        "6",
        "--sampling-cuda-visible-devices",
        visible,
        "--fid-cuda-visible-devices",
        str(gpus[0]),
        "--fid-batch-size",
        "8",
        "--fid-gpu-memory-fraction",
        "0.25",
        "--gpu-memory-ceiling-mib",
        str(gpu_memory_ceiling_mib),
        "--global-seed",
        "0",
    ]


def build_stages(args: argparse.Namespace) -> list[Stage]:
    smoke = args.profile == "smoke"
    max_steps = 2 if smoke else 50_000
    num_samples = 64 if smoke else 1_000
    global_batch = 16 if smoke else 256
    validation_every = 2 if smoke else 5_000
    validation_batches = 1 if smoke else 8
    save_every = max_steps if smoke else 10_000
    compile_model = not smoke
    nproc = len(args.gpu_indices)
    torchrun = shutil.which("torchrun") or "torchrun"
    python = sys.executable
    output = args.output_root
    train_environment = environment_for_training(args.gpu_indices)
    plain_environment = os.environ.copy()

    tiny_run = output / "runs/frozen-final-linear-x"
    depth8_v_run = output / "runs/frozen-depth8-v"
    depth8_x_run = output / "runs/frozen-depth8-x"
    depth8_eps_run = output / "runs/frozen-depth8-epsilon"
    depth12_x_run = output / "runs/frozen-depth12-x-full"
    train_checkpoint_name = f"step_{max_steps:08d}.pt"
    tiny_checkpoint = tiny_run / "checkpoints" / train_checkpoint_name
    depth8_v_checkpoint = depth8_v_run / "checkpoints" / train_checkpoint_name
    depth8_x_checkpoint = depth8_x_run / "checkpoints" / train_checkpoint_name
    depth8_eps_checkpoint = depth8_eps_run / "checkpoints" / train_checkpoint_name
    depth12_x_checkpoint = depth12_x_run / "checkpoints" / train_checkpoint_name

    train_prefix = (torchrun, "--standalone", f"--nproc_per_node={nproc}")
    common = dict(
        source_checkpoint=args.source_checkpoint,
        max_steps=max_steps,
        global_batch_size=global_batch,
        validation_every=validation_every,
        validation_batches=validation_batches,
        save_every=save_every,
        compile_model=compile_model,
    )
    tiny_protocol = "imagenet100_sit_frozen_v_clean_linear_probe_v1"
    internal_protocols = {
        "velocity": "imagenet100_sit_frozen_v_internal_velocity_head_v1",
        "clean": "imagenet100_sit_frozen_v_internal_clean_head_v1",
        "epsilon": "imagenet100_sit_frozen_v_internal_epsilon_head_v1",
    }

    if smoke:
        tiny_gammas = [0.05]
        v_gammas = [0.1]
        target_gammas = [0.1]
        final_gammas = [0.08]
        hidden_gammas = [0.01]
        output_gammas = [0.003]
        hidden_alphas = [0.02]
        output_alphas = [0.01]
    else:
        tiny_gammas = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
        v_gammas = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
        target_gammas = [
            0.01,
            0.03,
            0.05,
            0.08,
            0.1,
            0.12,
            0.15,
            0.18,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.5,
        ]
        final_gammas = [
            0.01,
            0.03,
            0.05,
            0.08,
            0.1,
            0.12,
            0.15,
            0.2,
            0.3,
            0.4,
            0.6,
            0.8,
            1.0,
            1.5,
        ]
        hidden_gammas = [
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.1,
            0.2,
            0.4,
            0.6,
            1.0,
            1.5,
            2.0,
            3.0,
        ]
        output_gammas = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.1, 0.2]
        hidden_alphas = [
            0.005,
            0.01,
            0.0125,
            0.015,
            0.0175,
            0.02,
            0.0225,
            0.025,
            0.03,
            0.05,
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
        ]
        output_alphas = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]

    stages: list[Stage] = [
        Stage(
            name="prepare_official_subset",
            command=(
                python,
                str(REPO_ROOT / "experiments/prepare_official_imagenet100_sit_s2.py"),
                "--raw-checkpoint",
                str(args.raw_checkpoint),
                "--output",
                str(args.source_checkpoint),
                "--device",
                "cuda:0",
            ),
            artifact=args.source_checkpoint,
            validator=valid_prepared_checkpoint,
            environment=train_environment,
        ),
        Stage(
            name="train_final_linear_x",
            command=(
                *train_prefix,
                str(REPO_ROOT / "experiments/train_imagenet100_sit_frozen_v_clean_head.py"),
                *common_train_arguments(output_dir=tiny_run, **common),
            ),
            artifact=tiny_checkpoint,
            validator=checkpoint_validator(
                max_steps,
                protocol=tiny_protocol,
                source_checkpoint=args.source_checkpoint,
            ),
            environment=train_environment,
        ),
        Stage(
            name="fid_final_linear_x",
            command=(
                python,
                str(REPO_ROOT / "experiments/run_imagenet100_sit_frozen_v_clean_head_fid1k.py"),
                *fid_arguments(
                    checkpoint=tiny_checkpoint,
                    output_root=output / "fid/frozen-final-linear-x",
                    num_samples=num_samples,
                    gpus=args.gpu_indices,
                    gpu_memory_ceiling_mib=args.gpu_memory_ceiling_mib,
                ),
                "--gammas",
                *(str(value) for value in tiny_gammas),
            ),
            artifact=(
                output / "fid/frozen-final-linear-x/frozen_v_clean_head_fid1k.json"
            ),
            validator=json_rows_validator(2 + len(tiny_gammas)),
            environment=plain_environment,
        ),
    ]

    internal_specs = (
        ("v", "velocity", depth8_v_run, depth8_v_checkpoint, v_gammas),
        ("x", "clean", depth8_x_run, depth8_x_checkpoint, target_gammas),
        (
            "epsilon",
            "epsilon",
            depth8_eps_run,
            depth8_eps_checkpoint,
            target_gammas,
        ),
    )
    for tag, target, run_dir, checkpoint, gammas in internal_specs:
        stages.append(
            Stage(
                name=f"train_depth8_{tag}",
                command=(
                    *train_prefix,
                    str(
                        REPO_ROOT
                        / "experiments/train_imagenet100_sit_frozen_internal_v_head.py"
                    ),
                    *common_train_arguments(output_dir=run_dir, **common),
                    "--internal-depth",
                    "8",
                    "--prediction-target",
                    target,
                    "--clean-velocity-denominator-floor",
                    "0.05",
                ),
                artifact=checkpoint,
                validator=checkpoint_validator(
                    max_steps,
                    protocol=internal_protocols[target],
                    prediction_target=target,
                    internal_depth=8,
                    source_checkpoint=args.source_checkpoint,
                ),
                environment=train_environment,
            )
        )
        stages.append(
            Stage(
                name=f"fid_depth8_{tag}",
                command=(
                    python,
                    str(
                        REPO_ROOT
                        / "experiments/run_imagenet100_sit_frozen_internal_v_head_fid1k.py"
                    ),
                    *fid_arguments(
                        checkpoint=checkpoint,
                        output_root=output / f"fid/frozen-depth8-{tag}",
                        num_samples=num_samples,
                        gpus=args.gpu_indices,
                        gpu_memory_ceiling_mib=args.gpu_memory_ceiling_mib,
                    ),
                    "--gammas",
                    *(str(value) for value in gammas),
                ),
                artifact=(
                    output
                    / f"fid/frozen-depth8-{tag}/"
                    f"frozen_internal_{target}_head_fid1k.json"
                ),
                validator=json_rows_validator(2 + len(gammas)),
                environment=plain_environment,
            )
        )
        if tag == "v":
            stages.append(
                Stage(
                    name="audit_hidden_gap",
                    command=(
                        python,
                        str(
                            REPO_ROOT
                            / "experiments/analyze_imagenet100_sit_hidden_state_gap.py"
                        ),
                        "--head-checkpoint",
                        str(checkpoint),
                        "--head-weights",
                        "ema",
                        "--output-root",
                        str(output / "audit/hidden-gap"),
                        "--internal-depth",
                        "8",
                        "--samples",
                        "8" if smoke else "32",
                    ),
                    artifact=output / "audit/hidden-gap/hidden_state_gap_audit.json",
                    validator=lambda path: valid_json(path),
                    environment=train_environment,
                )
            )

    hidden_rows = (
        2
        + len(hidden_gammas)
        + len(output_gammas)
        + len(hidden_alphas)
        + len(output_alphas)
    )
    stages.append(
        Stage(
            name="fid_hidden_state_mixing",
            command=(
                python,
                str(
                    REPO_ROOT
                    / "experiments/run_imagenet100_sit_hidden_state_extrapolation_fid1k.py"
                ),
                *fid_arguments(
                    checkpoint=args.source_checkpoint,
                    output_root=output / "fid/hidden-state-mixing",
                    num_samples=num_samples,
                    gpus=args.gpu_indices,
                    gpu_memory_ceiling_mib=args.gpu_memory_ceiling_mib,
                ),
                "--weights",
                "ema",
                "--internal-depth",
                "8",
                "--hidden-gammas",
                *(str(value) for value in hidden_gammas),
                "--output-gammas",
                *(str(value) for value in output_gammas),
                "--hidden-alphas",
                *(str(value) for value in hidden_alphas),
                "--output-alphas",
                *(str(value) for value in output_alphas),
            ),
            artifact=(
                output
                / "fid/hidden-state-mixing/hidden_state_extrapolation_fid1k.json"
            ),
            validator=json_rows_validator(hidden_rows),
            environment=plain_environment,
        )
    )

    stages.extend(
        (
            Stage(
                name="train_depth12_x_full",
                command=(
                    *train_prefix,
                    str(
                        REPO_ROOT
                        / "experiments/train_imagenet100_sit_frozen_internal_v_head.py"
                    ),
                    *common_train_arguments(output_dir=depth12_x_run, **common),
                    "--internal-depth",
                    "12",
                    "--prediction-target",
                    "clean",
                    "--clean-velocity-denominator-floor",
                    "0.05",
                ),
                artifact=depth12_x_checkpoint,
                validator=checkpoint_validator(
                    max_steps,
                    protocol=internal_protocols["clean"],
                    prediction_target="clean",
                    internal_depth=12,
                    source_checkpoint=args.source_checkpoint,
                ),
                environment=train_environment,
            ),
            Stage(
                name="fid_depth12_x_full",
                command=(
                    python,
                    str(
                        REPO_ROOT
                        / "experiments/run_imagenet100_sit_frozen_internal_v_head_fid1k.py"
                    ),
                    *fid_arguments(
                        checkpoint=depth12_x_checkpoint,
                        output_root=output / "fid/frozen-depth12-x-full",
                        num_samples=num_samples,
                        gpus=args.gpu_indices,
                        gpu_memory_ceiling_mib=args.gpu_memory_ceiling_mib,
                    ),
                    "--gammas",
                    *(str(value) for value in final_gammas),
                ),
                artifact=(
                    output
                    / "fid/frozen-depth12-x-full/"
                    "frozen_internal_clean_head_fid1k.json"
                ),
                validator=json_rows_validator(2 + len(final_gammas)),
                environment=plain_environment,
            ),
        )
    )
    return stages


def wait_for_download(path: Path, poll_seconds: float) -> None:
    while True:
        if path.is_file() and path.stat().st_size == RAW_CHECKPOINT_BYTES:
            digest = base.sha256_file(path)
            if digest != RAW_CHECKPOINT_SHA256:
                raise ValueError(
                    f"completed download has unexpected SHA256: {digest}"
                )
            return
        if path.is_file() and path.stat().st_size > RAW_CHECKPOINT_BYTES:
            raise ValueError("downloaded checkpoint is larger than official metadata")
        print(
            json.dumps(
                {
                    "event": "waiting_for_download",
                    "path": str(path),
                    "observed_bytes": path.stat().st_size if path.is_file() else 0,
                    "expected_bytes": RAW_CHECKPOINT_BYTES,
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def gpu_memory_used() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, int] = {}
    for line in output.splitlines():
        index, used = (item.strip() for item in line.split(",", maxsplit=1))
        result[int(index)] = int(used)
    return result


def wait_for_gpus(indices: list[int], ceiling_mib: int, poll_seconds: float) -> None:
    while True:
        used = gpu_memory_used()
        selected = {index: used[index] for index in indices}
        if all(value <= ceiling_mib for value in selected.values()):
            return
        print(
            json.dumps(
                {
                    "event": "waiting_for_gpus",
                    "memory_used_mib": selected,
                    "required_at_most_mib": ceiling_mib,
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def run_stage(stage: Stage, log_path: Path) -> None:
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
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, stage.command)
    if not stage.validator(stage.artifact):
        raise RuntimeError(
            f"stage {stage.name} exited successfully but artifact is invalid: "
            f"{stage.artifact}"
        )


def write_scope(args: argparse.Namespace, stages: list[Stage]) -> None:
    payload = {
        "format": "eqvae_official_sit_s2_recent_replication_scope_v1",
        "start_commit": START_COMMIT,
        "through_commit": base.git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "profile": args.profile,
        "official_checkpoint": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "filename": HF_FILENAME,
            "sha256": RAW_CHECKPOINT_SHA256,
        },
        "included": [stage.name for stage in stages],
        "not_identifiable": [
            {
                "commit": START_COMMIT,
                "experiment": "same-trajectory EMA weight extrapolation",
                "reason": (
                    "the official collection publishes only one final SiT-S/2 "
                    "state dict and no aligned intermediate checkpoint; inventing "
                    "a weak checkpoint would not reproduce the experiment"
                ),
                "substitute_run": False,
            }
        ],
    }
    base.atomic_json_dump(payload, args.output_root / "replication_scope.json")


def update_state(path: Path, payload: dict[str, object]) -> None:
    base.atomic_json_dump(payload, path)


def run(args: argparse.Namespace) -> None:
    args.raw_checkpoint = args.raw_checkpoint.expanduser().resolve()
    args.source_checkpoint = args.source_checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.gpu_indices = parse_gpu_list(args.gpus)
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    if shutil.disk_usage(args.output_root.parent).free < args.minimum_free_gib * 2**30:
        raise RuntimeError("insufficient free disk space for the replication artifacts")
    args.output_root.mkdir(parents=True, exist_ok=True)

    lock_path = args.output_root / "pipeline.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another pipeline owns {lock_path}") from error

    wait_for_download(args.raw_checkpoint, args.poll_seconds)
    stages = build_stages(args)
    write_scope(args, stages)
    if args.dry_run:
        for stage in stages:
            print(f"{stage.name}: {shlex.join(stage.command)}")
        return

    state_path = args.output_root / "pipeline_state.json"
    state: dict[str, object] = {
        "format": "eqvae_official_sit_s2_recent_replication_state_v1",
        "profile": args.profile,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stages": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("profile") != args.profile:
            raise ValueError("existing pipeline state uses a different profile")
        state = previous

    selected = False if args.start_stage else True
    for stage in stages:
        if not selected and stage.name == args.start_stage:
            selected = True
        if not selected:
            continue
        stage_states = state.setdefault("stages", {})
        assert isinstance(stage_states, dict)
        if stage.validator(stage.artifact):
            stage_states[stage.name] = {
                "status": "complete",
                "reused": True,
                "artifact": str(stage.artifact),
            }
            update_state(state_path, state)
            print(
                json.dumps({"event": "stage_reuse", "stage": stage.name}),
                flush=True,
            )
        else:
            wait_for_gpus(
                args.gpu_indices,
                args.max_existing_gpu_memory_mib,
                args.poll_seconds,
            )
            stage_states[stage.name] = {
                "status": "running",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "artifact": str(stage.artifact),
                "command": list(stage.command),
            }
            update_state(state_path, state)
            try:
                run_stage(stage, args.output_root / "logs" / f"{stage.name}.log")
            except Exception as error:
                stage_states[stage.name] = {
                    **stage_states[stage.name],
                    "status": "failed",
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "error": repr(error),
                }
                update_state(state_path, state)
                raise
            stage_states[stage.name] = {
                **stage_states[stage.name],
                "status": "complete",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            update_state(state_path, state)
        if args.stop_after == stage.name:
            break
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["status"] = "complete"
    update_state(state_path, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW_CHECKPOINT)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SUBSET_CHECKPOINT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-existing-gpu-memory-mib", type=int, default=2_048)
    parser.add_argument("--gpu-memory-ceiling-mib", type=int, default=22 * 1_024)
    parser.add_argument("--minimum-free-gib", type=float, default=40.0)
    parser.add_argument("--start-stage")
    parser.add_argument("--stop-after")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
