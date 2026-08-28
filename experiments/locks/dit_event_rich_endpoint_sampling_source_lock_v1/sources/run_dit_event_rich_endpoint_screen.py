#!/usr/bin/env python3
"""Four-GPU endpoint-only launcher for the event-rich DiT discovery screen.

The frozen axis is the 84-class roster crossed with global seeds 1000..1011
(1,008 endpoint PNGs).  Tasks are ordered seed-major, class-roster-minor and
split into four contiguous 252-pair shards.  Scientific randomness belongs to
the pair key rather than a shard or batch, so the assignment is operational
only and cannot affect a pair's RNG stream.

Real sampling is impossible until a final source lock binds the forthcoming
event-rich protocol v3.  This launcher reads no labels, reviews, metrics,
features, embeddings, or selection results.  It saves no trajectory arrays.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


def find_repo_root(source: Path) -> Path:
    for candidate in source.resolve().parents:
        if (candidate / ".git").exists() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate repository root from {source}")


ROOT = find_repo_root(Path(__file__))
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
DEFAULT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_event_rich_endpoint_sampling_source_lock_v1"
)
DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT / "cross_scale_evidence/dit_event_rich_endpoint_screen_v1"
)
DEFAULT_DIT_ROOT = DATA_ROOT / "baselines/DiT"
CHECKPOINT_FILENAME = "DiT-XL-2-256x256.pt"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
DEFAULT_VAE = (
    Path.home()
    / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
    / VAE_REVISION
)
RNG_DOMAIN = "eqvae.dit.event-rich.endpoint.v1"
WORKER_COUNT = 4
EXPECTED_SEEDS = tuple(range(1000, 1012))
EXPECTED_CLASS_COUNT = 84
EXPECTED_PAIR_COUNT = 1008
PAIRS_PER_WORKER = 252
SOURCE_BASENAMES = (
    "sample_dit_imagenet256_endpoint_pairs.py",
    "run_dit_event_rich_endpoint_screen.py",
    "freeze_dit_event_rich_endpoint_sampling_sources.py",
    "reproduce_dit_imagenet256.py",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def exclusive_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {path}")
    return path.resolve()


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"source-lock artifact must not be a symlink: {path}")
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_exact_source_lock_tree(root: Path) -> None:
    expected_files = {
        "sampling_protocol.json",
        "event_protocol.json",
        "manifest.json",
        "completion.json",
        *(f"sources/{name}" for name in SOURCE_BASENAMES),
    }
    expected_dirs = {"sources"}
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"source lock contains a symlink: {path}")
        if path.is_file():
            files.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise RuntimeError(f"source lock contains a special path: {path}")
    if files != expected_files or directories != expected_dirs:
        raise RuntimeError(
            "source-lock member set changed: "
            f"missing_files={sorted(expected_files-files)}, extra_files={sorted(files-expected_files)}, "
            f"missing_dirs={sorted(expected_dirs-directories)}, extra_dirs={sorted(directories-expected_dirs)}"
        )


def roster_from_event_protocol(protocol: Mapping[str, Any]) -> tuple[int, ...]:
    endpoint = protocol.get("endpoint_screen", {})
    rows = endpoint.get("class_roster")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CLASS_COUNT:
        raise RuntimeError("event protocol must freeze exactly 84 roster rows")
    classes: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("class_id")) is not int:
            raise RuntimeError("event protocol class roster row is malformed")
        class_id = row["class_id"]
        if not 0 <= class_id < 1000:
            raise RuntimeError("event protocol class ID lies outside [0,999]")
        classes.append(class_id)
    if len(set(classes)) != EXPECTED_CLASS_COUNT:
        raise RuntimeError("event protocol class roster has duplicate IDs")
    return tuple(classes)


def require_pair_rng_contract(event_protocol: Mapping[str, Any]) -> None:
    endpoint = event_protocol.get("endpoint_screen", {})
    rng = endpoint.get("batch_rng_contract", {})
    expected = {
        "rng_unit": ["global_seed", "class_id"],
        "domain": RNG_DOMAIN,
        "manual_seed_timing": (
            "call torch.manual_seed(pair_seed) after the frozen DiT model and VAE are "
            "fully loaded, immediately before drawing the singleton initial latent"
        ),
        "classes_per_invocation": 1,
        "same_global_seed_classes_share_initial_or_transition_innovation": False,
        "batch_order_shard_and_resume_invariant": True,
        "pair_seed_range": "unsigned 63-bit integer [0,2^63-1]",
        "seed_derivation": (
            "payload=UTF8(domain)+0x00+ASCII(str(global_seed))+0x00+ASCII(str(class_id)); "
            "digest=SHA256(payload); pair_seed=int.from_bytes(digest[0:8],"
            "byteorder='big',signed=False) & 0x7fffffffffffffff"
        ),
        "cfg_batch_contract": (
            "B=1 initial latent is duplicated into the ordered 2B conditional/null batch"
        ),
        "transition_rng_contract": (
            "every one of the 250 ancestral DDPM steps draws the full ordered 2B "
            "randn_like tensor; the t=0 draw is consumed before it is multiplied by zero"
        ),
        "compatibility_scope": (
            "same frozen model, 250-step sampler and full-2B draw semantics as the old "
            "pool; no claim of reproducing its three-class batch-correlated realizations"
        ),
    }
    if any(rng.get(key) != value for key, value in expected.items()):
        differing = {
            key: {"expected": value, "observed": rng.get(key)}
            for key, value in expected.items()
            if rng.get(key) != value
        }
        raise RuntimeError(
            "event protocol does not authorize the pair-keyed RNG contract: "
            + json.dumps(differing, sort_keys=True)
        )


def validate_event_protocol_snapshot(protocol: Mapping[str, Any]) -> tuple[int, ...]:
    identity = protocol.get("identity_sha256")
    endpoint = protocol.get("endpoint_screen", {})
    if (
        not isinstance(identity, str)
        or canonical_sha256(without_identity(protocol)) != identity
        or protocol.get("schema_version") != 3
        or protocol.get("status")
        != "FROZEN_BEFORE_REVIEWER_QUALIFICATION_OR_EVENT_RICH_SCREEN"
        or endpoint.get("model") != "DiT-XL/2 ImageNet-256"
        or endpoint.get("sampler") != "official 250-step ancestral DDPM"
        or endpoint.get("cfg_scale") != 4.0
        or endpoint.get("cfg_epsilon_channels") != 3
        or endpoint.get("endpoint_only_no_trace_saved") is not True
        or tuple(endpoint.get("discovery_seeds", ())) != EXPECTED_SEEDS
        or endpoint.get("discovery_samples_per_class") != 12
        or endpoint.get("discovery_endpoint_count") != EXPECTED_PAIR_COUNT
        or endpoint.get("class_count") != EXPECTED_CLASS_COUNT
    ):
        raise RuntimeError("event-rich protocol v3 endpoint scientific contract changed")
    require_pair_rng_contract(protocol)
    return roster_from_event_protocol(protocol)


def validate_source_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...]]:
    root = require_directory(root, "event-rich endpoint sampling source lock")
    validate_exact_source_lock_tree(root)
    protocol_path = require_regular(root / "sampling_protocol.json", "sampling protocol")
    event_path = require_regular(root / "event_protocol.json", "event protocol snapshot")
    manifest_path = require_regular(root / "manifest.json", "source-lock manifest")
    completion_path = require_regular(root / "completion.json", "source-lock completion")
    protocol = load_json(protocol_path)
    event_protocol = load_json(event_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = protocol.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    event_identity = event_protocol.get("identity_sha256")
    classes = validate_event_protocol_snapshot(event_protocol)
    scientific = protocol.get("scientific_contract", {})
    rng = protocol.get("rng_contract", {})
    if (
        not isinstance(identity, str)
        or canonical_sha256(without_identity(protocol)) != identity
        or protocol.get("schema_version") != 1
        or protocol.get("status") != "FROZEN_BEFORE_EVENT_RICH_ENDPOINT_GPU_SAMPLING"
        or protocol.get("event_protocol", {}).get("identity_sha256") != event_identity
        or protocol.get("event_protocol", {}).get("file_sha256") != sha256_file(event_path)
        or scientific.get("model") != "DiT-XL/2 ImageNet-256"
        or scientific.get("sampler") != "official 250-step ancestral DDPM"
        or scientific.get("sampling_steps") != 250
        or scientific.get("cfg_scale") != 4.0
        or scientific.get("cfg_epsilon_channels") != 3
        or tuple(scientific.get("classes_ordered", ())) != classes
        or tuple(scientific.get("global_seeds", ())) != EXPECTED_SEEDS
        or scientific.get("pair_count") != EXPECTED_PAIR_COUNT
        or scientific.get("endpoint_only") is not True
        or scientific.get("trace_saved") is not False
        or rng.get("domain") != RNG_DOMAIN
        or rng.get("unit") != "(global_seed,class_id)"
        or rng.get("same_global_seed_classes_share_initial_noise") is not False
        or rng.get("same_global_seed_classes_share_transition_innovations") is not False
        or rng.get("task_order_worker_shard_resume_invariant") is not True
        or rng.get("classes_per_sampler_invocation") != 1
        or protocol.get("evidence_access_audit")
        != {
            "labels_or_reviews_opened": False,
            "metrics_features_embeddings_or_scores_opened": False,
            "score_label_join_performed": False,
        }
        or not isinstance(manifest_identity, str)
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256") != identity
        or manifest.get("files") != artifact_records(root)
        or completion
        != {
            "complete": True,
            "sampling_protocol_identity_sha256": identity,
            "sampling_protocol_file_sha256": sha256_file(protocol_path),
            "event_protocol_identity_sha256": event_identity,
            "event_protocol_file_sha256": sha256_file(event_path),
            "manifest_identity_sha256": manifest_identity,
            "manifest_file_sha256": sha256_file(manifest_path),
        }
    ):
        raise RuntimeError("event-rich endpoint sampling source lock validation failed")
    sources = protocol.get("source_snapshots", {})
    if set(sources) != set(SOURCE_BASENAMES):
        raise RuntimeError("source snapshot set changed")
    for basename in SOURCE_BASENAMES:
        snapshot = require_regular(root / "sources" / basename, f"source {basename}")
        if sha256_file(snapshot) != sources[basename].get("sha256"):
            raise RuntimeError(f"source snapshot hash changed: {snapshot}")
    invoked = Path(__file__).resolve()
    if sha256_file(invoked) != sources[invoked.name].get("sha256"):
        raise RuntimeError("invoked launcher differs from frozen source snapshot")
    return protocol, manifest, classes


def load_frozen_strict(source_lock: Path) -> ModuleType:
    path = source_lock / "sources/reproduce_dit_imagenet256.py"
    spec = importlib.util.spec_from_file_location("_event_endpoint_launcher_strict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen strict helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_assets(
    protocol: Mapping[str, Any], source_lock: Path, dit_root: Path, checkpoint: Path, vae: Path
) -> tuple[Path, Path, Path]:
    dit_root = require_directory(dit_root, "DiT repository")
    checkpoint = require_regular(checkpoint, "DiT checkpoint")
    vae = require_directory(vae, "VAE snapshot")
    strict = load_frozen_strict(source_lock)
    observed = {
        "dit_repository": strict.validate_repository(dit_root, checkpoint),
        "checkpoint": strict.validate_checkpoint(checkpoint),
        "vae_snapshot": strict.validate_vae_snapshot(vae),
    }
    if observed != protocol.get("assets"):
        raise RuntimeError("model/checkpoint/VAE differ from source-lock identities")
    return dit_root, checkpoint, vae


def parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(result) != WORKER_COUNT or len(set(result)) != WORKER_COUNT:
        raise argparse.ArgumentTypeError("--gpus must list exactly four unique devices")
    return result


def pair_axis(classes: Sequence[int]) -> tuple[tuple[int, int], ...]:
    pairs = tuple((seed, int(class_id)) for seed in EXPECTED_SEEDS for class_id in classes)
    if len(pairs) != EXPECTED_PAIR_COUNT or len(set(pairs)) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("pair axis is not the exact 12 x 84 Cartesian product")
    return pairs


def assignments(
    gpus: Sequence[str], classes: Sequence[int]
) -> dict[str, tuple[tuple[int, int], ...]]:
    if len(gpus) != WORKER_COUNT or len(set(gpus)) != WORKER_COUNT:
        raise ValueError("exactly four unique GPUs are required")
    axis = pair_axis(classes)
    result = {
        gpu: axis[index * PAIRS_PER_WORKER : (index + 1) * PAIRS_PER_WORKER]
        for index, gpu in enumerate(gpus)
    }
    if tuple(pair for shard in result.values() for pair in shard) != axis:
        raise AssertionError("GPU shards changed, reordered, omitted, or duplicated a pair")
    if any(len(shard) != PAIRS_PER_WORKER for shard in result.values()):
        raise AssertionError("each GPU must receive exactly 252 pairs")
    return result


def validate_output_root(path: Path) -> Path:
    output = path.expanduser().absolute()
    if os.path.lexists(output) and (not output.is_dir() or output.is_symlink()):
        raise RuntimeError(f"output root must be a real directory: {output}")
    forbidden = {Path("/"), ROOT, ROOT.parent, DATA_ROOT, DATA_ROOT.parent}
    if output in forbidden:
        raise RuntimeError(f"refusing broad output root: {output}")
    return output


def build_plan(
    protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    classes: Sequence[int],
    gpus: Sequence[str],
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae: Path,
) -> dict[str, Any]:
    allocation = assignments(gpus, classes)
    return {
        "schema_version": 1,
        "status": "FROZEN_FOUR_GPU_ENDPOINT_EXECUTION_PLAN",
        "sampling_protocol_identity_sha256": protocol["identity_sha256"],
        "sampling_manifest_identity_sha256": source_manifest["identity_sha256"],
        "event_protocol_identity_sha256": protocol["event_protocol"]["identity_sha256"],
        "classes_ordered": list(classes),
        "global_seeds": list(EXPECTED_SEEDS),
        "pair_axis_order": "seed-major, frozen-class-roster-minor",
        "pair_count": EXPECTED_PAIR_COUNT,
        "gpus_ordered": list(gpus),
        "assignment": {
            gpu: [
                {"global_seed": seed, "class_id": class_id}
                for seed, class_id in shard
            ]
            for gpu, shard in allocation.items()
        },
        "assignment_kind": "four contiguous 252-pair operational shards",
        "rng_unit": "(global_seed,class_id), independent of assignment",
        "same_global_seed_classes_share_initial_or_transition_innovation": False,
        "output_root": str(output_root),
        "dit_root": str(dit_root),
        "checkpoint": str(checkpoint),
        "vae_snapshot": str(vae),
        "endpoint_only": True,
        "trace_saved": False,
        "labels_reviews_metrics_features_embeddings_or_scores_read": False,
    }


def task_file_value(tasks: Sequence[tuple[int, int]]) -> dict[str, Any]:
    rows = [{"global_seed": seed, "class_id": class_id} for seed, class_id in tasks]
    return {"tasks": rows, "tasks_sha256": canonical_sha256(rows)}


def write_or_validate_plan_files(
    output_root: Path,
    plan: Mapping[str, Any],
    allocation: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[Path, dict[str, Path]]:
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
        (output_root / "pairs").mkdir(exist_ok=False)
        (output_root / "_runner_tasks").mkdir(exist_ok=False)
        (output_root / "_runner_logs").mkdir(exist_ok=False)
    for name in ("pairs", "_runner_tasks", "_runner_logs"):
        require_directory(output_root / name, name)
    plan_path = output_root / "execution_plan.json"
    if os.path.lexists(plan_path):
        if plan_path.is_symlink() or load_json(plan_path) != plan:
            raise RuntimeError("existing execution plan differs; refusing overwrite")
    else:
        allowed = {"pairs", "_runner_tasks", "_runner_logs"}
        if {path.name for path in output_root.iterdir()} != allowed:
            raise RuntimeError("nonempty output root lacks its exact execution-plan skeleton")
        exclusive_json(plan_path, plan)
    task_paths: dict[str, Path] = {}
    for index, (gpu, tasks) in enumerate(allocation.items()):
        path = output_root / "_runner_tasks" / f"worker{index}.json"
        expected = task_file_value(tasks)
        if os.path.lexists(path):
            if path.is_symlink() or load_json(path) != expected:
                raise RuntimeError(f"existing worker task file changed: {path}")
        else:
            exclusive_json(path, expected)
        task_paths[gpu] = path
    return plan_path, task_paths


def next_log_path(output_root: Path, worker_index: int) -> Path:
    root = output_root / "_runner_logs"
    for attempt in range(1, 10_000):
        candidate = root / f"worker{worker_index}_attempt{attempt:04d}.log"
        if not os.path.lexists(candidate):
            return candidate
    raise RuntimeError(f"too many prior attempts for worker {worker_index}")


def run_worker_process(
    worker_index: int,
    gpu: str,
    task_path: Path,
    source_lock: Path,
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae: Path,
) -> None:
    worker_source = source_lock / "sources/sample_dit_imagenet256_endpoint_pairs.py"
    command = [
        sys.executable,
        str(worker_source),
        "--source-lock",
        str(source_lock),
        "--tasks",
        str(task_path),
        "--output-root",
        str(output_root),
        "--dit-root",
        str(dit_root),
        "--checkpoint",
        str(checkpoint),
        "--vae-snapshot",
        str(vae),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = next_log_path(output_root, worker_index)
    started = time.time()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {"started_unix": started, "worker_index": worker_index, "gpu": gpu, "command": command},
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        finished = time.time()
        log.write(
            json.dumps(
                {
                    "finished_unix": finished,
                    "elapsed_seconds": finished - started,
                    "returncode": completed.returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker {worker_index} failed closed on GPU {gpu}; inspect {log_path}"
        )


def log_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_root / "_runner_logs").iterdir()):
        if not path.is_file() or path.is_symlink() or path.suffix != ".log":
            raise RuntimeError(f"unexpected runner log entry: {path}")
        records.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def load_pair_module(source_lock: Path) -> ModuleType:
    path = source_lock / "sources/sample_dit_imagenet256_endpoint_pairs.py"
    spec = importlib.util.spec_from_file_location("_event_endpoint_pair_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen pair sampler: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_all_pairs(
    output_root: Path,
    classes: Sequence[int],
    source_lock: Path,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pair_module = load_pair_module(source_lock)
    strict = pair_module.load_frozen_strict(source_lock)
    expected_directories = {
        f"seed{seed:04d}_class{class_id:04d}"
        for seed, class_id in pair_axis(classes)
    }
    pairs_root = require_directory(output_root / "pairs", "endpoint pairs root")
    observed_directories: set[str] = set()
    unexpected: list[str] = []
    for path in pairs_root.iterdir():
        if path.is_symlink() or not path.is_dir():
            unexpected.append(path.name)
        else:
            observed_directories.add(path.name)
    if observed_directories != expected_directories or unexpected:
        raise RuntimeError(
            "endpoint pair directory axis changed: "
            f"missing={sorted(expected_directories-observed_directories)[:8]}, "
            f"extra={sorted(observed_directories-expected_directories)[:8]}, "
            f"unexpected={sorted(unexpected)[:8]}"
        )
    records = [
        pair_module.validate_pair_output(
            output_root, seed, class_id, source_lock, protocol, strict
        )
        for seed, class_id in pair_axis(classes)
    ]
    if [(row["global_seed"], row["class_id"]) for row in records] != list(
        pair_axis(classes)
    ):
        raise RuntimeError("validated pair receipts changed axis/order")
    return records


def publish_pool_receipts(
    output_root: Path,
    plan_path: Path,
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_path = output_root / "pool_manifest.json"
    completion_path = output_root / "pool_completion.json"
    if os.path.lexists(manifest_path) or os.path.lexists(completion_path):
        raise RuntimeError("refusing to overwrite prior pool receipts")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "sampling_protocol_identity_sha256": protocol["identity_sha256"],
        "event_protocol_identity_sha256": protocol["event_protocol"]["identity_sha256"],
        "execution_plan_sha256": sha256_file(plan_path),
        "class_count": EXPECTED_CLASS_COUNT,
        "global_seed_count": len(EXPECTED_SEEDS),
        "endpoint_count": EXPECTED_PAIR_COUNT,
        "pair_outputs": list(records),
        "runner_logs": log_records(output_root),
        "endpoint_only": True,
        "trace_saved": False,
        "labels_reviews_metrics_features_embeddings_or_scores_read": False,
    }
    manifest["identity_sha256"] = canonical_sha256(manifest)
    exclusive_json(manifest_path, manifest)
    completion = {
        "complete": True,
        "pool_identity_sha256": manifest["identity_sha256"],
        "pool_manifest_sha256": sha256_file(manifest_path),
        "execution_plan_sha256": sha256_file(plan_path),
        "endpoint_count": EXPECTED_PAIR_COUNT,
    }
    exclusive_json(completion_path, completion)
    return completion


def validate_complete_pool(
    output_root: Path,
    expected_plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    classes: Sequence[int],
    source_lock: Path,
) -> dict[str, Any]:
    allowed_top_level = {
        "pairs",
        "_runner_tasks",
        "_runner_logs",
        "execution_plan.json",
        "pool_manifest.json",
        "pool_completion.json",
    }
    observed_top_level = {path.name for path in output_root.iterdir()}
    if observed_top_level != allowed_top_level or any(
        path.is_symlink() for path in output_root.iterdir()
    ):
        raise RuntimeError("completed output root member set changed")
    task_root = require_directory(output_root / "_runner_tasks", "runner tasks")
    expected_task_names = {f"worker{index}.json" for index in range(WORKER_COUNT)}
    if {path.name for path in task_root.iterdir()} != expected_task_names or any(
        not path.is_file() or path.is_symlink() for path in task_root.iterdir()
    ):
        raise RuntimeError("runner task-file set changed")
    plan_gpus = expected_plan.get("gpus_ordered")
    plan_assignment = expected_plan.get("assignment")
    if not isinstance(plan_gpus, list) or not isinstance(plan_assignment, dict):
        raise RuntimeError("execution plan lacks frozen GPU/task assignment")
    for index, gpu in enumerate(plan_gpus):
        rows = plan_assignment.get(str(gpu))
        if not isinstance(rows, list):
            raise RuntimeError(f"execution plan lacks assignment for GPU {gpu}")
        expected_task_file = {
            "tasks": rows,
            "tasks_sha256": canonical_sha256(rows),
        }
        if load_json(task_root / f"worker{index}.json") != expected_task_file:
            raise RuntimeError(f"runner task file differs from execution plan: worker {index}")
    plan_path = require_regular(output_root / "execution_plan.json", "execution plan")
    manifest_path = require_regular(output_root / "pool_manifest.json", "pool manifest")
    completion_path = require_regular(output_root / "pool_completion.json", "pool completion")
    if load_json(plan_path) != expected_plan:
        raise RuntimeError("completed pool execution plan differs")
    records = validate_all_pairs(output_root, classes, source_lock, protocol)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("event_protocol_identity_sha256")
        != protocol["event_protocol"]["identity_sha256"]
        or manifest.get("execution_plan_sha256") != sha256_file(plan_path)
        or manifest.get("class_count") != EXPECTED_CLASS_COUNT
        or manifest.get("global_seed_count") != len(EXPECTED_SEEDS)
        or manifest.get("endpoint_count") != EXPECTED_PAIR_COUNT
        or manifest.get("pair_outputs") != records
        or manifest.get("runner_logs") != log_records(output_root)
        or manifest.get("endpoint_only") is not True
        or manifest.get("trace_saved") is not False
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or completion
        != {
            "complete": True,
            "pool_identity_sha256": manifest.get("identity_sha256"),
            "pool_manifest_sha256": sha256_file(manifest_path),
            "execution_plan_sha256": sha256_file(plan_path),
            "endpoint_count": EXPECTED_PAIR_COUNT,
        }
    ):
        raise RuntimeError("completed endpoint pool failed full validation")
    return completion


def output_state(output_root: Path, classes: Sequence[int]) -> dict[str, Any]:
    if not output_root.exists():
        return {"state": "absent", "completed_pairs": 0, "partial_pairs": 0}
    completed = 0
    partial = 0
    for seed, class_id in pair_axis(classes):
        path = output_root / f"pairs/seed{seed:04d}_class{class_id:04d}"
        if not os.path.lexists(path):
            continue
        if (path / "manifest.json").is_file() and (path / "completion.json").is_file():
            completed += 1
        else:
            partial += 1
    return {
        "state": "present",
        "completed_pair_receipts": completed,
        "partial_pair_paths": partial,
        "note": "dry-run counts receipts; real run fully rehashes every reused endpoint",
    }


def provisional_dry_run(event_protocol_lock: Path) -> None:
    root = require_directory(event_protocol_lock, "event protocol lock")
    protocol = load_json(root / "protocol.json")
    endpoint = protocol.get("endpoint_screen", {})
    classes = roster_from_event_protocol(protocol)
    report: dict[str, Any] = {
        "mode": "PROVISIONAL_PROTOCOL_INSPECTION_ONLY",
        "real_sampling_authorized": False,
        "protocol_schema_version": protocol.get("schema_version"),
        "protocol_identity_sha256": protocol.get("identity_sha256"),
        "class_count": len(classes),
        "global_seeds": endpoint.get("discovery_seeds"),
        "endpoint_count": len(classes) * len(endpoint.get("discovery_seeds", ())),
        "model": endpoint.get("model"),
        "sampler": endpoint.get("sampler"),
        "observed_batch_rng_contract": endpoint.get("batch_rng_contract"),
    }
    try:
        validate_event_protocol_snapshot(protocol)
    except RuntimeError as exc:
        report["compatible_with_final_pair_rng_contract"] = False
        report["blocking_reason"] = str(exc)
    else:
        report["compatible_with_final_pair_rng_contract"] = True
        report["blocking_reason"] = "final sampling source lock has not been created"
    print(json.dumps(report, indent=2, sort_keys=True))


def run_self_test() -> None:
    synthetic_classes = tuple(range(EXPECTED_CLASS_COUNT))
    axis = pair_axis(synthetic_classes)
    allocation = assignments(("0", "1", "2", "3"), synthetic_classes)
    if len(axis) != EXPECTED_PAIR_COUNT or any(
        len(shard) != PAIRS_PER_WORKER for shard in allocation.values()
    ):
        raise AssertionError("exact Cartesian pair axis or four-way split changed")
    if axis[0] != (1000, 0) or axis[-1] != (1011, 83):
        raise AssertionError("pair axis ordering changed")
    pair_source = Path(__file__).resolve().with_name(
        "sample_dit_imagenet256_endpoint_pairs.py"
    )
    completed = subprocess.run(
        [sys.executable, str(pair_source), "--self-test"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("pair sampler self-test failed:\n" + completed.stdout + completed.stderr)
    for invalid in ("0,1,2", "0,1,2,2", "0,1,2,3,4"):
        try:
            parse_gpus(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"invalid GPU list was accepted: {invalid}")
    print(
        "self-test passed: exact 1008 pair axis, four contiguous 252-pair shards, "
        "pair RNG known answers/order invariance, and no GPU sampling"
    )


def run_smoke_test() -> None:
    pair_source = Path(__file__).resolve().with_name(
        "sample_dit_imagenet256_endpoint_pairs.py"
    )
    completed = subprocess.run(
        [sys.executable, str(pair_source), "--smoke-test"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("pair sampler smoke-test failed:\n" + completed.stdout + completed.stderr)
    classes = tuple(range(EXPECTED_CLASS_COUNT))
    allocation = assignments(("0", "1", "2", "3"), classes)
    with tempfile.TemporaryDirectory(prefix="dit-endpoint-launcher-smoke-") as raw:
        root = Path(raw) / "pool"
        plan = {
            "synthetic": True,
            "axis_sha256": canonical_sha256(pair_axis(classes)),
        }
        plan_path, task_paths = write_or_validate_plan_files(root, plan, allocation)
        if load_json(plan_path) != plan or len(task_paths) != WORKER_COUNT:
            raise AssertionError("execution plan/task files failed round trip")
        write_or_validate_plan_files(root, plan, allocation)
        first = next(iter(task_paths.values()))
        changed = load_json(first)
        changed["tasks_sha256"] = "0" * 64
        first.write_text(json.dumps(changed), encoding="utf-8")
        try:
            write_or_validate_plan_files(root, plan, allocation)
        except RuntimeError:
            pass
        else:
            raise AssertionError("changed worker task file escaped no-overwrite validation")
    print(
        "smoke-test passed: synthetic endpoint receipt, execution-plan/task resume, "
        "and mutation rejection; no model load or GPU sampling"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--event-protocol-lock", type=Path)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path, default=DEFAULT_VAE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--validate-source-lock", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.smoke_test:
        run_smoke_test()
        return 0
    if args.dry_run and args.event_protocol_lock is not None:
        provisional_dry_run(args.event_protocol_lock)
        return 0
    source_lock = args.source_lock.expanduser().absolute()
    protocol, source_manifest, classes = validate_source_lock(source_lock)
    if args.validate_source_lock:
        print(
            json.dumps(
                {
                    "valid": True,
                    "sampling_protocol_identity_sha256": protocol["identity_sha256"],
                    "source_manifest_identity_sha256": source_manifest["identity_sha256"],
                    "event_protocol_identity_sha256": protocol["event_protocol"]["identity_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output_root = validate_output_root(args.output_root)
    checkpoint_arg = args.checkpoint or args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
    dit_root, checkpoint, vae = validate_assets(
        protocol, source_lock, args.dit_root, checkpoint_arg, args.vae_snapshot
    )
    plan = build_plan(
        protocol,
        source_manifest,
        classes,
        args.gpus,
        output_root,
        dit_root,
        checkpoint,
        vae,
    )
    if args.dry_run:
        print(
            json.dumps(
                {"execution_plan": plan, "output_state": output_state(output_root, classes)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    allocation = assignments(args.gpus, classes)
    plan_path, task_paths = write_or_validate_plan_files(
        output_root, plan, allocation
    )
    pool_manifest = output_root / "pool_manifest.json"
    pool_completion = output_root / "pool_completion.json"
    if pool_completion.exists():
        completion = validate_complete_pool(
            output_root, plan, protocol, classes, source_lock
        )
        print(json.dumps({**completion, "reused_complete_pool": True}, indent=2, sort_keys=True))
        return 0
    if pool_manifest.exists():
        raise RuntimeError("partial pool manifest exists; refusing overwrite")

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        futures = [
            executor.submit(
                run_worker_process,
                index,
                gpu,
                task_paths[gpu],
                source_lock,
                output_root,
                dit_root,
                checkpoint,
                vae,
            )
            for index, gpu in enumerate(args.gpus)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    records = validate_all_pairs(output_root, classes, source_lock, protocol)
    completion = publish_pool_receipts(
        output_root, plan_path, protocol, records
    )
    validated = validate_complete_pool(
        output_root, plan, protocol, classes, source_lock
    )
    if completion != validated:
        raise RuntimeError("new pool receipt failed validation round trip")
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
