#!/usr/bin/env python3
"""Resumable endpoint-only launcher for the scientific-v4.2.1 DiT screen.

The frozen axis is the 84-class roster crossed with global seeds 1000..1011
(1,008 endpoint PNGs).  Tasks are ordered seed-major, class-roster-minor and
split into four immutable *logical* 252-pair shards.  Any fixed subset of
logical workers 0..3 can be run later on any currently free physical GPU(s),
including sequentially on one GPU.  Scientific randomness belongs to the pair
key rather than a shard, launch order, or physical device, so this operational
scheduling cannot affect a pair's RNG stream.  Pool-level validation and
receipts are attempted only after all four logical-shard receipts exist.

This source is frozen with ``execution_ready=false`` while reviewer
qualification and the remaining execution chain are incomplete.  It reads no labels, reviews, metrics,
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

sys.dont_write_bytecode = True


def find_repo_root(source: Path) -> Path:
    for candidate in source.resolve().parents:
        if (candidate / ".git").exists() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate repository root from {source}")


ROOT = find_repo_root(Path(__file__))
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
DEFAULT_SOURCE_LOCK = (
    ROOT
    / "experiments/locks/dit_scientific_v4_2_1_endpoint_sampling_source_lock_v1"
)
DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT / "cross_scale_evidence/dit_scientific_v4_2_1_endpoint_screen_v1"
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
LOGICAL_WORKERS = tuple(range(WORKER_COUNT))
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
        if not path.is_file() or (
            path.parent == root and path.name in {"manifest.json", "completion.json"}
        ):
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
        "pre_sampling_zero_audit.json",
        "selftest_receipt.json",
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
        or protocol.get("schema_version") != 4
        or protocol.get("status")
        != "SCIENTIFIC_V4_2_1_CLAIM_LIMITED_FROZEN_EXECUTION_NOT_READY"
        or protocol.get("scientific_revision") != "v4.2.1"
        or endpoint.get("model") != "DiT-XL/2 ImageNet-256"
        or endpoint.get("sampler") != "official 250-step ancestral DDPM"
        or endpoint.get("cfg_scale") != 4.0
        or endpoint.get("cfg_epsilon_channels") != 3
        or endpoint.get("endpoint_only_no_trace_saved") is not True
        or tuple(endpoint.get("discovery_seeds", ())) != EXPECTED_SEEDS
        or endpoint.get("discovery_samples_per_class") != 12
        or endpoint.get("discovery_endpoint_count") != EXPECTED_PAIR_COUNT
        or endpoint.get("class_count") != EXPECTED_CLASS_COUNT
        or protocol.get("method_lock", {}).get("identity_sha256")
        != "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"
        or protocol.get("method_boundary", {}).get(
            "screen_is_external_population_design_not_a_method"
        )
        is not True
    ):
        raise RuntimeError("scientific-v4 endpoint scientific contract changed")
    require_pair_rng_contract(protocol)
    return roster_from_event_protocol(protocol)


def validate_source_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...]]:
    root = require_directory(root, "scientific-v4 endpoint sampling source lock")
    validate_exact_source_lock_tree(root)
    protocol_path = require_regular(root / "sampling_protocol.json", "sampling protocol")
    event_path = require_regular(root / "event_protocol.json", "event protocol snapshot")
    manifest_path = require_regular(root / "manifest.json", "source-lock manifest")
    completion_path = require_regular(root / "completion.json", "source-lock completion")
    zero_audit_path = require_regular(
        root / "pre_sampling_zero_audit.json", "pre-sampling zero-output audit"
    )
    selftest_path = require_regular(
        root / "selftest_receipt.json", "source-lock self-test receipt"
    )
    protocol = load_json(protocol_path)
    event_protocol = load_json(event_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    zero_audit = load_json(zero_audit_path)
    selftest_receipt = load_json(selftest_path)
    identity = protocol.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    event_identity = event_protocol.get("identity_sha256")
    classes = validate_event_protocol_snapshot(event_protocol)
    scientific = protocol.get("scientific_contract", {})
    rng = protocol.get("rng_contract", {})
    execution = protocol.get("execution_contract", {})
    method = protocol.get("method_lock", {})
    event_method = event_protocol.get("method_lock", {})
    if (
        not isinstance(identity, str)
        or canonical_sha256(without_identity(protocol)) != identity
        or protocol.get("schema_version") != 1
        or protocol.get("status")
        != "SCIENTIFIC_V4_2_1_ENDPOINT_SOURCE_FROZEN_EXECUTION_NOT_READY"
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
        or protocol.get("execution_ready") is not False
        or protocol.get("real_endpoint_outputs_present_at_freeze") is not False
        or protocol.get("real_expert_label_review_or_consensus_results_present")
        is not False
        or method.get("identity_sha256")
        != "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"
        or any(
            method.get(key) != event_method.get(key)
            for key in (
                "identity_sha256",
                "protocol_file_sha256",
                "manifest_file_sha256",
                "completion_file_sha256",
                "matched_q_power_gate_file_sha256",
                "adaptive_null_audit_file_sha256",
            )
        )
        or execution.get("logical_worker_count") != WORKER_COUNT
        or execution.get("pair_count_per_worker") != PAIRS_PER_WORKER
        or execution.get("allowed_logical_worker_subset") != list(LOGICAL_WORKERS)
        or execution.get(
            "pool_validation_and_receipts_only_after_all_four_logical_shard_receipts"
        )
        is not True
        or execution.get("physical_gpu_schedule_is_not_scientific_rng_input")
        is not True
        or protocol.get("external_evaluation_boundary", {}).get(
            "class_selection_is_evaluation_event_enrichment_not_method"
        )
        is not True
        or protocol.get("external_evaluation_boundary", {}).get(
            "endpoint_or_review_artifacts_are_forbidden_B_E_method_inputs"
        )
        is not True
        or set(zero_audit)
        != {
            "audit_status",
            "audited_paths",
            "completed_real_endpoint_pairs",
            "partial_real_endpoint_pairs",
            "real_endpoint_files",
            "real_sampling_started",
        }
        or zero_audit.get("audit_status")
        != "ZERO_REAL_OUTPUTS_BEFORE_V4_2_1_SOURCE_FREEZE"
        or zero_audit.get("completed_real_endpoint_pairs") != 0
        or zero_audit.get("partial_real_endpoint_pairs") != 0
        or zero_audit.get("real_endpoint_files") != 0
        or zero_audit.get("real_sampling_started") is not False
        or selftest_receipt
        != {
            "all_four_logical_shards_required_for_pool_receipts": True,
            "exact_tree_and_provenance_checked": True,
            "gpu_or_model_used": False,
            "logical_subset_and_physical_schedule_invariance_checked": True,
            "pair_rng_known_answers_checked": True,
            "status": "PASS_SYNTHETIC_NO_REAL_DATA",
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
        raise RuntimeError("scientific-v4 endpoint sampling source lock validation failed")
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
    if not 1 <= len(result) <= WORKER_COUNT or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "--gpus must list one to four unique currently free physical devices"
        )
    return result


def parse_logical_workers(value: str) -> tuple[int, ...]:
    pieces = tuple(part.strip() for part in value.split(",") if part.strip())
    try:
        result = tuple(int(part) for part in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--logical-workers must be a comma-separated subset of 0,1,2,3"
        ) from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(str(index) != part for index, part in zip(result, pieces))
        or any(index not in LOGICAL_WORKERS for index in result)
    ):
        raise argparse.ArgumentTypeError(
            "--logical-workers must be a duplicate-free nonempty subset of 0,1,2,3"
        )
    return result


def pair_axis(classes: Sequence[int]) -> tuple[tuple[int, int], ...]:
    pairs = tuple((seed, int(class_id)) for seed in EXPECTED_SEEDS for class_id in classes)
    if len(pairs) != EXPECTED_PAIR_COUNT or len(set(pairs)) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("pair axis is not the exact 12 x 84 Cartesian product")
    return pairs


def logical_assignments(
    classes: Sequence[int],
) -> dict[int, tuple[tuple[int, int], ...]]:
    axis = pair_axis(classes)
    result = {
        index: axis[index * PAIRS_PER_WORKER : (index + 1) * PAIRS_PER_WORKER]
        for index in LOGICAL_WORKERS
    }
    if tuple(pair for shard in result.values() for pair in shard) != axis:
        raise AssertionError("logical shards changed, reordered, omitted, or duplicated a pair")
    if any(len(shard) != PAIRS_PER_WORKER for shard in result.values()):
        raise AssertionError("each logical worker must receive exactly 252 pairs")
    return result


def physical_queues(
    logical_workers: Sequence[int], physical_gpus: Sequence[str]
) -> dict[str, tuple[int, ...]]:
    if (
        not logical_workers
        or len(set(logical_workers)) != len(logical_workers)
        or any(index not in LOGICAL_WORKERS for index in logical_workers)
    ):
        raise ValueError("logical worker subset changed")
    if (
        not physical_gpus
        or len(physical_gpus) > WORKER_COUNT
        or len(set(physical_gpus)) != len(physical_gpus)
    ):
        raise ValueError("physical GPU set changed")
    usable = tuple(physical_gpus[: min(len(physical_gpus), len(logical_workers))])
    queues = {
        gpu: tuple(logical_workers[offset:: len(usable)])
        for offset, gpu in enumerate(usable)
    }
    flattened = tuple(
        worker
        for position in range(max(map(len, queues.values())))
        for gpu in usable
        for worker in queues[gpu][position : position + 1]
    )
    if flattened != tuple(logical_workers):
        raise AssertionError("physical scheduling changed logical launch order/coverage")
    return queues


def all_logical_shards_complete(completed: Iterable[int]) -> bool:
    values = tuple(completed)
    return len(values) == WORKER_COUNT and set(values) == set(LOGICAL_WORKERS)


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
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae: Path,
) -> dict[str, Any]:
    allocation = logical_assignments(classes)
    return {
        "schema_version": 1,
        "status": "FROZEN_FOUR_LOGICAL_SHARD_ENDPOINT_EXECUTION_PLAN",
        "sampling_protocol_identity_sha256": protocol["identity_sha256"],
        "sampling_manifest_identity_sha256": source_manifest["identity_sha256"],
        "event_protocol_identity_sha256": protocol["event_protocol"]["identity_sha256"],
        "classes_ordered": list(classes),
        "global_seeds": list(EXPECTED_SEEDS),
        "pair_axis_order": "seed-major, frozen-class-roster-minor",
        "pair_count": EXPECTED_PAIR_COUNT,
        "logical_workers_ordered": list(LOGICAL_WORKERS),
        "logical_assignment": {
            str(worker_index): [
                {"global_seed": seed, "class_id": class_id}
                for seed, class_id in shard
            ]
            for worker_index, shard in allocation.items()
        },
        "assignment_kind": "four immutable contiguous 252-pair logical shards",
        "physical_gpu_schedule_in_plan": None,
        "physical_gpu_and_launch_order_are_not_scientific_inputs": True,
        "allowed_execution": (
            "any fixed nonempty subset of logical workers 0..3 on one to four "
            "currently free physical GPUs; workers assigned to one GPU run sequentially"
        ),
        "pool_receipt_requires_all_logical_workers": True,
        "rng_unit": "(global_seed,class_id), independent of logical/physical assignment",
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
    allocation: Mapping[int, Sequence[tuple[int, int]]],
) -> tuple[Path, dict[int, Path]]:
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
        (output_root / "pairs").mkdir(exist_ok=False)
        (output_root / "_runner_tasks").mkdir(exist_ok=False)
        (output_root / "_runner_logs").mkdir(exist_ok=False)
        (output_root / "_logical_shards").mkdir(exist_ok=False)
    for name in ("pairs", "_runner_tasks", "_runner_logs", "_logical_shards"):
        require_directory(output_root / name, name)
    plan_path = output_root / "execution_plan.json"
    if os.path.lexists(plan_path):
        if plan_path.is_symlink() or load_json(plan_path) != plan:
            raise RuntimeError("existing execution plan differs; refusing overwrite")
    else:
        allowed = {"pairs", "_runner_tasks", "_runner_logs", "_logical_shards"}
        if {path.name for path in output_root.iterdir()} != allowed:
            raise RuntimeError("nonempty output root lacks its exact execution-plan skeleton")
        exclusive_json(plan_path, plan)
    task_paths: dict[int, Path] = {}
    for worker_index, tasks in allocation.items():
        path = output_root / "_runner_tasks" / f"worker{worker_index}.json"
        expected = task_file_value(tasks)
        if os.path.lexists(path):
            if path.is_symlink() or load_json(path) != expected:
                raise RuntimeError(f"existing worker task file changed: {path}")
        else:
            exclusive_json(path, expected)
        task_paths[worker_index] = path
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


def run_physical_gpu_queue(
    gpu: str,
    logical_workers: Sequence[int],
    task_paths: Mapping[int, Path],
    source_lock: Path,
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae: Path,
) -> None:
    for logical_worker in logical_workers:
        run_worker_process(
            logical_worker,
            gpu,
            task_paths[logical_worker],
            source_lock,
            output_root,
            dit_root,
            checkpoint,
            vae,
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


def validate_logical_shard_pairs(
    output_root: Path,
    logical_worker: int,
    classes: Sequence[int],
    source_lock: Path,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if logical_worker not in LOGICAL_WORKERS:
        raise RuntimeError(f"invalid logical worker index: {logical_worker}")
    tasks = logical_assignments(classes)[logical_worker]
    pair_module = load_pair_module(source_lock)
    strict = pair_module.load_frozen_strict(source_lock)
    records = [
        pair_module.validate_pair_output(
            output_root, seed, class_id, source_lock, protocol, strict
        )
        for seed, class_id in tasks
    ]
    if [(row["global_seed"], row["class_id"]) for row in records] != list(tasks):
        raise RuntimeError(f"logical worker {logical_worker} pair axis/order changed")
    return records


def logical_shard_receipt_value(
    output_root: Path,
    logical_worker: int,
    classes: Sequence[int],
    source_lock: Path,
    protocol: Mapping[str, Any],
    plan_path: Path,
) -> dict[str, Any]:
    records = validate_logical_shard_pairs(
        output_root, logical_worker, classes, source_lock, protocol
    )
    task_path = require_regular(
        output_root / "_runner_tasks" / f"worker{logical_worker}.json",
        f"logical worker {logical_worker} task file",
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "logical_worker_index": logical_worker,
        "pair_count": PAIRS_PER_WORKER,
        "pair_axis_sha256": canonical_sha256(
            [
                {"global_seed": row["global_seed"], "class_id": row["class_id"]}
                for row in records
            ]
        ),
        "pair_outputs_sha256": canonical_sha256(records),
        "sampling_protocol_identity_sha256": protocol["identity_sha256"],
        "event_protocol_identity_sha256": protocol["event_protocol"][
            "identity_sha256"
        ],
        "execution_plan_sha256": sha256_file(plan_path),
        "task_file_sha256": sha256_file(task_path),
        "physical_gpu_or_launch_order_is_scientific_input": False,
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def logical_shard_receipt_path(output_root: Path, logical_worker: int) -> Path:
    return output_root / "_logical_shards" / f"worker{logical_worker}_completion.json"


def publish_or_validate_logical_shard_receipt(
    output_root: Path,
    logical_worker: int,
    classes: Sequence[int],
    source_lock: Path,
    protocol: Mapping[str, Any],
    plan_path: Path,
) -> dict[str, Any]:
    expected = logical_shard_receipt_value(
        output_root,
        logical_worker,
        classes,
        source_lock,
        protocol,
        plan_path,
    )
    path = logical_shard_receipt_path(output_root, logical_worker)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or load_json(path) != expected:
            raise RuntimeError(
                f"logical worker {logical_worker} completion receipt changed"
            )
    else:
        exclusive_json(path, expected)
    return expected


def completed_logical_shards(
    output_root: Path,
    classes: Sequence[int],
    source_lock: Path,
    protocol: Mapping[str, Any],
    plan_path: Path,
) -> dict[int, dict[str, Any]]:
    root = require_directory(output_root / "_logical_shards", "logical shard receipts")
    allowed_names = {
        f"worker{index}_completion.json" for index in LOGICAL_WORKERS
    }
    observed_names = {path.name for path in root.iterdir()}
    if not observed_names.issubset(allowed_names) or any(
        not path.is_file() or path.is_symlink() for path in root.iterdir()
    ):
        raise RuntimeError("logical shard receipt directory contains an unexpected entry")
    completed: dict[int, dict[str, Any]] = {}
    for logical_worker in LOGICAL_WORKERS:
        path = logical_shard_receipt_path(output_root, logical_worker)
        if not path.exists():
            continue
        completed[logical_worker] = publish_or_validate_logical_shard_receipt(
            output_root,
            logical_worker,
            classes,
            source_lock,
            protocol,
            plan_path,
        )
    return completed


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
    logical_shard_receipts: Mapping[int, Mapping[str, Any]],
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
        "logical_shard_receipts": [
            dict(logical_shard_receipts[index]) for index in LOGICAL_WORKERS
        ],
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
        "_logical_shards",
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
    plan_workers = expected_plan.get("logical_workers_ordered")
    plan_assignment = expected_plan.get("logical_assignment")
    if plan_workers != list(LOGICAL_WORKERS) or not isinstance(plan_assignment, dict):
        raise RuntimeError("execution plan lacks frozen logical-worker assignment")
    for index in LOGICAL_WORKERS:
        rows = plan_assignment.get(str(index))
        if not isinstance(rows, list):
            raise RuntimeError(f"execution plan lacks assignment for logical worker {index}")
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
    logical_receipts = completed_logical_shards(
        output_root, classes, source_lock, protocol, plan_path
    )
    if not all_logical_shards_complete(logical_receipts):
        raise RuntimeError("pool validation requires all four logical-shard receipts")
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
        or manifest.get("logical_shard_receipts")
        != [logical_receipts[index] for index in LOGICAL_WORKERS]
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
        return {
            "state": "absent",
            "completed_pairs": 0,
            "partial_pairs": 0,
            "logical_shard_receipts": [],
        }
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
        "logical_shard_receipts": [
            index
            for index in LOGICAL_WORKERS
            if logical_shard_receipt_path(output_root, index).is_file()
            and not logical_shard_receipt_path(output_root, index).is_symlink()
        ],
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
    allocation = logical_assignments(synthetic_classes)
    if len(axis) != EXPECTED_PAIR_COUNT or any(
        len(shard) != PAIRS_PER_WORKER for shard in allocation.values()
    ):
        raise AssertionError("exact Cartesian pair axis or four-way split changed")
    if axis[0] != (1000, 0) or axis[-1] != (1011, 83):
        raise AssertionError("pair axis ordering changed")
    one_gpu = physical_queues((3, 0, 2), ("7",))
    two_gpus = physical_queues((3, 0, 2), ("7", "5"))
    if one_gpu != {"7": (3, 0, 2)} or two_gpus != {"7": (3, 2), "5": (0,)}:
        raise AssertionError("logical-subset physical scheduling changed")
    if logical_assignments(tuple(reversed(synthetic_classes)))[0] == allocation[0]:
        raise AssertionError("logical assignment ignored the frozen class roster")
    for incomplete in ((), (0,), (0, 1, 2), (0, 1, 2, 2)):
        if all_logical_shards_complete(incomplete):
            raise AssertionError("an incomplete logical-shard set authorized pool receipts")
    if not all_logical_shards_complete((3, 1, 0, 2)):
        raise AssertionError("the exact four logical shards did not authorize pool receipts")
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
    for invalid in ("", "0,0", "0,1,2,3,4"):
        try:
            parse_gpus(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"invalid GPU list was accepted: {invalid}")
    for invalid in ("", "0,0", "-1", "4", "0,1,4"):
        try:
            parse_logical_workers(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"invalid logical worker subset was accepted: {invalid}")
    print(
        "self-test passed: exact 1008 pair axis, four immutable logical 252-pair "
        "shards, arbitrary subset/sequential physical scheduling, pair RNG known "
        "answers/order invariance, and no GPU sampling"
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
    allocation = logical_assignments(classes)
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
    parser.add_argument(
        "--logical-workers",
        type=parse_logical_workers,
        default=parse_logical_workers("0,1,2,3"),
        help=(
            "fixed logical shard subset to execute; e.g. 0,2.  This never changes "
            "the shard definitions or pair RNG"
        ),
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=parse_gpus("0,1,2,3"),
        help=(
            "one to four currently free physical CUDA device IDs; multiple logical "
            "workers assigned to the same device run sequentially"
        ),
    )
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
    if protocol.get("execution_ready") is not True:
        raise RuntimeError(
            "this design-only source lock does not authorize real endpoint sampling; "
            "complete the independent review qualification/execution chain and freeze a "
            "separate execution-ready authorization before launching GPUs"
        )
    output_root = validate_output_root(args.output_root)
    checkpoint_arg = args.checkpoint or args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
    dit_root, checkpoint, vae = validate_assets(
        protocol, source_lock, args.dit_root, checkpoint_arg, args.vae_snapshot
    )
    plan = build_plan(
        protocol,
        source_manifest,
        classes,
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

    allocation = logical_assignments(classes)
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

    already_complete = completed_logical_shards(
        output_root, classes, source_lock, protocol, plan_path
    )
    requested_pending = tuple(
        worker for worker in args.logical_workers if worker not in already_complete
    )
    queues = physical_queues(requested_pending, args.gpus) if requested_pending else {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(queues))) as executor:
        futures = [
            executor.submit(
                run_physical_gpu_queue,
                gpu,
                logical_queue,
                task_paths,
                source_lock,
                output_root,
                dit_root,
                checkpoint,
                vae,
            )
            for gpu, logical_queue in queues.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    for logical_worker in args.logical_workers:
        publish_or_validate_logical_shard_receipt(
            output_root,
            logical_worker,
            classes,
            source_lock,
            protocol,
            plan_path,
        )
    logical_receipts = completed_logical_shards(
        output_root, classes, source_lock, protocol, plan_path
    )
    if not all_logical_shards_complete(logical_receipts):
        print(
            json.dumps(
                {
                    "complete": False,
                    "pool_receipts_published": False,
                    "completed_logical_workers": list(logical_receipts),
                    "remaining_logical_workers": [
                        index for index in LOGICAL_WORKERS if index not in logical_receipts
                    ],
                    "note": (
                        "selected logical shards validated; full pair-axis validation and "
                        "pool receipts are intentionally deferred until all four logical "
                        "shard receipts exist"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    records = validate_all_pairs(output_root, classes, source_lock, protocol)
    completion = publish_pool_receipts(
        output_root, plan_path, protocol, records, logical_receipts
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
