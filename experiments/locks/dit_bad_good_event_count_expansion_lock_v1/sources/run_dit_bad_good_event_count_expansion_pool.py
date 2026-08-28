#!/usr/bin/env python3
"""Run the frozen 360-trajectory, event-count-only DiT expansion cohort.

Exactly four GPU workers process global seeds 130..249.  Each seed generates
the ordered class batch 207,602,795 with the hash-frozen observation-only
250-step ancestral DDPM trace source.  No score, threshold, alert, visual label,
or intervention code is imported or read by this runner.

Nothing is overwritten.  A completed seed may be reused only after every bound
payload hash and its scientific identity are revalidated.  Partial or changed
outputs are refused.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


def find_repo_root(source: Path) -> Path:
    for candidate in source.resolve().parents:
        if (candidate / ".git").exists() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate repository root from {source}")


ROOT = find_repo_root(Path(__file__))
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
DEFAULT_LOCK_ROOT = (
    ROOT / "experiments/locks/dit_bad_good_event_count_expansion_lock_v1"
)
DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/dit_bad_good_confirmation_expansion_v1_custom_traces_cfg_locked"
)
DEFAULT_DIT_ROOT = DATA_ROOT / "baselines/DiT"
CHECKPOINT_FILENAME = "DiT-XL-2-256x256.pt"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
DEFAULT_VAE = (
    Path.home()
    / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
    / VAE_REVISION
)

EXPECTED_CANDIDATE_IDENTITY = (
    "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"
)
EXPECTED_EVENT_RESULT_IDENTITY = (
    "4791dafe591823b22c3b89aaca5bcf287a06493cc65b13be740e81c389a88e31"
)
EXPECTED_CONSENSUS_IDENTITY = (
    "21c242dc796d5c8baa4568c9f82add0d1b64c984477cf8698efbbca5889e166a"
)
CLASSES = (207, 602, 795)
SEEDS = tuple(range(130, 250))
EXPECTED_OUTPUT_RELATIVE_PATHS = {
    "sample.png",
    "images/00_class0207.png",
    "images/01_class0602.png",
    "images/02_class0795.png",
    "trace.npz",
    "runner_source.py",
    "custom_baseline_helper.py",
    "strict_reproduction_helper.py",
}
SOURCE_BASENAMES = (
    "trace_dit_imagenet256_custom_batch.py",
    "sample_dit_imagenet256_custom.py",
    "reproduce_dit_imagenet256.py",
    "run_dit_bad_good_event_count_expansion_pool.py",
    "freeze_dit_bad_good_event_count_expansion.py",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
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


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def validate_lock(lock_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_root = lock_root.expanduser().absolute()
    if not lock_root.is_dir() or lock_root.is_symlink():
        raise RuntimeError(f"expansion lock must be a real directory: {lock_root}")
    protocol_path = lock_root / "expansion_protocol.json"
    manifest_path = lock_root / "manifest.json"
    completion_path = lock_root / "completion.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = protocol.get("identity_sha256")
    if not isinstance(identity, str) or canonical_sha256(without_identity(protocol)) != identity:
        raise RuntimeError("expansion protocol canonical identity is invalid")
    if (
        completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != identity
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != identity
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
    ):
        raise RuntimeError("expansion lock manifest/completion validation failed")

    listed = manifest.get("files")
    if not isinstance(listed, list) or not all(isinstance(item, dict) for item in listed):
        raise RuntimeError("expansion lock manifest member list is malformed")
    expected_names = {"expansion_protocol.json", *(f"sources/{x}" for x in SOURCE_BASENAMES)}
    observed_names = {item.get("name") for item in listed}
    if observed_names != expected_names or len(listed) != len(expected_names):
        raise RuntimeError("expansion lock member set changed")
    for item in listed:
        relative = str(item["name"])
        path = lock_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"lock member missing or indirect: {path}")
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            raise RuntimeError(f"lock member hash/size changed: {path}")

    scientific = protocol.get("scientific_contract", {})
    cohort = protocol.get("expansion_cohort", {})
    selection = protocol.get("selection_basis", {})
    audit = protocol.get("evidence_access_audit", {})
    if (
        protocol.get("status")
        != "FROZEN_AFTER_EVENT_COUNT_ONLY_GATE_BEFORE_EXPANSION_SAMPLING_OR_SCORE_ACCESS"
        or scientific.get("model") != "DiT-XL/2 ImageNet-256"
        or scientific.get("sampler") != "official 250-step ancestral DDPM"
        or scientific.get("sampling_steps") != 250
        or scientific.get("cfg_scale") != 4.0
        or scientific.get("cfg_epsilon_channels") != 3
        or tuple(scientific.get("classes_ordered", ())) != CLASSES
        or scientific.get("candidate_formula") != "max(z_A, z_B)"
        or tuple(cohort.get("global_seeds", ())) != SEEDS
        or cohort.get("global_seed_count") != 120
        or cohort.get("trajectory_count") != 360
        or selection.get("original_locked_clear_bad_events") != 8
        or selection.get("additional_events_needed") != 7
        or selection.get("detector_scores_thresholds_alerts_or_score_label_join_used")
        is not False
        or audit.get("candidate_score_files_opened") is not False
        or audit.get("calibration_threshold_members_opened") is not False
        or audit.get("evaluation_alert_files_opened") is not False
        or audit.get("sample_level_label_score_mapping_opened") is not False
    ):
        raise RuntimeError("expansion protocol scientific/blinding contract changed")

    lineage = protocol.get("input_lineage", {})
    candidate = lineage.get("candidate_v5", {})
    event = lineage.get("event_count_only_result", {})
    consensus = lineage.get("final_visual_consensus_aggregate", {})
    if (
        candidate.get("protocol_identity_sha256") != EXPECTED_CANDIDATE_IDENTITY
        or event.get("result_identity_sha256") != EXPECTED_EVENT_RESULT_IDENTITY
        or event.get("evidence_access_audit", {}).get("score_label_join_performed")
        is not False
        or consensus.get("consensus_identity_sha256") != EXPECTED_CONSENSUS_IDENTITY
        or consensus.get("aggregate_counts", {}).get("clear_bad") != 8
        or consensus.get("sample_level_consensus_decoded_by_this_locker") is not False
    ):
        raise RuntimeError("expansion input lineage changed")

    external_files = (
        (candidate, "candidate_protocol.json", "protocol_file_sha256"),
        (candidate, "manifest.json", "manifest_file_sha256"),
        (candidate, "completion.json", "completion_file_sha256"),
        (event, "confirmation_results.json", "result_file_sha256"),
        (event, "manifest.json", "manifest_file_sha256"),
        (event, "completion.json", "completion_file_sha256"),
        (consensus, "consensus_locked.json", "consensus_file_sha256"),
        (consensus, "manifest.json", "manifest_file_sha256"),
        (consensus, "completion.json", "completion_file_sha256"),
    )
    for record, basename, hash_key in external_files:
        root = Path(str(record.get("path", "")))
        path = root / basename
        if not path.is_file() or path.is_symlink() or sha256_file(path) != record.get(hash_key):
            raise RuntimeError(f"frozen external input changed: {path}")

    source_records = protocol.get("source_snapshots", {})
    if set(source_records) != set(SOURCE_BASENAMES):
        raise RuntimeError("source snapshot name set changed")
    for basename in SOURCE_BASENAMES:
        snapshot = lock_root / "sources" / basename
        if sha256_file(snapshot) != source_records[basename].get("sha256"):
            raise RuntimeError(f"source snapshot hash changed: {snapshot}")
    if sha256_file(Path(__file__).resolve()) != source_records[Path(__file__).name]["sha256"]:
        raise RuntimeError("the invoked expansion runner differs from its frozen snapshot")
    return protocol, manifest


def parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(result) != 4 or len(set(result)) != 4:
        raise argparse.ArgumentTypeError("--gpus must list exactly four unique device identifiers")
    return result


def assignments(gpus: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    if len(gpus) != 4:
        raise ValueError("exactly four GPUs are required")
    result = {gpu: SEEDS[index::4] for index, gpu in enumerate(gpus)}
    if sorted(seed for values in result.values() for seed in values) != list(SEEDS):
        raise AssertionError("GPU assignment lost or duplicated a seed")
    if any(len(values) != 30 for values in result.values()):
        raise AssertionError("the frozen 120-seed cohort must split 30 seeds per GPU")
    return result


def require_real_input(path: Path, description: str, *, directory: bool) -> Path:
    raw = path.expanduser().absolute()
    if os.path.lexists(raw) and raw.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink: {raw}")
    resolved = raw.resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        raise RuntimeError(f"missing {description}: {resolved}")
    return resolved


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
    lock_manifest: Mapping[str, Any],
    lock_root: Path,
    gpus: tuple[str, ...],
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
) -> dict[str, Any]:
    allocation = assignments(gpus)
    trace_source = lock_root / "sources/trace_dit_imagenet256_custom_batch.py"
    return {
        "schema_version": 1,
        "status": "FROZEN_FOUR_GPU_EXECUTION_PLAN",
        "expansion_protocol_identity_sha256": protocol["identity_sha256"],
        "expansion_lock_manifest_identity_sha256": lock_manifest["identity_sha256"],
        "candidate_protocol_identity_sha256": EXPECTED_CANDIDATE_IDENTITY,
        "event_count_only_result_identity_sha256": EXPECTED_EVENT_RESULT_IDENTITY,
        "trace_source": str(trace_source),
        "trace_source_sha256": sha256_file(trace_source),
        "classes_ordered": list(CLASSES),
        "global_seeds": list(SEEDS),
        "global_seed_count": len(SEEDS),
        "trajectory_count": len(SEEDS) * len(CLASSES),
        "gpus": list(gpus),
        "assignment": {gpu: list(values) for gpu, values in allocation.items()},
        "output_root": str(output_root),
        "dit_root": str(dit_root),
        "checkpoint": str(checkpoint),
        "vae_snapshot": str(vae_snapshot),
        "observation_only": True,
        "scores_thresholds_alerts_labels_or_interventions_read": False,
    }


def safe_payload_path(outdir: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError(f"unsafe trace payload path: {relative!r}")
    path = outdir / relative
    if path.resolve().parent != outdir.resolve() and outdir.resolve() not in path.resolve().parents:
        raise RuntimeError(f"trace payload escapes seed directory: {relative!r}")
    return path


def validate_seed_output(
    seed: int,
    outdir: Path,
    protocol: Mapping[str, Any],
    lock_root: Path,
) -> dict[str, Any]:
    if not outdir.is_dir() or outdir.is_symlink():
        raise RuntimeError(f"seed output is missing or indirect: {outdir}")
    manifest_path = outdir / "manifest.json"
    completion_path = outdir / "completion.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"partial seed output cannot be overwritten: {outdir}")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"seed manifest lacks scientific identity: {outdir}")
    identity_hash = canonical_sha256(identity)
    if (
        manifest.get("status") != "complete"
        or manifest.get("identity_sha256") != identity_hash
        or completion.get("identity_sha256") != identity_hash
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or completion.get("outputs_sha256") != manifest.get("outputs_sha256")
        or completion.get("output_count") != len(EXPECTED_OUTPUT_RELATIVE_PATHS)
    ):
        raise RuntimeError(f"seed completion/identity validation failed: {outdir}")
    trace_protocol = identity.get("protocol", {})
    if (
        identity.get("runner") != "trace_dit_imagenet256_custom_batch"
        or identity.get("observation_only") is not True
        or identity.get("quality_score") is not None
        or identity.get("selection") is not None
        or identity.get("intervention") is not None
        or tuple(trace_protocol.get("class_ids_ordered", ())) != CLASSES
        or trace_protocol.get("sampling_steps") != 250
        or trace_protocol.get("cfg_scale") != 4.0
        or trace_protocol.get("cfg_epsilon_channels") != 3
        or trace_protocol.get("global_torch_seed") != seed
        or trace_protocol.get("internal_timestep_order") != "249..0"
        or trace_protocol.get("raw_cfg_components_observed_from_same_model_forward")
        is not True
    ):
        raise RuntimeError(f"seed scientific contract changed: {outdir}")
    source_records = protocol["source_snapshots"]
    identity_source_pairs = {
        "runner_source": "trace_dit_imagenet256_custom_batch.py",
        "custom_baseline_helper": "sample_dit_imagenet256_custom.py",
        "strict_reproduction_helper": "reproduce_dit_imagenet256.py",
    }
    for identity_key, basename in identity_source_pairs.items():
        if identity.get(identity_key, {}).get("sha256") != source_records[basename]["sha256"]:
            raise RuntimeError(f"seed source identity changed: {identity_key}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(item, dict) for item in outputs):
        raise RuntimeError(f"seed manifest output list is malformed: {outdir}")
    names = {item.get("relative_path") for item in outputs}
    if names != EXPECTED_OUTPUT_RELATIVE_PATHS or len(outputs) != len(names):
        raise RuntimeError(f"seed payload set changed: {outdir}")
    for item in outputs:
        path = safe_payload_path(outdir, str(item["relative_path"]))
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"seed payload missing or indirect: {path}")
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            raise RuntimeError(f"seed payload hash/size changed: {path}")
    if canonical_sha256(outputs) != manifest.get("outputs_sha256"):
        raise RuntimeError(f"seed output aggregate hash changed: {outdir}")
    for snapshot_name, basename in {
        "runner_source.py": "trace_dit_imagenet256_custom_batch.py",
        "custom_baseline_helper.py": "sample_dit_imagenet256_custom.py",
        "strict_reproduction_helper.py": "reproduce_dit_imagenet256.py",
    }.items():
        if sha256_file(outdir / snapshot_name) != sha256_file(lock_root / "sources" / basename):
            raise RuntimeError(f"seed source snapshot differs from expansion lock: {outdir}")
    return {
        "seed": seed,
        "relative_output": outdir.name,
        "identity_sha256": identity_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "completion_sha256": sha256_file(completion_path),
        "outputs_sha256": manifest["outputs_sha256"],
        "output_count": completion["output_count"],
    }


def run_seed(
    seed: int,
    gpu: str,
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
    lock_root: Path,
    protocol: Mapping[str, Any],
    print_lock: threading.Lock,
) -> dict[str, Any]:
    outdir = output_root / f"expansion_v1_seed{seed:03d}"
    if os.path.lexists(outdir):
        record = validate_seed_output(seed, outdir, protocol, lock_root)
        with print_lock:
            print(json.dumps({"seed": seed, "gpu": gpu, "reused": True}), flush=True)
        return record

    logs = output_root / "_runner_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if logs.is_symlink():
        raise RuntimeError(f"runner log directory must not be a symlink: {logs}")
    log_path = logs / f"seed{seed:03d}.log"
    if os.path.lexists(log_path):
        raise RuntimeError(f"refusing to overwrite prior seed log: {log_path}")
    trace_source = lock_root / "sources/trace_dit_imagenet256_custom_batch.py"
    command = [
        sys.executable,
        str(trace_source),
        "--classes",
        ",".join(str(value) for value in CLASSES),
        "--seed",
        str(seed),
        "--dit-root",
        str(dit_root),
        "--checkpoint",
        str(checkpoint),
        "--vae-snapshot",
        str(vae_snapshot),
        "--outdir",
        str(outdir),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.time()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {"started_unix": started, "gpu": gpu, "command": command},
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
        raise RuntimeError(f"seed {seed} failed on GPU {gpu}; inspect {log_path}")
    record = validate_seed_output(seed, outdir, protocol, lock_root)
    with print_lock:
        print(
            json.dumps(
                {
                    "seed": seed,
                    "gpu": gpu,
                    "elapsed_seconds": round(finished - started, 3),
                    "reused": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return record


def worker(
    gpu: str,
    seeds: tuple[int, ...],
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
    lock_root: Path,
    protocol: Mapping[str, Any],
    print_lock: threading.Lock,
) -> list[dict[str, Any]]:
    return [
        run_seed(
            seed,
            gpu,
            output_root,
            dit_root,
            checkpoint,
            vae_snapshot,
            lock_root,
            protocol,
            print_lock,
        )
        for seed in seeds
    ]


def log_records(output_root: Path) -> list[dict[str, Any]]:
    logs = output_root / "_runner_logs"
    if not logs.exists():
        return []
    if not logs.is_dir() or logs.is_symlink():
        raise RuntimeError("runner logs path changed")
    records = []
    for path in sorted(logs.iterdir()):
        if not path.is_file() or path.is_symlink() or path.suffix != ".log":
            raise RuntimeError(f"unexpected runner log entry: {path}")
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_complete_pool(
    output_root: Path,
    expected_plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    lock_root: Path,
) -> dict[str, Any]:
    plan_path = output_root / "execution_plan.json"
    manifest_path = output_root / "pool_manifest.json"
    completion_path = output_root / "pool_completion.json"
    if load_json(plan_path) != expected_plan:
        raise RuntimeError("completed pool execution plan differs from this invocation")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    records = [
        validate_seed_output(
            seed, output_root / f"expansion_v1_seed{seed:03d}", protocol, lock_root
        )
        for seed in SEEDS
    ]
    if (
        manifest.get("status") != "complete"
        or manifest.get("expansion_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or manifest.get("seed_count") != len(SEEDS)
        or manifest.get("trajectory_count") != len(SEEDS) * len(CLASSES)
        or manifest.get("seed_outputs") != records
        or manifest.get("runner_logs") != log_records(output_root)
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or completion.get("complete") is not True
        or completion.get("pool_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("pool_manifest_sha256") != sha256_file(manifest_path)
        or completion.get("execution_plan_sha256") != sha256_file(plan_path)
        or completion.get("seed_count") != len(SEEDS)
        or completion.get("trajectory_count") != len(SEEDS) * len(CLASSES)
    ):
        raise RuntimeError("completed expansion pool failed full validation")
    return completion


def publish_pool_receipt(
    output_root: Path,
    plan_path: Path,
    records: list[dict[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output_root / "pool_manifest.json"
    completion_path = output_root / "pool_completion.json"
    if os.path.lexists(manifest_path) or os.path.lexists(completion_path):
        raise RuntimeError("refusing to overwrite a prior pool receipt")
    records = sorted(records, key=lambda item: int(item["seed"]))
    if [item["seed"] for item in records] != list(SEEDS):
        raise RuntimeError("pool result does not contain each frozen seed exactly once")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "expansion_protocol_identity_sha256": protocol["identity_sha256"],
        "candidate_protocol_identity_sha256": EXPECTED_CANDIDATE_IDENTITY,
        "event_count_only_result_identity_sha256": EXPECTED_EVENT_RESULT_IDENTITY,
        "execution_plan_sha256": sha256_file(plan_path),
        "seed_count": len(records),
        "trajectory_count": len(records) * len(CLASSES),
        "seed_outputs": records,
        "runner_logs": log_records(output_root),
        "observation_only": True,
        "scores_thresholds_alerts_labels_or_interventions_read": False,
    }
    manifest["identity_sha256"] = canonical_sha256(manifest)
    exclusive_json(manifest_path, manifest)
    completion = {
        "complete": True,
        "pool_identity_sha256": manifest["identity_sha256"],
        "pool_manifest_sha256": sha256_file(manifest_path),
        "execution_plan_sha256": sha256_file(plan_path),
        "seed_count": len(records),
        "trajectory_count": len(records) * len(CLASSES),
    }
    exclusive_json(completion_path, completion)
    return completion


def run_self_test(lock_root: Path) -> None:
    protocol, manifest = validate_lock(lock_root)
    assert parse_gpus("0,1,2,3") == ("0", "1", "2", "3")
    allocation = assignments(("0", "1", "2", "3"))
    assert all(len(values) == 30 for values in allocation.values())
    assert sorted(seed for values in allocation.values() for seed in values) == list(SEEDS)
    assert set(protocol["source_snapshots"]) == set(SOURCE_BASENAMES)
    assert manifest["protocol_identity_sha256"] == protocol["identity_sha256"]
    try:
        parse_gpus("0,1,2")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("three-GPU plan was accepted")
    print(
        "self-test passed: immutable lock, source/external hashes, four-GPU split, "
        "disjoint 120 seeds, and score-blind contract"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-root", type=Path, default=DEFAULT_LOCK_ROOT)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path, default=DEFAULT_VAE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lock_root = args.lock_root.expanduser().absolute()
    if args.self_test:
        run_self_test(lock_root)
        return 0
    protocol, lock_manifest = validate_lock(lock_root)
    output_root = validate_output_root(args.output_root)
    dit_root = require_real_input(args.dit_root, "DiT repository", directory=True)
    checkpoint = require_real_input(
        args.checkpoint if args.checkpoint is not None else dit_root / "pretrained_models" / CHECKPOINT_FILENAME,
        "DiT checkpoint",
        directory=False,
    )
    vae_snapshot = require_real_input(args.vae_snapshot, "VAE snapshot", directory=True)
    plan = build_plan(
        protocol,
        lock_manifest,
        lock_root,
        args.gpus,
        output_root,
        dit_root,
        checkpoint,
        vae_snapshot,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    plan_path = output_root / "execution_plan.json"
    if plan_path.exists():
        if plan_path.is_symlink() or load_json(plan_path) != plan:
            raise RuntimeError("existing execution plan differs; refusing overwrite")
    else:
        if any(output_root.iterdir()):
            raise RuntimeError("nonempty output root lacks a frozen execution plan")
        exclusive_json(plan_path, plan)

    pool_manifest = output_root / "pool_manifest.json"
    pool_completion = output_root / "pool_completion.json"
    if pool_completion.exists():
        completion = validate_complete_pool(output_root, plan, protocol, lock_root)
        print(json.dumps({**completion, "reused_complete_pool": True}, indent=2, sort_keys=True))
        return 0
    if pool_manifest.exists():
        raise RuntimeError("partial pool receipt exists; refusing overwrite")

    print_lock = threading.Lock()
    allocation = assignments(args.gpus)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                worker,
                gpu,
                seeds,
                output_root,
                dit_root,
                checkpoint,
                vae_snapshot,
                lock_root,
                protocol,
                print_lock,
            ): gpu
            for gpu, seeds in allocation.items()
        }
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    completion = publish_pool_receipt(output_root, plan_path, records, protocol)
    validated = validate_complete_pool(output_root, plan, protocol, lock_root)
    if completion != validated:
        raise RuntimeError("newly written pool receipt did not round-trip")
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
