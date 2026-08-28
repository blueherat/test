#!/usr/bin/env python3
"""Produce the frozen 1,800-trajectory third DiT confirmation pool.

Four GPU workers receive four *contiguous* 150-seed blocks.  Each global seed
generates the ordered class batch 207,602,795 with the hash-frozen,
observation-only DiT trace runner.  The launcher reads no visual labels,
reviews, screen results, or sample scores, and performs no score/label join.

Completed seeds are reusable only after their scientific identity and every
payload hash are revalidated.  An incomplete or changed seed directory is
preserved and refused (fail closed); no output is overwritten.
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
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping


def find_repo_root(source: Path) -> Path:
    for candidate in source.resolve().parents:
        if (candidate / ".git").exists() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate repository root from {source}")


ROOT = find_repo_root(Path(__file__))
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
DEFAULT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_sampling_source_lock_v1"
)
DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT
    / "cross_scale_evidence/dit_bad_good_third_pool_v1_custom_traces_cfg_locked"
)
DEFAULT_DIT_ROOT = DATA_ROOT / "baselines/DiT"
CHECKPOINT_FILENAME = "DiT-XL-2-256x256.pt"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
DEFAULT_VAE = (
    Path.home()
    / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
    / VAE_REVISION
)

EXPECTED_PROTOCOL_IDENTITY = (
    "0788c7074adc55f1896dea4e0626f57c8b2b4899a5b600f38d2f982a87acfed5"
)
EXPECTED_PROTOCOL_MANIFEST_IDENTITY = (
    "931c928054da332dd83501e09a69793b5217596609e085b441b7cb993a325aa2"
)
EXPECTED_THRESHOLD_IDENTITY = (
    "c89fee87731968aa0c8a7ef086cb9a95a578dc3462149a6135bb71275bdbe43d"
)
EXPECTED_THRESHOLD_MANIFEST_IDENTITY = (
    "6b2c117b0dcc2eb3e2be71e1f4838ffe8b56206c8ffa0a557a6868692a732fb4"
)
BLUR_FEATURE = "decoded_local_blur_severity__mean"
C3_FEATURE = "pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump"
CLASSES = (207, 602, 795)
SEEDS = tuple(range(250, 850))
WORKER_COUNT = 4
SEEDS_PER_WORKER = 150
TRACE_ARRAY_NAMES = {
    "state_before",
    "pred_xstart",
    "p_mean",
    "p_standard_deviation",
    "transition_innovation",
    "conditional_epsilon_raw",
    "unconditional_epsilon_raw",
    "conditional_variance_values_raw",
    "unconditional_variance_values_raw",
    "final_latents",
    "decoded_images",
    "internal_timestep",
    "alpha_bar",
}
STEP_ARRAY_NAMES = TRACE_ARRAY_NAMES - {
    "final_latents",
    "decoded_images",
    "internal_timestep",
    "alpha_bar",
}
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
    "run_dit_bad_good_third_pool.py",
    "freeze_dit_bad_good_third_pool_sampling_sources.py",
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


def artifact_records(root: Path, *, recursive: bool) -> list[dict[str, Any]]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    records: list[dict[str, Any]] = []
    for path in sorted(iterator):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"lock artifact must not be a symlink: {path}")
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def require_exact_tree(
    root: Path, *, expected_files: set[str], expected_directories: set[str]
) -> None:
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"lock tree contains a symlink: {path}")
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise RuntimeError(f"lock tree contains a non-file/non-directory: {path}")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise RuntimeError(
            "lock tree member set changed: "
            f"missing_files={sorted(expected_files - observed_files)}, "
            f"extra_files={sorted(observed_files - expected_files)}, "
            f"missing_dirs={sorted(expected_directories - observed_directories)}, "
            f"extra_dirs={sorted(observed_directories - expected_directories)}"
        )


def validate_phase1_protocol_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "third-pool protocol lock")
    protocol_snapshot_names = {
        "custom_sampler_helper.py",
        "primary_feature_extractor.py",
        "strict_reproduction_helper.py",
        "trace_runner.py",
        "visual_feature_extractor.py",
    }
    require_exact_tree(
        root,
        expected_files={
            "completion.json",
            "locker_source.py",
            "manifest.json",
            "third_pool_protocol.json",
            *(f"source_snapshots/{name}" for name in protocol_snapshot_names),
        },
        expected_directories={"source_snapshots"},
    )
    protocol_path = require_regular(root / "third_pool_protocol.json", "protocol")
    manifest_path = require_regular(root / "manifest.json", "protocol manifest")
    completion_path = require_regular(root / "completion.json", "protocol completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = protocol.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    pool = protocol.get("third_pool", {})
    family = protocol.get("co_primary_family", {})
    if (
        identity != EXPECTED_PROTOCOL_IDENTITY
        or canonical_sha256(without_identity(protocol)) != identity
        or manifest_identity != EXPECTED_PROTOCOL_MANIFEST_IDENTITY
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != identity
        or manifest.get("files") != artifact_records(root, recursive=True)
        or completion.get("complete") is not True
        or completion.get("protocol_identity_sha256") != identity
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or protocol.get("status")
        != "FROZEN_BEFORE_THIRD_POOL_SAMPLING_OR_VISUAL_REVIEW"
        or tuple(pool.get("classes_ordered", ())) != CLASSES
        or tuple(pool.get("global_seeds", ())) != SEEDS
        or pool.get("global_seed_count") != len(SEEDS)
        or pool.get("trajectory_count") != len(SEEDS) * len(CLASSES)
        or pool.get("model") != "DiT-XL/2 ImageNet-256"
        or pool.get("sampler") != "official 250-step ancestral DDPM"
        or pool.get("sampling_steps") != 250
        or pool.get("cfg_scale") != 4.0
        or pool.get("cfg_epsilon_channels") != 3
        or family.get("family_size") != 2
        or tuple(family.get("candidate_ids", ()))
        != ("B_blur_mean", "C_c3_low_jump")
        or family.get("combination_allowed") is not False
    ):
        raise RuntimeError("third-pool phase-1 protocol lock validation failed")
    candidates = protocol.get("candidates", {})
    blur = candidates.get("B_blur_mean", {})
    c3 = candidates.get("C_c3_low_jump", {})
    if (
        set(candidates) != {"B_blur_mean", "C_c3_low_jump"}
        or blur.get("feature") != BLUR_FEATURE
        or blur.get("raw_orientation") != "bad_high"
        or blur.get("primary_endpoint")
        != "blur_or_soft_fusion_clear_bad_vs_clean_good"
        or blur.get("latest_required_sampling_step") != 149
        or c3.get("feature") != C3_FEATURE
        or c3.get("raw_orientation") != "bad_low"
        or c3.get("primary_endpoint") != "all_clear_bad_vs_clean_good"
        or c3.get("latest_required_sampling_step") != 149
    ):
        raise RuntimeError("third-pool phase-1 candidate contract changed")
    return protocol, {
        "path": str(root),
        "identity_sha256": identity,
        "protocol_file_sha256": sha256_file(protocol_path),
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "completion_file_sha256": sha256_file(completion_path),
    }


def validate_phase1_threshold_lock(
    root: Path, protocol_identity: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "third-pool threshold lock")
    require_exact_tree(
        root,
        expected_files={
            "calibrator_source.py",
            "completion.json",
            "manifest.json",
            "thresholds_locked.json",
        },
        expected_directories=set(),
    )
    threshold_path = require_regular(root / "thresholds_locked.json", "thresholds")
    manifest_path = require_regular(root / "manifest.json", "threshold manifest")
    completion_path = require_regular(root / "completion.json", "threshold completion")
    record = load_json(threshold_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = record.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    if (
        identity != EXPECTED_THRESHOLD_IDENTITY
        or record.get("third_pool_protocol_identity_sha256") != protocol_identity
        or canonical_sha256(without_identity(record)) != identity
        or manifest_identity != EXPECTED_THRESHOLD_MANIFEST_IDENTITY
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("threshold_identity_sha256") != identity
        or manifest.get("third_pool_protocol_identity_sha256") != protocol_identity
        or manifest.get("files") != artifact_records(root, recursive=False)
        or completion.get("complete") is not True
        or completion.get("threshold_identity_sha256") != identity
        or completion.get("threshold_file_sha256") != sha256_file(threshold_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("third-pool phase-1 threshold lock validation failed")
    if record.get("status") != "FROZEN_LABEL_FREE_BEFORE_THIRD_POOL_SAMPLING_OR_SCORE_LABEL_JOIN":
        raise RuntimeError("third-pool threshold status changed")
    thresholds = record.get("thresholds", {})
    expected_candidates = {
        "B_blur_mean": (BLUR_FEATURE, "bad_high", "upper", ">"),
        "C_c3_low_jump": (C3_FEATURE, "bad_low", "lower", "<"),
    }
    if set(thresholds) != set(expected_candidates):
        raise RuntimeError("third-pool threshold candidate family changed")
    for candidate, (feature, orientation, tail, operator) in expected_candidates.items():
        candidate_record = thresholds[candidate]
        if (
            candidate_record.get("feature") != feature
            or candidate_record.get("raw_orientation") != orientation
            or set(candidate_record.get("classes", {}))
            != {str(value) for value in CLASSES}
        ):
            raise RuntimeError(f"third-pool threshold feature/orientation changed: {candidate}")
        for class_id in CLASSES:
            class_record = candidate_record["classes"][str(class_id)]
            if set(class_record) != {
                "alpha_0p10",
                "alpha_0p05",
                "calibration_values_ordered_by_seed_sha256",
            }:
                raise RuntimeError(f"threshold class record changed: {candidate}/{class_id}")
            for alpha_key, order, numerator in (
                ("alpha_0p10", 19 if tail == "upper" else 2, 2),
                ("alpha_0p05", 20 if tail == "upper" else 1, 1),
            ):
                row = class_record[alpha_key]
                threshold = row.get("threshold")
                if (
                    set(row)
                    != {
                        "calibration_count",
                        "calibration_order_statistic_1_based",
                        "finite_sample_bound_fraction",
                        "finite_sample_marginal_trigger_probability_upper_bound",
                        "strict_comparison",
                        "tail",
                        "threshold",
                    }
                    or row.get("calibration_count") != 20
                    or row.get("calibration_order_statistic_1_based") != order
                    or row.get("tail") != tail
                    or row.get("strict_comparison")
                    != f"third_pool_raw_score {operator} threshold"
                    or row.get("finite_sample_bound_fraction") != f"{numerator}/21"
                    or row.get("finite_sample_marginal_trigger_probability_upper_bound")
                    != numerator / 21
                    or not isinstance(threshold, (int, float))
                    or not float("-inf") < float(threshold) < float("inf")
                ):
                    raise RuntimeError(
                        f"threshold comparison/bound changed: {candidate}/{class_id}/{alpha_key}"
                    )
    return record, {
        "path": str(root),
        "identity_sha256": identity,
        "threshold_file_sha256": sha256_file(threshold_path),
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "completion_file_sha256": sha256_file(completion_path),
    }


def validate_source_lock(lock_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_root = require_directory(lock_root, "third-pool sampling source lock")
    require_exact_tree(
        lock_root,
        expected_files={
            "completion.json",
            "manifest.json",
            "sampling_protocol.json",
            *(f"sources/{name}" for name in SOURCE_BASENAMES),
        },
        expected_directories={"sources"},
    )
    protocol_path = require_regular(lock_root / "sampling_protocol.json", "sampling protocol")
    manifest_path = require_regular(lock_root / "manifest.json", "sampling manifest")
    completion_path = require_regular(lock_root / "completion.json", "sampling completion")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = protocol.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    expected_members = {"sampling_protocol.json", *(f"sources/{x}" for x in SOURCE_BASENAMES)}
    listed = manifest.get("files")
    if (
        not isinstance(identity, str)
        or canonical_sha256(without_identity(protocol)) != identity
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256") != identity
        or not isinstance(listed, list)
        or {row.get("name") for row in listed if isinstance(row, dict)} != expected_members
        or len(listed) != len(expected_members)
        or manifest.get("files") != artifact_records(lock_root, recursive=True)
        or completion.get("complete") is not True
        or completion.get("sampling_protocol_identity_sha256") != identity
        or completion.get("sampling_protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("third-pool sampling source lock validation failed")

    scientific = protocol.get("scientific_contract", {})
    execution = protocol.get("execution_contract", {})
    required = protocol.get("required_trace_arrays", ())
    if (
        protocol.get("status") != "FROZEN_BEFORE_THIRD_POOL_GPU_SAMPLING"
        or protocol.get("phase1_protocol", {}).get("identity_sha256")
        != EXPECTED_PROTOCOL_IDENTITY
        or protocol.get("phase1_protocol", {}).get("manifest_identity_sha256")
        != EXPECTED_PROTOCOL_MANIFEST_IDENTITY
        or protocol.get("phase1_thresholds", {}).get("identity_sha256")
        != EXPECTED_THRESHOLD_IDENTITY
        or protocol.get("phase1_thresholds", {}).get("manifest_identity_sha256")
        != EXPECTED_THRESHOLD_MANIFEST_IDENTITY
        or scientific.get("model") != "DiT-XL/2 ImageNet-256"
        or scientific.get("sampler") != "official 250-step ancestral DDPM"
        or scientific.get("sampling_steps") != 250
        or scientific.get("cfg_scale") != 4.0
        or scientific.get("cfg_epsilon_channels") != 3
        or tuple(scientific.get("classes_ordered", ())) != CLASSES
        or tuple(scientific.get("global_seeds", ())) != SEEDS
        or scientific.get("trajectory_count") != 1800
        or execution.get("gpu_worker_count") != WORKER_COUNT
        or execution.get("seeds_per_gpu") != SEEDS_PER_WORKER
        or execution.get("assignment_kind") != "four ordered contiguous seed blocks"
        or set(required) != TRACE_ARRAY_NAMES
        or protocol.get("evidence_access_audit", {}).get("labels_opened") is not False
        or protocol.get("evidence_access_audit", {}).get("screen_results_opened")
        is not False
        or protocol.get("evidence_access_audit", {}).get("sample_scores_opened")
        is not False
        or protocol.get("evidence_access_audit", {}).get("score_label_join_performed")
        is not False
    ):
        raise RuntimeError("sampling source scientific/blinding contract changed")

    source_records = protocol.get("source_snapshots", {})
    if set(source_records) != set(SOURCE_BASENAMES):
        raise RuntimeError("sampling source snapshot set changed")
    for basename in SOURCE_BASENAMES:
        snapshot = require_regular(lock_root / "sources" / basename, f"source {basename}")
        if sha256_file(snapshot) != source_records[basename].get("sha256"):
            raise RuntimeError(f"sampling source snapshot changed: {snapshot}")
    invoked = Path(__file__).resolve()
    if sha256_file(invoked) != source_records[invoked.name].get("sha256"):
        raise RuntimeError("invoked launcher differs from its frozen source snapshot")

    imported = protocol.get("imported_helper_sha256", {})
    expected_imported = {
        "source_locker_imports_launcher": source_records[
            "run_dit_bad_good_third_pool.py"
        ]["sha256"],
        "launcher_imports_strict_reproduction": source_records[
            "reproduce_dit_imagenet256.py"
        ]["sha256"],
        "trace_imports_custom_sampler": source_records[
            "sample_dit_imagenet256_custom.py"
        ]["sha256"],
        "trace_imports_strict_reproduction": source_records[
            "reproduce_dit_imagenet256.py"
        ]["sha256"],
        "custom_sampler_imports_strict_reproduction": source_records[
            "reproduce_dit_imagenet256.py"
        ]["sha256"],
    }
    if imported != expected_imported:
        raise RuntimeError("imported helper SHA contract changed")

    statistics = protocol.get("frozen_statistics", {})
    blur_stats = statistics.get("B_blur_mean", {})
    c3_stats = statistics.get("C_c3_low_jump", {})
    common_auc = {
        "aggregation": "micro over class-stratified bad-good pairs",
        "numerator": (
            "sum over classes of concordant oriented bad-good pairs, with ties counting 0.5"
        ),
        "denominator": "sum over classes of n_bad_class * n_clean_good_class",
        "zero_total_pair_denominator": "fail closed without evaluating the primary endpoint",
    }
    if (
        statistics.get("candidate_combination_allowed") is not False
        or blur_stats.get("feature") != BLUR_FEATURE
        or blur_stats.get("orientation") != "bad_high"
        or blur_stats.get("primary_endpoint")
        != "blur_or_soft_fusion_clear_bad_vs_clean_good"
        or blur_stats.get("primary_auc") != common_auc
        or blur_stats.get("alpha_0p10_operating_point")
        != {
            "strict_alert": "raw_score > class_specific_alpha_0p10_threshold",
            "TP": "blur-or-soft-fusion clear-bad trajectory with a strict alert",
            "FP": "clean_good trajectory with a strict alert",
            "TP_count": "sum of TP over all three classes",
            "TPR": "sum of TP over all three classes / total blur-or-soft-fusion clear-bad over all three classes",
            "FPR": "sum of FP over all three classes / total clean_good over all three classes",
            "zero_TPR_or_FPR_denominator": "fail closed",
        }
        or c3_stats.get("feature") != C3_FEATURE
        or c3_stats.get("orientation") != "bad_low"
        or c3_stats.get("primary_endpoint") != "all_clear_bad_vs_clean_good"
        or c3_stats.get("primary_auc") != common_auc
    ):
        raise RuntimeError("frozen primary statistic definitions changed")

    phase1_root = Path(str(protocol["phase1_protocol"]["path"]))
    _, protocol_binding = validate_phase1_protocol_lock(phase1_root)
    if protocol_binding != protocol["phase1_protocol"]:
        raise RuntimeError("phase-1 protocol binding changed")
    threshold_root = Path(str(protocol["phase1_thresholds"]["path"]))
    _, threshold_binding = validate_phase1_threshold_lock(
        threshold_root, EXPECTED_PROTOCOL_IDENTITY
    )
    if threshold_binding != protocol["phase1_thresholds"]:
        raise RuntimeError("phase-1 threshold binding changed")
    return protocol, manifest


def load_frozen_strict(lock_root: Path) -> ModuleType:
    path = lock_root / "sources/reproduce_dit_imagenet256.py"
    name = "_third_pool_frozen_strict_reproduction"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen strict helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_assets(
    protocol: Mapping[str, Any],
    lock_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
) -> tuple[Path, Path, Path]:
    dit_root = require_directory(dit_root, "DiT repository")
    checkpoint = require_regular(checkpoint, "DiT checkpoint")
    vae_snapshot = require_directory(vae_snapshot, "VAE snapshot")
    frozen = protocol.get("assets", {})
    if (
        str(dit_root) != frozen.get("dit_repository", {}).get("root")
        or str(checkpoint) != frozen.get("checkpoint", {}).get("path")
        or str(vae_snapshot) != frozen.get("vae_snapshot", {}).get("snapshot")
    ):
        raise RuntimeError("asset paths differ from the frozen source lock")
    strict = load_frozen_strict(lock_root)
    observed_repository = strict.validate_repository(dit_root, checkpoint)
    observed_checkpoint = strict.validate_checkpoint(checkpoint)
    observed_vae = strict.validate_vae_snapshot(vae_snapshot)
    if (
        observed_repository != frozen.get("dit_repository")
        or observed_checkpoint != frozen.get("checkpoint")
        or observed_vae != frozen.get("vae_snapshot")
    ):
        raise RuntimeError("model, checkpoint, or VAE differs from frozen identity")
    return dit_root, checkpoint, vae_snapshot


def parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(result) != WORKER_COUNT or len(set(result)) != WORKER_COUNT:
        raise argparse.ArgumentTypeError("--gpus must list exactly four unique devices")
    return result


def assignments(gpus: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    if len(gpus) != WORKER_COUNT or len(set(gpus)) != WORKER_COUNT:
        raise ValueError("exactly four unique GPUs are required")
    result = {
        gpu: SEEDS[index * SEEDS_PER_WORKER : (index + 1) * SEEDS_PER_WORKER]
        for index, gpu in enumerate(gpus)
    }
    if [seed for values in result.values() for seed in values] != list(SEEDS):
        raise AssertionError("contiguous GPU allocation lost, reordered, or duplicated a seed")
    if any(
        len(values) != SEEDS_PER_WORKER
        or values != tuple(range(values[0], values[-1] + 1))
        for values in result.values()
    ):
        raise AssertionError("each GPU must receive one contiguous 150-seed block")
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
    manifest: Mapping[str, Any],
    lock_root: Path,
    gpus: tuple[str, ...],
    output_root: Path,
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
) -> dict[str, Any]:
    allocation = assignments(gpus)
    source = lock_root / "sources/trace_dit_imagenet256_custom_batch.py"
    return {
        "schema_version": 1,
        "status": "FROZEN_FOUR_GPU_CONTIGUOUS_EXECUTION_PLAN",
        "sampling_source_lock": str(lock_root),
        "sampling_protocol_identity_sha256": protocol["identity_sha256"],
        "sampling_manifest_identity_sha256": manifest["identity_sha256"],
        "phase1_protocol_identity_sha256": EXPECTED_PROTOCOL_IDENTITY,
        "phase1_threshold_identity_sha256": EXPECTED_THRESHOLD_IDENTITY,
        "trace_source": str(source),
        "trace_source_sha256": sha256_file(source),
        "launcher_source_sha256": sha256_file(Path(__file__).resolve()),
        "classes_ordered": list(CLASSES),
        "global_seeds": list(SEEDS),
        "global_seed_count": len(SEEDS),
        "trajectory_count": len(SEEDS) * len(CLASSES),
        "gpus_ordered": list(gpus),
        "assignment": {gpu: list(values) for gpu, values in allocation.items()},
        "assignment_kind": "four ordered contiguous seed blocks",
        "output_root": str(output_root),
        "dit_root": str(dit_root),
        "checkpoint": str(checkpoint),
        "vae_snapshot": str(vae_snapshot),
        "required_trace_arrays": sorted(TRACE_ARRAY_NAMES),
        "observation_only": True,
        "labels_reviews_screen_results_or_sample_scores_read": False,
        "score_label_join_performed": False,
    }


def expected_trace_array_contract() -> dict[str, tuple[list[int], str]]:
    step_shape = [3, 250, 4, 32, 32]
    contract = {name: (step_shape, "<f4") for name in STEP_ARRAY_NAMES}
    contract.update(
        {
            "final_latents": ([3, 4, 32, 32], "<f4"),
            "decoded_images": ([3, 3, 256, 256], "<f4"),
            "internal_timestep": ([250], "<i2"),
            "alpha_bar": ([250], "<f8"),
        }
    )
    return contract


def safe_payload_path(outdir: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"unsafe trace payload path: {relative!r}")
    path = outdir / relative_path
    resolved_root = outdir.resolve()
    resolved = path.resolve()
    if resolved_root != resolved.parent and resolved_root not in resolved.parents:
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
        or trace_protocol.get("sampler")
        != "ancestral DDPM, manual statement-equivalent p_sample loop"
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
    frozen_assets = protocol["assets"]
    if (
        identity.get("source") != frozen_assets["dit_repository"]
        or identity.get("checkpoint") != frozen_assets["checkpoint"]
        or identity.get("vae_snapshot") != frozen_assets["vae_snapshot"]
    ):
        raise RuntimeError(f"seed model/checkpoint/VAE identity changed: {outdir}")

    array_records = manifest.get("trace_array_records")
    contract = expected_trace_array_contract()
    if not isinstance(array_records, dict) or set(array_records) != TRACE_ARRAY_NAMES:
        raise RuntimeError(f"seed trace array set changed: {outdir}")
    for name, (shape, dtype) in contract.items():
        row = array_records.get(name, {})
        digest = row.get("raw_sha256")
        if (
            row.get("shape") != shape
            or row.get("dtype") != dtype
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise RuntimeError(f"seed trace array contract changed: {outdir}/{name}")

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
        if sha256_file(outdir / snapshot_name) != source_records[basename]["sha256"]:
            raise RuntimeError(f"seed source snapshot differs from source lock: {outdir}")
    return {
        "seed": seed,
        "relative_output": outdir.name,
        "identity_sha256": identity_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "completion_sha256": sha256_file(completion_path),
        "outputs_sha256": manifest["outputs_sha256"],
        "output_count": completion["output_count"],
        "trace_npz_sha256": next(
            item["sha256"] for item in outputs if item["relative_path"] == "trace.npz"
        ),
    }


def next_log_path(output_root: Path, seed: int) -> Path:
    logs = output_root / "_runner_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if logs.is_symlink():
        raise RuntimeError(f"runner log directory must not be a symlink: {logs}")
    for attempt in range(1, 10_000):
        candidate = logs / f"seed{seed:03d}_attempt{attempt:04d}.log"
        if not os.path.lexists(candidate):
            return candidate
    raise RuntimeError(f"too many prior attempts for seed {seed}")


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
    outdir = output_root / f"third_pool_v1_seed{seed:03d}"
    if os.path.lexists(outdir):
        record = validate_seed_output(seed, outdir, protocol, lock_root)
        with print_lock:
            print(json.dumps({"seed": seed, "gpu": gpu, "reused": True}), flush=True)
        return record

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
    log_path = next_log_path(output_root, seed)
    started = time.time()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"started_unix": started, "gpu": gpu, "command": command}, sort_keys=True) + "\n")
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
            f"seed {seed} failed closed on GPU {gpu}; preserve and inspect {log_path} and {outdir}"
        )
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
    records: list[dict[str, Any]] = []
    for seed in seeds:
        records.append(
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
        )
    return records


def log_records(output_root: Path) -> list[dict[str, Any]]:
    logs = output_root / "_runner_logs"
    if not logs.exists():
        return []
    if not logs.is_dir() or logs.is_symlink():
        raise RuntimeError("runner logs path changed")
    records: list[dict[str, Any]] = []
    for path in sorted(logs.iterdir()):
        if not path.is_file() or path.is_symlink() or path.suffix != ".log":
            raise RuntimeError(f"unexpected runner log entry: {path}")
        records.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
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
            seed, output_root / f"third_pool_v1_seed{seed:03d}", protocol, lock_root
        )
        for seed in SEEDS
    ]
    if (
        manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256") != protocol["identity_sha256"]
        or manifest.get("phase1_protocol_identity_sha256") != EXPECTED_PROTOCOL_IDENTITY
        or manifest.get("phase1_threshold_identity_sha256") != EXPECTED_THRESHOLD_IDENTITY
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
        raise RuntimeError("completed third pool failed full validation")
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
        "sampling_protocol_identity_sha256": protocol["identity_sha256"],
        "phase1_protocol_identity_sha256": EXPECTED_PROTOCOL_IDENTITY,
        "phase1_threshold_identity_sha256": EXPECTED_THRESHOLD_IDENTITY,
        "execution_plan_sha256": sha256_file(plan_path),
        "seed_count": len(records),
        "trajectory_count": len(records) * len(CLASSES),
        "seed_outputs": records,
        "runner_logs": log_records(output_root),
        "observation_only": True,
        "labels_reviews_screen_results_or_sample_scores_read": False,
        "score_label_join_performed": False,
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


def output_state(output_root: Path) -> dict[str, Any]:
    if not output_root.exists():
        return {"state": "absent", "completed_seed_receipts": 0, "partial_seed_paths": 0}
    completed = 0
    partial = 0
    for seed in SEEDS:
        path = output_root / f"third_pool_v1_seed{seed:03d}"
        if not os.path.lexists(path):
            continue
        if (path / "manifest.json").is_file() and (path / "completion.json").is_file():
            completed += 1
        else:
            partial += 1
    return {
        "state": "present",
        "completed_seed_receipts": completed,
        "partial_seed_paths": partial,
        "note": "dry-run counts receipts only; run mode rehashes every reused payload",
    }


def run_self_test(lock_root: Path) -> None:
    protocol, manifest = validate_source_lock(lock_root)
    assert parse_gpus("0,1,2,3") == ("0", "1", "2", "3")
    allocation = assignments(("0", "1", "2", "3"))
    assert [values[0] for values in allocation.values()] == [250, 400, 550, 700]
    assert [values[-1] for values in allocation.values()] == [399, 549, 699, 849]
    assert all(len(values) == 150 for values in allocation.values())
    assert set(protocol["source_snapshots"]) == set(SOURCE_BASENAMES)
    assert manifest["sampling_protocol_identity_sha256"] == protocol["identity_sha256"]
    assert set(expected_trace_array_contract()) == TRACE_ARRAY_NAMES
    for invalid in ("0,1,2", "0,1,2,2", "0,1,2,3,4"):
        try:
            parse_gpus(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"invalid GPU plan was accepted: {invalid}")
    print(
        "self-test passed: phase-1 identities, immutable source lock, exact trace arrays, "
        "four contiguous 150-seed blocks, and score/label-blind contract"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
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
    lock_root = args.source_lock.expanduser().absolute()
    if args.self_test:
        run_self_test(lock_root)
        return 0
    protocol, source_manifest = validate_source_lock(lock_root)
    output_root = validate_output_root(args.output_root)
    checkpoint_arg = (
        args.checkpoint
        if args.checkpoint is not None
        else args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
    )
    dit_root, checkpoint, vae_snapshot = validate_assets(
        protocol,
        lock_root,
        args.dit_root,
        checkpoint_arg,
        args.vae_snapshot,
    )
    plan = build_plan(
        protocol,
        source_manifest,
        lock_root,
        args.gpus,
        output_root,
        dit_root,
        checkpoint,
        vae_snapshot,
    )
    if args.dry_run:
        print(json.dumps({"execution_plan": plan, "output_state": output_state(output_root)}, indent=2, sort_keys=True))
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
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
        raise RuntimeError("new pool receipt failed its validation round trip")
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
