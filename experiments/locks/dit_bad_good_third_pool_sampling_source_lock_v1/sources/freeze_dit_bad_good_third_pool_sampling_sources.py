#!/usr/bin/env python3
"""Freeze the source and asset identity for third-pool DiT trace production.

This phase reads only the already frozen phase-1 protocol and threshold locks,
source files, the DiT checkout, checkpoint, and VAE snapshot.  It never opens
endpoint images, trajectories, visual labels/reviews, screening results, or
sample-level feature tables.  The resulting lock is immutable and is required
by ``run_dit_bad_good_third_pool.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import run_dit_bad_good_third_pool as runner


ROOT = runner.ROOT
DATA_ROOT = runner.DATA_ROOT
DEFAULT_PROTOCOL_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_protocol_lock_v1"
)
DEFAULT_THRESHOLD_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_threshold_lock_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_sampling_source_lock_v1"
)
DEFAULT_DIT_ROOT = runner.DEFAULT_DIT_ROOT
DEFAULT_VAE = runner.DEFAULT_VAE

SOURCE_PATHS = {
    "trace_dit_imagenet256_custom_batch.py": (
        ROOT / "experiments/trace_dit_imagenet256_custom_batch.py"
    ),
    "sample_dit_imagenet256_custom.py": (
        ROOT / "experiments/sample_dit_imagenet256_custom.py"
    ),
    "reproduce_dit_imagenet256.py": (
        ROOT / "experiments/reproduce_dit_imagenet256.py"
    ),
    "run_dit_bad_good_third_pool.py": (
        ROOT / "experiments/run_dit_bad_good_third_pool.py"
    ),
    "freeze_dit_bad_good_third_pool_sampling_sources.py": Path(__file__).resolve(),
}


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def load_module(path: Path) -> ModuleType:
    path = require_regular(path, "strict reproduction source")
    spec = importlib.util.spec_from_file_location("_third_pool_source_freeze_strict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import strict reproduction source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_records(phase1_protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for basename, raw_path in SOURCE_PATHS.items():
        path = require_regular(raw_path, f"sampling source {basename}")
        records[basename] = {"live_path_at_freeze": str(path), "sha256": sha256_file(path)}
    if set(records) != set(runner.SOURCE_BASENAMES):
        raise RuntimeError("sampling source set differs from launcher contract")
    frozen_phase1 = phase1_protocol.get("source_snapshots", {})
    phase1_pairs = {
        "trace_dit_imagenet256_custom_batch.py": "trace_runner.py",
        "sample_dit_imagenet256_custom.py": "custom_sampler_helper.py",
        "reproduce_dit_imagenet256.py": "strict_reproduction_helper.py",
    }
    for basename, phase1_name in phase1_pairs.items():
        if records[basename]["sha256"] != frozen_phase1.get(phase1_name, {}).get("sha256"):
            raise RuntimeError(
                f"sampling source {basename} differs from the phase-1 protocol snapshot"
            )
    return records


def validate_assets(
    dit_root: Path, checkpoint: Path, vae_snapshot: Path
) -> dict[str, Any]:
    dit_root = runner.require_directory(dit_root, "DiT repository")
    checkpoint = runner.require_regular(checkpoint, "DiT checkpoint")
    vae_snapshot = runner.require_directory(vae_snapshot, "VAE snapshot")
    strict = load_module(SOURCE_PATHS["reproduce_dit_imagenet256.py"])
    repository = strict.validate_repository(dit_root, checkpoint)
    checkpoint_identity = strict.validate_checkpoint(checkpoint)
    vae_identity = strict.validate_vae_snapshot(vae_snapshot)
    if (
        strict.MODEL_NAME != "DiT-XL/2"
        or strict.NUM_SAMPLING_STEPS != 250
        or strict.CFG_SCALE != 4.0
        or strict.VAE_KIND != "mse"
        or strict.VAE_SCALING_FACTOR != 0.18215
    ):
        raise RuntimeError("strict DiT scientific constants changed")
    return {
        "dit_repository": repository,
        "checkpoint": checkpoint_identity,
        "vae_snapshot": vae_identity,
    }


def build_protocol(
    phase1_binding: Mapping[str, Any],
    threshold_binding: Mapping[str, Any],
    sources: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    blocks = [
        {
            "worker_index": index,
            "seed_start_inclusive": values[0],
            "seed_stop_inclusive": values[-1],
            "seed_count": len(values),
        }
        for index, values in enumerate(
            runner.assignments(("GPU0", "GPU1", "GPU2", "GPU3")).values()
        )
    ]
    common_auc = {
        "aggregation": "micro over class-stratified bad-good pairs",
        "numerator": (
            "sum over classes of concordant oriented bad-good pairs, with ties counting 0.5"
        ),
        "denominator": "sum over classes of n_bad_class * n_clean_good_class",
        "zero_total_pair_denominator": "fail closed without evaluating the primary endpoint",
    }
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_THIRD_POOL_GPU_SAMPLING",
        "phase1_protocol": dict(phase1_binding),
        "phase1_thresholds": dict(threshold_binding),
        "scientific_contract": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "sampler_implementation": (
                "manual statement-equivalent p_sample loop from the frozen trace source"
            ),
            "sampling_steps": 250,
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "classes_ordered": list(runner.CLASSES),
            "global_seeds": list(runner.SEEDS),
            "global_seed_start_inclusive": runner.SEEDS[0],
            "global_seed_stop_inclusive": runner.SEEDS[-1],
            "global_seed_count": len(runner.SEEDS),
            "trajectory_count": len(runner.SEEDS) * len(runner.CLASSES),
            "observation_only": True,
            "quality_score": None,
            "selection": None,
            "intervention": None,
        },
        "execution_contract": {
            "gpu_worker_count": runner.WORKER_COUNT,
            "seeds_per_gpu": runner.SEEDS_PER_WORKER,
            "assignment_kind": "four ordered contiguous seed blocks",
            "ordered_blocks": blocks,
            "one_process_per_gpu": True,
            "one_global_seed_per_trace_invocation": True,
            "classes_are_one_ordered_batch_per_seed": True,
        },
        "required_trace_arrays": sorted(runner.TRACE_ARRAY_NAMES),
        "candidate_extractability": {
            "B_blur_mean": {
                "source_array": "pred_xstart",
                "sampling_steps": [69, 79, 89, 99, 109, 119, 129, 139, 149],
                "requires_frozen_VAE_decode": True,
                "extractor_is_not_run_during_sampling": True,
            },
            "C_c3_low_jump": {
                "source_arrays": ["pred_xstart", "alpha_bar"],
                "sampling_steps_inclusive": [100, 149],
                "latent_channel_zero_based": 3,
                "extractor_is_not_run_during_sampling": True,
            },
            "candidates_are_not_combined": True,
        },
        "frozen_statistics": {
            "candidate_combination_allowed": False,
            "B_blur_mean": {
                "feature": runner.BLUR_FEATURE,
                "orientation": "bad_high",
                "primary_endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                "primary_auc": common_auc,
                "alpha_0p10_operating_point": {
                    "strict_alert": "raw_score > class_specific_alpha_0p10_threshold",
                    "TP": "blur-or-soft-fusion clear-bad trajectory with a strict alert",
                    "FP": "clean_good trajectory with a strict alert",
                    "TP_count": "sum of TP over all three classes",
                    "TPR": (
                        "sum of TP over all three classes / total blur-or-soft-fusion "
                        "clear-bad over all three classes"
                    ),
                    "FPR": (
                        "sum of FP over all three classes / total clean_good over all "
                        "three classes"
                    ),
                    "zero_TPR_or_FPR_denominator": "fail closed",
                },
            },
            "C_c3_low_jump": {
                "feature": runner.C3_FEATURE,
                "orientation": "bad_low",
                "primary_endpoint": "all_clear_bad_vs_clean_good",
                "primary_auc": common_auc,
            },
        },
        "output_contract": {
            "seed_directory_template": "third_pool_v1_seed{global_seed:03d}",
            "endpoint_pngs_per_seed": 3,
            "full_trace_npz_per_seed": 1,
            "expected_payload_relative_paths": sorted(
                runner.EXPECTED_OUTPUT_RELATIVE_PATHS
            ),
            "seed_manifest_and_completion_required": True,
            "pool_manifest_and_completion_required": True,
            "all_payloads_sha256_bound": True,
        },
        "resume_and_failure_contract": {
            "completed_seed_reuse": "full scientific identity and payload hash validation",
            "partial_or_changed_seed": "fail closed; preserve and refuse overwrite",
            "seed_logs": "append by exclusive monotonically numbered attempt files",
            "pool_receipts": "published only after all 600 seed outputs validate",
            "automatic_deletion_or_quarantine": False,
        },
        "assets": dict(assets),
        "source_snapshots": dict(sources),
        "imported_helper_sha256": {
            "source_locker_imports_launcher": sources[
                "run_dit_bad_good_third_pool.py"
            ]["sha256"],
            "launcher_imports_strict_reproduction": sources[
                "reproduce_dit_imagenet256.py"
            ]["sha256"],
            "trace_imports_custom_sampler": sources[
                "sample_dit_imagenet256_custom.py"
            ]["sha256"],
            "trace_imports_strict_reproduction": sources[
                "reproduce_dit_imagenet256.py"
            ]["sha256"],
            "custom_sampler_imports_strict_reproduction": sources[
                "reproduce_dit_imagenet256.py"
            ]["sha256"],
        },
        "phase1_access_semantics": {
            "protocol_locker_sample_features_access": (
                "sample_features.csv bytes were read sequentially only to compute and "
                "verify SHA-256 integrity; the CSV was not parsed or loaded as a table"
            ),
            "threshold_calibrator_sample_features_access": (
                "the complete CSV rows were read to audit row identifiers and cohort "
                "membership, but only the two frozen candidate columns were converted "
                "to numeric score values, and only calibration seeds 30..49 supplied "
                "numeric values to thresholds"
            ),
            "visual_labels_or_reviews_used": False,
            "screening_results_used": False,
        },
        "evidence_access_audit": {
            "phase1_protocol_lock_opened_and_verified": True,
            "phase1_threshold_lock_opened_and_verified": True,
            "endpoint_images_opened": False,
            "trajectory_archives_opened": False,
            "visual_labels_or_reviews_opened": False,
            "labels_opened": False,
            "screen_results_opened": False,
            "sample_scores_opened": False,
            "score_label_join_performed": False,
        },
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    return protocol


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"sampling lock artifact must not be a symlink: {path}")
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def prepare(
    protocol_lock: Path,
    threshold_lock: Path,
    dit_root: Path,
    checkpoint: Path,
    vae_snapshot: Path,
) -> dict[str, Any]:
    phase1_protocol, phase1_binding = runner.validate_phase1_protocol_lock(protocol_lock)
    _, threshold_binding = runner.validate_phase1_threshold_lock(
        threshold_lock, runner.EXPECTED_PROTOCOL_IDENTITY
    )
    sources = source_records(phase1_protocol)
    assets = validate_assets(dit_root, checkpoint, vae_snapshot)
    return build_protocol(phase1_binding, threshold_binding, sources, assets)


def publish(protocol: Mapping[str, Any], output: Path) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite sampling source lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "sampling_protocol.json", protocol)
        source_root = staging / "sources"
        source_root.mkdir()
        for basename, raw_path in SOURCE_PATHS.items():
            shutil.copy2(raw_path, source_root / basename)
            if sha256_file(source_root / basename) != protocol["source_snapshots"][basename]["sha256"]:
                raise RuntimeError(f"source changed while freezing: {raw_path}")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "sampling_protocol_identity_sha256": protocol["identity_sha256"],
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "sampling_protocol_file_sha256": sha256_file(
                    staging / "sampling_protocol.json"
                ),
                "sampling_protocol_identity_sha256": protocol["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
            },
        )
        validated, _ = runner.validate_source_lock(staging)
        if validated != protocol:
            raise RuntimeError("new sampling source lock failed round-trip validation")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    runner.validate_source_lock(output)
    return output


def run_self_test(protocol_lock: Path, threshold_lock: Path) -> None:
    protocol, protocol_binding = runner.validate_phase1_protocol_lock(protocol_lock)
    threshold, threshold_binding = runner.validate_phase1_threshold_lock(
        threshold_lock, runner.EXPECTED_PROTOCOL_IDENTITY
    )
    sources = source_records(protocol)
    allocation = runner.assignments(("GPU0", "GPU1", "GPU2", "GPU3"))
    assert protocol_binding["identity_sha256"] == runner.EXPECTED_PROTOCOL_IDENTITY
    assert threshold_binding["identity_sha256"] == runner.EXPECTED_THRESHOLD_IDENTITY
    assert threshold["third_pool_protocol_identity_sha256"] == runner.EXPECTED_PROTOCOL_IDENTITY
    assert set(sources) == set(runner.SOURCE_BASENAMES)
    assert [(values[0], values[-1]) for values in allocation.values()] == [
        (250, 399),
        (400, 549),
        (550, 699),
        (700, 849),
    ]
    print(
        "self-test passed: exact phase-1 protocol/threshold locks, unchanged sampling "
        "sources, and four contiguous 150-seed blocks"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--threshold-lock", type=Path, default=DEFAULT_THRESHOLD_LOCK)
    parser.add_argument("--dit-root", type=Path, default=DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test(args.protocol_lock, args.threshold_lock)
        return 0
    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else args.dit_root / "pretrained_models" / runner.CHECKPOINT_FILENAME
    )
    protocol = prepare(
        args.protocol_lock,
        args.threshold_lock,
        args.dit_root,
        checkpoint,
        args.vae_snapshot,
    )
    if args.dry_run:
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0
    output = publish(protocol, args.output)
    print(
        json.dumps(
            {
                "sampling_source_lock": str(output),
                "sampling_protocol_identity_sha256": protocol["identity_sha256"],
                "source_count": len(protocol["source_snapshots"]),
                "trajectory_count": protocol["scientific_contract"]["trajectory_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
