#!/usr/bin/env python3
"""Freeze event-rich endpoint sampler sources, protocol v3, and model assets.

The locker is deliberately unusable with protocol v2: v2 freezes a six-class
batch-dependent RNG contract, while this sampler requires the v3 pair-keyed
contract.  A successful lock is therefore the final authorization boundary
before real endpoint sampling.  No endpoint, label, review, metric, feature,
embedding, or score data are opened here.
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

import run_dit_event_rich_endpoint_screen as runner


ROOT = runner.ROOT
DEFAULT_EVENT_PROTOCOL_LOCK = (
    ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v3"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments/locks/dit_event_rich_endpoint_sampling_source_lock_v1"
)
SOURCE_PATHS = {
    "sample_dit_imagenet256_endpoint_pairs.py": (
        ROOT / "experiments/sample_dit_imagenet256_endpoint_pairs.py"
    ),
    "run_dit_event_rich_endpoint_screen.py": (
        ROOT / "experiments/run_dit_event_rich_endpoint_screen.py"
    ),
    "freeze_dit_event_rich_endpoint_sampling_sources.py": Path(__file__).resolve(),
    "reproduce_dit_imagenet256.py": (
        ROOT / "experiments/reproduce_dit_imagenet256.py"
    ),
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_module(path: Path) -> ModuleType:
    path = runner.require_regular(path, "strict reproduction source")
    spec = importlib.util.spec_from_file_location("_event_endpoint_freezer_strict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import strict helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_lock_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = runner.require_directory(root, "event-rich confirmation protocol v3 lock")
    protocol_path = runner.require_regular(root / "protocol.json", "event protocol")
    manifest_path = runner.require_regular(root / "manifest.json", "event manifest")
    completion_path = runner.require_regular(root / "completion.json", "event completion")
    protocol = runner.load_json(protocol_path)
    manifest = runner.load_json(manifest_path)
    completion = runner.load_json(completion_path)
    classes = runner.validate_event_protocol_snapshot(protocol)
    identity = protocol["identity_sha256"]
    manifest_identity = manifest.get("identity_sha256")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"event protocol lock contains a symlink: {path}")
        files.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": runner.sha256_file(path),
            }
        )
    if (
        not isinstance(manifest_identity, str)
        or runner.canonical_sha256(runner.without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("protocol_identity_sha256") != identity
        or manifest.get("files") != files
        or completion
        != {
            "complete": True,
            "protocol_identity_sha256": identity,
            "protocol_file_sha256": runner.sha256_file(protocol_path),
            "manifest_identity_sha256": manifest_identity,
            "manifest_file_sha256": runner.sha256_file(manifest_path),
        }
    ):
        raise RuntimeError("event-rich confirmation protocol v3 lock failed validation")
    return protocol, {
        "path": str(root),
        "identity_sha256": identity,
        "file_sha256": runner.sha256_file(protocol_path),
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": runner.sha256_file(manifest_path),
        "completion_file_sha256": runner.sha256_file(completion_path),
        "classes_ordered": list(classes),
    }


def source_records() -> dict[str, dict[str, Any]]:
    if set(SOURCE_PATHS) != set(runner.SOURCE_BASENAMES):
        raise RuntimeError("freezer and launcher source sets differ")
    records: dict[str, dict[str, Any]] = {}
    for basename, raw_path in SOURCE_PATHS.items():
        path = runner.require_regular(raw_path, f"sampling source {basename}")
        records[basename] = {
            "live_path_at_freeze": str(path),
            "sha256": runner.sha256_file(path),
        }
    return records


def validate_assets(
    dit_root: Path, checkpoint: Path, vae_snapshot: Path
) -> dict[str, Any]:
    strict = load_module(SOURCE_PATHS["reproduce_dit_imagenet256.py"])
    dit_root = runner.require_directory(dit_root, "DiT repository")
    checkpoint = runner.require_regular(checkpoint, "DiT checkpoint")
    vae_snapshot = runner.require_directory(vae_snapshot, "VAE snapshot")
    if (
        strict.MODEL_NAME != "DiT-XL/2"
        or strict.NUM_SAMPLING_STEPS != 250
        or strict.CFG_SCALE != 4.0
        or strict.VAE_KIND != "mse"
        or strict.VAE_SCALING_FACTOR != 0.18215
    ):
        raise RuntimeError("strict DiT scientific constants changed")
    return {
        "dit_repository": strict.validate_repository(dit_root, checkpoint),
        "checkpoint": strict.validate_checkpoint(checkpoint),
        "vae_snapshot": strict.validate_vae_snapshot(vae_snapshot),
    }


def build_protocol(
    event_binding: Mapping[str, Any],
    sources: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    classes = tuple(int(value) for value in event_binding["classes_ordered"])
    axis = runner.pair_axis(classes)
    shards = runner.assignments(("GPU0", "GPU1", "GPU2", "GPU3"), classes)
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EVENT_RICH_ENDPOINT_GPU_SAMPLING",
        "event_protocol": {
            key: value for key, value in event_binding.items() if key != "classes_ordered"
        },
        "scientific_contract": {
            "model": "DiT-XL/2 ImageNet-256",
            "sampler": "official 250-step ancestral DDPM",
            "sampling_steps": 250,
            "clip_denoised": False,
            "cfg_scale": 4.0,
            "cfg_epsilon_channels": 3,
            "vae": "mse",
            "vae_scaling_factor": 0.18215,
            "classes_ordered": list(classes),
            "global_seeds": list(runner.EXPECTED_SEEDS),
            "pair_axis_order": "seed-major, frozen-class-roster-minor",
            "pair_count": len(axis),
            "endpoint_only": True,
            "trace_saved": False,
            "quality_score": None,
            "selection": None,
            "intervention": None,
        },
        "rng_contract": {
            "unit": "(global_seed,class_id)",
            "domain": runner.RNG_DOMAIN,
            "derivation": (
                "uint64_be(first_8_bytes(SHA256(ASCII(domain + NUL + global_seed "
                "+ NUL + class_id)))) mod 2^63"
            ),
            "manual_seed_timing": (
                "after frozen model/VAE load, immediately before singleton initial latent"
            ),
            "classes_per_sampler_invocation": 1,
            "same_global_seed_classes_share_initial_noise": False,
            "same_global_seed_classes_share_transition_innovations": False,
            "task_order_worker_shard_resume_invariant": True,
            "initial_noise_shape": [1, 4, 32, 32],
            "duplicated_cfg_state_shape": [2, 4, 32, 32],
            "transition_randn_like_calls": 250,
            "transition_noise_shape_each_call": [2, 4, 32, 32],
            "full_2B_randn_like_each_transition_including_t0": True,
            "terminal_t0_randn_consumed_then_masked": True,
            "second_half_transition_noises_consumed_then_state_discarded": True,
            "relation_to_third_pool": (
                "same singleton-or-batch first-half latent duplication, upstream CFG and full-2B "
                "ancestral transition semantics; pair-keyed reseeding replaces the third pool's "
                "ordered three-class batch RNG so endpoint pixels are shard/order invariant"
            ),
        },
        "execution_contract": {
            "gpu_worker_count": runner.WORKER_COUNT,
            "pair_count_per_worker": runner.PAIRS_PER_WORKER,
            "assignment_kind": "four contiguous 252-pair operational shards",
            "assignment_is_not_scientific_rng_input": True,
            "one_model_and_vae_load_per_worker": True,
            "ordered_shards": [
                {
                    "worker_index": index,
                    "first_pair": {"global_seed": shard[0][0], "class_id": shard[0][1]},
                    "last_pair": {"global_seed": shard[-1][0], "class_id": shard[-1][1]},
                    "pair_count": len(shard),
                }
                for index, shard in enumerate(shards.values())
            ],
        },
        "output_contract": {
            "pair_directory_template": "pairs/seed{global_seed:04d}_class{class_id:04d}",
            "endpoint_filename": "endpoint.png",
            "endpoint_mode": "RGB",
            "endpoint_size": [256, 256],
            "files_per_pair": ["endpoint.png", "manifest.json", "completion.json"],
            "trajectory_or_latent_payload": None,
            "pair_and_pool_receipts_required": True,
            "all_endpoint_file_and_pixel_hashes_bound": True,
        },
        "resume_and_failure_contract": {
            "completed_pair_reuse": "full identity and endpoint file/pixel hash validation",
            "partial_or_changed_pair": "fail closed; preserve and refuse overwrite",
            "pool_receipts": "published only after all 1008 pair outputs fully revalidate",
            "automatic_deletion_or_quarantine": False,
        },
        "evidence_access_audit": {
            "labels_or_reviews_opened": False,
            "metrics_features_embeddings_or_scores_opened": False,
            "score_label_join_performed": False,
        },
        "assets": dict(assets),
        "source_snapshots": dict(sources),
    }
    protocol["identity_sha256"] = runner.canonical_sha256(protocol)
    return protocol


def freeze(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite source lock: {output}")
    event_protocol, event_binding = event_lock_binding(args.event_protocol_lock)
    sources = source_records()
    checkpoint = args.checkpoint or args.dit_root / "pretrained_models" / runner.CHECKPOINT_FILENAME
    assets = validate_assets(args.dit_root, checkpoint, args.vae_snapshot)
    protocol = build_protocol(event_binding, sources, assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=str(output.parent))
    )
    try:
        (temporary / "sources").mkdir(exist_ok=False)
        for basename, source_path in SOURCE_PATHS.items():
            shutil.copyfile(source_path, temporary / "sources" / basename)
        shutil.copyfile(args.event_protocol_lock / "protocol.json", temporary / "event_protocol.json")
        write_json(temporary / "sampling_protocol.json", protocol)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "sampling_protocol_identity_sha256": protocol["identity_sha256"],
            "event_protocol_identity_sha256": event_protocol["identity_sha256"],
            "files": runner.artifact_records(temporary),
        }
        manifest["identity_sha256"] = runner.canonical_sha256(manifest)
        write_json(temporary / "manifest.json", manifest)
        completion = {
            "complete": True,
            "sampling_protocol_identity_sha256": protocol["identity_sha256"],
            "sampling_protocol_file_sha256": runner.sha256_file(
                temporary / "sampling_protocol.json"
            ),
            "event_protocol_identity_sha256": event_protocol["identity_sha256"],
            "event_protocol_file_sha256": runner.sha256_file(
                temporary / "event_protocol.json"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": runner.sha256_file(temporary / "manifest.json"),
        }
        write_json(temporary / "completion.json", completion)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validated_protocol, _, _ = runner.validate_source_lock(output)
    if validated_protocol != protocol:
        raise RuntimeError("new source lock failed validation round trip")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-protocol-lock", type=Path, default=DEFAULT_EVENT_PROTOCOL_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dit-root", type=Path, default=runner.DEFAULT_DIT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path, default=runner.DEFAULT_VAE)
    parser.add_argument("--validate-input-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.validate_input_only:
        protocol, binding = event_lock_binding(args.event_protocol_lock)
        print(
            json.dumps(
                {
                    "valid": True,
                    "event_protocol_identity_sha256": protocol["identity_sha256"],
                    "event_binding": binding,
                    "real_sampling_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = freeze(args)
    print(json.dumps({"source_lock": str(output), "real_sampling_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
