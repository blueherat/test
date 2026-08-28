#!/usr/bin/env python3
"""Freeze the third-pool label-free product launcher before pool access.

This freezer validates only pre-existing source/protocol locks, implementation
sources, and local model assets.  It does not open, stat, list, or hash the
third sampling pool or any prospective product, label, review, screen, score,
threshold, or alert path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import run_dit_bad_good_third_pool_label_free_products as launcher


ROOT = launcher.ROOT
DEFAULT_OUTPUT = launcher.DEFAULT_SOURCE_LOCK

SOURCE_PATHS = {
    "run_dit_bad_good_third_pool_label_free_products.py": (
        ROOT / "experiments/run_dit_bad_good_third_pool_label_free_products.py"
    ),
    "freeze_dit_bad_good_third_pool_label_free_products.py": Path(__file__).resolve(),
    "run_dit_bad_good_third_pool.py": (
        ROOT / "experiments/run_dit_bad_good_third_pool.py"
    ),
    "analyze_dit_bad_good_custom_traces.py": (
        ROOT / "experiments/analyze_dit_bad_good_custom_traces.py"
    ),
    "extract_dit_predxstart_visual_tracks.py": (
        ROOT / "experiments/extract_dit_predxstart_visual_tracks.py"
    ),
    "extract_dit_endpoint_embeddings_label_free.py": (
        ROOT / "experiments/extract_dit_endpoint_embeddings_label_free.py"
    ),
    "reproduce_dit_imagenet256.py": (
        ROOT / "experiments/reproduce_dit_imagenet256.py"
    ),
}


def source_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for basename, raw_path in SOURCE_PATHS.items():
        path = launcher.require_regular(raw_path, f"product source {basename}")
        records[basename] = {
            "live_path_at_freeze": str(path),
            "bytes": path.stat().st_size,
            "sha256": launcher.sha256_file(path),
        }
    if set(records) != set(launcher.SOURCE_BASENAMES):
        raise RuntimeError("product source family differs from launcher")
    expected = {
        "run_dit_bad_good_third_pool.py": launcher.EXPECTED_SAMPLING_RUNNER_SHA256,
        "analyze_dit_bad_good_custom_traces.py": launcher.EXPECTED_PRIMARY_SOURCE_SHA256,
        "extract_dit_predxstart_visual_tracks.py": launcher.EXPECTED_VISUAL_SOURCE_SHA256,
        "extract_dit_endpoint_embeddings_label_free.py": launcher.EXPECTED_ENDPOINT_SOURCE_SHA256,
        "reproduce_dit_imagenet256.py": launcher.EXPECTED_REPRODUCTION_SOURCE_SHA256,
    }
    for basename, digest in expected.items():
        if records[basename]["sha256"] != digest:
            raise RuntimeError(f"product helper differs from its prior SHA pin: {basename}")
    return records


def _regular_asset_record(path: Path, expected_sha: str, description: str) -> dict[str, Any]:
    raw = path.expanduser().absolute()
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"{description} must resolve to a regular file: {raw}")
    observed = launcher.sha256_file(resolved)
    if observed != expected_sha:
        raise RuntimeError(f"{description} differs from its SHA pin")
    return {
        "path": str(raw),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": observed,
    }


def validate_foundations() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sampling_protocol, sampling_manifest = launcher.validate_sampling_source_lock_label_free(
        launcher.SAMPLING_SOURCE_LOCK
    )
    if (
        sampling_protocol.get("identity_sha256")
        != launcher.EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or sampling_manifest.get("identity_sha256")
        != launcher.EXPECTED_SAMPLING_MANIFEST_IDENTITY
    ):
        raise RuntimeError("sampling source foundation changed")
    endpoint_protocol, endpoint_manifest = (
        launcher.validate_endpoint_secondary_lock_label_free(
            launcher.ENDPOINT_SECONDARY_LOCK
        )
    )
    if (
        endpoint_protocol.get("identity_sha256")
        != launcher.EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or endpoint_manifest.get("identity_sha256")
        != launcher.EXPECTED_ENDPOINT_MANIFEST_IDENTITY
        or launcher.canonical_sha256(launcher.without_identity(endpoint_protocol))
        != launcher.EXPECTED_ENDPOINT_PROTOCOL_IDENTITY
        or launcher.canonical_sha256(launcher.without_identity(endpoint_manifest))
        != launcher.EXPECTED_ENDPOINT_MANIFEST_IDENTITY
    ):
        raise RuntimeError("endpoint-secondary foundation changed")
    return sampling_protocol, sampling_manifest, endpoint_protocol


def validate_assets(sampling_protocol: Mapping[str, Any]) -> dict[str, Any]:
    vae = dict(sampling_protocol["assets"]["vae_snapshot"])
    if vae.get("revision") != "31f26fdeee1355a5c34592e401dd41e45d25a493":
        raise RuntimeError("sampling VAE revision changed")
    for item in vae.get("files", ()):  # validate the resolved bytes now
        observed = _regular_asset_record(
            Path(item["path"]), item["sha256"], f"VAE {item['name']}"
        )
        if observed != {
            "path": item["path"],
            "resolved_path": item["resolved_path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }:
            raise RuntimeError(f"sampling VAE member changed: {item['name']}")

    resnet = _regular_asset_record(
        Path("/home/zhoushunyu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"),
        launcher.RESNET18_SHA256,
        "ResNet-18 weights",
    )
    inception = _regular_asset_record(
        Path(
            "/home/zhoushunyu/.cache/torch/hub/checkpoints/"
            "pt_inception-2015-12-05-6726825d.pth"
        ),
        launcher.INCEPTION_SHA256,
        "Inception compatibility weights",
    )
    dino_snapshot = Path(
        "/home/zhoushunyu/.cache/huggingface/hub/"
        "models--facebook--dinov2-with-registers-large/snapshots/"
        "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
    )
    if not dino_snapshot.is_dir() or dino_snapshot.is_symlink():
        raise RuntimeError("DINO snapshot must be a real directory")
    dino_files = []
    for name, digest in launcher.DINO_FILE_SHA256.items():
        record = _regular_asset_record(dino_snapshot / name, digest, f"DINO {name}")
        dino_files.append({"name": name, **record})
    assets = {
        "vae": vae,
        "resnet18": resnet,
        "inception": inception,
        "dinov2": {
            "snapshot": str(dino_snapshot),
            "model_id": "facebook/dinov2-with-registers-large",
            "revision": "e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
            "files": dino_files,
        },
    }
    launcher._validate_assets(assets)
    return assets


def build_protocol(
    *,
    sources: Mapping[str, Any],
    assets: Mapping[str, Any],
    sampling_protocol: Mapping[str, Any],
    sampling_manifest: Mapping[str, Any],
    endpoint_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "dit_bad_good_third_pool_label_free_products_v1",
        "status": "FROZEN_BEFORE_POOL_SEAL_OR_PRODUCT_ACCESS",
        "implementation_source_sha256": sources[
            "run_dit_bad_good_third_pool_label_free_products.py"
        ]["sha256"],
        "foundation_identity_pins": {
            "sampling_source_lock_path": str(launcher.SAMPLING_SOURCE_LOCK),
            "sampling_protocol_identity_sha256": sampling_protocol["identity_sha256"],
            "sampling_manifest_identity_sha256": sampling_manifest["identity_sha256"],
            "sampling_runner_source_sha256": launcher.EXPECTED_SAMPLING_RUNNER_SHA256,
            "endpoint_secondary_lock_path": str(launcher.ENDPOINT_SECONDARY_LOCK),
            "endpoint_secondary_protocol_identity_sha256": endpoint_protocol[
                "identity_sha256"
            ],
            "endpoint_secondary_manifest_identity_sha256": (
                launcher.EXPECTED_ENDPOINT_MANIFEST_IDENTITY
            ),
        },
        "cohort": {
            "classes_ordered": list(launcher.CLASSES),
            "global_seeds": list(launcher.SEEDS),
            "seed_count": len(launcher.SEEDS),
            "trajectory_count": launcher.TRAJECTORY_COUNT,
            "sample_order": "global_seed ascending, then class slots 207/602/795",
        },
        "outputs": {
            "pool_root": str(launcher.POOL_ROOT),
            "primary": str(launcher.PRIMARY_OUTPUT),
            "visual": str(launcher.VISUAL_OUTPUT),
            "endpoint": str(launcher.ENDPOINT_OUTPUT),
            "completion_receipt": str(launcher.RECEIPT_OUTPUT),
            "pool_binding": str(launcher.DEFAULT_POOL_BINDING),
        },
        "products": launcher.expected_products_contract(),
        "role_boundaries": launcher.expected_role_boundaries(),
        "execution": {
            "python_executable": sys.executable,
            "commands": launcher._expected_commands(),
            "dependency_order": [
                "validate bound complete pool",
                "primary",
                "visual || endpoint",
                "receipt",
            ],
            "parallel_after_primary": ["visual", "endpoint"],
            "gpu_routing": {
                "visual": {
                    "physical_cuda_visible_devices": launcher.VISUAL_PHYSICAL_GPU,
                    "logical_device": launcher.LOGICAL_DEVICE,
                },
                "endpoint": {
                    "physical_cuda_visible_devices": launcher.ENDPOINT_PHYSICAL_GPU,
                    "logical_device": launcher.LOGICAL_DEVICE,
                },
            },
            "primary_is_completed_and_validated_before_children_start": True,
        },
        "assets": dict(assets),
        "resume_and_overwrite": launcher.expected_resume_contract(),
        "supervision_policy": launcher.expected_supervision_contract(),
        "pool_binding_contract": launcher.expected_pool_binding_contract(),
        "source_snapshots": dict(sources),
        "imported_helper_sha256": launcher.expected_imported_helper_contract(),
        "threat_model": {
            "assumption": (
                "controlled static non-concurrently-rewritten local filesystem with "
                "Git or append-only chronology"
            ),
            "not_claimed": (
                "cryptographic authentication against malicious replacement and "
                "manual re-signing of a self-consistent artifact tree"
            ),
        },
        "evidence_access_audit": launcher.expected_evidence_access_audit(),
    }
    protocol["identity_sha256"] = launcher.canonical_sha256(protocol)
    return protocol


def build_all() -> tuple[dict[str, Any], dict[str, Any]]:
    sources = source_records()
    sampling_protocol, sampling_manifest, endpoint_protocol = validate_foundations()
    assets = validate_assets(sampling_protocol)
    protocol = build_protocol(
        sources=sources,
        assets=assets,
        sampling_protocol=sampling_protocol,
        sampling_manifest=sampling_manifest,
        endpoint_protocol=endpoint_protocol,
    )
    return protocol, sources


def publish(output: Path) -> Path:
    output = launcher.require_exact_path(
        output, DEFAULT_OUTPUT, "product source-lock output"
    )
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite product source lock: {output}")
    protocol, _sources = build_all()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        launcher.write_json(staging / "product_protocol.json", protocol)
        source_root = staging / "sources"
        source_root.mkdir()
        for basename, path in SOURCE_PATHS.items():
            shutil.copy2(path, source_root / basename)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "dit_bad_good_third_pool_label_free_products_source_lock_v1",
            "status": "complete",
            "product_protocol_identity_sha256": protocol["identity_sha256"],
            "files": launcher.artifact_records(staging),
        }
        manifest["identity_sha256"] = launcher.canonical_sha256(manifest)
        launcher.write_json(staging / "manifest.json", manifest)
        launcher.write_json(
            staging / "completion.json",
            {
                "complete": True,
                "product_protocol_identity_sha256": protocol["identity_sha256"],
                "product_protocol_file_sha256": launcher.sha256_file(
                    staging / "product_protocol.json"
                ),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": launcher.sha256_file(staging / "manifest.json"),
                "pool_or_product_access_performed": False,
            },
        )
        launcher._validate_source_lock_contents(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def dry_run() -> dict[str, Any]:
    protocol, sources = build_all()
    return {
        "status": "DRY_RUN_NO_OUTPUT_WRITTEN",
        "product_protocol_identity_sha256": protocol["identity_sha256"],
        "launcher_source_sha256": sources[
            "run_dit_bad_good_third_pool_label_free_products.py"
        ]["sha256"],
        "freezer_source_sha256": sources[
            "freeze_dit_bad_good_third_pool_label_free_products.py"
        ]["sha256"],
        "pool_or_product_access_performed": False,
        "commands": protocol["execution"]["commands"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        launcher.synthetic_self_test()
        print("product freezer self-test passed; pool/product paths remained unopened")
        return 0
    if args.dry_run:
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return 0
    path = publish(DEFAULT_OUTPUT)
    protocol, manifest = launcher.validate_source_lock(path)
    print(
        json.dumps(
            {
                "path": str(path),
                "product_protocol_identity_sha256": protocol["identity_sha256"],
                "manifest_identity_sha256": manifest["identity_sha256"],
                "pool_or_product_access_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
