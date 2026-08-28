#!/usr/bin/env python3
"""Freeze the v2.2 repairability pilot execution matrix and source bytes.

The lock binds the already frozen v1.2 internal-only selection, all selected
completed trace inputs, the suffix runner and its local helpers, the model /
checkpoint / VAE lineage, and the complete 32-job output matrix.  It does not
open or copy endpoint pixels, labels, FID, or endpoint representations.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from . import intervene_dit_v22_custom_trace_suffix as runner
    from . import reproduce_dit_imagenet256 as strict
except ImportError:  # pragma: no cover
    import intervene_dit_v22_custom_trace_suffix as runner
    import reproduce_dit_imagenet256 as strict


ROOT = Path(__file__).resolve().parents[1]
SELECTION_LOCK = ROOT / "experiments/locks/dit_v22_repairability_pilot_lock_v1_2"
SELECTION_LOCK_ID = "16acd0bffda207ed73ef78a62909e53997bef68baae66cdffedede1bb207fbd0"
SELECTION_PROTOCOL_ID = "f39c5a8bfbbc129d6e80ca5e38a07dfd886c6c41faff15337042127e78b3ae77"
TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_third_pool_v1_custom_traces_cfg_locked"
)
DIT_ROOT = Path("/data/users/zhoushunyu/eqvae/baselines/DiT")
CHECKPOINT = DIT_ROOT / "pretrained_models/DiT-XL-2-256x256.pt"
VAE_SNAPSHOT = Path(
    "/home/zhoushunyu/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/"
    "snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493"
)
OUTPUT_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_outputs"
)
SMOKE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_mechanics_smoke_v2_seed507_step149"
)
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_v22_repairability_execution_source_lock_v1"
ROLLBACK_STEPS = (109, 149)
SOURCE_FILES = (
    "intervene_dit_v22_custom_trace_suffix.py",
    "reproduce_dit_imagenet256.py",
    "sample_dit_imagenet256_custom.py",
    "trace_dit_imagenet256_custom_batch.py",
    "freeze_dit_v22_repairability_execution_sources.py",
    "run_dit_v22_repairability_pilot_shard.py",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or observed != canonical_sha256(payload):
        raise RuntimeError(f"invalid self hash: {path}/{key}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_selection() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = self_hashed(SELECTION_LOCK / "manifest.json", "identity_sha256")
    if (
        manifest.get("identity_sha256") != SELECTION_LOCK_ID
        or manifest.get("protocol_identity_sha256") != SELECTION_PROTOCOL_ID
        or manifest.get("artifact_kind") != "DIT_V22_REPAIRABILITY_PILOT_LOCK_V1_2"
    ):
        raise RuntimeError("selection lock identity changed")
    files = {row.get("name"): row for row in manifest.get("files", [])}
    for name in ("protocol.json", "source.py"):
        path = SELECTION_LOCK / name
        record = files.get(name)
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.is_symlink()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != strict.sha256_file(path)
        ):
            raise RuntimeError(f"selection lock member changed: {name}")
    protocol = self_hashed(SELECTION_LOCK / "protocol.json", "identity_sha256")
    if protocol.get("identity_sha256") != SELECTION_PROTOCOL_ID:
        raise RuntimeError("selection protocol identity changed")
    selected = protocol.get("selected_paths")
    if not isinstance(selected, list) or len(selected) != 16:
        raise RuntimeError("selection path axis changed")
    roles = [row.get("role") for row in selected if isinstance(row, dict)]
    if roles.count("joint_E_and_B") != 8 or roles.count(
        "B_only_exact_schedule_B_matched_control"
    ) != 8:
        raise RuntimeError("selection roles changed")
    return manifest, protocol


def validate_smoke(
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    vae_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    trace_dir = TRACE_ROOT / "third_pool_v1_seed507"
    trace = runner.load_saved_trace(
        trace_dir,
        source=source,
        checkpoint=checkpoint,
        vae_snapshot=vae_snapshot,
    )
    class Args:
        pass

    args = Args()
    args.trace_dir = trace_dir.resolve()
    args.target_slot = 0
    args.expect_class_id = 207
    args.rollback_sampling_step = 149
    args.pilot_lock = SELECTION_LOCK.resolve()
    args.dit_root = DIT_ROOT.resolve()
    args.checkpoint = CHECKPOINT.resolve()
    args.vae_snapshot = VAE_SNAPSHOT.resolve()
    args.outdir = SMOKE_ROOT.resolve()
    manifest = runner.build_manifest(
        args,
        trace,
        source=source,
        checkpoint=checkpoint,
        vae_snapshot=vae_snapshot,
    )
    runner.validate_bundle(SMOKE_ROOT, manifest=manifest, trace=trace, require_completion=True)
    stored = self_hashed(SMOKE_ROOT / runner.MANIFEST_NAME, "identity_sha256")
    return {
        "path": str(SMOKE_ROOT),
        "manifest_identity_sha256": stored["identity_sha256"],
        "manifest_file_sha256": strict.sha256_file(SMOKE_ROOT / runner.MANIFEST_NAME),
        "completion_file_sha256": strict.sha256_file(SMOKE_ROOT / runner.COMPLETION_NAME),
        "mechanical_validation_only": True,
        "endpoint_pixels_or_quality_opened": False,
    }


def freeze(output: Path) -> None:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite execution lock: {output}")
    if os.path.lexists(OUTPUT_ROOT):
        raise RuntimeError("formal repairability output root must be absent before source freeze")
    selection_manifest, selection = validate_selection()
    source = strict.validate_repository(DIT_ROOT, CHECKPOINT)
    checkpoint = strict.validate_checkpoint(CHECKPOINT)
    vae_snapshot = strict.validate_vae_snapshot(VAE_SNAPSHOT)

    selected = selection["selected_paths"]
    trace_records: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    seen_inputs: set[tuple[int, int]] = set()
    for selected_row in selected:
        seed = int(selected_row["global_seed"])
        class_id = int(selected_row["class_id"])
        slot = int(selected_row["class_slot"])
        role = str(selected_row["role"])
        pair_index = int(selected_row["pair_index"])
        key = (seed, class_id)
        if key in seen_inputs:
            raise RuntimeError("selection contains a duplicate seed/class path")
        seen_inputs.add(key)
        trace_dir = TRACE_ROOT / f"third_pool_v1_seed{seed}"
        trace = runner.load_saved_trace(
            trace_dir,
            source=source,
            checkpoint=checkpoint,
            vae_snapshot=vae_snapshot,
        )
        if trace.seed != seed or trace.classes[slot] != class_id:
            raise RuntimeError("selected trace seed/slot/class binding changed")
        trace_records.append(
            {
                "global_seed": seed,
                "class_id": class_id,
                "class_slot": slot,
                "root": str(trace_dir.resolve()),
                "trace_identity_sha256": trace.identity_sha256,
                "manifest_file_sha256": strict.sha256_file(trace_dir / runner.custom_trace.MANIFEST_NAME),
                "completion_file_sha256": strict.sha256_file(trace_dir / runner.custom_trace.COMPLETION_NAME),
                "trace_npz_sha256": strict.sha256_file(trace_dir / runner.custom_trace.TRACE_NAME),
            }
        )
        role_slug = "joint" if role == "joint_E_and_B" else "bonly"
        for rollback_step in ROLLBACK_STEPS:
            outdir = (
                OUTPUT_ROOT
                / f"pair{pair_index:02d}_{role_slug}_seed{seed}_class{class_id}_step{rollback_step}"
            )
            jobs.append(
                {
                    "job_index": len(jobs),
                    "pair_index": pair_index,
                    "role": role,
                    "global_seed": seed,
                    "class_id": class_id,
                    "class_slot": slot,
                    "rollback_sampling_step": rollback_step,
                    "trace_dir": str(trace_dir.resolve()),
                    "outdir": str(outdir),
                }
            )
    if len(jobs) != 32 or len({row["outdir"] for row in jobs}) != 32:
        raise RuntimeError("execution job matrix changed")

    smoke = validate_smoke(source, checkpoint, vae_snapshot)
    source_records = []
    for name in SOURCE_FILES:
        path = ROOT / "experiments" / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing real source file: {path}")
        source_records.append(
            {
                "name": f"sources/{name}",
                "origin": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": strict.sha256_file(path),
            }
        )

    contract: dict[str, Any] = {
        "schema_version": 1,
        "status": "EXECUTION_READY_RETROSPECTIVE_EXPLORATORY_ONLY",
        "selection_lock": {
            "path": str(SELECTION_LOCK),
            "manifest_identity_sha256": selection_manifest["identity_sha256"],
            "protocol_identity_sha256": selection["identity_sha256"],
        },
        "scientific_scope": {
            "question": selection["question"],
            "external_quality_inputs_used_by_execution": False,
            "all_fresh_attempts_retained": True,
            "attempt_ranking_or_best_of_n": False,
            "exploratory_only": True,
            "intervention_or_deployment_authority": False,
        },
        "decision_and_rollback_timing": selection["intervention"],
        "runner_contract": {
            "runner": runner.RUNNER_NAME,
            "fresh_attempts": runner.FRESH_ATTEMPTS,
            "branch_count_including_exact_replay": runner.BRANCH_COUNT,
            "rng_namespace": runner.RNG_NAMESPACE,
            "rollback_steps": list(ROLLBACK_STEPS),
        },
        "output_root": str(OUTPUT_ROOT),
        "output_root_absent_at_freeze": True,
        "lineage": {
            "dit_source": source,
            "checkpoint": checkpoint,
            "vae_snapshot": vae_snapshot,
        },
        "input_traces": trace_records,
        "jobs": jobs,
        "mechanics_smoke": smoke,
        "source_records": source_records,
        "forbidden_execution_inputs": selection["forbidden_selection_or_intervention_inputs"],
        "claim_limits": selection["evaluation_frozen_before_outputs"]["claim_limit"],
    }
    contract["identity_sha256"] = canonical_sha256(contract)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        sources_dir = staging / "sources"
        sources_dir.mkdir()
        for record in source_records:
            source_path = Path(record["origin"])
            target = staging / record["name"]
            shutil.copyfile(source_path, target)
            if (
                target.stat().st_size != record["bytes"]
                or strict.sha256_file(target) != record["sha256"]
            ):
                raise RuntimeError(f"source copy changed: {record['name']}")
        write_json(staging / "execution_contract.json", contract)
        files = [
            {
                "name": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": strict.sha256_file(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_REPAIRABILITY_EXECUTION_SOURCE_LOCK_V1",
            "execution_contract_identity_sha256": contract["identity_sha256"],
            "selection_lock_identity_sha256": SELECTION_LOCK_ID,
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "frozen",
        "output": str(output),
        "execution_contract_identity_sha256": contract["identity_sha256"],
        "job_count": len(jobs),
        "input_trace_count": len(trace_records),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    freeze(DEFAULT_OUTPUT)
