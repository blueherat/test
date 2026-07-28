"""Reject an RAE Flow/LPL pair unless its provenance and data stream match."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


PAIRED_MANIFEST_KEYS = (
    "source_checkpoint",
    "source_checkpoint_sha256",
    "source_checkpoint_type",
    "optimizer_state_at_branch_start",
    "config_sha256",
    "data_path",
    "dataset_split",
    "dataset_examples",
    "dataset_parquet_shards",
    "dataset_shard_names_sha256",
    "dataset_files_asserted_train_only",
    "global_seed",
    "world_size",
    "global_batch_size",
    "micro_batch_size",
    "grad_accum_steps",
    "branch_start_step",
    "endpoint_step",
    "precision",
    "tf32",
    "rae_decoder_sha256",
    "rae_statistics_sha256",
    "decoder_deterministic",
    "clean_estimate",
    "time_gate",
    "lpl_noise_to_signal_threshold",
    "lpl_max_samples_per_rank",
    "decoder_hidden_indices",
    "decoder_layer_weights",
    "outlier_quantile",
    "outlier_opening",
    "outlier_closing",
    "method_identity",
    "paper_code_available",
    "pairing_scope",
)

PAIRED_FINGERPRINT_KEYS = (
    "step",
    "indices_sha256",
    "images_sha256",
    "labels_sha256",
    "time_sha256",
    "noise_sha256",
    "noisy_latent_sha256",
    "target_velocity_sha256",
    "prediction_sha256",
    "labels",
    "time",
    "flow_loss",
    "eligible",
)

COPIED_SOURCE_FILES = (
    "train_rae_strict_lpl.py",
    "rae_strict_lpl.py",
    "rae_lpl_detach_audit.py",
)

STREAM_FIELDS = (
    "dataset_index",
    "label",
    "augmented_image_stride32",
    "time",
    "noise_channels8_stride4",
)

PAIRED_ENDPOINT_KEYS = (
    "epoch",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
    "scheduler_sha256",
    "optimizer_param_groups_sha256",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(value: Any) -> str:
    """Hash nested checkpoint state without relying on pickle byte stability."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def compare_keys(
    left: dict[str, Any],
    right: dict[str, Any],
    keys: tuple[str, ...],
    *,
    section: str,
) -> list[str]:
    return [
        f"{section}.{key}: {left.get(key)!r} != {right.get(key)!r}"
        for key in keys
        if left.get(key) != right.get(key)
    ]


def audit_metrics(
    path: Path,
    *,
    objective: str,
    branch_start_step: int,
    endpoint_step: int,
    lpl_weight: float,
) -> tuple[int, list[str]]:
    errors = []
    count = 0
    if not path.exists():
        return 0, [f"missing metrics file: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        row = json.loads(line)
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"{path}:{line_number} {key} is not finite")
        expected_step = int(branch_start_step) + count
        if int(row.get("step", -1)) != expected_step:
            errors.append(
                f"{path}:{line_number} step={row.get('step')} != {expected_step}"
            )
        if int(row.get("branch_update", -1)) != count:
            errors.append(
                f"{path}:{line_number} branch_update="
                f"{row.get('branch_update')} != {count}"
            )
        try:
            total = float(row["total_loss"])
            flow = float(row["flow_loss"])
            lpl = float(row["lpl_batch_contribution"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path}:{line_number} lacks scalar loss fields")
            continue
        expected_total = flow + float(lpl_weight) * lpl
        if not math.isclose(total, expected_total, rel_tol=1e-6, abs_tol=1e-7):
            errors.append(
                f"{path}:{line_number} total loss does not equal "
                "flow + weight * LPL"
            )
        if objective == "flow" and lpl != 0.0:
            errors.append(f"{path}:{line_number} Flow branch has nonzero LPL")
    expected_rows = int(endpoint_step) - int(branch_start_step)
    if count != expected_rows:
        errors.append(f"{path} has {count} rows, expected {expected_rows}")
    return count, errors


def audit_endpoint_checkpoint(
    branch: Path,
    *,
    branch_start_step: int,
    endpoint_step: int,
) -> tuple[dict[str, Any], list[str]]:
    checkpoint = branch / "checkpoints" / f"step-{int(endpoint_step):07d}.pt"
    if not checkpoint.exists():
        return {"path": str(checkpoint), "exists": False}, [
            f"missing endpoint checkpoint: {checkpoint}"
        ]
    errors = []
    try:
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:
        return {"path": str(checkpoint), "exists": True}, [
            f"failed to load endpoint checkpoint {checkpoint}: {error}"
        ]
    if int(state.get("step", -1)) != int(endpoint_step):
        errors.append(
            f"{checkpoint} step={state.get('step')} != {int(endpoint_step)}"
        )
    if int(state.get("branch_start_step", -1)) != int(branch_start_step):
        errors.append(
            f"{checkpoint} branch_start_step={state.get('branch_start_step')} "
            f"!= {int(branch_start_step)}"
        )
    required = {
        "model",
        "ema",
        "optimizer",
        "scheduler",
        "rng_cpu",
        "rng_cuda",
        "epoch",
    }
    missing = sorted(required.difference(state))
    if missing:
        errors.append(f"{checkpoint} is missing state keys: {missing}")
    return {
        "path": str(checkpoint),
        "exists": True,
        "bytes": checkpoint.stat().st_size,
        "step": state.get("step"),
        "branch_start_step": state.get("branch_start_step"),
        "epoch": state.get("epoch"),
        "rng_cpu_sha256": state_sha256(state.get("rng_cpu")),
        "rng_cuda_sha256": [
            state_sha256(value) for value in state.get("rng_cuda", ())
        ],
        "scheduler_sha256": state_sha256(state.get("scheduler")),
        "optimizer_param_groups_sha256": state_sha256(
            state.get("optimizer", {}).get("param_groups")
            if isinstance(state.get("optimizer"), dict)
            else None
        ),
        "state_keys": sorted(state),
    }, errors


def audit_pair(
    flow: Path,
    lpl: Path,
    *,
    require_endpoint_checkpoint: bool = True,
    minimum_free_memory_fraction: float = 0.10,
) -> dict[str, Any]:
    errors = []
    flow_manifest = read_json(flow / "manifest.json")
    lpl_manifest = read_json(lpl / "manifest.json")
    if flow_manifest.get("objective") != "flow":
        errors.append("the control branch objective is not flow")
    if lpl_manifest.get("objective") not in {"full", "lpl"}:
        errors.append("the treatment branch objective is not full LPL")
    errors.extend(
        compare_keys(
            flow_manifest,
            lpl_manifest,
            PAIRED_MANIFEST_KEYS,
            section="manifest",
        )
    )
    if float(flow_manifest.get("lpl_weight", float("nan"))) != 0.0:
        errors.append("the Flow control has a nonzero LPL weight")
    lpl_weight = float(lpl_manifest.get("lpl_weight", float("nan")))
    if not math.isfinite(lpl_weight) or lpl_weight <= 0.0:
        errors.append("the full-LPL treatment does not have a positive finite weight")
    if flow_manifest.get("cross_normalization") != "none":
        errors.append("the Flow control has unexpected feature normalization")
    lpl_normalization = str(lpl_manifest.get("cross_normalization", ""))
    if "differentiable prediction variance" not in lpl_normalization:
        errors.append("the full-LPL treatment is not prediction-stat cross-normalized")
    for name, manifest in (("flow", flow_manifest), ("lpl", lpl_manifest)):
        if manifest.get("dataset_split") != "train":
            errors.append(f"{name} did not use the train split")
        if manifest.get("evaluation_reference_loaded_by_trainer") is not False:
            errors.append(f"{name} trainer may have loaded an evaluation reference")
        for key in (
            "dataset_files_asserted_train_only",
            "encoder_frozen",
            "decoder_frozen",
            "frozen_boundary_runtime_assertions",
            "optimizer_exactly_stage2_parameters",
            "decoder_deterministic",
        ):
            if manifest.get(key) is not True:
                errors.append(f"{name}.{key} is not true")
        if manifest.get("fresh_initialization") is not False:
            errors.append(f"{name} did not start from a source checkpoint")
        if manifest.get("resumed_from_branch_checkpoint") is not False:
            errors.append(f"{name} resumed from a branch checkpoint")
        if manifest.get("resume_is_exact") is not True:
            errors.append(f"{name} is not marked as an exact fresh branch")
        if manifest.get("paper_code_available") is not False:
            errors.append(f"{name} does not disclose that official LPL code is unavailable")

    copied_sources = {}
    for filename in COPIED_SOURCE_FILES:
        flow_source = flow / filename
        lpl_source = lpl / filename
        if not flow_source.exists() or not lpl_source.exists():
            errors.append(f"missing copied source file: {filename}")
            continue
        flow_sha = file_sha256(flow_source)
        lpl_sha = file_sha256(lpl_source)
        if flow_sha != lpl_sha:
            errors.append(f"copied source SHA256 differs: {filename}")
        copied_sources[filename] = flow_sha

    flow_first = read_json(flow / "pair_fingerprint.json")
    lpl_first = read_json(lpl / "pair_fingerprint.json")
    errors.extend(
        compare_keys(
            flow_first,
            lpl_first,
            PAIRED_FINGERPRINT_KEYS,
            section="first_batch",
        )
    )

    world_size = int(flow_manifest.get("world_size", 0))
    expected_microbatches = (
        int(flow_manifest.get("endpoint_step", 0))
        - int(flow_manifest.get("branch_start_step", 0))
    ) * int(flow_manifest.get("grad_accum_steps", 0))
    streams = []
    for rank in range(world_size):
        flow_stream = read_json(flow / f"stream_audit_rank{rank}.json")
        lpl_stream = read_json(lpl / f"stream_audit_rank{rank}.json")
        for name, stream, objective in (
            ("flow", flow_stream, "flow"),
            ("lpl", lpl_stream, lpl_manifest.get("objective")),
        ):
            if int(stream.get("rank", -1)) != rank:
                errors.append(f"{name} stream rank {rank} has a rank mismatch")
            if int(stream.get("global_seed", -1)) != int(
                flow_manifest.get("global_seed", -2)
            ):
                errors.append(f"{name} stream rank {rank} has a seed mismatch")
            if stream.get("objective") != objective:
                errors.append(f"{name} stream rank {rank} has an objective mismatch")
            if tuple(stream.get("fields", ())) != STREAM_FIELDS:
                errors.append(f"{name} stream rank {rank} has unexpected hash fields")
            memory = stream.get("gpu_memory")
            if not isinstance(memory, dict):
                errors.append(f"{name} stream rank {rank} lacks GPU memory audit")
            else:
                free = float(memory.get("physical_free_at_end_mib", float("nan")))
                total = float(memory.get("physical_total_mib", float("nan")))
                if (
                    not math.isfinite(free)
                    or not math.isfinite(total)
                    or total <= 0.0
                ):
                    errors.append(f"{name} stream rank {rank} has invalid GPU memory audit")
                elif free / total < float(minimum_free_memory_fraction):
                    errors.append(
                        f"{name} stream rank {rank} retained only "
                        f"{free / total:.3%} free GPU memory"
                    )
        if flow_stream.get("sha256") != lpl_stream.get("sha256"):
            errors.append(f"rank {rank} data-stream SHA256 differs")
        if flow_stream.get("microbatches") != lpl_stream.get("microbatches"):
            errors.append(f"rank {rank} microbatch count differs")
        if int(flow_stream.get("microbatches", -1)) != expected_microbatches:
            errors.append(
                f"rank {rank} has {flow_stream.get('microbatches')} microbatches, "
                f"expected {expected_microbatches}"
            )
        streams.append(
            {
                "rank": rank,
                "sha256": flow_stream.get("sha256"),
                "microbatches": flow_stream.get("microbatches"),
            }
        )

    branch_start_step = int(flow_manifest.get("branch_start_step", 0))
    endpoint_step = int(flow_manifest.get("endpoint_step", 0))
    flow_rows, flow_metric_errors = audit_metrics(
        flow / "metrics.jsonl",
        objective="flow",
        branch_start_step=branch_start_step,
        endpoint_step=endpoint_step,
        lpl_weight=0.0,
    )
    lpl_rows, lpl_metric_errors = audit_metrics(
        lpl / "metrics.jsonl",
        objective="full",
        branch_start_step=branch_start_step,
        endpoint_step=endpoint_step,
        lpl_weight=lpl_weight,
    )
    errors.extend(flow_metric_errors)
    errors.extend(lpl_metric_errors)
    if flow_rows != lpl_rows:
        errors.append(f"metric row count differs: {flow_rows} != {lpl_rows}")

    if require_endpoint_checkpoint:
        flow_checkpoint, flow_checkpoint_errors = audit_endpoint_checkpoint(
            flow,
            branch_start_step=branch_start_step,
            endpoint_step=endpoint_step,
        )
        lpl_checkpoint, lpl_checkpoint_errors = audit_endpoint_checkpoint(
            lpl,
            branch_start_step=branch_start_step,
            endpoint_step=endpoint_step,
        )
        errors.extend(flow_checkpoint_errors)
        errors.extend(lpl_checkpoint_errors)
        errors.extend(
            compare_keys(
                flow_checkpoint,
                lpl_checkpoint,
                PAIRED_ENDPOINT_KEYS,
                section="endpoint_checkpoint",
            )
        )
    else:
        flow_checkpoint = {"required": False}
        lpl_checkpoint = {"required": False}

    return {
        "passed": not errors,
        "flow": str(flow),
        "lpl": str(lpl),
        "source_checkpoint_sha256": flow_manifest.get("source_checkpoint_sha256"),
        "training_seed": flow_manifest.get("global_seed"),
        "world_size": world_size,
        "copied_source_sha256": copied_sources,
        "expected_microbatches_per_rank": expected_microbatches,
        "minimum_free_memory_fraction": float(minimum_free_memory_fraction),
        "stream_audits": streams,
        "metric_rows": {"flow": flow_rows, "lpl": lpl_rows},
        "endpoint_checkpoints": {
            "flow": flow_checkpoint,
            "lpl": lpl_checkpoint,
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--lpl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-endpoint-checkpoint", action="store_true")
    parser.add_argument("--minimum-free-memory-fraction", type=float, default=0.10)
    args = parser.parse_args()

    result = audit_pair(
        args.flow.expanduser().resolve(),
        args.lpl.expanduser().resolve(),
        require_endpoint_checkpoint=not args.allow_missing_endpoint_checkpoint,
        minimum_free_memory_fraction=args.minimum_free_memory_fraction,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
