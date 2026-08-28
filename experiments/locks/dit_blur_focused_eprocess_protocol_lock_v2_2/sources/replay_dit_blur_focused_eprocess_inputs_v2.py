#!/usr/bin/env python3
"""Build method-v2 observer inputs from a completed preterminal DiT trace.

Replay tensors are intentionally unchanged from v1: the correction is wholly
inside the operational Q* accounting after ``theta`` and B have been observed.
This adapter reuses the audited v1 trace/model/VAE routines, validates the
result against the v2 input contract, and emits a v2-specific lineage receipt.
It never opens endpoints, labels, reviews, or external representations.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from . import calibrate_dit_blur_focused_eprocess_v2 as calibrate
    from . import observe_dit_blur_focused_eprocess_v2 as core
    from . import replay_dit_blur_focused_eprocess_inputs as v1
    from . import reproduce_dit_imagenet256 as strict
except ImportError:  # pragma: no cover
    import calibrate_dit_blur_focused_eprocess_v2 as calibrate
    import observe_dit_blur_focused_eprocess_v2 as core
    import replay_dit_blur_focused_eprocess_inputs as v1
    import reproduce_dit_imagenet256 as strict


EXPERIMENT = "dit_blur_focused_eprocess_replay_input_label_free_v2"
SCHEMA_VERSION = 2
OUTPUT_NAME = "observer_input.npz"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
V1_REPLAY_SOURCE_SHA256 = "723875e38aca1d300ca1f48085b8723cc0b9c4e35633b14eb6abc07f360053b1"
STRICT_REPRODUCER_SOURCE_SHA256 = "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"
TRACE_RUNNER_SOURCE_SHA256 = "6f4c94d3720717c3c7ce913ca6e928a30641aa5e4ddb0922bc2894e79aaf4e79"
CUSTOM_SAMPLER_SOURCE_SHA256 = "40a3a29a39f30545298e0d4c86367c87114751f3f7766560a649af91ecfd3e2c"

if core._sha256_file(Path(v1.__file__).resolve()) != V1_REPLAY_SOURCE_SHA256:
    raise RuntimeError("method v2 refuses an unpinned v1 replay dependency")
if core._sha256_file(Path(strict.__file__).resolve()) != STRICT_REPRODUCER_SOURCE_SHA256:
    raise RuntimeError("method v2 refuses an unpinned strict reproducer dependency")
if core._sha256_file(Path(v1.trace_runner.__file__).resolve()) != TRACE_RUNNER_SOURCE_SHA256:
    raise RuntimeError("method v2 refuses an unpinned trace-runner dependency")
if core._sha256_file(Path(v1.trace_runner.custom.__file__).resolve()) != CUSTOM_SAMPLER_SOURCE_SHA256:
    raise RuntimeError("method v2 refuses an unpinned custom-sampler dependency")


def publish(args: argparse.Namespace) -> Path:
    if args.outdir.exists() or args.outdir.is_symlink():
        raise RuntimeError(f"refusing pre-existing output path: {args.outdir}")
    arrays, metadata = v1._read_trace_preterminal(args.trace_dir)
    calibration_gate, calibration_score, calibration = v1._read_calibration(
        args.calibration, metadata["classes"]
    )
    observer_arrays, execution = v1.build_observer_arrays(
        arrays,
        metadata,
        calibration_gate=calibration_gate,
        calibration_score=calibration_score,
        dit_root=args.dit_root,
        checkpoint=args.checkpoint,
        vae_snapshot=args.vae_snapshot,
        decode_batch_size=args.decode_batch_size,
    )
    core.validate_observer_input(observer_arrays)
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent))
    try:
        output_path = staging / OUTPUT_NAME
        with output_path.open("wb") as handle:
            np.savez(handle, **observer_arrays)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "execution_ready": False,
            "scientific_protocol_requirement": (
                "a compatible immutable B/E scientific protocol bound to method v2"
            ),
            "trace": metadata,
            "calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": strict.sha256_file(args.calibration),
                "identity_sha256": calibration["identity_sha256"],
                "disjoint_calibration_required_for_confirmatory_mechanics_and_quality": True,
                "in_sample_operational_LR_exactness_may_remain_if_preinnovation_predictable": True,
            },
            "implementation": {
                "adapter_path": str(Path(__file__).resolve()),
                "adapter_sha256": strict.sha256_file(Path(__file__).resolve()),
                "v2_core_path": str(Path(core.__file__).resolve()),
                "v2_core_sha256": strict.sha256_file(Path(core.__file__).resolve()),
                "audited_v1_replay_dependency_path": str(Path(v1.__file__).resolve()),
                "audited_v1_replay_dependency_sha256": strict.sha256_file(
                    Path(v1.__file__).resolve()
                ),
            },
            "execution": execution,
            "output": {
                "relative_path": OUTPUT_NAME,
                "bytes": output_path.stat().st_size,
                "sha256": strict.sha256_file(output_path),
                "arrays": {
                    name: core._array_record(observer_arrays[name])
                    for name in core.INPUT_ARRAY_NAMES
                },
            },
        }
        manifest["identity_sha256"] = core._sha256_json(manifest)
        core._atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": manifest["identity_sha256"],
            "manifest_sha256": strict.sha256_file(staging / MANIFEST_NAME),
            "output_sha256": strict.sha256_file(output_path),
        }
        core._atomic_json_dump(completion, staging / COMPLETION_NAME)
        os.replace(staging, args.outdir)
        return args.outdir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    v1.self_test()
    calibrate.self_test()
    if core.CHECKPOINTS != v1.core.CHECKPOINTS:
        raise AssertionError("v2 replay checkpoints differ from the audited adapter")
    if core.SHIFTED_INTERNAL_TIMESTEPS != v1.core.SHIFTED_INTERNAL_TIMESTEPS:
        raise AssertionError("v2 replay cross-scale mapping changed")
    print(
        "v2 replay self-test passed: unchanged pre-innovation inputs, v2 validation, "
        "and explicit disjoint-calibration requirement for confirmatory use"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--dit-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--decode-batch-size", type=int, default=18)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        if any(
            value is not None
            for value in (
                args.trace_dir,
                args.calibration,
                args.dit_root,
                args.checkpoint,
                args.vae_snapshot,
                args.outdir,
            )
        ):
            raise RuntimeError("--self-test cannot be combined with real inputs")
        self_test()
        return 0
    required = ("trace_dir", "calibration", "dit_root", "checkpoint", "vae_snapshot", "outdir")
    if any(getattr(args, name) is None for name in required):
        raise RuntimeError("real replay requires trace, calibration, model, VAE, and output paths")
    if args.decode_batch_size <= 0:
        raise RuntimeError("--decode-batch-size must be positive")
    for name in required:
        value = getattr(args, name)
        setattr(args, name, value.expanduser().absolute())
    args.dit_root = args.dit_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.vae_snapshot = args.vae_snapshot.resolve()
    args.trace_dir = args.trace_dir.resolve()
    args.calibration = args.calibration.resolve()
    output = publish(args)
    print(f"published method-v2 replay input: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
