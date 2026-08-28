#!/usr/bin/env python3
"""Freeze corrected blur-latched directional e-process method v2.1.

This produces a non-executable method-definition lock.  It validates and
preserves immutable v1, runs all method/calibration/replay self-tests, freezes
the all-h conditional matched-Q power receipt, and copies every direct or v1
dependency needed to audit the implementation.  It opens no real trace,
endpoint, label, review, score, feature, or embedding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from . import calibrate_dit_blur_focused_eprocess as calibration_v1
    from . import calibrate_dit_blur_focused_eprocess_v2 as calibration_v2
    from . import freeze_dit_blur_focused_eprocess_protocol as locker_v1
    from . import observe_dit_blur_focused_eprocess as observer_v1
    from . import observe_dit_blur_focused_eprocess_v2 as core
    from . import replay_dit_blur_focused_eprocess_inputs as replay_v1
    from . import replay_dit_blur_focused_eprocess_inputs_v2 as replay_v2
except ImportError:  # pragma: no cover
    import calibrate_dit_blur_focused_eprocess as calibration_v1
    import calibrate_dit_blur_focused_eprocess_v2 as calibration_v2
    import freeze_dit_blur_focused_eprocess_protocol as locker_v1
    import observe_dit_blur_focused_eprocess as observer_v1
    import observe_dit_blur_focused_eprocess_v2 as core
    import replay_dit_blur_focused_eprocess_inputs as replay_v1
    import replay_dit_blur_focused_eprocess_inputs_v2 as replay_v2


SCHEMA_VERSION = 2
LOCK_NAME = "dit_blur_focused_eprocess_protocol_lock_v2_1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "locks" / LOCK_NAME
DEFAULT_PROTOCOL = ROOT / "experiments" / "configs" / "dit_blur_focused_eprocess_v2.json"
DEFAULT_DOC = ROOT / "docs" / "DIT_BLUR_LATCHED_DIRECTIONAL_EPROCESS_V2_THEORY_ZH.md"
DEFAULT_TEST = ROOT / "tests" / "test_observe_dit_blur_focused_eprocess_v2.py"
V1_LOCK = ROOT / "experiments" / "locks" / "dit_blur_focused_eprocess_protocol_lock_v1"
EXPECTED_V1_IDENTITY = "facef0f59d1f10cde339440db3bc47dc26ca7fcef012faca01f7f2dfbb31b985"
SUPERSEDED_V2_LOCK = ROOT / "experiments" / "locks" / "dit_blur_focused_eprocess_protocol_lock_v2"
EXPECTED_SUPERSEDED_V2_IDENTITY = "8932af74660eeab9ed3961ae598432b1af61d56331ddefff9005de7428ab618a"
SOURCE_FILES = {
    "sources/observe_dit_blur_focused_eprocess_v2.py": Path(core.__file__).resolve(),
    "sources/observe_dit_blur_focused_eprocess.py": Path(observer_v1.__file__).resolve(),
    "sources/calibrate_dit_blur_focused_eprocess_v2.py": Path(calibration_v2.__file__).resolve(),
    "sources/calibrate_dit_blur_focused_eprocess.py": Path(calibration_v1.__file__).resolve(),
    "sources/replay_dit_blur_focused_eprocess_inputs_v2.py": Path(replay_v2.__file__).resolve(),
    "sources/replay_dit_blur_focused_eprocess_inputs.py": Path(replay_v1.__file__).resolve(),
    "sources/reproduce_dit_imagenet256.py": ROOT / "experiments/reproduce_dit_imagenet256.py",
    "sources/trace_dit_imagenet256_custom_batch.py": ROOT / "experiments/trace_dit_imagenet256_custom_batch.py",
    "sources/sample_dit_imagenet256_custom.py": ROOT / "experiments/sample_dit_imagenet256_custom.py",
    "sources/freeze_dit_blur_focused_eprocess_protocol_v2.py": Path(__file__).resolve(),
    "sources/freeze_dit_blur_focused_eprocess_protocol.py": Path(locker_v1.__file__).resolve(),
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _copy_regular(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"source must be regular and non-symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    os.replace(temporary, destination)


def _records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"lock member may not be a symlink: {path}")
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": core._sha256_file(path),
            }
        )
    return rows


def _isolated_source_selftest(root: Path) -> dict[str, Any]:
    """Execute the copied sources with no repository PYTHONPATH fallback."""

    source_root = root / "sources"
    commands = [
        [sys.executable, str(source_root / "observe_dit_blur_focused_eprocess_v2.py"), "--self-test"],
        [sys.executable, str(source_root / "calibrate_dit_blur_focused_eprocess_v2.py"), "--self-test"],
        [sys.executable, str(source_root / "replay_dit_blur_focused_eprocess_inputs_v2.py"), "--self-test"],
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    rows = []
    with tempfile.TemporaryDirectory(prefix="dit-v2-isolated-cwd-") as temporary:
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=temporary,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "isolated frozen-source self-test failed: "
                    f"{Path(command[1]).name}\n{completed.stdout}\n{completed.stderr}"
                )
            rows.append(
                {
                    "entrypoint": Path(command[1]).name,
                    "returncode": completed.returncode,
                    "stdout_last_line": (
                        completed.stdout.strip().splitlines()[-1]
                        if completed.stdout.strip()
                        else ""
                    ),
                    "stderr_empty": completed.stderr == "",
                }
            )
    return {
        "status": "PASS",
        "repository_pythonpath_available": False,
        "bytecode_written": False,
        "entrypoints": rows,
    }


def _validate_v1() -> dict[str, Any]:
    manifest = locker_v1.validate_lock(V1_LOCK)
    if manifest.get("identity_sha256") != EXPECTED_V1_IDENTITY:
        raise RuntimeError("immutable v1 method identity changed")
    return manifest


def _validate_superseded_v2() -> dict[str, Any]:
    manifest_path = SUPERSEDED_V2_LOCK / "manifest.json"
    completion_path = SUPERSEDED_V2_LOCK / "completion.json"
    manifest = _load_json(manifest_path)
    completion = _load_json(completion_path)
    identity = dict(manifest)
    observed = identity.pop("identity_sha256", None)
    rows = _records(SUPERSEDED_V2_LOCK)
    if (
        observed != core._sha256_json(identity)
        or observed != EXPECTED_SUPERSEDED_V2_IDENTITY
        or manifest.get("lock_name") != "dit_blur_focused_eprocess_protocol_lock_v2"
        or manifest.get("execution_ready") is not False
        or rows != manifest.get("files")
        or completion.get("identity_sha256") != observed
        or completion.get("manifest_sha256") != core._sha256_file(manifest_path)
        or completion.get("execution_ready") is not False
    ):
        raise RuntimeError("superseded immutable v2 method lock changed")
    return manifest


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("scientific_revision") != "v2.1"
        or protocol.get("status") != "METHOD_PROTOCOL_ONLY_NOT_EXECUTION_READY"
        or protocol.get("execution_ready") is not False
    ):
        raise RuntimeError("v2 method status/schema/readiness changed")
    supersession = protocol.get("supersedes_method_v1")
    if (
        not isinstance(supersession, dict)
        or supersession.get("v1_identity_sha256") != EXPECTED_V1_IDENTITY
        or supersession.get("preserved_immutable") is not True
    ):
        raise RuntimeError("v2 does not bind immutable v1")
    superseded_v2 = protocol.get("supersedes_method_v2")
    if (
        not isinstance(superseded_v2, dict)
        or superseded_v2.get("v2_identity_sha256")
        != EXPECTED_SUPERSEDED_V2_IDENTITY
        or superseded_v2.get("preserved_immutable") is not True
        or superseded_v2.get("real_data_opened_before_correction") is not False
    ):
        raise RuntimeError("v2.1 does not bind/supersede immutable v2")
    cross = protocol.get("cross_scale_components")
    if (
        not isinstance(cross, dict)
        or cross.get("additive_heat_shifts") != list(core.HEAT_SHIFTS)
        or cross.get("effective_checkpoint_counts")
        != list(core.EFFECTIVE_STEP_COUNT_PER_SCALE)
        or cross.get("path_mixture_weights") != list(core.MIXTURE_WEIGHTS)
        or cross.get("initial_e_values") != [1.0, 1.0]
    ):
        raise RuntimeError("v2 scale/mixture/initial-e contract changed")
    calibration = protocol.get("label_free_calibration")
    if (
        not isinstance(calibration, dict)
        or calibration.get("count_per_selected_class")
        != calibration_v2.CALIBRATION_COUNT_PER_CLASS
        or calibration.get("confirmation_seed_disjointness_required") is not True
        or "may remain exact" not in str(
            calibration.get("in_sample_E_on_threshold_fitting_paths")
        )
        or "cannot support fresh rank" not in str(
            calibration.get("in_sample_E_on_threshold_fitting_paths")
        )
    ):
        raise RuntimeError("v2 calibration/disjointness boundary changed")
    q_star = protocol.get("blur_latched_directional_operational_Q_star")
    if (
        not isinstance(q_star, dict)
        or q_star.get("direction_norm_floor") != core.RAW_DIRECTION_NORM_FLOOR
        or q_star.get("per_step_information") != "kappa_d=K_total/h_d with K_total=2"
        or q_star.get("unused_allowance_carried_forward") is not False
        or q_star.get("total_K_for_every_started_path") != core.TOTAL_K_PER_SCALE
        or q_star.get("anytime_alpha") != core.ALPHA_E
        or q_star.get("operational_exactness") is not True
        or q_star.get("ideal_heat_marginal_ratio_claimed") is not False
    ):
        raise RuntimeError("v2 directional Q* contract changed")
    mechanics = protocol.get("pre_label_confirmation_path_mechanics_gate")
    if (
        not isinstance(mechanics, dict)
        or mechanics.get("minimum_confirmation_paths")
        != core.PATH_MECHANICS_MINIMUM_SAMPLES
        or mechanics.get("minimum_qualifying_started_paths_per_scale")
        != core.PATH_MECHANICS_MINIMUM_STARTED_PATHS_PER_SCALE
        or mechanics.get("minimum_qualifying_started_classes_per_scale")
        != core.PATH_MECHANICS_MINIMUM_STARTED_CLASSES_PER_SCALE
        or mechanics.get("minimum_complete_coverage_fraction_among_started_paths")
        != core.PATH_MECHANICS_COMPLETE_COVERAGE_FRACTION_MINIMUM
        or mechanics.get("maximum_last_valid_fallback_fraction_among_started_steps_per_scale")
        != core.PATH_MECHANICS_MAX_REUSED_DIRECTION_FRACTION
        or "before any confirmation label" not in str(mechanics.get("timing"))
    ):
        raise RuntimeError("v2 confirmation mechanics gate changed")
    power = protocol.get("pre_real_matched_Q_conditional_power_gate")
    expected_power = 0.5 * math.erfc(
        (math.log(20.0) - core.TOTAL_K_PER_SCALE)
        / math.sqrt(4.0 * core.TOTAL_K_PER_SCALE)
    )
    if (
        not isinstance(power, dict)
        or not math.isclose(
            float(power.get("conditional_terminal_power_lower_bound", -1.0)),
            expected_power,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or power.get("minimum_required") != core.MATCHED_Q_ANYTIME_POWER_MINIMUM
        or power.get("allowed_h_by_scale") != [[3, 4, 5], [3, 4, 5, 6, 7, 8]]
    ):
        raise RuntimeError("v2 conditional matched-Q power contract changed")
    ablations = protocol.get("fixed_ablations")
    if not isinstance(ablations, dict) or not {
        "E_no_state_gate",
        "E_first_hit_full_budget",
    }.issubset(ablations):
        raise RuntimeError("v2 exact ablation set changed")
    forbidden_text = " ".join(protocol.get("forbidden_method_inputs", []))
    for token in ("endpoint", "FID", "Inception", "DINO", "CLIP", "quality"):
        if token not in forbidden_text:
            raise RuntimeError(f"v2 forbidden-input boundary omits {token}")


def validate_lock(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("v2 method lock is missing or indirect")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    manifest = _load_json(manifest_path)
    completion = _load_json(completion_path)
    protocol = _load_json(root / "protocol.json")
    _validate_protocol(protocol)
    identity = dict(manifest)
    observed = identity.pop("identity_sha256", None)
    if (
        observed != core._sha256_json(identity)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("lock_name") != LOCK_NAME
        or manifest.get("status") != "METHOD_PROTOCOL_FROZEN_EXECUTION_BLOCKED"
        or manifest.get("execution_ready") is not False
        or manifest.get("v1_identity_sha256") != EXPECTED_V1_IDENTITY
        or manifest.get("superseded_v2_identity_sha256")
        != EXPECTED_SUPERSEDED_V2_IDENTITY
    ):
        raise RuntimeError("v2 method manifest identity/status mismatch")
    rows = _records(root)
    if rows != manifest.get("files") or core._sha256_json(rows) != manifest.get("files_sha256"):
        raise RuntimeError("v2 method exact member tree changed")
    expected_completion = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": observed,
        "manifest_sha256": core._sha256_file(manifest_path),
        "files_sha256": manifest["files_sha256"],
        "file_count": len(rows),
        "execution_ready": False,
    }
    if completion != expected_completion:
        raise RuntimeError("v2 completion receipt mismatch")
    power = _load_json(root / "matched_q_conditional_power_gate.json")
    if (
        core._sha256_json(power) != manifest.get("matched_q_power_gate_identity")
        or power.get("passes") is not True
        or power.get("dependence_robust_conditional_terminal_power_lower_bound", 0.0)
        < core.MATCHED_Q_ANYTIME_POWER_MINIMUM
    ):
        raise RuntimeError("v2 matched-Q conditional power receipt changed")
    adaptive = _load_json(root / "adaptive_predictable_null_audit.json")
    if (
        core._sha256_json(adaptive) != manifest.get("adaptive_null_audit_identity")
        or adaptive.get("passes") is not True
        or adaptive.get("anytime_trigger_fraction_under_P", 1.0) > core.ALPHA_E + 0.005
    ):
        raise RuntimeError("v2 adaptive predictable-null receipt changed")
    return manifest


def freeze(*, output: Path, protocol_path: Path, doc_path: Path, test_path: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing pre-existing lock path: {output}")
    for path in (protocol_path, doc_path, test_path, *SOURCE_FILES.values()):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular v2 method source: {path}")
    v1_manifest = _validate_v1()
    v2_manifest = _validate_superseded_v2()
    protocol = _load_json(protocol_path)
    _validate_protocol(protocol)
    core.self_test()
    calibration_v2.self_test()
    replay_v2.self_test()
    power = core.matched_q_power_reference()
    if power.get("passes") is not True:
        raise RuntimeError("v2 all-h conditional matched-Q power gate failed")
    adaptive = core.adaptive_predictable_null_reference()
    if adaptive.get("passes") is not True:
        raise RuntimeError("v2 adaptive predictable-null audit failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _copy_regular(protocol_path, staging / "protocol.json")
        _copy_regular(doc_path, staging / "theory_zh.md")
        _copy_regular(test_path, staging / "sources" / "test_observer_v2.py")
        for relative, source in SOURCE_FILES.items():
            _copy_regular(source, staging / relative)
        for name in ("protocol.json", "manifest.json", "completion.json"):
            _copy_regular(V1_LOCK / name, staging / "upstream_v1_lock" / name)
            _copy_regular(
                SUPERSEDED_V2_LOCK / name,
                staging / "upstream_superseded_v2_lock" / name,
            )
        core._atomic_json_dump(power, staging / "matched_q_conditional_power_gate.json")
        core._atomic_json_dump(adaptive, staging / "adaptive_predictable_null_audit.json")
        isolated = _isolated_source_selftest(staging)
        core._atomic_json_dump(isolated, staging / "isolated_source_selftest.json")
        rows = _records(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "lock_name": LOCK_NAME,
            "status": "METHOD_PROTOCOL_FROZEN_EXECUTION_BLOCKED",
            "execution_ready": False,
            "v1_identity_sha256": v1_manifest["identity_sha256"],
            "v1_preserved_immutable": True,
            "superseded_v2_identity_sha256": v2_manifest["identity_sha256"],
            "superseded_v2_preserved_immutable": True,
            "real_data_opened_before_v2_1_correction": False,
            "real_trace_endpoint_label_review_score_embedding_opened": False,
            "disjoint_calibration_required_for_fresh_rank_mechanics_and_confirmation": True,
            "in_sample_preinnovation_predictable_threshold_can_retain_operational_LR_exactness": True,
            "pinned_dependency_hashes": {
                "v1_observer": core.V1_OBSERVER_SOURCE_SHA256,
                "v1_calibrator": calibration_v2.V1_CALIBRATOR_SOURCE_SHA256,
                "v1_replay": replay_v2.V1_REPLAY_SOURCE_SHA256,
                "strict_reproducer": replay_v2.STRICT_REPRODUCER_SOURCE_SHA256,
                "trace_runner": replay_v2.TRACE_RUNNER_SOURCE_SHA256,
                "custom_sampler": replay_v2.CUSTOM_SAMPLER_SOURCE_SHA256,
            },
            "files": rows,
            "files_sha256": core._sha256_json(rows),
            "matched_q_power_gate_identity": core._sha256_json(power),
            "adaptive_null_audit_identity": core._sha256_json(adaptive),
        }
        manifest["identity_sha256"] = core._sha256_json(manifest)
        core._atomic_json_dump(manifest, staging / "manifest.json")
        completion = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": manifest["identity_sha256"],
            "manifest_sha256": core._sha256_file(staging / "manifest.json"),
            "files_sha256": manifest["files_sha256"],
            "file_count": len(rows),
            "execution_ready": False,
        }
        core._atomic_json_dump(completion, staging / "completion.json")
        os.replace(staging, output)
        validate_lock(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="dit-blur-eprocess-v2-lock-test-") as temporary:
        root = Path(temporary)
        result = freeze(
            output=root / "lock",
            protocol_path=DEFAULT_PROTOCOL,
            doc_path=DEFAULT_DOC,
            test_path=DEFAULT_TEST,
        )
        manifest = validate_lock(result)
        if manifest.get("execution_ready") is not False:
            raise AssertionError("method v2 lock unexpectedly authorizes execution")
    print("v2.1 lock self-test passed: immutable v1/v2 lineage, exact tree, all-h power, blocked execution")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        self_test()
        return 0
    output = freeze(
        output=args.output.expanduser().absolute(),
        protocol_path=args.protocol.expanduser().resolve(),
        doc_path=args.doc.expanduser().resolve(),
        test_path=args.test.expanduser().resolve(),
    )
    manifest = validate_lock(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "identity_sha256": manifest["identity_sha256"],
                "execution_ready": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
