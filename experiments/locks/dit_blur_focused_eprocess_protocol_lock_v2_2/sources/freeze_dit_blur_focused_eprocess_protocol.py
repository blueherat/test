#!/usr/bin/env python3
"""Freeze the label-free B-gated e-process method protocol and source bundle.

This lock is intentionally not a pool-execution authorization.  It records
that the method conflicts with the existing event-rich B/C scientific v3 and
requires either an explicit pre-sampling scientific v4 or a later independent
pool.  No trace, endpoint, label, review, score, or embedding is read.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    from . import calibrate_dit_blur_focused_eprocess as calibrate
    from . import observe_dit_blur_focused_eprocess as core
    from . import replay_dit_blur_focused_eprocess_inputs as replay
except ImportError:  # pragma: no cover
    import calibrate_dit_blur_focused_eprocess as calibrate
    import observe_dit_blur_focused_eprocess as core
    import replay_dit_blur_focused_eprocess_inputs as replay


SCHEMA_VERSION = 1
LOCK_NAME = "dit_blur_focused_eprocess_protocol_lock_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "locks" / LOCK_NAME
DEFAULT_PROTOCOL = ROOT / "experiments" / "configs" / "dit_blur_focused_eprocess_v1.json"
DEFAULT_DOC = ROOT / "docs" / "DIT_BLUR_FOCUSED_EPROCESS_THEORY_ZH.md"
DEFAULT_TEST = ROOT / "tests" / "test_observe_dit_blur_focused_eprocess.py"
SOURCE_FILES = {
    "sources/observer_core.py": Path(core.__file__).resolve(),
    "sources/label_free_calibrator.py": Path(calibrate.__file__).resolve(),
    "sources/replay_adapter.py": Path(replay.__file__).resolve(),
    "sources/protocol_locker.py": Path(__file__).resolve(),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise RuntimeError("method protocol schema changed")
    if protocol.get("status") != "METHOD_PROTOCOL_ONLY_NOT_EXECUTION_READY":
        raise RuntimeError("method protocol status must remain execution-blocked")
    if protocol.get("execution_ready") is not False:
        raise RuntimeError("method protocol must not authorize real execution")
    compatibility = protocol.get("scientific_protocol_compatibility")
    if not isinstance(compatibility, dict):
        raise RuntimeError("scientific protocol compatibility block is missing")
    if compatibility.get("current_real_event_screen_samples") != 0:
        raise RuntimeError("method lock assumes zero real event-screen samples")
    if "incompatible" not in str(compatibility.get("existing_event_rich_v3", "")):
        raise RuntimeError("existing B/C v3 incompatibility is not explicit")
    generator = protocol.get("frozen_generator")
    if not isinstance(generator, dict) or generator.get("cfg_scale") != 4.0:
        raise RuntimeError("frozen generator contract changed")
    q_star = protocol.get("blur_gated_operational_Q_star")
    if not isinstance(q_star, dict):
        raise RuntimeError("Q* block is missing")
    if q_star.get("per_scale_total_K") != core.TOTAL_K_PER_SCALE:
        raise RuntimeError("protocol K_total differs from implementation")
    if q_star.get("anytime_alpha") != core.ALPHA_E:
        raise RuntimeError("protocol alpha differs from implementation")
    if q_star.get("operational_exactness") is not True:
        raise RuntimeError("operational exactness scope is missing")
    if q_star.get("ideal_heat_marginal_ratio_claimed") is not False:
        raise RuntimeError("protocol improperly claims ideal marginal exactness")
    components = protocol.get("cross_scale_components")
    if not isinstance(components, dict):
        raise RuntimeError("cross-scale component block is missing")
    if components.get("additive_heat_shifts") != list(core.HEAT_SHIFTS):
        raise RuntimeError("protocol heat shifts differ from implementation")
    if components.get("path_mixture_weights") != list(core.MIXTURE_WEIGHTS):
        raise RuntimeError("protocol mixture weights differ from implementation")
    calibration = protocol.get("label_free_calibration")
    if not isinstance(calibration, dict):
        raise RuntimeError("label-free calibration block is missing")
    if calibration.get("count_per_selected_class") != (
        calibrate.CALIBRATION_COUNT_PER_CLASS
    ):
        raise RuntimeError("calibration path count differs from implementation")
    if "17th ascending" not in str(calibration.get("state_gate_threshold", "")):
        raise RuntimeError("state-gate order statistic differs from implementation")
    if "19th ascending" not in str(calibration.get("pure_B_alarm_threshold", "")):
        raise RuntimeError("pure-B order statistic differs from implementation")
    family = protocol.get("candidate_family")
    if (
        not isinstance(family, dict)
        or family.get("family_size") != 2
        or [row.get("id") for row in family.get("co_primary", [])]
        != ["B_persistence", "E_blur_gated_running_max_log"]
    ):
        raise RuntimeError("co-primary family differs from frozen B/E")
    forbidden = protocol.get("forbidden_method_inputs")
    forbidden_text = " ".join(forbidden) if isinstance(forbidden, list) else ""
    for token in ("FID", "Inception", "DINO", "CLIP", "endpoint"):
        if token not in forbidden_text:
            raise RuntimeError(f"forbidden method-input boundary omits {token}")
    power = protocol.get("pre_real_trace_matched_Q_power_gate")
    if not isinstance(power, dict):
        raise RuntimeError("matched-Q power gate is missing")
    if power.get("draws") != core.MATCHED_Q_POWER_DRAWS:
        raise RuntimeError("matched-Q draw count changed")
    if power.get("minimum_over_two_matched_components_anytime_power_at_least") != (
        core.MATCHED_Q_ANYTIME_POWER_MINIMUM
    ):
        raise RuntimeError("matched-Q minimum power changed")


def _copy_regular(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"source must be regular and non-symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    os.replace(temporary, destination)


def _records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in ("manifest.json", "completion.json"):
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": core._sha256_file(path),
            }
        )
    return rows


def validate_lock(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError("method lock is incomplete")
    manifest = _load_json(manifest_path)
    completion = _load_json(completion_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != (
        "METHOD_PROTOCOL_FROZEN_EXECUTION_BLOCKED"
    ):
        raise RuntimeError("method lock manifest status is invalid")
    expected_identity = dict(manifest)
    observed_identity = expected_identity.pop("identity_sha256", None)
    if observed_identity != core._sha256_json(expected_identity):
        raise RuntimeError("method lock identity hash is invalid")
    rows = _records(root)
    if rows != manifest.get("files") or core._sha256_json(rows) != manifest.get(
        "files_sha256"
    ):
        raise RuntimeError("method lock file records changed")
    expected_completion = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": observed_identity,
        "manifest_sha256": core._sha256_file(manifest_path),
        "files_sha256": manifest["files_sha256"],
        "file_count": len(rows),
        "execution_ready": False,
    }
    if completion != expected_completion:
        raise RuntimeError("method lock completion record is invalid")
    return manifest


def freeze(
    *, output: Path, protocol_path: Path, doc_path: Path, test_path: Path
) -> Path:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing pre-existing lock path: {output}")
    for path in (protocol_path, doc_path, test_path, *SOURCE_FILES.values()):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular method-lock source: {path}")
    protocol = _load_json(protocol_path)
    _validate_protocol(protocol)
    core.self_test()
    calibrate.self_test()
    replay.self_test()
    power = core.matched_q_power_reference()
    if power.get("passes") is not True:
        raise RuntimeError("frozen matched-Q power gate did not pass")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _copy_regular(protocol_path, staging / "protocol.json")
        _copy_regular(doc_path, staging / "theory_zh.md")
        _copy_regular(test_path, staging / "sources" / "test_observer.py")
        for relative, source in SOURCE_FILES.items():
            _copy_regular(source, staging / relative)
        core._atomic_json_dump(power, staging / "matched_q_power_gate.json")
        rows = _records(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "lock_name": LOCK_NAME,
            "status": "METHOD_PROTOCOL_FROZEN_EXECUTION_BLOCKED",
            "execution_ready": False,
            "scientific_protocol_boundary": {
                "existing_event_rich_v3": "incompatible B/C co-primary family",
                "same_pool_requirement": (
                    "explicit scientific v4 frozen before any real screen sampling"
                ),
                "alternative": "later independent B/E pool",
                "current_real_event_screen_samples": 0,
            },
            "supervision_opened": False,
            "endpoint_or_external_representation_opened": False,
            "files": rows,
            "files_sha256": core._sha256_json(rows),
            "matched_q_power_gate_identity": core._sha256_json(power),
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
    with tempfile.TemporaryDirectory(prefix="dit-bg-eprocess-lock-test-") as temporary:
        root = Path(temporary)
        result = freeze(
            output=root / "lock",
            protocol_path=DEFAULT_PROTOCOL,
            doc_path=DEFAULT_DOC,
            test_path=DEFAULT_TEST,
        )
        manifest = validate_lock(result)
        if manifest.get("execution_ready") is not False:
            raise AssertionError("self-test lock unexpectedly authorizes execution")
    print("self-test passed: immutable source/protocol/power lock and execution block")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
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
        "frozen B-gated e-process method protocol "
        f"{manifest['identity_sha256']} at {output}; execution remains blocked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
