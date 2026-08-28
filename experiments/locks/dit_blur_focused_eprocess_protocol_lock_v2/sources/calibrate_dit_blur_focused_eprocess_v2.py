#!/usr/bin/env python3
"""Method-v2 facade for the unchanged label-free B calibration.

The v2 correction begins only after the predictable B threshold is frozen.
Therefore the 17th-of-20 checkpoint threshold and 19th-of-20 B score threshold
must remain byte-for-byte the v1 calibration rule.  This facade deliberately
delegates publication and validation to that immutable implementation while
binding the v2 method core in its self-test and source lock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    from . import calibrate_dit_blur_focused_eprocess as v1
    from . import observe_dit_blur_focused_eprocess_v2 as core
except ImportError:  # pragma: no cover
    import calibrate_dit_blur_focused_eprocess as v1
    import observe_dit_blur_focused_eprocess_v2 as core


SCHEMA_VERSION = v1.SCHEMA_VERSION
EXPERIMENT = v1.EXPERIMENT
SOURCE_EXPERIMENT = v1.SOURCE_EXPERIMENT
SOURCE_ARCHIVE = v1.SOURCE_ARCHIVE
CALIBRATION_COUNT_PER_CLASS = v1.CALIBRATION_COUNT_PER_CLASS
STATE_GATE_ORDER_INDEX = v1.STATE_GATE_ORDER_INDEX
PURE_B_ORDER_INDEX = v1.PURE_B_ORDER_INDEX
CALIBRATION_KEYS = v1.CALIBRATION_KEYS
CALIBRATION_CLASS_KEYS = v1.CALIBRATION_CLASS_KEYS
LOADED_ARRAY_NAMES = v1.LOADED_ARRAY_NAMES
V1_CALIBRATOR_SOURCE_SHA256 = "042b7023060eafa86455a0e1911e9af8991d736d6b8dbce4a77855285fdd170a"

if core._sha256_file(Path(v1.__file__).resolve()) != V1_CALIBRATOR_SOURCE_SHA256:
    raise RuntimeError("method v2 refuses an unpinned v1 calibrator dependency")

derive_calibration = v1.derive_calibration
validate_calibration = v1.validate_calibration
publish = v1.publish
_synthetic_arrays = v1._synthetic_arrays


def self_test() -> None:
    v1.self_test()
    if core.CHECKPOINTS != v1.core.CHECKPOINTS:
        raise AssertionError("v2 changed the B calibration checkpoint axis")
    if core.INPUT_ARRAY_NAMES != v1.core.INPUT_ARRAY_NAMES:
        raise AssertionError("v2 changed the calibrated observer input schema")
    if (STATE_GATE_ORDER_INDEX, PURE_B_ORDER_INDEX) != (16, 18):
        raise AssertionError("v2 changed a B order statistic")
    print(
        "v2 calibration self-test passed: immutable v1 17th/20 and 19th/20 "
        "B thresholds are reused without outcome or endpoint access"
    )


def main(argv: Iterable[str] | None = None) -> int:
    # Publication is intentionally performed by the immutable implementation;
    # its source hash remains the honest calibrator lineage in the JSON.
    return v1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
