#!/usr/bin/env python3
"""Synthetic fail-closed tests for the design-only scientific v4 lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"not an object: {path}")
    return value


def write_consensus(
    path: Path,
    *,
    phase: str,
    classes: list[int],
    seeds: list[int],
    blur_pairs: set[tuple[int, int]],
    extra_column: str | None = None,
    omit_last: bool = False,
) -> None:
    fields = ["phase", "class_id", "global_seed", "final_severity", "blur_component"]
    if extra_column is not None:
        fields.append(extra_column)
    rows: list[dict[str, str]] = []
    for seed in seeds:
        for class_id in classes:
            blur = (class_id, seed) in blur_pairs
            row = {
                "phase": phase,
                "class_id": str(class_id),
                "global_seed": str(seed),
                "final_severity": "clear_bad" if blur else "clean_good",
                "blur_component": "1" if blur else "0",
            }
            if extra_column is not None:
                row[extra_column] = "poison"
            rows.append(row)
    if omit_last:
        rows.pop()
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(command: list[str], *, expect_success: bool) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if expect_success and completed.returncode != 0:
        raise AssertionError(
            f"expected success: {' '.join(command)}\n{completed.stdout}\n{completed.stderr}"
        )
    if not expect_success and completed.returncode == 0:
        raise AssertionError(f"poison command unexpectedly succeeded: {' '.join(command)}")
    return completed


def selftest(lock: Path) -> dict[str, Any]:
    lock = lock.expanduser().absolute()
    protocol = load_json(lock / "protocol.json")
    manifest = load_json(lock / "manifest.json")
    completion = load_json(lock / "completion.json")
    if canonical_sha256(without_identity(protocol)) != protocol.get("identity_sha256"):
        raise AssertionError("protocol identity mismatch")
    if canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256"):
        raise AssertionError("manifest identity mismatch")
    if completion.get("ready_for_real_sampling") is not False:
        raise AssertionError("scientific v4 unexpectedly authorizes sampling")
    if protocol["method_lock"]["identity_sha256"] != (
        "facef0f59d1f10cde339440db3bc47dc26ca7fcef012faca01f7f2dfbb31b985"
    ):
        raise AssertionError("method identity changed")
    if protocol["co_primary_family"]["Holm_family_exactly"] != [
        "B_persistence", "E_blur_gated_running_max_log"
    ]:
        raise AssertionError("co-primary family changed")
    screen = protocol["endpoint_screen"]
    classes = [int(row["class_id"]) for row in screen["class_roster"]]
    discovery_seeds = [int(seed) for seed in screen["discovery_seeds"]]
    anchor_seeds = [int(seed) for seed in screen["anchor_seeds"]]
    if len(classes) != 84 or discovery_seeds != list(range(1000, 1012)):
        raise AssertionError("discovery axis changed")
    if anchor_seeds != list(range(1012, 1036)):
        raise AssertionError("anchor axis changed")
    selector = lock / "sources/select_dit_event_rich_blur_classes_v4.py"

    with tempfile.TemporaryDirectory(prefix="dit-event-v4-selftest-") as temporary:
        work = Path(temporary)
        discovery = work / "discovery.csv"
        selected_expected = classes[:6]
        blur_pairs: set[tuple[int, int]] = set()
        for offset, class_id in enumerate(selected_expected):
            count = 12 - offset
            blur_pairs.update((class_id, seed) for seed in discovery_seeds[:count])
        write_consensus(
            discovery,
            phase="discovery",
            classes=classes,
            seeds=discovery_seeds,
            blur_pairs=blur_pairs,
        )
        selection = work / "selection.json"
        run(
            [
                sys.executable,
                str(selector),
                "--lock",
                str(lock),
                "rank",
                "--consensus",
                str(discovery),
                "--output",
                str(selection),
            ],
            expect_success=True,
        )
        selected = load_json(selection)
        if selected["selected_classes"] != selected_expected:
            raise AssertionError("blur-only ranking was not reproduced")

        anchor = work / "anchor.csv"
        anchor_blur = {
            (class_id, seed)
            for class_id in selected_expected[:3]
            for seed in anchor_seeds[:2]
        }
        write_consensus(
            anchor,
            phase="anchor",
            classes=selected_expected,
            seeds=anchor_seeds,
            blur_pairs=anchor_blur,
        )
        plan = work / "anchor_plan.json"
        run(
            [
                sys.executable,
                str(selector),
                "--lock",
                str(lock),
                "anchor",
                "--selection",
                str(selection),
                "--consensus",
                str(anchor),
                "--output",
                str(plan),
            ],
            expect_success=True,
        )
        plan_payload = load_json(plan)
        if (
            plan_payload["decision"]["go"] is not True
            or plan_payload["decision"]["blur_clear_bad"] != 6
            or plan_payload["decision"]["event_bearing_classes"] != 3
            or plan_payload["calibration_trace_rows"] != 120
            or plan_payload["confirmation_trace_rows"] != 768
        ):
            raise AssertionError("anchor GO or exact dynamic axes changed")

        poison = work / "poison_dino.csv"
        write_consensus(
            poison,
            phase="discovery",
            classes=classes,
            seeds=discovery_seeds,
            blur_pairs=set(),
            extra_column="DINO_distance",
        )
        run(
            [
                sys.executable,
                str(selector),
                "--lock",
                str(lock),
                "rank",
                "--consensus",
                str(poison),
                "--output",
                str(work / "poison_selection.json"),
            ],
            expect_success=False,
        )

        incomplete = work / "incomplete.csv"
        write_consensus(
            incomplete,
            phase="discovery",
            classes=classes,
            seeds=discovery_seeds,
            blur_pairs=set(),
            omit_last=True,
        )
        run(
            [
                sys.executable,
                str(selector),
                "--lock",
                str(lock),
                "rank",
                "--consensus",
                str(incomplete),
                "--output",
                str(work / "incomplete_selection.json"),
            ],
            expect_success=False,
        )

        forged = dict(selected)
        forged["selected_classes"] = list(reversed(selected_expected))
        forged["identity_sha256"] = canonical_sha256(without_identity(forged))
        forged_path = work / "forged_selection.json"
        forged_path.write_text(
            json.dumps(forged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run(
            [
                sys.executable,
                str(selector),
                "--lock",
                str(lock),
                "anchor",
                "--selection",
                str(forged_path),
                "--consensus",
                str(anchor),
                "--output",
                str(work / "forged_plan.json"),
            ],
            expect_success=False,
        )

    return {
        "protocol_identity_sha256": protocol["identity_sha256"],
        "manifest_identity_sha256": manifest["identity_sha256"],
        "exact_discovery_rows": 1008,
        "exact_anchor_rows": 144,
        "exact_calibration_rows": 120,
        "exact_confirmation_rows": 768,
        "extra_DINO_column_rejected": True,
        "incomplete_axis_rejected": True,
        "rehashed_forged_selection_rejected": True,
        "ready_for_real_sampling": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(selftest(args.source_lock), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
