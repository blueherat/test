#!/usr/bin/env python3
"""Prepare a score-blind visual audit of retrospective v2.2 E alarms.

All fixed ``E>=10`` alarms are paired with unique non-alarm controls from the
same class.  Exact ``(T1,h1,T4,h4)`` schedule matches are preferred; otherwise
the nearest prespecified start-summary match is used.  The delivery tree has
only anonymous native images, contact sheets, a rubric, and empty response
forms.  The score/arm mapping is written to a physically separate private
tree and must remain closed until reviewer responses are immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

try:
    from .explore_dit_v22_third_pool_retrospective import (
        EXPECTED_CLASSES,
        canonical_sha256,
        sha256_file,
        validate_score_shard,
        write_json,
    )
except ImportError:  # pragma: no cover
    from explore_dit_v22_third_pool_retrospective import (
        EXPECTED_CLASSES,
        canonical_sha256,
        sha256_file,
        validate_score_shard,
        write_json,
    )


DEFAULT_TRACE_ROOT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_third_pool_v1_custom_traces_cfg_locked"
)
SLOT = {207: 0, 602: 1, 795: 2}
RESPONSE_COLUMNS = (
    "anonymous_id",
    "relative_quality",
    "blur_or_soft_fusion",
    "topology_or_attachment_error",
    "localized_problem",
    "short_reason",
)


def hidden_key(*parts: Any) -> str:
    payload = "\0".join(["eqvae.v22.alarm.audit.v1", *map(str, parts)]).encode()
    return hashlib.sha256(payload).hexdigest()


def schedule(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(
        int(row[name]) for name in ("T_delta1", "h_delta1", "T_delta4", "h_delta4")
    )


def schedule_distance(left: dict[str, str], right: dict[str, str]) -> tuple[Any, ...]:
    l_schedule = schedule(left)
    r_schedule = schedule(right)
    exact = l_schedule == r_schedule
    # Treat no-start as a fixed endpoint and otherwise use the finite discrete
    # checkpoint/count distance.  G is a deterministic tiebreaking summary.
    coordinate_distance = sum(abs(a - b) for a, b in zip(l_schedule, r_schedule))
    g_distance = abs(
        float(left["G_start_schedule_diagnostic"])
        - float(right["G_start_schedule_diagnostic"])
    )
    return (
        0 if exact else 1,
        coordinate_distance,
        g_distance,
        hidden_key(right["global_seed"], right["class_id"]),
    )


def source_image(trace_root: Path, row: dict[str, str]) -> Path:
    seed = int(row["global_seed"])
    class_id = int(row["class_id"])
    slot = SLOT[class_id]
    path = (
        trace_root
        / f"third_pool_v1_seed{seed}"
        / "images"
        / f"{slot:02d}_class{class_id:04d}.png"
    )
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular endpoint image: {path}")
    return path.resolve()


def make_grid(paths: list[Path], labels: list[str], output: Path) -> None:
    if len(paths) != len(labels) or not paths:
        raise ValueError("contact-sheet rows are malformed")
    columns = 4
    tile = 256
    label_height = 22
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.size != (tile, tile):
                raise RuntimeError(f"unexpected endpoint size: {path}: {rgb.size}")
            x = (index % columns) * tile
            y = (index // columns) * (tile + label_height)
            canvas.paste(rgb, (x, y))
            draw.text((x + 4, y + tile + 3), label, fill="black")
    canvas.save(output)


def publish(args: argparse.Namespace) -> None:
    delivery = args.delivery.expanduser().absolute()
    private = args.private.expanduser().absolute()
    if os.path.lexists(delivery) or os.path.lexists(private):
        raise RuntimeError("refusing to overwrite a blind-audit tree")
    rows: list[dict[str, str]] = []
    score_manifest_ids: list[str] = []
    seen: set[tuple[int, int]] = set()
    for shard in args.score_shards:
        manifest, shard_rows = validate_score_shard(shard)
        score_manifest_ids.append(str(manifest["identity_sha256"]))
        for row in shard_rows:
            key = (int(row["global_seed"]), int(row["class_id"]))
            if key in seen:
                raise RuntimeError("score shards overlap")
            seen.add(key)
            rows.append(row)
    alarms = sorted(
        (row for row in rows if int(row["E_blur_gated_alarm"]) == 1),
        key=lambda row: (int(row["global_seed"]), int(row["class_id"])),
    )
    if len(alarms) != 26:
        raise RuntimeError(f"frozen old-pool E alarm count changed: {len(alarms)}")
    available = [row for row in rows if int(row["E_blur_gated_alarm"]) == 0]
    controls: list[dict[str, str]] = []
    exact_matches = 0
    for alarm in alarms:
        candidates = [
            row
            for row in available
            if row["class_id"] == alarm["class_id"] and row not in controls
        ]
        if not candidates:
            raise RuntimeError("ran out of unique same-class controls")
        control = min(candidates, key=lambda row: schedule_distance(alarm, row))
        controls.append(control)
        exact_matches += int(schedule(alarm) == schedule(control))
    selected = [
        {"arm": "alarm", "pair_index": index, "row": alarm}
        for index, alarm in enumerate(alarms)
    ] + [
        {"arm": "control", "pair_index": index, "row": control}
        for index, control in enumerate(controls)
    ]
    selected.sort(
        key=lambda item: hidden_key(
            item["arm"], item["row"]["global_seed"], item["row"]["class_id"]
        )
    )
    mapping: list[dict[str, Any]] = []
    delivery.parent.mkdir(parents=True, exist_ok=True)
    private.parent.mkdir(parents=True, exist_ok=True)
    delivery_stage = Path(
        tempfile.mkdtemp(prefix=f".{delivery.name}.tmp-", dir=delivery.parent)
    )
    private_stage = Path(
        tempfile.mkdtemp(prefix=f".{private.name}.tmp-", dir=private.parent)
    )
    try:
        native = delivery_stage / "native"
        grids = delivery_stage / "grids"
        native.mkdir()
        grids.mkdir()
        trace_root = args.trace_root.expanduser().resolve()
        for anonymous_index, item in enumerate(selected):
            anonymous_id = f"A{anonymous_index:04d}"
            source = source_image(trace_root, item["row"])
            target = native / f"{anonymous_id}.png"
            shutil.copyfile(source, target)
            mapping.append(
                {
                    "anonymous_id": anonymous_id,
                    "arm": item["arm"],
                    "pair_index": item["pair_index"],
                    "global_seed": int(item["row"]["global_seed"]),
                    "class_id": int(item["row"]["class_id"]),
                    "schedule": list(schedule(item["row"])),
                    "E_running_max_log": float(
                        item["row"]["E_blur_gated_running_max_log"]
                    ),
                    "B_persistence": float(item["row"]["B_persistence"]),
                    "G_start": float(item["row"]["G_start_schedule_diagnostic"]),
                    "source_image_sha256": sha256_file(source),
                    "anonymous_image_sha256": sha256_file(target),
                }
            )
        for start in range(0, len(mapping), 16):
            block = mapping[start : start + 16]
            make_grid(
                [native / f"{row['anonymous_id']}.png" for row in block],
                [row["anonymous_id"] for row in block],
                grids / f"sheet_{start // 16:02d}.png",
            )
        rubric = {
            "task": "Judge each anonymous endpoint relative to the ordinary quality of this same frozen DiT batch.",
            "relative_quality_values": ["clean_good", "mild_or_uncertain", "clear_bad"],
            "clear_bad_rule": (
                "Use only for an obvious defect materially below this model's ordinary batch level: "
                "strong blur/soft fusion, duplicated/fused/missing/misattached limb or object, "
                "severe spatial misregistration, or gross incoherent structure. Class recognizability "
                "does not override a structural defect. Do not mark merely imperfect average DiT texture as clear_bad."
            ),
            "blur_or_soft_fusion": "true only when conspicuous blur, melting, smearing, or soft fusion is part of the defect",
            "topology_or_attachment_error": "true only for conspicuous duplicated/fused/missing/misattached anatomy or object geometry",
            "localized_problem": "short region such as left arm, lower skis, sled body, face; empty only for clean_good",
            "review_order": "native PNG at 100% first; use sheets only for navigation; inspect every item independently",
            "forbidden_context": [
                "candidate scores or alarms",
                "arm or matched pair",
                "old labels or other reviewer votes",
                "FID/Inception/DINO/CLIP/embeddings",
            ],
        }
        write_json(delivery_stage / "rubric.json", rubric)
        for reviewer in ("reviewer_1", "reviewer_2"):
            with (delivery_stage / f"{reviewer}_response.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
                writer.writeheader()
                for row in mapping:
                    writer.writerow({"anonymous_id": row["anonymous_id"]})
        delivery_manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_ALARM_SCORE_BLIND_DELIVERY",
            "anonymous_count": len(mapping),
            "native_image_count": len(mapping),
            "sheet_count": (len(mapping) + 15) // 16,
            "score_arm_mapping_present": False,
            "old_labels_or_external_representations_present": False,
            "files": [
                {
                    "name": path.relative_to(delivery_stage).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(delivery_stage.rglob("*"))
                if path.is_file()
            ],
        }
        delivery_manifest["identity_sha256"] = canonical_sha256(delivery_manifest)
        write_json(delivery_stage / "manifest.json", delivery_manifest)
        private_payload: dict[str, Any] = {
            "status": "SEALED_UNTIL_BOTH_REVIEWS_COMPLETE",
            "artifact_kind": "DIT_V22_ALARM_PRIVATE_MAPPING",
            "delivery_identity_sha256": delivery_manifest["identity_sha256"],
            "score_shard_manifest_ids": score_manifest_ids,
            "alarm_count": len(alarms),
            "control_count": len(controls),
            "exact_schedule_matched_pair_count": exact_matches,
            "same_class_pair_count": len(alarms),
            "mapping": mapping,
            "old_labels_opened_for_pack_selection": False,
            "external_representations_used": False,
        }
        private_payload["identity_sha256"] = canonical_sha256(private_payload)
        write_json(private_stage / "sealed_mapping.json", private_payload)
        os.replace(delivery_stage, delivery)
        os.replace(private_stage, private)
    except Exception:
        shutil.rmtree(delivery_stage, ignore_errors=True)
        shutil.rmtree(private_stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "delivery": str(delivery),
                "private": str(private),
                "delivery_identity_sha256": delivery_manifest["identity_sha256"],
                "anonymous_count": len(mapping),
                "alarm_count": len(alarms),
                "control_count": len(controls),
                "exact_schedule_matched_pairs": exact_matches,
            },
            indent=2,
            sort_keys=True,
        )
    )


def self_test() -> None:
    alarm = {
        "T_delta1": "4",
        "h_delta1": "5",
        "T_delta4": "3",
        "h_delta4": "6",
        "G_start_schedule_diagnostic": "0.875",
    }
    exact = dict(alarm, global_seed="1", class_id="602")
    other = dict(alarm, T_delta4="4", h_delta4="5", global_seed="2", class_id="602")
    if not schedule_distance(alarm, exact) < schedule_distance(alarm, other):
        raise AssertionError("exact schedule matching priority changed")
    if hidden_key("alarm", 1, 2) == hidden_key("control", 1, 2):
        raise AssertionError("blind randomization domain separation changed")
    print("v2.2 alarm blind-audit preparation self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
    else:
        publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
