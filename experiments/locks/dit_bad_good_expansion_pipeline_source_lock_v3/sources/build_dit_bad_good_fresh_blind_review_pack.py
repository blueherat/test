#!/usr/bin/env python3
"""Build the frozen seed-50..129 DiT endpoint-only blind-review pack.

This program deliberately exposes only terminal PNG evidence.  It reads each
source run's provenance records and its three endpoint PNGs; it never opens a
trajectory archive, extracted feature table, candidate score, calibration
record, alert file, or prior quality label.  The output contains byte-identical
endpoint copies, native-resolution contact sheets, explicit cell mappings, and
self-hashed manifest/completion records.

The evaluation Cartesian product is intentionally hard-coded so that the
calibration seeds 30..49 cannot accidentally enter the review pack:

    classes = (207, 602, 795)
    seeds   = 50..129 inclusive

Every contact sheet contains one class and one consecutive block of 20 seeds.
Pixels are pasted at their native 256x256 size without resampling.  Reviewers
can open the byte-identical endpoint copies when a single cell needs closer
inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo


RUNNER_NAME = "build_dit_bad_good_fresh_blind_review_pack"
SCHEMA_VERSION = 1
SOURCE_RUNNER = "trace_dit_imagenet256_custom_batch"
SOURCE_MANIFEST_NAME = "manifest.json"
SOURCE_COMPLETION_NAME = "completion.json"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
SOURCE_INDEX_NAME = "source_endpoints.jsonl"
CELL_INDEX_NAME = "grid_cells.jsonl"
REVIEW_CONTRACT_NAME = "review_contract.json"

EVALUATION_CLASSES = (207, 602, 795)
EVALUATION_SEEDS = tuple(range(50, 130))
CALIBRATION_SEEDS_FORBIDDEN = tuple(range(30, 50))
RUN_NAME_TEMPLATE = "confirmation_v1_seed{seed:03d}"
ENDPOINT_RELATIVE_PATHS = {
    207: "images/00_class0207.png",
    602: "images/01_class0602.png",
    795: "images/02_class0795.png",
}

BLOCK_SIZE = 20
GRID_COLUMNS = 5
TILE_SIZE = 256
CELL_GAP = 8
OUTER_MARGIN = 8
LABEL_HEIGHT = 24

FORBIDDEN_EVIDENCE = (
    "trajectory archives (including trace.npz) and intermediate trajectory states",
    "extracted trajectory features or metrics of any kind",
    "candidate scores, ranks, thresholds, or metric-triggered selections",
    "calibration records or calibrated alert decisions",
    "prior, concurrent, or future human quality labels and adjudications",
)


@dataclass(frozen=True)
class Endpoint:
    class_id: int
    seed: int
    source_run: Path
    source_manifest_sha256: str
    source_identity_sha256: str
    source_relative_path: str
    source_path: Path
    byte_count: int
    sha256: str
    pixel_sha256: str
    mode: str
    size: tuple[int, int]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON record: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON record must be an object: {path}")
    return value


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl_dump(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pixel_record(path: Path) -> tuple[str, tuple[int, int], str]:
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = image.size
        pixels = image.tobytes()
    return mode, size, sha256_bytes(pixels)


def _source_contract(identity: dict[str, Any]) -> dict[str, Any]:
    protocol = identity.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("source identity has no protocol object")
    protocol_without_seed = dict(protocol)
    protocol_without_seed.pop("global_torch_seed", None)
    return {
        "runner": identity.get("runner"),
        "observation_only": identity.get("observation_only"),
        "quality_score": identity.get("quality_score"),
        "selection": identity.get("selection"),
        "intervention": identity.get("intervention"),
        "protocol_without_seed": protocol_without_seed,
        "checkpoint": identity.get("checkpoint"),
        "vae_snapshot": identity.get("vae_snapshot"),
        "source": identity.get("source"),
        "runner_source": identity.get("runner_source"),
        "custom_baseline_helper": identity.get("custom_baseline_helper"),
        "strict_reproduction_helper": identity.get("strict_reproduction_helper"),
    }


def load_source_run(trace_root: Path, seed: int) -> tuple[tuple[Endpoint, ...], str]:
    run = (trace_root / RUN_NAME_TEMPLATE.format(seed=seed)).resolve()
    if not run.is_dir() or run.is_symlink():
        raise RuntimeError(f"missing or unsafe source run for seed {seed}: {run}")
    manifest_path = run / SOURCE_MANIFEST_NAME
    completion_path = run / SOURCE_COMPLETION_NAME
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if manifest.get("schema") != 1 or manifest.get("status") != "complete":
        raise RuntimeError(f"source run is not complete: {run}")
    identity = manifest.get("identity")
    outputs = manifest.get("outputs")
    if not isinstance(identity, dict) or not isinstance(outputs, list):
        raise RuntimeError(f"source provenance is incomplete: {run}")
    identity_sha256 = sha256_json(identity)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest.get("identity_sha256") != identity_sha256:
        raise RuntimeError(f"source identity self-hash failed: {run}")
    if manifest.get("outputs_sha256") != sha256_json(outputs):
        raise RuntimeError(f"source output-list self-hash failed: {run}")
    expected_completion = {
        "schema": 1,
        "identity_sha256": identity_sha256,
        "manifest_sha256": manifest_sha256,
        "outputs_sha256": sha256_json(outputs),
        "output_count": len(outputs),
    }
    if completion != expected_completion:
        raise RuntimeError(f"source completion lock failed: {run}")
    if identity.get("runner") != SOURCE_RUNNER:
        raise RuntimeError(f"unexpected source runner: {run}")
    if identity.get("observation_only") is not True:
        raise RuntimeError(f"source is not observation-only: {run}")
    for null_field in ("quality_score", "selection", "intervention"):
        if identity.get(null_field) is not None:
            raise RuntimeError(f"source {null_field} is not null: {run}")
    protocol = identity.get("protocol", {})
    if protocol.get("global_torch_seed") != seed:
        raise RuntimeError(f"source seed mismatch: {run}")
    if tuple(protocol.get("class_ids_ordered", ())) != EVALUATION_CLASSES:
        raise RuntimeError(f"source class batch mismatch: {run}")
    if protocol.get("image_size") != TILE_SIZE:
        raise RuntimeError(f"unexpected source endpoint size contract: {run}")

    output_by_path = {
        record.get("relative_path"): record
        for record in outputs
        if isinstance(record, dict)
    }
    endpoints: list[Endpoint] = []
    for class_id in EVALUATION_CLASSES:
        relative = ENDPOINT_RELATIVE_PATHS[class_id]
        record = output_by_path.get(relative)
        if not isinstance(record, dict):
            raise RuntimeError(f"missing endpoint record: {run / relative}")
        path = (run / relative).resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe endpoint PNG: {path}")
        if path.parent.parent != run:
            raise RuntimeError(f"endpoint escaped its source run: {path}")
        byte_count = path.stat().st_size
        file_sha256 = sha256_file(path)
        mode, size, pixels_sha256 = pixel_record(path)
        if (
            record.get("bytes") != byte_count
            or record.get("sha256") != file_sha256
            or record.get("pixel_sha256") != pixels_sha256
        ):
            raise RuntimeError(f"endpoint provenance mismatch: {path}")
        if mode != "RGB" or size != (TILE_SIZE, TILE_SIZE):
            raise RuntimeError(f"endpoint is not a native RGB 256x256 PNG: {path}")
        endpoints.append(
            Endpoint(
                class_id=class_id,
                seed=seed,
                source_run=run,
                source_manifest_sha256=manifest_sha256,
                source_identity_sha256=identity_sha256,
                source_relative_path=relative,
                source_path=path,
                byte_count=byte_count,
                sha256=file_sha256,
                pixel_sha256=pixels_sha256,
                mode=mode,
                size=size,
            )
        )
    return tuple(endpoints), sha256_json(_source_contract(identity))


def load_endpoints(trace_root: Path) -> tuple[Endpoint, ...]:
    if not trace_root.is_dir() or trace_root.is_symlink():
        raise RuntimeError(f"trace root must be a non-symlink directory: {trace_root}")
    if set(EVALUATION_SEEDS) & set(CALIBRATION_SEEDS_FORBIDDEN):
        raise AssertionError("evaluation and calibration seed sets overlap")
    endpoints: list[Endpoint] = []
    contracts: set[str] = set()
    for seed in EVALUATION_SEEDS:
        records, contract_sha256 = load_source_run(trace_root, seed)
        endpoints.extend(records)
        contracts.add(contract_sha256)
    if len(contracts) != 1:
        raise RuntimeError("source runs do not share one seed-independent contract")
    expected_pairs = {
        (class_id, seed)
        for class_id in EVALUATION_CLASSES
        for seed in EVALUATION_SEEDS
    }
    actual_pairs = {(item.class_id, item.seed) for item in endpoints}
    if actual_pairs != expected_pairs or len(endpoints) != len(expected_pairs):
        raise RuntimeError("source endpoints are not the exact 3 x 80 Cartesian product")
    return tuple(sorted(endpoints, key=lambda item: (item.class_id, item.seed)))


def endpoint_copy_relative_path(class_id: int, seed: int) -> str:
    return f"endpoints/class{class_id:04d}/seed{seed:03d}.png"


def grid_relative_path(class_id: int, first_seed: int, last_seed: int) -> str:
    return f"grids/class{class_id:04d}/seeds{first_seed:03d}-{last_seed:03d}.png"


def grid_geometry(count: int) -> tuple[int, int]:
    if count != BLOCK_SIZE:
        raise RuntimeError("each review grid must contain exactly 20 endpoints")
    rows = (count + GRID_COLUMNS - 1) // GRID_COLUMNS
    width = 2 * OUTER_MARGIN + GRID_COLUMNS * TILE_SIZE + (GRID_COLUMNS - 1) * CELL_GAP
    height = (
        2 * OUTER_MARGIN
        + rows * (TILE_SIZE + LABEL_HEIGHT)
        + (rows - 1) * CELL_GAP
    )
    return width, height


def render_native_grid(
    block: Sequence[Endpoint],
    output_path: Path,
    *,
    grid_relative: str,
    identity_sha256: str,
) -> list[dict[str, Any]]:
    if len(block) != BLOCK_SIZE or len({item.class_id for item in block}) != 1:
        raise RuntimeError("invalid review-grid block")
    ordered = tuple(sorted(block, key=lambda item: item.seed))
    if tuple(item.seed for item in ordered) != tuple(
        range(ordered[0].seed, ordered[0].seed + BLOCK_SIZE)
    ):
        raise RuntimeError("review-grid block is not 20 consecutive seeds")
    canvas = Image.new("RGB", grid_geometry(len(ordered)), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        column = index % GRID_COLUMNS
        row = index // GRID_COLUMNS
        x0 = OUTER_MARGIN + column * (TILE_SIZE + CELL_GAP)
        y0 = OUTER_MARGIN + row * (TILE_SIZE + LABEL_HEIGHT + CELL_GAP)
        with Image.open(item.source_path) as source:
            source.load()
            if source.mode != "RGB" or source.size != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError(f"source endpoint changed during rendering: {item.source_path}")
            tile = source.copy()
        canvas.paste(tile, (x0, y0))
        draw.text(
            (x0 + 3, y0 + TILE_SIZE + 4),
            f"class {item.class_id:04d}   seed {item.seed:03d}",
            fill=(245, 245, 245),
        )
        rows.append(
            {
                "class_id": item.class_id,
                "seed": item.seed,
                "grid_relative_path": grid_relative,
                "cell_index_row_major": index,
                "row_zero_based": row,
                "column_zero_based": column,
                "endpoint_bounds_xyxy_half_open": [
                    x0,
                    y0,
                    x0 + TILE_SIZE,
                    y0 + TILE_SIZE,
                ],
                "label_bounds_xyxy_half_open": [
                    x0,
                    y0 + TILE_SIZE,
                    x0 + TILE_SIZE,
                    y0 + TILE_SIZE + LABEL_HEIGHT,
                ],
                "copied_endpoint_relative_path": endpoint_copy_relative_path(
                    item.class_id, item.seed
                ),
                "source_endpoint_path": str(item.source_path),
                "source_endpoint_sha256": item.sha256,
                "source_endpoint_pixel_sha256": item.pixel_sha256,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    png_fields = {
        "runner": RUNNER_NAME,
        "identity_sha256": identity_sha256,
        "class_id": str(ordered[0].class_id),
        "seeds_ordered": ",".join(str(item.seed) for item in ordered),
        "native_endpoint_pixels_without_resampling": "true",
        "automatic_quality_scoring": "false",
        "metric_or_label_evidence_included": "false",
    }
    for key, value in png_fields.items():
        metadata.add_text(key, value)
    temporary = output_path.with_name(output_path.name + ".tmp")
    canvas.save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, output_path)
    return rows


def source_index_rows(endpoints: Sequence[Endpoint]) -> list[dict[str, Any]]:
    return [
        {
            "class_id": item.class_id,
            "seed": item.seed,
            "source_run_path": str(item.source_run),
            "source_manifest_sha256": item.source_manifest_sha256,
            "source_identity_sha256": item.source_identity_sha256,
            "source_endpoint_relative_path": item.source_relative_path,
            "source_endpoint_path": str(item.source_path),
            "source_endpoint_bytes": item.byte_count,
            "source_endpoint_sha256": item.sha256,
            "source_endpoint_pixel_sha256": item.pixel_sha256,
            "source_endpoint_mode": item.mode,
            "source_endpoint_size": list(item.size),
            "copied_endpoint_relative_path": endpoint_copy_relative_path(
                item.class_id, item.seed
            ),
            "copy_contract": "byte-identical; no decoding, resampling, or re-encoding",
        }
        for item in endpoints
    ]


def expected_endpoint_paths() -> tuple[str, ...]:
    return tuple(
        endpoint_copy_relative_path(class_id, seed)
        for class_id in EVALUATION_CLASSES
        for seed in EVALUATION_SEEDS
    )


def expected_grid_paths() -> tuple[str, ...]:
    return tuple(
        grid_relative_path(class_id, seeds[0], seeds[-1])
        for class_id in EVALUATION_CLASSES
        for offset in range(0, len(EVALUATION_SEEDS), BLOCK_SIZE)
        for seeds in (EVALUATION_SEEDS[offset : offset + BLOCK_SIZE],)
    )


def build_review_contract(identity_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "role": "BLIND_HUMAN_ENDPOINT_REVIEW_ONLY",
        "allowed_evidence": [
            "the copied terminal endpoint PNGs in this pack",
            "the native-resolution grids in this pack",
            "class IDs and seed IDs used only as stable sample identifiers",
        ],
        "forbidden_evidence": list(FORBIDDEN_EVIDENCE),
        "calibration_seeds_explicitly_excluded": list(CALIBRATION_SEEDS_FORBIDDEN),
        "evaluation_seeds_included": list(EVALUATION_SEEDS),
        "review_independence_rule": (
            "Do not inspect any path outside this pack while assigning endpoint labels. "
            "In particular, do not query scores or use alerts to select images for review."
        ),
        "visual_content_contract": (
            "Every displayed sample is a terminal endpoint. Grid tiles retain the native "
            "256x256 pixels without resizing; endpoint copies are byte-identical to source."
        ),
        "automatic_quality_scoring": False,
        "automatic_ranking_or_selection": False,
    }


def build_identity(trace_root: Path, endpoints: Sequence[Endpoint]) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    source_manifest_hashes = [
        {
            "seed": seed,
            "manifest_sha256": next(
                item.source_manifest_sha256 for item in endpoints if item.seed == seed
            ),
            "identity_sha256": next(
                item.source_identity_sha256 for item in endpoints if item.seed == seed
            ),
        }
        for seed in EVALUATION_SEEDS
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "runner_source": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
        "role": "FRESH_EVALUATION_ENDPOINT_ONLY_BLIND_REVIEW_PACK",
        "trace_root": str(trace_root),
        "classes_ordered": list(EVALUATION_CLASSES),
        "seeds_ordered": list(EVALUATION_SEEDS),
        "calibration_seeds_explicitly_excluded": list(CALIBRATION_SEEDS_FORBIDDEN),
        "cartesian_product": {
            "class_count": len(EVALUATION_CLASSES),
            "seed_count": len(EVALUATION_SEEDS),
            "endpoint_count": len(endpoints),
            "exact": True,
        },
        "grid_contract": {
            "block_size": BLOCK_SIZE,
            "columns": GRID_COLUMNS,
            "rows": BLOCK_SIZE // GRID_COLUMNS,
            "tile_size": [TILE_SIZE, TILE_SIZE],
            "resampling": "none",
            "grid_count": len(expected_grid_paths()),
        },
        "endpoint_copy_contract": "byte-identical; no decoding, resampling, or re-encoding",
        "forbidden_evidence": list(FORBIDDEN_EVIDENCE),
        "builder_input_access_scope": [
            "source run manifest.json provenance",
            "source run completion.json hash lock",
            "the three terminal endpoint PNGs in each selected evaluation run",
        ],
        "builder_does_not_open": [
            "trace.npz",
            "any extracted feature or metric output",
            "any candidate-score or calibration-alert output",
            "any human-label or adjudication file",
        ],
        "source_run_locks": source_manifest_hashes,
        "expected_endpoint_paths": list(expected_endpoint_paths()),
        "expected_grid_paths": list(expected_grid_paths()),
        "expected_metadata_paths": [
            SOURCE_INDEX_NAME,
            CELL_INDEX_NAME,
            REVIEW_CONTRACT_NAME,
        ],
        "automatic_quality_scoring": False,
        "automatic_ranking_or_selection": False,
    }
    payload["identity_sha256"] = sha256_json(payload)
    return payload


def file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    record: dict[str, Any] = {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".png":
        mode, size, pixels_sha256 = pixel_record(path)
        record.update(
            {
                "pixel_sha256": pixels_sha256,
                "mode": mode,
                "size": list(size),
            }
        )
    return record


def expected_artifact_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            (*expected_endpoint_paths(), *expected_grid_paths(), SOURCE_INDEX_NAME,
             CELL_INDEX_NAME, REVIEW_CONTRACT_NAME)
        )
    )


def inspect_artifacts(root: Path) -> list[dict[str, Any]]:
    expected = set(expected_artifact_paths())
    expected_files = {
        (root / relative).resolve() for relative in expected
    } | {(root / MANIFEST_NAME).resolve(), (root / COMPLETION_NAME).resolve()}
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        raise RuntimeError(f"blind-pack file set mismatch; missing={missing}, extra={extra}")
    return [file_record(root, relative) for relative in sorted(expected)]


def validate_indexes(root: Path, identity: dict[str, Any]) -> None:
    source_rows = [
        json.loads(line)
        for line in (root / SOURCE_INDEX_NAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    cell_rows = [
        json.loads(line)
        for line in (root / CELL_INDEX_NAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_pairs = {
        (class_id, seed)
        for class_id in EVALUATION_CLASSES
        for seed in EVALUATION_SEEDS
    }
    if len(source_rows) != 240 or {
        (row.get("class_id"), row.get("seed")) for row in source_rows
    } != expected_pairs:
        raise RuntimeError("source endpoint index is not the exact 240-sample product")
    if len(cell_rows) != 240 or {
        (row.get("class_id"), row.get("seed")) for row in cell_rows
    } != expected_pairs:
        raise RuntimeError("grid cell index is not the exact 240-sample product")
    if any(30 <= int(row["seed"]) <= 49 for row in (*source_rows, *cell_rows)):
        raise RuntimeError("forbidden calibration seed leaked into the blind pack")
    for row in source_rows:
        copied = root / row["copied_endpoint_relative_path"]
        if sha256_file(copied) != row["source_endpoint_sha256"]:
            raise RuntimeError(f"endpoint copy is not byte-identical: {copied}")
        _, _, pixels_sha256 = pixel_record(copied)
        if pixels_sha256 != row["source_endpoint_pixel_sha256"]:
            raise RuntimeError(f"endpoint copy pixel hash changed: {copied}")
    contract = load_json(root / REVIEW_CONTRACT_NAME)
    if contract != build_review_contract(identity["identity_sha256"]):
        raise RuntimeError("review contract differs from its frozen definition")


def validate_completed(root: Path, identity: dict[str, Any]) -> None:
    manifest_path = root / MANIFEST_NAME
    completion_path = root / COMPLETION_NAME
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError("blind-pack manifest is not complete")
    identity_payload = dict(identity)
    stored_identity_sha256 = identity_payload.pop("identity_sha256", None)
    if stored_identity_sha256 != sha256_json(identity_payload):
        raise RuntimeError("blind-pack identity self-hash failed")
    if manifest.get("identity") != identity:
        raise RuntimeError("blind-pack identity changed")
    records = inspect_artifacts(root)
    outputs_sha256 = sha256_json(records)
    if manifest.get("artifacts") != records or manifest.get("artifacts_sha256") != outputs_sha256:
        raise RuntimeError("blind-pack artifact hashes changed")
    expected_completion = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity["identity_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts_sha256": outputs_sha256,
        "artifact_count": len(records),
        "endpoint_count": 240,
        "grid_count": 12,
        "grid_cell_count": 240,
        "calibration_seed_count": 0,
    }
    if completion != expected_completion:
        raise RuntimeError("blind-pack completion hash lock failed")
    validate_indexes(root, identity)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def build_pack(trace_root: Path, outdir: Path) -> None:
    endpoints = load_endpoints(trace_root)
    identity = build_identity(trace_root, endpoints)
    if outdir.exists():
        if not outdir.is_dir() or outdir.is_symlink():
            raise RuntimeError(f"output must be a non-symlink directory: {outdir}")
        if any(outdir.iterdir()):
            validate_completed(outdir, identity)
            print(f"validated completed endpoint-only blind-review pack: {outdir}")
            return
        raise RuntimeError(f"refusing to replace a pre-existing empty directory: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    if _paths_overlap(trace_root, outdir):
        raise RuntimeError("blind-review output overlaps the source trace root")

    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary)
        source_rows = source_index_rows(endpoints)

        # Preserve terminal PNG bytes exactly for one-image inspection.
        for item in endpoints:
            destination = staging / endpoint_copy_relative_path(item.class_id, item.seed)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.source_path, destination)
            if sha256_file(destination) != item.sha256:
                raise RuntimeError(f"endpoint copy hash mismatch: {destination}")

        cell_rows: list[dict[str, Any]] = []
        for class_id in EVALUATION_CLASSES:
            class_items = [item for item in endpoints if item.class_id == class_id]
            for offset in range(0, len(class_items), BLOCK_SIZE):
                block = class_items[offset : offset + BLOCK_SIZE]
                relative = grid_relative_path(class_id, block[0].seed, block[-1].seed)
                destination = staging / relative
                rows = render_native_grid(
                    block,
                    destination,
                    grid_relative=relative,
                    identity_sha256=identity["identity_sha256"],
                )
                cell_rows.extend(rows)

        atomic_jsonl_dump(source_rows, staging / SOURCE_INDEX_NAME)
        atomic_jsonl_dump(cell_rows, staging / CELL_INDEX_NAME)
        atomic_json_dump(
            build_review_contract(identity["identity_sha256"]),
            staging / REVIEW_CONTRACT_NAME,
        )

        # Reserve the two lock paths before exact file-set inspection.
        atomic_json_dump({}, staging / MANIFEST_NAME)
        atomic_json_dump({}, staging / COMPLETION_NAME)
        records = inspect_artifacts(staging)
        artifacts_sha256 = sha256_json(records)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "source_endpoints": source_rows,
            "grid_cells": cell_rows,
            "artifacts": records,
            "artifacts_sha256": artifacts_sha256,
        }
        atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": identity["identity_sha256"],
            "manifest_sha256": sha256_file(staging / MANIFEST_NAME),
            "artifacts_sha256": artifacts_sha256,
            "artifact_count": len(records),
            "endpoint_count": len(endpoints),
            "grid_count": len(expected_grid_paths()),
            "grid_cell_count": len(cell_rows),
            "calibration_seed_count": 0,
        }
        atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_completed(staging, identity)
        if outdir.exists():
            raise RuntimeError("output appeared during staging; refusing overwrite")
        os.replace(staging, outdir)

    print(
        json.dumps(
            {
                "complete": True,
                "outdir": str(outdir),
                "identity_sha256": identity["identity_sha256"],
                "classes": list(EVALUATION_CLASSES),
                "seeds": [EVALUATION_SEEDS[0], EVALUATION_SEEDS[-1]],
                "endpoint_count": len(endpoints),
                "grid_count": len(expected_grid_paths()),
                "calibration_seed_count": 0,
                "metric_or_label_evidence_included": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.trace_root = args.trace_root.expanduser().absolute().resolve()
    requested = args.outdir.expanduser().absolute()
    if os.path.lexists(requested) and requested.is_symlink():
        raise RuntimeError(f"output must not be a symlink: {requested}")
    args.outdir = requested.resolve()
    build_pack(args.trace_root, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
