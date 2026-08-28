#!/usr/bin/env python3
"""Build the endpoint-only blind pack for expansion seeds 130..249.

Only source provenance and terminal PNG bytes are opened.  The builder never
opens trace.npz, extracted features, candidate scores, calibration thresholds,
alerts, or any visual label.  Endpoint copies remain byte-identical and grid
tiles remain native 256x256 pixels without resampling.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

try:
    from .build_dit_bad_good_fresh_blind_review_pack import (
        Endpoint,
        _source_contract,
        pixel_record,
        sha256_json,
    )
    from .dit_bad_good_expansion_contract import (
        CALIBRATION_SEEDS,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        EXPANSION_SEEDS,
        ORIGINAL_EVALUATION_SEEDS,
        canonical_sha256,
        load_json,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )
except ImportError:  # pragma: no cover
    from build_dit_bad_good_fresh_blind_review_pack import (
        Endpoint,
        _source_contract,
        pixel_record,
        sha256_json,
    )
    from dit_bad_good_expansion_contract import (
        CALIBRATION_SEEDS,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        EXPANSION_SEEDS,
        ORIGINAL_EVALUATION_SEEDS,
        canonical_sha256,
        load_json,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )


RUNNER = "build_dit_bad_good_expansion_blind_review_pack"
BLOCK_SIZE = 20
GRID_COLUMNS = 5
TILE_SIZE = 256
CELL_GAP = 8
OUTER_MARGIN = 8
LABEL_HEIGHT = 24
SOURCE_INDEX = "source_endpoints.jsonl"
CELL_INDEX = "grid_cells.jsonl"
REVIEW_CONTRACT = "review_contract.json"

FORBIDDEN_EVIDENCE = (
    "trajectory archives and intermediate trajectory states",
    "extracted trajectory features or metrics",
    "candidate scores, ranks, calibration thresholds, or alerts",
    "prior, concurrent, or future visual labels and adjudications",
)


def endpoint_relative(class_id: int, seed: int) -> str:
    return f"endpoints/class{class_id:04d}/seed{seed:03d}.png"


def grid_relative(class_id: int, first_seed: int, last_seed: int) -> str:
    return f"grids/class{class_id:04d}/seeds{first_seed:03d}-{last_seed:03d}.png"


def expected_endpoint_paths() -> tuple[str, ...]:
    return tuple(
        endpoint_relative(class_id, seed)
        for class_id in CLASSES
        for seed in EXPANSION_SEEDS
    )


def expected_grid_paths() -> tuple[str, ...]:
    paths: list[str] = []
    for class_id in CLASSES:
        for offset in range(0, len(EXPANSION_SEEDS), BLOCK_SIZE):
            block = EXPANSION_SEEDS[offset : offset + BLOCK_SIZE]
            paths.append(grid_relative(class_id, block[0], block[-1]))
    return tuple(paths)


def _load_source_run(trace_root: Path, seed: int) -> tuple[tuple[Endpoint, ...], str]:
    """Validate one expansion trace while preserving the frozen base contract."""

    run = (trace_root / f"expansion_v1_seed{seed:03d}").resolve()
    if not run.is_dir() or run.is_symlink():
        raise RuntimeError(f"missing or unsafe source run for seed {seed}: {run}")
    manifest_path = run / "manifest.json"
    completion_path = run / "completion.json"
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
    if (
        manifest.get("identity_sha256") != identity_sha256
        or manifest.get("outputs_sha256") != sha256_json(outputs)
        or completion
        != {
            "schema": 1,
            "identity_sha256": identity_sha256,
            "manifest_sha256": manifest_sha256,
            "outputs_sha256": sha256_json(outputs),
            "output_count": len(outputs),
        }
        or identity.get("runner") != "trace_dit_imagenet256_custom_batch"
        or identity.get("observation_only") is not True
        or identity.get("quality_score") is not None
        or identity.get("selection") is not None
        or identity.get("intervention") is not None
    ):
        raise RuntimeError(f"source trace provenance contract failed: {run}")
    trace_protocol = identity.get("protocol", {})
    if (
        trace_protocol.get("global_torch_seed") != seed
        or tuple(trace_protocol.get("class_ids_ordered", ())) != CLASSES
        or trace_protocol.get("image_size") != TILE_SIZE
    ):
        raise RuntimeError(f"source trace scientific contract failed: {run}")
    output_by_path = {
        item.get("relative_path"): item for item in outputs if isinstance(item, dict)
    }
    endpoint_paths = {
        207: "images/00_class0207.png",
        602: "images/01_class0602.png",
        795: "images/02_class0795.png",
    }
    endpoints: list[Endpoint] = []
    for class_id in CLASSES:
        relative = endpoint_paths[class_id]
        record = output_by_path.get(relative)
        path = (run / relative).resolve()
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.is_symlink()
            or path.parent.parent != run
        ):
            raise RuntimeError(f"missing or unsafe endpoint: {path}")
        byte_count = path.stat().st_size
        file_sha256 = sha256_file(path)
        mode, size, pixels_sha256 = pixel_record(path)
        if (
            record.get("bytes") != byte_count
            or record.get("sha256") != file_sha256
            or record.get("pixel_sha256") != pixels_sha256
            or mode != "RGB"
            or size != (TILE_SIZE, TILE_SIZE)
        ):
            raise RuntimeError(f"endpoint provenance mismatch: {path}")
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


def _load_endpoints(trace_root: Path) -> tuple[Endpoint, ...]:
    if not trace_root.is_dir() or trace_root.is_symlink():
        raise RuntimeError(f"trace root must be a real directory: {trace_root}")
    endpoints: list[Endpoint] = []
    source_contracts: set[str] = set()
    for seed in EXPANSION_SEEDS:
        rows, source_contract = _load_source_run(trace_root, seed)
        endpoints.extend(rows)
        source_contracts.add(source_contract)
    expected = {(class_id, seed) for class_id in CLASSES for seed in EXPANSION_SEEDS}
    observed = {(row.class_id, row.seed) for row in endpoints}
    if len(endpoints) != 360 or observed != expected or len(source_contracts) != 1:
        raise RuntimeError("source traces are not one exact, homogeneous 3x120 cohort")
    return tuple(sorted(endpoints, key=lambda row: (row.class_id, row.seed)))


def _source_rows(endpoints: Sequence[Endpoint]) -> list[dict[str, Any]]:
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
            "copied_endpoint_relative_path": endpoint_relative(item.class_id, item.seed),
            "copy_contract": "byte-identical; no decoding, resampling, or re-encoding",
        }
        for item in endpoints
    ]


def _identity(trace_root: Path, endpoints: Sequence[Endpoint]) -> dict[str, Any]:
    helper = Path(__file__).resolve().with_name(
        "build_dit_bad_good_fresh_blind_review_pack.py"
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "runner": RUNNER,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "imported_endpoint_validator_sha256": sha256_file(helper),
        "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
        "role": "DISJOINT_EXPANSION_ENDPOINT_ONLY_BLIND_REVIEW_PACK",
        "trace_root": str(trace_root),
        "classes_ordered": list(CLASSES),
        "seeds_ordered": list(EXPANSION_SEEDS),
        "excluded_seed_sets": {
            "calibration": list(CALIBRATION_SEEDS),
            "original_evaluation": list(ORIGINAL_EVALUATION_SEEDS),
        },
        "cartesian_product": {
            "class_count": 3,
            "seed_count": 120,
            "endpoint_count": 360,
            "exact": True,
        },
        "grid_contract": {
            "block_size": BLOCK_SIZE,
            "columns": GRID_COLUMNS,
            "rows": BLOCK_SIZE // GRID_COLUMNS,
            "tile_size": [TILE_SIZE, TILE_SIZE],
            "resampling": "none",
            "grid_count": 18,
        },
        "endpoint_copy_contract": "byte-identical; no decoding, resampling, or re-encoding",
        "builder_input_access_scope": [
            "source manifest.json and completion.json provenance",
            "three terminal endpoint PNGs from each expansion run",
        ],
        "builder_does_not_open": list(FORBIDDEN_EVIDENCE),
        "source_run_locks": [
            {
                "seed": seed,
                "manifest_sha256": next(
                    row.source_manifest_sha256 for row in endpoints if row.seed == seed
                ),
                "identity_sha256": next(
                    row.source_identity_sha256 for row in endpoints if row.seed == seed
                ),
            }
            for seed in EXPANSION_SEEDS
        ],
        "automatic_quality_scoring": False,
        "automatic_ranking_or_selection": False,
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def _review_contract(identity: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity_sha256": identity,
        "role": "BLIND_HUMAN_ENDPOINT_REVIEW_ONLY",
        "allowed_evidence": [
            "copied terminal endpoint PNGs in this pack",
            "native-resolution grids in this pack",
            "class and seed identifiers solely as stable sample identifiers",
        ],
        "forbidden_evidence": list(FORBIDDEN_EVIDENCE),
        "evaluation_seeds_included": list(EXPANSION_SEEDS),
        "calibration_and_original_evaluation_seeds_excluded": list(
            (*CALIBRATION_SEEDS, *ORIGINAL_EVALUATION_SEEDS)
        ),
        "review_independence_rule": (
            "Do not inspect paths outside this pack; never query a metric, score, "
            "threshold, alert, trajectory, prior review, or research hypothesis."
        ),
        "visual_content_contract": (
            "Each tile is a terminal endpoint at native 256x256 resolution; copied "
            "endpoints are byte-identical to source."
        ),
        "automatic_quality_scoring": False,
        "automatic_ranking_or_selection": False,
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _render_grid(
    block: Sequence[Endpoint], output: Path, identity: str
) -> list[dict[str, Any]]:
    if len(block) != BLOCK_SIZE or len({row.class_id for row in block}) != 1:
        raise RuntimeError("grid block must contain 20 endpoints from one class")
    ordered = tuple(sorted(block, key=lambda row: row.seed))
    if tuple(row.seed for row in ordered) != tuple(
        range(ordered[0].seed, ordered[0].seed + BLOCK_SIZE)
    ):
        raise RuntimeError("grid block seeds are not consecutive")
    rows = BLOCK_SIZE // GRID_COLUMNS
    width = 2 * OUTER_MARGIN + GRID_COLUMNS * TILE_SIZE + (GRID_COLUMNS - 1) * CELL_GAP
    height = (
        2 * OUTER_MARGIN
        + rows * (TILE_SIZE + LABEL_HEIGHT)
        + (rows - 1) * CELL_GAP
    )
    canvas = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    grid_rel = grid_relative(ordered[0].class_id, ordered[0].seed, ordered[-1].seed)
    cells: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        column = index % GRID_COLUMNS
        row = index // GRID_COLUMNS
        x0 = OUTER_MARGIN + column * (TILE_SIZE + CELL_GAP)
        y0 = OUTER_MARGIN + row * (TILE_SIZE + LABEL_HEIGHT + CELL_GAP)
        with Image.open(item.source_path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError(f"source endpoint changed: {item.source_path}")
            canvas.paste(image, (x0, y0))
        draw.text(
            (x0 + 3, y0 + TILE_SIZE + 4),
            f"class {item.class_id:04d}   seed {item.seed:03d}",
            fill=(245, 245, 245),
        )
        cells.append(
            {
                "class_id": item.class_id,
                "seed": item.seed,
                "grid_relative_path": grid_rel,
                "cell_index_row_major": index,
                "row_zero_based": row,
                "column_zero_based": column,
                "endpoint_bounds_xyxy_half_open": [
                    x0,
                    y0,
                    x0 + TILE_SIZE,
                    y0 + TILE_SIZE,
                ],
                "copied_endpoint_relative_path": endpoint_relative(
                    item.class_id, item.seed
                ),
                "source_endpoint_sha256": item.sha256,
                "source_endpoint_pixel_sha256": item.pixel_sha256,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text("runner", RUNNER)
    metadata.add_text("identity_sha256", identity)
    metadata.add_text("native_endpoint_pixels_without_resampling", "true")
    metadata.add_text("metric_or_label_evidence_included", "false")
    canvas.save(output, format="PNG", pnginfo=metadata)
    return cells


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    record: dict[str, Any] = {
        "name": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".png":
        mode, size, pixels = pixel_record(path)
        record.update({"mode": mode, "size": list(size), "pixel_sha256": pixels})
    return record


def _artifact_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                *expected_endpoint_paths(),
                *expected_grid_paths(),
                SOURCE_INDEX,
                CELL_INDEX,
                REVIEW_CONTRACT,
            )
        )
    )


def validate_completed(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("blind pack has no identity")
    identity_without_hash = dict(identity)
    identity_hash = identity_without_hash.pop("identity_sha256", None)
    if identity_hash != canonical_sha256(identity_without_hash):
        raise RuntimeError("blind-pack identity failed")
    expected_files = {root / path for path in _artifact_paths()} | {
        manifest_path,
        root / "completion.json",
    }
    actual_files = {path for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("blind-pack exact file-set audit failed")
    artifacts = [_file_record(root, relative) for relative in _artifact_paths()]
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifacts") != artifacts
        or manifest.get("artifacts_sha256") != canonical_sha256(artifacts)
        or completion.get("complete") is not True
        or completion.get("identity_sha256") != identity_hash
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or completion.get("artifacts_sha256") != canonical_sha256(artifacts)
        or completion.get("endpoint_count") != 360
        or completion.get("grid_count") != 18
        or completion.get("grid_cell_count") != 360
        or completion.get("forbidden_seed_count") != 0
    ):
        raise RuntimeError("blind-pack completion binding failed")
    contract = load_json(root / REVIEW_CONTRACT)
    if contract != _review_contract(identity_hash):
        raise RuntimeError("blind-pack review contract changed")
    return identity


def build(trace_root: Path, output: Path) -> Path:
    validate_candidate_lock()
    validate_expansion_lock()
    validate_pipeline_source_lock(Path(__file__).name)
    trace_root = trace_root.expanduser().resolve()
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        if output.is_dir() and not output.is_symlink():
            validate_completed(output)
            return output
        raise RuntimeError(f"output exists or is indirect: {output}")
    endpoints = _load_endpoints(trace_root)
    identity = _identity(trace_root, endpoints)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        source_rows = _source_rows(endpoints)
        for item in endpoints:
            destination = staging / endpoint_relative(item.class_id, item.seed)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.source_path, destination)
            if sha256_file(destination) != item.sha256:
                raise RuntimeError(f"endpoint copy changed: {destination}")
        cells: list[dict[str, Any]] = []
        for class_id in CLASSES:
            selected = [item for item in endpoints if item.class_id == class_id]
            for offset in range(0, len(selected), BLOCK_SIZE):
                block = selected[offset : offset + BLOCK_SIZE]
                destination = staging / grid_relative(
                    class_id, block[0].seed, block[-1].seed
                )
                cells.extend(
                    _render_grid(block, destination, identity["identity_sha256"])
                )
        _write_jsonl(staging / SOURCE_INDEX, source_rows)
        _write_jsonl(staging / CELL_INDEX, cells)
        write_json(staging / REVIEW_CONTRACT, _review_contract(identity["identity_sha256"]))
        artifacts = [_file_record(staging, relative) for relative in _artifact_paths()]
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "source_endpoints": source_rows,
            "grid_cells": cells,
            "artifacts": artifacts,
            "artifacts_sha256": canonical_sha256(artifacts),
        }
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "identity_sha256": identity["identity_sha256"],
                "manifest_sha256": sha256_file(staging / "manifest.json"),
                "artifacts_sha256": canonical_sha256(artifacts),
                "artifact_count": len(artifacts),
                "endpoint_count": 360,
                "grid_count": 18,
                "grid_cell_count": 360,
                "forbidden_seed_count": 0,
            },
        )
        validate_completed(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    assert len(expected_endpoint_paths()) == 360
    assert len(expected_grid_paths()) == 18
    assert expected_grid_paths()[0].endswith("seeds130-149.png")
    assert expected_grid_paths()[-1].endswith("seeds230-249.png")
    assert not set(EXPANSION_SEEDS) & set((*CALIBRATION_SEEDS, *ORIGINAL_EVALUATION_SEEDS))
    print("self-test passed")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.trace_root is None or args.output is None:
        parser.error("--trace-root and --output are required")
    output = build(args.trace_root, args.output)
    identity = validate_completed(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "identity_sha256": identity["identity_sha256"],
                "endpoint_count": 360,
                "grid_count": 18,
                "metric_or_label_evidence_included": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
