#!/usr/bin/env python3
"""Build the frozen reviewer-only DiT class-207 visual-anchor pack.

The lineage-bearing repository configuration is the only place that names old
generation directories or binds old PNG hashes.  This builder validates those
bindings, decodes exactly seven 256x256 RGB images, and exports only canonical
metadata-free PNGs under fixed category IDs.  The public manifest contains the
anchor-configuration identity and output-side hashes, but no reversible source
lineage.

The output is a closed directory installed atomically with Linux
RENAME_NOREPLACE.  ``--self-test`` constructs all inputs inside a temporary
directory and deliberately never loads the release configuration or real
anchor images.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import stat
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

sys.dont_write_bytecode = True

from PIL import Image, PngImagePlugin


PACK_NAME = "dit_imagenet256_class207_visual_anchor_pack_v1"
SCHEMA_VERSION = 1
RUBRIC_SCHEMA_VERSION = 1
ANNOTATION_SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1
RELEASE_CONFIG_IDENTITY_SHA256 = (
    "e2a6ebf5eeb0e43cba302c40f3f54151bf542debff2d25df4c540a5d9cbdb38a"
)
FROZEN_STATUS = "FROZEN_BEFORE_CROSS_PREFIX_POOL"

RUNNER = Path(__file__).resolve()
DEFAULT_CONFIG = (
    RUNNER.parent / "configs/dit_imagenet256_class207_visual_anchors_v1.json"
)

ANCHOR_DIR_NAME = "anchors"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
RUBRIC_NAME = "rubric.json"
ANNOTATION_NAME = "annotation_template.json"
README_NAME = "README.txt"
ORDINARY_SHEET_NAME = "ordinary_anchors_sheet.png"
CLEAR_BAD_SHEET_NAME = "clear_bad_anchors_sheet.png"

IMAGE_SIZE = (256, 256)
IMAGE_MODE = "RGB"
ANCHOR_IDS = (
    "ordinary_anchor_01",
    "ordinary_anchor_02",
    "ordinary_anchor_03",
    "ordinary_anchor_04",
    "ordinary_anchor_05",
    "clear_bad_topology_01",
    "clear_bad_blur_01",
)
ANCHOR_CATEGORIES = (
    "ordinary",
    "ordinary",
    "ordinary",
    "ordinary",
    "ordinary",
    "clear_bad_topology",
    "clear_bad_blur",
)
ANCHOR_ROLES = (
    "ordinary_same_model_reference",
    "ordinary_same_model_reference",
    "ordinary_same_model_reference",
    "ordinary_same_model_reference",
    "ordinary_same_model_reference",
    "clear_bad_topology_calibration",
    "clear_bad_blur_calibration",
)
ORDINARY_IDS = ANCHOR_IDS[:5]
CLEAR_BAD_IDS = ANCHOR_IDS[5:]

SHEET_MARGIN = 8
SHEET_GAP = 8
SHEET_BACKGROUND = (238, 238, 238)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_CHUNK_ORDER = (b"IHDR", b"IDAT", b"IEND")

PUBLIC_TEXT_FILES = (
    RUBRIC_NAME,
    ANNOTATION_NAME,
    README_NAME,
    MANIFEST_NAME,
    COMPLETION_NAME,
)
PUBLIC_LINEAGE_FRAGMENTS = (
    "official_demo",
    "targeted_scan",
    "seed",
    "source_root",
    "source_lineage",
    "source_png_sha256",
    "source_pixel_rgb_sha256",
    "relative_png_path",
    "/home/",
    "/data/",
)


@dataclass(frozen=True)
class ValidatedAnchor:
    anchor_id: str
    category: str
    rgb_bytes: bytes


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return _sha256_bytes(_canonical_json_bytes(stripped))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected a plain JSON file: {path}")
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = _load_json(path)
    observed = payload.get(key)
    if not _is_sha256(observed) or observed != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid canonical {key}: {path}")
    return payload


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_bytes_noreplace(path: Path, value: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_noreplace(path: Path, value: Any) -> None:
    _write_bytes_noreplace(path, _pretty_json_bytes(value))


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _canonical_png_bytes(image: Image.Image) -> bytes:
    if image.mode != IMAGE_MODE:
        raise RuntimeError("canonical PNG encoder accepts RGB only")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise RuntimeError("canonical PNG encoder rejects empty geometry")
    pixels = image.tobytes()
    row_bytes = width * 3
    if len(pixels) != row_bytes * height:
        raise RuntimeError("RGB byte count does not match geometry")
    scanlines = b"".join(
        b"\x00" + pixels[offset : offset + row_bytes]
        for offset in range(0, len(pixels), row_bytes)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            PNG_SIGNATURE,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def _decode_canonical_png(path: Path) -> tuple[Image.Image, bytes]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected a plain PNG file: {path}")
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"invalid PNG signature: {path}")
    position = len(PNG_SIGNATURE)
    chunk_types: list[bytes] = []
    while position < len(data):
        if position + 12 > len(data):
            raise RuntimeError(f"truncated PNG chunk: {path}")
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(data):
            raise RuntimeError(f"truncated PNG payload: {path}")
        payload = data[position + 8 : position + 8 + length]
        observed_crc = struct.unpack(">I", data[position + 8 + length : end])[0]
        expected_crc = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise RuntimeError(f"PNG CRC mismatch: {path}")
        chunk_types.append(chunk_type)
        position = end
        if chunk_type == b"IEND":
            break
    if position != len(data) or tuple(chunk_types) != PNG_CHUNK_ORDER:
        raise RuntimeError(f"PNG is not the metadata-free canonical chunk set: {path}")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != IMAGE_MODE:
                raise RuntimeError(f"PNG is not decoded RGB: {path}")
            image = Image.frombytes(IMAGE_MODE, opened.size, opened.tobytes())
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"PNG decode failed: {path}") from exc
    if data != _canonical_png_bytes(image):
        raise RuntimeError(f"PNG bytes are not the sole canonical encoding: {path}")
    return image, data


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected a plain output file: {path}")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
    }


def _png_record(path: Path, root: Path) -> dict[str, Any]:
    image, data = _decode_canonical_png(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "file_sha256": _sha256_bytes(data),
        "pixel_rgb_sha256": _sha256_bytes(image.tobytes()),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }


def _require_nonempty_strings(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError(f"{label} schema changed")
    if not all(isinstance(value[key], str) and value[key] for key in keys):
        raise RuntimeError(f"{label} must contain nonempty strings")


def validate_anchor_config(
    payload: dict[str, Any],
    expected_identity_sha256: str | None = RELEASE_CONFIG_IDENTITY_SHA256,
) -> dict[str, Any]:
    expected_top_keys = {
        "anchor_config_identity_sha256",
        "anchor_config_name",
        "anchor_count",
        "anchor_status",
        "freeze_and_selection_provenance",
        "hash_conventions",
        "image_contract",
        "ordered_anchors",
        "reviewer_export_contract",
        "rubric_contract",
        "schema_version",
        "source_root",
    }
    if set(payload) != expected_top_keys:
        raise RuntimeError("anchor configuration top-level schema changed")
    observed_identity = payload.get("anchor_config_identity_sha256")
    if not _is_sha256(observed_identity) or observed_identity != _canonical_self_hash(
        payload, "anchor_config_identity_sha256"
    ):
        raise RuntimeError("anchor configuration canonical self-hash failed")
    if expected_identity_sha256 is not None and observed_identity != expected_identity_sha256:
        raise RuntimeError("builder is not bound to this anchor configuration identity")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("anchor configuration schema version changed")
    if not isinstance(payload.get("anchor_config_name"), str) or not payload[
        "anchor_config_name"
    ]:
        raise RuntimeError("anchor configuration name is invalid")
    if payload.get("anchor_status") != FROZEN_STATUS:
        raise RuntimeError("anchor configuration is not frozen before the new pool")
    if payload.get("anchor_count") != len(ANCHOR_IDS):
        raise RuntimeError("anchor count changed")
    source_root = payload.get("source_root")
    if not isinstance(source_root, str) or not Path(source_root).is_absolute():
        raise RuntimeError("source root must be an absolute path")

    _require_nonempty_strings(
        payload.get("freeze_and_selection_provenance"),
        {
            "current_pool_independence",
            "selection_basis",
            "selection_scope",
            "status_semantics",
        },
        "freeze/selection provenance",
    )
    image_contract = payload.get("image_contract")
    if image_contract != {
        "class_id": 207,
        "class_name": "golden retriever",
        "height": IMAGE_SIZE[1],
        "mode": IMAGE_MODE,
        "source_format": "PNG",
        "width": IMAGE_SIZE[0],
    }:
        raise RuntimeError("source image contract changed")
    conventions = payload.get("hash_conventions")
    if not isinstance(conventions, dict) or set(conventions) != {
        "canonical_json",
        "canonical_self_hash",
        "source_pixel_rgb_sha256",
        "source_png_sha256",
    }:
        raise RuntimeError("hash convention schema changed")
    if conventions.get("canonical_json") != {
        "encoding": "UTF-8",
        "ensure_ascii": False,
        "separators": [",", ":"],
        "sort_keys": True,
    }:
        raise RuntimeError("canonical JSON convention changed")
    if not all(
        isinstance(conventions.get(key), str) and conventions[key]
        for key in (
            "canonical_self_hash",
            "source_pixel_rgb_sha256",
            "source_png_sha256",
        )
    ):
        raise RuntimeError("hash convention text is invalid")

    anchors = payload.get("ordered_anchors")
    if not isinstance(anchors, list) or len(anchors) != len(ANCHOR_IDS):
        raise RuntimeError("ordered anchor list changed")
    relative_paths: list[str] = []
    source_file_hashes: list[str] = []
    source_pixel_hashes: list[str] = []
    for index, (anchor, expected_id, expected_category, expected_role) in enumerate(
        zip(anchors, ANCHOR_IDS, ANCHOR_CATEGORIES, ANCHOR_ROLES)
    ):
        if not isinstance(anchor, dict) or set(anchor) != {
            "category",
            "opaque_anchor_id",
            "review_role",
            "selection_note",
            "source_lineage",
        }:
            raise RuntimeError(f"anchor record {index} schema changed")
        if anchor.get("opaque_anchor_id") != expected_id:
            raise RuntimeError("opaque anchor order or identity changed")
        if anchor.get("category") != expected_category:
            raise RuntimeError("anchor category changed")
        if anchor.get("review_role") != expected_role:
            raise RuntimeError("anchor review role changed")
        if not isinstance(anchor.get("selection_note"), str) or not anchor[
            "selection_note"
        ]:
            raise RuntimeError("anchor selection note is invalid")
        lineage = anchor.get("source_lineage")
        if not isinstance(lineage, dict) or set(lineage) != {
            "relative_png_path",
            "source_pixel_rgb_sha256",
            "source_png_sha256",
        }:
            raise RuntimeError("source lineage schema changed")
        relative = lineage.get("relative_png_path")
        if not isinstance(relative, str):
            raise RuntimeError("source relative path is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in ("", ".", "..") for part in pure.parts)
            or pure.suffix.lower() != ".png"
        ):
            raise RuntimeError("source relative path escapes its root or is not PNG")
        file_hash = lineage.get("source_png_sha256")
        pixel_hash = lineage.get("source_pixel_rgb_sha256")
        if not _is_sha256(file_hash) or not _is_sha256(pixel_hash):
            raise RuntimeError("source hash is invalid")
        relative_paths.append(relative)
        source_file_hashes.append(file_hash)
        source_pixel_hashes.append(pixel_hash)
    if len(set(relative_paths)) != len(ANCHOR_IDS):
        raise RuntimeError("source paths are not unique")
    if len(set(source_file_hashes)) != len(ANCHOR_IDS):
        raise RuntimeError("source PNG hashes are not unique")
    if len(set(source_pixel_hashes)) != len(ANCHOR_IDS):
        raise RuntimeError("source pixel hashes are not unique")

    export = payload.get("reviewer_export_contract")
    if not isinstance(export, dict) or set(export) != {
        "allowed_anchor_filenames",
        "closed_directory",
        "manifest_privacy",
        "png_export",
        "sheet_order",
    }:
        raise RuntimeError("reviewer export contract schema changed")
    if export.get("allowed_anchor_filenames") != [f"{value}.png" for value in ANCHOR_IDS]:
        raise RuntimeError("allowed anchor filenames changed")
    if export.get("sheet_order") != {
        CLEAR_BAD_SHEET_NAME: list(CLEAR_BAD_IDS),
        ORDINARY_SHEET_NAME: list(ORDINARY_IDS),
    }:
        raise RuntimeError("sheet order changed")
    for key in ("closed_directory", "manifest_privacy", "png_export"):
        if not isinstance(export.get(key), str) or not export[key]:
            raise RuntimeError("reviewer export contract text is invalid")

    rubric = payload.get("rubric_contract")
    if not isinstance(rubric, dict) or set(rubric) != {
        "clear_bad_scope",
        "independent_per_image_rule",
        "ordinary_band",
        "primary_labels",
        "tail_rule",
        "viewing_rule",
    }:
        raise RuntimeError("rubric contract schema changed")
    if rubric.get("primary_labels") != [
        "clear_overall_structural_bad",
        "not_clear_overall_structural_bad",
        "uncertain",
    ]:
        raise RuntimeError("primary labels changed")
    for key in (
        "clear_bad_scope",
        "independent_per_image_rule",
        "ordinary_band",
        "tail_rule",
        "viewing_rule",
    ):
        if not isinstance(rubric.get(key), str) or not rubric[key]:
            raise RuntimeError("rubric contract text is invalid")
    return payload


def load_anchor_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return validate_anchor_config(_load_json(path), RELEASE_CONFIG_IDENTITY_SHA256)


def _read_bound_source(root: Path, relative: str) -> bytes:
    pure = PurePosixPath(relative)
    current = root
    for part in pure.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise RuntimeError("a bound source PNG is missing") from exc
        if stat.S_ISLNK(mode):
            raise RuntimeError("bound source path contains a symbolic link")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise RuntimeError("bound source PNG is not a regular file")
    return current.read_bytes()


def validate_sources(config: dict[str, Any]) -> tuple[ValidatedAnchor, ...]:
    root = Path(config["source_root"])
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise RuntimeError("source root is missing") from exc
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise RuntimeError("source root must be a plain directory")
    validated: list[ValidatedAnchor] = []
    for anchor in config["ordered_anchors"]:
        lineage = anchor["source_lineage"]
        source_bytes = _read_bound_source(root, lineage["relative_png_path"])
        if _sha256_bytes(source_bytes) != lineage["source_png_sha256"]:
            raise RuntimeError("bound source PNG SHA-256 mismatch")
        try:
            with Image.open(io.BytesIO(source_bytes)) as opened:
                opened.load()
                if opened.format != "PNG":
                    raise RuntimeError("bound source is not decoded as PNG")
                if opened.mode != IMAGE_MODE or opened.size != IMAGE_SIZE:
                    raise RuntimeError("bound source violates the exact 256x256 RGB contract")
                pixels = opened.tobytes()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("bound source PNG decode failed") from exc
        if len(pixels) != IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3:
            raise RuntimeError("bound source RGB byte count changed")
        if _sha256_bytes(pixels) != lineage["source_pixel_rgb_sha256"]:
            raise RuntimeError("bound source RGB-pixel SHA-256 mismatch")
        validated.append(
            ValidatedAnchor(
                anchor_id=anchor["opaque_anchor_id"],
                category=anchor["category"],
                rgb_bytes=pixels,
            )
        )
    if tuple(item.anchor_id for item in validated) != ANCHOR_IDS:
        raise RuntimeError("validated source ordering changed")
    return tuple(validated)


def _sheet_geometry(count: int, columns: int) -> tuple[int, int, int]:
    if count <= 0 or columns <= 0:
        raise RuntimeError("sheet geometry requires positive values")
    rows = math.ceil(count / columns)
    width = 2 * SHEET_MARGIN + columns * IMAGE_SIZE[0] + (columns - 1) * SHEET_GAP
    height = 2 * SHEET_MARGIN + rows * IMAGE_SIZE[1] + (rows - 1) * SHEET_GAP
    return width, height, rows


def _render_sheet(images: Iterable[Image.Image], columns: int) -> Image.Image:
    items = tuple(images)
    width, height, _ = _sheet_geometry(len(items), columns)
    canvas = Image.new(IMAGE_MODE, (width, height), SHEET_BACKGROUND)
    for index, image in enumerate(items):
        if image.mode != IMAGE_MODE or image.size != IMAGE_SIZE:
            raise RuntimeError("sheet input violates the 256x256 RGB contract")
        row, column = divmod(index, columns)
        left = SHEET_MARGIN + column * (IMAGE_SIZE[0] + SHEET_GAP)
        top = SHEET_MARGIN + row * (IMAGE_SIZE[1] + SHEET_GAP)
        canvas.paste(image, (left, top))
    return canvas


def _sheet_layout(count: int, columns: int) -> dict[str, Any]:
    width, height, rows = _sheet_geometry(count, columns)
    return {
        "columns": columns,
        "rows": rows,
        "margin_pixels": SHEET_MARGIN,
        "gap_pixels": SHEET_GAP,
        "width": width,
        "height": height,
        "order": "left_to_right_then_top_to_bottom",
    }


def build_rubric(anchor_config_identity_sha256: str) -> dict[str, Any]:
    if not _is_sha256(anchor_config_identity_sha256):
        raise RuntimeError("invalid anchor configuration identity for rubric")
    payload: dict[str, Any] = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_name": "class207_fixed_external_anchor_independent_review_v1",
        "anchor_config_identity_sha256": anchor_config_identity_sha256,
        "purpose": (
            "Classify visible quality for each new class-207 endpoint independently "
            "against a fixed external visual scale frozen before the current pool."
        ),
        "fixed_anchor_sets": {
            "ordinary": list(ORDINARY_IDS),
            "clear_bad": list(CLEAR_BAD_IDS),
        },
        "decision_unit": {
            "unit": "one new endpoint image",
            "independence_rule": (
                "Judge this image only against the written rule and the seven fixed anchors. "
                "The other 63 new images must not influence its label."
            ),
            "current_pool_prohibition": (
                "Do not estimate, recalibrate, or move the threshold from the current "
                "64-image pool's distribution, prevalence, quantiles, contact sheet, "
                "majority appearance, or emerging annotations."
            ),
            "viewing": "Inspect at native 256x256 resolution and at an enlarged view.",
        },
        "calibration_interpretation": {
            "ordinary_band": (
                "The five ordinary anchors define ordinary same-model quality. Common mild "
                "imperfections, mild oddness, imperfect texture, stylization, unusual pose, "
                "and ordinary model limitations are allowed and are not clear badness alone."
            ),
            "clear_bad_band": (
                "The clear-bad anchors demonstrate only conspicuous failures clearly below "
                "the ordinary band: a major topology/body-tail-attachment failure and an "
                "obvious global subject blur."
            ),
            "example_scope": (
                "The clear-bad anchors calibrate severity, not a closed defect vocabulary. "
                "Their exact poses, locations, and shapes need not be copied by a new failure."
            ),
        },
        "primary_endpoint": {
            "name": "overall_obvious_structural_bad_under_frozen_external_anchor_rubric",
            "labels": [
                "clear_overall_structural_bad",
                "not_clear_overall_structural_bad",
                "uncertain",
            ],
            "positive_definition": (
                "At native and enlarged view, an obvious blur, fusion, duplicated or floating "
                "part, broken attachment, impossible overlap, severe displacement, or major "
                "topology failure is comparable in conspicuousness to the clear-bad anchors "
                "and clearly below the ordinary anchors."
            ),
            "negative_definition": (
                "No visible defect reaches that conspicuous clear-bad level relative to the "
                "ordinary anchors. Mild oddness or a common model imperfection is insufficient."
            ),
            "uncertain_rule": (
                "Use uncertain only when the visible pixels do not support a confident call; "
                "do not force ambiguity into either analysis class."
            ),
            "pose_invariance": (
                "Apply the rule across poses and compositions; do not require one location, "
                "object count, orientation, or defect silhouette."
            ),
        },
        "secondary_checks": {
            "hind_limb_topology_labels": [
                "clear_failure",
                "not_clear_failure",
                "uncertain_or_not_scorable",
            ],
            "tail_identity_is_not_naturalness": (
                "Recognizing a depicted part as a tail does not establish natural root "
                "attachment, taper, volume, distal tip, or hair/feather flow. Record identity "
                "and naturalness separately."
            ),
            "tail_visibility_rule": (
                "A missing, hidden, or occluded tail is not automatically bad; use not "
                "scorable when its geometry cannot be judged."
            ),
            "secondary_role": (
                "Secondary checks describe visible structure but cannot redefine, rescue, "
                "or replace the primary endpoint."
            ),
        },
        "sheet_restriction": (
            "The two sheets in this pack display only the fixed anchors. Any sheet for the "
            "current pool may be used only for missing, duplicate, corruption, or layout QA, "
            "never to set the ordinary-quality threshold."
        ),
    }
    payload["rubric_identity_sha256"] = _canonical_self_hash(
        payload, "rubric_identity_sha256"
    )
    return payload


def build_annotation_template(
    anchor_config_identity_sha256: str, rubric_identity_sha256: str
) -> dict[str, Any]:
    if not _is_sha256(anchor_config_identity_sha256) or not _is_sha256(
        rubric_identity_sha256
    ):
        raise RuntimeError("invalid identity binding for annotation template")
    payload: dict[str, Any] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "template_name": "class207_independent_fixed_anchor_annotation_template_v1",
        "anchor_config_identity_sha256": anchor_config_identity_sha256,
        "rubric_identity_sha256": rubric_identity_sha256,
        "usage": (
            "Copy this file outside the immutable anchor pack, add exactly one row per opaque "
            "new-image ID, complete every field independently, and lock the completed copy "
            "before viewing evidence or lineage. The immutable template intentionally has no rows."
        ),
        "row_schema": {
            "image_id": "required opaque new-image ID",
            "primary_label": (
                "required enum: clear_overall_structural_bad | "
                "not_clear_overall_structural_bad | uncertain"
            ),
            "hind_limb_topology": (
                "required enum: clear_failure | not_clear_failure | "
                "uncertain_or_not_scorable"
            ),
            "tail_scorable": "required enum: yes | no",
            "tail_identity": "required enum: clear | plausible | unclear",
            "tail_naturalness": (
                "required enum: natural | odd | malformed | uncertain_or_not_scorable; "
                "must be judged separately from tail identity"
            ),
            "notes": "optional visible-appearance notes only",
        },
        "reviewer_declaration": {
            "statement": (
                "I judged every new image independently against the fixed anchors and rubric, "
                "did not use the current pool to move the threshold, and did not view evidence "
                "or lineage before annotation lock."
            ),
            "reviewer_name": None,
            "review_started_utc": None,
            "review_completed_utc": None,
            "signed_confirmation": None,
        },
        "rows": [],
    }
    payload["template_identity_sha256"] = _canonical_self_hash(
        payload, "template_identity_sha256"
    )
    return payload


def build_readme(
    anchor_config_identity_sha256: str,
    rubric_identity_sha256: str,
    template_identity_sha256: str,
) -> str:
    return "\n".join(
        (
            "DiT ImageNet-256 class-207 fixed visual anchors (reviewer-only)",
            "",
            f"Anchor configuration identity: {anchor_config_identity_sha256}",
            f"Rubric identity: {rubric_identity_sha256}",
            f"Annotation-template identity: {template_identity_sha256}",
            "",
            "This pack was frozen before the current 64-image pool.",
            "Judge each new image independently against these anchors and rubric.",
            "Never use the other new images or their distribution to move the threshold.",
            "Ordinary anchors permit common mild same-model imperfections.",
            "Clear-bad anchors illustrate conspicuous topology/attachment failure and blur only.",
            "Tail identity is not tail naturalness.",
            "",
            "ordinary_anchors_sheet.png order (left-to-right, then top-to-bottom):",
            *[f"  {value}" for value in ORDINARY_IDS],
            "",
            "clear_bad_anchors_sheet.png order (left-to-right, then top-to-bottom):",
            *[f"  {value}" for value in CLEAR_BAD_IDS],
            "",
            "annotation_template.json intentionally contains zero rows. Copy it outside this",
            "immutable pack and populate it with opaque IDs from the separate pool delivery.",
            "",
        )
    )


def _reject_special_entries(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("anchor-pack root must be a plain directory")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError("anchor pack contains a link or special entry")


def _payload_file_records(root: Path) -> list[dict[str, Any]]:
    paths = [
        *(root / ANCHOR_DIR_NAME / f"{anchor_id}.png" for anchor_id in ANCHOR_IDS),
        root / ORDINARY_SHEET_NAME,
        root / CLEAR_BAD_SHEET_NAME,
        root / RUBRIC_NAME,
        root / ANNOTATION_NAME,
        root / README_NAME,
    ]
    return [_file_record(path, root) for path in sorted(paths, key=lambda p: p.as_posix())]


def _pack_payload_sha256(records: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(records))


def _atomic_install_directory_noreplace(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, "refusing to replace anchor pack", target)
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("filesystem/kernel lacks atomic RENAME_NOREPLACE")
    raise OSError(error_number, os.strerror(error_number), target)


def _write_bundle(
    outdir: Path,
    config: dict[str, Any],
    expected_config_identity_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if os.path.lexists(outdir):
        raise RuntimeError("refusing to overwrite an existing anchor-pack path")
    validate_anchor_config(config, expected_config_identity_sha256)
    anchors = validate_sources(config)
    config_identity = config["anchor_config_identity_sha256"]

    outdir.parent.mkdir(parents=True, exist_ok=True)
    if not outdir.parent.is_dir() or outdir.parent.is_symlink():
        raise RuntimeError("anchor-pack parent must be a plain directory")
    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary) / "pack"
        staging.mkdir()
        anchor_dir = staging / ANCHOR_DIR_NAME
        anchor_dir.mkdir()

        anchor_records: list[dict[str, Any]] = []
        rendered: dict[str, Image.Image] = {}
        for anchor in anchors:
            image = Image.frombytes(IMAGE_MODE, IMAGE_SIZE, anchor.rgb_bytes)
            destination = anchor_dir / f"{anchor.anchor_id}.png"
            _write_bytes_noreplace(destination, _canonical_png_bytes(image))
            inspection = _png_record(destination, staging)
            if inspection["pixel_rgb_sha256"] != _sha256_bytes(anchor.rgb_bytes):
                raise RuntimeError("canonical export changed source RGB pixels")
            anchor_records.append(
                {
                    "anchor_id": anchor.anchor_id,
                    "category": anchor.category,
                    "image": inspection,
                }
            )
            rendered[anchor.anchor_id] = image

        ordinary_sheet = _render_sheet(
            (rendered[anchor_id] for anchor_id in ORDINARY_IDS), columns=3
        )
        ordinary_sheet_path = staging / ORDINARY_SHEET_NAME
        _write_bytes_noreplace(ordinary_sheet_path, _canonical_png_bytes(ordinary_sheet))
        clear_bad_sheet = _render_sheet(
            (rendered[anchor_id] for anchor_id in CLEAR_BAD_IDS), columns=2
        )
        clear_bad_sheet_path = staging / CLEAR_BAD_SHEET_NAME
        _write_bytes_noreplace(clear_bad_sheet_path, _canonical_png_bytes(clear_bad_sheet))

        rubric = build_rubric(config_identity)
        _write_json_noreplace(staging / RUBRIC_NAME, rubric)
        annotation = build_annotation_template(
            config_identity, rubric["rubric_identity_sha256"]
        )
        _write_json_noreplace(staging / ANNOTATION_NAME, annotation)
        readme = build_readme(
            config_identity,
            rubric["rubric_identity_sha256"],
            annotation["template_identity_sha256"],
        )
        _write_bytes_noreplace(staging / README_NAME, readme.encode("utf-8"))

        payload_records = _payload_file_records(staging)
        payload_identity = _pack_payload_sha256(payload_records)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pack_name": PACK_NAME,
            "role": "REVIEWER_ONLY_FIXED_EXTERNAL_VISUAL_CALIBRATION",
            "anchor_config_identity_sha256": config_identity,
            "frozen_before_current_pool": True,
            "anchor_count": len(ANCHOR_IDS),
            "anchor_order": list(ANCHOR_IDS),
            "anchors": anchor_records,
            "sheets": {
                "ordinary": {
                    "ordered_anchor_ids": list(ORDINARY_IDS),
                    "layout": _sheet_layout(len(ORDINARY_IDS), 3),
                    "image": _png_record(ordinary_sheet_path, staging),
                },
                "clear_bad": {
                    "ordered_anchor_ids": list(CLEAR_BAD_IDS),
                    "layout": _sheet_layout(len(CLEAR_BAD_IDS), 2),
                    "image": _png_record(clear_bad_sheet_path, staging),
                },
            },
            "rubric": _file_record(staging / RUBRIC_NAME, staging),
            "annotation_template": _file_record(staging / ANNOTATION_NAME, staging),
            "readme": _file_record(staging / README_NAME, staging),
            "payload_file_count": len(payload_records),
            "payload_files": payload_records,
            "pack_payload_sha256": payload_identity,
        }
        manifest["identity_sha256"] = _canonical_self_hash(manifest, "identity_sha256")
        _write_json_noreplace(staging / MANIFEST_NAME, manifest)
        completion: dict[str, Any] = {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "complete": True,
            "anchor_config_identity_sha256": config_identity,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": _sha256_file(staging / MANIFEST_NAME),
            "pack_payload_sha256": payload_identity,
            "anchor_count": len(ANCHOR_IDS),
            "file_count": len(payload_records) + 2,
        }
        completion["payload_sha256"] = _canonical_self_hash(
            completion, "payload_sha256"
        )
        _write_json_noreplace(staging / COMPLETION_NAME, completion)
        validate_output_bundle(staging, config_identity)
        _atomic_install_directory_noreplace(staging, outdir)
    return validate_output_bundle(outdir, config_identity)


def _validate_public_lineage_absence(root: Path) -> None:
    for filename in PUBLIC_TEXT_FILES:
        text = (root / filename).read_bytes().decode("utf-8").lower()
        for fragment in PUBLIC_LINEAGE_FRAGMENTS:
            if fragment in text:
                raise RuntimeError(
                    f"reviewer-visible text contains forbidden lineage fragment: {fragment}"
                )


def validate_output_bundle(
    root: Path,
    expected_config_identity_sha256: str | None = RELEASE_CONFIG_IDENTITY_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly validate one closed public pack without reading any source image."""

    root = Path(root)
    _reject_special_entries(root)
    expected_files = {
        (root / ANCHOR_DIR_NAME / f"{anchor_id}.png").resolve()
        for anchor_id in ANCHOR_IDS
    }
    expected_files.update(
        {
            (root / ORDINARY_SHEET_NAME).resolve(),
            (root / CLEAR_BAD_SHEET_NAME).resolve(),
            (root / RUBRIC_NAME).resolve(),
            (root / ANNOTATION_NAME).resolve(),
            (root / README_NAME).resolve(),
            (root / MANIFEST_NAME).resolve(),
            (root / COMPLETION_NAME).resolve(),
        }
    )
    actual_files = {
        path.resolve()
        for path in root.rglob("*")
        if stat.S_ISREG(path.lstat().st_mode)
    }
    actual_directories = {
        path.resolve()
        for path in root.rglob("*")
        if stat.S_ISDIR(path.lstat().st_mode)
    }
    if actual_files != expected_files or actual_directories != {
        (root / ANCHOR_DIR_NAME).resolve()
    }:
        raise RuntimeError("anchor pack is not the exact closed file/directory set")

    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "identity_sha256")
    expected_manifest_keys = {
        "schema_version",
        "pack_name",
        "role",
        "anchor_config_identity_sha256",
        "frozen_before_current_pool",
        "anchor_count",
        "anchor_order",
        "anchors",
        "sheets",
        "rubric",
        "annotation_template",
        "readme",
        "payload_file_count",
        "payload_files",
        "pack_payload_sha256",
        "identity_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise RuntimeError("public manifest schema changed")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
        "pack_name"
    ) != PACK_NAME:
        raise RuntimeError("public manifest identity changed")
    if manifest.get("role") != "REVIEWER_ONLY_FIXED_EXTERNAL_VISUAL_CALIBRATION":
        raise RuntimeError("public manifest role changed")
    config_identity = manifest.get("anchor_config_identity_sha256")
    if not _is_sha256(config_identity):
        raise RuntimeError("public manifest has invalid anchor configuration identity")
    if (
        expected_config_identity_sha256 is not None
        and config_identity != expected_config_identity_sha256
    ):
        raise RuntimeError("public pack is bound to the wrong anchor configuration")
    if manifest.get("frozen_before_current_pool") is not True:
        raise RuntimeError("public manifest lost its pre-pool freeze declaration")
    if manifest.get("anchor_count") != len(ANCHOR_IDS) or manifest.get(
        "anchor_order"
    ) != list(ANCHOR_IDS):
        raise RuntimeError("public anchor count/order changed")

    records = manifest.get("anchors")
    if not isinstance(records, list) or len(records) != len(ANCHOR_IDS):
        raise RuntimeError("public anchor records changed")
    decoded: dict[str, Image.Image] = {}
    for record, anchor_id, category in zip(records, ANCHOR_IDS, ANCHOR_CATEGORIES):
        path = root / ANCHOR_DIR_NAME / f"{anchor_id}.png"
        image, _ = _decode_canonical_png(path)
        if image.size != IMAGE_SIZE:
            raise RuntimeError("exported anchor geometry changed")
        expected_record = {
            "anchor_id": anchor_id,
            "category": category,
            "image": _png_record(path, root),
        }
        if record != expected_record:
            raise RuntimeError("public anchor output record changed")
        decoded[anchor_id] = image

    sheets = manifest.get("sheets")
    if not isinstance(sheets, dict) or set(sheets) != {"ordinary", "clear_bad"}:
        raise RuntimeError("public sheet schema changed")
    sheet_specs = (
        (
            "ordinary",
            ORDINARY_SHEET_NAME,
            ORDINARY_IDS,
            3,
        ),
        (
            "clear_bad",
            CLEAR_BAD_SHEET_NAME,
            CLEAR_BAD_IDS,
            2,
        ),
    )
    for key, filename, ordered_ids, columns in sheet_specs:
        expected_image = _render_sheet(
            (decoded[anchor_id] for anchor_id in ordered_ids), columns=columns
        )
        path = root / filename
        observed_image, _ = _decode_canonical_png(path)
        if (
            observed_image.size != expected_image.size
            or observed_image.tobytes() != expected_image.tobytes()
        ):
            raise RuntimeError("sheet pixels do not exactly reconstruct from ordered anchors")
        expected_sheet = {
            "ordered_anchor_ids": list(ordered_ids),
            "layout": _sheet_layout(len(ordered_ids), columns),
            "image": _png_record(path, root),
        }
        if sheets.get(key) != expected_sheet:
            raise RuntimeError("public sheet record changed")

    rubric = _read_self_hashed_json(root / RUBRIC_NAME, "rubric_identity_sha256")
    if rubric != build_rubric(config_identity):
        raise RuntimeError("rubric is not the exact fixed independent-image rubric")
    annotation = _read_self_hashed_json(
        root / ANNOTATION_NAME, "template_identity_sha256"
    )
    if annotation != build_annotation_template(
        config_identity, rubric["rubric_identity_sha256"]
    ):
        raise RuntimeError("annotation template changed")
    if annotation.get("rows") != []:
        raise RuntimeError("immutable annotation template is not empty")
    expected_readme = build_readme(
        config_identity,
        rubric["rubric_identity_sha256"],
        annotation["template_identity_sha256"],
    ).encode("utf-8")
    if (root / README_NAME).read_bytes() != expected_readme:
        raise RuntimeError("reviewer README changed")
    if manifest.get("rubric") != _file_record(root / RUBRIC_NAME, root):
        raise RuntimeError("manifest rubric binding changed")
    if manifest.get("annotation_template") != _file_record(
        root / ANNOTATION_NAME, root
    ):
        raise RuntimeError("manifest annotation binding changed")
    if manifest.get("readme") != _file_record(root / README_NAME, root):
        raise RuntimeError("manifest README binding changed")

    payload_records = _payload_file_records(root)
    payload_identity = _pack_payload_sha256(payload_records)
    if (
        manifest.get("payload_file_count") != len(payload_records)
        or manifest.get("payload_files") != payload_records
        or manifest.get("pack_payload_sha256") != payload_identity
    ):
        raise RuntimeError("public payload-tree binding changed")

    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    expected_completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "complete": True,
        "anchor_config_identity_sha256": config_identity,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": _sha256_file(root / MANIFEST_NAME),
        "pack_payload_sha256": payload_identity,
        "anchor_count": len(ANCHOR_IDS),
        "file_count": len(payload_records) + 2,
        "payload_sha256": completion.get("payload_sha256"),
    }
    if completion != expected_completion:
        raise RuntimeError("completion binding changed")
    _validate_public_lineage_absence(root)
    return manifest, completion


def _expect_failure(operation: Callable[[], Any], label: str) -> None:
    try:
        operation()
    except (RuntimeError, FileExistsError):
        return
    raise AssertionError(f"negative self-test did not fail: {label}")


def _synthetic_pixels(index: int, size: tuple[int, int] = IMAGE_SIZE) -> bytes:
    width, height = size
    value = bytearray(width * height * 3)
    cursor = 0
    for y in range(height):
        for x in range(width):
            value[cursor] = (x + 29 * index) % 256
            value[cursor + 1] = (y * 3 + 47 * index) % 256
            value[cursor + 2] = (x + y + 71 * index) % 256
            cursor += 3
    return bytes(value)


def _synthetic_png(
    path: Path, index: int, mode: str = IMAGE_MODE, size: tuple[int, int] = IMAGE_SIZE
) -> tuple[str, str]:
    if mode == IMAGE_MODE:
        pixels = _synthetic_pixels(index, size)
        image = Image.frombytes(IMAGE_MODE, size, pixels)
    elif mode == "L":
        pixels = bytes((offset + 17 * index) % 256 for offset in range(size[0] * size[1]))
        image = Image.frombytes("L", size, pixels)
    else:
        raise AssertionError("unsupported synthetic mode")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("synthetic_self_test_only", f"case-{index}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, pnginfo=metadata)
    encoded = buffer.getvalue()
    if b"tEXt" not in encoded:
        raise AssertionError("synthetic PNG did not contain test metadata")
    _write_bytes_noreplace(path, encoded)
    return _sha256_bytes(encoded), _sha256_bytes(pixels)


def _rehash_config(payload: dict[str, Any]) -> None:
    payload["anchor_config_identity_sha256"] = _canonical_self_hash(
        payload, "anchor_config_identity_sha256"
    )


def _make_synthetic_config(root: Path) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    for index, (anchor_id, category, role) in enumerate(
        zip(ANCHOR_IDS, ANCHOR_CATEGORIES, ANCHOR_ROLES)
    ):
        relative = f"input_{index:02d}.png"
        png_hash, pixel_hash = _synthetic_png(root / relative, index)
        anchors.append(
            {
                "category": category,
                "opaque_anchor_id": anchor_id,
                "review_role": role,
                "selection_note": "synthetic-only validation anchor",
                "source_lineage": {
                    "relative_png_path": relative,
                    "source_pixel_rgb_sha256": pixel_hash,
                    "source_png_sha256": png_hash,
                },
            }
        )
    payload: dict[str, Any] = {
        "anchor_config_identity_sha256": "0" * 64,
        "anchor_config_name": "synthetic_visual_anchor_self_test",
        "anchor_count": len(ANCHOR_IDS),
        "anchor_status": FROZEN_STATUS,
        "freeze_and_selection_provenance": {
            "current_pool_independence": "synthetic-only",
            "selection_basis": "synthetic-only",
            "selection_scope": "synthetic-only",
            "status_semantics": "synthetic-only",
        },
        "hash_conventions": {
            "canonical_json": {
                "encoding": "UTF-8",
                "ensure_ascii": False,
                "separators": [",", ":"],
                "sort_keys": True,
            },
            "canonical_self_hash": "synthetic-only",
            "source_pixel_rgb_sha256": "synthetic-only",
            "source_png_sha256": "synthetic-only",
        },
        "image_contract": {
            "class_id": 207,
            "class_name": "golden retriever",
            "height": IMAGE_SIZE[1],
            "mode": IMAGE_MODE,
            "source_format": "PNG",
            "width": IMAGE_SIZE[0],
        },
        "ordered_anchors": anchors,
        "reviewer_export_contract": {
            "allowed_anchor_filenames": [f"{value}.png" for value in ANCHOR_IDS],
            "closed_directory": "synthetic-only",
            "manifest_privacy": "synthetic-only",
            "png_export": "synthetic-only",
            "sheet_order": {
                CLEAR_BAD_SHEET_NAME: list(CLEAR_BAD_IDS),
                ORDINARY_SHEET_NAME: list(ORDINARY_IDS),
            },
        },
        "rubric_contract": {
            "clear_bad_scope": "synthetic-only",
            "independent_per_image_rule": "synthetic-only",
            "ordinary_band": "synthetic-only",
            "primary_labels": [
                "clear_overall_structural_bad",
                "not_clear_overall_structural_bad",
                "uncertain",
            ],
            "tail_rule": "synthetic-only",
            "viewing_rule": "synthetic-only",
        },
        "schema_version": SCHEMA_VERSION,
        "source_root": str(root),
    }
    _rehash_config(payload)
    return payload


def _replace_first_synthetic_source(
    payload: dict[str, Any], path: Path, mode: str, size: tuple[int, int], index: int
) -> dict[str, Any]:
    modified = copy.deepcopy(payload)
    png_hash, pixel_hash = _synthetic_png(path, index=index, mode=mode, size=size)
    lineage = modified["ordered_anchors"][0]["source_lineage"]
    lineage["relative_png_path"] = path.name
    lineage["source_png_sha256"] = png_hash
    lineage["source_pixel_rgb_sha256"] = pixel_hash
    _rehash_config(modified)
    return modified


def run_self_test() -> None:
    # This branch creates its complete private fixture from scratch.  It does
    # not call load_anchor_config(), open DEFAULT_CONFIG, or touch the release
    # source root.
    with tempfile.TemporaryDirectory(prefix="dit-class207-anchor-self-test-") as temporary:
        root = Path(temporary)
        inputs = root / "synthetic_inputs"
        inputs.mkdir()
        config = _make_synthetic_config(inputs)
        validate_anchor_config(config, expected_identity_sha256=None)
        validated = validate_sources(config)
        if len(validated) != len(ANCHOR_IDS):
            raise AssertionError("synthetic source count changed")

        tampered_config = copy.deepcopy(config)
        tampered_config["anchor_status"] = "tampered"
        _expect_failure(
            lambda: validate_anchor_config(tampered_config, None),
            "configuration self-hash",
        )
        bad_hash_config = copy.deepcopy(config)
        bad_hash_config["ordered_anchors"][0]["source_lineage"][
            "source_png_sha256"
        ] = "0" * 64
        _rehash_config(bad_hash_config)
        _expect_failure(lambda: validate_sources(bad_hash_config), "source PNG hash")

        wrong_size = _replace_first_synthetic_source(
            config,
            inputs / "wrong_size.png",
            IMAGE_MODE,
            (255, 256),
            20,
        )
        _expect_failure(lambda: validate_sources(wrong_size), "source dimensions")
        wrong_mode = _replace_first_synthetic_source(
            config,
            inputs / "wrong_mode.png",
            "L",
            IMAGE_SIZE,
            21,
        )
        _expect_failure(lambda: validate_sources(wrong_mode), "source RGB mode")

        outdir = root / "reviewer_pack"
        manifest, completion = _write_bundle(outdir, config, None)
        validate_output_bundle(outdir, config["anchor_config_identity_sha256"])
        if completion["file_count"] != 14 or manifest["payload_file_count"] != 12:
            raise AssertionError("closed-set file count changed")
        for record, synthetic in zip(manifest["anchors"], validated):
            if record["image"]["pixel_rgb_sha256"] != _sha256_bytes(
                synthetic.rgb_bytes
            ):
                raise AssertionError("synthetic output pixels changed")
            _decode_canonical_png(outdir / record["image"]["relative_path"])

        first_path = outdir / ANCHOR_DIR_NAME / f"{ANCHOR_IDS[0]}.png"
        clean_bytes = first_path.read_bytes()
        metadata_buffer = io.BytesIO()
        first_image = Image.frombytes(IMAGE_MODE, IMAGE_SIZE, validated[0].rgb_bytes)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("forbidden_ancillary", "synthetic")
        first_image.save(metadata_buffer, format="PNG", pnginfo=metadata)
        first_path.write_bytes(metadata_buffer.getvalue())
        _expect_failure(
            lambda: validate_output_bundle(
                outdir, config["anchor_config_identity_sha256"]
            ),
            "ancillary PNG metadata",
        )
        first_path.write_bytes(clean_bytes)

        extra = outdir / "unexpected.txt"
        extra.write_text("synthetic", encoding="utf-8")
        _expect_failure(
            lambda: validate_output_bundle(
                outdir, config["anchor_config_identity_sha256"]
            ),
            "closed file set",
        )
        extra.unlink()
        validate_output_bundle(outdir, config["anchor_config_identity_sha256"])

        completion_before = _sha256_file(outdir / COMPLETION_NAME)
        _expect_failure(lambda: _write_bundle(outdir, config, None), "no overwrite")
        if _sha256_file(outdir / COMPLETION_NAME) != completion_before:
            raise AssertionError("no-overwrite failure changed the existing pack")

        race_source = root / "race_source"
        race_target = root / "race_target"
        race_source.mkdir()
        race_target.mkdir()
        (race_source / "source_marker").write_text("source", encoding="utf-8")
        (race_target / "target_marker").write_text("target", encoding="utf-8")
        _expect_failure(
            lambda: _atomic_install_directory_noreplace(race_source, race_target),
            "atomic no-replace race",
        )
        if not (race_source / "source_marker").is_file() or not (
            race_target / "target_marker"
        ).is_file():
            raise AssertionError("atomic conflict modified source or target")

    print(
        "self-test passed: synthetic-only source hashes/dimensions/RGB, canonical "
        "metadata-free pixel-preserving PNGs, exact sheets, standalone fixed-anchor "
        "rubric, empty annotation template, public-lineage exclusion, self-hashes, "
        "closed set, ancillary-metadata rejection, and atomic no-overwrite"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_outdir = (
        data_root
        / "cross_scale_evidence/dit_imagenet256"
        / "class207_visual_anchor_pack_v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=default_outdir)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--validate",
        action="store_true",
        help="Validate an existing public pack without reading any source image.",
    )
    modes.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic-only CPU self-test; never reads release sources.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.validate:
        manifest, completion = validate_output_bundle(
            args.outdir, RELEASE_CONFIG_IDENTITY_SHA256
        )
    else:
        config = load_anchor_config(args.config)
        manifest, completion = _write_bundle(
            args.outdir, config, RELEASE_CONFIG_IDENTITY_SHA256
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "anchor_config_identity_sha256": manifest[
                    "anchor_config_identity_sha256"
                ],
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": completion["manifest_file_sha256"],
                "pack_payload_sha256": manifest["pack_payload_sha256"],
                "completion_payload_sha256": completion["payload_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
