#!/usr/bin/env python3
"""Build the reviewer-only blind pack from four completed t60 validation shards.

The sampling runner's strict validator is called on every shard with process
stdout/stderr suppressed.  The builder checks one common frozen protocol,
runner, saved prefix, model/schedule, and the exact disjoint 0..31 allocation.
It exports only clean RGB pixels under the runner's existing ``vp1_*`` blind IDs, one
contact sheet, a quality-only rubric, an empty annotation template, and
self-hashed manifest/completion records.

No trace, score, trigger result, random-stream value, explicit branch-number field, source
identity hash, or candidate configuration is exported.  The output is a closed
file set installed atomically without replacement.  There is intentionally no
unseal or analysis command in this module.  The IDs are not claimed to be
cryptographic anonymization: blinding is procedural and requires giving the
reviewer only the pack, without the runner or its ID convention.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import io
import json
import os
import platform
import stat
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

sys.dont_write_bytecode = True

import torch
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

try:
    from .run_dit_t60_within_prefix_validation_pool import (
        BRANCHES_PER_SHARD,
        EXPERIMENT as SHARD_EXPERIMENT,
        PROTOCOL_COPY_NAME,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        shard_global_indices,
        validate_output_bundle as validate_shard_bundle,
    )
    from .reproduce_dit_imagenet256 import (
        IMAGE_SIZE,
        atomic_json_dump,
        inspect_png,
        load_json,
        sha256_file,
        sha256_json,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from run_dit_t60_within_prefix_validation_pool import (
        BRANCHES_PER_SHARD,
        EXPERIMENT as SHARD_EXPERIMENT,
        PROTOCOL_COPY_NAME,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        shard_global_indices,
        validate_output_bundle as validate_shard_bundle,
    )
    from reproduce_dit_imagenet256 import (
        IMAGE_SIZE,
        atomic_json_dump,
        inspect_png,
        load_json,
        sha256_file,
        sha256_json,
    )


EXPERIMENT = "dit_imagenet256_t60_within_prefix_blind_review_pack"
SCHEMA_VERSION = 1
RUBRIC_SCHEMA_VERSION = 1
TEMPLATE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
RUBRIC_NAME = "rubric.json"
TEMPLATE_NAME = "annotation_template.json"
CONTACT_NAME = "contact_sheet.png"
IMAGE_DIR_NAME = "images"

CONTACT_COLUMNS = 8
CONTACT_ROWS = 4
CONTACT_MARGIN = 12
CONTACT_GAP = 8
CONTACT_LABEL_HEIGHT = 24
CONTACT_BACKGROUND = (245, 245, 245)
CONTACT_TEXT = (15, 15, 15)

RUNNER = Path(__file__).resolve()
SHARD_RUNNER = RUNNER.with_name("run_dit_t60_within_prefix_validation_pool.py")
PROTOCOL_SOURCE = RUNNER.parent / "configs/dit_imagenet256_t60_within_prefix_validation_v1.json"
CONTACT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_ALLOWED_CHUNKS = {b"IHDR", b"IDAT", b"IEND"}

ANNOTATION_FIELDS = (
    "overall_structural_secondary",
    "primary_hind_limb_topology",
    "tail_B_short_blunt",
    "tail_D_distal_tip",
    "tail_F_feather_or_hair_flow",
    "tail_P_paddle_like",
    "tail_R_root_attachment",
    "tail_S_abrupt_filament_transition",
    "tail_T_taper_and_volume",
    "tail_confidence",
    "tail_derived_label",
    "tail_identity",
    "tail_scorable",
    "notes",
)

ANNOTATION_ROW_SCHEMA: dict[str, str] = {
    "blind_id": "required runner blind ID matching vp1_[0-9a-f]{12}",
    "notes": "optional free text about visible endpoint appearance only",
    "overall_structural_secondary": (
        "required enum: clear_structural_bad | not_clear_structural_bad | uncertain"
    ),
    "primary_hind_limb_topology": (
        "required enum: clear_failure | not_clear_failure | uncertain"
    ),
    "tail_B_short_blunt": "required integer 0 or 1 when tail_scorable=yes; otherwise null",
    "tail_D_distal_tip": "required integer 0, 1, or 2 when tail_scorable=yes; otherwise null",
    "tail_F_feather_or_hair_flow": (
        "required integer 0, 1, or 2 when tail_scorable=yes; otherwise null"
    ),
    "tail_P_paddle_like": "required integer 0 or 1 when tail_scorable=yes; otherwise null",
    "tail_R_root_attachment": (
        "required integer 0, 1, or 2 when tail_scorable=yes; otherwise null"
    ),
    "tail_S_abrupt_filament_transition": (
        "required integer 0 or 1 when tail_scorable=yes; otherwise null"
    ),
    "tail_T_taper_and_volume": (
        "required integer 0, 1, or 2 when tail_scorable=yes; otherwise null"
    ),
    "tail_confidence": "required enum: high | medium | low",
    "tail_derived_label": "required enum: natural | odd | malformed | uncertain",
    "tail_identity": "required enum: clear | plausible | unclear",
    "tail_scorable": "required enum: yes | no",
}

DECLARATION_STATEMENT = (
    "I completed all visible-quality labels without seeing generation-path evidence, "
    "trigger outcomes, or rankings."
)

PUBLIC_FORBIDDEN_FRAGMENTS = (
    "global_index",
    "branch_index",
    "stream_seed",
    "branch_stream",
    "delta_nu",
    "tile_12",
    "tile_index",
    "alpha_e",
    "alarm_log_e",
    "primary_log_e",
    "likelihood_ratio",
    "sampling_runner_sha256",
    "observer_identity_sha256",
    "protocol_identity_sha256",
    "target_x60_raw_sha256",
    "source_shard",
)


@dataclass(frozen=True)
class BlindImage:
    blind_id: str
    source_path: Path
    pixel_sha256: str


@dataclass(frozen=True)
class ValidatedInputs:
    blind_images: tuple[BlindImage, ...]
    protocol: dict[str, Any]


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return sha256_json(stripped)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get(key) != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def _cpu_dependencies() -> dict[str, str | None]:
    def package_version(name: str) -> str | None:
        try:
            return distribution_version(name)
        except PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pillow": package_version("pillow"),
    }


def _validate_blind_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 16 or not value.startswith("vp1_"):
        raise RuntimeError("invalid runner blind ID")
    if any(character not in "0123456789abcdef" for character in value[4:]):
        raise RuntimeError("invalid runner blind ID")
    return value


def _blind_order_key(identifier: str) -> tuple[str, str]:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest(), identifier


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _reject_special_entries(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("artifact root must be a plain directory")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError("artifact tree contains a link or special entry")


def _plain_file_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("expected a plain file")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _image_record(path: Path, root: Path, size: tuple[int, int]) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        **inspect_png(path, "RGB", size),
    }


@contextlib.contextmanager
def _silence_process_output() -> Iterator[None]:
    """Suppress Python and file-descriptor stdout/stderr for private validation."""

    # Flush caller output before fd redirection.  More importantly, flush all
    # libc stdio buffers while fd 1/2 still point at /dev/null: otherwise a
    # validator's unflushed printf can escape after restoration or at exit.
    sys.stdout.flush()
    sys.stderr.flush()
    libc = ctypes.CDLL(None)
    fflush = getattr(libc, "fflush", None)
    if fflush is None:
        raise RuntimeError("cannot guarantee private-validator output suppression")
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with open(os.devnull, "w", encoding="utf-8") as null:
            os.dup2(null.fileno(), 1)
            os.dup2(null.fileno(), 2)
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
                captured_stderr
            ):
                yield
    finally:
        flush_result = fflush(None)
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        if flush_result != 0:
            raise RuntimeError("could not flush suppressed libc output")


def _strict_validate_silently(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with _silence_process_output():
            return validate_shard_bundle(root)
    except Exception as exc:
        del exc
        raise RuntimeError("an input shard failed strict validation") from None


def _extract_quality_endpoints(protocol: dict[str, Any]) -> dict[str, Any]:
    blind_review = protocol.get("blind_review")
    expected_source_keys = {
        "reviewer_input_boundary",
        "annotation_lock",
        "annotation_row_schema",
        "image_context",
        "overall_structural_secondary",
        "primary_endpoint",
        "tail_naturalness_separate",
    }
    if not isinstance(blind_review, dict) or set(blind_review) != expected_source_keys:
        raise RuntimeError("frozen blind-review schema changed")
    source_schema = blind_review.get("annotation_row_schema")
    if not isinstance(source_schema, dict) or set(source_schema) != {
        "blind_id",
        *ANNOTATION_FIELDS,
    }:
        raise RuntimeError("frozen annotation-row schema changed")

    image_context = blind_review.get("image_context")
    overall = blind_review.get("overall_structural_secondary")
    primary = blind_review.get("primary_endpoint")
    tail = blind_review.get("tail_naturalness_separate")
    if not isinstance(image_context, str):
        raise RuntimeError("quality image-context schema changed")
    if not isinstance(overall, dict) or set(overall) != {
        "labels",
        "positive_definition",
        "role",
    }:
        raise RuntimeError("overall quality endpoint schema changed")
    if overall.get("labels") != [
        "clear_structural_bad",
        "not_clear_structural_bad",
        "uncertain",
    ]:
        raise RuntimeError("overall quality endpoint labels changed")
    if not all(isinstance(overall.get(key), str) for key in ("positive_definition", "role")):
        raise RuntimeError("overall quality endpoint text changed type")

    primary_keys = {
        "labels",
        "name",
        "negative_definition",
        "positive_examples",
        "role",
        "tail_exclusion",
    }
    if not isinstance(primary, dict) or set(primary) != primary_keys:
        raise RuntimeError("primary quality endpoint schema changed")
    if primary.get("labels") != ["clear_failure", "not_clear_failure", "uncertain"]:
        raise RuntimeError("primary quality endpoint labels changed")
    if primary.get("name") != "clear_lower_left_hind_limb_topology_failure_under_shared_pose":
        raise RuntimeError("primary quality endpoint name changed")
    if not isinstance(primary.get("positive_examples"), list) or not all(
        isinstance(value, str) for value in primary["positive_examples"]
    ):
        raise RuntimeError("primary quality examples changed type")
    if not all(
        isinstance(primary.get(key), str)
        for key in ("negative_definition", "role", "tail_exclusion")
    ):
        raise RuntimeError("primary quality endpoint text changed type")

    tail_keys = {
        "confidence",
        "defect_dimensions",
        "derived_categorical_label",
        "identity",
        "scorable",
        "shape_flags_binary",
        "role",
    }
    if not isinstance(tail, dict) or set(tail) != tail_keys:
        raise RuntimeError("tail quality endpoint schema changed")
    if tail.get("confidence") != ["high", "medium", "low"]:
        raise RuntimeError("tail confidence labels changed")
    if tail.get("identity") != ["clear", "plausible", "unclear"]:
        raise RuntimeError("tail identity labels changed")
    nested = {
        "defect_dimensions": {
            "distal_tip",
            "feather_or_hair_flow",
            "root_attachment",
            "taper_and_volume",
        },
        "derived_categorical_label": {
            "malformed",
            "natural",
            "odd",
            "plausible_all_zero_resolution",
            "uncertain",
        },
        "scorable": {"allowed", "rule"},
        "shape_flags_binary": {
            "B_short_blunt",
            "P_paddle_like",
            "S_abrupt_filament_transition",
        },
    }
    for key, expected_keys in nested.items():
        value = tail.get(key)
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise RuntimeError("tail quality endpoint nested schema changed")
    if tail["scorable"].get("allowed") != ["yes", "no"]:
        raise RuntimeError("tail scorable labels changed")
    if not isinstance(tail["scorable"].get("rule"), str) or not isinstance(
        tail.get("role"), str
    ):
        raise RuntimeError("tail quality endpoint text changed type")

    endpoints = json.loads(
        json.dumps(
            {
                "image_context": image_context,
                "overall_structural_secondary": overall,
                "primary_endpoint": primary,
                "tail_naturalness_separate": tail,
            },
            ensure_ascii=False,
        )
    )
    serialized = json.dumps(endpoints, ensure_ascii=False, sort_keys=True).lower()
    if any(fragment in serialized for fragment in PUBLIC_FORBIDDEN_FRAGMENTS):
        raise RuntimeError("quality endpoints contain private configuration")
    return endpoints


def _load_frozen_protocol(root: Path) -> dict[str, Any]:
    protocol = _read_self_hashed_json(root / PROTOCOL_COPY_NAME, "protocol_identity_sha256")
    if protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION":
        raise RuntimeError("input shard protocol is not frozen")
    _extract_quality_endpoints(protocol)
    return protocol


def validate_input_shards(shard_roots: tuple[Path, ...]) -> ValidatedInputs:
    if len(shard_roots) != TOTAL_SHARDS or any(root.is_symlink() for root in shard_roots):
        raise RuntimeError("exactly four plain shard directories are required")
    # Resolve parent-directory aliases once so path containment and relative
    # records use the same namespace even when (for example) /home/data is a
    # mount alias for /data/users.
    shard_roots = tuple(root.resolve() for root in shard_roots)
    if len(set(shard_roots)) != TOTAL_SHARDS:
        raise RuntimeError("exactly four distinct shard directories are required")
    manifests: list[dict[str, Any]] = []
    results_list: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    for root in shard_roots:
        _reject_special_entries(root)
        manifest, results = _strict_validate_silently(root)
        if manifest.get("experiment") != SHARD_EXPERIMENT:
            raise RuntimeError("input is not a t60 validation shard")
        manifests.append(manifest)
        results_list.append(results)
        protocols.append(_load_frozen_protocol(root))

    def require_common(values: list[Any], label: str) -> Any:
        canonical = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}
        if len(canonical) != 1:
            raise RuntimeError(f"input shards disagree on {label}")
        return values[0]

    runner_hash = require_common(
        [manifest.get("runner", {}).get("sha256") for manifest in manifests], "runner"
    )
    protocol_identity = require_common(
        [
            manifest.get("protocol", {}).get("protocol_identity_sha256")
            for manifest in manifests
        ],
        "protocol",
    )
    protocol_file_hash = require_common(
        [manifest.get("protocol", {}).get("source_file_sha256") for manifest in manifests],
        "protocol file",
    )
    observer_identity = require_common(
        [
            manifest.get("input_prefix", {}).get("observer_manifest_identity_sha256")
            for manifest in manifests
        ],
        "saved-prefix observer",
    )
    x60_identity = require_common(
        [
            manifest.get("input_prefix", {}).get("target_x60_raw_sha256")
            for manifest in manifests
        ],
        "saved x60",
    )
    require_common([manifest.get("schedule") for manifest in manifests], "schedule")
    require_common([manifest.get("sources", {}).get("dit") for manifest in manifests], "DiT")
    require_common(
        [manifest.get("sources", {}).get("checkpoint") for manifest in manifests],
        "checkpoint",
    )
    require_common([manifest.get("sources", {}).get("vae") for manifest in manifests], "VAE")
    require_common(protocols, "frozen protocol content")
    require_common(
        [
            {
                key: manifest.get("pool", {}).get(key)
                for key in (
                    "prefix_seed",
                    "target_batch_index",
                    "target_class_id",
                    "rollback_internal_timestep",
                    "pool_seed",
                    "total_shards",
                    "branches_per_shard",
                    "total_pool_branches",
                )
            }
            for manifest in manifests
        ],
        "pool constants",
    )
    shared_hashes = (
        runner_hash,
        protocol_identity,
        protocol_file_hash,
        observer_identity,
        x60_identity,
    )
    if not all(_is_sha256(value) for value in shared_hashes):
        raise RuntimeError("a shared input identity is malformed")
    if runner_hash != sha256_file(SHARD_RUNNER):
        raise RuntimeError("input shards use a different sampling runner")
    for protocol in protocols:
        if protocol.get("protocol_identity_sha256") != protocol_identity:
            raise RuntimeError("protocol copy is not bound to its shard manifest")

    shard_indices = [manifest.get("pool", {}).get("this_shard_index") for manifest in manifests]
    if any(type(value) is not int for value in shard_indices):
        raise RuntimeError("shard index is not an exact integer")
    if sorted(shard_indices) != list(range(TOTAL_SHARDS)):
        raise RuntimeError("shards are not the complete 0..3 set")

    internal_indices: list[int] = []
    blind_images: list[BlindImage] = []
    for root, manifest, results in zip(shard_roots, manifests, results_list):
        shard_index = manifest["pool"]["this_shard_index"]
        indices = manifest["pool"].get("this_shard_global_branch_indices")
        if (
            not isinstance(indices, list)
            or any(type(value) is not int for value in indices)
            or indices != list(shard_global_indices(shard_index))
        ):
            raise RuntimeError("a shard has an invalid preassigned allocation")
        internal_indices.extend(indices)
        branch_records = results.get("branch_records")
        if not isinstance(branch_records, list) or len(branch_records) != BRANCHES_PER_SHARD:
            raise RuntimeError("a shard has an invalid endpoint-image set")
        for branch_record in branch_records:
            if not isinstance(branch_record, dict):
                raise RuntimeError("endpoint-image record is malformed")
            blind_id = _validate_blind_id(branch_record.get("blind_id"))
            image_record = branch_record.get("image")
            if not isinstance(image_record, dict):
                raise RuntimeError("endpoint-image record is malformed")
            relative_path = image_record.get("relative_path")
            if not isinstance(relative_path, str):
                raise RuntimeError("endpoint-image path is malformed")
            source_path = (root / relative_path).resolve()
            if root.resolve() not in source_path.parents:
                raise RuntimeError("endpoint image escapes its validated shard")
            if _image_record(source_path, root, (IMAGE_SIZE, IMAGE_SIZE)) != image_record:
                raise RuntimeError("endpoint image changed after strict validation")
            blind_images.append(
                BlindImage(
                    blind_id=blind_id,
                    source_path=source_path,
                    pixel_sha256=str(image_record["pixel_sha256"]),
                )
            )
    if sorted(internal_indices) != list(range(TOTAL_POOL_BRANCHES)):
        raise RuntimeError("shard allocations do not cover exactly 0..31")
    if len(blind_images) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("validated pool does not contain exactly 32 endpoint images")
    if len({image.blind_id for image in blind_images}) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("opaque blind IDs are not unique")

    ordered = tuple(sorted(blind_images, key=lambda image: _blind_order_key(image.blind_id)))
    del manifests, results_list, shard_indices, internal_indices
    return ValidatedInputs(blind_images=ordered, protocol=protocols[0])


def build_rubric(protocol: dict[str, Any]) -> dict[str, Any]:
    endpoints = _extract_quality_endpoints(protocol)
    payload: dict[str, Any] = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_name": "t60_within_prefix_endpoint_quality_review_v1",
        "purpose": "Visible endpoint quality review for one fixed 32-image closed set.",
        "annotation_row_schema": ANNOTATION_ROW_SCHEMA,
        **endpoints,
    }
    payload["rubric_identity_sha256"] = _canonical_self_hash(
        payload, "rubric_identity_sha256"
    )
    return payload


def build_annotation_template(
    blind_ids: tuple[str, ...], rubric_identity_sha256: str
) -> dict[str, Any]:
    if len(blind_ids) != TOTAL_POOL_BRANCHES or len(set(blind_ids)) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("annotation template requires 32 distinct IDs")
    rows: list[dict[str, Any]] = []
    for identifier in blind_ids:
        _validate_blind_id(identifier)
        row: dict[str, Any] = {"blind_id": identifier}
        row.update({field: None for field in ANNOTATION_FIELDS})
        rows.append(row)
    payload: dict[str, Any] = {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "template_name": "t60_within_prefix_blind_annotations_v1",
        "rubric_identity_sha256": rubric_identity_sha256,
        "instructions": (
            "Copy this file outside the immutable pack, fill every null field, then save "
            "and externally record the completed file SHA-256 before viewing private results."
        ),
        "unseen_information_declaration": {
            "statement": DECLARATION_STATEMENT,
            "reviewer_name": None,
            "review_started_utc": None,
            "review_completed_utc": None,
            "private_information_seen_before_annotation_lock": None,
            "signed_confirmation": None,
        },
        "rows": rows,
    }
    payload["template_identity_sha256"] = _canonical_self_hash(
        payload, "template_identity_sha256"
    )
    return payload


def _validate_metadata_free_png(path: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError("output is not a PNG")
    position = len(PNG_SIGNATURE)
    chunks: list[bytes] = []
    while position < len(data):
        if position + 12 > len(data):
            raise RuntimeError("truncated PNG chunk")
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(data):
            raise RuntimeError("truncated PNG payload")
        chunk_data = data[position + 8 : position + 8 + length]
        observed_crc = struct.unpack(">I", data[position + 8 + length : end])[0]
        expected_crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise RuntimeError("PNG chunk CRC changed")
        if chunk_type not in PNG_ALLOWED_CHUNKS:
            raise RuntimeError("PNG contains metadata or an unknown chunk")
        if chunk_type == b"IHDR" and length != 13:
            raise RuntimeError("PNG IHDR length changed")
        if chunk_type == b"IEND" and length != 0:
            raise RuntimeError("PNG IEND is not empty")
        chunks.append(chunk_type)
        position = end
        if chunk_type == b"IEND":
            break
    if position != len(data):
        raise RuntimeError("PNG contains trailing bytes")
    if not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND":
        raise RuntimeError("PNG critical chunk order changed")
    if chunks.count(b"IHDR") != 1 or chunks.count(b"IEND") != 1 or b"IDAT" not in chunks:
        raise RuntimeError("PNG critical chunk set changed")
    first_idat = chunks.index(b"IDAT")
    if any(chunk != b"IDAT" for chunk in chunks[first_idat:-1]):
        raise RuntimeError("PNG IDAT chunks are not contiguous")
    # An otherwise valid IDAT may hide bytes after the DEFLATE end marker.
    # Rebuild the decoded RGB pixels with the sole allowed encoder and require
    # byte equality, fixing compression, chunking, and all ancillary content.
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise RuntimeError("PNG mode is not canonical RGB")
        canonical_image = Image.frombytes("RGB", image.size, image.tobytes())
    canonical_buffer = io.BytesIO()
    canonical_image.save(canonical_buffer, format="PNG", optimize=False)
    if data != canonical_buffer.getvalue():
        raise RuntimeError("PNG bytes are not the fixed canonical RGB encoding")


def _clean_reencode(source: Path, destination: Path) -> dict[str, Any]:
    if os.path.lexists(destination):
        raise RuntimeError("refusing to overwrite an output image")
    with Image.open(source) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError("source image violates the RGB/256 contract")
        clean = Image.frombytes("RGB", image.size, image.tobytes())
    clean.save(destination, format="PNG", optimize=False)
    _validate_metadata_free_png(destination)
    return inspect_png(destination, "RGB", (IMAGE_SIZE, IMAGE_SIZE))


def _font() -> ImageFont.FreeTypeFont:
    if not CONTACT_FONT.is_file() or CONTACT_FONT.is_symlink():
        raise RuntimeError("required deterministic contact-sheet font is unavailable")
    return ImageFont.truetype(str(CONTACT_FONT), 14)


def _contact_size() -> tuple[int, int]:
    return (
        2 * CONTACT_MARGIN
        + CONTACT_COLUMNS * IMAGE_SIZE
        + (CONTACT_COLUMNS - 1) * CONTACT_GAP,
        2 * CONTACT_MARGIN
        + CONTACT_ROWS * (IMAGE_SIZE + CONTACT_LABEL_HEIGHT)
        + (CONTACT_ROWS - 1) * CONTACT_GAP,
    )


def render_contact_sheet(images: tuple[tuple[str, Path], ...]) -> Image.Image:
    if len(images) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("contact sheet requires exactly 32 images")
    canvas = Image.new("RGB", _contact_size(), CONTACT_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = _font()
    cell_height = IMAGE_SIZE + CONTACT_LABEL_HEIGHT
    for position, (identifier, path) in enumerate(images):
        _validate_blind_id(identifier)
        row, column = divmod(position, CONTACT_COLUMNS)
        left = CONTACT_MARGIN + column * (IMAGE_SIZE + CONTACT_GAP)
        top = CONTACT_MARGIN + row * (cell_height + CONTACT_GAP)
        with Image.open(path) as source:
            source.load()
            if source.mode != "RGB" or source.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError("contact-sheet source violates RGB/256 contract")
            canvas.paste(source, (left, top))
        bounds = draw.textbbox((0, 0), identifier, font=font)
        label_width = bounds[2] - bounds[0]
        draw.text(
            (left + (IMAGE_SIZE - label_width) // 2, top + IMAGE_SIZE + 3),
            identifier,
            fill=CONTACT_TEXT,
            font=font,
        )
    return canvas


def _atomic_install_directory_noreplace(source: Path, target: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
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
        raise FileExistsError(error_number, "refusing to replace existing blind pack", target)
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("filesystem/kernel lacks atomic RENAME_NOREPLACE")
    raise OSError(error_number, os.strerror(error_number), target)


def _write_bundle(args: argparse.Namespace, validated: ValidatedInputs) -> None:
    if os.path.lexists(args.outdir):
        raise RuntimeError("refusing to overwrite existing blind pack")
    images = validated.blind_images
    if images != tuple(sorted(images, key=lambda image: _blind_order_key(image.blind_id))):
        raise RuntimeError("validated images are not in frozen blind order")
    rubric = build_rubric(validated.protocol)
    template = build_annotation_template(
        tuple(image.blind_id for image in images), rubric["rubric_identity_sha256"]
    )

    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.outdir.name}.staging-", dir=args.outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        image_dir = staging / IMAGE_DIR_NAME
        image_dir.mkdir()
        atomic_json_dump(rubric, staging / RUBRIC_NAME)
        atomic_json_dump(template, staging / TEMPLATE_NAME)

        image_records: list[dict[str, Any]] = []
        contact_inputs: list[tuple[str, Path]] = []
        for position, item in enumerate(images):
            destination = image_dir / f"{item.blind_id}.png"
            inspection = _clean_reencode(item.source_path, destination)
            if inspection["pixel_sha256"] != item.pixel_sha256:
                raise RuntimeError("clean PNG re-encoding changed source pixels")
            image_records.append(
                {
                    "review_position": position,
                    "blind_id": item.blind_id,
                    "image": {
                        "relative_path": destination.relative_to(staging).as_posix(),
                        **inspection,
                    },
                }
            )
            contact_inputs.append((item.blind_id, destination))

        contact_path = staging / CONTACT_NAME
        render_contact_sheet(tuple(contact_inputs)).save(
            contact_path, format="PNG", optimize=False
        )
        _validate_metadata_free_png(contact_path)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "role": "BLIND_REVIEW_DELIVERY_ONLY",
            "image_count": TOTAL_POOL_BRANCHES,
            "input_shard_count": TOTAL_SHARDS,
            "input_validation": {
                "strict_validator_called_with_all_output_suppressed": True,
                "common_frozen_protocol_verified": True,
                "common_runner_model_schedule_verified": True,
                "common_saved_prefix_verified": True,
                "exact_disjoint_allocation_verified": True,
            },
            "blind_order": {
                "rule": "ascending SHA-256 of the UTF-8 runner blind ID",
                "ordered_blind_ids": [image.blind_id for image in images],
                "depends_on_private_result": False,
            },
            "images": image_records,
            "contact_sheet": _image_record(contact_path, staging, _contact_size()),
            "rubric": _plain_file_record(staging / RUBRIC_NAME, staging),
            "annotation_template": _plain_file_record(staging / TEMPLATE_NAME, staging),
            "contact_renderer": {
                "columns": CONTACT_COLUMNS,
                "rows": CONTACT_ROWS,
                "font_name": CONTACT_FONT.name,
                "font_file_sha256": sha256_file(CONTACT_FONT),
            },
            "builder": {"filename": RUNNER.name, "sha256": sha256_file(RUNNER)},
            "dependencies": _cpu_dependencies(),
        }
        manifest["identity_sha256"] = _canonical_self_hash(manifest, "identity_sha256")
        atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / MANIFEST_NAME),
            "image_count": TOTAL_POOL_BRANCHES,
            "file_count": TOTAL_POOL_BRANCHES + 5,
        }
        completion["payload_sha256"] = _canonical_self_hash(
            completion, "payload_sha256"
        )
        atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_output_bundle(staging)
        _atomic_install_directory_noreplace(staging, args.outdir)
    validate_output_bundle(args.outdir)


def _validate_rubric(rubric: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "rubric_name",
        "purpose",
        "annotation_row_schema",
        "image_context",
        "overall_structural_secondary",
        "primary_endpoint",
        "tail_naturalness_separate",
        "rubric_identity_sha256",
    }
    if set(rubric) != expected_keys:
        raise RuntimeError("rubric schema changed")
    if rubric.get("schema_version") != RUBRIC_SCHEMA_VERSION or rubric.get(
        "rubric_name"
    ) != "t60_within_prefix_endpoint_quality_review_v1":
        raise RuntimeError("rubric identity changed")
    if rubric.get("annotation_row_schema") != ANNOTATION_ROW_SCHEMA:
        raise RuntimeError("rubric annotation schema changed")
    synthetic_protocol = {
        "blind_review": {
            "reviewer_input_boundary": {},
            "annotation_lock": {},
            "annotation_row_schema": ANNOTATION_ROW_SCHEMA,
            "image_context": rubric["image_context"],
            "overall_structural_secondary": rubric["overall_structural_secondary"],
            "primary_endpoint": rubric["primary_endpoint"],
            "tail_naturalness_separate": rubric["tail_naturalness_separate"],
        }
    }
    _extract_quality_endpoints(synthetic_protocol)
    if rubric.get("rubric_identity_sha256") != _canonical_self_hash(
        rubric, "rubric_identity_sha256"
    ):
        raise RuntimeError("rubric self-hash changed")
    local_protocol = _read_self_hashed_json(
        PROTOCOL_SOURCE, "protocol_identity_sha256"
    )
    if local_protocol.get("protocol_status") != "FROZEN_BEFORE_GPU_EXECUTION":
        raise RuntimeError("local protocol is not frozen")
    if rubric != build_rubric(local_protocol):
        raise RuntimeError("rubric differs from the exact frozen quality rubric")


def _validate_template(
    template: dict[str, Any], blind_ids: tuple[str, ...], rubric: dict[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "template_name",
        "rubric_identity_sha256",
        "instructions",
        "unseen_information_declaration",
        "rows",
        "template_identity_sha256",
    }
    if set(template) != expected_keys:
        raise RuntimeError("annotation template schema changed")
    if template.get("schema_version") != TEMPLATE_SCHEMA_VERSION or template.get(
        "template_name"
    ) != "t60_within_prefix_blind_annotations_v1":
        raise RuntimeError("annotation template identity changed")
    if template.get("rubric_identity_sha256") != rubric["rubric_identity_sha256"]:
        raise RuntimeError("annotation template is not bound to its rubric")
    if template.get("template_identity_sha256") != _canonical_self_hash(
        template, "template_identity_sha256"
    ):
        raise RuntimeError("annotation template self-hash changed")
    declaration = template.get("unseen_information_declaration")
    declaration_keys = {
        "statement",
        "reviewer_name",
        "review_started_utc",
        "review_completed_utc",
        "private_information_seen_before_annotation_lock",
        "signed_confirmation",
    }
    if not isinstance(declaration, dict) or set(declaration) != declaration_keys:
        raise RuntimeError("annotation declaration schema changed")
    if declaration.get("statement") != DECLARATION_STATEMENT:
        raise RuntimeError("annotation declaration statement changed")
    if any(declaration[key] is not None for key in declaration_keys - {"statement"}):
        raise RuntimeError("annotation declaration is prefilled")
    rows = template.get("rows")
    if not isinstance(rows, list) or len(rows) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("annotation row count changed")
    for identifier, row in zip(blind_ids, rows):
        if not isinstance(row, dict) or set(row) != {"blind_id", *ANNOTATION_FIELDS}:
            raise RuntimeError("annotation row schema changed")
        if row["blind_id"] != identifier:
            raise RuntimeError("annotation rows are not in blind order")
        if any(row[field] is not None for field in ANNOTATION_FIELDS):
            raise RuntimeError("annotation row is prefilled")
    if template != build_annotation_template(
        blind_ids, rubric["rubric_identity_sha256"]
    ):
        raise RuntimeError("annotation template differs from the exact empty template")


def validate_output_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _reject_special_entries(root)
    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "identity_sha256")
    expected_manifest_keys = {
        "schema_version",
        "experiment",
        "role",
        "image_count",
        "input_shard_count",
        "input_validation",
        "blind_order",
        "images",
        "contact_sheet",
        "rubric",
        "annotation_template",
        "contact_renderer",
        "builder",
        "dependencies",
        "identity_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise RuntimeError("manifest schema changed")
    fixed_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "BLIND_REVIEW_DELIVERY_ONLY",
        "image_count": TOTAL_POOL_BRANCHES,
        "input_shard_count": TOTAL_SHARDS,
    }
    if any(manifest.get(key) != value for key, value in fixed_manifest.items()):
        raise RuntimeError("manifest identity changed")
    if manifest.get("input_validation") != {
        "strict_validator_called_with_all_output_suppressed": True,
        "common_frozen_protocol_verified": True,
        "common_runner_model_schedule_verified": True,
        "common_saved_prefix_verified": True,
        "exact_disjoint_allocation_verified": True,
    }:
        raise RuntimeError("input-validation declaration changed")
    if manifest.get("builder") != {"filename": RUNNER.name, "sha256": sha256_file(RUNNER)}:
        raise RuntimeError("builder identity changed")
    if manifest.get("dependencies") != _cpu_dependencies():
        raise RuntimeError("dependency identity changed")
    if manifest.get("contact_renderer") != {
        "columns": CONTACT_COLUMNS,
        "rows": CONTACT_ROWS,
        "font_name": CONTACT_FONT.name,
        "font_file_sha256": sha256_file(CONTACT_FONT),
    }:
        raise RuntimeError("contact renderer identity changed")
    blind_order = manifest.get("blind_order")
    if not isinstance(blind_order, dict) or set(blind_order) != {
        "rule",
        "ordered_blind_ids",
        "depends_on_private_result",
    }:
        raise RuntimeError("blind-order schema changed")
    if blind_order.get("rule") != "ascending SHA-256 of the UTF-8 runner blind ID":
        raise RuntimeError("blind-order rule changed")
    if blind_order.get("depends_on_private_result") is not False:
        raise RuntimeError("blind order depends on a private result")
    blind_ids_value = blind_order.get("ordered_blind_ids")
    if not isinstance(blind_ids_value, list) or len(blind_ids_value) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("blind-ID count changed")
    blind_ids = tuple(_validate_blind_id(value) for value in blind_ids_value)
    if len(set(blind_ids)) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("blind IDs are not unique")
    if blind_ids != tuple(sorted(blind_ids, key=_blind_order_key)):
        raise RuntimeError("blind IDs are not in frozen hash order")

    expected_files = {
        (root / MANIFEST_NAME).resolve(),
        (root / COMPLETION_NAME).resolve(),
        (root / RUBRIC_NAME).resolve(),
        (root / TEMPLATE_NAME).resolve(),
        (root / CONTACT_NAME).resolve(),
        *{
            (root / IMAGE_DIR_NAME / f"{identifier}.png").resolve()
            for identifier in blind_ids
        },
    }
    actual_files = {
        path.resolve() for path in root.rglob("*") if stat.S_ISREG(path.lstat().st_mode)
    }
    actual_directories = {
        path.resolve() for path in root.rglob("*") if stat.S_ISDIR(path.lstat().st_mode)
    }
    if actual_files != expected_files or actual_directories != {
        (root / IMAGE_DIR_NAME).resolve()
    }:
        raise RuntimeError("blind pack is not a closed file/directory set")

    rubric = _read_self_hashed_json(root / RUBRIC_NAME, "rubric_identity_sha256")
    _validate_rubric(rubric)
    template = _read_self_hashed_json(root / TEMPLATE_NAME, "template_identity_sha256")
    _validate_template(template, blind_ids, rubric)

    image_records = manifest.get("images")
    if not isinstance(image_records, list) or len(image_records) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("manifest image-record count changed")
    contact_inputs: list[tuple[str, Path]] = []
    for position, (identifier, record) in enumerate(zip(blind_ids, image_records)):
        path = root / IMAGE_DIR_NAME / f"{identifier}.png"
        _validate_metadata_free_png(path)
        expected_record = {
            "review_position": position,
            "blind_id": identifier,
            "image": _image_record(path, root, (IMAGE_SIZE, IMAGE_SIZE)),
        }
        if record != expected_record:
            raise RuntimeError("manifest native-image record changed")
        contact_inputs.append((identifier, path))
    contact_path = root / CONTACT_NAME
    _validate_metadata_free_png(contact_path)
    if manifest.get("contact_sheet") != _image_record(
        contact_path, root, _contact_size()
    ):
        raise RuntimeError("manifest contact-sheet record changed")
    expected_contact = render_contact_sheet(tuple(contact_inputs))
    with Image.open(contact_path) as observed_contact:
        observed_contact.load()
        if observed_contact.mode != "RGB" or observed_contact.size != _contact_size():
            raise RuntimeError("contact-sheet geometry changed")
        if observed_contact.tobytes() != expected_contact.tobytes():
            raise RuntimeError("contact sheet is not the exact ordered native-image rendering")
    if manifest.get("rubric") != _plain_file_record(root / RUBRIC_NAME, root):
        raise RuntimeError("rubric file record changed")
    if manifest.get("annotation_template") != _plain_file_record(root / TEMPLATE_NAME, root):
        raise RuntimeError("annotation-template file record changed")

    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    expected_completion_keys = {
        "complete",
        "manifest_identity_sha256",
        "manifest_file_sha256",
        "image_count",
        "file_count",
        "payload_sha256",
    }
    if set(completion) != expected_completion_keys:
        raise RuntimeError("completion schema changed")
    if completion != {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": sha256_file(root / MANIFEST_NAME),
        "image_count": TOTAL_POOL_BRANCHES,
        "file_count": TOTAL_POOL_BRANCHES + 5,
        "payload_sha256": completion["payload_sha256"],
    }:
        raise RuntimeError("completion binding changed")

    public_text = json.dumps(
        {
            "manifest": manifest,
            "completion": completion,
            "rubric": rubric,
            "template": template,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).lower()
    if any(fragment in public_text for fragment in PUBLIC_FORBIDDEN_FRAGMENTS):
        raise RuntimeError("public JSON contains a forbidden private field/configuration")
    return manifest, completion


def run_real(args: argparse.Namespace) -> None:
    validated = validate_input_shards(args.shard_dir)
    _write_bundle(args, validated)
    print(
        json.dumps(
            {"status": "complete", "blind_image_count": TOTAL_POOL_BRANCHES},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _toy_image(path: Path, position: int) -> None:
    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        ((position * 37) % 256, (position * 71) % 256, (position * 109) % 256),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 96, 96), outline=(255, 255, 255), width=3)
    draw.text((22, 42), f"{position:02d}", fill=(255, 255, 255), font=_font())
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("selftest_source_metadata", f"source-{position}")
    image.save(path, format="PNG", optimize=False, pnginfo=metadata)


def _make_mock_shards(root: Path, protocol: dict[str, Any]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for shard_index in range(TOTAL_SHARDS):
        shard_root = root / f"shard_{shard_index}"
        image_root = shard_root / "images"
        image_root.mkdir(parents=True)
        atomic_json_dump(protocol, shard_root / PROTOCOL_COPY_NAME)
        records: list[dict[str, Any]] = []
        for internal_index in shard_global_indices(shard_index):
            digest = hashlib.sha256(f"mock-blind-{internal_index}".encode()).hexdigest()
            identifier = f"vp1_{digest[:12]}"
            path = image_root / f"{identifier}.png"
            _toy_image(path, internal_index)
            records.append(
                {
                    "blind_id": identifier,
                    "image": {
                        "relative_path": path.relative_to(shard_root).as_posix(),
                        **inspect_png(path, "RGB", (IMAGE_SIZE, IMAGE_SIZE)),
                    },
                }
            )
        manifest: dict[str, Any] = {
            "experiment": SHARD_EXPERIMENT,
            "identity_sha256": hashlib.sha256(f"mock-{shard_index}".encode()).hexdigest(),
            "runner": {"sha256": sha256_file(SHARD_RUNNER)},
            "protocol": {
                "protocol_identity_sha256": protocol["protocol_identity_sha256"],
                "source_file_sha256": sha256_file(shard_root / PROTOCOL_COPY_NAME),
            },
            "input_prefix": {
                "observer_manifest_identity_sha256": "c" * 64,
                "target_x60_raw_sha256": "d" * 64,
            },
            "sources": {
                "dit": {"mock": "same"},
                "checkpoint": {"mock": "same"},
                "vae": {"mock": "same"},
            },
            "schedule": {"mock": "same"},
            "pool": {
                "prefix_seed": 2,
                "target_batch_index": 0,
                "target_class_id": 207,
                "rollback_internal_timestep": 60,
                "pool_seed": 20260827,
                "total_shards": TOTAL_SHARDS,
                "branches_per_shard": BRANCHES_PER_SHARD,
                "total_pool_branches": TOTAL_POOL_BRANCHES,
                "this_shard_index": shard_index,
                "this_shard_global_branch_indices": list(shard_global_indices(shard_index)),
            },
        }
        atomic_json_dump(manifest, shard_root / MANIFEST_NAME)
        atomic_json_dump({"branch_records": records}, shard_root / "results.json")
        roots.append(shard_root)
    return tuple(roots)


def _expect_failure(operation: Callable[[], Any], label: str) -> None:
    try:
        operation()
    except (RuntimeError, FileExistsError):
        return
    raise AssertionError(f"negative self-test did not fail: {label}")


def run_self_test() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must begin before CUDA initialization")
    libc_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ctypes\n"
                "from experiments.build_dit_t60_within_prefix_blind_pack import _silence_process_output\n"
                "with _silence_process_output():\n"
                "    ctypes.CDLL(None).printf(b'PRIVATE_LIBC_BUFFERED_SENTINEL')\n"
                "print('VISIBLE_AFTER_SUPPRESSION')\n"
            ),
        ],
        cwd=RUNNER.parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if (
        libc_probe.returncode != 0
        or libc_probe.stdout != b"VISIBLE_AFTER_SUPPRESSION\n"
        or libc_probe.stderr
    ):
        raise AssertionError("buffered libc output-suppression probe failed")
    protocol = json.loads(json.dumps(load_json(PROTOCOL_SOURCE)))
    protocol["protocol_status"] = "FROZEN_BEFORE_GPU_EXECUTION"
    protocol["protocol_identity_sha256"] = _canonical_self_hash(
        protocol, "protocol_identity_sha256"
    )
    with tempfile.TemporaryDirectory(prefix="t60-blind-pack-self-test-") as temporary:
        root = Path(temporary)
        shard_roots = _make_mock_shards(root, protocol)
        original_validator = globals()["validate_shard_bundle"]

        def fake_validator(shard_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            print("PRIVATE_VALIDATOR_SENTINEL")
            sys.stderr.write("PRIVATE_VALIDATOR_STDERR_SENTINEL\n")
            os.write(1, b"PRIVATE_FD_SENTINEL\n")
            return load_json(shard_root / MANIFEST_NAME), load_json(
                shard_root / "results.json"
            )

        globals()["validate_shard_bundle"] = fake_validator
        try:
            validated = validate_input_shards(shard_roots)
        finally:
            globals()["validate_shard_bundle"] = original_validator

        outdir = root / "blind_pack"
        args = argparse.Namespace(outdir=outdir, shard_dir=shard_roots)
        _write_bundle(args, validated)
        manifest, _ = validate_output_bundle(outdir)
        ids = tuple(manifest["blind_order"]["ordered_blind_ids"])
        if ids != tuple(sorted(ids, key=_blind_order_key)):
            raise AssertionError("blind ordering failed")
        for record in manifest["images"]:
            path = outdir / record["image"]["relative_path"]
            _validate_metadata_free_png(path)
        _validate_metadata_free_png(outdir / CONTACT_NAME)
        first_image_path = outdir / manifest["images"][0]["image"]["relative_path"]
        first_image_bytes = first_image_path.read_bytes()
        with Image.open(first_image_path) as first_image:
            first_image.load()
            metadata_image = Image.frombytes("RGB", first_image.size, first_image.tobytes())
        injected_metadata = PngImagePlugin.PngInfo()
        injected_metadata.add_text("hidden_private_value", "sentinel")
        metadata_image.save(
            first_image_path,
            format="PNG",
            optimize=False,
            pnginfo=injected_metadata,
        )
        _expect_failure(
            lambda: _validate_metadata_free_png(first_image_path),
            "PNG ancillary metadata",
        )
        first_image_path.write_bytes(first_image_bytes)
        template = _read_self_hashed_json(outdir / TEMPLATE_NAME, "template_identity_sha256")
        if any(
            row[field] is not None
            for row in template["rows"]
            for field in ANNOTATION_FIELDS
        ):
            raise AssertionError("annotation template is not empty")

        extra = outdir / "unexpected.txt"
        extra.write_text("unexpected", encoding="utf-8")
        _expect_failure(lambda: validate_output_bundle(outdir), "closed file set")
        extra.unlink()
        validate_output_bundle(outdir)
        completion_hash = sha256_file(outdir / COMPLETION_NAME)
        _expect_failure(lambda: _write_bundle(args, validated), "no overwrite")
        if sha256_file(outdir / COMPLETION_NAME) != completion_hash:
            raise AssertionError("existing bundle changed during no-overwrite test")
        race_source = root / "race_source"
        race_target = root / "race_target"
        race_source.mkdir()
        race_target.mkdir()
        (race_source / "source_marker").write_text("source", encoding="utf-8")
        (race_target / "target_marker").write_text("target", encoding="utf-8")
        _expect_failure(
            lambda: _atomic_install_directory_noreplace(race_source, race_target),
            "renameat2 no-replace race",
        )
        if not (race_source / "source_marker").is_file() or not (
            race_target / "target_marker"
        ).is_file():
            raise AssertionError("renameat2 conflict modified source or target")

    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: four-shard mock strict path, suppressed Python/fd/libc output, exact "
        "0..31 allocation, fixed blind-ID hash order, metadata-free pixel-preserving PNGs, "
        "exact contact reconstruction, quality-only rubric, empty annotation template, "
        "self-hashes, closed set, metadata-injection rejection, renameat2 no-overwrite, CPU only"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    default_out = (
        data_root
        / "cross_scale_evidence/dit_imagenet256_t60_within_prefix_blind_review"
        / f"pool32_blind_pack_{sha256_file(RUNNER)[:7]}"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed shard directory; provide exactly four times.",
    )
    parser.add_argument("--outdir", type=Path, default=default_out)
    parser.add_argument("--self-test", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        return
    if len(args.shard_dir) != TOTAL_SHARDS:
        parser.error("provide exactly four --shard-dir arguments")
    roots = tuple(path.expanduser().absolute().resolve() for path in args.shard_dir)
    if len(set(roots)) != TOTAL_SHARDS:
        parser.error("--shard-dir inputs must be distinct")
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            parser.error("an input shard is not a plain directory")
    args.shard_dir = roots
    requested = args.outdir.expanduser().absolute()
    if os.path.lexists(requested):
        parser.error("no-overwrite output already exists")
    args.outdir = requested.resolve()
    for protected in (*roots, RUNNER.parent.parent):
        if _paths_overlap(args.outdir, protected):
            parser.error("--outdir overlaps a protected source/input tree")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)
    if args.self_test:
        run_self_test()
    else:
        run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
