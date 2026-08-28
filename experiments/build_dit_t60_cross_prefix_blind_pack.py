#!/usr/bin/env python3
"""Build the sealed 64-image cross-prefix blind-review bundle.

The public reviewer pack contains only canonical metadata-free endpoint PNGs
under newly randomized opaque IDs, the frozen external visual-anchor pack, a
quality-only rubric, two independent-review templates, an adjudication
template, and integrity records.  Seed, shard, slot, generation order, runner
IDs, evidence, and the reversible blind mapping are never copied there.

The private mapping and public pack are installed together under one immutable
top-level directory.  Give reviewers only ``reviewer_pack/``.  The private
seal is consumed only by the aggregate unseal program after consensus lock.
All eight validated 8-image shards are required; there is no partial-pool mode.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import hmac
import io
import json
import os
import platform
import secrets
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
    from .run_dit_t60_cross_prefix_mixture_validation_pool import (
        BRANCHES_PER_SHARD,
        EXPERIMENT as SHARD_EXPERIMENT,
        PROTOCOL_COPY_NAME,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        blind_id as runner_blind_id,
        shard_global_indices,
        validate_output_bundle as validate_shard_bundle,
        validate_output_pool as validate_shard_pool,
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
    from run_dit_t60_cross_prefix_mixture_validation_pool import (
        BRANCHES_PER_SHARD,
        EXPERIMENT as SHARD_EXPERIMENT,
        PROTOCOL_COPY_NAME,
        TOTAL_POOL_BRANCHES,
        TOTAL_SHARDS,
        blind_id as runner_blind_id,
        shard_global_indices,
        validate_output_bundle as validate_shard_bundle,
        validate_output_pool as validate_shard_pool,
    )
    from reproduce_dit_imagenet256 import (
        IMAGE_SIZE,
        atomic_json_dump,
        inspect_png,
        load_json,
        sha256_file,
        sha256_json,
    )


EXPERIMENT = "dit_imagenet256_t60_cross_prefix_blind_review_bundle"
SCHEMA_VERSION = 1
RUBRIC_SCHEMA_VERSION = 1
TEMPLATE_SCHEMA_VERSION = 1
PUBLIC_DIR_NAME = "reviewer_pack"
PRIVATE_DIR_NAME = "private_seal"
TOP_MANIFEST_NAME = "bundle_manifest.json"
TOP_COMPLETION_NAME = "bundle_completion.json"
PUBLIC_MANIFEST_NAME = "manifest.json"
PUBLIC_COMPLETION_NAME = "completion.json"
PRIVATE_MAPPING_NAME = "blind_mapping_private.json"
PRIVATE_COMMITMENT_NAME = "blind_mapping_commitment_frozen.json"
PRIVATE_COMPLETION_NAME = "completion.json"
RUBRIC_NAME = "rubric.json"
REVIEW_A_TEMPLATE_NAME = "review_A_template.json"
REVIEW_B_TEMPLATE_NAME = "review_B_template.json"
ADJUDICATION_TEMPLATE_NAME = "adjudication_template.json"
CONTACT_NAME = "contact_sheet.png"
IMAGE_DIR_NAME = "images"
ANCHOR_DIR_NAME = "external_visual_anchors"
README_NAME = "README.txt"

CONTACT_COLUMNS = 8
CONTACT_ROWS = 8
CONTACT_MARGIN = 12
CONTACT_GAP = 8
CONTACT_LABEL_HEIGHT = 24
CONTACT_BACKGROUND = (245, 245, 245)
CONTACT_TEXT = (15, 15, 15)

RUNNER = Path(__file__).resolve()
SHARD_RUNNER = RUNNER.with_name(
    "run_dit_t60_cross_prefix_mixture_validation_pool.py"
)
CONSENSUS_LOCKER = RUNNER.with_name("lock_dit_t60_cross_prefix_consensus.py")
AGGREGATE_SUMMARIZER = RUNNER.with_name(
    "summarize_dit_t60_cross_prefix_mixture_validation.py"
)
ANCHOR_BUILDER = RUNNER.with_name("build_dit_class207_visual_anchor_pack.py")
ANCHOR_CONFIG_SOURCE = (
    RUNNER.parent / "configs/dit_imagenet256_class207_visual_anchors_v1.json"
)
PROTOCOL_SOURCE = (
    RUNNER.parent
    / "configs/dit_imagenet256_t60_cross_prefix_mixture_validation_v1.json"
)
CONTACT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_ALLOWED_CHUNKS = {b"IHDR", b"IDAT", b"IEND"}
PUBLIC_BLIND_ID_PREFIX = "xr1_"
PUBLIC_BLIND_ID_HEX_LENGTH = 16
MAPPING_COMMITMENT_SCHEMA = (
    "dit_t60_cross_prefix_randomized_blind_mapping_commitment_v1"
)
MAPPING_COMMITMENT_STATUS = "FROZEN_BEFORE_GPU_EXECUTION"

PRIMARY_LABELS = {
    "clear_overall_structural_bad",
    "not_clear_overall_structural_bad",
    "uncertain",
}
HIND_LIMB_LABELS = {
    "clear_failure",
    "not_clear_failure",
    "uncertain_or_not_scorable",
}
TAIL_IDENTITIES = {"clear", "plausible", "unclear"}
TAIL_CONFIDENCES = {"high", "medium", "low"}
TAIL_DERIVED_LABELS = {
    "natural",
    "odd",
    "malformed",
    "uncertain_or_not_scorable",
}
TAIL_SCORABLE_VALUES = {"yes", "no"}
TERNARY_TAIL_FIELDS = (
    "tail_R_root_attachment",
    "tail_T_taper_and_volume",
    "tail_F_feather_or_hair_flow",
    "tail_D_distal_tip",
)
BINARY_TAIL_FIELDS = (
    "tail_P_paddle_like",
    "tail_B_short_or_blunt",
    "tail_S_abrupt_filament_transition",
)
ANNOTATION_FIELDS = (
    "primary_overall_structural_quality",
    "secondary_hind_limb_topology",
    *TERNARY_TAIL_FIELDS,
    *BINARY_TAIL_FIELDS,
    "tail_identity",
    "tail_scorable",
    "tail_confidence",
    "tail_derived_label",
    "notes",
)

REVIEW_DECLARATION = (
    "I completed this independent visual review without seeing the other "
    "review, any seed/shard/slot/order lineage, or any path-evidence value, "
    "alarm, score, rank, trace, or evidence-bearing result."
)
ADJUDICATION_DECLARATION = (
    "I completed this visual adjudication after the two independent reviews "
    "were locked and without seeing any seed/shard/slot/order lineage or any "
    "path-evidence value, alarm, score, rank, trace, or evidence-bearing result."
)

REVIEWER_README_TEXT = (
    "REVIEWER-ONLY CLOSED PACK\n"
    "\n"
    "Use only this reviewer_pack directory. Judge every image independently "
    "against rubric.json and external_visual_anchors/. The 64-image contact "
    "sheet is only for missing/duplicate/layout QA and must not move the "
    "threshold. Complete review_A and review_B independently; neither reviewer "
    "may see the other's labels before that review is sealed. A separate "
    "evidence-blind adjudicator resolves all disagreements and uncertain calls.\n"
    "Recognizable as a tail is not the same as a natural tail.\n"
)

ANNOTATION_ROW_SCHEMA: dict[str, str] = {
    "blind_id": f"required opaque ID matching {PUBLIC_BLIND_ID_PREFIX}[0-9a-f]{{16}}",
    "primary_overall_structural_quality": (
        "required enum: clear_overall_structural_bad | "
        "not_clear_overall_structural_bad | uncertain"
    ),
    "secondary_hind_limb_topology": (
        "required enum: clear_failure | not_clear_failure | "
        "uncertain_or_not_scorable"
    ),
    "tail_R_root_attachment": "0 natural; 1 mildly odd/uncertain; 2 clear defect",
    "tail_T_taper_and_volume": "0 natural; 1 mildly odd/uncertain; 2 clear defect",
    "tail_F_feather_or_hair_flow": "0 coherent; 1 mildly odd/uncertain; 2 clear defect",
    "tail_D_distal_tip": "0 natural; 1 mildly odd/uncertain; 2 clear defect",
    "tail_P_paddle_like": "binary diagnostic flag",
    "tail_B_short_or_blunt": "binary diagnostic flag",
    "tail_S_abrupt_filament_transition": "binary diagnostic flag",
    "tail_identity": "required enum: clear | plausible | unclear",
    "tail_scorable": "required enum: yes | no",
    "tail_confidence": "required enum: high | medium | low",
    "tail_derived_label": (
        "required enum: natural | odd | malformed | uncertain_or_not_scorable"
    ),
    "notes": "visible appearance only",
}

PUBLIC_FORBIDDEN_FRAGMENTS = (
    "global_index",
    "local_index",
    "trajectory_index",
    "shard_index",
    "slot_index",
    "stream_seed",
    "branch_seed",
    "runner_blind",
    "source_path",
    "source_shard",
    "protocol_identity",
    "runner_sha256",
    "delta_nu",
    "alpha_e",
    "log_e",
    "e_mix",
    "likelihood_ratio",
    "alarm",
    "evidence",
    "rank",
    "trace_private",
)

# These structural tokens have no legitimate reviewer-facing use, including in
# the exact blindness declarations.  Scan every public JSON/TXT file for them;
# broader words such as "evidence", "alarm", and "rank" remain permitted only
# inside byte-exact templates/README declarations.
PUBLIC_ALL_TEXT_FORBIDDEN_FRAGMENTS = (
    "global_index",
    "local_index",
    "trajectory_index",
    "shard_index",
    "slot_index",
    "stream_seed",
    "branch_seed",
    "runner_blind",
    "source_path",
    "source_shard",
    "source_png_sha256",
    "source_pixel",
    "protocol_identity",
    "runner_sha256",
    "delta_nu",
    "alpha_e",
    "log_e",
    "e_mix",
    "likelihood_ratio",
    "trace_private",
    "blinding_secret",
    "random_order_key",
    "commitment_identity",
)


@dataclass(frozen=True)
class SourceImage:
    global_index: int
    shard_index: int
    local_index: int
    runner_id: str
    source_path: Path
    source_file_sha256: str
    pixel_sha256: str


@dataclass(frozen=True)
class BlindedImage:
    public_id: str
    order_key: str
    source: SourceImage


@dataclass(frozen=True)
class ValidatedInputs:
    images: tuple[SourceImage, ...]
    protocol: dict[str, Any]
    shard_records: tuple[dict[str, Any], ...]
    anchor_manifest: dict[str, Any]
    anchor_completion: dict[str, Any]


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


def _require_plain_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected a plain file: {path}")


def _reject_special_entries(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("artifact root must be a plain directory")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError("artifact tree contains a link or special entry")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _plain_file_record(path: Path, root: Path) -> dict[str, Any]:
    _require_plain_file(path)
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


@contextlib.contextmanager
def _silence_process_output() -> Iterator[None]:
    """Suppress Python, fd-level, and buffered libc output from private validation."""

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
    try:
        with open(os.devnull, "w", encoding="utf-8") as null:
            os.dup2(null.fileno(), 1)
            os.dup2(null.fileno(), 2)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                yield
            if fflush(None) != 0:
                raise RuntimeError("could not flush suppressed libc output")
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _validate_anchor_pack(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from .build_dit_class207_visual_anchor_pack import validate_output_bundle
    except ImportError:  # pragma: no cover - direct CLI execution.
        from build_dit_class207_visual_anchor_pack import validate_output_bundle
    return validate_output_bundle(root)


def _validate_public_blind_id(value: Any) -> str:
    expected_length = len(PUBLIC_BLIND_ID_PREFIX) + PUBLIC_BLIND_ID_HEX_LENGTH
    if (
        not isinstance(value, str)
        or len(value) != expected_length
        or not value.startswith(PUBLIC_BLIND_ID_PREFIX)
        or any(
            character not in "0123456789abcdef"
            for character in value[len(PUBLIC_BLIND_ID_PREFIX) :]
        )
    ):
        raise RuntimeError("invalid randomized public blind ID")
    return value


def _derive_public_id(secret: bytes, global_index: int) -> str:
    digest = hmac.new(
        secret, f"public-id\0{global_index}".encode("ascii"), hashlib.sha256
    ).hexdigest()
    return f"{PUBLIC_BLIND_ID_PREFIX}{digest[:PUBLIC_BLIND_ID_HEX_LENGTH]}"


def _derive_order_key(secret: bytes, global_index: int) -> str:
    return hmac.new(
        secret, f"public-order\0{global_index}".encode("ascii"), hashlib.sha256
    ).hexdigest()


def build_mapping_commitment(secret: bytes | None = None) -> dict[str, Any]:
    """Create the private randomized mapping that must be frozen before GPU use."""

    secret = secrets.token_bytes(32) if secret is None else secret
    if len(secret) != 32:
        raise RuntimeError("mapping commitment secret must be exactly 256 bits")
    entries = [
        {
            "global_index": index,
            "public_blind_id": _derive_public_id(secret, index),
            "random_order_key": _derive_order_key(secret, index),
        }
        for index in range(TOTAL_POOL_BRANCHES)
    ]
    if len({entry["public_blind_id"] for entry in entries}) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("mapping commitment public blind-ID collision")
    if len({entry["random_order_key"] for entry in entries}) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("mapping commitment order-key collision")
    entries.sort(key=lambda entry: (entry["random_order_key"], entry["public_blind_id"]))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "commitment_schema": MAPPING_COMMITMENT_SCHEMA,
        "status": MAPPING_COMMITMENT_STATUS,
        "role": "PRIVATE_PRE_GPU_BLIND_MAPPING_DO_NOT_GIVE_TO_REVIEWERS",
        "pool_size": TOTAL_POOL_BRANCHES,
        "mapping_builder_filename": RUNNER.name,
        "mapping_builder_sha256": sha256_file(RUNNER),
        "construction": (
            "HMAC-SHA256 with one private 256-bit nonce; domain-separated public-id "
            "and public-order messages over each fixed global index 0..63"
        ),
        "blinding_secret_hex": secret.hex(),
        "entries": entries,
    }
    payload["commitment_identity_sha256"] = _canonical_self_hash(
        payload, "commitment_identity_sha256"
    )
    return payload


def validate_mapping_commitment(
    path: Path,
    protocol: dict[str, Any] | None = None,
    *,
    enforce_protocol_path: bool = True,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("mapping commitment must be a plain file")
    payload = _read_self_hashed_json(path, "commitment_identity_sha256")
    expected_keys = {
        "schema_version",
        "commitment_schema",
        "status",
        "role",
        "pool_size",
        "mapping_builder_filename",
        "mapping_builder_sha256",
        "construction",
        "blinding_secret_hex",
        "entries",
        "commitment_identity_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("mapping commitment schema changed")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["commitment_schema"] != MAPPING_COMMITMENT_SCHEMA
        or payload["status"] != MAPPING_COMMITMENT_STATUS
        or payload["role"]
        != "PRIVATE_PRE_GPU_BLIND_MAPPING_DO_NOT_GIVE_TO_REVIEWERS"
        or payload["pool_size"] != TOTAL_POOL_BRANCHES
        or payload["mapping_builder_filename"] != RUNNER.name
        or payload["mapping_builder_sha256"] != sha256_file(RUNNER)
    ):
        raise RuntimeError("mapping commitment identity/source binding changed")
    try:
        secret = bytes.fromhex(payload["blinding_secret_hex"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("mapping commitment secret is malformed") from exc
    if len(secret) != 32:
        raise RuntimeError("mapping commitment secret is not 256 bits")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("mapping commitment must contain exactly 64 entries")
    observed_indices: set[int] = set()
    previous_order: tuple[str, str] | None = None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "global_index",
            "public_blind_id",
            "random_order_key",
        }:
            raise RuntimeError("mapping commitment entry schema changed")
        index = entry["global_index"]
        if type(index) is not int or index not in range(TOTAL_POOL_BRANCHES):
            raise RuntimeError("mapping commitment global index is invalid")
        if index in observed_indices:
            raise RuntimeError("mapping commitment duplicates a global index")
        observed_indices.add(index)
        if (
            entry["public_blind_id"] != _derive_public_id(secret, index)
            or entry["random_order_key"] != _derive_order_key(secret, index)
        ):
            raise RuntimeError("mapping commitment does not reconstruct from its nonce")
        current_order = (entry["random_order_key"], entry["public_blind_id"])
        if previous_order is not None and current_order <= previous_order:
            raise RuntimeError("mapping commitment entries are not in fixed random order")
        previous_order = current_order
    if observed_indices != set(range(TOTAL_POOL_BRANCHES)):
        raise RuntimeError("mapping commitment does not cover indices 0..63")
    if protocol is not None:
        bound_path = (
            str(path.resolve())
            if enforce_protocol_path
            else protocol.get("blind_mapping_commitment_binding", {}).get(
                "commitment_path"
            )
        )
        expected_binding = {
            "status": MAPPING_COMMITMENT_STATUS,
            "commitment_schema": MAPPING_COMMITMENT_SCHEMA,
            "pool_size": TOTAL_POOL_BRANCHES,
            "commitment_path": bound_path,
            "mapping_builder_filename": RUNNER.name,
            "mapping_builder_sha256": sha256_file(RUNNER),
            "commitment_identity_sha256": payload[
                "commitment_identity_sha256"
            ],
            "commitment_file_sha256": sha256_file(path),
        }
        if protocol.get("blind_mapping_commitment_binding") != expected_binding:
            raise RuntimeError("protocol does not bind this frozen blind mapping")
    return payload


def freeze_mapping_commitment(target: Path) -> dict[str, Any]:
    payload = build_mapping_commitment()
    if os.path.lexists(target):
        raise RuntimeError("refusing to overwrite a frozen mapping commitment")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.staging-", dir=target.parent
    ) as temporary:
        staged = Path(temporary) / target.name
        atomic_json_dump(payload, staged)
        try:
            os.link(staged, target, follow_symlinks=False)
        except FileExistsError:
            raise RuntimeError("refusing to overwrite a frozen mapping commitment") from None
    validate_mapping_commitment(target)
    return payload


def _pipeline_binding_expected() -> dict[str, Any]:
    paths = (RUNNER, CONSENSUS_LOCKER, AGGREGATE_SUMMARIZER)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise RuntimeError("a frozen blind-pipeline source is missing")
    return {
        "blind_pack_builder_filename": RUNNER.name,
        "blind_pack_builder_sha256": sha256_file(RUNNER),
        "consensus_locker_filename": CONSENSUS_LOCKER.name,
        "consensus_locker_sha256": sha256_file(CONSENSUS_LOCKER),
        "aggregate_unseal_summarizer_filename": AGGREGATE_SUMMARIZER.name,
        "aggregate_unseal_summarizer_sha256": sha256_file(AGGREGATE_SUMMARIZER),
    }


def _validate_protocol_for_pipeline(
    protocol: dict[str, Any], *, require_frozen: bool = True
) -> None:
    if protocol.get("protocol_identity_sha256") != _canonical_self_hash(
        protocol, "protocol_identity_sha256"
    ):
        raise RuntimeError("cross-prefix protocol self-hash changed")
    if protocol.get("protocol_name") != (
        "dit_imagenet256_t60_cross_prefix_mixture_validation_v1"
    ):
        raise RuntimeError("wrong cross-prefix protocol")
    expected_status = (
        "FROZEN_BEFORE_GPU_EXECUTION" if require_frozen else protocol.get("protocol_status")
    )
    if require_frozen and expected_status != protocol.get("protocol_status"):
        raise RuntimeError("cross-prefix protocol is not frozen")
    if require_frozen and protocol.get("authorization_gate", {}).get(
        "gpu_execution_authorized"
    ) is not True:
        raise RuntimeError("cross-prefix GPU authorization is not frozen true")
    if protocol.get("blind_pipeline_binding") != _pipeline_binding_expected():
        raise RuntimeError("protocol blind-pipeline source binding changed")
    commitment_binding = protocol.get("blind_mapping_commitment_binding")
    if (
        not isinstance(commitment_binding, dict)
        or set(commitment_binding)
        != {
            "status",
            "commitment_schema",
            "pool_size",
            "commitment_path",
            "mapping_builder_filename",
            "mapping_builder_sha256",
            "commitment_identity_sha256",
            "commitment_file_sha256",
        }
        or commitment_binding.get("status") != MAPPING_COMMITMENT_STATUS
        or commitment_binding.get("commitment_schema")
        != MAPPING_COMMITMENT_SCHEMA
        or commitment_binding.get("pool_size") != TOTAL_POOL_BRANCHES
        or not isinstance(commitment_binding.get("commitment_path"), str)
        or not Path(commitment_binding["commitment_path"]).is_absolute()
        or commitment_binding.get("mapping_builder_filename") != RUNNER.name
        or commitment_binding.get("mapping_builder_sha256") != sha256_file(RUNNER)
        or not _is_sha256(commitment_binding.get("commitment_identity_sha256"))
        or not _is_sha256(commitment_binding.get("commitment_file_sha256"))
    ):
        raise RuntimeError("protocol pre-GPU blind-mapping commitment binding changed")
    commitment_path = Path(commitment_binding["commitment_path"])
    if (
        commitment_path.resolve() != commitment_path
        or commitment_path.is_symlink()
        or not commitment_path.is_file()
    ):
        raise RuntimeError("frozen blind-mapping commitment path is missing or indirect")
    validate_mapping_commitment(commitment_path, protocol)
    review = protocol.get("blind_review", {})
    primary = review.get("primary_visual_endpoint", {})
    if (
        primary.get("name")
        != "overall_obvious_structural_bad_under_frozen_external_anchor_rubric"
        or primary.get("role") != "sole primary visual endpoint"
        or primary.get("labels")
        != [
            "clear_overall_structural_bad",
            "not_clear_overall_structural_bad",
            "uncertain",
        ]
    ):
        raise RuntimeError("frozen primary visual endpoint changed")
    if review.get("annotation_lock", {}).get("unseal_count") != 1:
        raise RuntimeError("protocol no longer requires one aggregate unseal")
    pool = protocol.get("pool", {})
    if (
        pool.get("shard_count") != TOTAL_SHARDS
        or pool.get("class207_trajectories_per_shard") != BRANCHES_PER_SHARD
        or pool.get("total_class207_trajectories") != TOTAL_POOL_BRANCHES
    ):
        raise RuntimeError("protocol 8x8 pool changed")


def _anchor_binding_from_manifest(
    anchor_root: Path,
    manifest: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, Any]:
    anchor_root = anchor_root.resolve()
    if anchor_root.is_symlink() or not anchor_root.is_dir():
        raise RuntimeError("external-anchor root must be one absolute plain directory")
    if (
        ANCHOR_CONFIG_SOURCE.is_symlink()
        or not ANCHOR_CONFIG_SOURCE.is_file()
        or ANCHOR_BUILDER.is_symlink()
        or not ANCHOR_BUILDER.is_file()
    ):
        raise RuntimeError("external-anchor frozen sources are unavailable or indirect")
    config = _read_self_hashed_json(
        ANCHOR_CONFIG_SOURCE, "anchor_config_identity_sha256"
    )
    rubric = _read_self_hashed_json(
        anchor_root / "rubric.json", "rubric_identity_sha256"
    )
    pack_payload = manifest.get("pack_payload_sha256")
    return {
        "status": "COMPLETE_AND_FROZEN_BEFORE_GPU_EXECUTION",
        "public_pack_root": str(anchor_root),
        "public_pack_root_identity_sha256": sha256_json(
            {
                "absolute_root": str(anchor_root),
                "pack_payload_sha256": pack_payload,
            }
        ),
        "anchor_config_path": str(ANCHOR_CONFIG_SOURCE.resolve()),
        "anchor_config_identity_sha256": config.get(
            "anchor_config_identity_sha256"
        ),
        "anchor_config_file_sha256": sha256_file(ANCHOR_CONFIG_SOURCE),
        "builder": {
            "filename": ANCHOR_BUILDER.name,
            "sha256": sha256_file(ANCHOR_BUILDER),
        },
        "manifest_identity_sha256": manifest.get("identity_sha256"),
        "manifest_file_sha256": sha256_file(anchor_root / "manifest.json"),
        "pack_payload_sha256": pack_payload,
        "rubric_identity_sha256": rubric.get("rubric_identity_sha256"),
        "rubric_file_sha256": sha256_file(anchor_root / "rubric.json"),
        "completion_payload_sha256": completion.get("payload_sha256"),
        "completion_file_sha256": sha256_file(anchor_root / "completion.json"),
    }


def _public_anchor_binding_from_manifest(
    anchor_root: Path,
    manifest: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, Any]:
    rubric = _read_self_hashed_json(
        anchor_root / "rubric.json", "rubric_identity_sha256"
    )
    return {
        "anchor_config_identity_sha256": manifest.get(
            "anchor_config_identity_sha256"
        ),
        "manifest_identity_sha256": manifest.get("identity_sha256"),
        "manifest_file_sha256": sha256_file(anchor_root / "manifest.json"),
        "pack_payload_sha256": manifest.get("pack_payload_sha256"),
        "rubric_identity_sha256": rubric.get("rubric_identity_sha256"),
        "rubric_file_sha256": sha256_file(anchor_root / "rubric.json"),
        "completion_payload_sha256": completion.get("payload_sha256"),
        "completion_file_sha256": sha256_file(anchor_root / "completion.json"),
    }


def _validate_anchor_protocol_binding(
    protocol: dict[str, Any],
    anchor_root: Path,
    manifest: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, Any]:
    observed = _anchor_binding_from_manifest(anchor_root, manifest, completion)
    expected = protocol.get("external_visual_anchor_binding", {}).get(
        "metadata_stripped_anchor_pack"
    )
    if expected != observed:
        raise RuntimeError("external visual-anchor pack differs from frozen protocol")
    hash_keys = {
        "public_pack_root_identity_sha256",
        "anchor_config_identity_sha256",
        "anchor_config_file_sha256",
        "manifest_identity_sha256",
        "manifest_file_sha256",
        "pack_payload_sha256",
        "rubric_identity_sha256",
        "rubric_file_sha256",
        "completion_payload_sha256",
        "completion_file_sha256",
    }
    if any(not _is_sha256(observed[key]) for key in hash_keys):
        raise RuntimeError("external visual-anchor binding contains malformed hashes")
    if observed["anchor_config_identity_sha256"] != manifest.get(
        "anchor_config_identity_sha256"
    ):
        raise RuntimeError("anchor manifest/config identity changed")
    if completion.get("manifest_identity_sha256") != manifest.get("identity_sha256"):
        raise RuntimeError("external anchor completion/manifest binding changed")
    return observed


def _strict_validate_shard_silently(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with _silence_process_output():
            return validate_shard_bundle(root)
    except Exception as exc:
        del exc
        raise RuntimeError("an input shard failed strict validation") from None


def validate_input_shards(
    shard_roots: tuple[Path, ...], anchor_root: Path
) -> ValidatedInputs:
    if len(shard_roots) != TOTAL_SHARDS:
        raise RuntimeError("exactly eight shard directories are required")
    if any(root.is_symlink() or not root.is_dir() for root in shard_roots):
        raise RuntimeError("every shard must be a plain directory")
    shard_roots = tuple(root.resolve() for root in shard_roots)
    if len(set(shard_roots)) != TOTAL_SHARDS:
        raise RuntimeError("the eight shard directories must be distinct")
    anchor_root = anchor_root.resolve()
    _reject_special_entries(anchor_root)
    anchor_manifest, anchor_completion = _validate_anchor_pack(anchor_root)

    manifests: list[dict[str, Any]] = []
    results_list: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    for root in shard_roots:
        _reject_special_entries(root)
        manifest, results = _strict_validate_shard_silently(root)
        if manifest.get("experiment") != SHARD_EXPERIMENT:
            raise RuntimeError("input is not a cross-prefix validation shard")
        protocol = _read_self_hashed_json(
            root / PROTOCOL_COPY_NAME, "protocol_identity_sha256"
        )
        _validate_protocol_for_pipeline(protocol)
        manifests.append(manifest)
        results_list.append(results)
        protocols.append(protocol)

    with _silence_process_output():
        pool_validation = validate_shard_pool(shard_roots)
    if pool_validation.get("status") != "valid-complete-pool":
        raise RuntimeError("strict runner did not validate the complete 8x8 pool")
    canonical_protocols = {
        json.dumps(value, ensure_ascii=False, sort_keys=True) for value in protocols
    }
    if len(canonical_protocols) != 1:
        raise RuntimeError("input shards do not contain one common frozen protocol")
    protocol = protocols[0]
    if PROTOCOL_SOURCE.is_file() and not PROTOCOL_SOURCE.is_symlink():
        local_protocol = _read_self_hashed_json(
            PROTOCOL_SOURCE, "protocol_identity_sha256"
        )
        if local_protocol != protocol:
            raise RuntimeError("input protocol differs from the local frozen protocol")
    anchor_binding = _validate_anchor_protocol_binding(
        protocol, anchor_root, anchor_manifest, anchor_completion
    )

    by_shard: dict[int, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for root, manifest, results in zip(shard_roots, manifests, results_list):
        shard_index = manifest.get("pool", {}).get("this_shard_index")
        if type(shard_index) is not int or shard_index not in range(TOTAL_SHARDS):
            raise RuntimeError("invalid shard index")
        if shard_index in by_shard:
            raise RuntimeError("duplicate shard index")
        by_shard[shard_index] = (root, manifest, results)
    if set(by_shard) != set(range(TOTAL_SHARDS)):
        raise RuntimeError("input pool is not exactly shard indices 0..7")

    images: list[SourceImage] = []
    shard_records: list[dict[str, Any]] = []
    for shard_index in range(TOTAL_SHARDS):
        root, manifest, results = by_shard[shard_index]
        expected_indices = list(shard_global_indices(shard_index))
        if manifest.get("pool", {}).get(
            "this_shard_global_branch_indices"
        ) != expected_indices:
            raise RuntimeError("shard allocation changed")
        records = results.get("branch_records")
        if not isinstance(records, list) or len(records) != BRANCHES_PER_SHARD:
            raise RuntimeError("shard endpoint record count changed")
        shard_record = {
            "shard_index": shard_index,
            "manifest_identity_sha256": manifest.get("identity_sha256"),
            "manifest_file_sha256": sha256_file(root / "manifest.json"),
            "results_payload_sha256": results.get("payload_sha256"),
            "results_file_sha256": sha256_file(root / "results.json"),
            "trace_file_sha256": results.get("private_trace", {}).get("sha256"),
            "completion_file_sha256": sha256_file(root / "completion.json"),
        }
        if any(
            not _is_sha256(value)
            for key, value in shard_record.items()
            if key != "shard_index"
        ):
            raise RuntimeError("malformed shard provenance hash")
        shard_records.append(shard_record)
        for local_index, (global_index, record) in enumerate(
            zip(expected_indices, records)
        ):
            if not isinstance(record, dict):
                raise RuntimeError("malformed branch record")
            expected_runner_id = runner_blind_id(global_index)
            if (
                record.get("global_index") != global_index
                or record.get("local_index") != local_index
                or record.get("blind_id") != expected_runner_id
            ):
                raise RuntimeError("branch record allocation/identity changed")
            image_record = record.get("image")
            if not isinstance(image_record, dict):
                raise RuntimeError("branch endpoint image record is malformed")
            relative = image_record.get("relative_path")
            if not isinstance(relative, str):
                raise RuntimeError("branch endpoint relative path is malformed")
            source = (root / relative).resolve()
            if root not in source.parents:
                raise RuntimeError("branch endpoint escapes its validated shard")
            observed = _image_record(source, root, (IMAGE_SIZE, IMAGE_SIZE))
            if observed != image_record:
                raise RuntimeError("branch endpoint changed after strict validation")
            images.append(
                SourceImage(
                    global_index=global_index,
                    shard_index=shard_index,
                    local_index=local_index,
                    runner_id=expected_runner_id,
                    source_path=source,
                    source_file_sha256=str(image_record["sha256"]),
                    pixel_sha256=str(image_record["pixel_sha256"]),
                )
            )
    if [item.global_index for item in images] != list(range(TOTAL_POOL_BRANCHES)):
        raise RuntimeError("validated endpoints do not cover trajectory indices 0..63")
    if len({item.source_file_sha256 for item in images}) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("duplicate endpoint PNG file detected")
    if len({item.pixel_sha256 for item in images}) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("duplicate endpoint RGB pixels detected")
    anchor_pixels = {
        record.get("image", {}).get("pixel_rgb_sha256")
        for record in anchor_manifest.get("anchors", [])
        if isinstance(record, dict)
    }
    if anchor_pixels & {item.pixel_sha256 for item in images}:
        raise RuntimeError("a current-pool endpoint duplicates an external anchor")
    if pool_validation.get("protocol_identity_sha256") != protocol.get(
        "protocol_identity_sha256"
    ):
        raise RuntimeError("runner pool/protocol identity changed")
    if pool_validation.get("runner_sha256") != sha256_file(SHARD_RUNNER):
        raise RuntimeError("runner pool/source identity changed")
    if anchor_binding != protocol["external_visual_anchor_binding"][
        "metadata_stripped_anchor_pack"
    ]:
        raise AssertionError("anchor binding postcondition failed")
    return ValidatedInputs(
        images=tuple(images),
        protocol=protocol,
        shard_records=tuple(shard_records),
        anchor_manifest=anchor_manifest,
        anchor_completion=anchor_completion,
    )


def expected_tail_derived_label(row: dict[str, Any]) -> str:
    if row["tail_scorable"] == "no" or row["tail_identity"] == "unclear":
        return "uncertain_or_not_scorable"
    dimensions = [int(row[key]) for key in TERNARY_TAIL_FIELDS]
    flags = [int(row[key]) for key in BINARY_TAIL_FIELDS]
    if any(value == 2 for value in dimensions):
        return "malformed"
    if (
        any(value == 1 for value in dimensions)
        or any(flags)
        or row["tail_identity"] == "plausible"
    ):
        return "odd"
    return "natural"


def validate_completed_annotation_row(row: Any, *, context: str) -> dict[str, Any]:
    expected_keys = {"blind_id", *ANNOTATION_FIELDS}
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise RuntimeError(f"{context}: annotation row schema changed")
    identifier = _validate_public_blind_id(row["blind_id"])
    if row["primary_overall_structural_quality"] not in PRIMARY_LABELS:
        raise RuntimeError(f"{context}/{identifier}: invalid primary label")
    if row["secondary_hind_limb_topology"] not in HIND_LIMB_LABELS:
        raise RuntimeError(f"{context}/{identifier}: invalid hind-limb label")
    if row["tail_scorable"] not in TAIL_SCORABLE_VALUES:
        raise RuntimeError(f"{context}/{identifier}: invalid tail_scorable")
    if row["tail_identity"] not in TAIL_IDENTITIES:
        raise RuntimeError(f"{context}/{identifier}: invalid tail identity")
    if row["tail_confidence"] not in TAIL_CONFIDENCES:
        raise RuntimeError(f"{context}/{identifier}: invalid tail confidence")
    if row["tail_derived_label"] not in TAIL_DERIVED_LABELS:
        raise RuntimeError(f"{context}/{identifier}: invalid tail derived label")
    notes = row["notes"]
    if not isinstance(notes, str) or len(notes) > 2_000:
        raise RuntimeError(f"{context}/{identifier}: invalid visible-only notes")
    forbidden_note_terms = (
        "seed",
        "shard",
        "slot",
        "order",
        "evidence",
        "alarm",
        "score",
        "rank",
        "likelihood",
        "theta",
        "delta_nu",
        "e_mix",
    )
    if any(term in notes.lower() for term in forbidden_note_terms):
        raise RuntimeError(f"{context}/{identifier}: notes mention forbidden information")
    tail_fields = TERNARY_TAIL_FIELDS + BINARY_TAIL_FIELDS
    if row["tail_scorable"] == "no":
        if row["tail_identity"] != "unclear":
            raise RuntimeError(f"{context}/{identifier}: unscorable tail must be unclear")
        if any(row[key] is not None for key in tail_fields):
            raise RuntimeError(f"{context}/{identifier}: unscorable tail fields must be null")
    else:
        if row["tail_identity"] == "unclear":
            raise RuntimeError(f"{context}/{identifier}: scorable tail cannot be unclear")
        for key in TERNARY_TAIL_FIELDS:
            if type(row[key]) is not int or row[key] not in (0, 1, 2):
                raise RuntimeError(f"{context}/{identifier}: invalid {key}")
        for key in BINARY_TAIL_FIELDS:
            if type(row[key]) is not int or row[key] not in (0, 1):
                raise RuntimeError(f"{context}/{identifier}: invalid {key}")
    expected_derived = expected_tail_derived_label(row)
    if row["tail_derived_label"] != expected_derived:
        raise RuntimeError(
            f"{context}/{identifier}: tail derived label must be {expected_derived}"
        )
    return row


def _rubric_payload(protocol: dict[str, Any], anchor_manifest: dict[str, Any]) -> dict[str, Any]:
    review = protocol["blind_review"]
    primary = review["primary_visual_endpoint"]
    secondary = review["secondary_visual_endpoints"]
    payload: dict[str, Any] = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_name": "dit_class207_cross_prefix_external_anchor_quality_v1",
        "purpose": (
            "Independent visible-quality review of one fixed 64-image pool. "
            "The other 63 current images may not move the frozen threshold."
        ),
        "external_anchor_manifest_identity_sha256": anchor_manifest[
            "identity_sha256"
        ],
        "external_anchor_pack_payload_sha256": anchor_manifest[
            "pack_payload_sha256"
        ],
        "annotation_row_schema": ANNOTATION_ROW_SCHEMA,
        "primary_endpoint": primary,
        "secondary_visual_endpoints": secondary,
        "tail_identity_is_not_naturalness": (
            "Recognizable as a tail is not equivalent to natural. Judge root attachment "
            "(R), taper/volume (T), feather or hair flow (F), and distal tip (D) "
            "separately; record paddle (P), short/blunt (B), and abrupt filament (S). "
            "A fluffy appearance is natural only when flow and geometry are coherent."
        ),
        "anchor_use_rule": (
            "Judge every endpoint independently against the five ordinary anchors, "
            "two clear-bad anchors, and written definitions. The current 64-image "
            "contact sheet is only for completeness/layout QA and must not recalibrate "
            "the badness threshold."
        ),
    }
    public_text = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    if any(fragment in public_text for fragment in PUBLIC_FORBIDDEN_FRAGMENTS):
        raise RuntimeError("quality-only rubric contains forbidden private information")
    payload["rubric_identity_sha256"] = _canonical_self_hash(
        payload, "rubric_identity_sha256"
    )
    return payload


def _empty_rows(blind_ids: tuple[str, ...], *, adjudication: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier in blind_ids:
        row: dict[str, Any] = {"blind_id": _validate_public_blind_id(identifier)}
        row.update({field: None for field in ANNOTATION_FIELDS})
        if adjudication:
            row["adjudication_reason"] = None
        rows.append(row)
    return rows


def _review_template(
    role: str,
    blind_ids: tuple[str, ...],
    public_manifest_identity: str,
    rubric_identity: str,
) -> dict[str, Any]:
    if role not in {"reviewer_A", "reviewer_B"}:
        raise ValueError("invalid independent-review role")
    payload: dict[str, Any] = {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "annotation_schema": "dit_t60_cross_prefix_independent_visual_review_v1",
        "role": role,
        "blind_pack_manifest_identity_sha256": public_manifest_identity,
        "rubric_identity_sha256": rubric_identity,
        "status": None,
        "reviewer": {
            "reviewer_id": None,
            "started_at_utc": None,
            "completed_at_utc": None,
            "evidence_and_lineage_unseen": None,
            "other_review_unseen_before_completion": None,
            "declaration": REVIEW_DECLARATION,
        },
        "rows": _empty_rows(blind_ids, adjudication=False),
        "annotation_identity_sha256": None,
    }
    return payload


def _adjudication_template(
    blind_ids: tuple[str, ...],
    public_manifest_identity: str,
    rubric_identity: str,
) -> dict[str, Any]:
    return {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "annotation_schema": "dit_t60_cross_prefix_visual_consensus_v1",
        "role": "consensus_adjudication",
        "blind_pack_manifest_identity_sha256": public_manifest_identity,
        "rubric_identity_sha256": rubric_identity,
        "status": None,
        "review_A_annotation_identity_sha256": None,
        "review_B_annotation_identity_sha256": None,
        "adjudicator": {
            "adjudicator_id": None,
            "started_at_utc": None,
            "completed_at_utc": None,
            "evidence_and_lineage_unseen": None,
            "inspected_every_disagreement_and_uncertain_call": None,
            "declaration": ADJUDICATION_DECLARATION,
        },
        "rows": _empty_rows(blind_ids, adjudication=True),
        "annotation_identity_sha256": None,
    }


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
        if observed_crc != expected_crc or chunk_type not in PNG_ALLOWED_CHUNKS:
            raise RuntimeError("PNG is noncanonical or contains metadata")
        chunks.append(chunk_type)
        position = end
        if chunk_type == b"IEND":
            break
    if position != len(data):
        raise RuntimeError("PNG contains trailing bytes")
    if (
        not chunks
        or chunks[0] != b"IHDR"
        or chunks[-1] != b"IEND"
        or chunks.count(b"IHDR") != 1
        or chunks.count(b"IEND") != 1
        or b"IDAT" not in chunks
    ):
        raise RuntimeError("PNG critical chunk structure changed")
    first_idat = chunks.index(b"IDAT")
    if any(chunk != b"IDAT" for chunk in chunks[first_idat:-1]):
        raise RuntimeError("PNG IDAT chunks are not contiguous")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise RuntimeError("PNG mode is not canonical RGB")
        canonical = Image.frombytes("RGB", image.size, image.tobytes())
    buffer = io.BytesIO()
    canonical.save(buffer, format="PNG", optimize=False)
    if data != buffer.getvalue():
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
        raise RuntimeError("contact sheet requires exactly 64 images")
    canvas = Image.new("RGB", _contact_size(), CONTACT_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = _font()
    cell_height = IMAGE_SIZE + CONTACT_LABEL_HEIGHT
    for position, (identifier, path) in enumerate(images):
        _validate_public_blind_id(identifier)
        row, column = divmod(position, CONTACT_COLUMNS)
        left = CONTACT_MARGIN + column * (IMAGE_SIZE + CONTACT_GAP)
        top = CONTACT_MARGIN + row * (cell_height + CONTACT_GAP)
        with Image.open(path) as source:
            source.load()
            if source.mode != "RGB" or source.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError("contact source violates RGB/256 contract")
            canvas.paste(source, (left, top))
        bounds = draw.textbbox((0, 0), identifier, font=font)
        width = bounds[2] - bounds[0]
        draw.text(
            (left + (IMAGE_SIZE - width) // 2, top + IMAGE_SIZE + 3),
            identifier,
            fill=CONTACT_TEXT,
            font=font,
        )
    return canvas


def _copy_closed_tree(source: Path, destination: Path) -> None:
    _reject_special_entries(source)
    if os.path.lexists(destination):
        raise RuntimeError("refusing to overwrite copied anchor pack")
    destination.mkdir()
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir()
        else:
            target.write_bytes(path.read_bytes())


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
        raise FileExistsError(error_number, "refusing to replace blind bundle", target)
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("filesystem/kernel lacks atomic RENAME_NOREPLACE")
    raise OSError(error_number, os.strerror(error_number), target)


def _blinded_images(
    images: tuple[SourceImage, ...], commitment: dict[str, Any]
) -> tuple[BlindedImage, ...]:
    by_index = {item.global_index: item for item in images}
    if set(by_index) != set(range(TOTAL_POOL_BRANCHES)):
        raise RuntimeError("blinding requires source indices exactly 0..63")
    values = tuple(
        BlindedImage(
            public_id=entry["public_blind_id"],
            order_key=entry["random_order_key"],
            source=by_index[entry["global_index"]],
        )
        for entry in commitment["entries"]
    )
    if len({value.public_id for value in values}) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("randomized public blind-ID collision")
    if len({value.order_key for value in values}) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("randomized public order-key collision")
    if values != tuple(sorted(values, key=lambda item: (item.order_key, item.public_id))):
        raise RuntimeError("pre-frozen mapping commitment is not in random review order")
    return values


def _write_bundle(
    outdir: Path,
    anchor_root: Path,
    mapping_commitment_path: Path,
    validated: ValidatedInputs,
    *,
    require_protocol_validation: bool = True,
) -> None:
    if os.path.lexists(outdir):
        raise RuntimeError("refusing to overwrite existing blind bundle")
    commitment = validate_mapping_commitment(
        mapping_commitment_path, validated.protocol
    )
    blinded = _blinded_images(validated.images, commitment)
    anchor_binding = _validate_anchor_protocol_binding(
        validated.protocol,
        anchor_root,
        validated.anchor_manifest,
        validated.anchor_completion,
    )
    public_anchor_binding = _public_anchor_binding_from_manifest(
        anchor_root, validated.anchor_manifest, validated.anchor_completion
    )
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        public = staging / PUBLIC_DIR_NAME
        private = staging / PRIVATE_DIR_NAME
        image_dir = public / IMAGE_DIR_NAME
        public.mkdir(parents=True)
        private.mkdir()
        image_dir.mkdir()
        _copy_closed_tree(anchor_root, public / ANCHOR_DIR_NAME)
        (private / PRIVATE_COMMITMENT_NAME).write_bytes(
            mapping_commitment_path.read_bytes()
        )

        image_records: list[dict[str, Any]] = []
        contact_inputs: list[tuple[str, Path]] = []
        for position, item in enumerate(blinded):
            destination = image_dir / f"{item.public_id}.png"
            inspection = _clean_reencode(item.source.source_path, destination)
            if inspection["pixel_sha256"] != item.source.pixel_sha256:
                raise RuntimeError("metadata stripping changed endpoint RGB pixels")
            image_records.append(
                {
                    "review_position": position,
                    "blind_id": item.public_id,
                    "image": {
                        "relative_path": destination.relative_to(public).as_posix(),
                        **inspection,
                    },
                }
            )
            contact_inputs.append((item.public_id, destination))
        contact_path = public / CONTACT_NAME
        render_contact_sheet(tuple(contact_inputs)).save(
            contact_path, format="PNG", optimize=False
        )
        _validate_metadata_free_png(contact_path)

        rubric = _rubric_payload(validated.protocol, validated.anchor_manifest)
        atomic_json_dump(rubric, public / RUBRIC_NAME)
        (public / README_NAME).write_text(REVIEWER_README_TEXT, encoding="utf-8")

        # Public manifest is created before templates so every template is bound to
        # one immutable image/order/anchor identity without exposing private lineage.
        public_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "role": "REVIEWER_ONLY_QUALITY_CONTENT",
            "image_count": TOTAL_POOL_BRANCHES,
            "randomized_blind_ids": True,
            "blind_order_depends_on_pixels_or_results": False,
            "blind_ids": [item.public_id for item in blinded],
            "images": image_records,
            "contact_sheet": _image_record(contact_path, public, _contact_size()),
            "external_visual_anchors": {
                **public_anchor_binding,
                "relative_path": ANCHOR_DIR_NAME,
            },
            "rubric": _plain_file_record(public / RUBRIC_NAME, public),
            "readme": _plain_file_record(public / README_NAME, public),
            "contact_renderer": {
                "columns": CONTACT_COLUMNS,
                "rows": CONTACT_ROWS,
                "font_file_sha256": sha256_file(CONTACT_FONT),
            },
            "dependencies": _cpu_dependencies(),
        }
        public_manifest["identity_sha256"] = _canonical_self_hash(
            public_manifest, "identity_sha256"
        )
        public_identity = public_manifest["identity_sha256"]
        templates = {
            REVIEW_A_TEMPLATE_NAME: _review_template(
                "reviewer_A",
                tuple(public_manifest["blind_ids"]),
                public_identity,
                rubric["rubric_identity_sha256"],
            ),
            REVIEW_B_TEMPLATE_NAME: _review_template(
                "reviewer_B",
                tuple(public_manifest["blind_ids"]),
                public_identity,
                rubric["rubric_identity_sha256"],
            ),
            ADJUDICATION_TEMPLATE_NAME: _adjudication_template(
                tuple(public_manifest["blind_ids"]),
                public_identity,
                rubric["rubric_identity_sha256"],
            ),
        }
        for name, payload in templates.items():
            atomic_json_dump(payload, public / name)
        public_manifest["templates"] = {
            name: _plain_file_record(public / name, public) for name in sorted(templates)
        }
        # The template records are intentionally excluded from the identity used
        # inside the templates; add a second payload hash that closes all public JSON.
        public_manifest["public_payload_sha256"] = _canonical_self_hash(
            public_manifest, "public_payload_sha256"
        )
        atomic_json_dump(public_manifest, public / PUBLIC_MANIFEST_NAME)
        public_completion: dict[str, Any] = {
            "complete": True,
            "manifest_identity_sha256": public_identity,
            "public_payload_sha256": public_manifest["public_payload_sha256"],
            "manifest_file_sha256": sha256_file(public / PUBLIC_MANIFEST_NAME),
            "image_count": TOTAL_POOL_BRANCHES,
        }
        public_completion["payload_sha256"] = _canonical_self_hash(
            public_completion, "payload_sha256"
        )
        atomic_json_dump(public_completion, public / PUBLIC_COMPLETION_NAME)

        mapping: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "role": "PRIVATE_DO_NOT_GIVE_TO_REVIEWERS",
            "public_manifest_identity_sha256": public_identity,
            "public_manifest_file_sha256": sha256_file(public / PUBLIC_MANIFEST_NAME),
            "public_completion_file_sha256": sha256_file(
                public / PUBLIC_COMPLETION_NAME
            ),
            "protocol_identity_sha256": validated.protocol[
                "protocol_identity_sha256"
            ],
            "protocol_file_sha256": sha256_file(PROTOCOL_SOURCE),
            "sampling_runner_sha256": sha256_file(SHARD_RUNNER),
            "blind_pack_builder_sha256": sha256_file(RUNNER),
            "external_visual_anchor_binding": anchor_binding,
            "blind_mapping_commitment_identity_sha256": commitment[
                "commitment_identity_sha256"
            ],
            "blind_mapping_commitment_file_sha256": sha256_file(
                private / PRIVATE_COMMITMENT_NAME
            ),
            "shards": list(validated.shard_records),
            "entries": [
                {
                    "public_blind_id": item.public_id,
                    "random_order_key": item.order_key,
                    "global_index": item.source.global_index,
                    "shard_index": item.source.shard_index,
                    "local_index": item.source.local_index,
                    "runner_blind_id": item.source.runner_id,
                    "source_png_sha256": item.source.source_file_sha256,
                    "source_pixel_sha256": item.source.pixel_sha256,
                    "public_png_sha256": image_records[position]["image"]["sha256"],
                    "public_pixel_sha256": image_records[position]["image"][
                        "pixel_sha256"
                    ],
                }
                for position, item in enumerate(blinded)
            ],
        }
        mapping["identity_sha256"] = _canonical_self_hash(
            mapping, "identity_sha256"
        )
        atomic_json_dump(mapping, private / PRIVATE_MAPPING_NAME)
        private_completion: dict[str, Any] = {
            "complete": True,
            "mapping_identity_sha256": mapping["identity_sha256"],
            "mapping_file_sha256": sha256_file(private / PRIVATE_MAPPING_NAME),
            "mapping_commitment_identity_sha256": commitment[
                "commitment_identity_sha256"
            ],
            "mapping_commitment_file_sha256": sha256_file(
                private / PRIVATE_COMMITMENT_NAME
            ),
            "public_manifest_identity_sha256": public_identity,
            "entry_count": TOTAL_POOL_BRANCHES,
        }
        private_completion["payload_sha256"] = _canonical_self_hash(
            private_completion, "payload_sha256"
        )
        atomic_json_dump(private_completion, private / PRIVATE_COMPLETION_NAME)

        top_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "role": "SEALED_PUBLIC_PRIVATE_PAIR",
            "reviewer_delivery_relative_path": PUBLIC_DIR_NAME,
            "private_seal_relative_path": PRIVATE_DIR_NAME,
            "public_manifest_identity_sha256": public_identity,
            "public_manifest_file_sha256": sha256_file(public / PUBLIC_MANIFEST_NAME),
            "public_completion_file_sha256": sha256_file(
                public / PUBLIC_COMPLETION_NAME
            ),
            "private_mapping_identity_sha256": mapping["identity_sha256"],
            "private_mapping_file_sha256": sha256_file(private / PRIVATE_MAPPING_NAME),
            "private_mapping_commitment_identity_sha256": commitment[
                "commitment_identity_sha256"
            ],
            "private_mapping_commitment_file_sha256": sha256_file(
                private / PRIVATE_COMMITMENT_NAME
            ),
            "private_completion_file_sha256": sha256_file(
                private / PRIVATE_COMPLETION_NAME
            ),
        }
        top_manifest["identity_sha256"] = _canonical_self_hash(
            top_manifest, "identity_sha256"
        )
        atomic_json_dump(top_manifest, staging / TOP_MANIFEST_NAME)
        top_completion: dict[str, Any] = {
            "complete": True,
            "bundle_manifest_identity_sha256": top_manifest["identity_sha256"],
            "bundle_manifest_file_sha256": sha256_file(staging / TOP_MANIFEST_NAME),
            "public_manifest_identity_sha256": public_identity,
            "private_mapping_identity_sha256": mapping["identity_sha256"],
            "private_mapping_commitment_identity_sha256": commitment[
                "commitment_identity_sha256"
            ],
        }
        top_completion["payload_sha256"] = _canonical_self_hash(
            top_completion, "payload_sha256"
        )
        atomic_json_dump(top_completion, staging / TOP_COMPLETION_NAME)
        validate_output_bundle(staging, require_protocol=require_protocol_validation)
        _atomic_install_directory_noreplace(staging, outdir)
    validate_output_bundle(outdir, require_protocol=require_protocol_validation)


def _validate_template_shape(
    payload: dict[str, Any],
    *,
    role: str,
    blind_ids: tuple[str, ...],
    public_identity: str,
    rubric_identity: str,
) -> None:
    adjudication = role == "consensus_adjudication"
    expected = (
        _adjudication_template(blind_ids, public_identity, rubric_identity)
        if adjudication
        else _review_template(role, blind_ids, public_identity, rubric_identity)
    )
    if payload != expected:
        raise RuntimeError(f"{role} template differs from the exact empty schema")


def _validate_exact_reviewer_readme(path: Path) -> None:
    _require_plain_file(path)
    if path.read_bytes() != REVIEWER_README_TEXT.encode("utf-8"):
        raise RuntimeError("reviewer README differs from the fixed quality-only text")


def _validate_all_reviewer_visible_text(root: Path) -> None:
    """Reject structural lineage/evidence tokens in every public JSON/TXT file."""

    text_paths = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".json", ".txt"}
    )
    if not text_paths:
        raise RuntimeError("reviewer pack contains no reviewer-visible text files")
    for path in text_paths:
        _require_plain_file(path)
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError as exc:
            raise RuntimeError("reviewer-visible text is not valid UTF-8") from exc
        for fragment in PUBLIC_ALL_TEXT_FORBIDDEN_FRAGMENTS:
            if fragment in text:
                relative = path.relative_to(root).as_posix()
                raise RuntimeError(
                    f"reviewer-visible text leaks {fragment} in {relative}"
                )


def validate_public_pack(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _reject_special_entries(root)
    manifest = _read_self_hashed_json(root / PUBLIC_MANIFEST_NAME, "public_payload_sha256")
    if manifest.get("identity_sha256") != _canonical_self_hash(
        {key: value for key, value in manifest.items() if key != "templates" and key != "public_payload_sha256"},
        "identity_sha256",
    ):
        raise RuntimeError("public pre-template manifest identity changed")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "REVIEWER_ONLY_QUALITY_CONTENT",
        "image_count": TOTAL_POOL_BRANCHES,
        "randomized_blind_ids": True,
        "blind_order_depends_on_pixels_or_results": False,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("public blind-pack identity changed")
    ids_value = manifest.get("blind_ids")
    if not isinstance(ids_value, list) or len(ids_value) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("public blind-ID count changed")
    blind_ids = tuple(_validate_public_blind_id(value) for value in ids_value)
    if len(set(blind_ids)) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("public blind IDs are duplicated")
    anchor_root = root / ANCHOR_DIR_NAME
    anchor_manifest, anchor_completion = _validate_anchor_pack(anchor_root)
    anchor_binding = _public_anchor_binding_from_manifest(
        anchor_root, anchor_manifest, anchor_completion
    )
    expected_anchor = {**anchor_binding, "relative_path": ANCHOR_DIR_NAME}
    if manifest.get("external_visual_anchors") != expected_anchor:
        raise RuntimeError("public pack external-anchor binding changed")
    rubric = _read_self_hashed_json(root / RUBRIC_NAME, "rubric_identity_sha256")
    if (
        rubric.get("rubric_name")
        != "dit_class207_cross_prefix_external_anchor_quality_v1"
        or rubric.get("annotation_row_schema") != ANNOTATION_ROW_SCHEMA
        or rubric.get("external_anchor_manifest_identity_sha256")
        != anchor_manifest.get("identity_sha256")
        or rubric.get("external_anchor_pack_payload_sha256")
        != anchor_manifest.get("pack_payload_sha256")
    ):
        raise RuntimeError("public quality rubric changed")
    if manifest.get("rubric") != _plain_file_record(root / RUBRIC_NAME, root):
        raise RuntimeError("public rubric file binding changed")
    _validate_exact_reviewer_readme(root / README_NAME)
    if manifest.get("readme") != _plain_file_record(root / README_NAME, root):
        raise RuntimeError("public README file binding changed")

    templates = manifest.get("templates")
    template_names = {
        REVIEW_A_TEMPLATE_NAME,
        REVIEW_B_TEMPLATE_NAME,
        ADJUDICATION_TEMPLATE_NAME,
    }
    if not isinstance(templates, dict) or set(templates) != template_names:
        raise RuntimeError("public annotation template set changed")
    for name in template_names:
        if templates[name] != _plain_file_record(root / name, root):
            raise RuntimeError("public annotation template file binding changed")
    _validate_template_shape(
        load_json(root / REVIEW_A_TEMPLATE_NAME),
        role="reviewer_A",
        blind_ids=blind_ids,
        public_identity=manifest["identity_sha256"],
        rubric_identity=rubric["rubric_identity_sha256"],
    )
    _validate_template_shape(
        load_json(root / REVIEW_B_TEMPLATE_NAME),
        role="reviewer_B",
        blind_ids=blind_ids,
        public_identity=manifest["identity_sha256"],
        rubric_identity=rubric["rubric_identity_sha256"],
    )
    _validate_template_shape(
        load_json(root / ADJUDICATION_TEMPLATE_NAME),
        role="consensus_adjudication",
        blind_ids=blind_ids,
        public_identity=manifest["identity_sha256"],
        rubric_identity=rubric["rubric_identity_sha256"],
    )

    records = manifest.get("images")
    if not isinstance(records, list) or len(records) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("public endpoint image-record count changed")
    contact_inputs: list[tuple[str, Path]] = []
    pixel_hashes: set[str] = set()
    file_hashes: set[str] = set()
    for position, (identifier, record) in enumerate(zip(blind_ids, records)):
        path = root / IMAGE_DIR_NAME / f"{identifier}.png"
        _validate_metadata_free_png(path)
        expected = {
            "review_position": position,
            "blind_id": identifier,
            "image": _image_record(path, root, (IMAGE_SIZE, IMAGE_SIZE)),
        }
        if record != expected:
            raise RuntimeError("public endpoint record changed")
        pixel_hashes.add(expected["image"]["pixel_sha256"])
        file_hashes.add(expected["image"]["sha256"])
        contact_inputs.append((identifier, path))
    if len(pixel_hashes) != TOTAL_POOL_BRANCHES or len(file_hashes) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("public pack contains duplicate endpoint images")
    contact_path = root / CONTACT_NAME
    _validate_metadata_free_png(contact_path)
    if manifest.get("contact_sheet") != _image_record(
        contact_path, root, _contact_size()
    ):
        raise RuntimeError("contact-sheet record changed")
    expected_contact = render_contact_sheet(tuple(contact_inputs))
    with Image.open(contact_path) as observed:
        observed.load()
        if observed.mode != "RGB" or observed.size != _contact_size():
            raise RuntimeError("contact-sheet geometry changed")
        if observed.tobytes() != expected_contact.tobytes():
            raise RuntimeError("contact sheet does not reconstruct from public images")

    completion = _read_self_hashed_json(root / PUBLIC_COMPLETION_NAME, "payload_sha256")
    expected_completion = {
        "complete": True,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "public_payload_sha256": manifest["public_payload_sha256"],
        "manifest_file_sha256": sha256_file(root / PUBLIC_MANIFEST_NAME),
        "image_count": TOTAL_POOL_BRANCHES,
        "payload_sha256": completion.get("payload_sha256"),
    }
    if completion != expected_completion:
        raise RuntimeError("public completion binding changed")
    expected_files = {
        (root / PUBLIC_MANIFEST_NAME).resolve(),
        (root / PUBLIC_COMPLETION_NAME).resolve(),
        (root / RUBRIC_NAME).resolve(),
        (root / README_NAME).resolve(),
        (root / REVIEW_A_TEMPLATE_NAME).resolve(),
        (root / REVIEW_B_TEMPLATE_NAME).resolve(),
        (root / ADJUDICATION_TEMPLATE_NAME).resolve(),
        (root / CONTACT_NAME).resolve(),
        *{
            (root / IMAGE_DIR_NAME / f"{identifier}.png").resolve()
            for identifier in blind_ids
        },
        *{
            path.resolve()
            for path in anchor_root.rglob("*")
            if path.is_file()
        },
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("public reviewer pack is not a closed file set")
    expected_dirs = {
        (root / IMAGE_DIR_NAME).resolve(),
        anchor_root.resolve(),
        *{path.resolve() for path in anchor_root.rglob("*") if path.is_dir()},
    }
    actual_dirs = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise RuntimeError("public reviewer pack is not a closed directory set")
    public_text = json.dumps(
        {"manifest": manifest, "completion": completion, "rubric": rubric},
        ensure_ascii=False,
        sort_keys=True,
    ).lower()
    if any(fragment in public_text for fragment in PUBLIC_FORBIDDEN_FRAGMENTS):
        raise RuntimeError("reviewer-visible JSON leaks lineage or evidence")
    _validate_all_reviewer_visible_text(root)
    return manifest, completion


def validate_output_bundle(
    root: Path, *, require_protocol: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    _reject_special_entries(root)
    public = root / PUBLIC_DIR_NAME
    private = root / PRIVATE_DIR_NAME
    public_manifest, public_completion = validate_public_pack(public)
    mapping = _read_self_hashed_json(
        private / PRIVATE_MAPPING_NAME, "identity_sha256"
    )
    expected_mapping_keys = {
        "schema_version",
        "experiment",
        "role",
        "public_manifest_identity_sha256",
        "public_manifest_file_sha256",
        "public_completion_file_sha256",
        "protocol_identity_sha256",
        "protocol_file_sha256",
        "sampling_runner_sha256",
        "blind_pack_builder_sha256",
        "external_visual_anchor_binding",
        "blind_mapping_commitment_identity_sha256",
        "blind_mapping_commitment_file_sha256",
        "shards",
        "entries",
        "identity_sha256",
    }
    if not isinstance(mapping, dict) or set(mapping) != expected_mapping_keys:
        raise RuntimeError("private blind-mapping schema changed")
    fixed_mapping_bindings = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "PRIVATE_DO_NOT_GIVE_TO_REVIEWERS",
        "public_manifest_identity_sha256": public_manifest["identity_sha256"],
        "public_manifest_file_sha256": sha256_file(public / PUBLIC_MANIFEST_NAME),
        "public_completion_file_sha256": sha256_file(public / PUBLIC_COMPLETION_NAME),
        "sampling_runner_sha256": sha256_file(SHARD_RUNNER),
        "blind_pack_builder_sha256": sha256_file(RUNNER),
    }
    for key, expected in fixed_mapping_bindings.items():
        if mapping.get(key) != expected:
            raise RuntimeError(f"private mapping source/public binding changed: {key}")
    commitment_path = private / PRIVATE_COMMITMENT_NAME
    commitment = validate_mapping_commitment(
        commitment_path,
        _read_self_hashed_json(PROTOCOL_SOURCE, "protocol_identity_sha256")
        if require_protocol
        else None,
        enforce_protocol_path=False,
    )
    if (
        mapping["blind_mapping_commitment_identity_sha256"]
        != commitment["commitment_identity_sha256"]
        or mapping["blind_mapping_commitment_file_sha256"]
        != sha256_file(commitment_path)
    ):
        raise RuntimeError("private mapping differs from its pre-GPU commitment")
    commitment_by_index = {
        entry["global_index"]: entry for entry in commitment["entries"]
    }
    entries = mapping.get("entries")
    if not isinstance(entries, list) or len(entries) != TOTAL_POOL_BRANCHES:
        raise RuntimeError("private mapping must contain exactly 64 entries")
    public_records = {
        record["blind_id"]: record for record in public_manifest["images"]
    }
    seen_global: set[int] = set()
    for position, entry in enumerate(entries):
        expected_keys = {
            "public_blind_id",
            "random_order_key",
            "global_index",
            "shard_index",
            "local_index",
            "runner_blind_id",
            "source_png_sha256",
            "source_pixel_sha256",
            "public_png_sha256",
            "public_pixel_sha256",
        }
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise RuntimeError("private mapping entry schema changed")
        global_index = entry["global_index"]
        if type(global_index) is not int or global_index not in range(TOTAL_POOL_BRANCHES):
            raise RuntimeError("private mapping global index is invalid")
        if global_index in seen_global:
            raise RuntimeError("private mapping duplicates a global index")
        seen_global.add(global_index)
        frozen_entry = commitment_by_index[global_index]
        expected_id = frozen_entry["public_blind_id"]
        expected_order = frozen_entry["random_order_key"]
        if (
            entry["public_blind_id"] != expected_id
            or entry["random_order_key"] != expected_order
            or entry["runner_blind_id"] != runner_blind_id(global_index)
            or entry["shard_index"] != global_index // BRANCHES_PER_SHARD
            or entry["local_index"] != global_index % BRANCHES_PER_SHARD
        ):
            raise RuntimeError("private randomized mapping does not reconstruct")
        public_record = public_records.get(expected_id)
        if public_record is None or public_record["review_position"] != position:
            raise RuntimeError("private mapping/public order join changed")
        image = public_record["image"]
        if (
            entry["public_png_sha256"] != image["sha256"]
            or entry["public_pixel_sha256"] != image["pixel_sha256"]
            or entry["source_pixel_sha256"] != image["pixel_sha256"]
        ):
            raise RuntimeError("private mapping/public endpoint pixels changed")
    if seen_global != set(range(TOTAL_POOL_BRANCHES)):
        raise RuntimeError("private mapping does not cover exactly indices 0..63")
    if [entry["random_order_key"] for entry in entries] != sorted(
        entry["random_order_key"] for entry in entries
    ):
        raise RuntimeError("private mapping is not in randomized review order")
    shards = mapping.get("shards")
    if not isinstance(shards, list) or len(shards) != TOTAL_SHARDS:
        raise RuntimeError("private mapping shard record count changed")
    if [record.get("shard_index") for record in shards] != list(range(TOTAL_SHARDS)):
        raise RuntimeError("private mapping shard records are not 0..7")
    for record in shards:
        if any(
            not _is_sha256(value)
            for key, value in record.items()
            if key != "shard_index"
        ):
            raise RuntimeError("private shard binding contains malformed hashes")

    if require_protocol:
        protocol = _read_self_hashed_json(PROTOCOL_SOURCE, "protocol_identity_sha256")
        _validate_protocol_for_pipeline(protocol)
        if (
            mapping["protocol_identity_sha256"]
            != protocol["protocol_identity_sha256"]
            or mapping["protocol_file_sha256"] != sha256_file(PROTOCOL_SOURCE)
            or mapping["external_visual_anchor_binding"]
            != protocol["external_visual_anchor_binding"][
                "metadata_stripped_anchor_pack"
            ]
        ):
            raise RuntimeError("private mapping differs from frozen local protocol")

    private_completion = _read_self_hashed_json(
        private / PRIVATE_COMPLETION_NAME, "payload_sha256"
    )
    expected_private_completion = {
        "complete": True,
        "mapping_identity_sha256": mapping["identity_sha256"],
        "mapping_file_sha256": sha256_file(private / PRIVATE_MAPPING_NAME),
        "mapping_commitment_identity_sha256": commitment[
            "commitment_identity_sha256"
        ],
        "mapping_commitment_file_sha256": sha256_file(commitment_path),
        "public_manifest_identity_sha256": public_manifest["identity_sha256"],
        "entry_count": TOTAL_POOL_BRANCHES,
        "payload_sha256": private_completion.get("payload_sha256"),
    }
    if private_completion != expected_private_completion:
        raise RuntimeError("private completion binding changed")

    top_manifest = _read_self_hashed_json(root / TOP_MANIFEST_NAME, "identity_sha256")
    expected_top_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "SEALED_PUBLIC_PRIVATE_PAIR",
        "reviewer_delivery_relative_path": PUBLIC_DIR_NAME,
        "private_seal_relative_path": PRIVATE_DIR_NAME,
        "public_manifest_identity_sha256": public_manifest["identity_sha256"],
        "public_manifest_file_sha256": sha256_file(public / PUBLIC_MANIFEST_NAME),
        "public_completion_file_sha256": sha256_file(public / PUBLIC_COMPLETION_NAME),
        "private_mapping_identity_sha256": mapping["identity_sha256"],
        "private_mapping_file_sha256": sha256_file(private / PRIVATE_MAPPING_NAME),
        "private_mapping_commitment_identity_sha256": commitment[
            "commitment_identity_sha256"
        ],
        "private_mapping_commitment_file_sha256": sha256_file(commitment_path),
        "private_completion_file_sha256": sha256_file(
            private / PRIVATE_COMPLETION_NAME
        ),
        "identity_sha256": top_manifest.get("identity_sha256"),
    }
    if top_manifest != expected_top_manifest:
        raise RuntimeError("top-level sealed-pair manifest changed")
    top_completion = _read_self_hashed_json(
        root / TOP_COMPLETION_NAME, "payload_sha256"
    )
    expected_top_completion = {
        "complete": True,
        "bundle_manifest_identity_sha256": top_manifest["identity_sha256"],
        "bundle_manifest_file_sha256": sha256_file(root / TOP_MANIFEST_NAME),
        "public_manifest_identity_sha256": public_manifest["identity_sha256"],
        "private_mapping_identity_sha256": mapping["identity_sha256"],
        "private_mapping_commitment_identity_sha256": commitment[
            "commitment_identity_sha256"
        ],
        "payload_sha256": top_completion.get("payload_sha256"),
    }
    if top_completion != expected_top_completion:
        raise RuntimeError("top-level sealed-pair completion changed")
    expected_files = {
        (root / TOP_MANIFEST_NAME).resolve(),
        (root / TOP_COMPLETION_NAME).resolve(),
        (private / PRIVATE_MAPPING_NAME).resolve(),
        commitment_path.resolve(),
        (private / PRIVATE_COMPLETION_NAME).resolve(),
        *{path.resolve() for path in public.rglob("*") if path.is_file()},
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("sealed blind bundle is not a closed file set")
    expected_directories = {
        public.resolve(),
        private.resolve(),
        *{path.resolve() for path in public.rglob("*") if path.is_dir()},
    }
    actual_directories = {
        path.resolve() for path in root.rglob("*") if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise RuntimeError("sealed blind bundle is not a closed directory set")
    return top_manifest, top_completion


def _toy_image(path: Path, index: int) -> None:
    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        ((index * 41) % 256, (index * 73) % 256, (index * 109) % 256),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 110, 110), outline=(255, 255, 255), width=3)
    draw.text((24, 48), f"{index:02d}", fill=(255, 255, 255), font=_font())
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private_source_metadata", f"trajectory-{index}")
    image.save(path, format="PNG", optimize=False, pnginfo=metadata)


def _expect_failure(operation: Callable[[], Any], label: str) -> None:
    try:
        operation()
    except (RuntimeError, FileExistsError):
        return
    raise AssertionError(f"negative self-test did not fail: {label}")


def run_self_test() -> None:
    """CPU-only structural tests; no real endpoint or private trace is opened."""

    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must begin before CUDA initialization")
    # Full integration uses the real runner and anchor validators.  Here we test
    # all local transformations with deterministic synthetic records so DRAFT
    # status cannot accidentally authorize or inspect real artifacts.
    secret = bytes(range(32))
    commitment = build_mapping_commitment(secret)
    sources: list[SourceImage] = []
    with tempfile.TemporaryDirectory(prefix="cross-prefix-blind-selftest-") as temporary:
        root = Path(temporary)
        source_root = root / "sources"
        source_root.mkdir()
        for index in range(TOTAL_POOL_BRANCHES):
            path = source_root / f"source_{index:02d}.png"
            _toy_image(path, index)
            inspection = inspect_png(path, "RGB", (IMAGE_SIZE, IMAGE_SIZE))
            sources.append(
                SourceImage(
                    global_index=index,
                    shard_index=index // BRANCHES_PER_SHARD,
                    local_index=index % BRANCHES_PER_SHARD,
                    runner_id=runner_blind_id(index),
                    source_path=path,
                    source_file_sha256=inspection["sha256"],
                    pixel_sha256=inspection["pixel_sha256"],
                )
            )
        blinded = _blinded_images(tuple(sources), commitment)
        if [item.source.global_index for item in blinded] == list(
            range(TOTAL_POOL_BRANCHES)
        ):
            raise AssertionError("synthetic randomized order unexpectedly stayed sequential")
        if any(
            item.public_id == item.source.runner_id or item.source.runner_id in item.public_id
            for item in blinded
        ):
            raise AssertionError("public IDs leaked runner IDs")
        if _blinded_images(tuple(sources), commitment) != blinded:
            raise AssertionError("fixed-secret mapping is not reproducible")
        if _blinded_images(
            tuple(sources), build_mapping_commitment(bytes(reversed(secret)))
        ) == blinded:
            raise AssertionError("different blind secrets produced the same mapping")
        commitment_probe = root / "commitment_probe.json"
        atomic_json_dump(commitment, commitment_probe)
        validate_mapping_commitment(commitment_probe)
        tampered_commitment = json.loads(json.dumps(commitment))
        tampered_commitment["entries"][0]["public_blind_id"] = (
            "xr1_ffffffffffffffff"
        )
        tampered_commitment["commitment_identity_sha256"] = _canonical_self_hash(
            tampered_commitment, "commitment_identity_sha256"
        )
        atomic_json_dump(tampered_commitment, root / "tampered_commitment.json")
        _expect_failure(
            lambda: validate_mapping_commitment(root / "tampered_commitment.json"),
            "pre-GPU mapping mutation",
        )
        frozen_probe = root / "frozen_mapping_probe.json"
        freeze_mapping_commitment(frozen_probe)
        validate_mapping_commitment(frozen_probe)
        frozen_probe_hash = sha256_file(frozen_probe)
        _expect_failure(
            lambda: freeze_mapping_commitment(frozen_probe),
            "mapping commitment no-overwrite",
        )
        if sha256_file(frozen_probe) != frozen_probe_hash:
            raise AssertionError("mapping no-overwrite changed the frozen commitment")

        canonical_dir = root / "canonical"
        canonical_dir.mkdir()
        record = _clean_reencode(sources[0].source_path, canonical_dir / "clean.png")
        if record["pixel_sha256"] != sources[0].pixel_sha256:
            raise AssertionError("metadata-free re-encoding changed pixels")
        _validate_metadata_free_png(canonical_dir / "clean.png")
        injected = PngImagePlugin.PngInfo()
        injected.add_text("hidden_lineage", "sentinel")
        with Image.open(canonical_dir / "clean.png") as observed:
            observed.load()
            copy = Image.frombytes("RGB", observed.size, observed.tobytes())
        copy.save(canonical_dir / "clean.png", format="PNG", pnginfo=injected)
        _expect_failure(
            lambda: _validate_metadata_free_png(canonical_dir / "clean.png"),
            "PNG metadata injection",
        )

        sample = {
            "blind_id": blinded[0].public_id,
            "primary_overall_structural_quality": "not_clear_overall_structural_bad",
            "secondary_hind_limb_topology": "not_clear_failure",
            "tail_R_root_attachment": 0,
            "tail_T_taper_and_volume": 1,
            "tail_F_feather_or_hair_flow": 0,
            "tail_D_distal_tip": 0,
            "tail_P_paddle_like": 0,
            "tail_B_short_or_blunt": 0,
            "tail_S_abrupt_filament_transition": 0,
            "tail_identity": "clear",
            "tail_scorable": "yes",
            "tail_confidence": "high",
            "tail_derived_label": "odd",
            "notes": "slightly strange taper but coherent fluffy hair flow",
        }
        validate_completed_annotation_row(sample, context="selftest")
        bad = dict(sample)
        bad["tail_derived_label"] = "natural"
        _expect_failure(
            lambda: validate_completed_annotation_row(bad, context="selftest"),
            "tail identity/naturalness contradiction",
        )
        bad = dict(sample)
        bad["notes"] = "alarm rank looked high"
        _expect_failure(
            lambda: validate_completed_annotation_row(bad, context="selftest"),
            "annotation evidence leakage",
        )
        if any(
            fragment in json.dumps(ANNOTATION_ROW_SCHEMA).lower()
            for fragment in ("seed", "shard", "delta_nu", "e_mix")
        ):
            raise AssertionError("public annotation schema leaks private configuration")

        readme_probe = root / "reviewer_readme_probe.txt"
        readme_probe.write_text(REVIEWER_README_TEXT, encoding="utf-8")
        _validate_exact_reviewer_readme(readme_probe)
        readme_probe.write_text(
            REVIEWER_README_TEXT + "hidden branch_seed lineage\n", encoding="utf-8"
        )
        _expect_failure(
            lambda: _validate_exact_reviewer_readme(readme_probe),
            "reviewer README mutation",
        )
        text_probe = root / "reviewer_text_probe"
        text_probe.mkdir()
        (text_probe / "safe.json").write_text(
            '{"purpose":"visible quality only"}\n', encoding="utf-8"
        )
        _validate_all_reviewer_visible_text(text_probe)
        (text_probe / "leak.txt").write_text(
            "branch_seed=123\n", encoding="utf-8"
        )
        _expect_failure(
            lambda: _validate_all_reviewer_visible_text(text_probe),
            "reviewer-visible all-text lineage injection",
        )

        # Full synthetic 8x8 input -> sealed public/private bundle.  The real
        # validators are replaced only inside this CPU fixture; all local
        # allocation, hash, randomization, PNG, closed-set, and leak checks run.
        try:
            from . import build_dit_class207_visual_anchor_pack as anchor_module
        except ImportError:  # direct CLI execution
            import build_dit_class207_visual_anchor_pack as anchor_module

        anchor_inputs = root / "synthetic_anchor_inputs"
        anchor_inputs.mkdir()
        anchor_config = anchor_module._make_synthetic_config(anchor_inputs)
        synthetic_anchor_config_path = root / "synthetic_anchor_config.json"
        atomic_json_dump(anchor_config, synthetic_anchor_config_path)
        original_anchor_config_source = globals()["ANCHOR_CONFIG_SOURCE"]
        globals()["ANCHOR_CONFIG_SOURCE"] = synthetic_anchor_config_path
        anchor_validated = anchor_module.validate_sources(anchor_config)
        if len(anchor_validated) != 7:
            raise AssertionError("synthetic anchor fixture count changed")
        anchor_pack = root / "synthetic_anchor_pack"
        anchor_manifest, anchor_completion = anchor_module._write_bundle(
            anchor_pack, anchor_config, None
        )

        anchor_binding = _anchor_binding_from_manifest(
            anchor_pack, anchor_manifest, anchor_completion
        )
        protocol = load_json(PROTOCOL_SOURCE)
        protocol["protocol_status"] = "FROZEN_BEFORE_GPU_EXECUTION"
        protocol["authorization_gate"]["gpu_execution_authorized"] = True
        protocol["blind_pipeline_binding"] = _pipeline_binding_expected()
        commitment_path = root / "synthetic_mapping_commitment.json"
        atomic_json_dump(commitment, commitment_path)
        protocol["blind_mapping_commitment_binding"] = {
            "status": MAPPING_COMMITMENT_STATUS,
            "commitment_schema": MAPPING_COMMITMENT_SCHEMA,
            "pool_size": TOTAL_POOL_BRANCHES,
            "commitment_path": str(commitment_path.resolve()),
            "mapping_builder_filename": RUNNER.name,
            "mapping_builder_sha256": sha256_file(RUNNER),
            "commitment_identity_sha256": commitment[
                "commitment_identity_sha256"
            ],
            "commitment_file_sha256": sha256_file(commitment_path),
        }
        protocol["external_visual_anchor_binding"][
            "metadata_stripped_anchor_pack"
        ] = anchor_binding
        protocol["protocol_identity_sha256"] = _canonical_self_hash(
            protocol, "protocol_identity_sha256"
        )
        protocol_path = root / "synthetic_protocol.json"
        atomic_json_dump(protocol, protocol_path)

        shard_roots: list[Path] = []
        fake_pairs: dict[Path, tuple[dict[str, Any], dict[str, Any]]] = {}
        for shard_index in range(TOTAL_SHARDS):
            shard_root = root / f"synthetic_shard_{shard_index}"
            image_root = shard_root / "blind_images"
            image_root.mkdir(parents=True)
            atomic_json_dump(protocol, shard_root / PROTOCOL_COPY_NAME)
            branch_records: list[dict[str, Any]] = []
            for local_index, global_index in enumerate(
                shard_global_indices(shard_index)
            ):
                source_item = sources[global_index]
                endpoint = image_root / f"{runner_blind_id(global_index)}.png"
                endpoint.write_bytes(source_item.source_path.read_bytes())
                image_record = _image_record(
                    endpoint, shard_root, (IMAGE_SIZE, IMAGE_SIZE)
                )
                branch_records.append(
                    {
                        "local_index": local_index,
                        "global_index": global_index,
                        "blind_id": runner_blind_id(global_index),
                        "image": image_record,
                    }
                )
            manifest = {
                "experiment": SHARD_EXPERIMENT,
                "identity_sha256": hashlib.sha256(
                    f"synthetic-manifest-{shard_index}".encode()
                ).hexdigest(),
                "pool": {
                    "this_shard_index": shard_index,
                    "this_shard_global_branch_indices": list(
                        shard_global_indices(shard_index)
                    ),
                },
            }
            results = {
                "payload_sha256": hashlib.sha256(
                    f"synthetic-results-{shard_index}".encode()
                ).hexdigest(),
                "private_trace": {"sha256": hashlib.sha256(
                    f"synthetic-trace-{shard_index}".encode()
                ).hexdigest()},
                "branch_records": branch_records,
            }
            atomic_json_dump(manifest, shard_root / "manifest.json")
            atomic_json_dump(results, shard_root / "results.json")
            (shard_root / "completion.json").write_text(
                f"synthetic-completion-{shard_index}\n", encoding="utf-8"
            )
            fake_pairs[shard_root.resolve()] = (manifest, results)
            shard_roots.append(shard_root)

        original_anchor_validator = globals()["_validate_anchor_pack"]
        original_shard_validator = globals()["validate_shard_bundle"]
        original_pool_validator = globals()["validate_shard_pool"]
        original_protocol_source = globals()["PROTOCOL_SOURCE"]

        def fake_anchor_validator(
            value: Path,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return anchor_module.validate_output_bundle(
                value, anchor_config["anchor_config_identity_sha256"]
            )

        def fake_shard_validator(
            value: Path,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            print("PRIVATE_SYNTHETIC_VALIDATOR_SENTINEL")
            return fake_pairs[value.resolve()]

        def fake_pool_validator(values: Iterable[Path]) -> dict[str, Any]:
            if {value.resolve() for value in values} != set(fake_pairs):
                raise RuntimeError("synthetic pool coverage changed")
            return {
                "status": "valid-complete-pool",
                "protocol_identity_sha256": protocol[
                    "protocol_identity_sha256"
                ],
                "runner_sha256": sha256_file(SHARD_RUNNER),
            }

        globals()["_validate_anchor_pack"] = fake_anchor_validator
        globals()["validate_shard_bundle"] = fake_shard_validator
        globals()["validate_shard_pool"] = fake_pool_validator
        globals()["PROTOCOL_SOURCE"] = protocol_path
        try:
            validated = validate_input_shards(tuple(shard_roots), anchor_pack)
            if len(validated.images) != TOTAL_POOL_BRANCHES:
                raise AssertionError("synthetic exact 8x8 validation failed")
            _expect_failure(
                lambda: validate_input_shards(tuple(shard_roots[:-1]), anchor_pack),
                "missing eighth shard",
            )
            blind_bundle = root / "synthetic_blind_bundle"
            _write_bundle(
                blind_bundle,
                anchor_pack,
                commitment_path,
                validated,
                require_protocol_validation=True,
            )
            top_manifest, _ = validate_output_bundle(
                blind_bundle, require_protocol=True
            )
            public_manifest, _ = validate_public_pack(
                blind_bundle / PUBLIC_DIR_NAME
            )
            private_mapping = _read_self_hashed_json(
                blind_bundle / PRIVATE_DIR_NAME / PRIVATE_MAPPING_NAME,
                "identity_sha256",
            )
            if top_manifest["public_manifest_identity_sha256"] != public_manifest[
                "identity_sha256"
            ]:
                raise AssertionError("synthetic public/private top binding failed")
            public_serialized = json.dumps(public_manifest, sort_keys=True).lower()
            if any(
                value in public_serialized
                for value in (
                    "global_index",
                    "runner_blind_id",
                    "stream_seed",
                    "blinding_secret_hex",
                    commitment["commitment_identity_sha256"],
                )
            ):
                raise AssertionError("synthetic reviewer pack leaked lineage")
            if len(private_mapping["entries"]) != TOTAL_POOL_BRANCHES:
                raise AssertionError("synthetic private mapping count changed")
            first_public = (
                blind_bundle
                / PUBLIC_DIR_NAME
                / public_manifest["images"][0]["image"]["relative_path"]
            )
            original_bytes = first_public.read_bytes()
            first_public.write_bytes(
                blind_bundle.joinpath(
                    PUBLIC_DIR_NAME,
                    public_manifest["images"][1]["image"]["relative_path"],
                ).read_bytes()
            )
            _expect_failure(
                lambda: validate_public_pack(blind_bundle / PUBLIC_DIR_NAME),
                "public endpoint mutation/duplication",
            )
            first_public.write_bytes(original_bytes)
            validate_output_bundle(blind_bundle, require_protocol=True)
        finally:
            globals()["_validate_anchor_pack"] = original_anchor_validator
            globals()["validate_shard_bundle"] = original_shard_validator
            globals()["validate_shard_pool"] = original_pool_validator
            globals()["PROTOCOL_SOURCE"] = original_protocol_source
            globals()["ANCHOR_CONFIG_SOURCE"] = original_anchor_config_source

        race_source = root / "race_source"
        race_target = root / "race_target"
        race_source.mkdir()
        race_target.mkdir()
        (race_source / "marker").write_text("source", encoding="utf-8")
        (race_target / "marker").write_text("target", encoding="utf-8")
        _expect_failure(
            lambda: _atomic_install_directory_noreplace(race_source, race_target),
            "atomic no-overwrite",
        )
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: CPU-only pre-GPU 64-ID mapping commitment, exact synthetic "
        "8x8 input validation, sealed public/private bundle, metadata stripping, "
        "mapping/pixel/README mutation and all-text reviewer-lineage leakage "
        "rejection, tail identity-vs-naturalness logic, closed sets, and atomic "
        "no-overwrite"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed cross-prefix shard directory; provide exactly eight.",
    )
    parser.add_argument("--anchor-pack", type=Path)
    parser.add_argument(
        "--mapping-commitment",
        type=Path,
        help="Private pre-GPU frozen 64-index randomized mapping commitment.",
    )
    parser.add_argument(
        "--freeze-mapping-output",
        type=Path,
        help=(
            "Create only the private mapping commitment before protocol/GPU freeze; "
            "no shard or image is opened."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=(
            data_root
            / "cross_scale_evidence/dit_imagenet256_t60_cross_prefix_blind_review"
            / f"pool64_sealed_blind_bundle_{sha256_file(RUNNER)[:8]}"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        return
    if args.freeze_mapping_output is not None:
        if args.shard_dir or args.anchor_pack is not None or args.mapping_commitment is not None:
            parser.error(
                "--freeze-mapping-output cannot be combined with shard/anchor/commitment inputs"
            )
        target = args.freeze_mapping_output.expanduser().absolute()
        if os.path.lexists(target):
            parser.error("--freeze-mapping-output already exists")
        args.freeze_mapping_output = target
        return
    if len(args.shard_dir) != TOTAL_SHARDS:
        parser.error("provide exactly eight --shard-dir arguments")
    roots = tuple(path.expanduser().absolute() for path in args.shard_dir)
    if any(path.is_symlink() or not path.is_dir() for path in roots):
        parser.error("each shard input must be a plain directory")
    roots = tuple(path.resolve() for path in roots)
    if len(set(roots)) != TOTAL_SHARDS:
        parser.error("the eight shard inputs must be distinct")
    if args.anchor_pack is None:
        parser.error("--anchor-pack is required")
    anchor = args.anchor_pack.expanduser().absolute()
    if anchor.is_symlink() or not anchor.is_dir():
        parser.error("--anchor-pack must be a plain directory")
    anchor = anchor.resolve()
    if args.mapping_commitment is None:
        parser.error("--mapping-commitment is required")
    commitment = args.mapping_commitment.expanduser().absolute()
    if commitment.is_symlink() or not commitment.is_file():
        parser.error("--mapping-commitment must be a plain file")
    commitment = commitment.resolve()
    requested = args.outdir.expanduser().absolute()
    if os.path.lexists(requested):
        parser.error("no-overwrite output already exists")
    for protected in (*roots, anchor, commitment, RUNNER.parent.parent):
        if _paths_overlap(requested, protected):
            parser.error("--outdir overlaps a protected input/source tree")
    args.shard_dir = roots
    args.anchor_pack = anchor
    args.mapping_commitment = commitment
    args.outdir = requested.resolve()


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_args(args, parser)
    if args.self_test:
        run_self_test()
    elif args.freeze_mapping_output is not None:
        payload = freeze_mapping_commitment(args.freeze_mapping_output)
        print(
            json.dumps(
                {
                    "status": "frozen_mapping_commitment_created",
                    "commitment_identity_sha256": payload[
                        "commitment_identity_sha256"
                    ],
                    "commitment_file_sha256": sha256_file(
                        args.freeze_mapping_output
                    ),
                    "private_do_not_give_to_reviewers": True,
                },
                sort_keys=True,
            )
        )
    else:
        validate_mapping_commitment(args.mapping_commitment)
        validated = validate_input_shards(args.shard_dir, args.anchor_pack)
        _write_bundle(
            args.outdir,
            args.anchor_pack,
            args.mapping_commitment,
            validated,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "reviewer_delivery": str(args.outdir / PUBLIC_DIR_NAME),
                    "image_count": TOTAL_POOL_BRANCHES,
                    "private_mapping_not_for_reviewers": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
