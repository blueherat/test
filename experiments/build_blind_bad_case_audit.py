#!/usr/bin/env python3
"""Build an offline, blinded human-review packet from a completed image run.

The reviewer packet contains only opaque image IDs, target class names, and
metadata-free copies of the full-resolution images.  In particular, it never
reads signal files and never exposes class IDs, seeds, ASD/evidence values, or
tail/rank membership.  The only link back to the run is written to a separate
private JSON mapping that must not be given to reviewers.

The packet is a self-contained static web interface.  It stores draft progress
in browser localStorage and can export annotations as JSON or CSV.

Confirmation protection is deliberately non-overridable in this pre-lock
version: paths or manifests marked as confirmation data, and the repository's
reserved confirmation seed range (seed >= 10,000), are rejected before any
source image is opened.  After the rubric, sampling plan, thresholds, and
reviewer/adjudication rules are frozen, confirmation access must be introduced
as an explicit audited schema revision; it must not be enabled by renaming a
directory or bypassing this guard.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


SCHEMA_VERSION = 2
CONFIRMATION_SEED_START = 10_000
ANON_HEX_LENGTH = 20

INSPECTION_CHECKS = (
    {
        "code": "global_count_pose_silhouette",
        "label": "1. Global count, pose, and silhouette",
        "description": "Check subject count, global pose/layout, and the complete outer silhouette.",
    },
    {
        "code": "primary_face_head_or_object_geometry",
        "label": "2. Face/head or primary-object geometry",
        "description": "Check face/head layout or the defining geometry of the requested object or plant.",
    },
    {
        "code": "parts_limbs_anatomy_and_symmetry",
        "label": "3. Parts, limbs, anatomy, and symmetry",
        "description": "Trace part count, origins, endpoints, joints, anatomy, and expected symmetries.",
    },
    {
        "code": "subject_subject_contacts_and_occlusion",
        "label": "4. Subject-subject contacts and occlusion",
        "description": "Inspect every overlap, shared boundary, attachment, and contact between subjects or parts.",
    },
    {
        "code": "subject_background_boundary_and_support",
        "label": "5. Subject-background boundary and support",
        "description": "Trace boundaries and verify support, contact, depth order, and separation from the background.",
    },
    {
        "code": "focus_texture_repetition_color_and_glyphs",
        "label": "6. Focus, texture, repetition, color, and glyphs",
        "description": "Look for local blur, liquefaction, over-sharpening, repeated texture, color patches, and broken glyphs.",
    },
    {
        "code": "background_secondary_objects_and_crop",
        "label": "7. Background, secondary objects, and crop",
        "description": "Inspect background objects, edge continuation, cropping, and small secondary structures.",
    },
)

INSPECTION_VALUES = (
    {"value": "clear", "label": "Checked — no defect found"},
    {"value": "defect", "label": "Defect found"},
    {"value": "unassessable", "label": "Unassessable"},
    {"value": "not_applicable", "label": "Not applicable"},
)

ASSESSABILITY_SCALE = (
    {
        "value": "sufficient",
        "label": "Sufficient",
        "description": "Primary structure is visible well enough to support a clean or bad judgment.",
    },
    {
        "value": "partial",
        "label": "Partial",
        "description": "Some primary structure can be judged, but important regions remain unresolved or occluded.",
    },
    {
        "value": "insufficient",
        "label": "Insufficient",
        "description": "The image does not support a reliable structural-cleanliness judgment.",
    },
)

ORIGIN_JUDGMENT_SCALE = (
    {
        "value": "none_observed",
        "label": "No defect observed",
        "description": "Use only with severity 0 and sufficient assessability.",
    },
    {
        "value": "model_likely",
        "label": "Model-likely defect",
        "description": "The contradiction exists in native pixels and is not explained by uniform resolution loss.",
    },
    {
        "value": "resolution_limited",
        "label": "Resolution-limited / unresolved",
        "description": "Pixel support is insufficient or the issue matches the image's uniform low-bandwidth limit.",
    },
    {
        "value": "rendering_only",
        "label": "Rendering/interpolation only",
        "description": "The issue appears only in the smooth enlarged view and is absent from nearest native pixels.",
    },
    {
        "value": "natural_occlusion_or_crop",
        "label": "Natural occlusion/crop plausible",
        "description": "The observation may be explained by ordinary occlusion or framing rather than generated topology.",
    },
    {
        "value": "uncertain",
        "label": "Origin uncertain",
        "description": "A visible or possible defect cannot be assigned reliably to the model or resolution/rendering.",
    },
)

# This fallback covers the repository's current smoke/pilot and the qualitative
# or density classes discussed in the baseline paper.  Any other class requires
# an explicit ImageNet class-index JSON instead of guessing a label.
KNOWN_CLASS_NAMES = {
    1: "goldfish",
    16: "bulbul",
    94: "hummingbird",
    95: "jacamar",
    207: "golden retriever",
    289: "snow leopard",
    336: "marmot",
    388: "giant panda",
    405: "airship",
    437: "beacon",
    520: "crib",
    562: "fountain",
    681: "notebook",
    701: "parachute",
    888: "viaduct",
    900: "water tower",
    936: "head cabbage",
    949: "strawberry",
}

SEVERITY_SCALE = (
    {
        "value": 0,
        "label": "0 — none observed",
        "description": "No specific structural or imaging defect is visible after full-resolution review.",
    },
    {
        "value": 1,
        "label": "1 — mild/local",
        "description": "A local, minor defect is visible but overall structure remains coherent.",
    },
    {
        "value": 2,
        "label": "2 — clear bad case",
        "description": "A clearly identifiable deformation, fusion, blur, or artifact is present.",
    },
    {
        "value": 3,
        "label": "3 — severe/systemic",
        "description": "Multiple defects or a failure of the subject's primary structure is present.",
    },
)

ARTIFACT_FLAGS = (
    {
        "code": "face_or_head_geometry",
        "label": "Face/head geometry",
        "description": "Eyes, mouth, muzzle, head, teeth, or facial layout is malformed.",
    },
    {
        "code": "limb_or_body_geometry",
        "label": "Limb/body geometry",
        "description": "Limbs, paws, hands, torso, pose, or body silhouette is incoherent.",
    },
    {
        "code": "object_or_plant_geometry",
        "label": "Object/plant geometry",
        "description": "Object shape, attachment, growth, support, or contact geometry is implausible.",
    },
    {
        "code": "object_boundary",
        "label": "Broken boundary",
        "description": "A subject or object boundary is cut, unstable, duplicated, or incomplete.",
    },
    {
        "code": "object_fusion",
        "label": "Object/part fusion",
        "description": "Parts or distinct objects merge into one another incoherently.",
    },
    {
        "code": "multi_subject_fusion",
        "label": "Multi-subject fusion",
        "description": "Multiple subjects overlap or share anatomy in an incoherent way.",
    },
    {
        "code": "subject_background_entanglement",
        "label": "Subject/background entanglement",
        "description": "The subject dissolves into, inherits from, or cannot be separated from the background.",
    },
    {
        "code": "subject_missing_or_too_small",
        "label": "Missing/tiny subject",
        "description": "The requested subject is absent, visually ambiguous, or too small to establish the class.",
    },
    {
        "code": "blur_smear_or_liquefaction",
        "label": "Blur/smear/liquefaction",
        "description": "Local regions are melted, smeared, unresolved, or inconsistently blurred.",
    },
    {
        "code": "texture_or_repetition",
        "label": "Texture/repetition",
        "description": "Plastic, over-sharp, tiled, repeated, or otherwise implausible texture is visible.",
    },
    {
        "code": "text_or_glyph_artifact",
        "label": "Text/glyph artifact",
        "description": "Corrupted letters, symbols, watermarks, or text-like marks are present.",
    },
    {
        "code": "color_or_lighting_artifact",
        "label": "Color/lighting artifact",
        "description": "A localized color patch, halo, shadow, highlight, or lighting transition is implausible.",
    },
    {
        "code": "human_anatomy",
        "label": "Human anatomy",
        "description": "A visible human face, hand, limb, or body is anatomically malformed.",
    },
    {
        "code": "missing_or_extra_parts",
        "label": "Missing/extra parts",
        "description": "A required part is absent, duplicated, or appears in an impossible count.",
    },
    {
        "code": "topology_attachment_or_contact",
        "label": "Topology/attachment/contact",
        "description": "Parts connect, terminate, overlap, or touch in a structurally impossible way.",
    },
    {
        "code": "perspective_or_support",
        "label": "Perspective/support",
        "description": "Depth, scale, perspective, gravity, or physical support is locally incoherent.",
    },
    {
        "code": "other_artifact",
        "label": "Other (describe in notes)",
        "description": "A specific defect is present but is not represented above; notes are required.",
    },
)

# Selecting any of these flags is already a clear structural bad case.  The UI
# upgrades severity to at least 2 and export validation rejects inconsistent
# lower severities.  Imaging-only flags (blur/texture/color/glyph) may remain 1
# when they are genuinely isolated and do not imply a structural contradiction.
HARD_ARTIFACT_FLAGS = frozenset(
    {
        "face_or_head_geometry",
        "limb_or_body_geometry",
        "object_or_plant_geometry",
        "object_boundary",
        "object_fusion",
        "multi_subject_fusion",
        "subject_background_entanglement",
        "subject_missing_or_too_small",
        "human_anatomy",
        "missing_or_extra_parts",
        "topology_attachment_or_contact",
        "perspective_or_support",
    }
)


class AuditBuildError(RuntimeError):
    """A user-facing validation failure while constructing a packet."""


@dataclass(frozen=True)
class SourceSample:
    class_id: int
    seed: int
    class_name: str
    image_path: Path

    @property
    def stable_key(self) -> str:
        return f"class={self.class_id};seed={self.seed}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise AuditBuildError(f"required JSON file is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditBuildError(f"invalid UTF-8 JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditBuildError(f"expected a JSON object: {path}")
    return payload, raw


def resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_within(path: Path, directory: Path) -> bool:
    try:
        resolved(path).relative_to(resolved(directory))
        return True
    except ValueError:
        return False


def path_mentions_confirmation(path: Path) -> bool:
    absolute_parts = Path(os.path.abspath(path.expanduser())).parts
    resolved_parts = resolved(path).parts
    return any("confirmation" in part.casefold() for part in (*absolute_parts, *resolved_parts))


def preflight_confirmation_path_guard(run_dir: Path, manifest_path: Path, images_dir: Path) -> None:
    """Reject obvious confirmation paths before opening even the manifest."""

    for label, path in (("run", run_dir), ("manifest", manifest_path), ("images", images_dir)):
        if path_mentions_confirmation(path):
            raise AuditBuildError(
                f"refusing to access {label} path marked as confirmation data: {path}. "
                "This guard has no override."
            )


def iter_split_markers(payload: Any, key_path: tuple[str, ...] = ()) -> Iterable[tuple[str, str]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_path = (*key_path, str(key))
            key_folded = str(key).casefold()
            if isinstance(value, str) and any(
                token in key_folded
                for token in ("protocol", "split", "phase", "role", "purpose", "partition", "dataset")
            ):
                yield ".".join(child_path), value
            yield from iter_split_markers(value, child_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from iter_split_markers(value, (*key_path, str(index)))


def validate_integer_list(payload: Any, name: str, *, minimum: int, maximum: int | None = None) -> tuple[int, ...]:
    if not isinstance(payload, list) or not payload:
        raise AuditBuildError(f"manifest {name} must be a non-empty JSON list")
    values: list[int] = []
    for value in payload:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AuditBuildError(f"manifest {name} must contain integers, found {value!r}")
        if value < minimum or (maximum is not None and value > maximum):
            interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
            raise AuditBuildError(f"manifest {name} value {value} is outside {interval}")
        values.append(value)
    if len(values) != len(set(values)):
        raise AuditBuildError(f"manifest {name} contains duplicate values")
    return tuple(values)


def validate_discovery_manifest(manifest: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    for key, value in iter_split_markers(manifest):
        if "confirmation" in value.casefold():
            raise AuditBuildError(
                f"refusing manifest marked as confirmation data at {key}={value!r}. "
                "This guard has no override."
            )

    class_ids = validate_integer_list(manifest.get("class_ids"), "class_ids", minimum=0, maximum=999)
    seeds = validate_integer_list(manifest.get("seeds"), "seeds", minimum=0)
    protected = [seed for seed in seeds if seed >= CONFIRMATION_SEED_START]
    if protected:
        raise AuditBuildError(
            "refusing the repository's reserved confirmation seed range "
            f"(seed >= {CONFIRMATION_SEED_START}); first protected seed={min(protected)}. "
            "This guard has no override."
        )

    expected_count = len(class_ids) * len(seeds)
    if expected_count < 2:
        raise AuditBuildError(
            "blind audit packets require at least two images so severity 0/1 labels can "
            "be left and revisited for the mandatory second review"
        )
    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise AuditBuildError("manifest sample_count must be an integer")
    if sample_count != expected_count:
        raise AuditBuildError(
            f"manifest sample_count mismatch: {sample_count} != "
            f"{len(class_ids)} classes x {len(seeds)} seeds ({expected_count})"
        )
    return class_ids, seeds


def parse_class_index_payload(payload: Any, source: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise AuditBuildError(f"unsupported class-index structure in {source}")

    for raw_index, raw_value in iterable:
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise AuditBuildError(f"non-integer class-index key {raw_index!r} in {source}") from exc
        if isinstance(raw_value, str):
            name = raw_value
        elif isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
            name = raw_value[1]
        elif isinstance(raw_value, dict):
            name = raw_value.get("name") or raw_value.get("label") or raw_value.get("class_name")
        else:
            name = None
        if not isinstance(name, str) or not name.strip():
            raise AuditBuildError(f"missing class name for index {index} in {source}")
        mapping[index] = name.strip()
    return mapping


def load_class_names(
    manifest: Mapping[str, Any],
    class_ids: Sequence[int],
    explicit_path: Path | None,
    run_dir: Path,
) -> tuple[dict[int, str], str]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    else:
        candidates.extend((run_dir / "imagenet_class_index.json", run_dir / "class_index.json"))
        data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
        candidates.append(
            data_root
            / "baselines"
            / "fkc-diffusion"
            / "applications"
            / "images"
            / "edm2"
            / "imagenet_class_index.json"
        )

    mapping: dict[int, str] = {}
    source = ""
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AuditBuildError(f"invalid class-index JSON in {candidate}: {exc}") from exc
            mapping.update(parse_class_index_payload(payload, str(candidate)))
            source = str(candidate.resolve())
            break
    if explicit_path is not None and not mapping:
        raise AuditBuildError(f"explicit class-index JSON does not exist: {explicit_path}")

    manifest_names = manifest.get("class_names")
    if manifest_names is not None:
        mapping.update(parse_class_index_payload(manifest_names, "manifest.class_names"))
        source = f"{source} + manifest.class_names" if source else "manifest.class_names"
    for class_id, name in KNOWN_CLASS_NAMES.items():
        mapping.setdefault(class_id, name)
    if not source:
        source = "built-in audited fallback"

    missing = [class_id for class_id in class_ids if class_id not in mapping]
    if missing:
        raise AuditBuildError(
            "class names are required for semantic_match, but no trusted label is available for "
            f"class IDs {missing}. Pass --class-index-json."
        )
    return {class_id: mapping[class_id] for class_id in class_ids}, source


def discover_source_samples(
    images_dir: Path,
    class_ids: Sequence[int],
    seeds: Sequence[int],
    class_names: Mapping[int, str],
) -> list[SourceSample]:
    canonical_images_dir = resolved(images_dir)
    expected_pairs = {(class_id, seed) for class_id in class_ids for seed in seeds}
    actual_by_pair: dict[tuple[int, int], Path] = {}
    canonical_to_pair: dict[Path, tuple[int, int]] = {}

    # EDM2 uses six-digit seed filenames while the ADM runner uses nineteen
    # digits so it can represent its full documented seed range.  Parse the
    # numeric identity instead of silently baking either width into the audit
    # schema.  The manifest Cartesian product remains the authority.
    for path in images_dir.glob("class_*/*.png"):
        if not path.is_file():
            continue
        class_match = re.fullmatch(r"class_(\d+)", path.parent.name)
        seed_match = re.fullmatch(r"(\d+)", path.stem)
        if class_match is None or seed_match is None:
            raise AuditBuildError(f"cannot parse class/seed identity from PNG path: {path}")
        pair = (int(class_match.group(1)), int(seed_match.group(1)))
        if pair in actual_by_pair:
            raise AuditBuildError(
                f"multiple PNG filenames encode the same numeric class/seed pair {pair}: "
                f"{actual_by_pair[pair]} and {path}"
            )
        canonical = resolved(path)
        if not is_within(canonical, canonical_images_dir):
            raise AuditBuildError(
                f"source image resolves outside the declared images directory: {path} -> {canonical}"
            )
        if path_mentions_confirmation(canonical):
            raise AuditBuildError(
                f"source image resolves into a confirmation-marked path: {path} -> {canonical}"
            )
        if canonical in canonical_to_pair:
            raise AuditBuildError(
                f"multiple source paths resolve to the same image: {path} and pair "
                f"{canonical_to_pair[canonical]}"
            )
        actual_by_pair[pair] = path
        canonical_to_pair[canonical] = pair

    missing_pairs = sorted(expected_pairs - set(actual_by_pair))
    if missing_pairs:
        raise AuditBuildError(
            f"run is incomplete: {len(missing_pairs)} manifest class/seed images are missing; "
            f"first={missing_pairs[:3]}"
        )
    extra_pairs = sorted(set(actual_by_pair) - expected_pairs)
    if extra_pairs:
        preview = [(pair, actual_by_pair[pair]) for pair in extra_pairs[:3]]
        raise AuditBuildError(
            f"images directory contains {len(extra_pairs)} PNGs not declared by the manifest; "
            f"first={preview}"
        )
    return [
        SourceSample(class_id, seed, class_names[class_id], actual_by_pair[(class_id, seed)])
        for class_id in class_ids
        for seed in seeds
    ]


def read_or_create_salt(path: Path, create: bool) -> bytes:
    path = path.expanduser()
    if path.exists():
        if not path.is_file():
            raise AuditBuildError(f"salt path is not a regular file: {path}")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(secrets.token_bytes(32))
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception as exc:
                try:
                    path.unlink()
                except OSError:
                    pass
                raise AuditBuildError(f"failed to create private salt file {path}: {exc}") from exc
    else:
        raise AuditBuildError(
            f"salt file is missing: {path}. Create an independent reviewer salt with --create-salt."
        )

    if not path.is_file():
        raise AuditBuildError(f"salt file could not be created: {path}")
    if os.name == "posix":
        permissions = stat.S_IMODE(path.stat().st_mode)
        if permissions & 0o077:
            raise AuditBuildError(
                f"salt file must not be group/world accessible: {path} has mode {permissions:04o}; "
                f"run chmod 600 {path}"
            )
    salt = path.read_bytes()
    if len(salt) < 16:
        raise AuditBuildError("reviewer salt must contain at least 16 bytes of independent entropy")
    return salt


def domain_hmac(
    salt: bytes,
    domain: str,
    reviewer_id: str,
    manifest_digest: str,
    stable_key: str,
) -> bytes:
    message = b"\x00".join(
        (
            f"blind-bad-case-audit/v{SCHEMA_VERSION}".encode("ascii"),
            domain.encode("ascii"),
            reviewer_id.encode("utf-8"),
            manifest_digest.encode("ascii"),
            stable_key.encode("utf-8"),
        )
    )
    return hmac.new(salt, message, hashlib.sha256).digest()


def blinded_order(
    samples: Sequence[SourceSample],
    salt: bytes,
    reviewer_id: str,
    manifest_digest: str,
) -> list[SourceSample]:
    return sorted(
        samples,
        key=lambda sample: (
            domain_hmac(salt, "order", reviewer_id, manifest_digest, sample.stable_key),
            sample.class_id,
            sample.seed,
        ),
    )


def anonymous_id(
    sample: SourceSample,
    salt: bytes,
    reviewer_id: str,
    manifest_digest: str,
) -> str:
    digest = domain_hmac(
        salt, "anonymous-id", reviewer_id, manifest_digest, sample.stable_key
    ).hex()
    return f"A-{digest[:ANON_HEX_LENGTH].upper()}"


def copy_sanitized_rgb_png(source: Path, destination: Path) -> tuple[int, int, str, str]:
    """Copy full RGB pixels while stripping filenames and PNG metadata."""

    source_digest = sha256_file(source)
    try:
        with Image.open(source) as image:
            image.load()
            if image.format != "PNG":
                raise AuditBuildError(f"source image is not a PNG: {source}")
            if image.mode != "RGB":
                raise AuditBuildError(
                    f"source image must be RGB to guarantee pixel-preserving sanitization: "
                    f"{source} has mode {image.mode}"
                )
            width, height = image.size
            pixel_bytes = image.tobytes()
    except AuditBuildError:
        raise
    except Exception as exc:
        raise AuditBuildError(f"cannot decode source image {source}: {exc}") from exc
    if width < 1 or height < 1:
        raise AuditBuildError(f"source image has invalid dimensions: {source} ({width}x{height})")

    clean = Image.frombytes("RGB", (width, height), pixel_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean.save(destination, format="PNG", optimize=False)
    packet_digest = sha256_file(destination)
    with Image.open(destination) as check:
        check.load()
        if check.mode != "RGB" or check.size != (width, height) or check.tobytes() != pixel_bytes:
            raise AuditBuildError(f"pixel verification failed after sanitizing {source}")
        residual_metadata = set(check.info) - {"interlace"}
        if residual_metadata:
            raise AuditBuildError(
                f"sanitized image unexpectedly retains metadata keys {sorted(residual_metadata)}: {destination}"
            )
    return width, height, source_digest, packet_digest


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes_exclusive(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as exc:
        raise AuditBuildError(f"refusing to overwrite existing output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def reviewer_readme(packet_id: str, reviewer_id: str) -> str:
    return f"""Blind bad-case audit packet

Packet: {packet_id}
Reviewer: {reviewer_id}

Open index.html in a modern browser.  No web server or network connection is
needed.  Inspect every image with every required view and complete all seven
region checks before assigning severity.  For 64x64 images, compare the default
nearest-neighbor 8x view with the smooth 8x view: a defect that exists only
after smooth interpolation is a rendering effect, while missing pixel support
must be marked unassessable rather than clean.  Semantic match, structural
assessability, and artifact severity are separate judgments.  Severity 0/1
labels require a second visit after leaving the image.  Progress is kept in
this browser's local storage; export JSON or CSV regularly and at completion.

This directory intentionally contains no source class IDs, seeds, path scores,
ASD/evidence values, or tail/rank labels.  Do not accept a source-to-anonymous
mapping, signal table, or ranked gallery from the study organizer before your
review is locked and exported.
"""


def build_html(public_payload: Mapping[str, Any]) -> str:
    encoded_payload = base64.b64encode(
        json.dumps(public_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file: data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>Blind bad-case audit</title>
  <style>
    :root {{ color-scheme: light dark; --accent: #1e66d0; --danger: #b42318; --muted: #667085; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 15px/1.4 system-ui, sans-serif; background: Canvas; color: CanvasText; }}
    header {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 16px; align-items: center; padding: 10px 16px; border-bottom: 1px solid GrayText; background: Canvas; }}
    header .grow {{ flex: 1; }}
    button, input, textarea, select {{ font: inherit; }}
    button {{ cursor: pointer; padding: 7px 10px; }}
    button.primary {{ background: var(--accent); color: white; border: 1px solid var(--accent); border-radius: 5px; }}
    button:disabled {{ cursor: not-allowed; opacity: .45; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 430px; min-height: calc(100vh - 58px); }}
    .viewer {{ min-width: 0; padding: 12px; border-right: 1px solid GrayText; }}
    .sample-head {{ display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; margin-bottom: 8px; }}
    .sample-head h1 {{ font-size: 20px; margin: 0; }}
    .opaque {{ color: var(--muted); font-family: ui-monospace, monospace; }}
    .toolbar {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
    .view-required {{ margin: 0 0 8px; color: var(--muted); }}
    .view-required.problem {{ color: var(--danger); font-weight: 650; }}
    .stage {{ height: calc(100vh - 150px); min-height: 440px; overflow: auto; background: #171717; border: 1px solid #555; display: grid; place-items: start center; padding: 16px; }}
    .stage img {{ display: block; max-width: none; image-rendering: auto; transform-origin: top left; box-shadow: 0 1px 10px #000; }}
    .panel {{ padding: 14px 18px 28px; overflow-y: auto; max-height: calc(100vh - 58px); }}
    fieldset {{ border: 1px solid GrayText; border-radius: 6px; margin: 0 0 14px; padding: 10px 12px; }}
    legend {{ font-weight: 700; padding: 0 5px; }}
    .option {{ display: block; padding: 5px 0; }}
    .option small {{ display: block; color: var(--muted); margin-left: 25px; }}
    .flag-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px 12px; }}
    .check-grid {{ display: grid; gap: 9px; }}
    .check-row {{ display: grid; grid-template-columns: minmax(0, 1fr) 175px; gap: 8px; align-items: start; }}
    .check-row small {{ display: block; color: var(--muted); }}
    .check-row select {{ width: 100%; }}
    textarea {{ width: 100%; min-height: 100px; resize: vertical; padding: 8px; }}
    textarea.short {{ min-height: 62px; }}
    .nav {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    .status {{ min-height: 22px; color: var(--muted); }}
    .status.problem {{ color: var(--danger); font-weight: 650; }}
    progress {{ width: min(280px, 28vw); }}
    dialog {{ max-width: 700px; }}
    .confidential {{ padding: 8px; border-left: 4px solid var(--accent); background: color-mix(in srgb, var(--accent) 9%, Canvas); margin-bottom: 14px; }}
    .lock-box {{ border-left: 4px solid var(--accent); padding: 9px 11px; margin: 12px 0; background: color-mix(in srgb, var(--accent) 7%, Canvas); }}
    .lock-box.problem {{ border-left-color: var(--danger); }}
    kbd {{ border: 1px solid #999; border-radius: 3px; padding: 1px 4px; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      .viewer {{ border-right: 0; }}
      .stage {{ height: 65vh; }}
      .panel {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <strong id="packet-label">Blind audit</strong>
    <span id="position"></span>
    <progress id="progress" value="0" max="1"></progress>
    <span id="completion"></span>
    <span class="grow"></span>
    <button id="import-json">Import JSON</button>
    <input id="import-file" type="file" accept="application/json,.json" hidden>
    <button id="export-json">Export JSON</button>
    <button id="export-csv">Export CSV</button>
  </header>
  <main>
    <section class="viewer">
      <div class="sample-head">
        <h1>Target: <span id="target-class"></span></h1>
        <span class="opaque" id="anonymous-id"></span>
        <span id="dimensions"></span>
      </div>
      <div class="toolbar">
        <button id="fit">Fit</button>
        <button id="actual">100% / actual pixels</button>
        <button id="zoom-2x">Smooth 2× detail</button>
        <button id="nearest-8x">Nearest 8×</button>
        <button id="smooth-8x">Smooth 8×</button>
        <button id="zoom-out">−</button>
        <span id="zoom-label">100%</span>
        <button id="zoom-in">+</button>
        <button id="fullscreen">Fullscreen</button>
        <span>Double-click image: fit ↔ 100%</span>
      </div>
      <p class="view-required" id="view-requirement"></p>
      <div class="stage" id="stage"><img id="sample-image" alt="Image under review" draggable="false"></div>
    </section>
    <aside class="panel">
      <div class="confidential">
        Judge the image only. Seed, class ID, ASD/path evidence, and tail/rank are intentionally absent.
        Required views and all seven region checks must be completed before locking a label.
      </div>
      <fieldset>
        <legend>Seven-region anti-miss checklist (required)</legend>
        <div class="check-grid" id="inspection-checklist"></div>
      </fieldset>
      <fieldset>
        <legend>Structural assessability (required)</legend>
        <div id="assessability-options"></div>
        <label for="assessability-reason"><strong>Reason</strong> — required for partial/insufficient.</label>
        <textarea class="short" id="assessability-reason" placeholder="Example: face occupies too few native pixels to judge eye geometry..."></textarea>
      </fieldset>
      <fieldset>
        <legend>Origin judgment (required; observation, not ground truth)</legend>
        <div id="origin-options"></div>
      </fieldset>
      <fieldset id="severity-fieldset">
        <legend>Artifact severity (required)</legend>
        <div id="severity-options"></div>
      </fieldset>
      <fieldset>
        <legend>Semantic match (required)</legend>
        <label class="option"><input type="radio" name="semantic" value="yes"> Yes — target class is present</label>
        <label class="option"><input type="radio" name="semantic" value="no"> No — absent or wrong class</label>
        <label class="option"><input type="radio" name="semantic" value="uncertain"> Uncertain</label>
      </fieldset>
      <fieldset>
        <legend>Artifact flags</legend>
        <div class="flag-grid" id="flag-options"></div>
      </fieldset>
      <label for="notes"><strong>Notes</strong> — identify the location and defect; required for every severity &gt; 0.</label>
      <textarea id="notes" placeholder="Example: lower-left forelimb merges into the grass..."></textarea>
      <div class="lock-box problem" id="lock-box">
        <strong>Label lock</strong>
        <div id="lock-status"></div>
        <div class="nav">
          <button id="lock-initial" class="primary">Lock initial review</button>
          <button id="lock-secondary">Complete required second review</button>
          <button id="reopen-label">Reopen locked label</button>
        </div>
      </div>
      <div class="status" id="item-status"></div>
      <div class="nav">
        <button id="previous">← Previous</button>
        <button id="next" class="primary">Next →</button>
        <button id="next-incomplete">Next incomplete</button>
      </div>
      <p><small>Shortcuts outside text fields: <kbd>0</kbd>–<kbd>3</kbd> severity; <kbd>←</kbd>/<kbd>→</kbd> navigate. Drafts auto-save locally.</small></p>
    </aside>
  </main>
  <script>
  'use strict';
  const PACKET = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('{encoded_payload}'), c => c.charCodeAt(0))));
  const storageKey = `blind-audit:${{PACKET.packet_id}}:${{PACKET.reviewer_id}}`;
  const byId = id => document.getElementById(id);
  const image = byId('sample-image');
  const stage = byId('stage');
  let currentIndex = 0;
  let zoom = 1;
  let zoomMode = 'fit';
  let renderingMode = 'smooth';
  let state = {{ annotations: {{}}, current_index: 0 }};
  const HARD_FLAGS = new Set(PACKET.hard_artifact_flags);
  const INSPECTION_CODES = PACKET.inspection_checks.map(entry => entry.code);
  const INSPECTION_VALUES = new Set(PACKET.inspection_values.map(entry => entry.value));

  function freshAnnotation() {{
    return {{
      artifact_severity: null,
      artifact_flags: [],
      semantic_match: null,
      structural_assessability: null,
      assessability_reason: '',
      artifact_origin_judgment: null,
      inspection_checks: {{}},
      notes: '',
      view_events: [],
      initial_locked_at: null,
      left_after_initial_lock: false,
      secondary_reviewed_at: null,
      label_locked_at: null,
      updated_at: null,
    }};
  }}
  function normalizeAnnotation(annotation) {{
    const fresh = freshAnnotation();
    for (const [key, value] of Object.entries(fresh)) {{
      if (!(key in annotation)) annotation[key] = value;
    }}
    if (!annotation.inspection_checks || typeof annotation.inspection_checks !== 'object') annotation.inspection_checks = {{}};
    if (!Array.isArray(annotation.artifact_flags)) annotation.artifact_flags = [];
    if (!Array.isArray(annotation.view_events)) annotation.view_events = [];
    return annotation;
  }}
  function annotationFor(id) {{
    if (!state.annotations[id]) state.annotations[id] = freshAnnotation();
    return normalizeAnnotation(state.annotations[id]);
  }}
  function saveLocal() {{
    state.current_index = currentIndex;
    try {{ localStorage.setItem(storageKey, JSON.stringify(state)); }}
    catch (error) {{ byId('item-status').textContent = `Local auto-save failed: ${{error}}`; byId('item-status').classList.add('problem'); }}
  }}
  function loadLocal() {{
    try {{
      const raw = localStorage.getItem(storageKey);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.annotations === 'object') state = parsed;
      if (Number.isInteger(parsed.current_index)) currentIndex = Math.max(0, Math.min(PACKET.items.length - 1, parsed.current_index));
    }} catch (error) {{ console.warn('Ignoring invalid local draft', error); }}
  }}
  function isLowResolution(item) {{ return Math.max(item.width, item.height) <= 64; }}
  function viewedKinds(annotation, after=null) {{
    const threshold = after ? Date.parse(after) : null;
    return new Set(annotation.view_events.filter(event => {{
      if (threshold === null) return true;
      const eventTime = Date.parse(event.at);
      return Number.isFinite(eventTime) && Number.isFinite(threshold) && eventTime > threshold;
    }}).map(event => event.kind));
  }}
  function requiredViewIssue(annotation, item, after=null) {{
    const viewed = viewedKinds(annotation, after);
    const required = isLowResolution(item) ? ['nearest_8x', 'smooth_8x'] : ['native_100', 'smooth_2x'];
    const missing = required.filter(kind => !viewed.has(kind));
    if (missing.length) return `${{after ? 'Second-review views' : 'Required views'}} not completed: ${{missing.join(', ')}}.`;
    return null;
  }}
  function contentIssueFor(annotation, item) {{
    const viewIssue = requiredViewIssue(annotation, item);
    if (viewIssue) return viewIssue;
    for (const code of INSPECTION_CODES) {{
      if (!INSPECTION_VALUES.has(annotation.inspection_checks[code])) return `Complete checklist item: ${{code}}.`;
    }}
    if (!PACKET.assessability_scale.some(entry => entry.value === annotation.structural_assessability)) return 'Choose structural assessability.';
    if (annotation.structural_assessability !== 'sufficient' && !annotation.assessability_reason.trim()) return 'Partial/insufficient assessability requires a reason.';
    if (!PACKET.origin_judgment_scale.some(entry => entry.value === annotation.artifact_origin_judgment)) return 'Choose an origin judgment.';
    if (!Number.isInteger(annotation.artifact_severity) || annotation.artifact_severity < 0 || annotation.artifact_severity > 3) return 'Choose severity 0–3.';
    if (!['yes', 'no', 'uncertain'].includes(annotation.semantic_match)) return 'Choose semantic match.';
    if (annotation.artifact_severity === 0 && annotation.artifact_flags.length) return 'Severity 0 cannot have artifact flags.';
    if (annotation.artifact_severity === 0 && annotation.artifact_origin_judgment !== 'none_observed') return 'Severity 0 requires origin judgment “none observed”.';
    if (annotation.artifact_severity > 0 && annotation.artifact_origin_judgment === 'none_observed') return 'Severity > 0 cannot use origin judgment “none observed”.';
    if (annotation.artifact_severity > 0 && !annotation.artifact_flags.length) return 'Severity > 0 requires at least one artifact flag.';
    if (annotation.artifact_severity > 0 && !annotation.notes.trim()) return 'Every observed defect requires a location-specific note.';
    if (annotation.artifact_flags.some(code => HARD_FLAGS.has(code)) && annotation.artifact_severity < 2) return 'A hard structural flag requires severity >= 2.';
    if (Object.values(annotation.inspection_checks).includes('defect') && annotation.artifact_severity === 0) return 'A checklist defect cannot be paired with severity 0.';
    if (Object.values(annotation.inspection_checks).includes('unassessable') && annotation.structural_assessability === 'sufficient') return 'An unassessable region is inconsistent with sufficient assessability.';
    if (Object.values(annotation.inspection_checks).includes('not_applicable') && !annotation.notes.trim()) return 'Every not-applicable checklist item requires an explanation in notes.';
    if (annotation.artifact_severity === 0 && annotation.structural_assessability !== 'sufficient') return 'Severity 0 requires sufficient structural assessability; unresolved is not clean.';
    return null;
  }}
  function issueFor(annotation, item) {{
    const contentIssue = contentIssueFor(annotation, item);
    if (contentIssue) return contentIssue;
    if (!annotation.initial_locked_at) return 'Lock the initial review.';
    if (!Number.isFinite(Date.parse(annotation.initial_locked_at))) return 'Initial lock timestamp is invalid.';
    if (annotation.artifact_severity <= 1 && !annotation.left_after_initial_lock) return 'Severity 0/1 must leave this image before its second review.';
    if (annotation.artifact_severity <= 1 && !annotation.secondary_reviewed_at) {{
      if (annotation.initial_locked_at && annotation.left_after_initial_lock) {{
        const secondViewIssue = requiredViewIssue(annotation, item, annotation.initial_locked_at);
        if (secondViewIssue) return secondViewIssue;
      }}
      return 'Severity 0/1 requires a delayed second review after visiting another image.';
    }}
    if (!annotation.label_locked_at) return 'Complete the label lock.';
    if (!Number.isFinite(Date.parse(annotation.label_locked_at)) || Date.parse(annotation.label_locked_at) < Date.parse(annotation.initial_locked_at)) return 'Final lock timestamp is invalid.';
    if (annotation.artifact_severity <= 1 && (!Number.isFinite(Date.parse(annotation.secondary_reviewed_at)) || Date.parse(annotation.secondary_reviewed_at) <= Date.parse(annotation.initial_locked_at))) return 'Second-review timestamp must be after the initial lock.';
    return null;
  }}
  function derivedStatuses(annotation) {{
    const defectPresent = Number.isInteger(annotation.artifact_severity) && annotation.artifact_severity >= 1;
    const clearBad = Number.isInteger(annotation.artifact_severity) && annotation.artifact_severity >= 2;
    const semanticBad = annotation.semantic_match === 'no';
    const overallBad = clearBad || semanticBad;
    const possibleBad = !overallBad && (
      annotation.artifact_severity === 1 ||
      annotation.semantic_match === 'uncertain' ||
      ['partial', 'insufficient'].includes(annotation.structural_assessability)
    );
    const clean = Boolean(annotation.label_locked_at) && !overallBad && !possibleBad &&
      annotation.artifact_severity === 0 && annotation.semantic_match === 'yes' &&
      annotation.structural_assessability === 'sufficient';
    return {{ defect_present: defectPresent, clear_bad: clearBad, semantic_bad: semanticBad, overall_bad: overallBad, possible_bad: possibleBad, clean }};
  }}
  function completedCount() {{ return PACKET.items.filter(item => !issueFor(annotationFor(item.anonymous_id), item)).length; }}
  function updateProgress() {{
    const completed = completedCount();
    byId('progress').max = PACKET.items.length;
    byId('progress').value = completed;
    byId('completion').textContent = `${{completed}} / ${{PACKET.items.length}} complete`;
  }}
  function recordView(kind, details={{}}) {{
    const annotation = annotationFor(currentItem().anonymous_id);
    annotation.view_events.push({{ kind, at: new Date().toISOString(), zoom, rendering: renderingMode, ...details }});
    if (annotation.view_events.length > 200) annotation.view_events = annotation.view_events.slice(-200);
    annotation.updated_at = new Date().toISOString();
    saveLocal();
    updateViewRequirement();
    updateProgress();
  }}
  function setRendering(mode) {{
    renderingMode = mode;
    image.style.imageRendering = mode === 'nearest' ? 'pixelated' : 'auto';
  }}
  function setZoom(value, mode='manual') {{
    if (!image.naturalWidth) return;
    zoom = Math.max(0.1, Math.min(8, value));
    zoomMode = mode;
    image.style.width = `${{Math.round(image.naturalWidth * zoom)}}px`;
    image.style.height = `${{Math.round(image.naturalHeight * zoom)}}px`;
    byId('zoom-label').textContent = `${{Math.round(zoom * 100)}}%`;
  }}
  function fitImage(record=true) {{
    if (!image.naturalWidth) return;
    setRendering('smooth');
    const usableWidth = Math.max(100, stage.clientWidth - 34);
    const usableHeight = Math.max(100, stage.clientHeight - 34);
    setZoom(Math.min(1, usableWidth / image.naturalWidth, usableHeight / image.naturalHeight), 'fit');
    if (record) recordView('fit');
  }}
  function currentItem() {{ return PACKET.items[currentIndex]; }}
  function showNative100() {{ setRendering('smooth'); setZoom(1, 'actual'); recordView('native_100'); }}
  function showSmooth2x() {{ setRendering('smooth'); setZoom(2, 'smooth_2x'); recordView('smooth_2x'); }}
  function showNearest8x() {{ setRendering('nearest'); setZoom(8, 'nearest_8x'); recordView('nearest_8x'); }}
  function showSmooth8x() {{ setRendering('smooth'); setZoom(8, 'smooth_8x'); recordView('smooth_8x'); }}
  function updateViewRequirement() {{
    const item = currentItem();
    const annotation = annotationFor(item.anonymous_id);
    const issue = requiredViewIssue(annotation, item);
    const element = byId('view-requirement');
    element.textContent = isLowResolution(item)
      ? (issue || 'ADM64/low-resolution gate complete: nearest 8× and smooth 8× both viewed.')
      : (issue || 'High-resolution view gate complete: native 100% and smooth 2× both viewed.');
    element.classList.toggle('problem', Boolean(issue));
  }}
  function setAnnotationControlsDisabled(disabled) {{
    document.querySelectorAll('#inspection-checklist select, input[name="severity"], input[name="semantic"], input[name="assessability"], input[name="origin"], input[data-flag], #assessability-reason, #notes').forEach(control => {{ control.disabled = disabled; }});
  }}
  function lockStatus(annotation) {{
    if (annotation.label_locked_at) return `Final label locked at ${{annotation.label_locked_at}}.`;
    if (annotation.initial_locked_at && annotation.artifact_severity <= 1) {{
      return annotation.left_after_initial_lock
        ? 'Initial 0/1 review locked. Repeat both required views, re-inspect, then complete the second review.'
        : 'Initial 0/1 review locked. Visit at least one other image, then return for a second review.';
    }}
    return 'Complete views, checklist, labels, and notes, then lock the initial review.';
  }}
  function updateLockControls(annotation) {{
    const contentIssue = contentIssueFor(annotation, currentItem());
    const secondViewIssue = annotation.initial_locked_at
      ? requiredViewIssue(annotation, currentItem(), annotation.initial_locked_at)
      : 'Initial review is not locked.';
    byId('lock-status').textContent = lockStatus(annotation);
    byId('lock-box').classList.toggle('problem', !annotation.label_locked_at);
    byId('lock-initial').disabled = Boolean(annotation.label_locked_at || annotation.initial_locked_at || contentIssue);
    byId('lock-secondary').disabled = Boolean(
      annotation.label_locked_at || !annotation.initial_locked_at ||
      annotation.artifact_severity > 1 || !annotation.left_after_initial_lock || contentIssue || secondViewIssue
    );
    byId('reopen-label').disabled = !annotation.label_locked_at;
    setAnnotationControlsDisabled(Boolean(annotation.label_locked_at));
  }}
  function render() {{
    const item = currentItem();
    const annotation = annotationFor(item.anonymous_id);
    byId('position').textContent = `Image ${{currentIndex + 1}} of ${{PACKET.items.length}}`;
    byId('target-class').textContent = item.target_class_name;
    byId('anonymous-id').textContent = item.anonymous_id;
    byId('dimensions').textContent = `${{item.width}} × ${{item.height}} px`;
    image.alt = `Image under review; target ${{item.target_class_name}}`;
    const expectedAnonymousId = item.anonymous_id;
    image.onload = () => {{
      if (currentItem().anonymous_id !== expectedAnonymousId) return;
      if (isLowResolution(item)) showNearest8x();
      else if (zoomMode === 'actual') showNative100();
      else fitImage();
      updateViewRequirement();
    }};
    image.src = item.image;
    document.querySelectorAll('input[name="severity"]').forEach(input => {{ input.checked = Number(input.value) === annotation.artifact_severity; }});
    document.querySelectorAll('input[name="semantic"]').forEach(input => {{ input.checked = input.value === annotation.semantic_match; }});
    document.querySelectorAll('input[name="assessability"]').forEach(input => {{ input.checked = input.value === annotation.structural_assessability; }});
    document.querySelectorAll('input[name="origin"]').forEach(input => {{ input.checked = input.value === annotation.artifact_origin_judgment; }});
    document.querySelectorAll('input[data-flag]').forEach(input => {{ input.checked = annotation.artifact_flags.includes(input.dataset.flag); }});
    document.querySelectorAll('select[data-check]').forEach(select => {{ select.value = annotation.inspection_checks[select.dataset.check] || ''; }});
    byId('assessability-reason').value = annotation.assessability_reason || '';
    byId('notes').value = annotation.notes || '';
    byId('previous').disabled = currentIndex === 0;
    byId('next').disabled = currentIndex === PACKET.items.length - 1;
    const issue = issueFor(annotation, item);
    byId('item-status').textContent = issue || 'This item is complete.';
    byId('item-status').classList.toggle('problem', Boolean(issue));
    updateProgress();
    updateViewRequirement();
    updateLockControls(annotation);
    saveLocal();
  }}
  function resetLocksAfterMaterialChange(annotation) {{
    if (!annotation.initial_locked_at && !annotation.label_locked_at) return;
    annotation.view_events.push({{ kind: 'label_changed_after_lock', at: new Date().toISOString(), zoom, rendering: renderingMode }});
    annotation.initial_locked_at = null;
    annotation.left_after_initial_lock = false;
    annotation.secondary_reviewed_at = null;
    annotation.label_locked_at = null;
  }}
  function touch(annotation, material=true) {{
    if (material) resetLocksAfterMaterialChange(annotation);
    annotation.updated_at = new Date().toISOString();
    saveLocal(); updateProgress();
    const issue = issueFor(annotation, currentItem());
    byId('item-status').textContent = issue || 'This item is complete.';
    byId('item-status').classList.toggle('problem', Boolean(issue));
    updateViewRequirement(); updateLockControls(annotation);
  }}
  function goTo(index) {{
    const destination = Math.max(0, Math.min(PACKET.items.length - 1, index));
    if (destination !== currentIndex) {{
      const departing = annotationFor(currentItem().anonymous_id);
      if (departing.initial_locked_at && !departing.label_locked_at && departing.artifact_severity <= 1) {{
        departing.left_after_initial_lock = true;
        const now = new Date().toISOString();
        departing.view_events.push({{ kind: 'left_for_secondary_review', at: now, zoom, rendering: renderingMode }});
        departing.updated_at = now;
        saveLocal();
      }}
    }}
    currentIndex = destination; zoomMode = 'fit'; renderingMode = 'smooth'; render(); window.scrollTo({{top: 0, behavior: 'instant'}});
  }}
  function goNextIncomplete() {{
    for (let offset = 1; offset <= PACKET.items.length; offset++) {{
      const index = (currentIndex + offset) % PACKET.items.length;
      if (issueFor(annotationFor(PACKET.items[index].anonymous_id), PACKET.items[index])) {{ goTo(index); return; }}
    }}
    alert('All items pass completeness checks.');
  }}
  function buildExport() {{
    const issues = [];
    const annotations = PACKET.items.map((item, index) => {{
      const annotation = annotationFor(item.anonymous_id);
      const issue = issueFor(annotation, item);
      if (issue) issues.push({{ anonymous_id: item.anonymous_id, order_index: index + 1, issue }});
      return {{
        order_index: index + 1,
        anonymous_id: item.anonymous_id,
        target_class_name: item.target_class_name,
        artifact_severity: annotation.artifact_severity,
        artifact_flags: [...annotation.artifact_flags].sort(),
        semantic_match: annotation.semantic_match,
        structural_assessability: annotation.structural_assessability,
        assessability_reason: annotation.assessability_reason || '',
        artifact_origin_judgment: annotation.artifact_origin_judgment,
        inspection_checks: Object.fromEntries(INSPECTION_CODES.map(code => [code, annotation.inspection_checks[code] || null])),
        notes: annotation.notes || '',
        view_events: annotation.view_events,
        initial_locked_at: annotation.initial_locked_at,
        left_after_initial_lock: annotation.left_after_initial_lock,
        secondary_reviewed_at: annotation.secondary_reviewed_at,
        label_locked_at: annotation.label_locked_at,
        ...derivedStatuses(annotation),
        updated_at: annotation.updated_at,
      }};
    }});
    return {{
      schema: 'blind_bad_case_annotations/v2',
      packet_id: PACKET.packet_id,
      reviewer_id: PACKET.reviewer_id,
      exported_at: new Date().toISOString(),
      status: issues.length ? 'draft_incomplete' : 'complete',
      bad_case_definitions: {{
        defect_present: 'artifact_severity >= 1',
        clear_bad: 'artifact_severity >= 2',
        semantic_bad: 'semantic_match == no',
        overall_bad: 'clear_bad OR semantic_bad',
        possible_bad: 'NOT overall_bad AND (severity == 1 OR semantic uncertain OR assessability partial/insufficient)',
        clean: 'locked AND severity == 0 AND semantic yes AND assessability sufficient',
      }},
      item_count: annotations.length,
      validation_issues: issues,
      annotations,
    }};
  }}
  function allowDraft(result) {{
    return !result.validation_issues.length || confirm(`${{result.validation_issues.length}} items are incomplete or inconsistent. Export a clearly marked draft anyway?`);
  }}
  function download(filename, content, type) {{
    const blob = new Blob([content], {{type}});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }}
  function csvCell(value) {{
    let text = value == null ? '' : String(value);
    if ('=+-@'.includes(text.charAt(0)) || [9, 13].includes(text.charCodeAt(0))) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  }}
  function exportJson() {{
    const result = buildExport(); if (!allowDraft(result)) return;
    download(`${{PACKET.packet_id}}_${{PACKET.reviewer_id}}.json`, JSON.stringify(result, null, 2) + String.fromCharCode(10), 'application/json;charset=utf-8');
  }}
  function exportCsv() {{
    const result = buildExport(); if (!allowDraft(result)) return;
    const header = ['schema','packet_id','reviewer_id','status','order_index','anonymous_id','target_class_name','artifact_severity','artifact_flags','semantic_match','structural_assessability','assessability_reason','artifact_origin_judgment','inspection_checks_json','view_events_json','initial_locked_at','left_after_initial_lock','secondary_reviewed_at','label_locked_at','defect_present','clear_bad','semantic_bad','overall_bad','possible_bad','clean','notes','updated_at'];
    const rows = result.annotations.map(row => [result.schema,result.packet_id,result.reviewer_id,result.status,row.order_index,row.anonymous_id,row.target_class_name,row.artifact_severity,row.artifact_flags.join(';'),row.semantic_match,row.structural_assessability,row.assessability_reason,row.artifact_origin_judgment,JSON.stringify(row.inspection_checks),JSON.stringify(row.view_events),row.initial_locked_at,row.left_after_initial_lock,row.secondary_reviewed_at,row.label_locked_at,row.defect_present,row.clear_bad,row.semantic_bad,row.overall_bad,row.possible_bad,row.clean,row.notes,row.updated_at]);
    const bom = String.fromCharCode(0xfeff); const crlf = String.fromCharCode(13, 10);
    const csv = bom + [header, ...rows].map(row => row.map(csvCell).join(',')).join(crlf) + crlf;
    download(`${{PACKET.packet_id}}_${{PACKET.reviewer_id}}.csv`, csv, 'text/csv;charset=utf-8');
  }}
  function importJson(file) {{
    const reader = new FileReader();
    reader.onload = () => {{
      try {{
        const payload = JSON.parse(reader.result);
        if (payload.schema !== 'blind_bad_case_annotations/v2') throw new Error('only annotation schema v2 can be imported into this packet');
        if (payload.packet_id !== PACKET.packet_id || payload.reviewer_id !== PACKET.reviewer_id) throw new Error('packet_id or reviewer_id does not match');
        const incoming = new Map((payload.annotations || []).map(row => [row.anonymous_id, row]));
        if (incoming.size !== PACKET.items.length || PACKET.items.some(item => !incoming.has(item.anonymous_id))) throw new Error('anonymous ID set does not match this packet');
        for (const item of PACKET.items) {{
          const row = incoming.get(item.anonymous_id);
          state.annotations[item.anonymous_id] = {{
            artifact_severity: Number.isInteger(row.artifact_severity) && row.artifact_severity >= 0 && row.artifact_severity <= 3 ? row.artifact_severity : null,
            artifact_flags: Array.isArray(row.artifact_flags) ? row.artifact_flags.filter(code => PACKET.artifact_flags.some(flag => flag.code === code)) : [],
            semantic_match: ['yes','no','uncertain'].includes(row.semantic_match) ? row.semantic_match : null,
            structural_assessability: PACKET.assessability_scale.some(entry => entry.value === row.structural_assessability) ? row.structural_assessability : null,
            assessability_reason: typeof row.assessability_reason === 'string' ? row.assessability_reason : '',
            artifact_origin_judgment: PACKET.origin_judgment_scale.some(entry => entry.value === row.artifact_origin_judgment) ? row.artifact_origin_judgment : null,
            inspection_checks: Object.fromEntries(INSPECTION_CODES.map(code => [code, INSPECTION_VALUES.has(row.inspection_checks && row.inspection_checks[code]) ? row.inspection_checks[code] : null])),
            notes: typeof row.notes === 'string' ? row.notes : '',
            view_events: Array.isArray(row.view_events) ? row.view_events.filter(event => event && typeof event.kind === 'string' && typeof event.at === 'string').slice(-200) : [],
            initial_locked_at: typeof row.initial_locked_at === 'string' ? row.initial_locked_at : null,
            left_after_initial_lock: row.left_after_initial_lock === true,
            secondary_reviewed_at: typeof row.secondary_reviewed_at === 'string' ? row.secondary_reviewed_at : null,
            label_locked_at: typeof row.label_locked_at === 'string' ? row.label_locked_at : null,
            updated_at: row.updated_at || null,
          }};
        }}
        saveLocal(); render(); alert('Progress imported.');
      }} catch (error) {{ alert(`Import failed: ${{error.message || error}}`); }}
    }};
    reader.readAsText(file);
  }}

  byId('packet-label').textContent = `${{PACKET.packet_id}} · reviewer ${{PACKET.reviewer_id}}`;
  PACKET.inspection_checks.forEach(check => {{
    const row = document.createElement('label'); row.className = 'check-row';
    const text = document.createElement('span');
    const strong = document.createElement('strong'); strong.textContent = check.label;
    const small = document.createElement('small'); small.textContent = check.description;
    text.append(strong, small);
    const select = document.createElement('select'); select.dataset.check = check.code;
    const prompt = document.createElement('option'); prompt.value = ''; prompt.textContent = 'Choose…';
    select.append(prompt);
    PACKET.inspection_values.forEach(entry => {{
      const option = document.createElement('option'); option.value = entry.value; option.textContent = entry.label; select.append(option);
    }});
    select.addEventListener('change', () => {{ const a = annotationFor(currentItem().anonymous_id); a.inspection_checks[check.code] = select.value || null; touch(a); }});
    row.append(text, select); byId('inspection-checklist').append(row);
  }});
  PACKET.assessability_scale.forEach(entry => {{
    const label = document.createElement('label'); label.className = 'option';
    const input = document.createElement('input'); input.type = 'radio'; input.name = 'assessability'; input.value = entry.value;
    input.addEventListener('change', () => {{ const a = annotationFor(currentItem().anonymous_id); a.structural_assessability = input.value; touch(a); }});
    const text = document.createTextNode(' ' + entry.label); const small = document.createElement('small'); small.textContent = entry.description;
    label.append(input, text, small); byId('assessability-options').append(label);
  }});
  PACKET.origin_judgment_scale.forEach(entry => {{
    const label = document.createElement('label'); label.className = 'option';
    const input = document.createElement('input'); input.type = 'radio'; input.name = 'origin'; input.value = entry.value;
    input.addEventListener('change', () => {{ const a = annotationFor(currentItem().anonymous_id); a.artifact_origin_judgment = input.value; touch(a); }});
    const text = document.createTextNode(' ' + entry.label); const small = document.createElement('small'); small.textContent = entry.description;
    label.append(input, text, small); byId('origin-options').append(label);
  }});
  PACKET.severity_scale.forEach(entry => {{
    const label = document.createElement('label'); label.className = 'option';
    const input = document.createElement('input'); input.type = 'radio'; input.name = 'severity'; input.value = entry.value;
    input.addEventListener('change', () => {{
      const a = annotationFor(currentItem().anonymous_id); const requested = Number(input.value);
      a.artifact_severity = requested < 2 && a.artifact_flags.some(code => HARD_FLAGS.has(code)) ? 2 : requested;
      document.querySelectorAll('input[name="severity"]').forEach(candidate => {{ candidate.checked = Number(candidate.value) === a.artifact_severity; }});
      touch(a);
    }});
    const text = document.createTextNode(' ' + entry.label); const small = document.createElement('small'); small.textContent = entry.description;
    label.append(input, text, small); byId('severity-options').append(label);
  }});
  PACKET.artifact_flags.forEach(flag => {{
    const label = document.createElement('label'); label.className = 'option'; label.title = flag.description;
    const input = document.createElement('input'); input.type = 'checkbox'; input.dataset.flag = flag.code;
    input.addEventListener('change', () => {{
      const a = annotationFor(currentItem().anonymous_id); const values = new Set(a.artifact_flags);
      input.checked ? values.add(flag.code) : values.delete(flag.code); a.artifact_flags = [...values];
      if (input.checked && HARD_FLAGS.has(flag.code) && (!Number.isInteger(a.artifact_severity) || a.artifact_severity < 2)) {{
        a.artifact_severity = 2;
        document.querySelectorAll('input[name="severity"]').forEach(candidate => {{ candidate.checked = Number(candidate.value) === 2; }});
      }}
      touch(a);
    }});
    label.append(input, document.createTextNode(' ' + flag.label)); byId('flag-options').append(label);
  }});
  document.querySelectorAll('input[name="semantic"]').forEach(input => input.addEventListener('change', () => {{ const a = annotationFor(currentItem().anonymous_id); a.semantic_match = input.value; touch(a); }}));
  byId('assessability-reason').addEventListener('input', event => {{ const a = annotationFor(currentItem().anonymous_id); a.assessability_reason = event.target.value; touch(a); }});
  byId('notes').addEventListener('input', event => {{ const a = annotationFor(currentItem().anonymous_id); a.notes = event.target.value; touch(a); }});
  byId('lock-initial').addEventListener('click', () => {{
    const a = annotationFor(currentItem().anonymous_id); const issue = contentIssueFor(a, currentItem());
    if (issue) {{ alert(issue); return; }}
    const now = new Date().toISOString();
    a.initial_locked_at = now; a.left_after_initial_lock = false;
    a.view_events.push({{ kind: 'initial_review_locked', at: now, zoom, rendering: renderingMode }});
    if (a.artifact_severity >= 2) a.label_locked_at = now;
    touch(a, false); render();
  }});
  byId('lock-secondary').addEventListener('click', () => {{
    const a = annotationFor(currentItem().anonymous_id); const issue = contentIssueFor(a, currentItem());
    if (issue) {{ alert(issue); return; }}
    if (!a.initial_locked_at || !a.left_after_initial_lock || a.artifact_severity > 1) {{ alert('A 0/1 label must be initially locked, left, and revisited before second-review completion.'); return; }}
    const secondViewIssue = requiredViewIssue(a, currentItem(), a.initial_locked_at);
    if (secondViewIssue) {{ alert(secondViewIssue); return; }}
    const now = new Date().toISOString();
    a.secondary_reviewed_at = now; a.label_locked_at = now;
    a.view_events.push({{ kind: 'secondary_review_locked', at: now, zoom, rendering: renderingMode }});
    touch(a, false); render();
  }});
  byId('reopen-label').addEventListener('click', () => {{
    const a = annotationFor(currentItem().anonymous_id);
    if (!a.label_locked_at || !confirm('Reopen this label? It must pass the complete lock workflow again.')) return;
    const now = new Date().toISOString();
    a.view_events.push({{ kind: 'final_label_reopened', at: now, zoom, rendering: renderingMode }});
    a.initial_locked_at = null; a.left_after_initial_lock = false; a.secondary_reviewed_at = null; a.label_locked_at = null;
    touch(a, false); render();
  }});
  byId('previous').addEventListener('click', () => goTo(currentIndex - 1));
  byId('next').addEventListener('click', () => goTo(currentIndex + 1));
  byId('next-incomplete').addEventListener('click', goNextIncomplete);
  byId('fit').addEventListener('click', () => fitImage(true));
  byId('actual').addEventListener('click', showNative100);
  byId('zoom-2x').addEventListener('click', showSmooth2x);
  byId('nearest-8x').addEventListener('click', showNearest8x);
  byId('smooth-8x').addEventListener('click', showSmooth8x);
  byId('zoom-in').addEventListener('click', () => {{ setRendering('smooth'); setZoom(zoom * 1.25); if (zoom >= 2) recordView('smooth_2x'); }});
  byId('zoom-out').addEventListener('click', () => {{ setRendering('smooth'); setZoom(zoom / 1.25); }});
  byId('fullscreen').addEventListener('click', () => stage.requestFullscreen && stage.requestFullscreen());
  image.addEventListener('dblclick', () => zoomMode === 'actual' ? fitImage(true) : showNative100());
  window.addEventListener('resize', () => {{ if (zoomMode === 'fit') fitImage(false); }});
  byId('export-json').addEventListener('click', exportJson);
  byId('export-csv').addEventListener('click', exportCsv);
  byId('import-json').addEventListener('click', () => byId('import-file').click());
  byId('import-file').addEventListener('change', event => {{ if (event.target.files[0]) importJson(event.target.files[0]); event.target.value = ''; }});
  document.addEventListener('keydown', event => {{
    if (['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
    if (/^[0-3]$/.test(event.key)) {{ const input = document.querySelector(`input[name="severity"][value="${{event.key}}"]`); input.click(); }}
    else if (event.key === 'ArrowLeft') goTo(currentIndex - 1);
    else if (event.key === 'ArrowRight') goTo(currentIndex + 1);
  }});
  loadLocal(); render();
  </script>
</body>
</html>
'''


def validate_output_locations(packet_dir: Path, private_map: Path, salt_file: Path) -> None:
    packet_dir = resolved(packet_dir)
    private_map = resolved(private_map)
    salt_file = resolved(salt_file)
    if packet_dir.exists():
        raise AuditBuildError(f"refusing to overwrite existing reviewer packet: {packet_dir}")
    if private_map.exists():
        raise AuditBuildError(f"refusing to overwrite existing private mapping: {private_map}")
    if is_within(private_map, packet_dir):
        raise AuditBuildError("--private-map must be outside --packet-dir; reviewers must never receive it")
    if is_within(salt_file, packet_dir):
        raise AuditBuildError("--salt-file must be outside --packet-dir; reviewers must never receive it")
    if resolved(private_map) == resolved(salt_file):
        raise AuditBuildError("--private-map and --salt-file must be different files")


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser()
    manifest_path = args.manifest.expanduser() if args.manifest else run_dir / "manifest.json"
    images_dir = args.images_dir.expanduser() if args.images_dir else run_dir / "images"
    packet_dir = args.packet_dir.expanduser()
    private_map_path = args.private_map.expanduser()
    salt_path = args.salt_file.expanduser()

    # These checks precede every source read, including the manifest.
    preflight_confirmation_path_guard(run_dir, manifest_path, images_dir)
    validate_output_locations(packet_dir, private_map_path, salt_path)
    if not run_dir.is_dir():
        raise AuditBuildError(f"run directory does not exist: {run_dir}")
    if not images_dir.is_dir():
        raise AuditBuildError(f"images directory does not exist: {images_dir}")

    manifest, manifest_raw = load_json_object(manifest_path)
    class_ids, seeds = validate_discovery_manifest(manifest)
    class_names, class_name_source = load_class_names(
        manifest, class_ids, args.class_index_json, run_dir
    )
    source_samples = discover_source_samples(images_dir, class_ids, seeds, class_names)
    salt = read_or_create_salt(salt_path, args.create_salt)
    salt_fingerprint = sha256_bytes(salt)
    manifest_digest = sha256_bytes(manifest_raw)
    ordered = blinded_order(source_samples, salt, args.reviewer_id, manifest_digest)

    anon_ids = [
        anonymous_id(sample, salt, args.reviewer_id, manifest_digest) for sample in ordered
    ]
    if len(anon_ids) != len(set(anon_ids)):
        raise AuditBuildError(
            "anonymous ID collision; use a new independent salt (no output was written)"
        )
    packet_id_material = b"\x00".join(
        (
            b"blind-bad-case-packet/v2",
            manifest_digest.encode("ascii"),
            args.reviewer_id.encode("utf-8"),
            salt_fingerprint.encode("ascii"),
        )
    )
    packet_id = f"blind-{sha256_bytes(packet_id_material)[:16]}"

    packet_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = packet_dir.parent / f".{packet_dir.name}.staging-{os.getpid()}-{secrets.token_hex(6)}"
    if staging.exists():  # Effectively impossible, but never delete an unresolved collision.
        raise AuditBuildError(f"unexpected staging-path collision: {staging}")
    staging.mkdir(mode=0o700)

    public_items: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    preserve_staging = False
    try:
        for order_index, (sample, anon_id) in enumerate(zip(ordered, anon_ids), start=1):
            relative_image = Path("images") / f"{anon_id}.png"
            destination = staging / relative_image
            width, height, source_digest, packet_digest = copy_sanitized_rgb_png(
                sample.image_path, destination
            )
            public_items.append(
                {
                    "anonymous_id": anon_id,
                    "image": relative_image.as_posix(),
                    "target_class_name": sample.class_name,
                    "width": width,
                    "height": height,
                }
            )
            private_items.append(
                {
                    "order_index": order_index,
                    "anonymous_id": anon_id,
                    "class_id": sample.class_id,
                    "class_name": sample.class_name,
                    "seed": sample.seed,
                    "source_image": str(sample.image_path.resolve()),
                    "source_png_sha256": source_digest,
                    "packet_png_sha256": packet_digest,
                }
            )

        public_payload = {
            "schema": "blind_bad_case_packet/v2",
            "packet_id": packet_id,
            "reviewer_id": args.reviewer_id,
            "item_count": len(public_items),
            "blinding": {
                "visible": ["anonymous_id", "target_class_name", "full_resolution_image"],
                "withheld": [
                    "class_id",
                    "seed",
                    "ASD",
                    "path_evidence",
                    "tail_or_rank",
                    "source_path",
                ],
                "order": "deterministic keyed random permutation unique to reviewer salt",
            },
            "severity_scale": list(SEVERITY_SCALE),
            "artifact_flags": list(ARTIFACT_FLAGS),
            "hard_artifact_flags": sorted(HARD_ARTIFACT_FLAGS),
            "inspection_checks": list(INSPECTION_CHECKS),
            "inspection_values": list(INSPECTION_VALUES),
            "assessability_scale": list(ASSESSABILITY_SCALE),
            "origin_judgment_scale": list(ORIGIN_JUDGMENT_SCALE),
            "semantic_match_values": ["yes", "no", "uncertain"],
            "required_views": {
                "max_dimension_le_64": ["nearest_8x", "smooth_8x"],
                "larger_images": ["native_100", "smooth_2x"],
            },
            "bad_case_definitions": {
                "defect_present": "artifact_severity >= 1",
                "clear_bad": "artifact_severity >= 2",
                "semantic_bad": "semantic_match == no",
                "overall_bad": "clear_bad OR semantic_bad",
                "possible_bad": (
                    "NOT overall_bad AND (severity == 1 OR semantic uncertain "
                    "OR assessability partial/insufficient)"
                ),
                "clean": (
                    "locked AND severity == 0 AND semantic yes AND assessability sufficient"
                ),
            },
            "items": public_items,
        }
        private_payload = {
            "schema": "blind_bad_case_private_mapping/v2",
            "warning": "PRIVATE: never provide this file or the salt to a reviewer before labels are locked.",
            "packet_id": packet_id,
            "reviewer_id": args.reviewer_id,
            "run_dir": str(run_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_digest,
            "class_name_source": class_name_source,
            "salt_file": str(salt_path.resolve()),
            "salt_sha256_fingerprint": salt_fingerprint,
            "salt_is_not_embedded": True,
            "confirmation_guard": {
                "mode": "pre_lock_non_overridable",
                "path_marker_rejected": "confirmation",
                "reserved_seed_start_rejected": CONFIRMATION_SEED_START,
                "future_access_rule": (
                    "requires a new audited schema revision after rubric, sampling plan, "
                    "thresholds, and reviewer/adjudication rules are frozen"
                ),
            },
            "item_count": len(private_items),
            "items": private_items,
        }

        (staging / "packet_manifest.json").write_bytes(json_bytes(public_payload))
        (staging / "index.html").write_text(build_html(public_payload), encoding="utf-8")
        (staging / "README_REVIEWER.txt").write_text(
            reviewer_readme(packet_id, args.reviewer_id), encoding="utf-8"
        )

        # Write the private map first and packet last.  Thus a visible packet is
        # never published without its organizer-side decoding map.
        write_bytes_exclusive(private_map_path, json_bytes(private_payload), 0o600)
        try:
            os.replace(staging, packet_dir)
        except Exception as exc:
            preserve_staging = True
            raise AuditBuildError(
                f"private map was written, but publishing reviewer packet failed; "
                f"keep {private_map_path} and inspect staging path {staging}: {exc}"
            ) from exc
    except Exception:
        if staging.exists() and not preserve_staging:
            shutil.rmtree(staging)
        raise

    return {
        "packet_dir": str(packet_dir.resolve()),
        "private_map": str(private_map_path.resolve()),
        "salt_file": str(salt_path.resolve()),
        "packet_id": packet_id,
        "reviewer_id": args.reviewer_id,
        "item_count": len(public_items),
        "source_manifest_sha256": manifest_digest,
        "reviewer_entrypoint": str((packet_dir / "index.html").resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed discovery/validation run")
    parser.add_argument("--manifest", type=Path, default=None, help="Default: RUN_DIR/manifest.json")
    parser.add_argument("--images-dir", type=Path, default=None, help="Default: RUN_DIR/images")
    parser.add_argument(
        "--class-index-json",
        type=Path,
        default=None,
        help="ImageNet mapping; supports {index: [WNID, name]}, {index: name}, or a list",
    )
    parser.add_argument("--reviewer-id", required=True, help="Opaque reviewer identifier")
    parser.add_argument("--salt-file", type=Path, required=True, help="Private, independent salt file")
    parser.add_argument(
        "--create-salt",
        action="store_true",
        help="Create --salt-file with 32 random bytes and mode 0600 if it does not exist",
    )
    parser.add_argument("--packet-dir", type=Path, required=True, help="Directory that may be given to reviewer")
    parser.add_argument(
        "--private-map",
        type=Path,
        required=True,
        help="Organizer-only JSON outside packet-dir; never give it to reviewer",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.reviewer_id or len(args.reviewer_id) > 64 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in args.reviewer_id
    ):
        parser.error("--reviewer-id must be 1-64 characters from A-Z, a-z, 0-9, _, ., or -")
    try:
        summary = build_packet(args)
    except AuditBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
