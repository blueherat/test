#!/usr/bin/env python3
"""Decode selected states from one strictly validated DiT evidence trace.

The default product is a temporal view of ``pred_xstart`` for one image in the
frozen official batch.  Each requested internal timestep is decoded with the
same pinned ``sd-vae-ft-mse`` used by the baseline, saved as a native 256x256
PNG, and included in a timestep-labelled contact sheet.

``x_t`` is deliberately excluded by default.  If ``--include-noisy-x-t`` is
given, it is decoded into a separate directory and separate contact sheet;
every filename, PNG metadata record, label, and manifest entry calls it a
NOISY STATE.  It is never presented as a clean prediction.

Before reading any latent, this tool revalidates the source tree, checkpoint,
VAE snapshot, frozen baseline, observer manifest/completion, every trace array,
the complete transition hash trail, and the operational likelihood-ratio math
through ``observe_dit_imagenet256_path_evidence``.  This is visualization only:
there is no quality score, ranking, selection, alarm, resampling, guidance, or
intervention.  Outputs are staged atomically, self-hashed, and never
overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

try:  # Package and direct CLI imports.
    from . import observe_dit_imagenet256_path_evidence as observer
    from . import reproduce_dit_imagenet256 as strict
except ImportError:  # pragma: no cover - direct CLI invocation.
    import observe_dit_imagenet256_path_evidence as observer
    import reproduce_dit_imagenet256 as strict


RUNNER_NAME = "visualize_dit_imagenet256_trace"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
DEFAULT_INTERNAL_TIMESTEPS = (249, 225, 200, 175, 150, 149, 125, 100, 75, 50, 25, 0)
DEFAULT_COLUMNS = 4
CELL_GAP = 8
OUTER_MARGIN = 8
LABEL_HEIGHT = 38
BACKGROUND = (26, 26, 26)
TEXT = (245, 245, 245)
NOISY_TEXT = (255, 210, 80)


@dataclass(frozen=True)
class BundleContext:
    root: Path
    manifest: dict[str, Any]
    results: dict[str, Any]
    baseline: observer.BaselineRun
    spec: observer.EvidenceSpec
    arrays: dict[str, np.ndarray]
    source: dict[str, Any]
    checkpoint: dict[str, Any]
    vae: dict[str, Any]


@dataclass(frozen=True)
class Target:
    batch_index: int
    class_id: int
    timesteps: tuple[int, ...]
    trace_rows: tuple[int, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_raw_sha256(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    return strict.sha256_json(stripped)


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    payload = strict.load_json(path)
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _canonical_self_hash(payload, key):
        raise RuntimeError(f"invalid {key} in {path}")
    return payload


def _manifest_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"observer manifest lacks {context}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"observer manifest {context} is not absolute: {path}")
    return path.resolve()


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a staged directory without an overwrite race.

    Linux ``renameat2(RENAME_NOREPLACE)`` combines the atomic directory rename
    with an existence check in one kernel operation.  We fail closed if that
    primitive is unavailable instead of falling back to racy ``os.replace``.
    """

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable; refusing racy publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(f"refusing to overwrite output path: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def runtime_device_identity(device_name: str, *, query_runtime: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "requested": device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_queried": query_runtime,
    }
    if not query_runtime:
        record["reason_not_queried"] = "dry-run must not initialize CUDA or load the VAE"
        return record
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        record.update(
            {
                "torch_device": f"cuda:{index}",
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    elif device_name == "cpu":
        record.update(
            {
                "torch_device": "cpu",
                "machine": platform.machine(),
                "processor": platform.processor(),
            }
        )
    else:
        raise ValueError("decoder device must be cpu or cuda")
    return record


def visualizer_dependency_identity(*, query_cuda_runtime: bool) -> dict[str, Any]:
    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    record: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "diffusers": package_version("diffusers"),
        "safetensors": package_version("safetensors"),
        "huggingface_hub": package_version("huggingface-hub"),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn_runtime": None,
        "cudnn_runtime_queried": query_cuda_runtime,
    }
    if query_cuda_runtime:
        record["cudnn_runtime"] = torch.backends.cudnn.version()
    return record


def validate_bundle(root: Path) -> BundleContext:
    """Fail-closed validation of one observer bundle, including its trace math."""

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"bundle must be a non-symlink directory: {root}")
    root = root.resolve()
    # This preliminary self-hash read is used only to locate immutable inputs.
    # Acceptance still requires the observer's complete validator below.
    manifest = observer._read_self_hashed_json(  # noqa: SLF001 - canonical validator primitive.
        root / MANIFEST_NAME, "identity_sha256"
    )
    seed = manifest.get("seed")
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise RuntimeError("observer manifest has an invalid seed")
    classes = manifest.get("class_ids_in_official_batch_order")
    if classes != list(strict.CLASS_IDS):
        raise RuntimeError("observer class order differs from the frozen official batch")

    sources = manifest.get("sources")
    frozen = manifest.get("frozen_baseline")
    if not isinstance(sources, dict) or not isinstance(frozen, dict):
        raise RuntimeError("observer manifest lacks source/baseline provenance")
    recorded_source = sources.get("dit")
    recorded_checkpoint = sources.get("checkpoint")
    recorded_vae = sources.get("vae")
    if not all(isinstance(item, dict) for item in (recorded_source, recorded_checkpoint, recorded_vae)):
        raise RuntimeError("observer source/checkpoint/VAE records are incomplete")

    dit_root = _manifest_path(recorded_source.get("root"), "DiT source root")
    checkpoint_path = _manifest_path(recorded_checkpoint.get("path"), "checkpoint path")
    vae_snapshot = _manifest_path(recorded_vae.get("snapshot"), "VAE snapshot path")
    baseline_root = _manifest_path(frozen.get("root"), "frozen baseline root")

    source = observer.validate_repository(dit_root, checkpoint_path)
    checkpoint = observer.validate_checkpoint(checkpoint_path)
    vae = observer.validate_vae_snapshot(vae_snapshot)
    if recorded_source != source:
        raise RuntimeError("observer DiT source record differs from current strict validation")
    if recorded_checkpoint != checkpoint:
        raise RuntimeError("observer checkpoint record differs from current strict validation")
    if recorded_vae != vae:
        raise RuntimeError("observer VAE record differs from current strict validation")

    baseline = observer.validate_baseline_run(
        baseline_root,
        seed=seed,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
    )
    alpha, timestep_map = observer.load_schedule(dit_root)
    spec = observer.build_evidence_spec(
        alpha,
        timestep_map,
        total_K_budget=observer.TOTAL_K_BUDGET,
    )
    results = observer.validate_output_bundle(root, baseline=baseline, spec=spec)

    # Reuse the same exact trace loader after whole-bundle validation so the
    # arrays passed to the decoder are checked for path, file, shape, dtype,
    # per-array raw hash, and finiteness rather than reopened ad hoc.
    arrays = observer._load_trace_exact(  # noqa: SLF001 - requested canonical trace validator.
        root / observer.TRACE_NAME,
        results.get("trace", {}),
        root,
    )
    if not np.array_equal(
        arrays["internal_timestep"],
        np.arange(strict.NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int16),
    ):
        raise RuntimeError("validated trace lost reverse internal-timestep order")
    if not np.array_equal(arrays["pred_xstart"][-1], arrays["final_latents_first_half"]):
        raise RuntimeError("t=0 pred_xstart is not exactly the observer's final baseline latent")
    return BundleContext(
        root=root,
        manifest=manifest,
        results=results,
        baseline=baseline,
        spec=spec,
        arrays=arrays,
        source=source,
        checkpoint=checkpoint,
        vae=vae,
    )


def resolve_target(
    context: BundleContext,
    *,
    batch_index: int | None,
    class_id: int | None,
    timesteps: Sequence[int],
) -> Target:
    classes = tuple(context.manifest["class_ids_in_official_batch_order"])
    if batch_index is None and class_id is None:
        batch_index = 0
    if batch_index is not None and not 0 <= batch_index < len(classes):
        raise ValueError(f"batch index must lie in [0,{len(classes)-1}]")
    if class_id is not None:
        if class_id not in classes:
            raise ValueError(f"class {class_id} is absent from the observer batch {classes}")
        class_index = classes.index(class_id)
        if batch_index is not None and class_index != batch_index:
            raise ValueError(
                f"batch index {batch_index} contains class {classes[batch_index]}, not {class_id}"
            )
        batch_index = class_index
    assert batch_index is not None
    class_id = classes[batch_index]

    selected = tuple(timesteps)
    if not selected:
        raise ValueError("at least one internal timestep is required")
    if any(type(timestep) is not int for timestep in selected):
        raise ValueError("internal timesteps must be integers")
    if len(set(selected)) != len(selected):
        raise ValueError("internal timesteps must be unique")
    if any(not 0 <= timestep < strict.NUM_SAMPLING_STEPS for timestep in selected):
        raise ValueError(
            f"internal timesteps must lie in [0,{strict.NUM_SAMPLING_STEPS-1}]"
        )
    trace_t = context.arrays["internal_timestep"]
    row_by_t = {int(timestep): row for row, timestep in enumerate(trace_t.tolist())}
    try:
        rows = tuple(row_by_t[timestep] for timestep in selected)
    except KeyError as exc:  # Defensive: full observer traces contain all 250.
        raise RuntimeError(f"requested timestep is absent from validated trace: {exc.args[0]}") from exc
    return Target(batch_index, class_id, selected, rows)


def representation_directory(kind: str) -> str:
    if kind == "pred_xstart":
        return "pred_xstart"
    if kind == "x_t":
        return "x_t_NOISY_STATE"
    raise ValueError(f"unknown trace representation: {kind}")


def native_relative_path(kind: str, timestep: int) -> str:
    directory = representation_directory(kind)
    suffix = "_NOISY_STATE" if kind == "x_t" else ""
    return f"{directory}/native/internal_t{timestep:03d}{suffix}.png"


def sheet_relative_path(kind: str) -> str:
    directory = representation_directory(kind)
    suffix = "_NOISY_STATE" if kind == "x_t" else ""
    return f"{directory}/contact_sheet{suffix}.png"


def output_relative_paths(target: Target, include_noisy_x_t: bool) -> tuple[str, ...]:
    kinds = ("pred_xstart", "x_t") if include_noisy_x_t else ("pred_xstart",)
    paths: list[str] = []
    for kind in kinds:
        paths.extend(native_relative_path(kind, timestep) for timestep in target.timesteps)
        paths.append(sheet_relative_path(kind))
    return tuple(paths)


def build_frame_records(
    context: BundleContext,
    target: Target,
    *,
    include_noisy_x_t: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ordinal, (timestep, row) in enumerate(zip(target.timesteps, target.trace_rows, strict=True)):
        record: dict[str, Any] = {
            "ordinal": ordinal,
            "internal_timestep": timestep,
            "trace_row": row,
            "pred_xstart_latent_raw_sha256": _array_raw_sha256(
                context.arrays["pred_xstart"][row, target.batch_index]
            ),
        }
        if include_noisy_x_t:
            record["x_t_NOISY_STATE_latent_raw_sha256"] = _array_raw_sha256(
                context.arrays["x_t"][row, target.batch_index]
            )
        records.append(record)
    return records


def build_identity(
    context: BundleContext,
    target: Target,
    *,
    include_noisy_x_t: bool,
    columns: int,
    device: str,
    decode_batch_size: int,
    device_runtime: dict[str, Any],
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    observer_runner = Path(observer.__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "runner_source": {"path": str(runner), "sha256": strict.sha256_file(runner)},
        "role": "TRACE_VISUALIZATION_ONLY",
        "visualization_only": True,
        "automatic_image_quality_scoring": False,
        "automatic_ranking_or_selection": False,
        "alarm_or_threshold_selection": False,
        "sampling_or_resampling": False,
        "guidance": False,
        "intervention": False,
        "observer_bundle": {
            "root": str(context.root),
            "manifest_identity_sha256": context.manifest["identity_sha256"],
            "manifest_file_sha256": strict.sha256_file(context.root / MANIFEST_NAME),
            "results_payload_sha256": context.results["payload_sha256"],
            "results_file_sha256": strict.sha256_file(context.root / "results.json"),
            "completion_file_sha256": strict.sha256_file(context.root / COMPLETION_NAME),
            "trace_relative_path": observer.TRACE_NAME,
            "trace_sha256": context.results["trace"]["sha256"],
            "strict_observer_bundle_and_trace_math_validated": True,
            "observer_validator_source": {
                "path": str(observer_runner),
                "sha256": strict.sha256_file(observer_runner),
            },
        },
        "frozen_baseline_identity_sha256": context.baseline.identity_sha256,
        "target": {
            "batch_index": target.batch_index,
            "class_id": target.class_id,
            "internal_timesteps_in_display_order": list(target.timesteps),
            "trace_rows_in_display_order": list(target.trace_rows),
        },
        "frames": build_frame_records(
            context, target, include_noisy_x_t=include_noisy_x_t
        ),
        "representations": {
            "pred_xstart": {
                "included": True,
                "meaning": "model clean-latent prediction at the recorded current state",
            },
            "x_t": {
                "included": include_noisy_x_t,
                "display_name": "x_t — NOISY STATE",
                "separate_from_pred_xstart": True,
                "warning": "a VAE decode of the current noisy latent, not a clean prediction",
            },
        },
        "decoder": {
            "model_id": strict.VAE_MODEL_ID,
            "kind": strict.VAE_KIND,
            "revision": strict.VAE_REVISION,
            "snapshot": context.vae["snapshot"],
            "snapshot_files": context.vae["files"],
            "latent_divisor": strict.VAE_SCALING_FACTOR,
            "device": device,
            "device_runtime": device_runtime,
            "decode_batch_size": decode_batch_size,
            "dtype": "float32",
            "eval_mode": True,
            "inference_mode": True,
            "offline_only": True,
            "png_mapping": "clip decoded tensor to [-1,1], map to [0,255], round-to-nearest",
            "official_torchvision_float32_mapping_reproduced": True,
            "t0_pred_xstart_latent_exactly_equals_observer_final_latent": True,
            "redecoded_t0_pixel_equality_to_frozen_baseline_required": False,
            "redecoded_t0_pixel_equality_caveat": (
                "VAE backend and decode batch shape may change pixels at quantization boundaries; "
                "the real-run manifest reports the observed comparison instead of assuming equality"
            ),
        },
        "dependencies": visualizer_dependency_identity(
            query_cuda_runtime=(device == "cuda" and bool(device_runtime["runtime_queried"]))
        ),
        "layout": {
            "native_image_size": [strict.IMAGE_SIZE, strict.IMAGE_SIZE],
            "native_resampling": "none",
            "contact_sheet_columns": columns,
            "contact_sheet_cell_image_size": [strict.IMAGE_SIZE, strict.IMAGE_SIZE],
            "contact_sheet_resampling": "none",
            "contact_sheet_has_internal_timestep_labels": True,
        },
        "expected_outputs": list(output_relative_paths(target, include_noisy_x_t)),
        "no_overwrite": True,
    }
    payload["identity_sha256"] = _canonical_self_hash(payload, "identity_sha256")
    return payload


def decode_latents(
    latents: np.ndarray,
    *,
    vae_snapshot: Path,
    device_name: str,
    batch_size: int,
) -> np.ndarray:
    if latents.ndim != 4 or latents.shape[1:] != (
        strict.LATENT_CHANNELS,
        strict.LATENT_SIZE,
        strict.LATENT_SIZE,
    ):
        raise ValueError(f"latent batch has the wrong shape: {latents.shape}")
    if latents.dtype != np.float32 or not np.isfinite(latents).all():
        raise ValueError("decoder input must be finite float32 latent data")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("decoder device must be cpu or cuda")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    from diffusers.models import AutoencoderKL

    device = torch.device(device_name)
    vae = AutoencoderKL.from_pretrained(
        str(vae_snapshot),
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    vae.eval().requires_grad_(False)
    decoded: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(latents), batch_size):
            stop = min(len(latents), start + batch_size)
            latent_batch = torch.from_numpy(
                np.ascontiguousarray(latents[start:stop])
            ).to(device=device, dtype=torch.float32)
            output = vae.decode(latent_batch / strict.VAE_SCALING_FACTOR).sample
            decoded.append(
                np.ascontiguousarray(output.detach().to(device="cpu", dtype=torch.float32).numpy())
            )
    result = np.ascontiguousarray(np.concatenate(decoded, axis=0), dtype=np.float32)
    expected = (len(latents), 3, strict.IMAGE_SIZE, strict.IMAGE_SIZE)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"VAE returned invalid decoded tensor: {result.shape}")
    del vae
    if device_name == "cuda":
        torch.cuda.synchronize()
    return result


def decoded_to_pil(decoded: np.ndarray) -> Image.Image:
    if decoded.shape != (3, strict.IMAGE_SIZE, strict.IMAGE_SIZE):
        raise ValueError(f"decoded frame has the wrong shape: {decoded.shape}")
    if decoded.dtype != np.float32 or not np.isfinite(decoded).all():
        raise ValueError("decoded frame must be finite float32")
    # Mirror torchvision.save_image(..., normalize=True,
    # value_range=(-1,1)) in float32, while retaining our provenance metadata.
    tensor = torch.from_numpy(np.ascontiguousarray(decoded)).clone()
    tensor.clamp_(min=-1.0, max=1.0).sub_(-1.0).div_(2.0)
    pixels = (
        tensor.mul(255.0)
        .add_(0.5)
        .clamp_(0.0, 255.0)
        .permute(1, 2, 0)
        .to(dtype=torch.uint8)
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def _native_metadata(
    identity: dict[str, Any],
    frame: dict[str, Any],
    *,
    kind: str,
) -> dict[str, str]:
    if kind == "pred_xstart":
        display = "pred_xstart"
        latent_hash = frame["pred_xstart_latent_raw_sha256"]
        noisy = "false"
    elif kind == "x_t":
        display = "x_t — NOISY STATE"
        latent_hash = frame["x_t_NOISY_STATE_latent_raw_sha256"]
        noisy = "true"
    else:
        raise ValueError(kind)
    target = identity["target"]
    return {
        "runner": RUNNER_NAME,
        "identity_sha256": identity["identity_sha256"],
        "representation": kind,
        "display_name": display,
        "is_noisy_state": noisy,
        "batch_index": str(target["batch_index"]),
        "class_id": str(target["class_id"]),
        "internal_timestep": str(frame["internal_timestep"]),
        "trace_row": str(frame["trace_row"]),
        "trace_sha256": identity["observer_bundle"]["trace_sha256"],
        "latent_raw_sha256": latent_hash,
        "native_resampling": "none",
        "visualization_only": "true",
    }


def _save_png(image: Image.Image, path: Path, metadata_fields: dict[str, str]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite PNG: {path}")
    if image.mode != "RGB":
        raise ValueError("visualization PNGs must be RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    for key, value in metadata_fields.items():
        metadata.add_text(key, value)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, path)


def contact_sheet_size(count: int, columns: int) -> tuple[int, int]:
    used_columns = min(columns, count)
    rows = math.ceil(count / used_columns)
    width = 2 * OUTER_MARGIN + used_columns * strict.IMAGE_SIZE + (used_columns - 1) * CELL_GAP
    height = (
        2 * OUTER_MARGIN
        + rows * (strict.IMAGE_SIZE + LABEL_HEIGHT)
        + (rows - 1) * CELL_GAP
    )
    return width, height


def render_contact_sheet(
    images: Sequence[Image.Image],
    identity: dict[str, Any],
    *,
    kind: str,
    columns: int,
) -> Image.Image:
    if not images:
        raise ValueError("contact sheet needs at least one image")
    target = identity["target"]
    timesteps = target["internal_timesteps_in_display_order"]
    if len(images) != len(timesteps):
        raise ValueError("image/timestep count mismatch")
    used_columns = min(columns, len(images))
    sheet = Image.new("RGB", contact_sheet_size(len(images), columns), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for ordinal, (image, timestep) in enumerate(zip(images, timesteps, strict=True)):
        if image.mode != "RGB" or image.size != (strict.IMAGE_SIZE, strict.IMAGE_SIZE):
            raise ValueError("contact-sheet source must be a native 256x256 RGB image")
        column = ordinal % used_columns
        row = ordinal // used_columns
        x = OUTER_MARGIN + column * (strict.IMAGE_SIZE + CELL_GAP)
        y = OUTER_MARGIN + row * (strict.IMAGE_SIZE + LABEL_HEIGHT + CELL_GAP)
        sheet.paste(image, (x, y))
        if kind == "pred_xstart":
            first_line = "pred_xstart"
            color = TEXT
        elif kind == "x_t":
            first_line = "x_t | NOISY STATE"
            color = NOISY_TEXT
        else:
            raise ValueError(kind)
        draw.text((x + 3, y + strict.IMAGE_SIZE + 2), first_line, fill=color)
        draw.text(
            (x + 3, y + strict.IMAGE_SIZE + 18),
            f"class {target['class_id']:04d} | batch {target['batch_index']} | internal t={timestep}",
            fill=color,
        )
    return sheet


def _sheet_metadata(identity: dict[str, Any], *, kind: str) -> dict[str, str]:
    target = identity["target"]
    if kind == "pred_xstart":
        display = "pred_xstart temporal contact sheet"
        noisy = "false"
    elif kind == "x_t":
        display = "x_t — NOISY STATE temporal contact sheet"
        noisy = "true"
    else:
        raise ValueError(kind)
    return {
        "runner": RUNNER_NAME,
        "identity_sha256": identity["identity_sha256"],
        "representation": kind,
        "display_name": display,
        "is_noisy_state": noisy,
        "batch_index": str(target["batch_index"]),
        "class_id": str(target["class_id"]),
        "internal_timesteps_in_display_order": ",".join(
            str(item) for item in target["internal_timesteps_in_display_order"]
        ),
        "trace_sha256": identity["observer_bundle"]["trace_sha256"],
        "contact_sheet_resampling": "none",
        "visualization_only": "true",
    }


def _expected_png_metadata(
    identity: dict[str, Any], relative: str
) -> dict[str, str]:
    for kind in ("pred_xstart", "x_t"):
        if relative == sheet_relative_path(kind):
            return _sheet_metadata(identity, kind=kind)
        for frame in identity["frames"]:
            if relative == native_relative_path(kind, frame["internal_timestep"]):
                return _native_metadata(identity, frame, kind=kind)
    raise RuntimeError(f"unexpected visualization path: {relative}")


def inspect_png_outputs(root: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    expected = tuple(identity["expected_outputs"])
    records: list[dict[str, Any]] = []
    for relative in sorted(expected):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/non-regular visualization PNG: {path}")
        with Image.open(path) as image:
            image.load()
            metadata = dict(image.info)
            pixels = image.tobytes()
            mode = image.mode
            size = tuple(image.size)
        if mode != "RGB":
            raise RuntimeError(f"visualization PNG is not RGB: {path}")
        is_sheet = relative.endswith("contact_sheet.png") or "contact_sheet_NOISY_STATE.png" in relative
        expected_size = (
            contact_sheet_size(len(identity["frames"]), identity["layout"]["contact_sheet_columns"])
            if is_sheet
            else (strict.IMAGE_SIZE, strict.IMAGE_SIZE)
        )
        if size != expected_size:
            raise RuntimeError(f"visualization PNG size changed: {path}: {size} != {expected_size}")
        expected_metadata = _expected_png_metadata(identity, relative)
        if metadata != expected_metadata:
            raise RuntimeError(f"visualization PNG provenance changed: {path}")
        records.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": strict.sha256_file(path),
                "pixel_sha256": _sha256_bytes(pixels),
                "mode": mode,
                "size": list(size),
            }
        )
    return records


def _validate_closed_output_tree(root: Path, identity: dict[str, Any]) -> None:
    expected_files = {
        (root / relative).absolute() for relative in identity["expected_outputs"]
    } | {(root / MANIFEST_NAME).absolute(), (root / COMPLETION_NAME).absolute()}
    representation_dirs = {
        (root / Path(relative).parts[0]).absolute()
        for relative in identity["expected_outputs"]
    }
    expected_dirs = {root.absolute()} | representation_dirs | {
        (directory / "native").absolute() for directory in representation_dirs
    }
    actual_files: set[Path] = set()
    actual_dirs: set[Path] = {root.absolute()}
    for path in root.rglob("*"):
        absolute = path.absolute()
        if path.is_symlink():
            raise RuntimeError(f"visualization output contains a symlink: {path}")
        if path.is_file():
            actual_files.add(absolute)
        elif path.is_dir():
            actual_dirs.add(absolute)
        else:
            raise RuntimeError(f"visualization output contains a special filesystem entry: {path}")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise RuntimeError(
            "visualization tree is not closed; "
            f"missing_files={sorted(expected_files-actual_files)[:2]}, "
            f"extra_files={sorted(actual_files-expected_files)[:2]}, "
            f"missing_dirs={sorted(expected_dirs-actual_dirs)[:2]}, "
            f"extra_dirs={sorted(actual_dirs-expected_dirs)[:2]}"
        )


def validate_completed_output(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"visualization output must be a non-symlink directory: {root}")
    for metadata_name in (MANIFEST_NAME, COMPLETION_NAME):
        metadata_path = root / metadata_name
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise RuntimeError(f"visualization metadata is missing/non-regular: {metadata_path}")
    manifest = _read_self_hashed_json(root / MANIFEST_NAME, "payload_sha256")
    completion = _read_self_hashed_json(root / COMPLETION_NAME, "payload_sha256")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError("visualization manifest is not complete")
    if manifest.get("identity") != identity:
        raise RuntimeError("visualization identity changed")
    if identity.get("identity_sha256") != _canonical_self_hash(identity, "identity_sha256"):
        raise RuntimeError("visualization identity self-hash changed")
    runner = Path(__file__).resolve()
    if identity.get("runner_source") != {
        "path": str(runner),
        "sha256": strict.sha256_file(runner),
    }:
        raise RuntimeError("visualization was produced by a different runner source")
    records = inspect_png_outputs(root, identity)
    outputs_sha256 = strict.sha256_json(records)
    if manifest.get("outputs") != records or manifest.get("outputs_sha256") != outputs_sha256:
        raise RuntimeError("visualization output hashes changed")
    _validate_closed_output_tree(root, identity)
    fixed_completion = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "identity_sha256": identity["identity_sha256"],
        "manifest_payload_sha256": manifest["payload_sha256"],
        "manifest_file_sha256": strict.sha256_file(root / MANIFEST_NAME),
        "outputs_sha256": outputs_sha256,
        "output_count": len(records),
    }
    mismatches = {
        key: (completion.get(key), value)
        for key, value in fixed_completion.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"visualization completion links/hashes changed: {mismatches}")
    return manifest


def _selected_latents(context: BundleContext, target: Target, kind: str) -> np.ndarray:
    source = context.arrays[kind]
    selected = np.stack(
        [source[row, target.batch_index] for row in target.trace_rows], axis=0
    )
    return np.ascontiguousarray(selected, dtype=np.float32)


def compare_t0_to_frozen_baseline(
    context: BundleContext,
    target: Target,
    pred_images: Sequence[Image.Image],
) -> dict[str, Any]:
    if 0 not in target.timesteps:
        return {
            "t0_requested": False,
            "pixel_equality_evaluated": False,
            "pixel_equality_required": False,
        }
    ordinal = target.timesteps.index(0)
    visualized = np.asarray(pred_images[ordinal], dtype=np.uint8)
    relative = strict.individual_relative_paths()[target.batch_index]
    baseline_path = context.baseline.root / relative
    with Image.open(baseline_path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (strict.IMAGE_SIZE, strict.IMAGE_SIZE):
            raise RuntimeError("validated baseline endpoint image properties changed")
        baseline = np.asarray(image, dtype=np.uint8)
    absolute_difference = np.abs(
        visualized.astype(np.int16) - baseline.astype(np.int16)
    )
    return {
        "t0_requested": True,
        "pixel_equality_evaluated": True,
        "pixel_equality_required": False,
        "t0_pred_xstart_latent_exactly_equals_observer_final_latent": True,
        "baseline_relative_path": relative,
        "baseline_png_sha256": strict.sha256_file(baseline_path),
        "pixel_equal": bool(np.array_equal(visualized, baseline)),
        "differing_pixels": int(np.count_nonzero(np.any(absolute_difference != 0, axis=2))),
        "differing_channel_values": int(np.count_nonzero(absolute_difference)),
        "maximum_absolute_channel_difference": int(absolute_difference.max(initial=0)),
        "interpretation": (
            "This audits decoder/backend reproducibility only; it is not an image-quality score. "
            "The latent endpoint equality is strict even if boundary pixels differ after re-decoding."
        ),
    }


def run_real(
    args: argparse.Namespace,
    *,
    context: BundleContext,
    target: Target,
    identity: dict[str, Any],
) -> None:
    outdir = args.outdir
    if os.path.lexists(outdir):
        raise RuntimeError(f"refusing to overwrite existing output path: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    kinds = ("pred_xstart", "x_t") if args.include_noisy_x_t else ("pred_xstart",)
    selected_by_kind = {
        kind: _selected_latents(context, target, kind) for kind in kinds
    }
    counts = [len(selected_by_kind[kind]) for kind in kinds]
    decoded_all = decode_latents(
        np.ascontiguousarray(
            np.concatenate([selected_by_kind[kind] for kind in kinds], axis=0),
            dtype=np.float32,
        ),
        vae_snapshot=Path(context.vae["snapshot"]),
        device_name=args.device,
        batch_size=args.decode_batch_size,
    )
    decoded_by_kind: dict[str, np.ndarray] = {}
    offset = 0
    for kind, count in zip(kinds, counts, strict=True):
        decoded_by_kind[kind] = np.ascontiguousarray(decoded_all[offset : offset + count])
        offset += count
    if offset != len(decoded_all):
        raise AssertionError("decoded representation split failed")

    with tempfile.TemporaryDirectory(
        prefix=f".{outdir.name}.staging-", dir=outdir.parent
    ) as temporary:
        staging = Path(temporary)
        images_by_kind: dict[str, list[Image.Image]] = {}
        for kind in kinds:
            native_images: list[Image.Image] = []
            for decoded, frame in zip(decoded_by_kind[kind], identity["frames"], strict=True):
                image = decoded_to_pil(decoded)
                native_images.append(image)
                relative = native_relative_path(kind, frame["internal_timestep"])
                _save_png(
                    image,
                    staging / relative,
                    _native_metadata(identity, frame, kind=kind),
                )
            sheet = render_contact_sheet(
                native_images, identity, kind=kind, columns=args.columns
            )
            _save_png(
                sheet,
                staging / sheet_relative_path(kind),
                _sheet_metadata(identity, kind=kind),
            )
            images_by_kind[kind] = native_images

        output_records = inspect_png_outputs(staging, identity)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "outputs": output_records,
            "outputs_sha256": strict.sha256_json(output_records),
            "endpoint_decode_audit": compare_t0_to_frozen_baseline(
                context, target, images_by_kind["pred_xstart"]
            ),
        }
        manifest["payload_sha256"] = _canonical_self_hash(manifest, "payload_sha256")
        strict.atomic_json_dump(manifest, staging / MANIFEST_NAME)
        completion: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "identity_sha256": identity["identity_sha256"],
            "manifest_payload_sha256": manifest["payload_sha256"],
            "manifest_file_sha256": strict.sha256_file(staging / MANIFEST_NAME),
            "outputs_sha256": manifest["outputs_sha256"],
            "output_count": len(output_records),
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        strict.atomic_json_dump(completion, staging / COMPLETION_NAME)
        validate_completed_output(staging, identity)
        _publish_directory_noreplace(staging, outdir)
    validate_completed_output(outdir, identity)
    print(
        json.dumps(
            {
                "complete": True,
                "outdir": str(outdir),
                "batch_index": target.batch_index,
                "class_id": target.class_id,
                "internal_timesteps": list(target.timesteps),
                "representations": list(kinds),
                "visualization_only": True,
                "automatic_quality_scoring": False,
                "intervention": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def dry_run(
    args: argparse.Namespace,
    *,
    context: BundleContext,
    target: Target,
    identity: dict[str, Any],
) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("dry-run unexpectedly initialized CUDA")
    payload = {
        "status": "dry-run",
        "runner": RUNNER_NAME,
        "strict_observer_bundle_and_trace_math_validated": True,
        "gpu_or_vae_model_loaded": False,
        "bundle": str(context.root),
        "trace_sha256": context.results["trace"]["sha256"],
        "target": identity["target"],
        "representations": identity["representations"],
        "decoder": identity["decoder"],
        "expected_outputs": identity["expected_outputs"],
        "outdir": str(args.outdir),
        "outdir_exists": os.path.lexists(args.outdir),
        "torch_cuda_initialized": torch.cuda.is_initialized(),
        "visualization_only": True,
        "automatic_image_quality_scoring": False,
        "automatic_ranking_or_selection": False,
        "intervention": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run_self_test() -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("self-test must start without CUDA initialization")
    # Exact display ordering and class/index cross-check semantics.
    synthetic_t = np.arange(strict.NUM_SAMPLING_STEPS - 1, -1, -1, dtype=np.int16)
    rows = tuple(int(np.flatnonzero(synthetic_t == t)[0]) for t in DEFAULT_INTERNAL_TIMESTEPS)
    if rows != tuple(strict.NUM_SAMPLING_STEPS - 1 - t for t in DEFAULT_INTERNAL_TIMESTEPS):
        raise AssertionError("internal-timestep row mapping failed")

    # Exercise the native no-resampling tensor-to-PNG mapping.
    decoded = np.zeros((3, strict.IMAGE_SIZE, strict.IMAGE_SIZE), dtype=np.float32)
    decoded[0] = -1.0
    decoded[1] = 0.0
    decoded[2] = 1.0
    image = decoded_to_pil(decoded)
    pixels = np.asarray(image)
    if image.size != (strict.IMAGE_SIZE, strict.IMAGE_SIZE) or image.mode != "RGB":
        raise AssertionError("native image dimensions/mode changed")
    if pixels[0, 0].tolist() != [0, 128, 255]:
        raise AssertionError("[-1,1] decoded-PNG mapping changed")

    # Exercise separate, visibly different pred_xstart/x_t contact-sheet labels
    # and self-hashed manifest/completion primitives without loading a VAE.
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "observer_bundle": {"trace_sha256": "a" * 64},
        "target": {
            "batch_index": 0,
            "class_id": strict.CLASS_IDS[0],
            "internal_timesteps_in_display_order": [249, 0],
            "trace_rows_in_display_order": [0, 249],
        },
        "frames": [
            {
                "ordinal": 0,
                "internal_timestep": 249,
                "trace_row": 0,
                "pred_xstart_latent_raw_sha256": "b" * 64,
                "x_t_NOISY_STATE_latent_raw_sha256": "c" * 64,
            },
            {
                "ordinal": 1,
                "internal_timestep": 0,
                "trace_row": 249,
                "pred_xstart_latent_raw_sha256": "d" * 64,
                "x_t_NOISY_STATE_latent_raw_sha256": "e" * 64,
            },
        ],
        "layout": {"contact_sheet_columns": 2},
        "expected_outputs": [],
    }
    identity["identity_sha256"] = _canonical_self_hash(identity, "identity_sha256")
    pred_sheet = render_contact_sheet([image, image], identity, kind="pred_xstart", columns=2)
    noisy_sheet = render_contact_sheet([image, image], identity, kind="x_t", columns=2)
    if pred_sheet.tobytes() == noisy_sheet.tobytes():
        raise AssertionError("x_t NOISY STATE labelling is not visibly distinct")
    with tempfile.TemporaryDirectory(prefix="dit-trace-visualizer-self-test-") as temporary:
        temporary_root = Path(temporary)
        root = temporary_root / "bundle"
        root.mkdir()
        runner = Path(__file__).resolve()
        identity["runner_source"] = {
            "path": str(runner),
            "sha256": strict.sha256_file(runner),
        }
        identity["expected_outputs"] = list(
            output_relative_paths(Target(0, strict.CLASS_IDS[0], (249, 0), (0, 249)), True)
        )
        identity["identity_sha256"] = _canonical_self_hash(identity, "identity_sha256")
        for kind, sheet in (("pred_xstart", pred_sheet), ("x_t", noisy_sheet)):
            for frame in identity["frames"]:
                _save_png(
                    image,
                    root / native_relative_path(kind, frame["internal_timestep"]),
                    _native_metadata(identity, frame, kind=kind),
                )
            _save_png(
                sheet,
                root / sheet_relative_path(kind),
                _sheet_metadata(identity, kind=kind),
            )
        records = inspect_png_outputs(root, identity)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "outputs": records,
            "outputs_sha256": strict.sha256_json(records),
        }
        manifest["payload_sha256"] = _canonical_self_hash(manifest, "payload_sha256")
        strict.atomic_json_dump(manifest, root / MANIFEST_NAME)
        completion: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "identity_sha256": identity["identity_sha256"],
            "manifest_payload_sha256": manifest["payload_sha256"],
            "manifest_file_sha256": strict.sha256_file(root / MANIFEST_NAME),
            "outputs_sha256": manifest["outputs_sha256"],
            "output_count": len(records),
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        strict.atomic_json_dump(completion, root / COMPLETION_NAME)
        validate_completed_output(root, identity)

        extra = root / "unexpected_empty_directory"
        extra.mkdir()
        try:
            validate_completed_output(root, identity)
        except RuntimeError as exc:
            if "tree is not closed" not in str(exc):
                raise
        else:
            raise AssertionError("closed-tree validator accepted an extra directory")
        extra.rmdir()

        alias = root / "unexpected_symlink"
        alias.symlink_to(root / MANIFEST_NAME)
        try:
            validate_completed_output(root, identity)
        except RuntimeError as exc:
            if "symlink" not in str(exc):
                raise
        else:
            raise AssertionError("closed-tree validator accepted a symlink")
        alias.unlink()

        staged = temporary_root / "publish_staged"
        published = temporary_root / "publish_final"
        staged.mkdir()
        _publish_directory_noreplace(staged, published)
        if staged.exists() or not published.is_dir():
            raise AssertionError("atomic no-replace publication failed")
        collision_source = temporary_root / "collision_staged"
        collision_source.mkdir()
        try:
            _publish_directory_noreplace(collision_source, published)
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic publication overwrote an existing directory")
        if not collision_source.is_dir() or not published.is_dir():
            raise AssertionError("failed no-replace publication changed source/destination")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    print(
        "self-test passed: timestep mapping, native PNG mapping, separate visible "
        "x_t NOISY STATE labels, closed tree, self-hashes, atomic no-overwrite, CPU-only"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="First-half official batch index. Defaults to 0 unless --class-id is supplied.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional class selector/cross-check within the frozen official batch.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=list(DEFAULT_INTERNAL_TIMESTEPS),
        metavar="T",
        help="Internal timesteps in desired display order.",
    )
    parser.add_argument(
        "--include-noisy-x-t",
        action="store_true",
        help="Also decode x_t into a separate, explicitly NOISY STATE product.",
    )
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--outdir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def normalize_output_path(
    requested: Path | None,
    *,
    context: BundleContext,
    target: Target,
) -> Path:
    if requested is None:
        name = f"{context.root.name}_trace_visualization_b{target.batch_index}_class{target.class_id:04d}"
        requested = context.root.parent / name
    requested = requested.expanduser().absolute()
    if os.path.lexists(requested) and requested.is_symlink():
        raise ValueError(f"output path must not be a symlink: {requested}")
    outdir = requested.resolve()
    protected = {
        "observer bundle": context.root,
        "frozen baseline": context.baseline.root,
        "DiT source": Path(context.source["root"]),
        "checkpoint": Path(context.checkpoint["path"]),
        "VAE snapshot": Path(context.vae["snapshot"]),
        "research repository": Path(__file__).resolve().parent.parent,
    }
    overlaps = [label for label, path in protected.items() if _paths_overlap(outdir, path)]
    if overlaps:
        raise ValueError("output path overlaps protected input/source path(s): " + ", ".join(overlaps))
    return outdir


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.bundle_dir is None:
        parser.error("--bundle-dir is required unless --self-test is used")
    if not 1 <= args.columns <= 8:
        parser.error("--columns must lie in [1,8]")
    if args.decode_batch_size < 1:
        parser.error("--decode-batch-size must be positive")
    if args.class_id is not None and not 0 <= args.class_id < strict.NUM_CLASSES:
        parser.error(f"--class-id must lie in [0,{strict.NUM_CLASSES-1}]")
    bundle_requested = args.bundle_dir.expanduser().absolute()
    if bundle_requested.is_symlink():
        parser.error(f"--bundle-dir must not be a symlink: {bundle_requested}")
    try:
        context = validate_bundle(bundle_requested.resolve())
        target = resolve_target(
            context,
            batch_index=args.batch_index,
            class_id=args.class_id,
            timesteps=args.timesteps,
        )
        args.outdir = normalize_output_path(args.outdir, context=context, target=target)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    if os.path.lexists(args.outdir):
        parser.error(f"no-overwrite target already exists: {args.outdir}")
    identity = build_identity(
        context,
        target,
        include_noisy_x_t=args.include_noisy_x_t,
        columns=args.columns,
        device=args.device,
        decode_batch_size=args.decode_batch_size,
        device_runtime=runtime_device_identity(
            args.device, query_runtime=not args.dry_run
        ),
    )
    if args.dry_run:
        dry_run(args, context=context, target=target, identity=identity)
        return 0
    run_real(args, context=context, target=target, identity=identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
