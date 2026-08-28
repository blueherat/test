#!/usr/bin/env python3
"""Strictly reproduce the official single-GPU DiT ImageNet-256 demo.

This is a baseline-only runner.  It executes the frozen upstream DiT-XL/2
model with the exact public ``sample.py`` demo configuration: the eight fixed
ImageNet classes, 250 ancestral DDPM steps, CFG scale 4.0, and the MSE
Stable-Diffusion VAE.  It does not define a counterfactual Q process, compute
a likelihood ratio, or implement rollback sampling.

The upstream objects are deliberately called instead of reimplementing the
sampler.  This preserves four easy-to-miss details of the released demo:

* the initial B latents are duplicated into a 2B conditional/null batch;
* classifier-free guidance is applied to only the first three epsilon channels;
* every transition draws 2B noises although the next model call discards the
  second half of the transitioned batch; and
* the terminal t=0 transition still draws a full noise tensor before its
  nonzero mask multiplies that noise by zero.

Each invocation has one global ``torch.manual_seed`` seed and produces the
official-style ``sample.png`` grid plus eight native 256x256 PNGs.  Completed
directories are immutable: rerunning an identical invocation validates the
source, artifacts, manifest, completion record, and every output hash, then
exits without sampling.  A non-empty incomplete or incompatible directory is
refused rather than overwritten.

The released checkpoint is pinned by its independently verified SHA-256.
A real run fails closed if the local file differs; ``--dry-run`` reports
input readiness without loading the checkpoint or a GPU.

Official source: https://github.com/facebookresearch/DiT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Iterable, Sequence

# Do not create untracked bytecode inside the frozen upstream checkout.
sys.dont_write_bytecode = True

import numpy as np
import PIL
import torch
from PIL import Image


# These two settings are at module scope in the official sample.py.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


DIT_REVISION = "ed81ce2229091fd4ecc9a223645f95cf379d582b"
MODEL_NAME = "DiT-XL/2"
IMAGE_SIZE = 256
LATENT_SIZE = 32
LATENT_CHANNELS = 4
NUM_CLASSES = 1_000
NUM_SAMPLING_STEPS = 250
CFG_SCALE = 4.0
VAE_KIND = "mse"
VAE_SCALING_FACTOR = 0.18215
CLASS_IDS = (207, 360, 387, 974, 88, 979, 417, 279)
NULL_CLASS_ID = 1_000

CHECKPOINT_FILENAME = "DiT-XL-2-256x256.pt"
CHECKPOINT_BYTES = 2_700_611_775
CHECKPOINT_SHA256: str | None = (
    "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"
)

VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
VAE_FILE_SPECS: dict[str, tuple[int, str]] = {
    "config.json": (
        547,
        "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e",
    ),
    "diffusion_pytorch_model.safetensors": (
        334_643_276,
        "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
    ),
}

PINNED_SOURCE_SHA256: dict[str, str] = {
    "sample.py": "e82038ecd6d6303208b6a9940705f9c3cd49219f5d76e2caa8d07813d086e2dc",
    "models.py": "1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca",
    "download.py": "f9ea211a0fa8f5ad18e3b3059dbcad38d8e6598a308bc0de744758bcb649a857",
    "diffusion/__init__.py": (
        "0128da38ee27ab91b78458809f9c50d188d7377b60cc73d3ee777bca8029425d"
    ),
    "diffusion/gaussian_diffusion.py": (
        "d7d095d98ff4666e565d0a854790992e24cc09a4ac75e8eaae9a4ee186a8887b"
    ),
    "diffusion/respace.py": (
        "9dde1492dd9d03e47caa12c7a7293dd7a5af1a79f3ddb0629f1692d97c856903"
    ),
}

MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
MANIFEST_SCHEMA = 1
COMPLETION_SCHEMA = 1


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON record: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON record must be an object: {path}")
    return value


def git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", b"")
        detail = output.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed below {root}: {detail}") from exc


def tracked_tree_sha256(root: Path) -> tuple[str, int]:
    """Hash the paths and working-tree bytes of every tracked file."""

    relative_names = [
        item.decode("utf-8", errors="surrogateescape")
        for item in git_bytes(root, "ls-files", "-z").split(b"\0")
        if item
    ]
    if not relative_names:
        raise RuntimeError(f"no tracked files found below upstream root: {root}")
    digest = hashlib.sha256()
    for relative in sorted(relative_names):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"tracked upstream file is missing: {path}")
        encoded_name = relative.encode("utf-8", errors="surrogateescape")
        contents = path.read_bytes()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest(), len(relative_names)


def validate_source_contract(root: Path) -> None:
    """Check the exact files that encode the official demo and RNG quirks."""

    for relative, expected_sha256 in PINNED_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing pinned DiT source file: {path}")
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"wrong source SHA-256 for {relative}: {observed} != {expected_sha256}"
            )

    fragments: dict[str, Sequence[str]] = {
        "sample.py": (
            "class_labels = [207, 360, 387, 974, 88, 979, 417, 279]",
            "z = torch.cat([z, z], 0)",
            "y = torch.cat([y, y_null], 0)",
            "model.forward_with_cfg, z.shape, z, clip_denoised=False",
            "samples, _ = samples.chunk(2, dim=0)",
            "samples = vae.decode(samples / 0.18215).sample",
            'save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))',
        ),
        "models.py": (
            "half = x[: len(x) // 2]",
            "combined = torch.cat([half, half], dim=0)",
            "eps, rest = model_out[:, :3], model_out[:, 3:]",
            "eps = torch.cat([half_eps, half_eps], dim=0)",
        ),
        "diffusion/gaussian_diffusion.py": (
            "noise = th.randn_like(x)",
            "(t != 0).float().view",
            'sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise',
        ),
    }
    for relative, required in fragments.items():
        source = (root / relative).read_text(encoding="utf-8")
        missing = [fragment for fragment in required if fragment not in source]
        if missing:
            raise RuntimeError(f"pinned upstream contract changed in {relative}: {missing}")


def validate_repository(root: Path, checkpoint: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing pinned DiT checkout: {root}")
    revision = git_bytes(root, "rev-parse", "HEAD").decode().strip()
    if revision != DIT_REVISION:
        raise RuntimeError(f"wrong DiT revision: {revision} != {DIT_REVISION}")

    status_records = [
        record.decode("utf-8", errors="surrogateescape")
        for record in git_bytes(
            root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
        ).split(b"\0")
        if record
    ]
    try:
        allowed_untracked = checkpoint.relative_to(root).as_posix()
    except ValueError:
        allowed_untracked = None
    unexpected: list[str] = []
    for record in status_records:
        if len(record) < 4:
            unexpected.append(record)
            continue
        code, relative = record[:2], record[3:]
        if code == "??" and relative == allowed_untracked:
            continue
        unexpected.append(record)
    if unexpected:
        raise RuntimeError(
            "DiT checkout is not the frozen source tree; unexpected git status: "
            + repr(unexpected[:8])
        )

    validate_source_contract(root)
    working_tree_sha256, tracked_file_count = tracked_tree_sha256(root)
    commit_tree = git_bytes(root, "rev-parse", "HEAD^{tree}").decode().strip()
    return {
        "url": "https://github.com/facebookresearch/DiT",
        "root": str(root),
        "revision": revision,
        "commit_tree": commit_tree,
        "working_tracked_tree_sha256": working_tree_sha256,
        "tracked_file_count": tracked_file_count,
        "tracked_and_index_clean": True,
        "allowed_untracked_checkpoint": allowed_untracked,
        "pinned_source_sha256": dict(PINNED_SOURCE_SHA256),
    }


def validate_checkpoint(path: Path) -> dict[str, Any]:
    if CHECKPOINT_SHA256 is None:
        raise RuntimeError(
            "CHECKPOINT_SHA256 is not pinned; refusing a real sampling run. "
            "Wait for the 2,700,611,775-byte download, independently compute SHA-256, "
            "and fill the constant in this runner."
        )
    if not path.is_file():
        raise FileNotFoundError(f"missing official DiT checkpoint: {path}")
    if path.is_symlink():
        raise RuntimeError(f"checkpoint must be a regular non-symlink file: {path}")
    byte_count = path.stat().st_size
    if byte_count != CHECKPOINT_BYTES:
        raise RuntimeError(
            f"wrong checkpoint size: {byte_count:,} != {CHECKPOINT_BYTES:,} ({path})"
        )
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"wrong checkpoint SHA-256: {digest} != {CHECKPOINT_SHA256} ({path})"
        )
    return {
        "filename": CHECKPOINT_FILENAME,
        "path": str(path),
        "bytes": byte_count,
        "sha256": digest,
        "url": (
            "https://dl.fbaipublicfiles.com/DiT/models/"
            "DiT-XL-2-256x256.pt"
        ),
    }


def checkpoint_dry_probe(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    byte_count = path.stat().st_size if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "observed_bytes": byte_count,
        "expected_bytes": CHECKPOINT_BYTES,
        "size_matches": byte_count == CHECKPOINT_BYTES,
        "sha256_pinned": CHECKPOINT_SHA256 is not None,
        "sha256_computed": False,
    }


def validate_vae_snapshot(snapshot: Path) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing local MSE VAE snapshot: {snapshot}")
    if snapshot.name != VAE_REVISION:
        raise RuntimeError(
            f"wrong local VAE snapshot revision: {snapshot.name} != {VAE_REVISION}"
        )
    actual_names = {entry.name for entry in snapshot.iterdir()}
    expected_names = set(VAE_FILE_SPECS)
    if actual_names != expected_names:
        raise RuntimeError(
            "local VAE snapshot file set changed; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    records: list[dict[str, Any]] = []
    for name in sorted(expected_names):
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"local VAE artifact is not a readable file: {path}")
        expected_bytes, expected_sha256 = VAE_FILE_SPECS[name]
        byte_count = path.stat().st_size
        if byte_count != expected_bytes:
            raise RuntimeError(
                f"wrong VAE file size for {name}: {byte_count:,} != {expected_bytes:,}"
            )
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise RuntimeError(
                f"wrong VAE SHA-256 for {name}: {digest} != {expected_sha256}"
            )
        records.append(
            {
                "name": name,
                "path": str(path),
                "resolved_path": str(path.resolve()),
                "bytes": byte_count,
                "sha256": digest,
            }
        )
    return {
        "model_id": VAE_MODEL_ID,
        "kind": VAE_KIND,
        "snapshot": str(snapshot),
        "revision": VAE_REVISION,
        "offline_only": True,
        "files": records,
    }


def package_version(name: str) -> str | None:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


def dependency_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "timm": package_version("timm"),
        "diffusers": package_version("diffusers"),
        "safetensors": package_version("safetensors"),
        "huggingface_hub": package_version("huggingface-hub"),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def canonical_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--seed",
        str(args.seed),
        "--dit-root",
        str(args.dit_root),
        "--checkpoint",
        str(args.checkpoint),
        "--vae-snapshot",
        str(args.vae_snapshot),
        "--outdir",
        str(args.outdir),
    ]


def build_identity(args: argparse.Namespace) -> dict[str, Any]:
    source = validate_repository(args.dit_root, args.checkpoint)
    checkpoint = validate_checkpoint(args.checkpoint)
    vae = validate_vae_snapshot(args.vae_snapshot)
    command = canonical_command(args)
    identity: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "runner": "reproduce_dit_imagenet256",
        "runner_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "baseline_only": True,
        "counterfactual_q": None,
        "path_likelihood_ratio": None,
        "rollback": None,
        "protocol": {
            "upstream_entry": "sample.py",
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "latent_shape_before_duplication": [len(CLASS_IDS), 4, 32, 32],
            "latent_shape_after_duplication": [2 * len(CLASS_IDS), 4, 32, 32],
            "num_classes": NUM_CLASSES,
            "class_ids": list(CLASS_IDS),
            "null_class_id": NULL_CLASS_ID,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "sampler": "ancestral DDPM (upstream p_sample_loop)",
            "clip_denoised": False,
            "cfg_scale": CFG_SCALE,
            "cfg_epsilon_channels": 3,
            "vae": VAE_KIND,
            "vae_scaling_factor": VAE_SCALING_FACTOR,
            "global_torch_seed": args.seed,
            "one_seed_per_eight_image_run": True,
        },
        "rng_contract": {
            "torch_manual_seed_once": args.seed,
            "initial_noise_shape": [len(CLASS_IDS), 4, 32, 32],
            "duplicated_state_shape": [2 * len(CLASS_IDS), 4, 32, 32],
            "transition_randn_like_calls": NUM_SAMPLING_STEPS,
            "transition_noise_shape_each_call": [2 * len(CLASS_IDS), 4, 32, 32],
            "terminal_t0_randn_consumed_then_masked": True,
            "second_half_transition_noises_consumed_then_state_discarded": True,
            "cfg_duplicates_first_half_before_every_model_call": True,
            "cfg_guides_first_three_epsilon_channels_only": True,
            "descriptive_normal_scalar_draws": {
                "initial": len(CLASS_IDS) * LATENT_CHANNELS * LATENT_SIZE**2,
                "all_transitions_including_t0": (
                    NUM_SAMPLING_STEPS
                    * 2
                    * len(CLASS_IDS)
                    * LATENT_CHANNELS
                    * LATENT_SIZE**2
                ),
                "discarded_second_half_transitions_including_t0": (
                    NUM_SAMPLING_STEPS
                    * len(CLASS_IDS)
                    * LATENT_CHANNELS
                    * LATENT_SIZE**2
                ),
            },
        },
        "outputs": {
            "official_grid": "sample.png",
            "official_grid_nrow": 4,
            "native_individuals": list(individual_relative_paths()),
            "expected_png_count": 1 + len(CLASS_IDS),
        },
        "source": source,
        "checkpoint": checkpoint,
        "vae_snapshot": vae,
        "dependencies": dependency_identity(),
        "canonical_command": command,
        "canonical_command_sha256": sha256_json(command),
    }
    return identity


def individual_relative_paths() -> tuple[str, ...]:
    return tuple(
        f"images/{index:02d}_class{class_id:04d}.png"
        for index, class_id in enumerate(CLASS_IDS)
    )


def expected_output_specs() -> dict[str, tuple[str, tuple[int, int]]]:
    specs: dict[str, tuple[str, tuple[int, int]]] = {
        # torchvision.make_grid default padding=2: 4 columns x 2 rows.
        "sample.png": ("RGB", (1_034, 518)),
    }
    specs.update(
        {relative: ("RGB", (IMAGE_SIZE, IMAGE_SIZE)) for relative in individual_relative_paths()}
    )
    return specs


def inspect_png(path: Path, expected_mode: str, expected_size: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = tuple(image.size)
        pixel_digest = sha256_bytes(image.tobytes())
    if mode != expected_mode or size != expected_size:
        raise RuntimeError(
            f"unexpected PNG properties for {path}: mode={mode}, size={size}, "
            f"expected mode={expected_mode}, size={expected_size}"
        )
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_digest,
        "mode": mode,
        "size": list(size),
    }


def collect_output_records(outdir: Path, *, allow_metadata: bool) -> list[dict[str, Any]]:
    specs = expected_output_specs()
    metadata = {MANIFEST_NAME, COMPLETION_NAME} if allow_metadata else set()
    actual: dict[str, Path] = {}
    unexpected: list[str] = []
    observed_directories: set[str] = set()
    for path in sorted(outdir.rglob("*")):
        relative = path.relative_to(outdir).as_posix()
        if path.is_symlink():
            unexpected.append(relative + " (symlink)")
            continue
        if path.is_dir():
            observed_directories.add(relative)
            continue
        if not path.is_file():
            unexpected.append(relative + " (special file)")
            continue
        if relative in metadata:
            continue
        if relative not in specs:
            unexpected.append(relative)
        else:
            actual[relative] = path
    expected_directories = {"images"}
    unexpected.extend(
        relative + " (directory)"
        for relative in sorted(observed_directories - expected_directories)
    )
    missing_directories = sorted(expected_directories - observed_directories)
    missing = sorted(set(specs) - set(actual))
    if missing or missing_directories or unexpected:
        raise RuntimeError(
            f"output path mismatch; missing={missing[:8]}, "
            f"missing_directories={missing_directories}, unexpected={unexpected[:8]}"
        )

    records: list[dict[str, Any]] = []
    for relative in sorted(specs):
        expected_mode, expected_size = specs[relative]
        record = {"relative_path": relative}
        record.update(inspect_png(actual[relative], expected_mode, expected_size))
        records.append(record)
    return records


def validate_completed_output(outdir: Path, identity: dict[str, Any]) -> None:
    manifest_path = outdir / MANIFEST_NAME
    completion_path = outdir / COMPLETION_NAME
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError(
            f"non-empty output directory is incomplete; refusing to overwrite: {outdir}"
        )
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError(f"manifest is not a supported complete record: {manifest_path}")
    if manifest.get("identity") != identity:
        raise RuntimeError("existing output identity differs from this locked invocation")
    identity_sha256 = sha256_json(identity)
    if manifest.get("identity_sha256") != identity_sha256:
        raise RuntimeError("manifest identity SHA-256 is invalid")
    records = collect_output_records(outdir, allow_metadata=True)
    records_sha256 = sha256_json(records)
    if manifest.get("outputs") != records:
        raise RuntimeError("one or more output PNG records or hashes have changed")
    if manifest.get("outputs_sha256") != records_sha256:
        raise RuntimeError("manifest output aggregate SHA-256 is invalid")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise RuntimeError("completion schema is unsupported")
    if completion.get("identity_sha256") != identity_sha256:
        raise RuntimeError("completion identity SHA-256 is invalid")
    if completion.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("completion record does not authenticate manifest bytes")
    if completion.get("outputs_sha256") != records_sha256:
        raise RuntimeError("completion output aggregate SHA-256 is invalid")
    if completion.get("output_count") != len(records):
        raise RuntimeError("completion output count is invalid")
    print(f"validated completed DiT output: {outdir} ({len(records)} PNG files); no sampling run")


def ensure_single_process() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (world_size, rank, local_rank) != (1, 0, 0):
        raise RuntimeError(
            "the official demo wrapper requires one process "
            "(WORLD_SIZE=1, RANK=0, LOCAL_RANK=0)"
        )
    if torch.distributed.is_initialized():
        raise RuntimeError("an initialized distributed process group is incompatible with sample.py")


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def cuda_rng_state_sha256() -> str:
    state = torch.cuda.get_rng_state()
    return sha256_bytes(state.cpu().numpy().tobytes())


def save_outputs(samples: torch.Tensor, outdir: Path, save_image: Any) -> None:
    # This call is intentionally identical to the final call in upstream sample.py.
    save_image(
        samples,
        outdir / "sample.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    images_dir = outdir / "images"
    images_dir.mkdir(parents=False, exist_ok=False)
    for sample, relative in zip(samples, individual_relative_paths(), strict=True):
        save_image(
            sample,
            outdir / relative,
            nrow=1,
            padding=0,
            normalize=True,
            value_range=(-1, 1),
        )


def run_official_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Call the exact pinned upstream model, CFG method, and DDPM loop."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DiT-XL/2 baseline reproduction")
    ensure_single_process()

    # Import before torch.manual_seed, matching sample.py's module-level imports.
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    prior_grad_enabled = torch.is_grad_enabled()
    preexisting_upstream_modules = {
        name for name in sys.modules if name == "models" or name == "download" or name == "diffusion" or name.startswith("diffusion.")
    }
    if preexisting_upstream_modules:
        raise RuntimeError(
            "ambiguous pre-imported upstream module names: "
            + repr(sorted(preexisting_upstream_modules))
        )

    # A local path plus both offline flags makes accidental network fallback impossible.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    try:
        os.chdir(args.dit_root)
        sys.path.insert(0, str(args.dit_root))
        from diffusion import create_diffusion
        from diffusers.models import AutoencoderKL
        from download import find_model
        from models import DiT_models
        from torchvision.utils import save_image

        imported_paths = {
            "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
            "download": Path(sys.modules["download"].__file__).resolve(),
            "models": Path(sys.modules["models"].__file__).resolve(),
        }
        expected_paths = {
            "diffusion": (args.dit_root / "diffusion/__init__.py").resolve(),
            "download": (args.dit_root / "download.py").resolve(),
            "models": (args.dit_root / "models.py").resolve(),
        }
        if imported_paths != expected_paths:
            raise RuntimeError(
                f"upstream import shadowing detected: {imported_paths} != {expected_paths}"
            )

        # From here through decoding, preserve the statement order in sample.py.
        torch.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        rng_after_manual_seed = cuda_rng_state_sha256()

        model = DiT_models[MODEL_NAME](
            input_size=LATENT_SIZE,
            num_classes=NUM_CLASSES,
        ).to(device)
        state_dict = find_model(str(args.checkpoint))
        model.load_state_dict(state_dict)
        model.eval()
        diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
        vae = AutoencoderKL.from_pretrained(
            str(args.vae_snapshot),
            local_files_only=True,
            use_safetensors=True,
        ).to(device)

        n = len(CLASS_IDS)
        z_initial = torch.randn(n, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, device=device)
        initial_noise_sha256 = tensor_sha256(z_initial)
        rng_after_initial_noise = cuda_rng_state_sha256()
        y_conditional = torch.tensor(CLASS_IDS, device=device)

        z = torch.cat([z_initial, z_initial], 0)
        y_null = torch.tensor([NULL_CLASS_ID] * n, device=device)
        y = torch.cat([y_conditional, y_null], 0)
        model_kwargs = {"y": y, "cfg_scale": CFG_SCALE}

        latent_samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
        rng_after_diffusion = cuda_rng_state_sha256()
        latent_samples, discarded_half = latent_samples.chunk(2, dim=0)
        latent_sha256 = tensor_sha256(latent_samples)
        discarded_final_half_sha256 = tensor_sha256(discarded_half)
        samples = vae.decode(latent_samples / VAE_SCALING_FACTOR).sample
        decoded_tensor_sha256 = tensor_sha256(samples)
        save_outputs(samples, args.outdir, save_image)
        torch.cuda.synchronize()

        return {
            "rng_state_sha256": {
                "after_manual_seed": rng_after_manual_seed,
                "after_initial_noise": rng_after_initial_noise,
                "after_250_transition_noise_draws": rng_after_diffusion,
            },
            "tensor_sha256": {
                "initial_noise_b": initial_noise_sha256,
                "final_latents_first_half_b": latent_sha256,
                "final_latents_discarded_second_half_b": discarded_final_half_sha256,
                "decoded_samples_b": decoded_tensor_sha256,
            },
            "observed_shapes": {
                "initial_noise": list(z_initial.shape),
                "duplicated_sampler_state": list(z.shape),
                "returned_first_half_latents": list(latent_samples.shape),
                "decoded_samples": list(samples.shape),
            },
        }
    finally:
        torch.set_grad_enabled(prior_grad_enabled)
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        for name in list(sys.modules):
            if name == "models" or name == "download" or name == "diffusion" or name.startswith("diffusion."):
                if name not in preexisting_upstream_modules:
                    sys.modules.pop(name, None)


def dry_run(args: argparse.Namespace) -> None:
    source = validate_repository(args.dit_root, args.checkpoint)
    vae = validate_vae_snapshot(args.vae_snapshot)
    checkpoint = checkpoint_dry_probe(args.checkpoint)
    blockers: list[str] = []
    if not checkpoint["exists"]:
        blockers.append("checkpoint file is missing")
    elif not checkpoint["size_matches"]:
        blockers.append("checkpoint download is incomplete or has the wrong size")
    if not checkpoint["sha256_pinned"]:
        blockers.append("CHECKPOINT_SHA256 is still None")
    summary = {
        "status": "dry-run",
        "baseline_only": True,
        "configuration": {
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "class_ids": list(CLASS_IDS),
            "steps": NUM_SAMPLING_STEPS,
            "sampler": "ancestral DDPM",
            "cfg_scale": CFG_SCALE,
            "vae": VAE_KIND,
            "global_seed": args.seed,
        },
        "source": source,
        "checkpoint_probe": checkpoint,
        "vae_snapshot": vae,
        "expected_outputs": sorted(expected_output_specs()),
        "outdir": str(args.outdir),
        "real_run_blockers": blockers,
        "static_inputs_ready": not blockers,
        "cuda_available": torch.cuda.is_available(),
        "canonical_command": canonical_command(args),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _toy_cfg_first_three(model_out: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    """Tiny CPU mirror used only to test the audited three-channel contract."""

    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    return torch.cat([torch.cat([half_eps, half_eps], 0), rest], 1)


def run_self_test() -> None:
    # Verify that exactly three channels are guided and the remaining channels
    # retain their separate conditional/null values.
    model_out = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[3.0]], [[40.0]], [[50.0]]],
            [[[10.0]], [[20.0]], [[30.0]], [[400.0]], [[500.0]]],
        ]
    )
    guided = _toy_cfg_first_three(model_out, 4.0)
    assert guided.shape == model_out.shape
    assert torch.equal(guided[0, :3], torch.tensor([[[-26.0]], [[-52.0]], [[-78.0]]]))
    assert torch.equal(guided[0, 3:], model_out[0, 3:])
    assert torch.equal(guided[1, :3], guided[0, :3])
    assert torch.equal(guided[1, 3:], model_out[1, 3:])

    # A terminal draw is consumed even when its multiplier is zero.
    generator_a = torch.Generator(device="cpu").manual_seed(91)
    x_a = torch.randn((2, 3), generator=generator_a)
    for timestep in (2, 1, 0):
        noise = torch.randn((4, 3), generator=generator_a)
        x_a = x_a + float(timestep != 0) * noise[:2]
    final_state_a = generator_a.get_state()

    generator_b = torch.Generator(device="cpu").manual_seed(91)
    x_b = torch.randn((2, 3), generator=generator_b)
    for timestep in (2, 1, 0):
        noise = torch.randn((4, 3), generator=generator_b)
        x_b = x_b + float(timestep != 0) * noise[:2]
    assert torch.equal(x_a, x_b)
    assert torch.equal(final_state_a, generator_b.get_state())

    # Drawing only the retained first half is not RNG-equivalent: the omitted
    # second-half draws shift every later innovation.
    generator_c = torch.Generator(device="cpu").manual_seed(91)
    x_c = torch.randn((2, 3), generator=generator_c)
    for timestep in (2, 1, 0):
        noise = torch.randn((2, 3), generator=generator_c)
        x_c = x_c + float(timestep != 0) * noise
    assert not torch.equal(x_a, x_c)
    assert not torch.equal(final_state_a, generator_c.get_state())

    # Omitting only the zero-multiplied terminal draw leaves the sample equal
    # but still leaves the generator in a different state.
    generator_d = torch.Generator(device="cpu").manual_seed(91)
    x_d = torch.randn((2, 3), generator=generator_d)
    for timestep in (2, 1):
        noise = torch.randn((4, 3), generator=generator_d)
        x_d = x_d + noise[:2]
    assert torch.equal(x_a, x_d)
    assert not torch.equal(final_state_a, generator_d.get_state())

    # Exercise exact output-set, manifest, completion, and immutable validation.
    with tempfile.TemporaryDirectory(prefix="dit-runner-self-test-") as temporary:
        outdir = Path(temporary)
        Image.new("RGB", (1_034, 518), (10, 20, 30)).save(outdir / "sample.png")
        (outdir / "images").mkdir()
        for relative in individual_relative_paths():
            Image.new("RGB", (256, 256), (1, 2, 3)).save(outdir / relative)
        records = collect_output_records(outdir, allow_metadata=False)
        assert len(records) == 9
        identity = {"self_test": True, "runner": "reproduce_dit_imagenet256"}
        identity_sha256 = sha256_json(identity)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "outputs": records,
            "outputs_sha256": sha256_json(records),
        }
        manifest_path = outdir / MANIFEST_NAME
        atomic_json_dump(manifest, manifest_path)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "identity_sha256": identity_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "outputs_sha256": sha256_json(records),
            "output_count": len(records),
        }
        atomic_json_dump(completion, outdir / COMPLETION_NAME)
        validate_completed_output(outdir, identity)

        (outdir / individual_relative_paths()[0]).unlink()
        try:
            validate_completed_output(outdir, identity)
        except RuntimeError as exc:
            assert "output path mismatch" in str(exc)
        else:  # pragma: no cover - defensive assertion.
            raise AssertionError("missing output PNG was not rejected")

    assert expected_output_specs()["sample.png"] == ("RGB", (1_034, 518))
    assert len(individual_relative_paths()) == len(CLASS_IDS) == 8
    assert CHECKPOINT_BYTES == 2_700_611_775
    print(
        "self-test passed: three-channel CFG, discarded/t=0 RNG consumption, "
        "and strict PNG/manifest/completion validation"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    dit_root = data_root / "baselines/DiT"
    vae_snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / VAE_REVISION
    )
    parser = argparse.ArgumentParser(
        description="Strict baseline-only reproduction of the official DiT ImageNet-256 demo."
    )
    parser.add_argument("--seed", type=int, default=0, help="One global torch seed for all 8 images.")
    parser.add_argument("--dit-root", type=Path, default=dit_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <dit-root>/pretrained_models/DiT-XL-2-256x256.pt.",
    )
    parser.add_argument("--vae-snapshot", type=Path, default=vae_snapshot)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate static inputs without hashing/loading DiT.")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only tests without external artifacts.")
    return parser


def normalize_paths(args: argparse.Namespace) -> None:
    raw_root = args.dit_root.expanduser().absolute()
    if os.path.lexists(raw_root) and raw_root.is_symlink():
        raise RuntimeError(f"DiT root must not be a symlink: {raw_root}")
    args.dit_root = raw_root.resolve()
    if args.checkpoint is None:
        args.checkpoint = args.dit_root / "pretrained_models" / CHECKPOINT_FILENAME
    else:
        requested_checkpoint = args.checkpoint.expanduser().absolute()
        if os.path.lexists(requested_checkpoint) and requested_checkpoint.is_symlink():
            raise RuntimeError(f"checkpoint must not be a symlink: {requested_checkpoint}")
        args.checkpoint = requested_checkpoint.resolve()
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    if args.outdir is not None:
        requested = args.outdir.expanduser().absolute()
        if os.path.lexists(requested) and requested.is_symlink():
            raise RuntimeError(f"output directory must not be a symlink: {requested}")
        args.outdir = requested.resolve()


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.seed < 0 or args.seed >= 1 << 63:
        parser.error("--seed must be in [0, 2^63 - 1]")
    if args.outdir is None:
        parser.error("--outdir is required unless --self-test is used")
    for label, protected in (
        ("DiT source tree", args.dit_root),
        ("VAE snapshot", args.vae_snapshot),
    ):
        overlaps = (
            args.outdir == protected
            or args.outdir.is_relative_to(protected)
            or protected.is_relative_to(args.outdir)
        )
        if overlaps:
            parser.error(f"--outdir must not overlap the protected {label}: {protected}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    validate_args(args, parser)

    if args.outdir.exists() and not args.outdir.is_dir():
        raise RuntimeError(f"output path is not a directory: {args.outdir}")
    if args.dry_run:
        dry_run(args)
        return 0

    identity = build_identity(args)
    identity_sha256 = sha256_json(identity)
    if args.outdir.exists() and any(args.outdir.iterdir()):
        validate_completed_output(args.outdir, identity)
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    if any(args.outdir.iterdir()):
        raise RuntimeError(f"output directory ceased to be empty: {args.outdir}")

    started_at = time.time()
    running_manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "started_unix": started_at,
    }
    atomic_json_dump(running_manifest, args.outdir / MANIFEST_NAME)
    try:
        execution = run_official_demo(args)
        outputs = collect_output_records(args.outdir, allow_metadata=True)
        outputs_sha256 = sha256_json(outputs)
        finished_at = time.time()
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "started_unix": started_at,
            "finished_unix": finished_at,
            "elapsed_seconds": finished_at - started_at,
            "execution": execution,
            "outputs": outputs,
            "outputs_sha256": outputs_sha256,
            "platform": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "dependencies": dependency_identity(),
                "cuda_device_count_visible": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda_device_capability": list(torch.cuda.get_device_capability()),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
        }
        manifest_path = args.outdir / MANIFEST_NAME
        atomic_json_dump(manifest, manifest_path)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "identity_sha256": identity_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "outputs_sha256": outputs_sha256,
            "output_count": len(outputs),
        }
        atomic_json_dump(completion, args.outdir / COMPLETION_NAME)
        validate_completed_output(args.outdir, identity)
    except BaseException as exc:
        failed = dict(running_manifest)
        failed.update(
            {
                "status": "failed",
                "failed_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_json_dump(failed, args.outdir / MANIFEST_NAME)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
