#!/usr/bin/env python3
"""Reproduce OpenAI's classifier-guided ADM ImageNet-64 sampler.

This is a baseline-only runner.  It implements the official 64x64 model and
classifier configuration, classifier scale 1, and 250-step *stochastic DDPM*
sampling.  It deliberately does not define a high-noise counterfactual Q and
does not compute a path likelihood ratio.

Each seed owns a deterministic ``torch.Generator`` stream.  Reusing a seed
across classes reuses the initial and reverse Gaussian innovations, matching
the paired class/seed protocol used by the EDM2 baseline in this repository.
Neural-network evaluations are always singleton evaluations even when several
paths are scheduled together with ``--batch``.  This correctness-first choice
makes a path independent of its neighbours and of batch grouping (subject to
the recorded PyTorch/CUDA platform's deterministic-kernel guarantees).

The final spaced timestep ``t=0`` is executed as the required deterministic
posterior-mean transition.  Unlike the upstream ``p_sample`` helper, this
runner does not draw an unused Gaussian and multiply it by zero at that step.

Official source and checkpoints:
https://github.com/openai/guided-diffusion
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Deterministic cuBLAS needs this value before the first CUDA context is made.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.PngImagePlugin import PngInfo


GUIDED_DIFFUSION_REVISION = "22e0df8183507e13a7813f8d38d51b072ca1e67c"
SEED_RNG_NAMESPACE = "eqvae-adm64-guided-seed-v1"
IMAGE_SIZE = 64
NUM_CLASSES = 1_000
NUM_SPACED_STEPS = 250
CLASSIFIER_SCALE = 1.0

SMOKE_CLASS_IDS = (207, 388, 949)  # golden retriever, giant panda, strawberry
SMOKE_SEEDS = (0, 1)

OFFICIAL_MODEL_CONFIG: dict[str, Any] = {
    "image_size": 64,
    "num_channels": 192,
    "num_res_blocks": 3,
    "num_heads": 4,
    "num_heads_upsample": -1,
    "num_head_channels": 64,
    "attention_resolutions": "32,16,8",
    "channel_mult": "",
    "dropout": 0.1,
    "class_cond": True,
    "use_checkpoint": False,
    "use_scale_shift_norm": True,
    "resblock_updown": True,
    "use_fp16": True,
    "use_new_attention_order": True,
    "learn_sigma": True,
    "diffusion_steps": 1_000,
    "noise_schedule": "cosine",
    "timestep_respacing": "250",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": False,
    "rescale_learned_sigmas": False,
}

OFFICIAL_CLASSIFIER_CONFIG: dict[str, Any] = {
    "image_size": 64,
    "classifier_use_fp16": False,
    "classifier_width": 128,
    "classifier_depth": 4,
    "classifier_attention_resolutions": "32,16,8",
    "classifier_use_scale_shift_norm": True,
    "classifier_resblock_updown": True,
    "classifier_pool": "attention",
}


@dataclass(frozen=True)
class CheckpointSpec:
    filename: str
    byte_count: int
    sha256: str
    url: str


DIFFUSION_CHECKPOINT = CheckpointSpec(
    filename="64x64_diffusion.pt",
    byte_count=1_183_736_577,
    sha256="a18558f9a2499615a3ff9759ad12299690ad36ee3378c395adbb94855e2b634f",
    url="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt",
)
CLASSIFIER_CHECKPOINT = CheckpointSpec(
    filename="64x64_classifier.pt",
    byte_count=261_889_658,
    sha256="d5c4c240e4f0d36460f58520c2803a11490db7e5540e42d2ad75f0cf75bb3586",
    url="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt",
)

Pair = tuple[int, int]


@dataclass(frozen=True)
class Protocol:
    class_ids: tuple[int, ...]
    seeds: tuple[int, ...]

    @property
    def pairs(self) -> tuple[Pair, ...]:
        return tuple((class_id, seed) for class_id in self.class_ids for seed in self.seeds)


def parse_int_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ranges such as 1,4-7."""

    result: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)-(-?\d+)", token)
        if match:
            start, stop = int(match.group(1)), int(match.group(2))
            if stop < start:
                raise argparse.ArgumentTypeError(f"descending range is not allowed: {token}")
            result.extend(range(start, stop + 1))
        else:
            try:
                result.append(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid integer specification: {token}") from exc
    if not result:
        raise argparse.ArgumentTypeError("integer specification is empty")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("integer specification contains duplicates")
    return tuple(result)


def protocol_from_args(args: argparse.Namespace) -> Protocol:
    if args.protocol == "smoke":
        class_ids = args.classes if args.classes is not None else SMOKE_CLASS_IDS
        seeds = args.seeds if args.seeds is not None else SMOKE_SEEDS
    elif args.protocol == "custom":
        if args.classes is None or args.seeds is None:
            raise ValueError("--protocol custom requires both --classes and --seeds")
        class_ids, seeds = args.classes, args.seeds
    else:  # pragma: no cover - argparse constrains this value.
        raise AssertionError(args.protocol)

    if any(class_id < 0 or class_id >= NUM_CLASSES for class_id in class_ids):
        raise ValueError("ImageNet class IDs must be in [0, 999]")
    if any(seed < 0 or seed > (1 << 63) - 1 for seed in seeds):
        raise ValueError("seeds must be in [0, 2^63 - 1]")
    return Protocol(tuple(class_ids), tuple(seeds))


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
    return hashlib.sha256(encoded).hexdigest()


def sha256_python_tree(root: Path) -> str:
    """Hash Python source paths and bytes, including uncommitted source edits."""

    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no Python source files below {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_tracked_dirty(path: Path) -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_checkpoint(path: Path, spec: CheckpointSpec) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing official checkpoint: {path}\n"
            "Run experiments/download_cross_scale_baselines.sh first or pass its exact path."
        )
    byte_count = path.stat().st_size
    if byte_count != spec.byte_count:
        raise RuntimeError(
            f"wrong checkpoint size for {path}: {byte_count:,} != {spec.byte_count:,}"
        )
    digest = sha256_file(path)
    if digest != spec.sha256:
        raise RuntimeError(f"wrong checkpoint SHA-256 for {path}: {digest} != {spec.sha256}")
    return {
        "path": str(path.resolve()),
        "filename": spec.filename,
        "bytes": byte_count,
        "sha256": digest,
        "url": spec.url,
    }


def sample_stream_seed(seed: int) -> int:
    """Map a public sample seed to a stable, non-negative 63-bit torch seed."""

    payload = f"{SEED_RNG_NAMESPACE}\0{seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


class SeedRandomStreams:
    """One generator per path, initialized solely from its public seed."""

    def __init__(self, device: torch.device, pairs: Sequence[Pair]) -> None:
        self.device = device
        self.pairs = tuple(pairs)
        self.generators = [
            torch.Generator(device=device).manual_seed(sample_stream_seed(pair[1])) for pair in pairs
        ]
        self.draw_counts = [0 for _ in pairs]

    def randn(self, index: int, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        self.draw_counts[index] += 1
        return torch.randn(
            tuple(shape),
            generator=self.generators[index],
            device=self.device,
            dtype=dtype,
        )


def chunks(items: Sequence[Pair], size: int) -> Iterable[Sequence[Pair]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sample_batch_invariant(
    diffusion: Any,
    model_fn: Callable[..., torch.Tensor],
    cond_fn: Callable[..., torch.Tensor],
    pairs: Sequence[Pair],
    *,
    device: torch.device,
    channels: int = 3,
    image_size: int = IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Run ancestral DDPM while keeping every neural evaluation singleton.

    The singleton calls are intentional: merely assigning an independent RNG
    to each row is not enough for bitwise batch-grouping invariance, because a
    GPU convolution or classifier input-gradient kernel can change arithmetic
    when its batch shape changes.
    """

    if not pairs:
        raise ValueError("cannot sample an empty pair batch")
    if diffusion.num_timesteps < 1:
        raise ValueError("diffusion must contain at least one timestep")

    streams = SeedRandomStreams(device, pairs)
    states = torch.cat(
        [
            streams.randn(index, (1, channels, image_size, image_size), dtype)
            for index in range(len(pairs))
        ],
        dim=0,
    )

    for timestep in range(diffusion.num_timesteps - 1, -1, -1):
        next_states: list[torch.Tensor] = []
        for index, (class_id, _) in enumerate(pairs):
            # Shape is fixed at one regardless of --batch or neighbouring paths.
            x = states[index : index + 1]
            t = torch.tensor([timestep], dtype=torch.long, device=device)
            y = torch.tensor([class_id], dtype=torch.long, device=device)
            model_kwargs = {"y": y}
            with torch.no_grad():
                out = diffusion.p_mean_variance(
                    model_fn,
                    x,
                    t,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                )
                guided_mean = diffusion.condition_mean(
                    cond_fn,
                    out,
                    x,
                    t,
                    model_kwargs=model_kwargs,
                )
                if timestep > 0:
                    noise = streams.randn(index, x.shape, dtype)
                    x_next = guided_mean + torch.exp(0.5 * out["log_variance"]) * noise
                else:
                    # The final transition is deterministic.  Do not consume an
                    # unused Gaussian as upstream p_sample() does before masking.
                    x_next = guided_mean
            next_states.append(x_next.detach())
        states = torch.cat(next_states, dim=0)

    expected_draws = diffusion.num_timesteps  # initial draw + (T - 1) noisy transitions
    if streams.draw_counts != [expected_draws] * len(pairs):
        raise AssertionError(
            f"unexpected RNG consumption: {streams.draw_counts} != {expected_draws} per path"
        )
    return states, {
        "reverse_steps": int(diffusion.num_timesteps),
        "stochastic_reverse_steps": int(diffusion.num_timesteps - 1),
        "deterministic_final_steps": 1,
        "gaussian_draws_per_path_including_initial": int(expected_draws),
        "neural_eval_batch_size": 1,
    }


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def add_guided_diffusion_to_import_path(root: Path) -> tuple[Callable[..., Any], Callable[..., Any]]:
    required = root / "guided_diffusion" / "script_util.py"
    if not required.is_file():
        raise FileNotFoundError(f"not an OpenAI guided-diffusion checkout: {root}")
    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from guided_diffusion.script_util import create_classifier, create_model_and_diffusion

    return create_model_and_diffusion, create_classifier


def load_state_dict(path: Path) -> Any:
    """Load tensor-only official checkpoints across supported torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def load_official_models(
    root: Path,
    diffusion_checkpoint: Path,
    classifier_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, torch.nn.Module]:
    create_model_and_diffusion, create_classifier = add_guided_diffusion_to_import_path(root)

    model, diffusion = create_model_and_diffusion(**OFFICIAL_MODEL_CONFIG)
    model.load_state_dict(load_state_dict(diffusion_checkpoint), strict=True)
    model.to(device)
    model.convert_to_fp16()
    model.eval().requires_grad_(False)

    classifier = create_classifier(**OFFICIAL_CLASSIFIER_CONFIG)
    classifier.load_state_dict(load_state_dict(classifier_checkpoint), strict=True)
    classifier.to(device)
    # The official 64x64 command leaves classifier_use_fp16 at False.
    # Keep parameter requires_grad flags intact.  The released classifier uses
    # a custom activation-checkpoint backward that explicitly differentiates
    # with respect to both its input and parameters; disabling parameter grads
    # makes that official backward fail even though we only retain the input
    # gradient and never optimize or store parameter .grad buffers.
    classifier.eval()

    if diffusion.num_timesteps != NUM_SPACED_STEPS:
        raise RuntimeError(
            f"official timestep_respacing=250 produced {diffusion.num_timesteps} steps"
        )
    return model, diffusion, classifier


def make_guided_functions(
    model: torch.nn.Module, classifier: torch.nn.Module
) -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
    def model_fn(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is None:
            raise ValueError("class label y is required")
        return model(x, t, y)

    def cond_fn(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is None:
            raise ValueError("class label y is required")
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probabilities = F.log_softmax(logits, dim=-1)
            selected = log_probabilities[torch.arange(len(logits), device=x.device), y]
            gradient = torch.autograd.grad(selected.sum(), x_in)[0]
        return gradient * CLASSIFIER_SCALE

    return model_fn, cond_fn


def pair_path(output_dir: Path, pair: Pair) -> Path:
    class_id, seed = pair
    return output_dir / "images" / f"class_{class_id:04d}" / f"{seed:019d}.png"


def pixels_from_sample(sample: torch.Tensor) -> np.ndarray:
    image = ((sample + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return np.ascontiguousarray(image.permute(1, 2, 0).detach().cpu().numpy())


def save_sample_png(
    pixels: np.ndarray,
    path: Path,
    pair: Pair,
    manifest_identity_sha256: str,
    runner_sha256: str,
) -> None:
    class_id, seed = pair
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"unexpected output pixels: shape={pixels.shape}, dtype={pixels.dtype}")
    pixel_digest = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
    metadata = PngInfo()
    fields = {
        "experiment": "adm64_classifier_guided_reproduction",
        "class_id": str(class_id),
        "seed": str(seed),
        "sample_stream_seed": str(sample_stream_seed(seed)),
        "pixel_sha256": pixel_digest,
        "manifest_identity_sha256": manifest_identity_sha256,
        "runner_sha256": runner_sha256,
    }
    for key, value in fields.items():
        metadata.add_text(key, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG", pnginfo=metadata)
    os.replace(temporary, path)


def validate_sample_png(
    path: Path,
    pair: Pair,
    manifest_identity_sha256: str,
    runner_sha256: str,
) -> None:
    class_id, seed = pair
    expected_metadata = {
        "experiment": "adm64_classifier_guided_reproduction",
        "class_id": str(class_id),
        "seed": str(seed),
        "sample_stream_seed": str(sample_stream_seed(seed)),
        "manifest_identity_sha256": manifest_identity_sha256,
        "runner_sha256": runner_sha256,
    }
    try:
        with Image.open(path) as image:
            metadata = dict(image.info)
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"mode/size is {image.mode}/{image.size}")
            image.verify()
        with Image.open(path) as image:
            pixels = np.ascontiguousarray(np.asarray(image))
    except Exception as exc:
        raise RuntimeError(f"invalid existing PNG {path}: {exc}") from exc

    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"existing PNG metadata does not match its pair: {path}: {mismatches}")
    pixel_digest = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
    if metadata.get("pixel_sha256") != pixel_digest:
        raise RuntimeError(f"existing PNG pixel hash is invalid: {path}")


def validate_output_set(
    output_dir: Path,
    pairs: Sequence[Pair],
    manifest_identity_sha256: str,
    runner_sha256: str,
    *,
    require_all: bool,
) -> set[Pair]:
    expected_paths = {pair_path(output_dir, pair).resolve(): pair for pair in pairs}
    image_root = output_dir / "images"
    actual_paths = set(path.resolve() for path in image_root.rglob("*.png")) if image_root.exists() else set()
    unexpected = sorted(actual_paths - set(expected_paths))
    if unexpected:
        preview = ", ".join(str(path) for path in unexpected[:5])
        raise RuntimeError(f"output contains unexpected PNG files: {preview}")

    complete: set[Pair] = set()
    for path, pair in expected_paths.items():
        if path.is_file():
            validate_sample_png(path, pair, manifest_identity_sha256, runner_sha256)
            complete.add(pair)
        elif path.exists():
            raise RuntimeError(f"expected PNG path is not a regular file: {path}")
    if require_all and len(complete) != len(pairs):
        raise RuntimeError(f"only {len(complete)}/{len(pairs)} expected PNG files are complete")
    return complete


def create_or_validate_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot read existing manifest: {manifest_path}") from exc
        if existing != manifest:
            keys = sorted(
                key for key in set(existing) | set(manifest) if existing.get(key) != manifest.get(key)
            )
            raise RuntimeError(
                f"output directory has an incompatible manifest; differing keys: {', '.join(keys)}"
            )
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory without a manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest, manifest_path)


def validate_existing_completion(
    path: Path,
    *,
    manifest_identity_sha256: str,
    pair_set_sha256: str,
    total_expected: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        completion = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read existing completion record: {path}") from exc
    expected = {
        "complete": True,
        "manifest_identity_sha256": manifest_identity_sha256,
        "pair_set_sha256": pair_set_sha256,
        "total_expected": total_expected,
        "total_complete": total_expected,
    }
    mismatches = {
        key: (completion.get(key), value)
        for key, value in expected.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"existing completion record is incompatible: {mismatches}")
    return completion


def build_manifest(
    args: argparse.Namespace,
    protocol: Protocol,
    device: torch.device,
    diffusion_checkpoint_record: dict[str, Any],
    classifier_checkpoint_record: dict[str, Any],
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    runner_sha = sha256_file(runner_path)
    source_root = args.guided_diffusion_root.resolve()
    source_revision = git_revision(source_root)
    source_dirty = git_tracked_dirty(source_root)
    if source_revision != GUIDED_DIFFUSION_REVISION:
        raise RuntimeError(
            "guided-diffusion is not at the pinned official revision: "
            f"{source_revision} != {GUIDED_DIFFUSION_REVISION}"
        )
    if source_dirty:
        raise RuntimeError(
            "guided-diffusion has tracked source edits; use a clean pinned checkout for reproduction"
        )
    pairs = protocol.pairs
    pair_set_sha = sha256_json([[class_id, seed] for class_id, seed in pairs])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "adm64_classifier_guided_reproduction",
        "role": "stochastic_baseline_only_no_counterfactual_Q_no_path_LR",
        "protocol": args.protocol,
        "class_ids": list(protocol.class_ids),
        "seeds": list(protocol.seeds),
        "pair_order": "class_major_then_seed_major",
        "pair_set_sha256": pair_set_sha,
        "sample_count": len(pairs),
        "checkpoints": {
            "diffusion": diffusion_checkpoint_record,
            "classifier": classifier_checkpoint_record,
        },
        "official_model_config": OFFICIAL_MODEL_CONFIG,
        "official_classifier_config": OFFICIAL_CLASSIFIER_CONFIG,
        "sampler": {
            "name": "OpenAI classifier-guided ancestral DDPM",
            "use_ddim": False,
            "timestep_respacing": "250",
            "spaced_reverse_steps": NUM_SPACED_STEPS,
            "classifier_scale": CLASSIFIER_SCALE,
            "clip_denoised": True,
            "stochastic_reverse_steps": NUM_SPACED_STEPS - 1,
            "final_step": (
                "spaced t=0 guided posterior mean is executed; it is deterministic and no unused "
                "Gaussian is drawn; no LR is computed in this baseline runner"
            ),
        },
        "rng": {
            "owner": "one torch.Generator per path, initialized solely from sample seed",
            "seed_namespace": SEED_RNG_NAMESPACE,
            "stream_seed_derivation": (
                "low 63 bits of big-endian first 8 bytes of SHA256(namespace\\0seed)"
            ),
            "cross_class_pairing": (
                "the same public seed reuses identical initial and reverse Gaussian innovations "
                "across class IDs; inference must cluster or pair by seed"
            ),
            "initial_noise": "one [1,3,64,64] float32 Gaussian draw per pair",
            "reverse_noise": "one same-shape float32 Gaussian draw at each spaced t>0",
        },
        "batch_invariance": {
            "logical_batch_argument_affects_only_path_scheduling": True,
            "neural_eval_batch_size": 1,
            "reason": (
                "fix both RNG streams and CUDA model/classifier kernel input shape, so neighbours "
                "and logical batch grouping cannot alter a path"
            ),
        },
        "outputs": {
            "format": "RGB PNG",
            "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "quantization": "((x + 1) * 127.5).clamp(0,255).to(torch.uint8)",
            "path_template": "images/class_{class_id:04d}/{seed:019d}.png",
            "integrity": "pair identity, manifest identity, runner hash, and decoded-pixel SHA256 in PNG text",
        },
        "sources": {
            "repo": "https://github.com/openai/guided-diffusion",
            "root": str(source_root),
            "revision": source_revision,
            "expected_revision_at_setup": GUIDED_DIFFUSION_REVISION,
            "tracked_dirty": source_dirty,
            "guided_diffusion_python_tree_sha256": sha256_python_tree(
                source_root / "guided_diffusion"
            ),
            "official_readme_sampling_section": (
                "https://github.com/openai/guided-diffusion#classifier-guidance"
            ),
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "scope": (
                "same recorded software, CUDA device model, checkpoints, code, and singleton kernel shapes; "
                "cross-platform bitwise identity is not claimed"
            ),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", None),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "runner": {"path": str(runner_path), "sha256": runner_sha},
    }
    manifest["identity_sha256"] = sha256_json(manifest)
    return manifest


def run_sampling(args: argparse.Namespace, protocol: Protocol) -> None:
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the official FP16 ADM64 baseline requires an available CUDA device")

    configure_determinism()
    torch.cuda.set_device(device)
    diffusion_checkpoint_record = validate_checkpoint(args.model_path, DIFFUSION_CHECKPOINT)
    classifier_checkpoint_record = validate_checkpoint(
        args.classifier_path, CLASSIFIER_CHECKPOINT
    )
    manifest = build_manifest(
        args,
        protocol,
        device,
        diffusion_checkpoint_record,
        classifier_checkpoint_record,
    )
    output_dir = args.output_dir
    create_or_validate_manifest(output_dir, manifest)

    pairs = protocol.pairs
    manifest_identity = manifest["identity_sha256"]
    runner_sha = manifest["runner"]["sha256"]
    complete_pairs = validate_output_set(
        output_dir,
        pairs,
        manifest_identity,
        runner_sha,
        require_all=False,
    )
    completion_path = output_dir / "completion.json"
    existing_completion = validate_existing_completion(
        completion_path,
        manifest_identity_sha256=manifest_identity,
        pair_set_sha256=manifest["pair_set_sha256"],
        total_expected=len(pairs),
    )
    if existing_completion is not None:
        if len(complete_pairs) != len(pairs):
            raise RuntimeError("completion.json exists but one or more validated PNGs are missing")
        print(json.dumps(existing_completion, ensure_ascii=False, indent=2))
        return

    pending_pairs = [pair for pair in pairs if pair not in complete_pairs]
    start_time = time.monotonic()
    generated = 0
    if pending_pairs:
        model, diffusion, classifier = load_official_models(
            args.guided_diffusion_root,
            args.model_path,
            args.classifier_path,
            device,
        )
        model_fn, cond_fn = make_guided_functions(model, classifier)
        for logical_batch in chunks(pending_pairs, args.batch):
            samples, sampling_record = sample_batch_invariant(
                diffusion,
                model_fn,
                cond_fn,
                logical_batch,
                device=device,
            )
            if sampling_record["stochastic_reverse_steps"] != NUM_SPACED_STEPS - 1:
                raise AssertionError("unexpected reverse-step accounting")
            for index, pair in enumerate(logical_batch):
                pixels = pixels_from_sample(samples[index])
                save_sample_png(
                    pixels,
                    pair_path(output_dir, pair),
                    pair,
                    manifest_identity,
                    runner_sha,
                )
                generated += 1
            elapsed = time.monotonic() - start_time
            print(
                f"generated {generated}/{len(pending_pairs)} new paths "
                f"({len(complete_pairs)} already complete, {elapsed:.1f}s)",
                flush=True,
            )

    final_pairs = validate_output_set(
        output_dir,
        pairs,
        manifest_identity,
        runner_sha,
        require_all=True,
    )
    completion = {
        "complete": True,
        "manifest_identity_sha256": manifest_identity,
        "pair_set_sha256": manifest["pair_set_sha256"],
        "generated_this_run": generated,
        "already_complete": len(complete_pairs),
        "total_expected": len(pairs),
        "total_complete": len(final_pairs),
        "logical_batch_requested": args.batch,
        "neural_eval_batch_size": 1,
        "wall_seconds": time.monotonic() - start_time,
        "finished_at_unix": time.time(),
    }
    atomic_json_dump(completion, completion_path)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


class _FakeDiffusion:
    """Tiny deterministic arithmetic used only by --self-test."""

    num_timesteps = 5

    def p_mean_variance(
        self,
        model: Callable[..., torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        model_kwargs = kwargs["model_kwargs"]
        prediction = model(x, t, **model_kwargs)
        variance = torch.full_like(x, 0.04) + t.view(-1, 1, 1, 1).float() * 0.001
        return {
            "mean": x * 0.91 + prediction,
            "variance": variance,
            "log_variance": variance.log(),
            "pred_xstart": prediction,
        }

    def condition_mean(
        self,
        cond_fn: Callable[..., torch.Tensor],
        out: dict[str, torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return out["mean"] + out["variance"] * cond_fn(
            x, t, **kwargs["model_kwargs"]
        )


def run_self_test() -> None:
    assert parse_int_spec("1,4-6") == (1, 4, 5, 6)
    assert sample_stream_seed(2) == sample_stream_seed(2)
    assert sample_stream_seed(2) != sample_stream_seed(1)

    device = torch.device("cpu")
    pairs: tuple[Pair, ...] = ((3, 7), (3, 8), (9, 7), (11, 99))
    paired_streams = SeedRandomStreams(device, ((3, 7), (9, 7)))
    paired_noise_a = paired_streams.randn(0, (1, 1, 2, 2), torch.float32)
    paired_noise_b = paired_streams.randn(1, (1, 1, 2, 2), torch.float32)
    assert torch.equal(paired_noise_a, paired_noise_b)

    def fake_model(
        x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert y is not None and x.shape[0] == 1
        return t.view(-1, 1, 1, 1).float() * 0.002 + y.view(-1, 1, 1, 1) * 0.0001

    def fake_cond(
        x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert y is not None and x.shape[0] == 1
        return torch.full_like(x, 0.003) + y.view(-1, 1, 1, 1) * 0.00001

    def grouped(logical_batch_size: int) -> dict[Pair, torch.Tensor]:
        outputs: dict[Pair, torch.Tensor] = {}
        for logical_batch in chunks(pairs, logical_batch_size):
            samples, record = sample_batch_invariant(
                _FakeDiffusion(),
                fake_model,
                fake_cond,
                logical_batch,
                device=device,
                channels=1,
                image_size=2,
            )
            assert record == {
                "reverse_steps": 5,
                "stochastic_reverse_steps": 4,
                "deterministic_final_steps": 1,
                "gaussian_draws_per_path_including_initial": 5,
                "neural_eval_batch_size": 1,
            }
            outputs.update({pair: samples[index].clone() for index, pair in enumerate(logical_batch)})
        return outputs

    singleton = grouped(1)
    for logical_batch_size in (2, 3, 4):
        regrouped = grouped(logical_batch_size)
        assert singleton.keys() == regrouped.keys()
        for pair in pairs:
            assert torch.equal(singleton[pair], regrouped[pair]), (
                pair,
                logical_batch_size,
            )
    assert not torch.equal(singleton[(3, 7)], singleton[(3, 8)])

    # Exercise the atomic manifest and self-authenticating PNG resume path
    # without importing guided-diffusion or loading either checkpoint.
    with tempfile.TemporaryDirectory(prefix="adm64-runner-self-test-") as temporary:
        output_dir = Path(temporary)
        test_manifest: dict[str, Any] = {
            "schema_version": 0,
            "runner": {"sha256": "a" * 64},
        }
        test_manifest["identity_sha256"] = sha256_json(test_manifest)
        create_or_validate_manifest(output_dir, test_manifest)
        create_or_validate_manifest(output_dir, test_manifest)
        test_pair = (3, 7)
        test_pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        test_pixels[..., 0] = 17
        test_path = pair_path(output_dir, test_pair)
        save_sample_png(
            test_pixels,
            test_path,
            test_pair,
            test_manifest["identity_sha256"],
            test_manifest["runner"]["sha256"],
        )
        assert validate_output_set(
            output_dir,
            (test_pair,),
            test_manifest["identity_sha256"],
            test_manifest["runner"]["sha256"],
            require_all=True,
        ) == {test_pair}

    print(
        "self-test passed: seed-paired RNG, final-step accounting, grouping invariance, "
        "and strict PNG/manifest resume validation"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
    guided_root = data_root / "baselines" / "guided-diffusion"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("smoke", "custom"), default="smoke")
    parser.add_argument("--classes", type=parse_int_spec, default=None, help="e.g. 207,388,949")
    parser.add_argument("--seeds", type=parse_int_spec, default=None, help="e.g. 0-7,19")
    parser.add_argument("--guided-diffusion-root", type=Path, default=guided_root)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=guided_root / "checkpoints" / DIFFUSION_CHECKPOINT.filename,
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=guided_root / "checkpoints" / CLASSIFIER_CHECKPOINT.filename,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="logical paths scheduled together; neural evaluations remain singleton and invariant",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="resolve protocol without hashing/loading models")
    parser.add_argument("--self-test", action="store_true", help="run CPU tests without model weights")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    protocol = protocol_from_args(args)
    if args.output_dir is None:
        args.output_dir = (
            Path(os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae"))
            / "cross_scale_evidence"
            / "adm64_guided"
            / args.protocol
        )
    args.output_dir = args.output_dir.resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": args.protocol,
                    "class_ids": list(protocol.class_ids),
                    "seeds": list(protocol.seeds),
                    "sample_count": len(protocol.pairs),
                    "output_dir": str(args.output_dir),
                    "sampler": "stochastic DDPM, 250 spaced steps, classifier scale 1",
                    "logical_batch": args.batch,
                    "neural_eval_batch_size": 1,
                    "counterfactual_Q_or_LR": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_sampling(args, protocol)


if __name__ == "__main__":
    main()
