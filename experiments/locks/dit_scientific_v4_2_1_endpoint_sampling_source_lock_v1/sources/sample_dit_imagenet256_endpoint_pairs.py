#!/usr/bin/env python3
"""Sample immutable endpoint-only DiT images with pair-keyed RNG streams.

This worker loads the frozen DiT-XL/2 and MSE VAE once, then samples singleton
``(global_seed, class_id)`` tasks.  Each task gets a domain-separated derived
63-bit torch seed.  The seed is reset immediately before its initial latent,
so task order, worker assignment, batching of the task list, resume, and retry
cannot change its random stream.  Different classes with the same global seed
do *not* share an initial latent or transition innovations.

The diffusion transition itself retains the released DiT baseline semantics:
the singleton latent is duplicated to a 2-path conditional/null CFG batch,
CFG changes only the first three epsilon channels, all 250 ancestral DDPM
steps draw a complete 2-path ``randn_like`` tensor (including t=0), and only
the first final path is decoded.  No trace, score, selection, or intervention
is computed.  Existing pair directories are either fully hash-validated and
reused or refused without overwrite.

This file is intended to be invoked from its immutable source-lock snapshot by
``run_dit_event_rich_endpoint_screen.py``.  It is not a public standalone
sampling interface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import torch
from PIL import Image


RNG_DOMAIN = "eqvae.dit.event-rich.endpoint.v1"
RNG_MODULUS = 1 << 63
PNG_NAME = "endpoint.png"
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
PAIR_SCHEMA_VERSION = 1


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def exclusive_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {path}")
    return path.resolve()


def derive_torch_seed(global_seed: int, class_id: int) -> int:
    """Map a scientific pair key to a stable non-negative signed-63-bit seed."""

    if type(global_seed) is not int or global_seed < 0:
        raise ValueError("global_seed must be a non-negative integer")
    if type(class_id) is not int or not 0 <= class_id < 1_000:
        raise ValueError("class_id must lie in [0,999]")
    message = f"{RNG_DOMAIN}\0{global_seed}\0{class_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % RNG_MODULUS


def pair_relative_directory(global_seed: int, class_id: int) -> str:
    return f"pairs/seed{global_seed:04d}_class{class_id:04d}"


def load_frozen_strict(source_lock: Path) -> ModuleType:
    path = require_regular(
        source_lock / "sources/reproduce_dit_imagenet256.py",
        "frozen strict reproduction helper",
    )
    spec = importlib.util.spec_from_file_location("_event_endpoint_frozen_strict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen strict helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_endpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"endpoint PNG is missing or indirect: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or tuple(image.size) != (256, 256):
            raise RuntimeError(
                f"endpoint PNG contract changed: mode={image.mode}, size={image.size}"
            )
        pixel_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
    return {
        "relative_path": PNG_NAME,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_sha256,
        "mode": "RGB",
        "size": [256, 256],
    }


def pair_identity(
    global_seed: int,
    class_id: int,
    source_lock: Path,
    sampling_protocol: Mapping[str, Any],
    strict: ModuleType,
) -> dict[str, Any]:
    derived = derive_torch_seed(global_seed, class_id)
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "runner": "sample_dit_imagenet256_endpoint_pairs",
        "sampling_source_lock": str(source_lock),
        "sampling_protocol_identity_sha256": sampling_protocol["identity_sha256"],
        "event_protocol_identity_sha256": sampling_protocol["event_protocol"][
            "identity_sha256"
        ],
        "baseline_only": True,
        "endpoint_only": True,
        "trace": None,
        "quality_score": None,
        "selection": None,
        "intervention": None,
        "pair_key": {"global_seed": global_seed, "class_id": class_id},
        "scientific_contract": {
            "model": strict.MODEL_NAME,
            "image_size": strict.IMAGE_SIZE,
            "sampler": "official 250-step ancestral DDPM",
            "sampling_steps": strict.NUM_SAMPLING_STEPS,
            "clip_denoised": False,
            "cfg_scale": strict.CFG_SCALE,
            "cfg_epsilon_channels": 3,
            "class_batch_size": 1,
            "duplicated_cfg_batch_size": 2,
            "vae": strict.VAE_KIND,
            "vae_scaling_factor": strict.VAE_SCALING_FACTOR,
        },
        "rng_contract": {
            "unit": "(global_seed,class_id)",
            "domain": RNG_DOMAIN,
            "derivation": (
                "uint64_be(first_8_bytes(SHA256(ASCII(domain + NUL + global_seed "
                "+ NUL + class_id)))) mod 2^63"
            ),
            "derived_torch_seed": derived,
            "manual_seed_timing": (
                "after frozen model/VAE load, immediately before singleton initial latent"
            ),
            "initial_noise_shape": [1, 4, 32, 32],
            "duplicated_state_shape": [2, 4, 32, 32],
            "transition_randn_like_calls": 250,
            "transition_noise_shape_each_call": [2, 4, 32, 32],
            "full_2B_randn_like_each_transition_including_t0": True,
            "terminal_t0_randn_consumed_then_masked": True,
            "second_half_transition_noises_consumed_then_state_discarded": True,
            "same_global_seed_classes_share_initial_noise": False,
            "same_global_seed_classes_share_transition_innovations": False,
            "task_order_worker_shard_resume_invariant_rng": True,
        },
    }


def validate_pair_output(
    root: Path,
    global_seed: int,
    class_id: int,
    source_lock: Path,
    sampling_protocol: Mapping[str, Any],
    strict: ModuleType,
) -> dict[str, Any]:
    relative = pair_relative_directory(global_seed, class_id)
    outdir = root / relative
    if not outdir.is_dir() or outdir.is_symlink():
        raise RuntimeError(f"pair output is missing or indirect: {outdir}")
    observed_names = {path.name for path in outdir.iterdir()}
    expected_names = {PNG_NAME, MANIFEST_NAME, COMPLETION_NAME}
    if observed_names != expected_names or any(path.is_symlink() for path in outdir.iterdir()):
        raise RuntimeError(
            f"pair output member set changed: {outdir}; observed={sorted(observed_names)}"
        )
    manifest_path = require_regular(outdir / MANIFEST_NAME, "pair manifest")
    completion_path = require_regular(outdir / COMPLETION_NAME, "pair completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = pair_identity(
        global_seed, class_id, source_lock, sampling_protocol, strict
    )
    identity_sha256 = canonical_sha256(identity)
    endpoint = inspect_endpoint(outdir / PNG_NAME)
    if (
        manifest.get("schema_version") != PAIR_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("identity_sha256") != identity_sha256
        or manifest.get("endpoint") != endpoint
        or completion
        != {
            "complete": True,
            "identity_sha256": identity_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "endpoint_sha256": endpoint["sha256"],
            "endpoint_pixel_sha256": endpoint["pixel_sha256"],
        }
    ):
        raise RuntimeError(f"completed pair failed full validation: {outdir}")
    return {
        "global_seed": global_seed,
        "class_id": class_id,
        "relative_directory": relative,
        "identity_sha256": identity_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "completion_sha256": sha256_file(completion_path),
        "endpoint_sha256": endpoint["sha256"],
        "endpoint_pixel_sha256": endpoint["pixel_sha256"],
        "endpoint_bytes": endpoint["bytes"],
    }


def publish_pair(
    samples: torch.Tensor,
    root: Path,
    global_seed: int,
    class_id: int,
    source_lock: Path,
    sampling_protocol: Mapping[str, Any],
    strict: ModuleType,
    save_image: Any,
) -> dict[str, Any]:
    relative = pair_relative_directory(global_seed, class_id)
    outdir = root / relative
    outdir.mkdir(parents=False, exist_ok=False)
    save_image(
        samples,
        outdir / PNG_NAME,
        nrow=1,
        padding=0,
        normalize=True,
        value_range=(-1, 1),
    )
    endpoint = inspect_endpoint(outdir / PNG_NAME)
    identity = pair_identity(
        global_seed, class_id, source_lock, sampling_protocol, strict
    )
    identity_sha256 = canonical_sha256(identity)
    manifest = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "endpoint": endpoint,
    }
    exclusive_json(outdir / MANIFEST_NAME, manifest)
    completion = {
        "complete": True,
        "identity_sha256": identity_sha256,
        "manifest_sha256": sha256_file(outdir / MANIFEST_NAME),
        "endpoint_sha256": endpoint["sha256"],
        "endpoint_pixel_sha256": endpoint["pixel_sha256"],
    }
    exclusive_json(outdir / COMPLETION_NAME, completion)
    return validate_pair_output(
        root, global_seed, class_id, source_lock, sampling_protocol, strict
    )


def parse_tasks(path: Path) -> tuple[tuple[int, int], ...]:
    value = load_json(require_regular(path, "worker task file"))
    rows = value.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("worker task file must contain a nonempty tasks list")
    tasks: list[tuple[int, int]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"global_seed", "class_id"}:
            raise RuntimeError("worker task row schema changed")
        pair = (row["global_seed"], row["class_id"])
        derive_torch_seed(*pair)
        tasks.append(pair)
    if len(set(tasks)) != len(tasks):
        raise RuntimeError("worker task file contains duplicate pairs")
    expected_hash = value.get("tasks_sha256")
    if expected_hash != canonical_sha256(rows):
        raise RuntimeError("worker task list hash is invalid")
    return tuple(tasks)


def load_models(
    strict: ModuleType, dit_root: Path, checkpoint: Path, vae_snapshot: Path
) -> tuple[Any, Any, Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DiT endpoint sampling")
    strict.ensure_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.chdir(dit_root)
    sys.path.insert(0, str(dit_root))
    from diffusion import create_diffusion
    from diffusers.models import AutoencoderKL
    from download import find_model
    from models import DiT_models
    from torchvision.utils import save_image

    imported = {
        "diffusion": Path(sys.modules["diffusion"].__file__).resolve(),
        "download": Path(sys.modules["download"].__file__).resolve(),
        "models": Path(sys.modules["models"].__file__).resolve(),
    }
    expected = {
        "diffusion": (dit_root / "diffusion/__init__.py").resolve(),
        "download": (dit_root / "download.py").resolve(),
        "models": (dit_root / "models.py").resolve(),
    }
    if imported != expected:
        raise RuntimeError(f"upstream import shadowing detected: {imported} != {expected}")
    device = torch.device("cuda")
    model = DiT_models[strict.MODEL_NAME](
        input_size=strict.LATENT_SIZE, num_classes=strict.NUM_CLASSES
    ).to(device)
    model.load_state_dict(find_model(str(checkpoint)))
    model.eval()
    diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
    vae = AutoencoderKL.from_pretrained(
        str(vae_snapshot), local_files_only=True, use_safetensors=True
    ).to(device)
    return model, diffusion, vae, save_image


def sample_one(
    global_seed: int,
    class_id: int,
    strict: ModuleType,
    model: Any,
    diffusion: Any,
    vae: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    derived = derive_torch_seed(global_seed, class_id)
    torch.manual_seed(derived)
    device = torch.device("cuda")
    rng_after_seed = strict.cuda_rng_state_sha256()
    initial = torch.randn(1, 4, 32, 32, device=device)
    rng_after_initial = strict.cuda_rng_state_sha256()
    state = torch.cat([initial, initial], dim=0)
    y = torch.tensor([class_id, strict.NULL_CLASS_ID], device=device)
    kwargs = {"y": y, "cfg_scale": strict.CFG_SCALE}
    latent = diffusion.p_sample_loop(
        model.forward_with_cfg,
        state.shape,
        state,
        clip_denoised=False,
        model_kwargs=kwargs,
        progress=False,
        device=device,
    )
    rng_after_diffusion = strict.cuda_rng_state_sha256()
    kept, discarded = latent.chunk(2, dim=0)
    decoded = vae.decode(kept / strict.VAE_SCALING_FACTOR).sample
    torch.cuda.synchronize()
    return decoded, {
        "derived_torch_seed": derived,
        "rng_state_sha256": {
            "after_pair_seed_reset": rng_after_seed,
            "after_initial_noise": rng_after_initial,
            "after_250_full_2B_transition_draws": rng_after_diffusion,
        },
        "tensor_sha256": {
            "initial_noise_b1": strict.tensor_sha256(initial),
            "final_latent_kept_b1": strict.tensor_sha256(kept),
            "final_latent_discarded_b1": strict.tensor_sha256(discarded),
            "decoded_sample_b1": strict.tensor_sha256(decoded),
        },
    }


def run_worker(args: argparse.Namespace) -> None:
    source_lock = require_directory(args.source_lock, "sampling source lock")
    protocol = load_json(source_lock / "sampling_protocol.json")
    strict = load_frozen_strict(source_lock)
    dit_root = require_directory(args.dit_root, "DiT repository")
    checkpoint = require_regular(args.checkpoint, "DiT checkpoint")
    vae_snapshot = require_directory(args.vae_snapshot, "VAE snapshot")
    if (
        strict.validate_repository(dit_root, checkpoint)
        != protocol.get("assets", {}).get("dit_repository")
        or strict.validate_checkpoint(checkpoint)
        != protocol.get("assets", {}).get("checkpoint")
        or strict.validate_vae_snapshot(vae_snapshot)
        != protocol.get("assets", {}).get("vae_snapshot")
    ):
        raise RuntimeError("worker assets differ from frozen sampling source lock")
    tasks = parse_tasks(args.tasks)
    allowed = {
        (int(seed), int(class_id))
        for seed in protocol["scientific_contract"]["global_seeds"]
        for class_id in protocol["scientific_contract"]["classes_ordered"]
    }
    if not set(tasks) <= allowed:
        raise RuntimeError("worker received a task outside the frozen pair axis")
    output_root = require_directory(args.output_root, "endpoint output root")
    pairs_root = require_directory(output_root / "pairs", "endpoint pairs root")

    reusable: list[tuple[int, int]] = []
    pending: list[tuple[int, int]] = []
    for task in tasks:
        outdir = output_root / pair_relative_directory(*task)
        if os.path.lexists(outdir):
            validate_pair_output(
                output_root, *task, source_lock, protocol, strict
            )
            reusable.append(task)
        else:
            pending.append(task)
    print(
        json.dumps(
            {"tasks": len(tasks), "reusable": len(reusable), "pending": len(pending)},
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        return

    prior_cwd = Path.cwd()
    prior_sys_path = list(sys.path)
    prior_grad = torch.is_grad_enabled()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    try:
        model, diffusion, vae, save_image = load_models(
            strict, dit_root, checkpoint, vae_snapshot
        )
        for index, (global_seed, class_id) in enumerate(pending, start=1):
            started = time.time()
            decoded, execution = sample_one(
                global_seed, class_id, strict, model, diffusion, vae
            )
            record = publish_pair(
                decoded,
                output_root,
                global_seed,
                class_id,
                source_lock,
                protocol,
                strict,
                save_image,
            )
            # Execution hashes are operational diagnostics only and are printed
            # to the immutable worker log; the minimum pair receipt binds pixels.
            print(
                json.dumps(
                    {
                        "completed": index,
                        "pending_total": len(pending),
                        "global_seed": global_seed,
                        "class_id": class_id,
                        "elapsed_seconds": round(time.time() - started, 3),
                        "record": record,
                        "execution": execution,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        torch.set_grad_enabled(prior_grad)
        os.chdir(prior_cwd)
        sys.path[:] = prior_sys_path


def run_self_test() -> None:
    expected = {
        (1000, 0): 3026363209052735318,
        (1000, 1): 3606479167075842380,
        (1011, 999): 8394018843514802193,
    }
    observed = {key: derive_torch_seed(*key) for key in expected}
    if observed != expected:
        raise AssertionError(f"pair RNG known-answer test changed: {observed}")
    axis = [(seed, class_id) for seed in range(1000, 1012) for class_id in range(84)]
    derived = [derive_torch_seed(*pair) for pair in axis]
    if len(set(derived)) != len(derived):
        raise AssertionError("derived RNG collision in synthetic 1008-pair axis")
    reordered = list(reversed(axis))
    if {pair: derive_torch_seed(*pair) for pair in axis} != {
        pair: derive_torch_seed(*pair) for pair in reordered
    }:
        raise AssertionError("pair RNG derivation depends on task order")
    print(
        "self-test passed: pair-keyed known answers, range checks, collision check, "
        "and task-order-invariant derivation; no GPU sampling"
    )


def run_smoke_test() -> None:
    """Exercise immutable endpoint receipts with a synthetic image only."""

    class SyntheticStrict:
        MODEL_NAME = "DiT-XL/2"
        IMAGE_SIZE = 256
        NUM_SAMPLING_STEPS = 250
        CFG_SCALE = 4.0
        VAE_KIND = "mse"
        VAE_SCALING_FACTOR = 0.18215

    protocol = {
        "identity_sha256": "1" * 64,
        "event_protocol": {"identity_sha256": "2" * 64},
    }
    with tempfile.TemporaryDirectory(prefix="dit-endpoint-pair-smoke-") as raw:
        root = Path(raw)
        (root / "pairs").mkdir()
        source_lock = root / "synthetic_source_lock"
        source_lock.mkdir()
        outdir = root / pair_relative_directory(1000, 7)
        outdir.mkdir()
        Image.new("RGB", (256, 256), (17, 29, 41)).save(outdir / PNG_NAME)
        endpoint = inspect_endpoint(outdir / PNG_NAME)
        identity = pair_identity(
            1000, 7, source_lock, protocol, SyntheticStrict
        )
        identity_sha256 = canonical_sha256(identity)
        manifest = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "endpoint": endpoint,
        }
        exclusive_json(outdir / MANIFEST_NAME, manifest)
        exclusive_json(
            outdir / COMPLETION_NAME,
            {
                "complete": True,
                "identity_sha256": identity_sha256,
                "manifest_sha256": sha256_file(outdir / MANIFEST_NAME),
                "endpoint_sha256": endpoint["sha256"],
                "endpoint_pixel_sha256": endpoint["pixel_sha256"],
            },
        )
        validate_pair_output(
            root, 1000, 7, source_lock, protocol, SyntheticStrict
        )
        Image.new("RGB", (256, 256), (18, 29, 41)).save(outdir / PNG_NAME)
        try:
            validate_pair_output(
                root, 1000, 7, source_lock, protocol, SyntheticStrict
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("modified endpoint PNG escaped full-hash validation")
    print(
        "smoke-test passed: synthetic endpoint receipt round trip and corruption "
        "rejection; no model load or GPU sampling"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dit-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae-snapshot", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.smoke_test:
        run_smoke_test()
        return 0
    required = (
        "source_lock",
        "tasks",
        "output_root",
        "dit_root",
        "checkpoint",
        "vae_snapshot",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing worker arguments: " + ", ".join(missing))
    run_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
