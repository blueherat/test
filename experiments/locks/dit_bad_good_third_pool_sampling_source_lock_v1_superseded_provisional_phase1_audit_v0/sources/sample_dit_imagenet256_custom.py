#!/usr/bin/env python3
"""Strict custom-class DiT-XL/2 ImageNet-256 baseline scanner.

This runner extends the frozen official-demo reproduction to an ordered list
of one through eight ImageNet class IDs.  It remains a baseline-only sampler:
there is no counterfactual process, detector, quality score, rejection,
rollback, attempt selection, or intervention.

The sampling path deliberately preserves the released ``sample.py`` contract:

* one global ``torch.manual_seed`` is set before model construction;
* B initial latents are sampled and copied into a 2B conditional/null batch;
* upstream ``forward_with_cfg`` guides only the first three epsilon channels;
* all 250 ancestral DDPM transitions draw a 2B noise tensor;
* the t=0 transition also consumes its 2B draw before masking it to zero; and
* only the first B final latents are decoded by the pinned MSE VAE.

Consequently, changing the number or order of classes changes the batch RNG
contract.  A singleton class run is not promised to reproduce the matching
image from an eight-class run.  With the exact official eight classes in their
official order, however, this code follows the frozen reproduction statement
for statement and is intended to be pixel-identical under the same software
and hardware environment.

Real runs fail closed on the DiT revision/source hashes, checkpoint hash, MSE
VAE snapshot hashes, helper-runner hash, single-process environment, and output
directory state.  Completed outputs are immutable and are validated on rerun.
Each run writes an official-style grid and one native 256x256 PNG per class.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

# Do not create bytecode inside the frozen upstream checkout.
sys.dont_write_bytecode = True

import torch
from PIL import Image

try:  # Package and direct CLI imports.
    from . import reproduce_dit_imagenet256 as strict
except ImportError:  # pragma: no cover - exercised by direct CLI invocation.
    import reproduce_dit_imagenet256 as strict


# These are module-scope settings in upstream sample.py and in the frozen
# reproduction.  Keep them explicit here rather than relying on import effects.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


RUNNER_NAME = "sample_dit_imagenet256_custom"
MANIFEST_SCHEMA = 1
COMPLETION_SCHEMA = 1
MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"
MAX_CLASSES = 8

# Start the challenge-class scanner from the official demo set.  The CLI is the
# source of truth for future challenge sets, so changing classes never requires
# editing this file.  This default also provides a full-batch reproduction
# control against reproduce_dit_imagenet256.py.
DEFAULT_CHALLENGE_CLASS_IDS = tuple(strict.CLASS_IDS)

# This helper owns the audited source/checkpoint/VAE validators used below.
# Pinning it makes this new runner fail closed if that implementation drifts.
STRICT_HELPER_SHA256 = "4d7d360c2621586fe3e751d7d73537784c436d5cee78be83448ce676d6fae746"


def parse_classes(value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError(
            "--classes must be a comma-separated list of 1-8 ImageNet class IDs"
        )
    try:
        classes = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--classes entries must be integers") from exc
    if not 1 <= len(classes) <= MAX_CLASSES:
        raise argparse.ArgumentTypeError(f"--classes must contain 1-{MAX_CLASSES} IDs")
    if any(class_id < 0 or class_id >= strict.NUM_CLASSES for class_id in classes):
        raise argparse.ArgumentTypeError("--classes IDs must lie in [0,999]")
    if len(set(classes)) != len(classes):
        raise argparse.ArgumentTypeError("--classes IDs must be unique")
    return classes


def validate_strict_helper() -> dict[str, Any]:
    path = Path(strict.__file__).resolve()
    if path.name != "reproduce_dit_imagenet256.py" or not path.is_file():
        raise RuntimeError(f"unexpected strict DiT helper module: {path}")
    digest = strict.sha256_file(path)
    if digest != STRICT_HELPER_SHA256:
        raise RuntimeError(
            "strict DiT helper source changed: "
            f"{digest} != {STRICT_HELPER_SHA256} ({path})"
        )
    return {"path": str(path), "sha256": digest}


def grid_size(image_count: int) -> tuple[int, int]:
    """Return torchvision make_grid dimensions for nrow=4, padding=2."""

    if not 1 <= image_count <= MAX_CLASSES:
        raise ValueError(f"image_count must lie in [1,{MAX_CLASSES}]")
    # torchvision.make_grid returns tensor.squeeze(0) for a singleton 4-D
    # batch, before constructing the padded multi-image grid.
    if image_count == 1:
        return strict.IMAGE_SIZE, strict.IMAGE_SIZE
    columns = min(4, image_count)
    rows = (image_count + columns - 1) // columns
    padding = 2
    return (
        strict.IMAGE_SIZE * columns + padding * (columns + 1),
        strict.IMAGE_SIZE * rows + padding * (rows + 1),
    )


def individual_relative_paths(classes: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        f"images/{index:02d}_class{int(class_id):04d}.png"
        for index, class_id in enumerate(classes)
    )


def expected_output_specs(
    classes: Sequence[int],
) -> dict[str, tuple[str, tuple[int, int]]]:
    specs: dict[str, tuple[str, tuple[int, int]]] = {
        "sample.png": ("RGB", grid_size(len(classes)))
    }
    specs.update(
        {
            relative: ("RGB", (strict.IMAGE_SIZE, strict.IMAGE_SIZE))
            for relative in individual_relative_paths(classes)
        }
    )
    return specs


def canonical_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--classes",
        ",".join(str(class_id) for class_id in args.classes),
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
    helper = validate_strict_helper()
    source = strict.validate_repository(args.dit_root, args.checkpoint)
    checkpoint = strict.validate_checkpoint(args.checkpoint)
    vae = strict.validate_vae_snapshot(args.vae_snapshot)
    classes = tuple(args.classes)
    count = len(classes)
    command = canonical_command(args)
    official_batch = classes == tuple(strict.CLASS_IDS)
    identity: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "runner": RUNNER_NAME,
        "runner_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": strict.sha256_file(Path(__file__).resolve()),
        },
        "strict_helper": helper,
        "baseline_only": True,
        "custom_class_scanner": True,
        "counterfactual_q": None,
        "path_likelihood_ratio": None,
        "rollback": None,
        "quality_scoring": None,
        "selection": None,
        "protocol": {
            "upstream_entry": "sample.py",
            "model": strict.MODEL_NAME,
            "image_size": strict.IMAGE_SIZE,
            "batch_size_before_duplication": count,
            "latent_shape_before_duplication": [
                count,
                strict.LATENT_CHANNELS,
                strict.LATENT_SIZE,
                strict.LATENT_SIZE,
            ],
            "latent_shape_after_duplication": [
                2 * count,
                strict.LATENT_CHANNELS,
                strict.LATENT_SIZE,
                strict.LATENT_SIZE,
            ],
            "num_classes": strict.NUM_CLASSES,
            "class_ids_ordered": list(classes),
            "class_count": count,
            "maximum_class_count": MAX_CLASSES,
            "null_class_id": strict.NULL_CLASS_ID,
            "num_sampling_steps": strict.NUM_SAMPLING_STEPS,
            "sampler": "ancestral DDPM (upstream p_sample_loop)",
            "clip_denoised": False,
            "cfg_scale": strict.CFG_SCALE,
            "cfg_epsilon_channels": 3,
            "vae": strict.VAE_KIND,
            "vae_scaling_factor": strict.VAE_SCALING_FACTOR,
            "global_torch_seed": args.seed,
            "one_global_seed_per_ordered_class_batch": True,
        },
        "official_demo_compatibility": {
            "official_class_ids_ordered": list(strict.CLASS_IDS),
            "same_ordered_eight_class_batch": official_batch,
            "statement_order_matches_frozen_reproduction": True,
            "pixel_equivalence_control_eligible": official_batch,
            "requires_same_seed_software_hardware_and_artifacts": True,
            "singleton_or_subset_is_not_official_batch_prefix_equivalent": True,
        },
        "rng_contract": {
            "torch_manual_seed_once_before_model_construction": args.seed,
            "initial_noise_shape": [
                count,
                strict.LATENT_CHANNELS,
                strict.LATENT_SIZE,
                strict.LATENT_SIZE,
            ],
            "duplicated_state_shape": [
                2 * count,
                strict.LATENT_CHANNELS,
                strict.LATENT_SIZE,
                strict.LATENT_SIZE,
            ],
            "transition_randn_like_calls": strict.NUM_SAMPLING_STEPS,
            "transition_noise_shape_each_call": [
                2 * count,
                strict.LATENT_CHANNELS,
                strict.LATENT_SIZE,
                strict.LATENT_SIZE,
            ],
            "terminal_t0_randn_consumed_then_masked": True,
            "second_half_transition_noises_consumed_then_state_discarded": True,
            "cfg_duplicates_first_half_before_every_model_call": True,
            "cfg_guides_first_three_epsilon_channels_only": True,
            "batch_shape_changes_future_rng_alignment": True,
            "descriptive_normal_scalar_draws": {
                "initial": count
                * strict.LATENT_CHANNELS
                * strict.LATENT_SIZE**2,
                "all_transitions_including_t0": (
                    strict.NUM_SAMPLING_STEPS
                    * 2
                    * count
                    * strict.LATENT_CHANNELS
                    * strict.LATENT_SIZE**2
                ),
                "discarded_second_half_transitions_including_t0": (
                    strict.NUM_SAMPLING_STEPS
                    * count
                    * strict.LATENT_CHANNELS
                    * strict.LATENT_SIZE**2
                ),
            },
        },
        "outputs": {
            "official_style_grid": "sample.png",
            "grid_nrow": 4,
            "grid_pixel_size": list(grid_size(count)),
            "native_individuals": list(individual_relative_paths(classes)),
            "expected_png_count": 1 + count,
            "automatic_quality_scoring": False,
            "automatic_ranking_or_selection": False,
        },
        "source": source,
        "checkpoint": checkpoint,
        "vae_snapshot": vae,
        "dependencies": strict.dependency_identity(),
        "canonical_command": command,
        "canonical_command_sha256": strict.sha256_json(command),
    }
    return identity


def inspect_png(
    path: Path, expected_mode: str, expected_size: tuple[int, int]
) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = tuple(image.size)
        pixel_digest = strict.sha256_bytes(image.tobytes())
    if mode != expected_mode or size != expected_size:
        raise RuntimeError(
            f"unexpected PNG properties for {path}: mode={mode}, size={size}, "
            f"expected mode={expected_mode}, size={expected_size}"
        )
    return {
        "bytes": path.stat().st_size,
        "sha256": strict.sha256_file(path),
        "pixel_sha256": pixel_digest,
        "mode": mode,
        "size": list(size),
    }


def collect_output_records(
    outdir: Path,
    classes: Sequence[int],
    *,
    allow_metadata: bool,
) -> list[dict[str, Any]]:
    specs = expected_output_specs(classes)
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


def validate_completed_output(
    outdir: Path, identity: dict[str, Any], classes: Sequence[int]
) -> None:
    manifest_path = outdir / MANIFEST_NAME
    completion_path = outdir / COMPLETION_NAME
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError(
            f"non-empty output directory is incomplete; refusing to overwrite: {outdir}"
        )
    manifest = strict.load_json(manifest_path)
    completion = strict.load_json(completion_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError(f"manifest is not a supported complete record: {manifest_path}")
    if manifest.get("identity") != identity:
        raise RuntimeError("existing output identity differs from this locked invocation")
    identity_sha256 = strict.sha256_json(identity)
    if manifest.get("identity_sha256") != identity_sha256:
        raise RuntimeError("manifest identity SHA-256 is invalid")
    records = collect_output_records(outdir, classes, allow_metadata=True)
    records_sha256 = strict.sha256_json(records)
    if manifest.get("outputs") != records:
        raise RuntimeError("one or more output PNG records or hashes have changed")
    if manifest.get("outputs_sha256") != records_sha256:
        raise RuntimeError("manifest output aggregate SHA-256 is invalid")
    expected_completion_keys = {
        "schema",
        "identity_sha256",
        "manifest_sha256",
        "outputs_sha256",
        "output_count",
    }
    if set(completion) != expected_completion_keys:
        raise RuntimeError("completion record key set is invalid")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise RuntimeError("completion schema is unsupported")
    if completion.get("identity_sha256") != identity_sha256:
        raise RuntimeError("completion identity SHA-256 is invalid")
    if completion.get("manifest_sha256") != strict.sha256_file(manifest_path):
        raise RuntimeError("completion record does not authenticate manifest bytes")
    if completion.get("outputs_sha256") != records_sha256:
        raise RuntimeError("completion output aggregate SHA-256 is invalid")
    if completion.get("output_count") != len(records):
        raise RuntimeError("completion output count is invalid")
    print(
        f"validated completed custom DiT output: {outdir} "
        f"({len(records)} PNG files); no sampling run"
    )


def save_outputs(
    samples: torch.Tensor,
    outdir: Path,
    classes: Sequence[int],
    save_image: Any,
) -> None:
    # Identical grid arguments to upstream sample.py.  In particular, default
    # padding=2 is retained rather than hand-building the contact sheet.
    save_image(
        samples,
        outdir / "sample.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    images_dir = outdir / "images"
    images_dir.mkdir(parents=False, exist_ok=False)
    for sample, relative in zip(
        samples, individual_relative_paths(classes), strict=True
    ):
        save_image(
            sample,
            outdir / relative,
            nrow=1,
            padding=0,
            normalize=True,
            value_range=(-1, 1),
        )


def run_custom_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Call the pinned upstream model, CFG method, and DDPM loop."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DiT-XL/2 baseline scanner")
    strict.ensure_single_process()

    # Import before torch.manual_seed, matching sample.py and the frozen runner.
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    prior_grad_enabled = torch.is_grad_enabled()
    preexisting_upstream_modules = {
        name
        for name in sys.modules
        if name == "models"
        or name == "download"
        or name == "diffusion"
        or name.startswith("diffusion.")
    }
    if preexisting_upstream_modules:
        raise RuntimeError(
            "ambiguous pre-imported upstream module names: "
            + repr(sorted(preexisting_upstream_modules))
        )

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

        # From here through decoding, preserve statement order from sample.py.
        torch.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        rng_after_manual_seed = strict.cuda_rng_state_sha256()

        model = DiT_models[strict.MODEL_NAME](
            input_size=strict.LATENT_SIZE,
            num_classes=strict.NUM_CLASSES,
        ).to(device)
        state_dict = find_model(str(args.checkpoint))
        model.load_state_dict(state_dict)
        model.eval()
        diffusion = create_diffusion(str(strict.NUM_SAMPLING_STEPS))
        vae = AutoencoderKL.from_pretrained(
            str(args.vae_snapshot),
            local_files_only=True,
            use_safetensors=True,
        ).to(device)

        classes = tuple(args.classes)
        n = len(classes)
        z_initial = torch.randn(
            n,
            strict.LATENT_CHANNELS,
            strict.LATENT_SIZE,
            strict.LATENT_SIZE,
            device=device,
        )
        initial_noise_sha256 = strict.tensor_sha256(z_initial)
        rng_after_initial_noise = strict.cuda_rng_state_sha256()
        y_conditional = torch.tensor(classes, device=device)

        z = torch.cat([z_initial, z_initial], 0)
        y_null = torch.tensor([strict.NULL_CLASS_ID] * n, device=device)
        y = torch.cat([y_conditional, y_null], 0)
        model_kwargs = {"y": y, "cfg_scale": strict.CFG_SCALE}

        latent_samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
        rng_after_diffusion = strict.cuda_rng_state_sha256()
        latent_samples, discarded_half = latent_samples.chunk(2, dim=0)
        latent_sha256 = strict.tensor_sha256(latent_samples)
        discarded_final_half_sha256 = strict.tensor_sha256(discarded_half)
        samples = vae.decode(latent_samples / strict.VAE_SCALING_FACTOR).sample
        decoded_tensor_sha256 = strict.tensor_sha256(samples)
        save_outputs(samples, args.outdir, classes, save_image)
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
            "class_ids_ordered": list(classes),
            "automatic_quality_scoring": False,
            "automatic_ranking_or_selection": False,
        }
    finally:
        torch.set_grad_enabled(prior_grad_enabled)
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        for name in list(sys.modules):
            if (
                name == "models"
                or name == "download"
                or name == "diffusion"
                or name.startswith("diffusion.")
            ) and name not in preexisting_upstream_modules:
                sys.modules.pop(name, None)


def dry_run(args: argparse.Namespace) -> None:
    helper = validate_strict_helper()
    source = strict.validate_repository(args.dit_root, args.checkpoint)
    vae = strict.validate_vae_snapshot(args.vae_snapshot)
    checkpoint = strict.checkpoint_dry_probe(args.checkpoint)
    blockers: list[str] = []
    if not checkpoint["exists"]:
        blockers.append("checkpoint file is missing")
    elif not checkpoint["size_matches"]:
        blockers.append("checkpoint download is incomplete or has the wrong size")
    if not checkpoint["sha256_pinned"]:
        blockers.append("checkpoint SHA-256 is not pinned")
    official_batch = tuple(args.classes) == tuple(strict.CLASS_IDS)
    summary = {
        "status": "dry-run",
        "runner": RUNNER_NAME,
        "baseline_only": True,
        "automatic_quality_scoring": False,
        "configuration": {
            "model": strict.MODEL_NAME,
            "image_size": strict.IMAGE_SIZE,
            "class_ids_ordered": list(args.classes),
            "class_count": len(args.classes),
            "steps": strict.NUM_SAMPLING_STEPS,
            "sampler": "ancestral DDPM",
            "cfg_scale": strict.CFG_SCALE,
            "cfg_epsilon_channels": 3,
            "vae": strict.VAE_KIND,
            "global_seed": args.seed,
        },
        "rng_contract": {
            "initial_B_then_duplicate_to_2B": True,
            "transition_draw_shape": [
                2 * len(args.classes),
                strict.LATENT_CHANNELS,
                strict.LATENT_SIZE,
                strict.LATENT_SIZE,
            ],
            "transition_draw_count_including_t0": strict.NUM_SAMPLING_STEPS,
            "subset_runs_are_not_official_batch_prefix_equivalent": True,
        },
        "official_demo_compatibility": {
            "same_ordered_eight_class_batch": official_batch,
            "pixel_equivalence_control_eligible": official_batch,
        },
        "strict_helper": helper,
        "source": source,
        "checkpoint_probe": checkpoint,
        "vae_snapshot": vae,
        "expected_outputs": sorted(expected_output_specs(args.classes)),
        "expected_grid_size": list(grid_size(len(args.classes))),
        "outdir": str(args.outdir),
        "real_run_blockers": blockers,
        "static_inputs_ready": not blockers,
        "cuda_available": torch.cuda.is_available(),
        "canonical_command": canonical_command(args),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _toy_cfg_first_three(model_out: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    return torch.cat([torch.cat([half_eps, half_eps], 0), rest], 1)


def _toy_official_rng(seed: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    retained = torch.randn((batch_size, 4), generator=generator)
    for timestep in (2, 1, 0):
        noise_2b = torch.randn((2 * batch_size, 4), generator=generator)
        retained = retained + float(timestep != 0) * noise_2b[:batch_size]
    return retained, generator.get_state()


def run_self_test() -> None:
    assert DEFAULT_CHALLENGE_CLASS_IDS == tuple(strict.CLASS_IDS)
    assert parse_classes("207") == (207,)
    assert parse_classes("207, 360,387") == (207, 360, 387)
    assert parse_classes(",".join(map(str, strict.CLASS_IDS))) == tuple(strict.CLASS_IDS)
    for invalid in ("", "207,", "-1", "1000", "1,1", "1,2,3,4,5,6,7,8,9"):
        try:
            parse_classes(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:  # pragma: no cover - defensive assertion.
            raise AssertionError(f"invalid class list was accepted: {invalid!r}")

    assert [grid_size(n) for n in range(1, 9)] == [
        (256, 256),
        (518, 260),
        (776, 260),
        (1_034, 260),
        (1_034, 518),
        (1_034, 518),
        (1_034, 518),
        (1_034, 518),
    ]
    assert expected_output_specs(strict.CLASS_IDS) == strict.expected_output_specs()
    assert individual_relative_paths(strict.CLASS_IDS) == strict.individual_relative_paths()

    model_out = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[3.0]], [[40.0]], [[50.0]]],
            [[[10.0]], [[20.0]], [[30.0]], [[400.0]], [[500.0]]],
        ]
    )
    guided = _toy_cfg_first_three(model_out, strict.CFG_SCALE)
    assert torch.equal(guided[0, :3], torch.tensor([[[-26.0]], [[-52.0]], [[-78.0]]]))
    assert torch.equal(guided[0, 3:], model_out[0, 3:])
    assert torch.equal(guided[1, :3], guided[0, :3])
    assert torch.equal(guided[1, 3:], model_out[1, 3:])

    singleton, singleton_state = _toy_official_rng(91, 1)
    full, full_state = _toy_official_rng(91, 8)
    assert not torch.equal(singleton[0], full[0])
    assert not torch.equal(singleton_state, full_state)

    # Omitting only the masked t=0 draw preserves the value but changes RNG.
    generator = torch.Generator(device="cpu").manual_seed(91)
    no_terminal = torch.randn((1, 4), generator=generator)
    for _ in (2, 1):
        no_terminal = no_terminal + torch.randn((2, 4), generator=generator)[:1]
    assert torch.equal(singleton, no_terminal)
    assert not torch.equal(singleton_state, generator.get_state())

    classes = (207, 360, 387)
    with tempfile.TemporaryDirectory(prefix="dit-custom-runner-self-test-") as temporary:
        outdir = Path(temporary)
        Image.new("RGB", grid_size(len(classes)), (10, 20, 30)).save(
            outdir / "sample.png"
        )
        (outdir / "images").mkdir()
        for relative in individual_relative_paths(classes):
            Image.new("RGB", (strict.IMAGE_SIZE, strict.IMAGE_SIZE), (1, 2, 3)).save(
                outdir / relative
            )
        records = collect_output_records(outdir, classes, allow_metadata=False)
        assert len(records) == 1 + len(classes)
        identity = {
            "self_test": True,
            "runner": RUNNER_NAME,
            "class_ids_ordered": list(classes),
        }
        identity_sha256 = strict.sha256_json(identity)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "outputs": records,
            "outputs_sha256": strict.sha256_json(records),
        }
        manifest_path = outdir / MANIFEST_NAME
        strict.atomic_json_dump(manifest, manifest_path)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "identity_sha256": identity_sha256,
            "manifest_sha256": strict.sha256_file(manifest_path),
            "outputs_sha256": strict.sha256_json(records),
            "output_count": len(records),
        }
        strict.atomic_json_dump(completion, outdir / COMPLETION_NAME)
        validate_completed_output(outdir, identity, classes)

        (outdir / individual_relative_paths(classes)[0]).unlink()
        try:
            validate_completed_output(outdir, identity, classes)
        except RuntimeError as exc:
            assert "output path mismatch" in str(exc)
        else:  # pragma: no cover - defensive assertion.
            raise AssertionError("missing output PNG was not rejected")

    assert strict.sha256_file(Path(strict.__file__).resolve()) == STRICT_HELPER_SHA256
    print(
        "self-test passed: custom class parsing/grid geometry, official 2B/t=0 RNG, "
        "three-channel CFG, default-eight compatibility, and immutable output validation"
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(
        os.environ.get("EQVAE_DATA_ROOT", "/home/zhoushunyu/data/eqvae")
    )
    dit_root = data_root / "baselines/DiT"
    vae_snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / strict.VAE_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default=DEFAULT_CHALLENGE_CLASS_IDS,
        help=(
            "Ordered comma-separated ImageNet IDs (1-8). Defaults to the official "
            "eight-class challenge/control batch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="One global torch seed for the entire ordered class batch.",
    )
    parser.add_argument("--dit-root", type=Path, default=dit_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <dit-root>/pretrained_models/DiT-XL-2-256x256.pt.",
    )
    parser.add_argument("--vae-snapshot", type=Path, default=vae_snapshot)
    parser.add_argument("--outdir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate static inputs without hashing/loading the 2.7GB checkpoint or a GPU.",
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="Run CPU-only contract tests without external model artifacts.",
    )
    return parser


def normalize_paths(args: argparse.Namespace) -> None:
    raw_root = args.dit_root.expanduser().absolute()
    if os.path.lexists(raw_root) and raw_root.is_symlink():
        raise RuntimeError(f"DiT root must not be a symlink: {raw_root}")
    args.dit_root = raw_root.resolve()
    if args.checkpoint is None:
        args.checkpoint = (
            args.dit_root / "pretrained_models" / strict.CHECKPOINT_FILENAME
        )
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


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.seed < 0 or args.seed >= 1 << 63:
        parser.error("--seed must be in [0, 2^63 - 1]")
    if not isinstance(args.classes, tuple) or not 1 <= len(args.classes) <= MAX_CLASSES:
        parser.error(f"--classes must contain 1-{MAX_CLASSES} IDs")
    if args.outdir is None:
        parser.error("--outdir is required unless --self-test is used")
    for label, protected in (
        ("DiT source tree", args.dit_root),
        ("VAE snapshot", args.vae_snapshot),
        ("checkpoint", args.checkpoint),
    ):
        if _paths_overlap(args.outdir, protected):
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
    identity_sha256 = strict.sha256_json(identity)
    if args.outdir.exists() and any(args.outdir.iterdir()):
        validate_completed_output(args.outdir, identity, args.classes)
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
    strict.atomic_json_dump(running_manifest, args.outdir / MANIFEST_NAME)
    try:
        execution = run_custom_demo(args)
        outputs = collect_output_records(
            args.outdir, args.classes, allow_metadata=True
        )
        outputs_sha256 = strict.sha256_json(outputs)
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
                "dependencies": strict.dependency_identity(),
                "cuda_device_count_visible": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(
                    torch.cuda.current_device()
                ),
                "cuda_device_capability": list(torch.cuda.get_device_capability()),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
        }
        manifest_path = args.outdir / MANIFEST_NAME
        strict.atomic_json_dump(manifest, manifest_path)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "identity_sha256": identity_sha256,
            "manifest_sha256": strict.sha256_file(manifest_path),
            "outputs_sha256": outputs_sha256,
            "output_count": len(outputs),
        }
        strict.atomic_json_dump(completion, args.outdir / COMPLETION_NAME)
        validate_completed_output(args.outdir, identity, args.classes)
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
        strict.atomic_json_dump(failed, args.outdir / MANIFEST_NAME)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
