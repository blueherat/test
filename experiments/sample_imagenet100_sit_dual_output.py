"""Compare the three sampling paths of an ImageNet-100 dual-output SiT.

The clean, epsilon, and learned dynamic paths use the same checkpoint,
initial noise, labels, adaptive ODE settings, and SD-VAE decoder. Sampling is
unguided so classifier-free guidance cannot confound the prediction-path test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

try:
    from experiments.imagenet100_sit_dual_output import dual_output_velocities
    from experiments.sample_imagenet100_sit_flow import (
        DEFAULT_CLASS_MANIFEST,
        DEFAULT_LABELS,
        integrate_velocity,
        load_class_records,
        load_font,
        parse_int_list,
        tensor_to_pil,
    )
    from experiments.train_imagenet100_sit_dual_output import (
        PROTOCOL,
        create_dual_output_sit,
    )
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        SD_VAE_SCALING_FACTOR,
        load_official_sit_module,
        sha256_file,
    )
except ModuleNotFoundError:
    from imagenet100_sit_dual_output import dual_output_velocities
    from sample_imagenet100_sit_flow import (
        DEFAULT_CLASS_MANIFEST,
        DEFAULT_LABELS,
        integrate_velocity,
        load_class_records,
        load_font,
        parse_int_list,
        tensor_to_pil,
    )
    from train_imagenet100_sit_dual_output import PROTOCOL, create_dual_output_sit
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        SD_VAE_SCALING_FACTOR,
        load_official_sit_module,
        sha256_file,
    )


SamplingMode = Literal["x", "epsilon", "dynamic"]
DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "runs/sit-s-2_dual-output_seed0/checkpoints/step_00800000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "samples/sit-s-2_dual-output_seed0"
)


def parse_modes(value: str) -> list[SamplingMode]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"x", "epsilon", "dynamic"}
    if not modes or any(mode not in allowed for mode in modes):
        raise argparse.ArgumentTypeError(
            "modes must be a comma-separated subset of x,epsilon,dynamic"
        )
    if len(modes) != len(set(modes)):
        raise argparse.ArgumentTypeError("sampling modes must not repeat")
    return modes  # type: ignore[return-value]


def dual_output_velocity(
    model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    mode: SamplingMode,
    gate_activation: str,
    denominator_floor: float,
    autocast_dtype: torch.dtype | None,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dict[str, object]]:
    """Build one unguided path and collect rollout gate statistics."""

    diagnostics: dict[str, object] = {"nfe": 0, "gate_trace": []}

    def velocity(time_value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        diagnostics["nfe"] = int(diagnostics["nfe"]) + 1
        times = time_value.expand(len(state))
        if autocast_dtype is None:
            output = model(state, times, labels)
        else:
            with torch.autocast(device_type=state.device.type, dtype=autocast_dtype):
                output = model(state, times, labels)
        paths = dual_output_velocities(
            output,
            state=state,
            time_value=times,
            gate_activation=gate_activation,  # type: ignore[arg-type]
            denominator_floor=denominator_floor,
        )
        gate = paths["gate"]
        trace = diagnostics["gate_trace"]
        assert isinstance(trace, list)
        trace.append(
            {
                "t": float(time_value.item()),
                "mean": float(gate.mean().item()),
                "std": float(gate.std(unbiased=False).item()),
            }
        )
        return paths[mode].float()

    return velocity, diagnostics


def make_mode_sheet(
    rows: list[list[Image.Image]],
    modes: list[SamplingMode],
    class_names: list[str],
) -> Image.Image:
    if not rows or len(rows) != len(modes):
        raise ValueError("one image row is required for every sampling mode")
    if any(len(row) != len(class_names) for row in rows):
        raise ValueError("each row must match the class-name count")
    tile_width, tile_height = rows[0][0].size
    left_margin, top_margin = 130, 50
    canvas = Image.new(
        "RGB",
        (left_margin + tile_width * len(class_names), top_margin + tile_height * len(rows)),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(16)
    row_font = load_font(19)
    for column, name in enumerate(class_names):
        draw.text(
            (left_margin + column * tile_width + 5, 15),
            name.split(",", maxsplit=1)[0][:20],
            fill="black",
            font=header_font,
        )
    for row_index, (mode, images) in enumerate(zip(modes, rows, strict=True)):
        y = top_margin + row_index * tile_height
        draw.text((12, y + tile_height // 2 - 12), mode, fill="black", font=row_font)
        for column, image in enumerate(images):
            canvas.paste(image, (left_margin + column * tile_width, y))
    return canvas


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SiT sampling")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
    config = checkpoint["config"]
    sit_module, source_metadata = load_official_sit_module(
        Path(args.official_sit_repo), verify_source=args.verify_sit_source
    )
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("checkpoint and sampler use different official SiT revisions")
    model = create_dual_output_sit(
        sit_module,
        model_name=config["model_name"],
        cfg_dropout=float(config["cfg_dropout"]),
    )
    state_key = "ema" if args.weights == "ema" else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval().requires_grad_(False)

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", local_files_only=True
    )
    vae.to(device).eval().requires_grad_(False)

    label_values = list(args.labels)
    labels = torch.tensor(label_values, device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    initial_noise = torch.randn(
        len(labels), *LATENT_SHAPE, generator=generator, device=device
    )
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16
    decoded_rows: list[torch.Tensor] = []
    sample_rows: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)

    for mode in args.modes:
        velocity, diagnostics = dual_output_velocity(
            model,
            labels,
            mode=mode,
            gate_activation=config["gate_activation"],
            denominator_floor=float(config["denominator_floor"]),
            autocast_dtype=autocast_dtype,
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        latents = integrate_velocity(
            initial_noise.clone(),
            velocity,
            num_output_points=args.num_output_points,
            atol=args.atol,
            rtol=args.rtol,
        )
        decoded = vae.decode(latents / SD_VAE_SCALING_FACTOR).sample
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        decoded_rows.append(decoded.cpu())
        image_name = f"{mode}.png"
        save_image(
            decoded,
            output_dir / image_name,
            nrow=len(labels),
            normalize=True,
            value_range=(-1, 1),
            padding=0,
        )
        row = {
            "mode": mode,
            "elapsed_seconds": elapsed,
            "nfe": diagnostics["nfe"],
            "latent_mean": float(latents.mean().item()),
            "latent_std": float(latents.std().item()),
            "image": image_name,
            "gate_trace": diagnostics["gate_trace"],
        }
        sample_rows.append(row)
        print(json.dumps({"event": "sampled", **row}, sort_keys=True), flush=True)

    class_records = load_class_records(Path(args.class_manifest), label_values)
    comparison = make_mode_sheet(
        [tensor_to_pil(row) for row in decoded_rows],
        list(args.modes),
        [record["name"] for record in class_records],
    )
    comparison_path = output_dir / "mode_comparison.png"
    comparison.save(comparison_path, format="PNG", optimize=True)
    manifest = {
        "format": "eqvae_imagenet100_sit_dual_output_sample_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": args.weights,
        "model_name": config["model_name"],
        "seed": int(args.seed),
        "same_initial_noise_and_labels_across_modes": True,
        "guidance": False,
        "classes": class_records,
        "sampler": {
            "method": "dopri5",
            "interval": [0.0, 1.0],
            "num_output_points": int(args.num_output_points),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "denominator_floor": float(config["denominator_floor"]),
            "precision": args.precision,
        },
        "samples": sample_rows,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "comparison_image": comparison_path.name,
    }
    (output_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "comparison": str(comparison_path)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-sit-repo", type=Path, default=DEFAULT_OFFICIAL_SIT_REPO)
    parser.add_argument("--class-manifest", type=Path, default=DEFAULT_CLASS_MANIFEST)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--modes", type=parse_modes, default=["x", "epsilon", "dynamic"])
    parser.add_argument("--labels", type=parse_int_list, default=list(DEFAULT_LABELS))
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-sit-source", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
