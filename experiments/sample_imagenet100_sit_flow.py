"""Sample the ImageNet-100 SiT flow baseline with the official ODE protocol.

The sampler intentionally mirrors the official SiT implementation:

* linear velocity is integrated from noise at ``t=0`` to data at ``t=1``;
* Dopri5 uses ``atol=1e-6`` and ``rtol=1e-3`` by default;
* classifier-free guidance is applied to the first three latent channels only;
* SD-VAE latents are decoded after division by ``0.18215``.

All compared CFG scales share the same initial noise and class labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageDraw, ImageFont
from torchdiffeq import odeint
from torchvision.utils import save_image

try:
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        load_official_sit_module,
        sha256_file,
    )
except ModuleNotFoundError:
    from train_imagenet100_sit_flow import (
        DEFAULT_OFFICIAL_SIT_REPO,
        LATENT_SHAPE,
        NUM_CLASSES,
        SD_VAE_SCALING_FACTOR,
        load_official_sit_module,
        sha256_file,
    )


DEFAULT_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "runs/sit-s-2_seed0/checkpoints/step_00100000.pt"
)
DEFAULT_CLASS_MANIFEST = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "imagenet100_cmc/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/"
    "samples/sit-s-2_step100000_seed0"
)
DEFAULT_LABELS = (0, 6, 13, 27, 42, 56, 70, 99)


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    if any(scale < 1.0 for scale in values):
        raise argparse.ArgumentTypeError("CFG scales must be at least 1.0")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    if any(label < 0 or label >= NUM_CLASSES for label in values):
        raise argparse.ArgumentTypeError(f"labels must be in [0, {NUM_CLASSES - 1}]")
    return values


def official_cfg_velocity(
    model: torch.nn.Module,
    labels: torch.Tensor,
    cfg_scale: float,
    *,
    autocast_dtype: torch.dtype | None,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dict[str, int]]:
    """Return the official SiT CFG velocity and a mutable NFE counter."""

    batch_size = labels.numel()
    null_labels = torch.full_like(labels, NUM_CLASSES)
    doubled_labels = torch.cat([labels, null_labels], dim=0)
    counter = {"nfe": 0}

    def velocity(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        counter["nfe"] += 1
        doubled_state = torch.cat([state, state], dim=0)
        doubled_t = t.expand(2 * batch_size)
        if autocast_dtype is None:
            prediction = model.forward_with_cfg(
                doubled_state, doubled_t, doubled_labels, float(cfg_scale)
            )
        else:
            with torch.autocast(
                device_type=state.device.type,
                dtype=autocast_dtype,
            ):
                prediction = model.forward_with_cfg(
                    doubled_state, doubled_t, doubled_labels, float(cfg_scale)
                )
        guided, _ = prediction.chunk(2, dim=0)
        return guided.float()

    return velocity, counter


def integrate_velocity(
    initial_noise: torch.Tensor,
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_output_points: int,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    """Run the official adaptive Dopri5 integration from ``t=0`` to ``t=1``."""

    if num_output_points < 2:
        raise ValueError("num_output_points must be at least 2")
    time_points = torch.linspace(
        0.0,
        1.0,
        num_output_points,
        device=initial_noise.device,
        dtype=torch.float32,
    )
    trajectory = odeint(
        velocity,
        initial_noise.float(),
        time_points,
        method="dopri5",
        atol=float(atol),
        rtol=float(rtol),
    )
    return trajectory[-1]


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = (
        images.detach()
        .float()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [Image.fromarray(image, mode="RGB") for image in images]


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_comparison_sheet(
    rows: list[list[Image.Image]],
    cfg_scales: list[float],
    class_names: list[str],
) -> Image.Image:
    if not rows or any(len(row) != len(class_names) for row in rows):
        raise ValueError("comparison rows must match the class-name count")
    tile_width, tile_height = rows[0][0].size
    left_margin = 120
    top_margin = 50
    canvas = Image.new(
        "RGB",
        (left_margin + tile_width * len(class_names), top_margin + tile_height * len(rows)),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(16)
    row_font = load_font(19)

    for column, name in enumerate(class_names):
        short_name = name.split(",", maxsplit=1)[0][:20]
        x = left_margin + column * tile_width + 5
        draw.text((x, 15), short_name, fill="black", font=header_font)

    for row_index, (scale, images) in enumerate(zip(cfg_scales, rows, strict=True)):
        y = top_margin + row_index * tile_height
        draw.text((12, y + tile_height // 2 - 12), f"CFG {scale:g}", fill="black", font=row_font)
        for column, image in enumerate(images):
            x = left_margin + column * tile_width
            canvas.paste(image, (x, y))
    return canvas


def load_class_records(path: Path, labels: list[int]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != NUM_CLASSES:
        raise ValueError(f"invalid ImageNet-100 class manifest: {path}")
    by_label = {int(record["label"]): record for record in classes}
    return [by_label[label] for label in labels]


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

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name")
    if model_name is None:
        raise ValueError("checkpoint is missing config.model_name")

    sit_module, source_metadata = load_official_sit_module(
        Path(args.official_sit_repo), verify_source=args.verify_sit_source
    )
    if checkpoint.get("official_sit") != source_metadata:
        raise ValueError("checkpoint and sampler use different official SiT source revisions")
    model = sit_module.SiT_models[model_name](
        input_size=LATENT_SHAPE[-1],
        num_classes=NUM_CLASSES,
        class_dropout_prob=float(config.get("cfg_dropout", 0.1)),
    )
    state_key = "ema" if args.weights == "ema" else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval().requires_grad_(False)

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        local_files_only=True,
    )
    vae.to(device).eval().requires_grad_(False)

    labels_list = list(args.labels)
    labels = torch.tensor(labels_list, device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    initial_noise = torch.randn(
        len(labels_list),
        *LATENT_SHAPE,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    autocast_dtype = None if args.precision == "fp32" else torch.bfloat16
    decoded_rows: list[torch.Tensor] = []
    sampling_rows: list[dict] = []

    for cfg_scale in args.cfg_scales:
        velocity, counter = official_cfg_velocity(
            model,
            labels,
            cfg_scale,
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
        filename = f"cfg_{cfg_scale:g}.png"
        save_image(
            decoded,
            output_dir / filename,
            nrow=len(labels_list),
            normalize=True,
            value_range=(-1, 1),
            padding=0,
        )
        row = {
            "cfg_scale": float(cfg_scale),
            "nfe": int(counter["nfe"]),
            "elapsed_seconds": elapsed,
            "image": filename,
            "latent_mean": float(latents.float().mean().item()),
            "latent_std": float(latents.float().std().item()),
        }
        sampling_rows.append(row)
        print(json.dumps({"event": "sampled", **row}), flush=True)

    class_records = load_class_records(Path(args.class_manifest), labels_list)
    sheet = make_comparison_sheet(
        [tensor_to_pil(images) for images in decoded_rows],
        list(args.cfg_scales),
        [record["name"] for record in class_records],
    )
    comparison_path = output_dir / "cfg_comparison.png"
    sheet.save(comparison_path, format="PNG", optimize=True)

    manifest = {
        "format": "eqvae_imagenet100_sit_sample_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": args.weights,
        "model_name": model_name,
        "official_sit": source_metadata,
        "class_manifest": str(Path(args.class_manifest).resolve()),
        "classes": class_records,
        "seed": int(args.seed),
        "same_initial_noise_across_cfg": True,
        "sampler": {
            "path": "linear",
            "prediction": "velocity",
            "method": "dopri5",
            "interval": [0.0, 1.0],
            "num_output_points": int(args.num_output_points),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "cfg_channels": "first_three_official_sit",
            "precision": args.precision,
            "allow_tf32": bool(args.allow_tf32),
        },
        "vae": {
            "model": "stabilityai/sd-vae-ft-mse",
            "local_files_only": True,
            "scaling_factor": SD_VAE_SCALING_FACTOR,
        },
        "samples": sampling_rows,
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
    parser.add_argument("--labels", type=parse_int_list, default=list(DEFAULT_LABELS))
    parser.add_argument("--cfg-scales", type=parse_float_list, default=[1.0, 1.5, 2.0, 4.0])
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--num-output-points", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-sit-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
