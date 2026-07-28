"""Measure latent-condition responsibility in an official PiD checkpoint.

PiD's released DINOv2 and SigLIP-2 models are four-step distilled students.
Accordingly, this script evaluates only their four supported student times. It
does not interpret the resulting table as a continuous diffusion-time curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.noise_responsibility_profile import (  # noqa: E402
    ResponsibilityBatch,
    aggregate_profile,
    derangement,
    identity_control_error,
    responsibility_rows,
)


BACKBONES = {
    "dinov2": {
        "experiment": "PiD_res2k_sr4x_official_dinov2_distill_4step",
        "checkpoint": "checkpoints/PiD_res2k_sr4x_official_dinov2_distill_4step/model_ema_bf16.pth",
        "lq_size": 512,
        "scale": 4,
    },
    "siglip": {
        "experiment": "PiD_res2k_sr8x_official_siglip_distill_4step",
        "checkpoint": "checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth",
        "lq_size": 256,
        "scale": 8,
    },
    "sd3": {
        "experiment": "PiD_res2k_sr4x_official_sd3_distill_4step",
        "checkpoint": "checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        "lq_size": 512,
        "scale": 4,
    },
}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    original_width: int
    original_height: int
    target_size: int
    upsampled: bool


def flow_noisy_state(x0: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """PiD flow path: x_t = (1 - t) x_0 + t epsilon."""

    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have identical shapes")
    if timestep.ndim != 1 or timestep.shape[0] != x0.shape[0]:
        raise ValueError("timestep must have shape [B]")
    shaped_t = timestep.view(x0.shape[0], *([1] * (x0.ndim - 1)))
    return (1.0 - shaped_t) * x0 + shaped_t * noise


def output_to_x0(
    x_t: torch.Tensor,
    net_output: torch.Tensor,
    timestep: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Convert PiD's velocity or x0 network output to a clean prediction."""

    if prediction_type == "x0":
        return net_output.to(x_t.dtype)
    if prediction_type != "velocity":
        raise ValueError(f"unsupported prediction_type: {prediction_type}")
    shaped_t = timestep.double().view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
    return (x_t.double() - shaped_t * net_output.double()).to(x_t.dtype)


def advance_student_state(
    x_t: torch.Tensor,
    x0_prediction: torch.Tensor,
    t_current: torch.Tensor,
    t_next: torch.Tensor,
    *,
    sample_type: str,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the official distilled-student SDE or ODE transition."""

    shape = (x_t.shape[0], *([1] * (x_t.ndim - 1)))
    current = t_current.view(shape)
    following = t_next.view(shape)
    if sample_type == "sde":
        if noise is None or noise.shape != x_t.shape:
            raise ValueError("SDE transition requires noise with the same shape as x_t")
        return (1.0 - following) * x0_prediction + following * noise
    if sample_type == "ode":
        velocity = (x_t - x0_prediction) / current.clamp_min(5e-2)
        return x_t + (following - current) * velocity
    raise ValueError(f"unsupported student sample type: {sample_type}")


def discover_images(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def load_square_target(path: Path, target_size: int) -> tuple[torch.Tensor, ImageRecord]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        if side != target_size:
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1.0)
    record = ImageRecord(
        path=str(path),
        original_width=width,
        original_height=height,
        target_size=target_size,
        upsampled=side < target_size,
    )
    return tensor, record


def paired_predictions(
    *,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    latents: Mapping[str, torch.Tensor],
    predict: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    prediction_type: str,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Evaluate all branches on exactly the same noisy state and timestep."""

    predictions: dict[str, torch.Tensor] = {}
    for name in ("real", "null", "shuffle"):
        output = predict(x_t, timestep, latents[name])
        predictions[name] = output_to_x0(x_t, output, timestep, prediction_type)
    repeated = output_to_x0(
        x_t,
        predict(x_t, timestep, latents["real"]),
        timestep,
        prediction_type,
    )
    identity = identity_control_error(predictions["real"], repeated)
    return predictions, identity


def batch_order_control_table(
    rows: pd.DataFrame,
    reordered_rows: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Compare reordered teacher-forced rows without mixing rollout modes."""

    metric_columns = [
        "loss_real",
        "loss_null",
        "loss_shuffle",
        "delta_null",
        "delta_shuffle",
    ]
    reference = rows[(rows.seed == seed) & (rows["mode"] == "teacher_forced")][
        ["sample_index", "timestep", *metric_columns]
    ]
    comparison = reference.merge(
        reordered_rows[["sample_index", "timestep", *metric_columns]],
        on=["sample_index", "timestep"],
        suffixes=("_reference", "_reordered"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "max_absolute_difference": float(
                    (
                        comparison[f"{metric}_reference"]
                        - comparison[f"{metric}_reordered"]
                    )
                    .abs()
                    .max()
                ),
            }
            for metric in metric_columns
        ]
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pid_model(args: argparse.Namespace):
    pid_root = Path(args.pid_root).resolve()
    if str(pid_root) not in sys.path:
        sys.path.insert(0, str(pid_root))
    os.chdir(pid_root)

    from pid._src.utils.model_loader import load_model_from_checkpoint

    spec = BACKBONES[args.backbone]
    checkpoint = pid_root / spec["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PiD checkpoint not found: {checkpoint}")

    experiment_opts = list(args.experiment_opt)
    if args.backbone in {"dinov2", "siglip"} and args.encoder_only_tokenizer:
        experiment_opts.append("model.config.tokenizer.pretrained_decoder_path=null")

    model, config = load_model_from_checkpoint(
        experiment_name=spec["experiment"],
        checkpoint_path=str(checkpoint),
        config_file="pid/_src/configs/pid/config.py",
        enable_fsdp=False,
        experiment_opts=experiment_opts,
        strict=False,
        load_ema_to_reg=False,
        seed=args.seed,
    )
    model.eval()
    return model, config, checkpoint


def _encode_conditions(model, targets_cpu: torch.Tensor, lq_size: int) -> torch.Tensor:
    lq = F.interpolate(
        targets_cpu,
        size=(lq_size, lq_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(-1.0, 1.0)
    encoded = []
    for image in lq:
        latent = model.encode_lq_latent(image[None].to(device="cuda", dtype=torch.bfloat16))
        encoded.append(latent.float().cpu())
    return torch.cat(encoded, dim=0)


def _make_predictor(model, captions: list[str], degrade_sigma: float):
    caption_embs, _ = model._encode_text_raw(captions)
    caption_embs = caption_embs.to(**model.tensor_kwargs)
    sigma = torch.full((len(captions),), degrade_sigma, device="cuda", dtype=torch.float32)
    timescale = float(model.fm_trainer.timescale)
    net = model.net
    net.eval()

    @torch.no_grad()
    def predict(x_t: torch.Tensor, timestep: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if x_t.shape[0] != 1 or latent.shape[0] != 1:
            raise ValueError("the memory-conservative PiD probe expects batch size one")
        with torch.autocast("cuda", dtype=model.autocast_dtype) if model.autocast_dtype else torch.no_grad():
            return net(
                x_t.to(**model.tensor_kwargs),
                timestep.to(device="cuda", dtype=torch.float32) * timescale,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=latent.to(**model.tensor_kwargs),
                degrade_sigma=sigma,
            )

    return predict


def release_text_encoder(model) -> None:
    """Free the frozen LM after the single fixed caption has been encoded."""

    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        return
    text_encoder.to("cpu")
    object.__delattr__(model, "text_encoder")
    torch.cuda.empty_cache()


def release_image_tokenizer(model) -> None:
    """Free the frozen image encoder after all screening latents are cached."""

    tokenizer = getattr(model, "vae_encoder", None)
    wrapper = getattr(tokenizer, "model", None)
    module = getattr(wrapper, "model", wrapper)
    if hasattr(module, "to"):
        module.to("cpu")
    model.vae_encoder = None
    torch.cuda.empty_cache()


def run_teacher_forced_probe(
    *,
    model,
    targets_cpu: torch.Tensor,
    latents_cpu: torch.Tensor,
    seeds: list[int],
    caption: str,
    batch_order_control: bool = True,
    real_rollout_control: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate supported student times on paired forward-noised real images."""

    permutation = derangement(len(targets_cpu), seed=seeds[0]).cpu()
    t_values = model._get_t_list(torch.device("cuda"))[:-1].float().cpu()
    prediction_type = str(model.config.prediction_type)
    all_rows = []
    control_rows = []
    predict = _make_predictor(model, [caption], degrade_sigma=0.0)
    release_text_encoder(model)

    for seed in seeds:
        for sample_index in range(len(targets_cpu)):
            target = targets_cpu[sample_index : sample_index + 1].to(device="cuda", dtype=torch.float32)
            generator = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + sample_index)
            noise = torch.randn(target.shape, device="cuda", dtype=torch.float32, generator=generator)
            real = latents_cpu[sample_index : sample_index + 1]
            shuffled = latents_cpu[permutation[sample_index] : permutation[sample_index] + 1]
            branches = {"real": real, "null": torch.zeros_like(real), "shuffle": shuffled}

            for timestep_value in t_values:
                timestep = timestep_value.reshape(1).to(device="cuda", dtype=torch.float32)
                x_t = flow_noisy_state(target, noise, timestep)
                predictions, identity = paired_predictions(
                    x_t=x_t,
                    timestep=timestep,
                    latents=branches,
                    predict=predict,
                    prediction_type=prediction_type,
                )
                rows = responsibility_rows(
                    ResponsibilityBatch(
                        timestep=timestep,
                        target=target,
                        predictions=predictions,
                        sample_index=torch.tensor([sample_index], device="cuda"),
                    )
                )
                rows.insert(0, "seed", seed)
                rows.insert(1, "mode", "teacher_forced")
                rows["shuffle_index"] = int(permutation[sample_index])
                all_rows.append(rows)
                control_rows.append(
                    {
                        "mode": "teacher_forced",
                        "seed": seed,
                        "sample_index": sample_index,
                        "timestep": float(timestep_value),
                        **identity,
                    }
                )
    if real_rollout_control:
        full_t_values = model._get_t_list(torch.device("cuda")).float().cpu()
        sample_type = str(model.config.student_sample_type)
        for seed in seeds:
            for sample_index in range(len(targets_cpu)):
                target = targets_cpu[sample_index : sample_index + 1].to(device="cuda", dtype=torch.float32)
                generator = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + sample_index)
                x_t = torch.randn(target.shape, device="cuda", dtype=torch.float32, generator=generator)
                real = latents_cpu[sample_index : sample_index + 1]
                shuffled = latents_cpu[permutation[sample_index] : permutation[sample_index] + 1]
                branches = {"real": real, "null": torch.zeros_like(real), "shuffle": shuffled}
                for timestep_value, next_value in zip(full_t_values[:-1], full_t_values[1:]):
                    timestep = timestep_value.reshape(1).to(device="cuda", dtype=torch.float32)
                    predictions, identity = paired_predictions(
                        x_t=x_t,
                        timestep=timestep,
                        latents=branches,
                        predict=predict,
                        prediction_type=prediction_type,
                    )
                    rows = responsibility_rows(
                        ResponsibilityBatch(
                            timestep=timestep,
                            target=target,
                            predictions=predictions,
                            sample_index=torch.tensor([sample_index], device="cuda"),
                        )
                    )
                    rows.insert(0, "seed", seed)
                    rows.insert(1, "mode", "real_rollout")
                    rows["shuffle_index"] = int(permutation[sample_index])
                    all_rows.append(rows)
                    control_rows.append(
                        {
                            "mode": "real_rollout",
                            "seed": seed,
                            "sample_index": sample_index,
                            "timestep": float(timestep_value),
                            **identity,
                        }
                    )
                    if float(next_value) > 0.0:
                        transition_noise = torch.randn(
                            x_t.shape,
                            device="cuda",
                            dtype=torch.float32,
                            generator=generator,
                        )
                        x_t = advance_student_state(
                            x_t,
                            predictions["real"],
                            timestep,
                            next_value.reshape(1).to(device="cuda", dtype=torch.float32),
                            sample_type=sample_type,
                            noise=transition_noise,
                        )

    rows_frame = pd.concat(all_rows, ignore_index=True)
    order_control_rows = []
    if batch_order_control:
        seed = seeds[0]
        reordered_rows = []
        for sample_index in reversed(range(len(targets_cpu))):
            target = targets_cpu[sample_index : sample_index + 1].to(device="cuda", dtype=torch.float32)
            generator = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + sample_index)
            noise = torch.randn(target.shape, device="cuda", dtype=torch.float32, generator=generator)
            real = latents_cpu[sample_index : sample_index + 1]
            shuffled = latents_cpu[permutation[sample_index] : permutation[sample_index] + 1]
            branches = {"real": real, "null": torch.zeros_like(real), "shuffle": shuffled}
            for timestep_value in t_values:
                timestep = timestep_value.reshape(1).to(device="cuda", dtype=torch.float32)
                predictions, _ = paired_predictions(
                    x_t=flow_noisy_state(target, noise, timestep),
                    timestep=timestep,
                    latents=branches,
                    predict=predict,
                    prediction_type=prediction_type,
                )
                reordered = responsibility_rows(
                    ResponsibilityBatch(
                        timestep=timestep,
                        target=target,
                        predictions=predictions,
                        sample_index=torch.tensor([sample_index], device="cuda"),
                    )
                )
                reordered.insert(0, "seed", seed)
                reordered_rows.append(reordered)

        reordered_frame = pd.concat(reordered_rows, ignore_index=True)
        order_control_rows = batch_order_control_table(
            rows_frame, reordered_frame, seed=seed
        ).to_dict("records")
    return rows_frame, pd.DataFrame(control_rows), pd.DataFrame(order_control_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-root", default=str(Path.home() / "data/eqvae/external/PiD"))
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="dinov2")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--caption", default="A high quality natural photograph.")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "data/eqvae/pid_responsibility")
    parser.add_argument("--require-native-resolution", action="store_true")
    parser.add_argument("--encoder-only-tokenizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experiment-opt", action="append", default=[])
    parser.add_argument("--batch-order-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-rollout-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0, help="model construction seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images < 2:
        raise ValueError("shuffle control requires --num-images >= 2")
    torch.set_grad_enabled(False)

    model, _, checkpoint = _load_pid_model(args)
    spec = BACKBONES[args.backbone]
    target_size = int(spec["lq_size"] * spec["scale"])

    targets = []
    records = []
    for path in discover_images(args.image_root):
        target, record = load_square_target(path, target_size)
        if args.require_native_resolution and record.upsampled:
            continue
        targets.append(target)
        records.append(record)
        if len(targets) == args.num_images:
            break
    if len(targets) < args.num_images:
        raise ValueError(
            f"found only {len(targets)} usable images under {args.image_root}; "
            f"need {args.num_images}"
        )
    targets_cpu = torch.stack(targets)
    latents_cpu = _encode_conditions(model, targets_cpu, int(spec["lq_size"]))
    release_image_tokenizer(model)

    rows, controls, order_controls = run_teacher_forced_probe(
        model=model,
        targets_cpu=targets_cpu,
        latents_cpu=latents_cpu,
        seeds=list(args.seeds),
        caption=args.caption,
        batch_order_control=args.batch_order_control,
        real_rollout_control=args.real_rollout_control,
    )
    profile = pd.concat(
        [
            aggregate_profile(frame).assign(mode=mode)
            for mode, frame in rows.groupby("mode", sort=True)
        ],
        ignore_index=True,
    )
    profile.insert(0, "mode", profile.pop("mode"))
    output_dir = args.output_root / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "paired_rows.csv", index=False)
    profile.to_csv(output_dir / "profile.csv", index=False)
    controls.to_csv(output_dir / "identity_controls.csv", index=False)
    order_controls.to_csv(output_dir / "batch_order_controls.csv", index=False)
    provenance = {
        "args": vars(args) | {"image_root": str(args.image_root), "output_root": str(args.output_root)},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "supported_student_times": [float(value) for value in model._get_t_list(torch.device("cuda")).cpu()],
        "prediction_type": str(model.config.prediction_type),
        "images": [asdict(record) for record in records],
        "latent_shape": list(latents_cpu.shape),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    with (output_dir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, default=str)
    print(profile.to_string(index=False))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
