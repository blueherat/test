"""Audit the RAE-adapted LPL gradient bridge on official DINOv2 S/B/L RAEs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
RAE_ROOT = ROOT / "external/RAE"
RAE_SRC = RAE_ROOT / "src"
MODEL_ROOT = Path.home() / "data/eqvae/models/RAE"
for path in (ROOT, RAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rae_strict_lpl import (  # noqa: E402
    decoder_feature_pyramid,
    decoder_hidden_indices,
    strict_lpl_per_sample,
)
from experiments.train_rae_strict_lpl import (  # noqa: E402
    assert_frozen_modules_have_no_grad,
    module_state_versions,
)
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import ParquetImageNetDataset, center_crop_arr  # noqa: E402


SPECS = {
    "small": {
        "encoder": "facebook/dinov2-with-registers-small",
        "channels": 384,
        "decoder": MODEL_ROOT
        / "decoders/dinov2/wReg_small/ViTXL_n08/model.pt",
        "statistics": MODEL_ROOT
        / "stats/dinov2/wReg_small/imagenet1k/stat.pt",
    },
    "base": {
        "encoder": "facebook/dinov2-with-registers-base",
        "channels": 768,
        "decoder": MODEL_ROOT
        / "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        "statistics": MODEL_ROOT
        / "stats/dinov2/wReg_base/imagenet1k/stat.pt",
    },
    "large": {
        "encoder": "facebook/dinov2-with-registers-large",
        "channels": 1024,
        "decoder": MODEL_ROOT
        / "decoders/dinov2/wReg_large/ViTXL_n08/model.pt",
        "statistics": MODEL_ROOT
        / "stats/dinov2/wReg_large/imagenet1k/stat.pt",
    },
}


def write_payload(output: Path, payload: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def stage1_config(spec: dict[str, object]) -> OmegaConf:
    return OmegaConf.create(
        {
            "target": "stage1.RAE",
            "params": {
                "encoder_cls": "Dinov2withNorm",
                "encoder_config_path": str(spec["encoder"]),
                "encoder_input_size": 224,
                "encoder_params": {
                    "dinov2_path": str(spec["encoder"]),
                    "normalize": True,
                },
                "decoder_config_path": str(RAE_ROOT / "configs/decoder/ViTXL"),
                "pretrained_decoder_path": str(spec["decoder"]),
                "noise_tau": 0.0,
                "reshape_to_2d": True,
                "normalization_stat_path": str(spec["statistics"]),
            },
        }
    )


def audit_size(
    name: str,
    spec: dict[str, object],
    image: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    for key in ("decoder", "statistics"):
        if not Path(spec[key]).exists():
            raise FileNotFoundError(spec[key])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    rae = instantiate_from_config(stage1_config(spec)).to(
        device=device, dtype=torch.float32
    )
    decoder_state = torch.load(
        Path(spec["decoder"]), map_location="cpu", weights_only=True
    )
    if not isinstance(decoder_state, dict):
        raise TypeError(f"{name}: decoder checkpoint is not a state dict")
    rae.decoder.load_state_dict(decoder_state, strict=True)
    rae.requires_grad_(False).eval()
    versions = module_state_versions(rae)
    x = image.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        clean = rae.encode(x)
        reconstruction = rae.decode(clean)
        indices = decoder_hidden_indices(len(rae.decoder.decoder_layers))
        target_features = decoder_feature_pyramid(
            rae, clean, layer_indices=indices
        )
    expected_shape = (1, int(spec["channels"]), 16, 16)
    if tuple(clean.shape) != expected_shape:
        raise RuntimeError(f"{name}: {tuple(clean.shape)} != {expected_shape}")

    predicted = (
        clean.detach() + 0.01 * torch.randn_like(clean)
    ).requires_grad_(True)
    predicted_features = decoder_feature_pyramid(
        rae, predicted, layer_indices=indices
    )
    losses, details = strict_lpl_per_sample(
        target_features,
        predicted_features,
        layer_weights=[1.0] * len(indices),
    )
    latent_gradient = torch.autograd.grad(losses.sum(), predicted)[0]
    assert_frozen_modules_have_no_grad((rae,))
    if module_state_versions(rae) != versions:
        raise RuntimeError(f"{name}: frozen RAE parameters or buffers changed")
    if not torch.isfinite(losses).all() or not torch.isfinite(latent_gradient).all():
        raise RuntimeError(f"{name}: LPL or latent gradient is non-finite")
    if float(latent_gradient.square().mean()) == 0:
        raise RuntimeError(f"{name}: frozen decoder did not pass a latent gradient")

    result = {
        "size": name,
        "encoder": str(spec["encoder"]),
        "decoder": str(spec["decoder"]),
        "decoder_strict_load": True,
        "decoder_state_key_count": len(decoder_state),
        "statistics": str(spec["statistics"]),
        "latent_shape": list(clean.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "decoder_hidden_indices": list(indices),
        "decoder_feature_shapes": [list(value.shape) for value in target_features],
        "lpl": float(losses.item()),
        "latent_gradient_rms": float(latent_gradient.square().mean().sqrt()),
        "mask_keep_fraction": [
            float(value) for value in details["mask_keep_fraction"][0]
        ],
        "rae_gradients_none": True,
        "rae_parameters_unchanged": True,
        "finite": True,
    }
    del (
        rae,
        decoder_state,
        clean,
        reconstruction,
        target_features,
        predicted_features,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("/data/shared/imagenet-1k"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / "data/eqvae/experiments/rae_lpl_authenticity/stage1_sizes.json",
    )
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=tuple(SPECS),
        default=list(SPECS),
        help="Official DINOv2 RAE sizes to audit, in execution order.",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = ParquetImageNetDataset(
        args.data_path.expanduser().resolve(), split="validation"
    )
    pil_image, label = dataset[0]
    image = TF.to_tensor(center_crop_arr(pil_image, 256)).unsqueeze(0)
    output = args.output.expanduser().resolve()
    payload = {
        "data_split": "validation",
        "dataset_index": 0,
        "label": int(label),
        "seed": int(args.seed),
        "dtype": "float32",
        "tf32": False,
        "requested_sizes": list(args.sizes),
        "completed_sizes": [],
        "complete": False,
        "results": [],
    }
    write_payload(output, payload)
    for name in args.sizes:
        result = audit_size(
            name,
            SPECS[name],
            image,
            device=torch.device(args.device),
            seed=args.seed,
        )
        payload["results"].append(result)
        payload["completed_sizes"].append(name)
        write_payload(output, payload)
    payload["complete"] = True
    write_payload(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
