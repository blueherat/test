"""Compare clean-room and public AdvFD timm features and input gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from experiments.advfd_cleanroom.feature_extractors import (
    MAE_LARGE_224,
    SIGLIP2_SO400M_224,
    DifferentiableTimmFeature,
    TimmFeatureSpec,
)


DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=260823)
    return parser.parse_args()


def tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = left.detach().to(torch.float64).cpu() - right.detach().to(torch.float64).cpu()
    return {
        "max_abs": float(delta.abs().max()),
        "rms": float(delta.square().mean().sqrt()),
    }


def state_dict_error(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, object]:
    left_keys = set(left)
    right_keys = set(right)
    common = sorted(left_keys & right_keys)
    max_abs = 0.0
    mismatched_shapes: list[str] = []
    for key in common:
        if left[key].shape != right[key].shape:
            mismatched_shapes.append(key)
            continue
        delta = (
            left[key].detach().to(torch.float64).cpu()
            - right[key].detach().to(torch.float64).cpu()
        )
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "left_tensors": len(left),
        "right_tensors": len(right),
        "common_tensors": len(common),
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "shape_mismatches": mismatched_shapes,
        "max_abs_common": max_abs,
    }


def audit_spec(
    spec: TimmFeatureSpec,
    official_cls,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    official = official_cls(
        spec.model_name,
        device=str(device),
        target_size=spec.input_size,
        grad_checkpointing=False,
    )
    clean = DifferentiableTimmFeature(spec, pretrained=True).to(device).eval()

    weights = state_dict_error(official.model.state_dict(), clean.model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    image = torch.rand((1, 3, 256, 256), generator=generator)
    projection = torch.randn(
        (1, spec.output_dim), generator=torch.Generator().manual_seed(seed + 2)
    )

    official_image = image.to(device).requires_grad_(True)
    official_feature = official(official_image)[0]
    official_scalar = (official_feature * projection.to(device)).sum()
    official_gradient = torch.autograd.grad(official_scalar, official_image)[0]

    clean_image = image.to(device).requires_grad_(True)
    clean_feature = clean(clean_image)
    clean_scalar = (clean_feature * projection.to(device)).sum()
    clean_gradient = torch.autograd.grad(clean_scalar, clean_image)[0]

    result = {
        "model_name": spec.model_name,
        "feature_shape": list(official_feature.shape),
        "official_num_prefix_tokens": int(official.num_prefix_tokens),
        "official_has_attn_pool": bool(official.has_attn_pool),
        "clean_global_pool": str(getattr(clean.model, "global_pool", "")),
        "weights": weights,
        "feature_error": tensor_error(official_feature, clean_feature),
        "input_gradient_error": tensor_error(official_gradient, clean_gradient),
        "official_feature_rms": float(
            official_feature.detach().to(torch.float64).square().mean().sqrt().cpu()
        ),
        "official_input_gradient_rms": float(
            official_gradient.detach().to(torch.float64).square().mean().sqrt().cpu()
        ),
    }
    del official, clean
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))
    from frechet_distance.repr_models import TimmReprModel  # noqa: PLC0415

    device = torch.device(args.device)
    results = {
        "protocol": "official_vs_cleanroom_timm_feature_and_input_gradient_v1",
        "device": str(device),
        "seed": args.seed,
        "models": {},
    }
    for spec in (MAE_LARGE_224, SIGLIP2_SO400M_224):
        results["models"][spec.name] = audit_spec(
            spec, TimmReprModel, device=device, seed=args.seed
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
