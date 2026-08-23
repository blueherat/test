"""Differentiable feature extractors specified by the FD-Loss papers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_fidelity.feature_extractor_inceptionv3 import (
    FeatureExtractorInceptionV3,
)
from torch_fidelity.interpolate_compat_tensorflow import (
    interpolate_bilinear_2d_like_tensorflow1x,
)


@dataclass(frozen=True)
class TimmFeatureSpec:
    name: str
    model_name: str
    output_dim: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    model_kwargs: tuple[tuple[str, object], ...] = ()


MAE_LARGE_224 = TimmFeatureSpec(
    name="mae",
    model_name="vit_large_patch16_224.mae",
    output_dim=1024,
    input_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

SIGLIP2_SO400M_224 = TimmFeatureSpec(
    name="siglip2",
    model_name="vit_so400m_patch16_siglip_256.v2_webli",
    output_dim=1152,
    input_size=224,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    model_kwargs=(("img_size", 224),),
)


class DifferentiableTimmFeature(nn.Module):
    """A frozen timm representation that keeps image-input gradients.

    The feature is the model's pre-logits representation. For MAE this is its
    normalized CLS token. The paper calls the specified SigLIP2 feature a CLS
    token, but that exact timm architecture has no prefix token and uses MAP
    attention pooling. Using ``forward_head(..., pre_logits=True)`` is therefore
    the only architecture-defined 1152-dimensional pooled representation; this
    clean-room choice is recorded as a paper ambiguity.
    """

    def __init__(
        self,
        spec: TimmFeatureSpec,
        *,
        pretrained: bool = True,
        trainable: bool = False,
        gradient_checkpointing: bool = False,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if model is None:
            try:
                import timm
            except ImportError as error:
                raise ImportError(
                    "DifferentiableTimmFeature requires timm; formal AdvFD runs "
                    "use the isolated timm 1.0.28 environment"
                ) from error
            model = timm.create_model(
                spec.model_name,
                pretrained=pretrained,
                num_classes=0,
                **dict(spec.model_kwargs),
            )

        self.spec = spec
        self.model = model
        self.model.eval()
        self.model.requires_grad_(trainable)
        if gradient_checkpointing:
            setter = getattr(self.model, "set_grad_checkpointing", None)
            if setter is None:
                raise ValueError(f"{spec.model_name} does not support checkpointing")
            setter(True)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(spec.mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(spec.std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "DifferentiableTimmFeature":
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected float images [B,3,H,W], got {tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError("Differentiable timm features expect floating point images")

        values = F.interpolate(
            images,
            size=(self.spec.input_size, self.spec.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        values = (values - self.pixel_mean) / self.pixel_std
        hidden = self.model.forward_features(values)
        features = self.model.forward_head(hidden, pre_logits=True)
        if features.ndim != 2 or features.shape[1] != self.spec.output_dim:
            raise RuntimeError(
                f"{self.spec.model_name} returned {tuple(features.shape)}, expected "
                f"[batch,{self.spec.output_dim}]"
            )
        return features.to(torch.float32)


def build_sim_feature_extractors(
    *, pretrained: bool = True, gradient_checkpointing: bool = False
) -> nn.ModuleDict:
    """Build the three unit-weight static representations used by FD-SIM."""

    return nn.ModuleDict(
        {
            "inception": DifferentiableInception2048(),
            "mae": DifferentiableTimmFeature(
                MAE_LARGE_224,
                pretrained=pretrained,
                gradient_checkpointing=gradient_checkpointing,
            ),
            "siglip2": DifferentiableTimmFeature(
                SIGLIP2_SO400M_224,
                pretrained=pretrained,
                gradient_checkpointing=gradient_checkpointing,
            ),
        }
    )


class DifferentiableInception2048(nn.Module):
    """torch-fidelity Inception-2048 with a differentiable float image input.

    ``torch-fidelity`` intentionally requires uint8 input for metric
    reproducibility. Post-training needs gradients with respect to continuous
    generated pixels, so this wrapper keeps the same resize, normalization and
    network weights while accepting images in [0, 1] without quantization.
    """

    def __init__(self, *, trainable: bool = False) -> None:
        super().__init__()
        self.extractor = FeatureExtractorInceptionV3(
            "inception-v3-compat", ["2048"], verbose=False
        )
        self.extractor.eval()
        self.extractor.requires_grad_(trainable)

    def train(self, mode: bool = True) -> "DifferentiableInception2048":
        # The representation weights may be optimized, but inference-mode layer
        # behavior remains fixed, matching a pretrained visual representation.
        super().train(False)
        self.extractor.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected float images [B,3,H,W], got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError("Differentiable Inception expects floating point images")

        model = self.extractor
        x = images.to(model.feature_extractor_internal_dtype) * 255.0
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(model.INPUT_IMAGE_SIZE, model.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        x = (x - 128.0) / 128.0

        x = model.Conv2d_1a_3x3(x)
        x = model.Conv2d_2a_3x3(x)
        x = model.Conv2d_2b_3x3(x)
        x = model.MaxPool_1(x)
        x = model.Conv2d_3b_1x1(x)
        x = model.Conv2d_4a_3x3(x)
        x = model.MaxPool_2(x)
        x = model.Mixed_5b(x)
        x = model.Mixed_5c(x)
        x = model.Mixed_5d(x)
        x = model.Mixed_6a(x)
        x = model.Mixed_6b(x)
        x = model.Mixed_6c(x)
        x = model.Mixed_6d(x)
        x = model.Mixed_6e(x)
        x = model.Mixed_7a(x)
        x = model.Mixed_7b(x)
        x = model.Mixed_7c(x)
        x = model.AvgPool(x)
        return torch.flatten(x, 1).to(torch.float32)


def generator_output_to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    """Convert the pMF/JiT [-1, 1] convention to encoder input range."""

    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)
