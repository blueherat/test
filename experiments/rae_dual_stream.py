"""Semantic-conditioned detail flow components for the RAE dual-stream gate."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = ROOT / "external/RAE/src"
if str(RAE_SRC) not in sys.path:
    sys.path.insert(0, str(RAE_SRC))

from stage2.models.DDT import DiTwDDTHead  # noqa: E402


def split_semantic_coefficients(
    final: torch.Tensor, basis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return endpoint-preserving semantic latent and rank-k detail coefficients."""

    if final.ndim != 4 or basis.ndim != 2 or basis.shape[0] != final.shape[1]:
        raise ValueError("expected final BCHW and basis [C,K]")
    basis = basis.to(device=final.device, dtype=final.dtype)
    mean = final.mean(dim=(-2, -1), keepdim=True)
    residual = final - mean
    coefficients = torch.einsum("bchw,ck->bkhw", residual, basis)
    detail = torch.einsum("bkhw,ck->bchw", coefficients, basis)
    semantic = final - detail
    return semantic, coefficients


def fuse_semantic_coefficients(
    semantic: torch.Tensor, coefficients: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    if semantic.ndim != 4 or coefficients.ndim != 4:
        raise ValueError("semantic and coefficients must be BCHW tensors")
    basis = basis.to(device=semantic.device, dtype=semantic.dtype)
    if basis.shape != (semantic.shape[1], coefficients.shape[1]):
        raise ValueError("basis shape disagrees with semantic/detail channels")
    return semantic + torch.einsum("bkhw,ck->bchw", coefficients, basis)


class SemanticConditionedDetailDDT(nn.Module):
    """A small DDT whose encoder sees semantic tokens and decoder sees detail."""

    def __init__(
        self,
        *,
        detail_channels: int = 16,
        semantic_channels: int = 768,
        input_size: int = 16,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.context_projector = nn.Conv2d(
            semantic_channels, detail_channels, kernel_size=1, bias=True
        )
        self.core = DiTwDDTHead(
            input_size=input_size,
            patch_size=1,
            in_channels=detail_channels,
            hidden_size=[256, 768],
            depth=[6, 2],
            num_heads=[4, 12],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=num_classes,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed=True,
        )
        nn.init.xavier_uniform_(self.context_projector.weight)
        nn.init.zeros_(self.context_projector.bias)

    def forward(
        self,
        detail: torch.Tensor,
        time: torch.Tensor,
        labels: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        core = self.core
        context = self.context_projector(semantic)
        time_embedding = core.t_embedder(time)
        label_embedding = core.y_embedder(labels, self.training)
        conditioning = nn.functional.silu(time_embedding + label_embedding)

        encoded = core.s_embedder(context)
        if core.use_pos_embed:
            encoded = encoded + core.pos_embed
        for index in range(core.num_encoder_blocks):
            encoded = core.blocks[index](
                encoded, conditioning, feat_rope=core.enc_feat_rope
            )
        repeated_time = time_embedding.unsqueeze(1).repeat(1, encoded.shape[1], 1)
        encoded = nn.functional.silu(repeated_time + encoded)
        encoded = core.s_projector(encoded)

        decoded = core.x_embedder(detail)
        if core.use_pos_embed and core.x_pos_embed is not None:
            decoded = decoded + core.x_pos_embed
        for index in range(core.num_encoder_blocks, core.num_blocks):
            decoded = core.blocks[index](
                decoded, encoded, feat_rope=core.dec_feat_rope
            )
        decoded = core.final_layer(decoded, encoded)
        return core.unpatchify(decoded)
