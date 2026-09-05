"""Attention routing negatives for the frozen RAEv2 DDT head.

Identity is standard PAG; uniform is an existing perturbation baseline.
``preserve_self`` keeps each attention row's self mass and replaces its
remaining probabilities by their mean. It is the unique maximum-entropy
row with that self mass, not a theorem about generation quality or novelty.
``native_explicit`` is an audit-only numerical control: it retains the
attention probabilities while using the same FP32 path as the negatives.

The native full/base branches share their encoder prefix with the negative.
Only the negative decoder is recomputed, with one attention operation edited.
No module, forward method, global backend setting, or hook is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


MODES = ("identity", "uniform", "preserve_self")
AUDIT_MODES = ("native_explicit",)


@dataclass(frozen=True)
class RoutingReference:
    full: Tensor
    base: Tensor
    negative: Tensor
    telemetry: dict[str, Tensor]


def _validate_mode(mode: str) -> None:
    if mode not in MODES + AUDIT_MODES:
        raise ValueError(f"unknown routing reference mode: {mode}")


def routing_probability_reference(attention: Tensor, mode: str) -> Tensor:
    """Transform square row probabilities without changing the input.

    Leading dimensions are arbitrary. Inputs must already be nonnegative,
    normalized probabilities. Summing the actual off-diagonal probabilities
    avoids cancellation when the self mass rounds to one. The one-token
    case is the identity for all modes. Input precision is retained.
    """
    _validate_mode(mode)
    if attention.ndim < 2 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError("attention must contain square probability matrices")
    if not attention.is_floating_point() or attention.shape[-1] < 1:
        raise ValueError("attention must be floating point with at least one token")
    if mode == "native_explicit":
        return attention
    count = attention.shape[-1]
    diagonal_mask = torch.eye(count, device=attention.device, dtype=torch.bool)
    if mode == "identity" or count == 1:
        return diagonal_mask.to(attention.dtype).expand_as(attention)
    if mode == "uniform":
        return torch.full_like(attention, 1.0 / count)
    diagonal = attention.diagonal(dim1=-2, dim2=-1)
    off_mass = attention.masked_fill(diagonal_mask, 0).sum(dim=-1)
    uniform_off = (off_mass / (count - 1)).unsqueeze(-1).expand_as(attention)
    return torch.where(diagonal_mask, diagonal.unsqueeze(-1), uniform_off)


def routing_kl(first: Tensor, second: Tensor) -> Tensor:
    """Row-wise KL(first || second), including exact support infinities.

    Zero probability contributes zero; positive mass against a zero target
    contributes +inf. In particular native-to-identity KL is normally inf.
    No epsilon-smoothed surrogate is substituted for this diagnostic.
    """
    if first.shape != second.shape or first.ndim < 1:
        raise ValueError("KL probability arrays must have identical nonempty shapes")
    positive = first > 0
    safe_first = torch.where(positive, first, torch.ones_like(first))
    safe_second = torch.where(positive, second, torch.ones_like(second))
    terms = first * (safe_first.log() - safe_second.log())
    return terms.sum(dim=-1)


def _routing_attention(attention_module, x: Tensor, rope, mode: str):
    """NormAttention projections/norms/RoPE, then explicit FP32 routing.

    Only decoder image tokens are supported here; native decoder attention
    has no mask. Under autocast, RMSNorm/RoPE can promote q/k to FP32, but
    native SDPA casts them back to the autocast value dtype. Reproduce that
    input quantization before FP32 logits. Projected values retain their
    native dtype before the final projection. Autocast is restored on exit.
    """
    batch, tokens, _ = x.shape
    heads, head_dim = attention_module.num_heads, attention_module.head_dim

    def shape(projected):
        return projected.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3)

    q = shape(attention_module.q(x))
    k = shape(attention_module.k(x))
    v = shape(attention_module.v(x))
    q = attention_module.q_norm(q)
    k = attention_module.k_norm(k)
    q, k = rope(q), rope(k)
    if torch.is_autocast_enabled(x.device.type):
        # The native SDPA autocast policy lowers all floating inputs. Here v
        # already has that dtype from its Linear, whereas q/k may have been
        # promoted by FP32 RMSNorm weights and RoPE buffers. Do not evaluate
        # a higher-precision routing matrix than the native branch received.
        q, k = q.to(v.dtype), k.to(v.dtype)
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
        probabilities = logits.softmax(dim=-1)
        negative_probabilities = routing_probability_reference(probabilities, mode)
        self_mass = probabilities.diagonal(dim1=-2, dim2=-1)
        values = v.float()
        if mode == "native_explicit":
            modified = probabilities @ values
        elif mode == "identity" or tokens == 1:
            modified = values
        elif mode == "uniform":
            modified = values.mean(dim=-2, keepdim=True).expand_as(values)
        else:
            diagonal_mask = torch.eye(tokens, device=x.device, dtype=torch.bool)
            off_mass = probabilities.masked_fill(diagonal_mask, 0).sum(dim=-1)
            off_mean = (values.sum(dim=-2, keepdim=True) - values) / (tokens - 1)
            modified = self_mass.unsqueeze(-1) * values + off_mass.unsqueeze(-1) * off_mean
        information_reference = routing_probability_reference(probabilities, "preserve_self")
        telemetry = {
            "self_mass": self_mass,
            "negative_self_mass": negative_probabilities.diagonal(dim1=-2, dim2=-1),
            "routing_kl": routing_kl(probabilities, negative_probabilities),
            "routing_reverse_kl": routing_kl(negative_probabilities, probabilities),
            "routing_information": routing_kl(probabilities, information_reference),
        }
        telemetry = {key: value.detach() for key, value in telemetry.items()}
    modified = modified.to(v.dtype).permute(0, 2, 1, 3).reshape(batch, tokens, attention_module.dim)
    return attention_module.proj(modified), telemetry


def _edited_decoder_block(block, x: Tensor, condition: Tensor, rope, mode: str):
    """Faithful DDTDecoderBlock arithmetic with only attention replaced."""
    modulation = block.adaln_modulation(condition)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    attention_input = block.norm1(x) * (1 + scale_msa) + shift_msa
    output, telemetry = _routing_attention(block.attn, attention_input, rope, mode)
    x = x + gate_msa * output
    x = x + gate_mlp * block.mlp(block.norm2(x) * (1 + scale_mlp) + shift_mlp)
    return x, telemetry


@torch.no_grad()
def evaluate_routing_reference(
    model,
    state: Tensor,
    times: Tensor,
    labels: Tensor,
    mode: str = "preserve_self",
    decoder_block: int = 1,
) -> RoutingReference:
    """Return native full/base plus one edited-decoder full prediction.

    ``decoder_block`` is zero-based within the DDT decoder (0 or 1 for the
    official K7 checkpoint), not the complete block list. The default edits
    the final decoder block. Full/base use native operations in native order;
    the full encoder prefix runs once. ``telemetry`` has [batch, head, token]
    entries from the edited layer, before its output projection.

    This inference helper expects the official label-conditioned dual-head
    model in evaluation mode. Real-checkpoint numerical parity and quality
    are deliberately left to the separate sampling/audit entry points.
    ``native_explicit`` measures the SDPA-to-explicit arithmetic residual;
    it is not a guidance candidate. A candidate routing gap must materially
    exceed that residual before any norm matching to the ordinary IG gap.
    """
    _validate_mode(mode)
    if model.training:
        raise ValueError("routing reference requires a model in evaluation mode")
    if state.ndim != 4 or times.shape != (state.shape[0],) or labels.shape != times.shape:
        raise ValueError("expected BCHW state and one time and label per sample")
    if not isinstance(decoder_block, int) or not 0 <= decoder_block < model.num_dec_blocks:
        raise ValueError("decoder_block must index an existing DDT decoder block")
    if not 1 <= model.base_model_depth <= model.num_enc_blocks:
        raise ValueError("base head depth must lie within the encoder prefix")
    if getattr(model, "use_cfg_conds", False):
        raise ValueError("CFG-scale condition tokens are outside this label-only helper")

    # Preserve the exact order of DiTwDDTHeadIG.forward(return_intermediate=False).
    conditions = {"context": labels, "attn_mask": None}
    seq, t_emb_base = model._build_sequence(state, times, conditions)
    attn_mask = model._build_attn_mask(seq, conditions)
    base_features = None
    for index in range(model.num_enc_blocks):
        seq = model.blocks[index](seq, model.enc_rope, attn_mask)
        if index + 1 == model.base_model_depth:
            base_features = seq[:, :model.s_embedder.num_patches, :]
    seq = model.s_projector(F.silu(t_emb_base + seq[:, :model.s_embedder.num_patches, :]))
    full = model.x_embedder(state)
    for index in range(model.num_dec_blocks):
        full = model.blocks[model.num_enc_blocks + index](full, seq, model.dec_rope)
    full = model.final_layer(full, seq)
    full = model.unpatchify(full, model.x_patch_size)
    base = F.silu(t_emb_base + base_features)
    base = model.base_final_layer(base, base)
    base = model.unpatchify(base, model.s_patch_size)

    # Recompute only the decoder; no hook or persistent model mutation.
    negative = model.x_embedder(state)
    telemetry = {}
    for index in range(model.num_dec_blocks):
        block = model.blocks[model.num_enc_blocks + index]
        if index == decoder_block:
            negative, telemetry = _edited_decoder_block(block, negative, seq, model.dec_rope, mode)
        else:
            negative = block(negative, seq, model.dec_rope)
    negative = model.final_layer(negative, seq)
    negative = model.unpatchify(negative, model.x_patch_size)
    return RoutingReference(full=full, base=base, negative=negative, telemetry=telemetry)
