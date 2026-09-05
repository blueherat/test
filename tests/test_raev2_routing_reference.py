from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_routing_reference import (  # noqa: E402
    AUDIT_MODES,
    MODES,
    _routing_attention,
    evaluate_routing_reference,
    routing_kl,
    routing_probability_reference,
)


def probabilities():
    return torch.randn(2, 3, 5, 5, generator=torch.Generator().manual_seed(841), dtype=torch.float64).softmax(-1)


@pytest.mark.parametrize("mode", MODES + AUDIT_MODES)
def test_probability_operator_is_stochastic_and_idempotent(mode):
    original = probabilities()
    saved = original.clone()
    result = routing_probability_reference(original, mode)
    assert (result >= 0).all()
    torch.testing.assert_close(result.sum(-1), torch.ones_like(result.sum(-1)))
    torch.testing.assert_close(routing_probability_reference(result, mode), result)
    assert torch.equal(original, saved)


def test_native_explicit_is_the_unchanged_probability_operator():
    original = probabilities()
    result = routing_probability_reference(original, "native_explicit")
    assert result is original
    assert torch.equal(routing_kl(original, result), torch.zeros_like(original[..., 0]))


def test_preserve_self_retains_diagonal_and_removes_only_offself_routing():
    original = probabilities()
    result = routing_probability_reference(original, "preserve_self")
    assert torch.equal(result.diagonal(dim1=-2, dim2=-1), original.diagonal(dim1=-2, dim2=-1))
    for row in range(original.shape[-1]):
        others = torch.arange(original.shape[-1]) != row
        torch.testing.assert_close(
            result[..., row, others],
            original[..., row, others].mean(-1, keepdim=True).expand_as(result[..., row, others]),
        )


def test_maximum_entropy_and_conditional_kl_identity():
    original = probabilities()
    result = routing_probability_reference(original, "preserve_self")
    entropy_before = -(original * original.log()).sum(-1)
    entropy_after = -(result * result.log()).sum(-1)
    assert (entropy_after >= entropy_before).all()
    torch.testing.assert_close(routing_kl(original, result), entropy_after - entropy_before)
    for row in range(original.shape[-1]):
        others = torch.arange(original.shape[-1]) != row
        off = original[..., row, others]
        mass = off.sum(-1)
        conditional = off / mass.unsqueeze(-1)
        uniform = torch.full_like(conditional, 1 / conditional.shape[-1])
        torch.testing.assert_close(
            routing_kl(original, result)[..., row], mass * routing_kl(conditional, uniform)
        )


def test_exact_kl_support_and_boundary_self_masses():
    original = torch.tensor([[1.0, 0.0, 0.0], [0.25, 0.0, 0.75], [0.1, 0.2, 0.7]])
    identity = routing_probability_reference(original, "identity")
    kl = routing_kl(original, identity)
    assert kl[0] == 0
    assert torch.isposinf(kl[1:]).all()
    result = routing_probability_reference(original, "preserve_self")
    assert torch.isfinite(routing_kl(original, result)).all()
    assert torch.equal(routing_kl(result, result), torch.zeros(3))
    torch.testing.assert_close(result[1], torch.tensor([0.5, 0.0, 0.5]))


@pytest.mark.parametrize("mode", MODES)
def test_single_token_operator(mode):
    original = torch.ones(2, 3, 1, 1)
    assert torch.equal(routing_probability_reference(original, mode), original)


class FakeRoPE(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        angles = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1) * 0.37
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, value):
        left, right = value.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return value * self.cos + rotated * self.sin


class FakeNormAttention(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.num_heads, self.dim, self.head_dim = heads, dim, dim // heads
        self.q, self.k, self.v, self.proj = (nn.Linear(dim, dim) for _ in range(4))
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        batch, tokens, _ = x.shape
        q, k, v = (
            layer(x).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            for layer in (self.q, self.k, self.v)
        )
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.permute(0, 2, 1, 3).reshape(batch, tokens, self.dim))


class NativeRMSNorm(nn.Module):
    """RAEv2's dtype-promoting RMSNorm arithmetic, without external imports."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value):
        fp32 = value.float()
        normalized = fp32 * torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return normalized.type_as(value) * self.weight


@pytest.mark.parametrize("mode", MODES + AUDIT_MODES)
@pytest.mark.parametrize("tokens", [1, 4])
def test_explicit_attention_matches_probability_matrix_reference(mode, tokens):
    torch.manual_seed(431)
    attention, rope = FakeNormAttention(), FakeRoPE(tokens)
    x = torch.randn(2, tokens, 8)
    actual, telemetry = _routing_attention(attention, x, rope, mode)
    q, k, v = (
        layer(x).reshape(2, tokens, 2, 4).permute(0, 2, 1, 3)
        for layer in (attention.q, attention.k, attention.v)
    )
    q, k = rope(attention.q_norm(q)), rope(attention.k_norm(k))
    probabilities = ((q @ k.transpose(-2, -1)) / math.sqrt(4)).softmax(-1)
    reference = routing_probability_reference(probabilities, mode)
    expected = reference @ v
    expected = attention.proj(expected.permute(0, 2, 1, 3).reshape(2, tokens, 8))
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(telemetry["self_mass"], probabilities.diagonal(dim1=-2, dim2=-1))
    torch.testing.assert_close(telemetry["routing_kl"], routing_kl(probabilities, reference))
    assert all(value.shape == (2, 2, tokens) and not value.requires_grad for value in telemetry.values())


def test_bf16_autocast_uses_fp32_routing_and_casts_before_projection():
    torch.manual_seed(432)
    attention, rope = FakeNormAttention(), FakeRoPE(4)
    observed = []
    handle = attention.proj.register_forward_pre_hook(lambda module, args: observed.append(args[0].dtype))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, telemetry = _routing_attention(attention, torch.randn(2, 4, 8), rope, "preserve_self")
    finally:
        handle.remove()
    assert observed == [torch.bfloat16]
    assert output.dtype == torch.bfloat16
    assert all(value.dtype == torch.float32 for value in telemetry.values())


def test_explicit_routing_matches_native_autocast_input_quantization_after_rope():
    torch.manual_seed(811)
    attention, rope = FakeNormAttention(), FakeRoPE(7)
    attention.q_norm, attention.k_norm = NativeRMSNorm(4), NativeRMSNorm(4)
    x = torch.randn(2, 7, 8)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        q, k, v = (
            layer(x).reshape(2, 7, 2, 4).permute(0, 2, 1, 3)
            for layer in (attention.q, attention.k, attention.v)
        )
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        q, k = rope(attention.q_norm(q)), rope(attention.k_norm(k))
        assert q.dtype == k.dtype == torch.float32
        native = F.scaled_dot_product_attention(q, k, v)
        with torch.autocast("cpu", enabled=False):
            lowered_q, lowered_k = q.to(v.dtype), k.to(v.dtype)
            native_explicit_inputs = F.scaled_dot_product_attention(lowered_q, lowered_k, v)
            probability = (lowered_q.float() @ lowered_k.float().transpose(-2, -1) / 2).softmax(-1)
            higher_precision_probability = (q @ k.transpose(-2, -1) / 2).softmax(-1)
            expected_values = (probability @ v.float()).to(v.dtype)
        expected = attention.proj(expected_values.permute(0, 2, 1, 3).reshape(2, 7, 8))
        actual, telemetry = _routing_attention(attention, x, rope, "native_explicit")
    assert torch.equal(native, native_explicit_inputs)
    assert torch.equal(actual, expected)
    expected_mass = probability.diagonal(dim1=-2, dim2=-1)
    assert torch.equal(telemetry["self_mass"], expected_mass)
    # The old unquantized FP32 q/k path is observably different in this probe.
    assert not torch.equal(expected_mass, higher_precision_probability.diagonal(dim1=-2, dim2=-1))


class FakePatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = 4
        self.proj = nn.Conv2d(3, 8, 1)
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return self.proj(value).flatten(2).transpose(1, 2)


class FakeEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
        self.attn = FakeNormAttention()
        self.mlp = nn.Sequential(nn.Linear(8, 13), nn.SiLU(), nn.Linear(13, 8))
        self.calls = 0

    def forward(self, x, rope, attn_mask=None):
        self.calls += 1
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        return x + self.mlp(self.norm2(x))


class FakeDecoderBlock(FakeEncoderBlock):
    def __init__(self):
        super().__init__()
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(8, 48))

    def forward(self, x, c, rope, attn_mask=None):
        self.calls += 1
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln_modulation(c).chunk(6, -1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1, rope, attn_mask)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class FakeFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.condition = nn.Linear(8, 16)
        self.linear = nn.Linear(8, 3)

    def forward(self, x, condition):
        shift, scale = self.condition(condition).chunk(2, -1)
        return self.linear(self.norm(x) * (1 + scale) + shift)


class FakeDDT(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_enc_blocks = self.num_dec_blocks = 2
        self.base_model_depth = 1
        self.x_patch_size = self.s_patch_size = 1
        self.s_embedder, self.x_embedder = FakePatchEmbed(), FakePatchEmbed()
        self.time_embed = nn.Linear(1, 8)
        self.class_embed = nn.Embedding(4, 8)
        self.s_projector = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([FakeEncoderBlock(), FakeEncoderBlock(), FakeDecoderBlock(), FakeDecoderBlock()])
        self.final_layer, self.base_final_layer = FakeFinal(), FakeFinal()
        self.enc_rope, self.dec_rope = FakeRoPE(6), FakeRoPE(4)

    def _build_sequence(self, x, t, kwargs):
        t_base = self.time_embed(t[:, None])[:, None, :]
        return torch.cat((self.s_embedder(x), t_base, self.class_embed(kwargs["context"])[:, None, :]), 1), t_base

    def _build_attn_mask(self, seq, kwargs):
        return torch.zeros(seq.shape[0], 1, 1, seq.shape[1], dtype=seq.dtype)

    def unpatchify(self, x, patch):
        return x.transpose(1, 2).reshape(x.shape[0], 3, 2, 2)

    def forward(self, x, t, **kwargs):
        seq, t_base = self._build_sequence(x, t, kwargs)
        mask = self._build_attn_mask(seq, kwargs)
        base = None
        for index in range(self.num_enc_blocks):
            seq = self.blocks[index](seq, self.enc_rope, mask)
            if index + 1 == self.base_model_depth:
                base = seq[:, :self.s_embedder.num_patches, :]
        seq = self.s_projector(F.silu(t_base + seq[:, :self.s_embedder.num_patches, :]))
        x = self.x_embedder(x)
        for index in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + index](x, seq, self.dec_rope)
        x = self.unpatchify(self.final_layer(x, seq), self.x_patch_size)
        base = F.silu(t_base + base)
        base = self.unpatchify(self.base_final_layer(base, base), self.s_patch_size)
        return x, base


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("edited_block", [0, 1])
def test_shared_prefix_preserves_native_full_and_base_without_model_mutation(mode, edited_block):
    torch.manual_seed(789)
    model = FakeDDT().eval()
    state, times, labels = torch.randn(2, 3, 2, 2), torch.tensor([0.8, 0.35]), torch.tensor([1, 2])
    saved_parameters = {name: value.clone() for name, value in model.state_dict().items()}
    forward_functions = [module.forward.__func__ for module in model.modules()]
    with torch.no_grad():
        expected_full, expected_base = model(state, times, context=labels, attn_mask=None)
    for module in model.modules():
        if hasattr(module, "calls"):
            module.calls = 0
    result = evaluate_routing_reference(model, state, times, labels, mode, edited_block)
    assert torch.equal(result.full, expected_full)
    assert torch.equal(result.base, expected_base)
    assert result.negative.shape == result.full.shape
    assert torch.isfinite(result.negative).all()
    assert not torch.allclose(result.negative, result.full, atol=1e-6, rtol=1e-6)
    assert model.s_embedder.calls == 1 and model.x_embedder.calls == 2
    assert [block.calls for block in model.blocks[:2]] == [1, 1]
    assert [block.calls for block in model.blocks[2:]] == [1 if i == edited_block else 2 for i in range(2)]
    assert [module.forward.__func__ for module in model.modules()] == forward_functions
    assert all(torch.equal(value, saved_parameters[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("edited_block", [0, 1])
def test_native_explicit_decoder_replay_only_changes_numerical_attention_path(edited_block):
    torch.manual_seed(790)
    model = FakeDDT().eval()
    state, times, labels = torch.randn(2, 3, 2, 2), torch.tensor([0.8, 0.35]), torch.tensor([1, 2])
    result = evaluate_routing_reference(
        model, state, times, labels, mode="native_explicit", decoder_block=edited_block
    )
    torch.testing.assert_close(result.full, result.negative, atol=2e-6, rtol=2e-6)
    assert all(torch.isfinite(value).all() for value in result.telemetry.values())
    assert torch.equal(result.telemetry["routing_kl"], torch.zeros_like(result.telemetry["routing_kl"]))
    assert torch.equal(result.telemetry["routing_reverse_kl"], torch.zeros_like(result.telemetry["routing_reverse_kl"]))
    assert torch.equal(result.telemetry["self_mass"], result.telemetry["negative_self_mass"])


def test_invalid_mode_shape_and_decoder_index_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        routing_probability_reference(probabilities(), "mystery")
    with pytest.raises(ValueError, match="square"):
        routing_probability_reference(torch.ones(2, 3), "identity")
    model = FakeDDT().eval()
    state, times, labels = torch.randn(2, 3, 2, 2), torch.ones(2), torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="decoder_block"):
        evaluate_routing_reference(model, state, times, labels, decoder_block=2)
    with pytest.raises(ValueError, match="time and label"):
        evaluate_routing_reference(model, state, times[:1], labels)
    model.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        evaluate_routing_reference(model, state, times, labels)
