from __future__ import annotations

import torch
from torch import nn

from experiments.raev2_common_adapter import (
    CommonResidualAdapter,
    ContrastPreservingCommonAdapterModel,
    forward_with_internalguidance_common_adapter,
    internal_guidance_prediction,
)


class ToyDualHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_channels = channels
        self.full = nn.Conv2d(channels, channels, kernel_size=1)
        self.base = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        return_intermediate: bool = False,
        **_: torch.Tensor,
    ):
        full = self.full(x) + time[:, None, None, None]
        base = self.base(x) - time[:, None, None, None]
        output = (full, base)
        if return_intermediate:
            return output, x.mean(dim=(2, 3))
        return output


class ToyBFloatDualHead(ToyDualHead):
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if kwargs.get("return_intermediate", False):
            heads, intermediate = output
            return tuple(value.bfloat16() for value in heads), intermediate
        return tuple(value.bfloat16() for value in output)


def _fixture():
    torch.manual_seed(3)
    source = ToyDualHead(channels=4)
    adapter = CommonResidualAdapter(channels=4, hidden_channels=8)
    wrapped = ContrastPreservingCommonAdapterModel(source, adapter)
    x = torch.randn(2, 4, 5, 5)
    time = torch.tensor([0.2, 0.7])
    return source, adapter, wrapped, x, time


def test_zero_initialized_adapter_is_exact_identity():
    source, _, wrapped, x, time = _fixture()
    expected = source(x, time)
    actual = wrapped(x, time)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_common_adapter_preserves_contrast_after_nonzero_update():
    source, adapter, wrapped, x, time = _fixture()
    nn.init.normal_(adapter.output_projection.weight, std=0.05)
    nn.init.normal_(adapter.output_projection.bias, std=0.02)
    source_full, source_base = source(x, time)
    full, base = wrapped(x, time)
    torch.testing.assert_close(
        full - base,
        source_full - source_base,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(full - source_full, base - source_base)


def test_guided_output_changes_only_by_common_correction():
    source, adapter, wrapped, x, time = _fixture()
    nn.init.normal_(adapter.output_projection.weight, std=0.05)
    scale = 1.78
    source_full, source_base = source(x, time)
    full, base = wrapped(x, time)
    source_guided = source_base + scale * (source_full - source_base)
    guided = base + scale * (full - base)
    torch.testing.assert_close(guided - source_guided, full - source_full)


def test_only_adapter_receives_gradients():
    source, adapter, wrapped, x, time = _fixture()
    wrapped.train()
    full, base = wrapped(x, time)
    (full.square().mean() + base.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert all(parameter.grad is None for parameter in source.parameters())
    assert not source.training
    assert adapter.training


def test_intermediate_output_is_forwarded_unchanged():
    source, _, wrapped, x, time = _fixture()
    expected_output, expected_intermediate = source(
        x,
        time,
        return_intermediate=True,
    )
    actual_output, actual_intermediate = wrapped(
        x,
        time,
        return_intermediate=True,
    )
    torch.testing.assert_close(actual_output[0], expected_output[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_output[1], expected_output[1], rtol=0, atol=0)
    torch.testing.assert_close(actual_intermediate, expected_intermediate)


def test_sampling_path_preserves_official_bfloat_guidance_at_zero_init():
    torch.manual_seed(11)
    source = ToyBFloatDualHead(channels=4)
    adapter = CommonResidualAdapter(channels=4, hidden_channels=8)
    noisy = torch.randn(4, 4, 5, 5)
    time = torch.tensor([0.2, 0.7, 0.2, 0.7])
    scale = 1.78

    full, base = source(noisy[:2], time[:2])
    expected_half = base + scale * (full - base)
    expected = torch.cat((expected_half, expected_half), dim=0).float()
    actual = forward_with_internalguidance_common_adapter(
        source,
        adapter,
        noisy,
        time,
        ig_scale=scale,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sampling_path_adds_only_one_common_correction():
    torch.manual_seed(17)
    source = ToyBFloatDualHead(channels=4)
    adapter = CommonResidualAdapter(channels=4, hidden_channels=8)
    nn.init.normal_(adapter.output_projection.weight, std=0.05)
    nn.init.normal_(adapter.output_projection.bias, std=0.02)
    noisy = torch.randn(4, 4, 5, 5)
    time = torch.tensor([0.2, 0.7, 0.2, 0.7])
    scale = 1.78

    full, base = source(noisy[:2], time[:2])
    source_guided = (base + scale * (full - base)).float()
    correction = adapter(noisy[:2], time[:2], full, base).float()
    actual = forward_with_internalguidance_common_adapter(
        source,
        adapter,
        noisy,
        time,
        ig_scale=scale,
    )[:2]

    torch.testing.assert_close(
        actual - source_guided,
        correction,
        rtol=1e-5,
        atol=1e-6,
    )


def test_internal_guidance_respects_sampling_interval():
    full = torch.tensor([[[[2.0]]], [[[3.0]]], [[[4.0]]]])
    base = torch.tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]]])
    time = torch.tensor([0.05, 0.5, 0.95])
    actual = internal_guidance_prediction(
        full,
        base,
        time,
        scale=2.0,
        interval=(0.1, 0.9),
    )
    expected = torch.tensor([[[[2.0]]], [[[5.0]]], [[[4.0]]]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
