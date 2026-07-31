import torch

from experiments.raev2_head_swap import (
    combine_full_and_contrast,
    forward_with_head_swap,
)


class TwoHeadModel(torch.nn.Module):
    in_channels = 1

    def __init__(self, full_offset: float, base_offset: float) -> None:
        super().__init__()
        self.full_offset = float(full_offset)
        self.base_offset = float(base_offset)
        self.calls = 0

    def forward(self, x, t, **kwargs):
        del t, kwargs
        self.calls += 1
        return x + self.full_offset, x + self.base_offset


def test_combine_identical_source_matches_internal_guidance() -> None:
    full = torch.tensor([[[[5.0]]]])
    base = torch.tensor([[[[1.0]]]])

    result = combine_full_and_contrast(
        (full, base),
        (full, base),
        guidance_scale=1.75,
        identical_sources=True,
    )

    torch.testing.assert_close(result, base + 1.75 * (full - base))


def test_combine_swaps_full_and_contrast_independently() -> None:
    full_a = torch.tensor([[[[10.0]]]])
    base_a = torch.tensor([[[[8.0]]]])
    full_b = torch.tensor([[[[5.0]]]])
    base_b = torch.tensor([[[[1.0]]]])

    result = combine_full_and_contrast(
        (full_a, base_a),
        (full_b, base_b),
        guidance_scale=1.5,
    )

    torch.testing.assert_close(result, torch.tensor([[[[12.0]]]]))


def test_inactive_samples_use_full_source_without_guidance() -> None:
    full = torch.tensor([[[[5.0]]], [[[7.0]]]])
    base = torch.tensor([[[[1.0]]], [[[2.0]]]])
    result = combine_full_and_contrast(
        (full, base),
        (full, base),
        guidance_scale=2.0,
        active=torch.tensor([True, False]),
    )

    torch.testing.assert_close(result, torch.tensor([[[[9.0]]], [[[7.0]]]]))


def test_forward_queries_each_source_on_same_half_batch() -> None:
    full_model = TwoHeadModel(full_offset=10.0, base_offset=8.0)
    contrast_model = TwoHeadModel(full_offset=5.0, base_offset=1.0)
    x = torch.tensor([[[[2.0]]], [[[99.0]]]])
    t = torch.tensor([0.5, 0.5])

    result = forward_with_head_swap(
        full_model,
        contrast_model,
        x,
        t,
        guidance_scale=1.5,
        guidance_interval=(0.1, 1.0),
        context=torch.tensor([3, 1000]),
        attn_mask=None,
    )

    torch.testing.assert_close(result, torch.tensor([[[[14.0]]], [[[14.0]]]]))
    assert full_model.calls == 1
    assert contrast_model.calls == 1


def test_forward_reuses_one_call_when_sources_match() -> None:
    model = TwoHeadModel(full_offset=5.0, base_offset=1.0)
    x = torch.tensor([[[[2.0]]], [[[99.0]]]])
    t = torch.tensor([0.5, 0.5])

    result = forward_with_head_swap(
        model,
        model,
        x,
        t,
        guidance_scale=1.5,
        guidance_interval=(0.1, 1.0),
    )

    torch.testing.assert_close(result, torch.tensor([[[[9.0]]], [[[9.0]]]]))
    assert model.calls == 1
