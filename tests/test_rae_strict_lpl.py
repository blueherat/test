from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from experiments.rae_strict_lpl import (
    cross_normalize_decoder_features,
    decoder_feature_pyramid,
    decoder_hidden_indices,
    decoder_outlier_mask,
    flow_clean_estimate,
    lpl_time_gate,
    noise_to_signal_ratio,
    strict_lpl_per_sample,
)
from experiments.train_rae_strict_lpl import (
    IndexedDataset,
    assert_frozen_modules_have_no_grad,
    assert_optimizer_boundary,
    model_state_from_checkpoint,
    moments_from_totals,
    text_sequence_sha256,
    validate_resume_policy,
    variance_matched_weight,
)


def test_linear_flow_clean_estimate_recovers_clean_latent() -> None:
    clean = torch.randn(3, 2, 4, 4)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.1, 0.5, 0.9])
    scale = time[:, None, None, None]
    noisy = (1.0 - scale) * clean + scale * noise
    velocity = noise - clean
    recovered = flow_clean_estimate(noisy, velocity, time)
    torch.testing.assert_close(recovered, clean)


def test_flow_lpl_gate_uses_noise_to_signal_ratio() -> None:
    time = torch.tensor([0.25, 0.75, 0.80])
    torch.testing.assert_close(noise_to_signal_ratio(time), torch.tensor([1 / 3, 3.0, 4.0]))
    assert lpl_time_gate(time, 3.0).tolist() == [True, True, False]


def test_outlier_mask_keeps_regular_features() -> None:
    feature = torch.linspace(-1.0, 1.0, 16 * 16).reshape(1, 1, 16, 16)
    mask = decoder_outlier_mask(feature)
    assert mask.dtype == torch.bool
    assert float(mask.float().mean()) > 0.95


def test_outlier_mask_matches_independent_paper_pseudocode() -> None:
    torch.manual_seed(17)
    feature = torch.randn(2, 3, 16, 16)
    feature[0, 1, 4, 7] = 1e4
    flat = feature.flatten(-2)
    lower = flat.kthvalue(int(flat.shape[-1] * 0.02), dim=-1).values[
        ..., None, None
    ]
    upper = flat.kthvalue(int(flat.shape[-1] * 0.98), dim=-1).values[
        ..., None, None
    ]
    margin = 2 * flat.std(-1)[..., None, None].detach()
    reference = ((lower - margin < feature) & (feature < upper + margin)).float()
    reference = F.max_pool2d(reference, 3, stride=1, padding=1)
    reference = -F.max_pool2d(-reference, 5, stride=1, padding=2)

    actual = decoder_outlier_mask(feature)

    assert torch.equal(actual, reference.bool())


def test_cross_normalization_uses_prediction_statistics() -> None:
    prediction = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = prediction + 2.0
    mask = torch.ones_like(prediction, dtype=torch.bool)
    normalized_target, normalized_prediction = cross_normalize_decoder_features(
        target, prediction, mask, eps=0.0
    )
    torch.testing.assert_close(normalized_prediction.mean((-2, -1)), torch.zeros(1, 1))
    torch.testing.assert_close(
        normalized_prediction.square().mean((-2, -1)), torch.ones(1, 1)
    )
    assert torch.all(normalized_target > normalized_prediction)


def test_strict_lpl_is_zero_for_equal_features_and_backpropagates() -> None:
    prediction = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = prediction.detach().clone()
    loss, details = strict_lpl_per_sample([target], [prediction])
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    assert details["layer_losses"].shape == (2, 1)
    loss.sum().backward()
    assert prediction.grad is not None


def test_strict_lpl_has_gradient_only_through_prediction() -> None:
    target = torch.randn(2, 4, 16, 16, requires_grad=True)
    prediction = (target.detach() + 0.1 * torch.randn_like(target)).requires_grad_(True)
    loss, _ = strict_lpl_per_sample([target.detach()], [prediction])
    loss.mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert target.grad is None


def test_strict_lpl_matches_expanded_equation_three() -> None:
    target = torch.tensor(
        [
            [
                [[-1.0, 0.0], [1.0, 2.0]],
                [[2.0, 0.0], [-2.0, 1.0]],
            ]
        ]
    )
    prediction = target + torch.tensor(
        [
            [
                [[0.2, -0.1], [0.3, -0.2]],
                [[-0.3, 0.2], [0.1, 0.4]],
            ]
        ]
    )
    mask = decoder_outlier_mask(
        prediction,
        quantile=0.25,
        opening=1,
        closing=1,
    )
    weights = mask.to(prediction.dtype)
    count = weights.sum((-2, -1), keepdim=True).clamp_min(1)
    prediction_mean = (prediction * weights).sum((-2, -1), keepdim=True) / count
    prediction_variance = (
        ((prediction - prediction_mean) * weights)
        .square()
        .sum((-2, -1), keepdim=True)
        / count
    )
    expected = (
        ((target - prediction).square() * weights)
        / (prediction_variance + 1e-6)
    ).sum((-2, -1)).mean(1)

    actual, _ = strict_lpl_per_sample(
        [target],
        [prediction],
        outlier_quantile=0.25,
        outlier_opening=1,
        outlier_closing=1,
        eps=1e-6,
    )

    torch.testing.assert_close(actual, expected)


def test_decoder_hidden_indices_cover_five_relative_depths() -> None:
    assert decoder_hidden_indices(28) == (6, 11, 17, 22, 28)


class _ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, tokens, **_kwargs):
        cls = torch.zeros(tokens.shape[0], 1, tokens.shape[2], device=tokens.device)
        state0 = torch.cat([cls, tokens], dim=1)
        return SimpleNamespace(hidden_states=(state0, state0 + 1.0, state0 + 2.0))


class _ToyRAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _ToyDecoder()
        self.reshape_to_2d = True
        self.do_normalization = False


def test_decoder_feature_pyramid_preserves_spatial_token_layout() -> None:
    latent = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    features = decoder_feature_pyramid(_ToyRAE(), latent, layer_indices=(1, 2))
    assert [tuple(value.shape) for value in features] == [(1, 2, 4, 4), (1, 2, 4, 4)]
    torch.testing.assert_close(features[0], latent + 1.0)
    torch.testing.assert_close(features[1], latent + 2.0)


def test_model_state_from_checkpoint_supports_official_and_wrapped_weights() -> None:
    state = {"weight": torch.ones(2)}
    assert model_state_from_checkpoint(state) is state
    assert model_state_from_checkpoint({"ema": state}) is state
    assert model_state_from_checkpoint({"model": state}) is state
    assert model_state_from_checkpoint({"state_dict": state}) is state


def test_model_state_from_checkpoint_removes_uniform_ddp_prefix() -> None:
    state = model_state_from_checkpoint({"module.weight": torch.ones(2)})
    assert list(state) == ["weight"]


def test_incomplete_legacy_resume_requires_explicit_opt_in(tmp_path) -> None:
    checkpoint = tmp_path / "step-0000500.pt"
    checkpoint.touch()

    try:
        validate_resume_policy(
            checkpoint,
            endpoint_step=2000,
            allow_nonexact_resume=False,
        )
    except RuntimeError as error:
        assert "dataloader cursor" in str(error)
    else:
        raise AssertionError("incomplete legacy resume should be rejected")

    validate_resume_policy(
        checkpoint,
        endpoint_step=2000,
        allow_nonexact_resume=True,
    )
    validate_resume_policy(
        checkpoint,
        endpoint_step=500,
        allow_nonexact_resume=False,
    )


def test_variance_calibration_matches_requested_ratio() -> None:
    flow = torch.tensor([1.0, 2.0, 4.0, 7.0], dtype=torch.float64)
    feature = torch.tensor([0.0, 1.0, 1.0, 3.0], dtype=torch.float64)
    flow_moments = moments_from_totals(
        flow.numel(), float(flow.sum()), float(flow.square().sum())
    )
    feature_moments = moments_from_totals(
        feature.numel(), float(feature.sum()), float(feature.square().sum())
    )
    weight = variance_matched_weight(
        flow_moments["variance"],
        feature_moments["variance"],
        target_ratio=0.1,
    )

    measured_ratio = (
        (weight * feature).var(unbiased=False) / flow.var(unbiased=False)
    )
    torch.testing.assert_close(measured_ratio, torch.tensor(0.1, dtype=torch.float64))


def test_indexed_dataset_preserves_item_and_adds_source_index() -> None:
    base = [(torch.tensor([1.0]), 4), (torch.tensor([2.0]), 7)]
    dataset = IndexedDataset(base)

    image, label, index = dataset[1]

    torch.testing.assert_close(image, torch.tensor([2.0]))
    assert label == 7
    assert index == 1


def test_text_sequence_hash_is_ordered_and_delimited() -> None:
    assert text_sequence_sha256(["train-a", "train-b"]) == text_sequence_sha256(
        ["train-a", "train-b"]
    )
    assert text_sequence_sha256(["train-a", "train-b"]) != text_sequence_sha256(
        ["train-b", "train-a"]
    )
    assert text_sequence_sha256(["ab", "c"]) != text_sequence_sha256(["a", "bc"])


def test_optimizer_and_frozen_gradient_boundaries_are_enforced() -> None:
    model = nn.Linear(2, 2)
    frozen = nn.Linear(2, 2).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    assert_optimizer_boundary(
        optimizer,
        trainable_model=model,
        frozen_modules=(frozen,),
    )
    assert_frozen_modules_have_no_grad((frozen,))

    next(frozen.parameters()).grad = torch.ones_like(next(frozen.parameters()))
    try:
        assert_frozen_modules_have_no_grad((frozen,))
    except RuntimeError as error:
        assert "frozen parameters received gradients" in str(error)
    else:
        raise AssertionError("a frozen-module gradient should fail the audit")
