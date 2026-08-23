import torch
import torch.nn as nn

from torch_fidelity.feature_extractor_inceptionv3 import (
    FeatureExtractorInceptionV3,
)

from experiments.advfd_cleanroom.feature_extractors import (
    DifferentiableInception2048,
    DifferentiableTimmFeature,
    TimmFeatureSpec,
    generator_output_to_unit_interval,
)


class FakeTimmModel(nn.Module):
    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=(-2, -1))

    def forward_head(self, hidden: torch.Tensor, *, pre_logits: bool) -> torch.Tensor:
        assert pre_logits
        return torch.cat((hidden, hidden[:, :1]), dim=1)


def test_differentiable_inception_matches_torch_fidelity_uint8_path() -> None:
    generator = torch.Generator().manual_seed(31)
    images_uint8 = torch.randint(
        0, 256, (2, 3, 64, 64), generator=generator, dtype=torch.uint8
    )
    reference = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], verbose=False
    ).eval()
    candidate = DifferentiableInception2048().eval()
    with torch.no_grad():
        expected = reference(images_uint8)[0]
        actual = candidate(images_uint8.float() / 255.0)
    assert actual.shape == (2, 2048)
    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_differentiable_inception_propagates_pixel_gradients() -> None:
    generator = torch.Generator().manual_seed(37)
    images = torch.rand(1, 3, 64, 64, generator=generator, requires_grad=True)
    model = DifferentiableInception2048().eval()
    model(images).square().mean().backward()
    assert images.grad is not None
    assert images.grad.abs().sum().item() > 0.0


def test_generator_output_conversion_clamps_to_valid_range() -> None:
    values = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    converted = generator_output_to_unit_interval(values)
    assert torch.equal(
        converted, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    )


def test_timm_feature_preprocesses_and_propagates_image_gradients() -> None:
    spec = TimmFeatureSpec(
        name="fake",
        model_name="fake",
        output_dim=4,
        input_size=8,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
    )
    images = torch.rand(2, 3, 12, 12, requires_grad=True)
    encoder = DifferentiableTimmFeature(
        spec, pretrained=False, model=FakeTimmModel()
    )
    features = encoder(images)
    assert features.shape == (2, 4)
    features.square().mean().backward()
    assert images.grad is not None
    assert images.grad.abs().sum().item() > 0.0
