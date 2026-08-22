from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenet100_sit_progressive_innovation import (
    configure_training_phase,
    create_progressive_innovation_sit,
    innovation_losses,
    trainable_parameter_names,
)
from experiments.train_imagenet100_sit_flow import (
    DEFAULT_OFFICIAL_SIT_REPO,
    load_official_sit_module,
)


def make_tiny_progressive(split_depth: int = 2):
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    source = sit_module.SiT(
        input_size=4,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=4,
        num_heads=4,
        num_classes=10,
        learn_sigma=True,
    )
    weak_head = sit_module.FinalLayer(32, 2, 4)
    with torch.no_grad():
        for parameter in weak_head.parameters():
            parameter.zero_()
    from experiments.imagenet100_sit_progressive_innovation import (
        ProgressiveInnovationSiT,
    )

    return ProgressiveInnovationSiT(
        source,
        weak_head,
        split_depth=split_depth,
        latent_channels=4,
    )


def test_factory_preserves_baseline_source_initialization() -> None:
    sit_module, _ = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    torch.manual_seed(17)
    baseline = sit_module.SiT_models["SiT-S/2"](
        input_size=32,
        num_classes=100,
        class_dropout_prob=0.1,
    )
    torch.manual_seed(17)
    progressive = create_progressive_innovation_sit(
        sit_module,
        model_name="SiT-S/2",
        num_classes=100,
        input_size=32,
        cfg_dropout=0.1,
        split_depth=6,
        latent_channels=4,
    )

    for name, value in baseline.state_dict().items():
        assert torch.equal(value, progressive.source.state_dict()[name]), name


def test_cumulative_output_is_exact_sum_of_stages() -> None:
    torch.manual_seed(23)
    model = make_tiny_progressive().eval()
    with torch.no_grad():
        model.weak_head.linear.weight.normal_()
        model.source.final_layer.linear.weight.normal_()
    state = torch.randn(3, 4, 4, 4)
    time_value = torch.tensor([0.2, 0.5, 0.8])
    labels = torch.tensor([1, 2, 3])

    with torch.inference_mode():
        weak, innovation, cumulative = model.forward_components(
            state, time_value, labels
        )
        direct = model(state, time_value, labels)

    assert torch.equal(cumulative, weak + innovation)
    assert torch.equal(direct, cumulative)


def test_training_phases_have_disjoint_parameter_sets() -> None:
    model = make_tiny_progressive()
    configure_training_phase(model, "weak")
    weak_names = set(trainable_parameter_names(model))
    assert any(name.startswith("source.x_embedder") for name in weak_names)
    assert any(name.startswith("source.blocks.0") for name in weak_names)
    assert any(name.startswith("weak_head") for name in weak_names)
    assert not any(name.startswith("source.blocks.2") for name in weak_names)
    assert not any(name.startswith("source.final_layer") for name in weak_names)

    configure_training_phase(model, "innovation")
    innovation_names = set(trainable_parameter_names(model))
    assert any(name.startswith("source.blocks.2") for name in innovation_names)
    assert any(name.startswith("source.final_layer") for name in innovation_names)
    assert not any(name.startswith("source.x_embedder") for name in innovation_names)
    assert not any(name.startswith("source.blocks.0") for name in innovation_names)
    assert not any(name.startswith("weak_head") for name in innovation_names)
    assert weak_names.isdisjoint(innovation_names)


def test_innovation_loss_cannot_backpropagate_into_weak_stage() -> None:
    torch.manual_seed(29)
    model = make_tiny_progressive().train()
    configure_training_phase(model, "innovation")
    with torch.no_grad():
        model.weak_head.linear.weight.normal_()
        model.source.final_layer.linear.weight.normal_()
    state = torch.randn(2, 4, 4, 4)
    time_value = torch.tensor([0.3, 0.7])
    labels = torch.tensor([1, 2])
    target = torch.randn_like(state)

    weak, innovation, _ = model.forward_components(state, time_value, labels)
    losses = innovation_losses(weak, innovation, target)
    losses["optimized"].backward()

    for name, parameter in model.named_parameters():
        if name.startswith("weak_head") or name.startswith("source.blocks.0"):
            assert parameter.grad is None, name
    assert model.source.final_layer.linear.weight.grad is not None
    assert model.source.blocks[2].attn.qkv.weight.grad is not None


def test_residual_and_cumulative_losses_are_numerically_identical() -> None:
    weak = torch.randn(3, 4, 5, 5, requires_grad=True)
    innovation = torch.randn_like(weak, requires_grad=True)
    target = torch.randn_like(weak)

    losses = innovation_losses(weak, innovation, target)

    torch.testing.assert_close(losses["optimized"], losses["cumulative"])
    losses["optimized"].backward()
    assert weak.grad is None
    assert innovation.grad is not None
