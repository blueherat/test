import torch

from experiments.mnist_generation_time_prior import (
    LatentVelocityMLP,
    PriorConfig,
    categorical_metrics,
    sample_prior,
    train_prior,
)
from experiments.mnist_generation_time_bottleneck import BottleneckConfig


def test_latent_prior_forward_and_sampling_shapes():
    model = LatentVelocityMLP(4, hidden_size=16, depth=1)
    value = torch.randn(8, 4)
    time = torch.rand(8)
    assert model(value, time).shape == value.shape
    sample = sample_prior(model, 7, 4, 2, seed=0, device=torch.device("cpu"))
    assert sample.shape == (7, 4)
    assert torch.isfinite(sample).all()


def test_prior_training_smoke_uses_stage2_latent_dim(tmp_path):
    run = tmp_path / "stage2"
    run.mkdir()
    values = BottleneckConfig(
        mode="high_noise", latent_dim=4, device="cpu", save=False
    )
    config = values.__dict__.copy()
    config["data_root"] = str(config["data_root"])
    config["output_root"] = str(config["output_root"])
    config["eval_times"] = list(config["eval_times"])
    import json

    (run / "config.json").write_text(json.dumps(config))
    prior, history = train_prior(
        torch.randn(64, 4),
        PriorConfig(
            stage2_run=run,
            steps=3,
            batch_size=16,
            hidden_size=16,
            depth=1,
            device="cpu",
            save=False,
        ),
    )
    assert isinstance(prior, LatentVelocityMLP)
    assert len(history) == 3
    assert torch.isfinite(torch.tensor(history.loss.to_numpy())).all()


def test_categorical_metrics_report_effective_support():
    logits = torch.eye(10) * 20.0
    metrics = categorical_metrics(logits)
    assert metrics["effective_classes"] > 9.9
    assert metrics["classifier_confidence"] > 0.99
