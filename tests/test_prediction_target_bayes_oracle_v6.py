from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.run_prediction_target_bayes_oracle_v5 import (
    TangentGaussianMixture,
    build_same_init_models,
    sample_condition,
)
from experiments.run_prediction_target_bayes_oracle_v6_trajectory import (
    checkpoint_path,
    condition_spec,
    latest_checkpoint,
    load_checkpoint,
    recent_relative_change,
    recent_relative_span,
    sample_conditions_joint,
    save_checkpoint,
    validate_setting_manifest,
)


def test_recent_relative_change_uses_latest_points() -> None:
    history = [
        {"x_excess_mse": 10.0},
        {"x_excess_mse": 4.0},
        {"x_excess_mse": 3.0},
        {"x_excess_mse": 2.0},
    ]
    assert recent_relative_change(history, "x", points=3) == -0.5
    assert recent_relative_span(history, "x", points=3) == 2.0 / 3.0


def test_condition_spec_has_one_shared_baseline() -> None:
    rows = condition_spec([-0.03, 0.03], [-0.03, 0.03])
    names = [row[0] for row in rows]
    assert len(names) == len(set(names))
    assert names[:4] == ["bayes", "x", "v", "eps"]
    assert ("xv_g0p03", "xv", 0.03) in rows


def test_checkpoint_restores_model_optimizer_and_generator(tmp_path: Path) -> None:
    device = torch.device("cpu")
    models = build_same_init_models(
        "residual", D=5, hidden=7, depth=3, time_dim=4, device=device, seed=17
    )
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=1e-3)
        for name, model in models.items()
    }
    generator = torch.Generator().manual_seed(23)
    expected_generator = generator.get_state().clone()
    expected_parameter = next(models["x"].parameters()).detach().clone()
    path = checkpoint_path(tmp_path, 20)
    save_checkpoint(
        path=path,
        step=20,
        models=models,
        optimizers=optimizers,
        generator=generator,
        history=[{"step": 20, "x_excess_mse": 0.2}],
    )
    with torch.no_grad():
        next(models["x"].parameters()).add_(1.0)
    generator.manual_seed(99)
    step, history = load_checkpoint(
        path=path,
        models=models,
        optimizers=optimizers,
        generator=generator,
        device=device,
    )
    assert step == 20
    assert history[0]["step"] == 20
    assert torch.equal(next(models["x"].parameters()), expected_parameter)
    assert torch.equal(generator.get_state(), expected_generator)
    assert latest_checkpoint(tmp_path, 19) is None
    assert latest_checkpoint(tmp_path, 20) == path


def test_setting_manifest_rejects_changed_semantics(tmp_path: Path) -> None:
    path = tmp_path / "training_manifest.json"
    validate_setting_manifest(path=path, expected={"lr": 1e-3}, resume=False)
    validate_setting_manifest(path=path, expected={"lr": 1e-3}, resume=True)
    try:
        validate_setting_manifest(path=path, expected={"lr": 2e-3}, resume=True)
    except ValueError as error:
        assert "manifest mismatch" in str(error)
    else:
        raise AssertionError("changed semantics should not resume")


def test_joint_sampler_matches_individual_rollouts() -> None:
    device = torch.device("cpu")
    mixture = TangentGaussianMixture(
        D=5,
        components=3,
        curvature=0.2,
        frequency_scale=3.0,
        center_rms=0.6,
        sigma_tangent=0.2,
        sigma_normal=0.03,
        seed=11,
        device=device,
    )
    models = build_same_init_models(
        "residual", D=5, hidden=7, depth=3, time_dim=4, device=device, seed=17
    )
    conditions = condition_spec([-0.03, 0.03], [-0.03, 0.03])
    joint = sample_conditions_joint(
        models=models,
        mixture=mixture,
        conditions=conditions,
        sample_count=8,
        batch_size=4,
        steps=4,
        t_max=0.9,
        t_min=0.1,
        clip=0.02,
        seed=31,
    )
    for name, kind, strength in conditions:
        if kind == "bayes":
            continue
        individual = sample_condition(
            models=models,
            mixture=mixture,
            kind=kind,
            strength=strength,
            sample_count=8,
            batch_size=4,
            steps=4,
            t_max=0.9,
            t_min=0.1,
            clip=0.02,
            seed=31,
        )
        np.testing.assert_allclose(joint[name], individual, rtol=2e-5, atol=2e-5)
