from __future__ import annotations

from pathlib import Path

import torch
import pytest
from omegaconf import OmegaConf

from experiments.rae_layerwise_path import plan_layerwise_path, random_detail_basis
from experiments.train_rae_layerwise_path import (
    active_path_mode,
    exact_resume_requested,
    parse_save_steps,
    resolve_stage1_paths,
    verify_restored_optimizer_config,
)


def test_static_path_is_official_linear_flow_path() -> None:
    generator = torch.Generator().manual_seed(31)
    clean = torch.randn((4, 10, 3, 3), generator=generator)
    noise = torch.randn((4, 10, 3, 3), generator=generator)
    time = torch.rand((4,), generator=generator)
    basis = random_detail_basis(10, 3, seed=37)
    plan = plan_layerwise_path(clean, noise, time, basis, mode="static", power=2.0)
    expanded = time[:, None, None, None]
    torch.testing.assert_close(
        plan.state,
        (1.0 - expanded) * clean + expanded * noise,
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(plan.target, noise - clean, atol=2e-6, rtol=0)


def test_stage1_paths_are_resolved_from_rae_root() -> None:
    config = OmegaConf.create(
        {
            "stage_1": {
                "params": {
                    "decoder_config_path": "configs/decoder/ViTXL",
                    "pretrained_decoder_path": "models/decoder.pt",
                    "normalization_stat_path": None,
                }
            }
        }
    )
    resolve_stage1_paths(config, Path("/tmp/rae"))
    assert config.stage_1.params.decoder_config_path == "/tmp/rae/configs/decoder/ViTXL"
    assert config.stage_1.params.pretrained_decoder_path == "/tmp/rae/models/decoder.pt"
    assert config.stage_1.params.normalization_stat_path is None


def test_restored_optimizer_must_match_declared_config() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [parameter], lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    training = {
        "optimizer": {"lr": 2e-4, "betas": [0.9, 0.95], "weight_decay": 0.0},
        "scheduler": {"base_lr": 2e-4},
    }
    restored = verify_restored_optimizer_config(optimizer, scheduler, training)
    assert restored["lr"] == 2e-4

    training["optimizer"]["lr"] = 1e-4
    with pytest.raises(ValueError, match="restored lr"):
        verify_restored_optimizer_config(optimizer, scheduler, training)


def test_resumed_optimizer_accepts_scheduler_current_lr_but_checks_base() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [parameter], lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - 0.1 * step
    )
    optimizer.step()
    scheduler.step()
    training = {
        "optimizer": {"lr": 2e-4, "betas": [0.9, 0.95], "weight_decay": 0.0},
        "scheduler": {"base_lr": 2e-4},
    }
    restored = verify_restored_optimizer_config(
        optimizer, scheduler, training, resumed=True
    )
    assert restored["lr"] == pytest.approx(1.8e-4)

    scheduler.base_lrs[0] = 1e-4
    with pytest.raises(ValueError, match="scheduler_base_lr"):
        verify_restored_optimizer_config(
            optimizer, scheduler, training, resumed=True
        )


def test_full_state_fork_uses_exact_resume_semantics() -> None:
    checkpoint = Path("/tmp/local/checkpoint.pt")
    assert exact_resume_requested(checkpoint, False)
    assert exact_resume_requested(None, True)
    assert not exact_resume_requested(None, False)


def test_path_switch_has_exact_optimizer_step_boundary() -> None:
    assert active_path_mode(0, "annealed", 2000, "static") == "annealed"
    assert active_path_mode(1999, "annealed", 2000, "static") == "annealed"
    assert active_path_mode(2000, "annealed", 2000, "static") == "static"
    assert active_path_mode(4999, "annealed", 2000, "static") == "static"
    assert active_path_mode(10, "static", None, None) == "static"


def test_path_switch_rejects_incomplete_configuration() -> None:
    with pytest.raises(ValueError, match="requires path-mode-after-switch"):
        active_path_mode(0, "annealed", 2000, None)
    with pytest.raises(ValueError, match="requires path-switch-step"):
        active_path_mode(0, "annealed", None, "static")


def test_parse_save_steps() -> None:
    assert parse_save_steps("2000,5000") == {2000, 5000}
    with pytest.raises(ValueError, match="positive"):
        parse_save_steps("0,5000")
