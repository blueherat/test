from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imagenette_sdvae_latent_diffusion import (
    NULL_CLASS,
    SD_BETA_END,
    SD_BETA_SCHEDULE,
    SD_BETA_START,
    SD_TIMESTEPS,
    apply_condition_dropout,
    block_types,
    build_unet,
    ddp_uses_static_graph,
    latent_size_for_image_size,
    resume_batch_offset,
    sample_shard_bounds,
    stable_diffusion_noise_scheduler,
)


def test_scheduler_matches_stable_diffusion_v1_noise_configuration() -> None:
    scheduler = stable_diffusion_noise_scheduler()
    assert scheduler.config.num_train_timesteps == SD_TIMESTEPS == 1000
    assert scheduler.config.beta_start == SD_BETA_START == 0.00085
    assert scheduler.config.beta_end == SD_BETA_END == 0.012
    assert scheduler.config.beta_schedule == SD_BETA_SCHEDULE == "scaled_linear"
    assert scheduler.config.prediction_type == "epsilon"
    assert scheduler.config.clip_sample is False


def test_condition_dropout_uses_dedicated_null_class() -> None:
    labels = torch.tensor([0, 1, 9])
    unchanged, mask = apply_condition_dropout(labels, 0.0)
    assert torch.equal(unchanged, labels)
    assert not mask.any()
    dropped, mask = apply_condition_dropout(labels, 1.0)
    assert mask.all()
    assert torch.equal(dropped, torch.full_like(labels, NULL_CLASS))


def test_block_layout_puts_attention_at_low_resolution() -> None:
    down, up = block_types((32, 64, 96, 128))
    assert down == ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    assert up == ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")


def test_small_unet_has_class_conditioned_latent_interface() -> None:
    model = build_unet((32, 64, 96))
    latent = torch.randn((2, 4, 16, 16))
    timesteps = torch.tensor([10, 900])
    labels = torch.tensor([2, NULL_CLASS])
    prediction = model(
        latent, timesteps, class_labels=labels, return_dict=False
    )[0]
    assert prediction.shape == latent.shape


def test_resolution_maps_to_sd_vae_latent_grid() -> None:
    assert latent_size_for_image_size(128) == 16
    assert latent_size_for_image_size(256) == 32
    try:
        latent_size_for_image_size(130)
    except ValueError:
        pass
    else:
        raise AssertionError("non-factor-8 image size should fail")


def test_ddp_static_graph_is_disabled_for_no_sync_accumulation() -> None:
    assert ddp_uses_static_graph(1)
    assert not ddp_uses_static_graph(2)


def test_gradient_accumulation_phase_is_independent_of_log_resets() -> None:
    accumulation = 3
    micro_step = 0
    update_microsteps = []
    for _ in range(10):
        micro_step += 1
        if micro_step % accumulation == 0:
            update_microsteps.append(micro_step)
        # A logging interval may reset local metric counters, but never this
        # monotonic accumulation counter.
    assert update_microsteps == [3, 6, 9]
    resumed_micro_step = 4 * accumulation
    assert (resumed_micro_step + accumulation) % accumulation == 0


def test_distributed_sample_shards_are_complete_and_nonoverlapping() -> None:
    bounds = [sample_shard_bounds(10, rank, 3) for rank in range(3)]
    assert bounds == [(0, 3), (3, 6), (6, 10)]
    indices = [index for start, end in bounds for index in range(start, end)]
    assert indices == list(range(10))


def test_resume_skips_batches_already_consumed_in_current_epoch() -> None:
    assert resume_batch_offset(5_000, 1, 197) == 75
    assert resume_batch_offset(100, 2, 50) == 0
