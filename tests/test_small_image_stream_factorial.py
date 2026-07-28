import math

import pandas as pd
import torch

from experiments.mnist_spectral_rollout_toy import (
    MNISTToyConfig,
    train_paired_velocity_fields,
)
from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from experiments.small_image_stream_factorial import (
    stream_factorial_effects,
    train_paired_mixed_streams,
)


def test_uncrossed_mixed_stream_exactly_replays_original_training():
    clean = torch.randn(
        (8, 1, 4, 4), generator=torch.Generator().manual_seed(1)
    )
    analyzer = DCTDirectionLoss(4, [1.0, 0.8], gamma=0.5)
    config = MNISTToyConfig(
        batch_size=4,
        steps=3,
        width=4,
        depth=1,
        learning_rate=2e-4,
        device="cpu",
        save=False,
    )
    original, _ = train_paired_velocity_fields(
        clean,
        config,
        analyzer,
        init_seed=4,
        stream_seed=3,
    )
    replay = train_paired_mixed_streams(
        clean,
        config,
        analyzer,
        init_seed=4,
        batch_seed=3,
        bridge_seed=3,
    )
    for variant in original:
        for name, value in original[variant].state_dict().items():
            assert torch.equal(value, replay[variant].state_dict()[name])


def test_stream_factorial_effects_recover_known_contrasts():
    rows = []
    for batch_seed in (3, 4):
        for bridge_seed in (3, 4):
            batch_code = -1 if batch_seed == 3 else 1
            bridge_code = -1 if bridge_seed == 3 else 1
            log_ratio = 0.3 * batch_code - 0.2 * bridge_code
            rows.append(
                {
                    "batch_seed": batch_seed,
                    "bridge_seed": bridge_seed,
                    "metric": "feature_fid",
                    "ratio_mean": math.exp(log_ratio),
                }
            )
    effects = stream_factorial_effects(pd.DataFrame(rows)).set_index("term")
    assert math.isclose(effects.loc["batch", "log_ratio_effect"], 0.6)
    assert math.isclose(effects.loc["bridge", "log_ratio_effect"], -0.4)
    assert abs(effects.loc["batch:bridge", "log_ratio_effect"]) < 1e-12
