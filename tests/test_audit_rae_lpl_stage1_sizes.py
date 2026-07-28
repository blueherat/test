from pathlib import Path

from experiments.audit_rae_lpl_stage1_sizes import stage1_config


def test_stage1_size_config_preserves_official_assets() -> None:
    config = stage1_config(
        {
            "encoder": "encoder-id",
            "channels": 384,
            "decoder": Path("/models/decoder.pt"),
            "statistics": Path("/models/stat.pt"),
        }
    )

    assert config.params.encoder_config_path == "encoder-id"
    assert config.params.encoder_params.dinov2_path == "encoder-id"
    assert config.params.pretrained_decoder_path == "/models/decoder.pt"
    assert config.params.normalization_stat_path == "/models/stat.pt"
    assert config.params.noise_tau == 0
