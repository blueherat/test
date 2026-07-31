import json
from pathlib import Path

from experiments.raev2_stage1_compat import resolve_decoder_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = (
    ROOT / "external" / "RAEv2" / "configs" / "decoder" / "ViTXL" / "config.json"
)


def test_raev2_decoder_placeholder_is_resolved_before_validation(tmp_path) -> None:
    payload = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    assert payload["patch_size"] == "SHOULD BE RELOADED"
    config_dir = tmp_path / "decoder"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    config = resolve_decoder_config(
        config_dir,
        hidden_size=7168,
        patch_size=16,
        num_patches=256,
    )

    assert config is not None
    assert config.hidden_size == 7168
    assert config.patch_size == 16
    assert config.image_size == 256


def test_normal_decoder_config_uses_official_loader(tmp_path) -> None:
    config_dir = tmp_path / "decoder"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"model_type": "vit_mae", "patch_size": 16}),
        encoding="utf-8",
    )

    assert (
        resolve_decoder_config(
            config_dir,
            hidden_size=768,
            patch_size=16,
            num_patches=256,
        )
        is None
    )
