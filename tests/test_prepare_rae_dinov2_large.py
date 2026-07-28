from experiments.prepare_rae_dinov2_large import convert_diffusers_decoder_key


def test_decoder_key_conversion_matches_rae_main_attention_names() -> None:
    assert (
        convert_diffusers_decoder_key(
            "decoder.decoder_layers.3.attention.to_q.weight"
        )
        == "decoder_layers.3.attention.attention.query.weight"
    )
    assert (
        convert_diffusers_decoder_key(
            "decoder.decoder_layers.3.attention.to_k.bias"
        )
        == "decoder_layers.3.attention.attention.key.bias"
    )
    assert (
        convert_diffusers_decoder_key(
            "decoder.decoder_layers.3.attention.to_v.weight"
        )
        == "decoder_layers.3.attention.attention.value.weight"
    )
    assert (
        convert_diffusers_decoder_key(
            "decoder.decoder_layers.3.attention.to_out.0.bias"
        )
        == "decoder_layers.3.attention.output.dense.bias"
    )


def test_decoder_key_conversion_keeps_non_attention_names() -> None:
    assert (
        convert_diffusers_decoder_key("decoder.decoder_embed.weight")
        == "decoder_embed.weight"
    )
    assert convert_diffusers_decoder_key("encoder.embeddings.cls_token") is None
