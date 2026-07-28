import torch

from experiments.rae_encoder_decoder_atlas import (
    CKAMoments,
    RidgeMoments,
    compare_atlases,
    cross_layer_cka,
    fit_ridge_map,
    linear_probe_scores,
    paired_and_mismatched_cka,
    summarize_atlas,
    token_gram_vectors,
)


def test_token_cka_is_invariant_to_orthogonal_channel_basis():
    generator = torch.Generator().manual_seed(8)
    state = torch.randn((4, 9, 6), generator=generator)
    basis, _ = torch.linalg.qr(torch.randn((6, 6), generator=generator))
    transformed = state @ basis
    cka = cross_layer_cka([state], [transformed])
    torch.testing.assert_close(cka[:, 0, 0], torch.ones(4), atol=1e-6, rtol=1e-6)


def test_image_mismatch_control_detects_sample_specific_geometry():
    generator = torch.Generator().manual_seed(9)
    states = []
    for sample in range(4):
        value = torch.randn((9, 6), generator=generator)
        value[:, sample] += torch.linspace(-4.0, 4.0, 9)
        states.append(value)
    encoder = torch.stack(states)
    decoder = encoder @ torch.linalg.qr(torch.randn((6, 6), generator=generator))[0]
    paired, mismatched = paired_and_mismatched_cka([encoder], [decoder])
    assert mismatched is not None
    assert float(paired.mean()) > float(mismatched.mean()) + 0.1


def test_moments_match_direct_mean_and_std():
    values = torch.tensor(
        [
            [[0.2, 0.4], [0.3, 0.5]],
            [[0.6, 0.8], [0.7, 0.9]],
        ]
    )
    control = values - 0.1
    moments = CKAMoments.zeros(2, 2, device="cpu")
    moments.update(values, control)
    summary = moments.summary()
    torch.testing.assert_close(summary["paired_mean"], values.double().mean(dim=0))
    torch.testing.assert_close(
        summary["paired_std"], values.double().std(dim=0, unbiased=False)
    )
    torch.testing.assert_close(
        summary["excess_mean"],
        torch.full((2, 2), 0.1, dtype=torch.float64),
    )


def test_reverse_hierarchy_summary_and_comparison():
    atlas = torch.zeros((4, 7))
    mapping = [3, 3, 2, 2, 1, 1, 0]
    for decoder, encoder in enumerate(mapping):
        atlas[encoder, decoder] = 1.0
    summary = summarize_atlas(atlas)
    assert summary["reverse_spearman_argmax"] > 0.9
    assert summary["reverse_spearman_soft"] > 0.9
    assert summary["nonincreasing_pair_fraction"] == 1.0
    comparison = compare_atlases(atlas, atlas.clone())
    assert comparison["pearson"] > 0.999
    assert comparison["rms_distance"] == 0.0
    assert comparison["exact_mapping_rate"] == 1.0


def test_constant_token_state_remains_finite():
    vectors = token_gram_vectors([torch.ones((2, 4, 3))])
    assert torch.isfinite(vectors).all()
    torch.testing.assert_close(vectors, torch.zeros_like(vectors))


def test_ridge_map_generalizes_known_channel_transform():
    generator = torch.Generator().manual_seed(11)
    transform = torch.randn((7, 5), generator=generator)
    train_decoder = torch.randn((8, 9, 7), generator=generator)
    train_encoder = train_decoder @ transform
    moments = RidgeMoments.zeros(7, 5, device="cpu")
    moments.update(train_decoder, train_encoder)
    mapping, ridge_scale = fit_ridge_map(moments, ridge=1e-7)
    assert ridge_scale > 0.0

    test_decoder = torch.randn((4, 9, 7), generator=generator)
    test_encoder = test_decoder @ transform
    error, cosine = linear_probe_scores(test_decoder, test_encoder, mapping)
    assert float(error.max()) < 1e-4
    assert float(cosine.min()) > 0.9999
