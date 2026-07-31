from experiments.summarize_raev2_gradient_audits import recommended_lpl_weight


def test_recommended_lpl_weight_matches_target_gradient_ratio() -> None:
    weight = recommended_lpl_weight(
        flow_gradient_norm=0.5,
        probe_gradient_norm=0.25,
        probe_weight=1e-4,
        target_ratio=0.2,
    )
    assert weight == 4e-5
    weighted_lpl_gradient = (0.25 / 1e-4) * weight
    assert abs(weighted_lpl_gradient / 0.5 - 0.2) < 1e-12
