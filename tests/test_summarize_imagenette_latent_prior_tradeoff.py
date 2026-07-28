import pandas as pd

from experiments.summarize_imagenette_latent_prior_tradeoff import (
    evaluate_gates,
    prediction_table,
)


def _rows(pattern: str) -> pd.DataFrame:
    rows = []
    for seed in range(5):
        for latent_dim in (16, 64, 256):
            oracle = {16: 123.0, 64: 119.0, 256: 116.0}[latent_dim] + 0.1 * seed
            if pattern == "tradeoff":
                end_to_end = {16: 145.0, 64: 134.0, 256: 142.0}[latent_dim] + 0.1 * seed
                modeling_gap = {16: 10.0, 64: 14.0, 256: 20.0}[latent_dim] + 0.1 * seed
            else:
                end_to_end = {16: 145.0, 64: 135.0, 256: 125.0}[latent_dim] + 0.1 * seed
                modeling_gap = {16: 10.0, 64: 10.3, 256: 10.5}[latent_dim] + 0.1 * seed
            rows.append(
                {
                    "latent_dim": latent_dim,
                    "frozen_seed": seed,
                    "oracle_feature_fid": oracle,
                    "empirical_feature_fid": end_to_end - modeling_gap,
                    "end_to_end_feature_fid": end_to_end,
                    "gaussian_feature_fid": end_to_end + 10.0,
                    "modeling_gap": modeling_gap,
                    "total_prior_gap": end_to_end - oracle,
                    "source_final_validation_velocity_mse": 0.2 - latent_dim / 10_000,
                    "real_latent_effective_rank": {16: 8.0, 64: 15.0, 256: 22.0}[latent_dim]
                    + 0.1 * seed,
                    "heldout_prior_flow_mse": 0.5 + 0.01 * modeling_gap,
                    "frozen_hashes_unchanged": True,
                    "orthogonal_roundtrip_max_abs": 1e-7,
                    "source_oracle_fid_1024_abs_diff": 0.0,
                    "train_path_sha256": "train-paths",
                    "val_path_sha256": "val-paths",
                    "prior_initial_sha256": f"init-{seed}",
                    "prior_parameters": 100,
                    "stream_indices_first_32_sha256": f"indices-{seed}",
                    "stream_base_noise_first_32_sha256": f"noise-{seed}",
                    "stream_time_first_32_sha256": f"time-{seed}",
                }
            )
    return pd.DataFrame(rows)


def test_positive_pattern_passes_directional_tradeoff_gates():
    table = _rows("tradeoff")
    prediction, _ = prediction_table(table)
    gates, paired = evaluate_gates(table, prediction)
    assert gates["decoder_benefit"]
    assert gates["prior_difficulty"]
    assert gates["middle_optimum"]
    assert gates["trained_prior_valid"]
    assert paired.middle_is_best.all()
    assert not gates["opposite_candidate_requires_independent_audit"]


def test_monotonic_pattern_is_flagged_as_opposite_candidate():
    table = _rows("opposite")
    prediction, _ = prediction_table(table)
    gates, paired = evaluate_gates(table, prediction)
    assert gates["decoder_benefit"]
    assert not gates["prior_difficulty"]
    assert not gates["middle_optimum"]
    assert paired.monotonic_larger_is_better.all()
    assert gates["opposite_candidate_requires_independent_audit"]
