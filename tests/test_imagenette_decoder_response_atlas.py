import pandas as pd
import torch

from experiments.imagenette_decoder_response_atlas import (
    FixedFeatureProjector,
    decoder_forward_trace,
    distribution_core_metrics,
    rollout_response_features,
)
from experiments.imagenette_noise_responsibility import ImagenetteConditionalUNet
from experiments.summarize_imagenette_decoder_response_atlas import (
    CAPACITIES,
    LAYER_ORDER,
    SEEDS,
    aggregate_times,
    evaluate_gates,
    heldout_predictions,
    prediction_summary,
)


def test_decoder_trace_matches_original_forward_exactly():
    torch.manual_seed(11)
    model = ImagenetteConditionalUNet(latent_dim=7, width=8).eval()
    state = torch.randn(3, 3, 16, 16)
    time = torch.tensor([0.2, 0.5, 0.8])
    condition = torch.randn(3, 7)

    expected = model(state, time, condition)
    observed, trace = decoder_forward_trace(model, state, time, condition)

    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    assert tuple(trace) == LAYER_ORDER
    assert trace["velocity"] is observed


def test_zero_condition_has_exactly_zero_condition_contribution():
    torch.manual_seed(13)
    model = ImagenetteConditionalUNet(latent_dim=5, width=8).eval()
    state = torch.randn(2, 3, 16, 16)
    time = torch.tensor([0.3, 0.7])
    condition = torch.zeros(2, 5)

    _velocity, trace = decoder_forward_trace(model, state, time, condition)
    _null_velocity, null_trace = decoder_forward_trace(model, state, time, condition)

    for layer in LAYER_ORDER:
        torch.testing.assert_close(trace[layer] - null_trace[layer], torch.zeros_like(trace[layer]))


def test_fixed_projection_is_deterministic_and_shape_stable():
    value = torch.randn(4, 17, 8, 8, generator=torch.Generator().manual_seed(17))
    first = FixedFeatureProjector(output_dim=11, seed=23)("middle", value)
    second = FixedFeatureProjector(output_dim=11, seed=23)("middle", value)
    changed = FixedFeatureProjector(output_dim=11, seed=29)("middle", value)

    assert first.shape == (4, 11)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, changed)


def test_distribution_metrics_have_identity_floor_and_detect_covariance_change():
    generator = torch.Generator().manual_seed(31)
    real = torch.randn(256, 12, generator=generator)
    identical = distribution_core_metrics(real, real.clone(), seed=37)
    stretched = real.clone()
    stretched[:, 0] *= 3.0
    changed = distribution_core_metrics(real, stretched, seed=37)

    assert identical["mean_relative_error"] < 1e-12
    assert identical["covariance_relative_error"] < 1e-12
    assert identical["normalized_frechet"] < 1e-12
    assert identical["normalized_swd"] < 1e-12
    assert changed["covariance_relative_error"] > 0.5
    assert changed["normalized_frechet"] > 0.05
    assert changed["normalized_swd"] > 0.05


def test_rollout_response_feature_grid_is_complete():
    torch.manual_seed(41)
    model = ImagenetteConditionalUNet(latent_dim=6, width=8).eval()
    conditions = torch.randn(4, 6)
    noise = torch.randn(4, 3, 16, 16)

    features = rollout_response_features(
        model,
        conditions,
        noise,
        steps=4,
        probe_times=(0.75, 0.5, 0.25),
        batch_size=2,
        projection_dim=9,
        projection_seed=43,
    )

    assert len(features) == 2 * 3 * len(LAYER_ORDER)
    assert all(value.shape[0] == 4 for value in features.values())
    assert all(value.shape[1] <= 9 for value in features.values())
    assert all(torch.isfinite(value).all() for value in features.values())


def _synthetic_atlas_frames():
    runs = []
    response = []
    paired = []
    for latent_dim in CAPACITIES:
        for seed in SEEDS:
            gap = 0.08 * latent_dim + 0.1 * seed
            common = {
                "latent_dim": latent_dim,
                "frozen_seed": seed,
                "modeling_gap": gap,
                "count": 256,
                "paired_count": 128,
                "pixel_steps": 50,
                "projection_dim": 128,
                "projection_seed": 48_271,
                "frozen_decoder_matches_formal": True,
                "run": f"d{latent_dim}_seed{seed}",
            }
            runs.append(common)
            for representation in ("raw", "condition"):
                for time in (0.9, 0.5, 0.1):
                    for layer in LAYER_ORDER:
                        response.append(
                            {
                                **common,
                                "representation": representation,
                                "time": time,
                                "layer": layer,
                                "feature_dim": 128,
                                "mean_relative_error": 0.1 * gap,
                                "covariance_relative_error": 0.2 * gap,
                                "normalized_frechet": gap,
                                "normalized_swd": 0.3 * gap,
                                "linear_c2st_auc": 0.5 + 0.01 * gap,
                                "real_effective_rank": 10.0,
                                "generated_effective_rank": 9.0,
                                "real_real_mean_relative_error": 0.1,
                                "real_real_covariance_relative_error": 0.1,
                                "real_real_normalized_frechet": gap / 2.0,
                                "real_real_normalized_swd": 0.1,
                                "real_real_real_effective_rank": 10.0,
                                "real_real_generated_effective_rank": 10.0,
                                "real_real_linear_c2st_auc": 0.52,
                                "frechet_over_real_floor": 2.0,
                                "swd_over_real_floor": 2.0,
                            }
                        )
            for time in (0.9, 0.5, 0.1):
                paired.append(
                    {
                        **common,
                        "time": time,
                        "matched_velocity_mse": 1.0,
                        "shuffled_velocity_mse": 1.2,
                        "null_velocity_mse": 1.3,
                        "shuffled_over_matched": 1.2,
                        "null_over_matched": 1.3,
                    }
                )
    return pd.DataFrame(runs), pd.DataFrame(response), pd.DataFrame(paired)


def test_preregistered_response_atlas_gates_accept_constructed_signal():
    runs, response, paired = _synthetic_atlas_frames()
    aggregated = aggregate_times(response)
    prediction, _heldout = prediction_summary(aggregated)
    gates, primary = evaluate_gates(runs, response, paired, aggregated, prediction)

    assert gates["implementation_audit"]
    assert gates["primary_prediction_gate"]
    assert gates["floor_gate"]
    assert gates["shuffled_gate"]
    assert gates["decoder_response_target_supported"]
    assert primary.both_pass.all()


def test_heldout_prediction_handles_a_single_group_without_fitting_empty_data():
    table = pd.DataFrame(
        {
            "latent_dim": [16, 64],
            "frozen_seed": [0, 0],
            "modeling_gap": [2.0, 8.0],
            "normalized_frechet": [0.1, 0.2],
        }
    )
    frame, correlation, rmse = heldout_predictions(
        table,
        predictor="normalized_frechet",
        group="frozen_seed",
    )

    assert frame.empty
    assert correlation != correlation
    assert rmse != rmse
