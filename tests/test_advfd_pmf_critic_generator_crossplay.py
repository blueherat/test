import math

import torch

from experiments.advfd_cleanroom.audit_pmf_critic_generator_crossplay import (
    calibrate_crossplay_rows,
    interaction_summary,
    linear_cka,
)


def test_linear_cka_is_invariant_to_orthogonal_feature_rotation() -> None:
    torch.manual_seed(3)
    features = torch.randn(64, 9)
    rotation, _ = torch.linalg.qr(torch.randn(9, 9))
    assert math.isclose(
        linear_cka(features, features @ rotation, 64),
        1.0,
        abs_tol=1e-10,
    )


def test_interaction_fraction_detects_pair_specific_matrix() -> None:
    additive_rows = [
        {"critic": critic, "generator": generator, "full_fd": row + column}
        for critic, row in (("c0", 1.0), ("c1", 2.0))
        for generator, column in (("g0", 3.0), ("g1", 5.0))
    ]
    pair_specific_rows = [
        {"critic": "c0", "generator": "g0", "full_fd": 2.0},
        {"critic": "c0", "generator": "g1", "full_fd": 0.0},
        {"critic": "c1", "generator": "g0", "full_fd": 0.0},
        {"critic": "c1", "generator": "g1", "full_fd": 2.0},
    ]

    assert interaction_summary(additive_rows, "full_fd")[
        "interaction_frobenius_fraction"
    ] < 1e-12
    assert interaction_summary(pair_specific_rows, "full_fd")[
        "interaction_frobenius_fraction"
    ] > 0.99


def test_anchor_calibration_removes_critic_row_scale() -> None:
    rows = [
        {"critic": critic, "generator": generator, **values}
        for critic, scale in (("c0", 1.0), ("c1", 1e8))
        for generator, values in (
            (
                "static",
                {
                    "mean_fd": 2.0 * scale,
                    "covariance_fd": 3.0 * scale,
                    "full_fd": 5.0 * scale,
                },
            ),
            (
                "candidate",
                {
                    "mean_fd": 1.0 * scale,
                    "covariance_fd": 1.5 * scale,
                    "full_fd": 2.5 * scale,
                },
            ),
        )
    ]
    null = {
        critic: {
            "mean_fd": 0.2 * scale,
            "covariance_fd": 0.3 * scale,
            "full_fd": 0.5 * scale,
        }
        for critic, scale in (("c0", 1.0), ("c1", 1e8))
    }

    calibrate_crossplay_rows(
        rows,
        anchor_generator="static",
        real_null_by_critic=null,
    )

    candidates = [row for row in rows if row["generator"] == "candidate"]
    for key in (
        "mean_fd_over_anchor",
        "covariance_fd_over_anchor",
        "full_fd_over_anchor",
        "mean_fd_over_real_null",
        "covariance_fd_over_real_null",
        "full_fd_over_real_null",
    ):
        assert math.isclose(candidates[0][key], candidates[1][key])
