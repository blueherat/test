"""Paper-only building blocks for the AdvFD reproduction.

This package is intentionally independent of the official AdvFD implementation.
"""

from .core import (
    AffineCalibration,
    EMAMomentTracker,
    FrechetComponents,
    Moments,
    batch_moments,
    calibrate_features,
    frechet_from_features,
    frechet_from_moments,
    fit_calibration,
    normalized_frechet_loss,
    symmetric_matrix_inverse_sqrt,
    symmetric_matrix_sqrt,
)

__all__ = [
    "AffineCalibration",
    "EMAMomentTracker",
    "FrechetComponents",
    "Moments",
    "batch_moments",
    "calibrate_features",
    "frechet_from_features",
    "frechet_from_moments",
    "fit_calibration",
    "normalized_frechet_loss",
    "symmetric_matrix_inverse_sqrt",
    "symmetric_matrix_sqrt",
]
