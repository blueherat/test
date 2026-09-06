"""Fixed, additive class guidance about the unchanged native IG anchor.

The single 0.15 strength is the historical, unevaluated preset, not a fitted
coefficient. Orthogonalization solves a per-image constrained quadratic.
"""
from experiments.raev2_semantic_quality_guidance import semantic_orthogonal_residual

STRENGTH = .15


def semantic_correction(full, base, null_full, *, orthogonal=False):
    conditional_difference = full.float() - null_full.float()
    if orthogonal:
        conditional_difference = semantic_orthogonal_residual(
            full.float() - base.float(), conditional_difference)
    return STRENGTH * conditional_difference
