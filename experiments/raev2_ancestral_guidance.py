"""Gaussian-channel ancestral guidance; no fitted time schedule.

Time is noise time: z_t=(1-t)x+t*epsilon, and 0 <= s < t <= 1.
eta=1 is the forward Markov channel's conditional variance. eta=0 is DDIM.
The prediction is shared by the clean and residual-noise terms.
"""
from __future__ import annotations

import math


def coefficients(t: float, s: float, eta: float = 1.0):
    if not 0 <= s < t <= 1 or not 0 <= eta <= 1:
        raise ValueError('require 0 <= s < t <= 1 and 0 <= eta <= 1')
    a, b = 1-t, 1-s
    conditional_variance = s*s * (1 - (a*s/(b*t))**2)
    variance = eta*eta*max(0.0, conditional_variance)
    residual = math.sqrt(max(0.0, s*s-variance)) / t
    return residual, b-residual*a, math.sqrt(variance)


def ancestral_step(state, clean, t, s, noise, *, eta=1.0):
    if state.shape != clean.shape or state.shape != noise.shape:
        raise ValueError('state, clean, and independent noise must match')
    r, q, std = coefficients(t, s, eta)
    # Euler plus a residual correction avoids cancellation when eta is small.
    euler = state - (t-s)*((state-clean)/t)
    return euler + (r-s/t)*(state-(1-t)*clean) + std*noise
