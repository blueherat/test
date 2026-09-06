"""Noise-paired density-ratio loss with a finite high-noise expansion.

The classifier logit is d(z,t,c) = (1-t) f(z,t,c). At each positive signal
level the loss has the usual Bayes density-ratio minimizer. Sharing the noise
in each real/fake pair cancels the leading stochastic gradient at t -> 1.
"""
import math
import torch


def paired_scaled_logistic(real_f, fake_f, signal):
    """Per-pair centered BCE / signal**2, evaluated stably in FP64.

    Inputs must use the same label, noise and signal for each pair. The
    population marginals remain the ordinary real/fake noising marginals.
    Training uses positive signal; the endpoint is handled by its limit.
    """
    p, q, a = real_f.double(), fake_f.double(), signal.double()
    lp = torch.logaddexp(a * p / 2, -a * p / 2) - math.log(2)
    lq = torch.logaddexp(a * q / 2, -a * q / 2) - math.log(2)
    return (q - p) / (4 * a) + (lp + lq) / (2 * a.square())
