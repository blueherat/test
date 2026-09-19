"""Distribution targets; pure tensor functions with no model or device side effects."""
import torch


def smoothing_pair(clean, time, noise, tau, smooth):
    """Rao--Blackwellized FM pair for Y=X+tau*xi; smooth is per image.

    The unsmoothed branch uses bitwise-identical native arithmetic. Labels identifying
    the branch are never supplied to the predictor. Extra noise is not clipped.
    """
    shape = (-1,) + (1,) * (clean.ndim - 1)
    a = time.reshape(shape)
    b = 1 - a
    z = a * clean + b * noise
    target = clean - noise
    if bool(smooth.any()):
        variance = b.square() + a.square() * tau ** 2
        scale = variance.sqrt()
        smoothed_z = a * clean + scale * noise
        smoothed_target = clean + (a * tau ** 2 - b) / scale * noise
        mask = smooth.reshape(shape)
        z = torch.where(mask, smoothed_z, z)
        target = torch.where(mask, smoothed_target, target)
    return z, target


def excess_weights(prob_generated, strength=1.0):
    """Bounded endpoint weights; the classifier is used only offline."""
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    p = prob_generated.clamp(0, 1)
    excess = torch.where(p > .5, (2 * p - 1) / p.clamp_min(.5), torch.zeros_like(p))
    return 1 + strength * excess


def weighted_mse(prediction, target, weights):
    """Weights are normalized over the frozen class bank, never per minibatch."""
    per_image = (prediction.float() - target.float()).square().flatten(1).mean(1)
    return (weights.detach() * per_image).mean()
