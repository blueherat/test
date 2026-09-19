"""Direct vector supervision from source-centered endpoint denoising pairs.

source=1 denotes real endpoints, source=0 generated endpoints. Both sources
must use the same forward corruption. Nuisances are trained on other endpoint
folds and detached here; their fitting pipeline is deliberately not implied.
"""
import torch


def source_residual(source, probability_real, reference):
    assert source.ndim == probability_real.ndim == 1
    assert len(source) == len(probability_real) == len(reference)
    assert torch.all((source == 0) | (source == 1))
    assert torch.isfinite(probability_real).all()
    assert torch.all((probability_real >= 0) & (probability_real <= 1))
    shape = (len(source),) + (1,) * (reference.ndim - 1)
    return (source.detach() - probability_real.detach()).reshape(shape)


def contrast_residual(correction, velocity_target, source, probability_real, baseline):
    """Residual whose square identifies v_real-v_generated with oracle eta.

    Any baseline measurable from the noisy input is unbiased at oracle eta.
    Joint first-order robustness requires a consistent mixture-mean baseline.
    """
    assert correction.shape == velocity_target.shape == baseline.shape
    centered = source_residual(source, probability_real, correction)
    return velocity_target.detach() - baseline.detach() - centered * correction


def contrast_loss(correction, velocity_target, source, probability_real, baseline):
    residual = contrast_residual(correction, velocity_target, source, probability_real, baseline)
    return residual.square().flatten(1).mean(1).mean()


def covariance_target(velocity_target, source, probability_real, mixture_mean):
    """Targets eta*(1-eta)*(v_real-v_generated), not the unscaled contrast.

    Its conditional bias is a product of the two nuisance errors. No inverse
    probability, classifier input gradient, or inference-time nuisance is used.
    """
    assert velocity_target.shape == mixture_mean.shape
    centered = source_residual(source, probability_real, velocity_target)
    return centered * (velocity_target.detach() - mixture_mean.detach())
