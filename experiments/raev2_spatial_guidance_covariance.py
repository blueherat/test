"""Spatial residual covariance nested around the frozen global guidance model.

The Gaussian likelihood statement concerns a fixed guided mean. It is not a
claim that calibrated covariance improves FID or recovers a true posterior.
"""
from __future__ import annotations

import torch

from experiments.raev2_directional_variance import colored_noise as global_colored_noise


PLAN = {
    'name': 'spatial_guidance_covariance', 'train': 64000, 'validation': 8000,
    'batch': 8, 'steps': 100, 'spatial_dimension': 256, 'fitted_dimension': 255,
    'spherical_head_sha256': '4b714004cd57addae8c833970574c13fd6f3c3addfb9b119a913ae0767108faa',
    'basis': 'one channel-normalized Full.float()-Base.float() direction per spatial token',
    'reference': 'frozen conditional scalar variance and frozen global directional kappa',
    'frame': 'V = U H(w)[:,1:], H is the Householder reflection mapping e0 to global coordinates w',
    'residual': 'r = V^T (X-G_native) / sqrt(m), V^T u_global=0',
    'fit': 'K = train mean(r r^T), uncentered maximum likelihood with fixed mean',
    'full_parameters': 32640, 'diagonal_control_parameters': 255,
    'sampling': 'S_global xi + V(K^(1/2)-I)V^T xi',
    'invariant': 'global direction remains an eigenvector with the original variance m*kappa',
    'unit_matrix': 'exactly recovers the existing global directional covariance sampler',
    'zero_token_rule': 'if any token gap vanishes, spatial transform is identity for that image',
    'entry': 'SPD K; validation full-minus-global and full-minus-diagonal mean NLL+2 class SE < 0',
    'validation_uncertainty': 'existing 8K held-out rows; descriptive 1000-class SE, not an untouched holdout',
    'sample_count': 5000, 'discovery_seed': 202609072, 'confirmation_seed': 202609073,
    'no_centering_shrinkage_optimizer_grid_or_time_strength_fit': True,
    'no_fid_used_in_fit': True,
}


def token_basis(full, base):
    """Implicit orthonormal columns U and the coordinates of the global gap."""
    delta = (full.float() - base.float()).double()
    norm2 = delta.square().sum(dim=1)
    active = (norm2 > 0).flatten(1).all(dim=1)
    denominator = torch.where(norm2 > 0, norm2.sqrt(), torch.ones_like(norm2))
    u = delta / denominator[:, None]
    u = u * active[:, None, None, None]
    lengths = norm2.sqrt().flatten(1)
    total = lengths.square().sum(dim=1).sqrt()
    weights = lengths / torch.where(total > 0, total, torch.ones_like(total))[:, None]
    return u, weights, active


def project(value, u):
    return (value.double() * u).sum(dim=1).flatten(1)


def reflect(coordinates, weights):
    """H(w)=I-2vv^T with v=normalize(e0-w); apply without dense Bx256x256."""
    v = -weights.clone()
    v[:, 0] += 1
    length = v.square().sum(dim=1, keepdim=True).sqrt()
    v = v / torch.where(length > 0, length, torch.ones_like(length))
    return coordinates - 2 * v * (v * coordinates).sum(dim=1, keepdim=True)


def standardized_residual(error, full, base, predicted_mse):
    """Existing covariance whitening is identity on the global-orthogonal space."""
    u, weights, active = token_basis(full, base)
    p = project(error, u) / predicted_mse.double().sqrt()[:, None]
    along = (weights * p).sum(dim=1, keepdim=True)
    r = reflect(p, weights)[:, 1:]
    return r, active, p.square().sum(dim=1), along[:, 0].square()


def spatial_update(noise, full, base, covariance_root):
    """Apply V(K^1/2-I)V^T, which annihilates the global direction."""
    u, weights, active = token_basis(full, base)
    projected = reflect(project(noise, u), weights)[:, 1:]
    dimension = projected.shape[1]
    if covariance_root.shape != (dimension, dimension):
        raise ValueError('covariance root and spatial geometry differ')
    change = covariance_root.double() - torch.eye(dimension, device=noise.device, dtype=torch.float64)
    coefficients = torch.cat((torch.zeros_like(projected[:, :1]), projected @ change.T), dim=1)
    lifted = reflect(coefficients, weights)
    update = (u * lifted.reshape(noise.shape[0], 1, *noise.shape[2:])).float()
    return update, active


def colored_noise(noise, full, base, covariance_root, kappa):
    # Orthogonality makes the two factors commute. Compute the added component
    # from original xi so rounded global coloring cannot leak into that space.
    update, active = spatial_update(noise, full, base, covariance_root)
    colored, _ = global_colored_noise(noise, full, base, kappa)
    return colored + update, active
