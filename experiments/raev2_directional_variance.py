"""One fitted global covariance ratio along the native Full/Base difference."""
import torch

PLAN = {
    'name': 'directional_variance', 'train': 64000, 'validation': 8000, 'batch': 8, 'steps': 100,
    'spherical_head_sha256': '4b714004cd57addae8c833970574c13fd6f3c3addfb9b119a913ae0767108faa',
    'direction': 'normalize(full.float()-base.float()) in FP64; zero gap uses spherical covariance',
    'covariance': 'm(z,t,c) * (I + (kappa-1) u u^T)',
    'global_fitted_parameters': 1, 'fit': 'kappa = train mean((u dot (X-G))^2/m) among nonzero gaps',
    'entry': 'finite positive kappa; validation mean NLL change + 2 class SE < 0',
    'all100_times': True, 'reuse_original64k8k_real_noise_times_and_frozen_spherical_head': True,
    'native_features_and_total_mse_bitwise_match_previous_extraction': True,
    'no_optimizer_grid_regularizer_retraining_or_time_coefficients': True,
    'samples': [1000, 5000], 'seeds': [202609071, 202609072],
    'both_scales_regardless_1k_fid': True, 'maximum_research_round': 8,
    'no_further_covariance_directions_or_scalars_after_this_candidate': True,
}


def unit_disagreement(full, base):
    delta = (full.float()-base.float()).double()
    norm2 = delta.square().flatten(1).sum(1)
    active = norm2 > 0
    denominator = torch.where(active, norm2.sqrt(), torch.ones_like(norm2))
    return delta/denominator[:, None, None, None], active, norm2


def fixed_mse(head, features, mse0):
    h = ((features.double()-head.mean)/head.scale)@head.weight+head.bias
    return float(mse0)*h.exp()


def colored_noise(noise, full, base, kappa):
    assert kappa > 0
    u, active, _ = unit_disagreement(full, base)
    component = (u*noise.double()).flatten(1).sum(1)
    update = ((kappa**.5-1)*component[:, None, None, None]*u).float()
    return noise+update, active
