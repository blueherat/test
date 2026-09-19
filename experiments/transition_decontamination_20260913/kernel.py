"""Positive part of N(0,1) - kappa N(d,1), including its exact scalar moments.

The inverse CDF table is a numerical implementation, not a learned network.
All table settings are selected by numerical error before any image evaluation.
"""
from pathlib import Path
import math
import numpy as np
from scipy.special import ndtr
import torch
import torch.nn.functional as F

ALPHAS = (0.8 * 6 / 7, 0.8, 1.25)
DMAX, ZMAX, ND, NZ = 16., 7., 1025, 1025


def moments_np(d, alpha):
    d = np.asarray(d, dtype=np.float64)
    k = alpha / (1 + alpha)
    cut = d / 2 - math.log(k) / np.maximum(d, 1e-12)
    a = ndtr(cut) - k * ndtr(cut - d)
    phi = np.exp(-.5 * cut**2) / math.sqrt(2 * math.pi)
    mean = -k * d * ndtr(cut - d) / a
    variance = 1 - k * d**2 * ndtr(cut - d) / a + d * phi / a - mean**2
    negative = k * ndtr(d - cut) - ndtr(-cut)
    return mean, variance, negative, a, cut


def inverse_np(d, normal, alpha):
    d, normal = np.broadcast_arrays(np.asarray(d, dtype=np.float64), np.asarray(normal, dtype=np.float64))
    k = alpha / (1 + alpha)
    _, _, _, mass, cut = moments_np(d, alpha)
    target = ndtr(normal) * mass
    survival_target = ndtr(-normal) * mass
    lo, hi = np.full(d.shape, -16.), np.minimum(cut, 16.)
    for _ in range(56):
        mid = (lo + hi) / 2
        def interval(left, right):
            return np.where(left > 0, ndtr(-left) - ndtr(-right), ndtr(right) - ndtr(left))
        survival = interval(mid, cut) - k * interval(mid - d, cut - d)
        lower = np.where(normal > 0, survival > survival_target,
                         ndtr(mid) - k * ndtr(mid - d) < target)
        lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
    return np.where(d == 0, normal, (lo + hi) / 2)


def build(path):
    path = Path(path)
    if path.exists():
        return
    ds = np.linspace(0, math.sqrt(DMAX), ND)[:, None] ** 2
    zs = np.linspace(-ZMAX, ZMAX, NZ)[None, :]
    shifts = np.stack([(inverse_np(ds, zs, a) - zs).astype(np.float32) for a in ALPHAS])
    assert np.isfinite(shifts).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, shifts=shifts, alphas=np.array(ALPHAS), dmax=DMAX, zmax=ZMAX)


class Kernel:
    def __init__(self, path, device='cuda'):
        with np.load(path) as data:
            np.testing.assert_array_equal(data['alphas'], ALPHAS)
            self.tables = torch.from_numpy(data['shifts'].copy()).to(device)[:, None]

    def shift(self, d, normal, alpha):
        index = min(range(len(ALPHAS)), key=lambda j: abs(ALPHAS[j] - alpha))
        assert abs(ALPHAS[index] - alpha) < 1e-8
        grid = torch.stack((normal / ZMAX, 2 * d.clamp_min(0).sqrt() / math.sqrt(DMAX) - 1), -1).reshape(1, -1, 1, 2)
        value = F.grid_sample(self.tables[index:index + 1], grid.float(), mode='bilinear',
                              padding_mode='border', align_corners=True).flatten()
        return torch.where(d > DMAX, torch.zeros_like(value), value)

    @staticmethod
    def moments(d, alpha):
        dd = d.double()
        k = alpha / (1 + alpha)
        cut = dd / 2 - math.log(k) / dd.clamp_min(1e-12)
        cdf = torch.special.ndtr
        a = cdf(cut) - k * cdf(cut - dd)
        phi = torch.exp(-.5 * cut.square()) / math.sqrt(2 * math.pi)
        mean = -k * dd * cdf(cut - dd) / a
        var = 1 - k * dd.square() * cdf(cut - dd) / a + dd * phi / a - mean.square()
        return mean.float(), var.clamp_min(1e-12).float()

    def update(self, mean_s, mean_w, sigma, epsilon, alpha, method, delta=None):
        if alpha == 0:
            return mean_s + sigma * epsilon
        if method == 'sde':
            return mean_s + alpha * (mean_s - mean_w) + sigma * epsilon
        delta = mean_w - mean_s if delta is None else delta
        norm = delta.flatten(1).norm(dim=1)
        view = (-1,) + (1,) * (delta.ndim - 1)
        direction = delta / norm.clamp_min(1e-20).reshape(view)
        normal = (direction * epsilon).flatten(1).sum(1)
        d = norm / sigma
        if method == 'positive':
            shift = self.shift(d, normal, alpha)
        else:
            mean, variance = self.moments(d, alpha)
            if method == 'mean_gaussian':
                shift = mean
            elif method == 'moment_gaussian':
                shift = mean + (variance.sqrt() - 1) * normal
            else:
                raise ValueError(method)
        return mean_s + sigma * (epsilon + direction * shift.reshape(view))


def numerical_checks(path):
    from scipy.integrate import quad
    rng = np.random.default_rng(2026121417)
    lut = Kernel(path, 'cpu')
    errors, rows = [], []
    for alpha in ALPHAS:
        d = rng.uniform(0, DMAX, 6000)
        z = rng.uniform(-6, 6, 6000)
        reference = inverse_np(d, z, alpha) - z
        result = lut.shift(torch.tensor(d, dtype=torch.float32), torch.tensor(z, dtype=torch.float32), alpha).numpy()
        errors.append(float(np.max(np.abs(reference - result))))
        for separation in (.01, .1, .5, 1., 2., 4., 8.):
            mean, variance, negative, a, cut = moments_np(separation, alpha)
            k = alpha / (1 + alpha)
            def pdf(x):
                return max(math.exp(-x*x/2) - k * math.exp(-(x-separation)**2/2), 0) / math.sqrt(2*math.pi) / a
            moments = [quad(lambda x, j=j: x**j * pdf(x), -14, min(float(cut), 14), epsabs=1e-10)[0] for j in range(3)]
            assert abs(moments[0] - 1) < 1e-8
            assert abs(moments[1] - mean) < 1e-8
            assert abs(moments[2] - moments[1]**2 - variance) < 1e-8
            assert variance > 0 and a >= 1-k-1e-12
            # q is a nearest probability measure to the signed normalized inverse in TV.
            tv_distance = .5 * quad(lambda x: abs(pdf(x) - (math.exp(-x*x/2)-k*math.exp(-(x-separation)**2/2))
                / math.sqrt(2*math.pi)/(1-k)), -16, 20, points=[min(float(cut), 19)], epsabs=1e-8)[0] if cut < 19 else 0.
            assert abs(tv_distance - negative/(1-k)) < 1e-6
            rows.append(dict(alpha=alpha, separation=separation, mean=float(mean), variance=float(variance),
                             negative_mass=float(negative), normalization=float(a)))
    assert max(errors) < .002, errors
    # Distribution checks compare the implementation with analytical moments, not only its own code path.
    g = torch.Generator().manual_seed(2026121421)
    eps = torch.randn((100000, 3), generator=g)
    ms, mw = torch.zeros_like(eps), torch.zeros_like(eps)
    mw[:, 0] = 2.
    sampled = lut.update(ms, mw, 1., eps, .8, 'positive')
    mean, variance, *_ = moments_np(2., .8)
    assert abs(float(sampled[:, 0].mean()) - mean) < .01
    assert abs(float(sampled[:, 0].var()) - variance) < .02
    assert torch.equal(sampled[:, 1:], eps[:, 1:])
    assert torch.equal(lut.update(ms, mw, 1., eps, 0., 'positive'), ms + eps)
    return dict(passed=True, lookup_max_abs_shift_errors=errors, inverse_table_shape=list(lut.tables.shape),
                quadrature_cases=rows, distribution_draws=100000, perpendicular_noise_exact=True,
                zero_guidance_exact=True, gaussian_projection_tail_bound=2*float(ndtr(-ZMAX)),
                finite_step_method=True, endpoint_density_recovery_proven=False)
