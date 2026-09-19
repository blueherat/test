"""Deterministic coupling-covariance and native-arithmetic checks; CPU only."""
from __future__ import annotations
import json
import sys
import types
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT'))
from experiments.official_sit_sde_pfr_control_core_20260909 import (
    counts, grid, bridge_draw_count, couple_normals, make_noises, sample_sde)
from samplers import euler_maruyama_ig_sampler


def covariance_check(source_steps, target_steps):
    a, b = grid(source_steps), grid(target_steps)
    k = bridge_draw_count(a, b)
    total = source_steps-1+k
    basis = torch.eye(total, dtype=torch.float64)
    base = basis[:source_steps-1]
    draws = iter(basis[source_steps-1:])
    refined = couple_normals(base, a, b, lambda: next(draws))
    assert next(draws, None) is None
    err = (refined@refined.T-torch.eye(target_steps-1, dtype=torch.float64)).abs().max().item()
    endpoint_error = ((base*(a[:-1]-a[1:]).sqrt()[:, None]).sum(0) -
                      (refined*(b[:-1]-b[1:]).sqrt()[:, None]).sum(0)).abs().max().item()
    assert err < 1e-11 and endpoint_error < 1e-12
    return dict(source_steps=source_steps, target_steps=target_steps,
                bridge_draws=k, covariance_identity_max_error=err, endpoint_max_error=endpoint_error)


class Dummy:
    def __call__(self, x, t, y):
        label = (y % 7).float().reshape(-1, 1, 1, 1)*.001
        full = .2*x + t.reshape(-1, 1, 1, 1)*.01 + label
        return full, full*.9+.005, None
    def future_base(self, x, t, y):
        return self(x, t, y)[1]


class NoiseTorch:
    """Used only to compare the unchanged author code with explicit test noise."""
    def __init__(self, noises):
        self.noises = iter(noises)
    def __getattr__(self, name):
        return getattr(torch, name)
    def randn_like(self, x):
        value = next(self.noises)
        assert value.shape == x.shape and value.dtype == x.dtype and value.device == x.device
        return value


def main():
    torch.set_num_threads(4)
    covariance = [covariance_check(a, b) for a, b in ((250,281), (7,41), (11,11), (41,7))]
    torch.manual_seed(91234)
    initial = torch.randn(4, 2, 3, 3)
    labels = torch.arange(4)
    base, refined, _ = make_noises(initial.double(), 71234, 81234)
    torch.manual_seed(71234)
    assert torch.equal(base, torch.stack([torch.randn_like(initial.double()) for _ in range(249)]))
    records = []
    for steps, normals in ((250, base), (281, refined)):
        proxy = NoiseTorch(normals)
        original = euler_maruyama_ig_sampler
        patched_globals = dict(original.__globals__, torch=proxy)
        reference = types.FunctionType(original.__code__, patched_globals, original.__name__,
                                       original.__defaults__, original.__closure__)
        expected = reference(Dummy(), initial, labels, num_steps=steps, cfg_scale=1.35,
                             sg_scale=1.4, guidance_high=.7)
        assert next(proxy.noises, None) is None
        actual = sample_sde(Dummy(), initial, labels, normals, steps=steps, pfr=False)
        zero = sample_sde(Dummy(), initial, labels, normals, steps=steps, pfr=True, response_scale=0.)
        assert torch.equal(expected, actual) and torch.equal(expected, zero)
        records.append(dict(steps=steps, copied_sampler_exact=True, zero_response_exact=True))
    a, b = counts(250, True), counts(281, False)
    assert a['full_per_image']==422 and a['prefix_per_image']==182
    assert b['full_per_image']==474 and b['prefix_per_image']==0
    assert a['block_evaluations_per_image']==b['block_evaluations_per_image']==13272
    print(json.dumps(dict(passed=True, scope='CPU arithmetic and Brownian distribution, not real-model quality',
                         covariance=covariance, native_arithmetic=records, pfr_cost=a, ordinary_cost=b), indent=2))


if __name__ == '__main__':
    main()
