"""Independent rule checks plus optional real SiT zero/parity checks."""
from contextlib import nullcontext
import argparse
import json
import torch
from experiments.self_guidance_20260913 import sampler as s


class Analytic:
    def __init__(self):
        self.labels = None
        self.counts = dict(full=0, prefix=0)
        self.queries = []
    def context(self):
        return nullcontext()
    def field(self, z, t, kind):
        self.counts['full'] += 1
        self.queries.append((z.clone(), float(t), self.labels.clone()))
        return .1 * z + t.square() + self.labels[:, None, None, None] * .001


def checks():
    rt = Analytic()
    x = torch.ones(2, 1, 2, 2)
    y = torch.tensor([3, 9])
    cfg = dict(kind='sg', alpha=1.25, omega=3., shift=.1, steps=8)
    v, raw, _, active = s.velocity(rt, x, .4, y, cfg)
    conditional = .1 * x + .4**2 + y[:, None, None, None] * .001
    unconditional = .1 * x + .4**2 + .1
    reference = .1 * x + .3**2 + y[:, None, None, None] * .001
    torch.testing.assert_close(v, conditional + 1.25 * (conditional-unconditional) + 3 * (conditional-reference))
    assert active and abs(rt.queries[-1][1] - .3) < 1e-7
    assert torch.equal(rt.queries[-1][0], x) and rt.labels is None
    # Independent Euler rollout: previous raw CONDITIONAL is cached even before gate.
    for omega in (0., 1., 3.):
        cfg = dict(kind='sg_prev', alpha=1.25, omega=omega, steps=8)
        actual = s.sample(Analytic(), x, y, cfg)['latents']
        z, old = x.clone(), None
        for k in range(8):
            t = k / 8
            c = .1*z + t*t + y[:, None, None, None]*.001
            v = c + (1.25 * (c-(.1*z+t*t+.1)) if t < .75 else 0)
            if old is not None and t > .5:
                v = v + omega*(c-old)
            z, old = z + v/8, c.clone()
        torch.testing.assert_close(actual, z, rtol=0, atol=0)
    for kind in ('sg', 'sg_prev'):
        cfg = dict(kind=kind, alpha=1.25, omega=0., steps=8)
        out = s.sample(Analytic(), x, y, cfg)
        base = s.sample(Analytic(), x, y, dict(cfg, kind='baseline'))
        assert torch.equal(out['latents'], base['latents']) and out['counts'] == base['counts']
    cfg = dict(kind='sg_prev', alpha=1.25, omega=3., steps=8)
    a = s.sample(rt, x, y, cfg); b = s.sample(rt, x, y, cfg)
    assert torch.equal(a['latents'], b['latents']) and rt.labels is None
    return dict(cpu_passed=True, checks=['same_state_noisier_time', 'additive_CFG_SG',
        'raw_conditional_history', 'strict_halfway_gate', 'zero_scale_exact_parity',
        'history_reset', 'label_restore', 'actual_branch_counts'])


def model_checks():
    from experiments.guidance_pasted_20260912 import common as c
    from experiments.cfg_transport_search_20260913 import baselines
    rt = c.runtime('sit_small')
    torch.manual_seed(2026091307)
    x = torch.randn(2, 4, 32, 32, device='cuda')
    y = torch.tensor([4, 61], device='cuda')
    cfg = dict(kind='baseline', alpha=1.25, omega=0., steps=64)
    reference = s.sample(rt, x, y, cfg)
    for kind in ('sg', 'sg_prev'):
        result = s.sample(rt, x, y, dict(cfg, kind=kind))
        assert torch.equal(result['latents'], reference['latents'])
        assert result['counts'] == reference['counts']
    ours = s.sample(rt, x, y, dict(cfg, solver='heun'))
    native = baselines.sample(rt, x, y, dict(kind='cfg', alpha=1.25, steps=64))
    assert torch.equal(ours['latents'], native['latents'])
    records = {}
    for kind in ('sg', 'sg_prev'):
        result = s.sample(rt, x, y, dict(cfg, kind=kind, omega=3.))
        records[kind] = dict(counts=result['counts'], rms=float(result['latents'].square().mean().sqrt()))
        assert not torch.equal(result['latents'], reference['latents'])
    return dict(model_passed=True, zero_scale_bitwise=True, native_heun_bitwise=True, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model', action='store_true')
    args = parser.parse_args()
    result = checks()
    if args.model:
        result.update(model_checks())
    print(json.dumps(result, indent=2))
