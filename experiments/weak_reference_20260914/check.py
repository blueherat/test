"""Checks of scientific invariants and native baseline equivalence."""
import argparse
import json
from pathlib import Path
import torch
from experiments.self_guidance_20260913.check import Analytic
from experiments.self_guidance_20260913 import sampler as native
from experiments.weak_reference_20260914 import sampler as s


def cpu_checks():
    torch.manual_seed(1401)
    x = torch.randn(3, 2, 3, 3)
    y = torch.tensor([2, 4, 8])
    cfg = dict(arm='check', kind='log', alpha=1.25, omega=1, kappa=.2, steps=8)
    rt = Analytic()
    a = s.sample(rt, x, y, cfg)
    b = native.sample(Analytic(), x, y, dict(cfg, kind='baseline', omega=0))
    torch.testing.assert_close(a['latents'], b['latents'], rtol=1e-6, atol=1e-6)
    assert a['counts']['full'] == 30 and b['counts']['full'] == 14
    # Full anisotropic, non-diagonal affine score: exact algebra up to FP64 roundoff.
    cov = torch.tensor([[2., .4], [.4, .1]], dtype=torch.float64)
    precision = torch.linalg.inv(cov)
    u, eps = torch.randn(100, 2, dtype=torch.float64), torch.randn(100, 2, dtype=torch.float64)
    affine = lambda z: -(z-.3) @ precision
    error = (affine(u)-.5*(affine(u+eps)+affine(u-eps))).abs().max().item()
    assert error < 1e-11
    for key in ('omega', 'kappa'):
        z = s.sample(Analytic(), x, y, dict(cfg, **{key:0}))
        assert torch.equal(z['latents'], b['latents']) and z['counts'] == b['counts']
    # Nonlinear field exposes wrong signs, asymmetric query locations and RNG leaks.
    class Nonlinear(Analytic):
        def field(self, z, t, kind):
            super().field(z,t,kind)
            return .1*z.square() + t + self.labels[:,None,None,None]*.001
    rt = Nonlinear()
    out = s.sample(rt,x,y,cfg)
    repeat = s.sample(Nonlinear(),x,y,cfg)
    assert torch.equal(out['latents'],repeat['latents'])
    split = torch.cat([s.sample(Nonlinear(), x[i:i+1],y[i:i+1],cfg)['latents'] for i in range(3)])
    assert torch.equal(out['latents'],split)
    queries = rt.queries[:4]
    torch.testing.assert_close(.5*(queries[2][0]+queries[3][0]),x)
    assert all(q[1]==0 for q in queries)
    assert not torch.equal(out['latents'],b['latents'])
    return dict(cpu_passed=True, gaussian_fp64_max_error=error,
        checks=['affine_null', 'zero_strength_and_radius_bitwise', 'same_time_symmetric_queries',
                'per_image_probe_rng_batch_invariance', 'repeatable', 'actual_forward_counts'])


def real_checks():
    from experiments.guidance_pasted_20260912 import common as c
    rt = c.runtime('sit_small')
    torch.manual_seed(1402)
    x = torch.randn(2,4,32,32,device='cuda'); y=torch.tensor([2,61],device='cuda')
    cfg=dict(arm='check',kind='log',alpha=1.25,kappa=.2,omega=0,steps=64)
    a=s.sample(rt,x,y,cfg)
    b=native.sample(rt,x,y,dict(cfg,kind='baseline'))
    assert torch.equal(a['latents'],b['latents']) and a['counts']==b['counts']
    guided=s.sample(rt,x,y,dict(cfg,omega=1),snapshots=True)
    assert guided['counts']['full']==240 and torch.isfinite(guided['latents']).all()
    assert not torch.equal(guided['latents'],b['latents'])
    pixels=rt.decode(guided['latents'])
    assert pixels.shape==(2,256,256,3)
    return dict(model_passed=True,native_zero_bitwise=True,counts=guided['counts'],
        max_memory_allocated=torch.cuda.max_memory_allocated(),
        latent_delta_rms=float((guided['latents']-b['latents']).square().mean().sqrt()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',action='store_true');p.add_argument('--output')
    args=p.parse_args(); result=cpu_checks()
    if args.model: result.update(real_checks())
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
