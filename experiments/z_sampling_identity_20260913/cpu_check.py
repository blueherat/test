"""CPU-only algebra, endpoint, query-budget, clock, and baseline checks.

Run: python -m experiments.z_sampling_identity_20260913.cpu_check
No checkpoints or images are loaded. The output documents implementation
properties only; the fake fields provide no image quality evidence.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import sampler as s
from experiments.cfg_transport_search_20260913 import baselines as b


class FakeRuntime:
    name = 'sit_small'

    def __init__(self, mode='affine'):
        self.mode = mode
        self.labels = None
        self.counts = dict(full=0, prefix=0)
        self.queries = []
        self.mu = np.array([[.11, .13, 0., 0.], [-.07, .16, .04, 0.],
                            [0., 0., -.12, .09], [.02, 0., -.03, .08]])
        self.mc = np.array([[.27, -.04, .03, 0.], [.12, -.1, 0., .02],
                            [.03, 0., .19, -.08], [0., .03, .11, -.04]])
        self.bu = np.array([.06, -.02, .04, -.01])
        self.bc = np.array([-.03, .05, -.01, .07])

    def context(self):
        return nullcontext()

    def value(self, z, t, conditional):
        if self.mode == 'polynomial':
            a, offset = (.21, .035) if conditional else (.13, -.025)
            return a*z + .07*z.square() + offset + .03*t
        matrix = self.mc if conditional else self.mu
        bias = self.bc if conditional else self.bu
        if self.mode == 'constant':
            return torch.as_tensor(bias, dtype=z.dtype, device=z.device)[None, :, None, None].expand_as(z)
        return (torch.einsum('ij,bjhw->bihw', torch.as_tensor(matrix, dtype=z.dtype, device=z.device), z)
                + torch.as_tensor(bias, dtype=z.dtype, device=z.device)[None, :, None, None])

    def field(self, z, t, kind):
        assert kind == 'full' and self.labels.shape == (len(z),)
        mask = self.labels < 100
        assert mask.all() or (~mask).all(), 'Unexpected mixed branch query'
        conditional = bool(mask.all())
        self.counts['full'] += 1
        self.queries.append((float(t), 'c' if conditional else 'u'))
        return self.value(z, t, conditional)


def main(args):
    torch.set_num_threads(1)
    labels = torch.tensor([0, 1], dtype=torch.long)
    x = torch.linspace(-.7, .8, 16, dtype=torch.float64).reshape(2, 4, 1, 2)
    same_field = []
    for weight in (0., 1.):
        rt = FakeRuntime('polynomial')
        expected = x + .125*rt.value(x, .2, bool(weight))
        advanced, detail = s.z_event(rt, x, .2, .125, labels,
                                    strong_w=weight, backward_w=weight, kind='z_anchored_euler')
        assert torch.equal(detail['reflected'], x)
        assert torch.equal(detail['predictor'], x)
        assert torch.equal(advanced, expected)
        assert rt.counts == dict(full=3, prefix=0) and rt.labels is None
        errors = []
        for h in (.125, .0625, .03125):
            old_rt = FakeRuntime('polynomial')
            old, data = s.z_event(old_rt, x, .2, h, labels,
                                 strong_w=weight, backward_w=weight, kind='z_release_euler')
            assert old_rt.counts == dict(full=3, prefix=0)
            velocity = old_rt.value(x, .2, bool(weight))
            derivative = (.21 if weight else .13) + .14*x
            error = data['reflected'] - x + h*h*derivative*velocity
            errors.append(float(torch.linalg.vector_norm(error)))
            assert not torch.equal(old, x+h*velocity)
        ratios = [errors[i]/errors[i+1] for i in range(2)]
        assert all(7.7 < v < 8.3 for v in ratios)
        same_field.append(dict(weight=weight, optimized_calls=3,
                               anchored_bitwise_native=True, old_self_drift_remainder_ratios=ratios))

    # The excluded alpha=1, wb=0 setting really has unequal minimal budgets.
    endpoint_counts = {}
    for kind in s.Z_KINDS:
        rt=FakeRuntime()
        s.z_event(rt,x,.2,.125,labels,strong_w=1.,backward_w=0.,kind=kind)
        endpoint_counts[kind] = rt.counts['full']
    assert endpoint_counts == dict(z_release_euler=3,z_anchored_euler=4)

    # Affine noncommuting fields admit an independent exact matrix inverse.
    linear = []
    for wb in (0., 1.):
        proto = FakeRuntime()
        ws = 1.625
        ma = proto.mc+(ws-1)*(proto.mc-proto.mu)
        ba = proto.bc+(ws-1)*(proto.bc-proto.bu)
        mb, bb = (proto.mu,proto.bu) if wb == 0 else (proto.mc,proto.bc)
        xv=x.numpy().transpose(0,2,3,1).reshape(-1,4)
        anchored_errors=[]; release_errors=[]
        for h in (.125,.0625,.03125,.015625):
            y=xv@((np.eye(4)+h*ma).T)+h*ba
            exact=np.linalg.solve(np.eye(4)+h*mb,(y-h*bb).T).T
            # Closed polynomial q2, derived independently of field calls.
            md,bd=ma-mb,ba-bb
            q2=xv@(np.eye(4)+h*md-h*h*mb@md).T+h*bd-h*h*mb@bd
            expected=q2@(np.eye(4)+h*ma).T+h*ba
            rt=FakeRuntime()
            actual,info=s.z_event(rt,x,.2,h,labels,strong_w=ws,backward_w=wb,kind='z_anchored_euler')
            actual_flat=actual.numpy().transpose(0,2,3,1).reshape(-1,4)
            np.testing.assert_allclose(actual_flat,expected,rtol=1e-14,atol=1e-14)
            reflected=info['reflected'].numpy().transpose(0,2,3,1).reshape(-1,4)
            anchored_errors.append(float(np.linalg.norm(reflected-exact)))
            old_rt=FakeRuntime()
            _,old=s.z_event(old_rt,x,.2,h,labels,strong_w=ws,backward_w=wb,kind='z_release_euler')
            old_reflected=old['reflected'].numpy().transpose(0,2,3,1).reshape(-1,4)
            release_errors.append(float(np.linalg.norm(old_reflected-exact)))
            assert rt.counts['full']==old_rt.counts['full']==5
        ar=np.array(anchored_errors[:-1])/anchored_errors[1:]
        zr=np.array(release_errors[:-1])/release_errors[1:]
        assert np.all((ar>7.7)&(ar<8.3)) and np.all((zr>3.8)&(zr<4.2))
        linear.append(dict(backward_w=wb,anchored_inverse_error_ratios=ar.tolist(),
                           release_inverse_error_ratios=zr.tolist()))

    # Generic settings are all real five-query events. Constant fields give
    # exact effective guidance algebra; float64 allows only rounding error.
    alphas=(.75,1.125,1.25,1.5,2.)
    max_constant_error=0.
    for a in alphas:
        for wb in (0.,1.):
            ws=(1+a+wb)/2
            for kind in s.Z_KINDS:
                rt=FakeRuntime('constant')
                advanced,_=s.z_event(rt,x,.2,1/56,labels,strong_w=ws,backward_w=wb,kind=kind)
                c,u=rt.value(x,.2,True),rt.value(x,.2,False)
                target=x+(1/56)*(c+a*(c-u))
                error=float((advanced-target).abs().max())
                assert error<1e-14 and rt.counts==dict(full=5,prefix=0)
                max_constant_error=max(max_constant_error,error)

    # Full trajectories: count real calls, inspect every query's old physical
    # time, and compare delegated baselines using the real integrator code.
    noise=torch.randn((2,4,32,32),generator=torch.Generator().manual_seed(913))
    original=noise.clone()
    rollout=[]
    for kind in s.Z_KINDS:
        for a in alphas:
            for wb in (0.,1.):
                rt=FakeRuntime()
                cfg=dict(kind=kind,alpha=a,backward_w=wb,steps=56,cutoff=.75)
                result=s.sample(rt,noise,labels,cfg,snapshots=True)
                assert result['counts']==dict(full=224,prefix=0)
                assert set(result['snapshots'])=={'step_000','step_014','step_028','step_042','step_056'}
                np.testing.assert_allclose(result['trace'][:,0,1],np.arange(56)/56,atol=3e-8,rtol=0)
                assert result['trace'].shape==(56,2,6)
                cursor=0
                for k in range(56):
                    count=5 if k<42 else 1
                    block=rt.queries[cursor:cursor+count]
                    assert len(block)==count and all(abs(t-k/56)<3e-8 for t,_ in block)
                    if k<42:
                        assert [branch for _,branch in block]==['c','u','c' if wb==1 else 'u','c','u']
                    else:assert block[0][1]=='c'
                    cursor+=count
                assert cursor==224 and torch.equal(noise,original) and rt.labels is None
                rollout.append(dict(kind=kind,alpha=a,backward_w=wb,full_calls=224))
    for kind in ('cfg_euler','cfg_euler128','cfg','apg','ctrl'):
        rt=FakeRuntime()
        cfg=dict(kind=kind,alpha=1.25,steps=128 if kind in s.EULER_KINDS else 64,cutoff=.75,beta=-.5)
        result=s.sample(rt,noise,labels,cfg,snapshots=True)
        assert result['counts']==dict(full=224,prefix=0)
        if kind in s.EULER_KINDS:
            expected=noise.clone();independent=FakeRuntime()
            for k in range(128):
                conditional=independent.value(expected,k/128,True)
                velocity=conditional if k>=96 else conditional+1.25*(conditional-independent.value(expected,k/128,False))
                expected=expected+(1/128)*velocity
            assert torch.equal(result['latents'],expected)
            assert set(result['snapshots'])=={'step_000','step_032','step_064','step_096','step_128'}
        else:
            expected=b.sample(FakeRuntime(),noise,labels,cfg,snapshots=True)
            assert torch.equal(result['latents'],expected['latents'])
            for key in expected['snapshots']:
                assert torch.equal(result['snapshots'][key],expected['snapshots'][key])
        rollout.append(dict(kind=kind,full_calls=224,independent_or_baseline_bitwise=True))

    for kind in s.Z_KINDS:
        try:
            s.sample(FakeRuntime(),noise,labels,dict(kind=kind,alpha=1.,backward_w=0.,steps=56))
            raise RuntimeError('Excluded strong=1 configuration was accepted')
        except ValueError:pass
        try:
            s.sample(FakeRuntime(),noise,labels,dict(kind=kind,alpha=1.25,backward_w=0.,forward_w=9.,steps=56))
            raise RuntimeError('Inconsistent forward_w was accepted')
        except ValueError:pass
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(s.__file__),Path(__file__))}
    report=dict(complete=True,cpu_only=True,same_field=same_field,excluded_endpoint_calls=endpoint_counts,
                affine_inverse=linear,max_constant_field_error=max_constant_error,rollouts=rollout,
                source_sha256=hashes,interpretation='Implementation and algebra only; no image quality evidence.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,output=str(args.output),rollouts=len(rollout),source_sha256=hashes)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('/tmp/z_sampling_identity_cpu_20260913.json'))
    main(parser.parse_args())
