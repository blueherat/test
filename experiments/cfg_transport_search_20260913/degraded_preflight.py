"""Actual-model endpoint parity and same-state degraded-reference geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from . import baselines as b, degraded_condition as d
from experiments.guidance_pasted_20260912 import common as c

@torch.inference_mode()
def main():
    rt=c.runtime('sit_small');root=c.EXPS/'cfg_transport_search_20260913'
    with np.load(root/'baseline_1k/inputs.npz') as data:
        noise=c.cuda(data['noise'][:8]);labels=c.cuda(data['labels'][:8])
    for kind in ('cfg','apg'):
        base=b.sample(rt,noise,labels,dict(kind=kind,alpha=2.))
        zero=d.sample(rt,noise,labels,dict(kind='degraded_'+kind,alpha=2.,condition_scale=0.))
        assert torch.equal(base['latents'],zero['latents'])
        assert zero['counts']==dict(full=224,prefix=0)
    plain=b.sample(rt,noise,labels,dict(kind='cfg',alpha=0.))
    one=d.sample(rt,noise,labels,dict(kind='degraded_apg',alpha=4.,condition_scale=1.))
    assert torch.equal(plain['latents'],one['latents']) and one['counts']==dict(full=224,prefix=0)
    half=d.sample(rt,noise,labels,dict(kind='degraded_apg',alpha=4.,condition_scale=.5))
    assert torch.isfinite(half['latents']).all()
    rows=[]
    with np.load(root/'baseline_1k/apg_a2_s64/snapshots.npz') as states, rt.context():
        for step in (0,16,32,48):
            z=c.cuda(states[f'step_{step:03d}'][:8]);t=step/64
            vc,vhalf=d.query_pair(rt,z,t,labels,.5)
            old=rt.labels;rt.labels=torch.full_like(labels,100)
            vu=rt.field(z,z.new_tensor(t),'full');rt.labels=old
            gap=vc-vu;new=vc-vhalf;old_secant=vhalf-vu
            pn=b.projection(new,gap);po=b.projection(old_secant,gap)
            torch.testing.assert_close(new-pn,-(old_secant-po),atol=1e-5,rtol=1e-4)
            cosine=(new*gap).flatten(1).sum(1)/(b.norm(new)*b.norm(gap)).flatten().clamp_min(1e-12)
            ratios=(b.norm(new)/b.norm(gap).clamp_min(1e-12)).flatten()
            orth=(b.norm(new-pn)/b.norm(new).clamp_min(1e-12)).flatten()
            rows.append(dict(time=t,new_gap_norm_over_cfg=ratios.cpu().tolist(),
                cosine_with_cfg=cosine.cpu().tolist(),orthogonal_fraction=orth.cpu().tolist()))
    out=dict(samples=8,scale0_cfg_apg_bitwise=True,scale1_conditional_bitwise=True,
             half_finite=True,counts=half['counts'],geometry=rows,module_sha256=c.sha(Path(d.__file__).resolve()))
    target=root/'degraded_condition_model.json';c.atomic(target,out)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
