"""Check paired two-scale residuals and the real fixed-weight sampler."""
import json
from pathlib import Path
import torch
from experiments.self_guidance_20260913.check import Analytic
from experiments.weak_reference_20260914 import calibrated as b


def main():
    torch.manual_seed(1415)
    x=torch.randn(3,2,4,4);labels=torch.tensor([1,3,9]);probe=torch.randn_like(x)
    rt=Analytic();full=b.plain.sg.query(rt,x,.3,labels)
    delta=b.residual(rt,x,.3,labels,full,probe,'band')
    assert delta.abs().max()<1e-6
    # A quadratic field has a closed-form two-scale symmetric difference.
    class Quadratic(Analytic):
        def field(self,z,t,kind):
            super().field(z,t,kind)
            return .1*z.square()+t
    rt=Quadratic();full=b.plain.sg.query(rt,x,.3,labels)
    actual=b.residual(rt,x,.3,labels,full,probe,'band')
    expected=-.3*(.2*.7*probe).square()
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)
    from experiments.guidance_pasted_20260912 import common as c
    rt=c.runtime('sit_small')
    noise=torch.randn(2,4,32,32,device='cuda');labels=torch.tensor([2,61],device='cuda')
    cfg=dict(arm='check',kind='calibrated',alpha=0,kernel='band',weights=[0]*8,steps=64)
    zero=b.sample(rt,noise,labels,cfg)
    base=b.plain.sample(rt,noise,labels,dict(cfg,kind='baseline',omega=0))
    assert torch.equal(zero['latents'],base['latents']) and zero['counts']==base['counts']
    result=b.sample(rt,noise,labels,dict(cfg,weights=[1]*8))
    assert result['counts']==dict(full=320,prefix=0)
    assert not torch.equal(result['latents'],base['latents'])
    assert rt.decode(result['latents']).shape==(2,256,256,3)
    out=dict(affine_null=True,quadratic_sign_and_scale=True,native_zero_bitwise=True,
             model_passed=True,counts=result['counts'],fixed_weights_not_DSM_fitted=True)
    Path('docs/data/weak_reference_20260914/band_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
