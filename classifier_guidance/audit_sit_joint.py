"""Actual SiT full-trajectory joint gradients versus checkpointed autograd."""
import argparse
from pathlib import Path
import time

import torch
from torch.utils.checkpoint import checkpoint

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from .evaluate_sit_transformer import load_head, Field, integrate
from .sit_joint import JointGuidance, JointField, make_sampler


def main(args):
    _,world = c.setup(); assert world==1
    for path,digest in c.read(args.output/'request.json')['sources'].items():
        assert c.sha(path)==digest
    torch.manual_seed(2026092207)
    adapter = Adapter('sit_small')
    weak,_ = load_head(adapter,args.head_checkpoint)
    joint = JointGuidance(weak.requires_grad_(True),64,.75).cuda().eval()
    frozen = fingerprint(adapter.model)
    z = torch.randn(args.microbatch,4,32,32,device='cuda')
    y = torch.arange(args.microbatch,device='cuda')
    begin = time.perf_counter()
    engine = make_sampler(adapter,joint,z,y)
    with torch.no_grad():
        initial = engine(z,y)
        baseline = integrate(Field(adapter,joint.weak,'guided'),z,y,.75,'full','guided')
    torch.testing.assert_close(initial,baseline,rtol=2e-4,atol=2e-5)
    initial_error = (initial-baseline).abs().max().item()
    with torch.no_grad():
        joint.schedule.coefficients.copy_(torch.linspace(.75,-.25,64,device='cuda'))
        joint.schedule.coefficients[32] = 0
    z.requires_grad_(True)
    field = JointField(adapter,joint)
    grid = torch.linspace(0,1,65,device='cuda')
    def ordinary_step(x,t,u,index):
        first = field(x,t,y,index,True)
        second = field(x+(u-t)*first,u,y,index,True)
        return x+(u-t)/2*(first+second)
    reference = z
    for i,(t,u) in enumerate(zip(grid[:-1],grid[1:])):
        reference = checkpoint(ordinary_step,reference,t,u,grid.new_tensor(i),use_reentrant=False)
    cotangent = torch.randn_like(reference)/reference.numel()
    variables = (z,*joint.parameters())
    expected = torch.autograd.grad((reference*cotangent).sum(),variables)
    actual = engine(z,y)
    got = torch.autograd.grad((actual*cotangent).sum(),variables)
    errors = []
    for name,a,b in zip(['noise',*[n for n,_ in joint.named_parameters()]],got,expected):
        delta = (a-b).double().norm().item()
        relative = delta/max(b.double().norm().item(),1e-12)
        torch.testing.assert_close(a,b,rtol=3e-3,atol=2e-6)
        errors.append(dict(name=name,max_abs=(a-b).abs().max().item(),relative_l2=relative,
                           norm=b.norm().item()))
    torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-5)
    assert got[-1][32].abs().item()>0, 'Zero coefficient lost its derivative'
    assert frozen==fingerprint(adapter.model)
    assert all(p.grad is None for p in adapter.model.parameters())
    torch.cuda.synchronize()
    c.atomic(args.output/'audit.json',dict(passed=True,microbatch=args.microbatch,steps=64,nfe=128,
        baseline_max_abs=initial_error,endpoint_max_abs=(actual-reference).abs().max().item(),
        gradients=errors,zero_coefficient_gradient=got[-1][32].item(),
        seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        frozen_unchanged=True))
    print(c.read(args.output/'audit.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--head-checkpoint',type=Path,required=True)
    p.add_argument('--microbatch',type=int,default=8)
    main(p.parse_args())
