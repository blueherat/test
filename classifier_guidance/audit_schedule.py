"""Leased real-SiT audit of signed coefficients, zero tail and graph replay."""
import argparse
import fcntl
import json
import os
from pathlib import Path


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    from experiments.weak_reference_loss_20260914.idle import gpu_snapshot,eligible
    gpu=next(r for r in gpu_snapshot() if str(r['index'])==args.gpu)
    lease=Path('/tmp',f"eqvae_idle_{gpu['uuid']}.lock").open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert eligible(next(r for r in gpu_snapshot() if r['uuid']==gpu['uuid']))
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    os.environ['TORCH_COMPILE_DISABLE']='1'
    import torch
    from experiments.guidance_dynamic_50k_20260915.models import Adapter,fingerprint
    from .schedules import GuidanceSchedule,for_native_sit,native_reference
    from .adapters import image_decoder
    from .features import enable_feedback_checkpointing
    from .training import step
    from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
    from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
    from .data import ImageNetData
    from experiments.adversarial_weak_training_20260915 import common as c

    torch.manual_seed(2026091922)
    adapter=Adapter('sit_small');adapter.native.eval().requires_grad_(False)
    schedule=GuidanceSchedule().cuda().eval()
    signatures={name:fingerprint(m) for name,m in [('strong',adapter.model),('weak',adapter.native)]}
    z=torch.randn(1,4,32,32,device='cuda');labels=torch.tensor([2],device='cuda')
    engine=for_native_sit(adapter,schedule,z,labels)
    rows=[]
    for iteration in range(2):
        expected=native_reference(adapter,schedule,z,labels)
        reference,=torch.autograd.grad(expected.square().mean(),(schedule.coefficients,))
        actual=engine(z,labels)
        gradient,=torch.autograd.grad(actual.square().mean(),(schedule.coefficients,))
        row=dict(iteration=iteration,endpoint_max_error=float((actual-expected).abs().max()),
            gradient_relative_error=float((gradient-reference).norm()/reference.norm()),
            first_half_gradient_norm=float(gradient[:32].norm()),
            second_half_gradient_norm=float(gradient[32:].norm()),
            nonzero_tail_gradient_count=int((gradient[32:].abs()>1e-10).sum()),
            coefficients=schedule.coefficients.tolist())
        assert row['endpoint_max_error']==0,row
        assert row['gradient_relative_error']<1e-4,row
        assert row['nonzero_tail_gradient_count']==32,row
        if iteration==0:
            # Independent finite differences for a zero late-time coefficient.
            epsilon=.003
            with torch.no_grad():
                schedule.coefficients[48]+=epsilon
                plus=engine(z,labels).square().mean()
                schedule.coefficients[48]-=2*epsilon
                minus=engine(z,labels).square().mean()
                schedule.coefficients[48]+=epsilon
            finite=float((plus-minus)/(2*epsilon))
            error=abs(finite-float(gradient[48]))/(abs(float(gradient[48]))+1e-12)
            row['late_finite_difference']=dict(step=48,epsilon=epsilon,gradient=float(gradient[48]),
                                             finite_difference=finite,relative_error=error)
            assert error<.03,row
        print({k:v for k,v in row.items() if k!='coefficients'},flush=True)
        rows.append(row)
        with torch.no_grad():schedule.coefficients[32:]=torch.linspace(-.2,.15,32,device='cuda')
        del expected,actual,reference,gradient
        adapter.values.clear()
        z=torch.randn_like(z);labels=(labels+17)%100
    # Full GAN update verifies that D sees the generated endpoint and the
    # coefficient receives the updated critic's image-feedback gradient.
    feature=DifferentiableInception2048().cuda().eval().requires_grad_(False)
    enable_feedback_checkpointing(adapter,feature)
    critic=BinaryCritic(classes=100).cuda()
    optimizer=torch.optim.Adam(schedule.parameters(),lr=1e-3)
    optimizer_d=torch.optim.Adam(critic.parameters(),lr=1e-4)
    real,noise,labels=ImageNetData('sit_small',2026091922).draw(1)
    before=schedule.coefficients.detach().clone()
    metrics=step(head=schedule,critic=critic,optimizer_w=optimizer,optimizer_d=optimizer_d,
        sample=engine,decode=image_decoder(adapter),feature=feature,real=real,noise=noise,
        labels=labels,feature_chunk=0)
    assert torch.all(schedule.coefficients!=before)
    assert all(p.grad is None for m in (adapter.model,adapter.native) for p in m.parameters())
    assert signatures=={name:fingerprint(m) for name,m in [('strong',adapter.model),('weak',adapter.native)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    c.atomic(args.output,dict(passed=True,gpu=gpu,checks=rows,full_gan_metrics=metrics.tolist(),
        frozen_unchanged=True,all_64_coefficients_updated=True,
        sources={str(p.resolve()):c.sha(p) for p in Path(__file__).parent.glob('*.py')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
