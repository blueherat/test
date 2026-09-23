"""Matched real checkpoint/batch, single versus two-GPU SiT joint updates."""
import argparse
import copy
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from experiments.guidance_dynamic_50k_20260915.models import Adapter,fingerprint
from .evaluate_sit_transformer import load_head
from .sit_joint import JointGuidance,GapProbe,make_sampler,joint_step
from .data import ImageNetData
from .features import enable_feedback_checkpointing
from .adapters import image_decoder


def main(args):
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==2
    torch.cuda.set_device(0);torch.set_num_threads(2)
    for path,digest in c.read(args.output/'request.json')['sources'].items():assert c.sha(path)==digest
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    adapter=Adapter('sit_small');weak,_=load_head(adapter,args.head_checkpoint)
    reference=copy.deepcopy(weak).requires_grad_(False)
    joint=JointGuidance(weak.requires_grad_(True),64,.75).cuda().eval()
    critic=BinaryCritic(classes=100).cuda()
    feature=DifferentiableInception2048().cuda().eval().requires_grad_(False)
    enable_feedback_checkpointing(adapter,feature)
    data=ImageNetData('sit_small',state['args']['seed']+1009);data.load_state_dict(state['data_rng'])
    real,noise,labels=data.draw(32)
    probe=GapProbe(adapter,joint,reference,state['args']['seed']+2018)
    sample=make_sampler(adapter,joint,noise[:8],labels[:8])
    frozen=fingerprint(adapter.model)
    def update(batch):
        joint.load_state_dict(state['joint']);critic.load_state_dict(state['critic'])
        probe.generator.set_state(state['probe_rng'])
        ow=torch.optim.Adam([dict(params=joint.weak.parameters(),lr=1e-6),
            dict(params=joint.schedule.parameters(),lr=2e-4)],betas=(.9,.99))
        od=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.99))
        ow.load_state_dict(copy.deepcopy(state['optimizer']));od.load_state_dict(copy.deepcopy(state['optimizer_d']))
        diagnostics={};torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        metrics=joint_step(joint=joint,critic=critic,optimizer=ow,optimizer_d=od,sample=sample,
            decode=image_decoder(adapter),feature=feature,real=batch[0],noise=batch[1],labels=batch[2],
            microbatch=8,probe=probe,probe_real=real[:8],probe_labels=labels[:8],
            norm_weight=state['args']['norm_weight'],r1=state['args']['r1'],diagnostics=diagnostics)
        torch.cuda.synchronize()
        return dict(seconds=time.perf_counter()-begin,metrics=metrics,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            gradients={k:v.cpu() for k,v in diagnostics.items()},
            joint={k:v.detach().cpu().clone() for k,v in joint.state_dict().items()},
            critic={k:v.detach().cpu().clone() for k,v in critic.state_dict().items()})
    if rank==0:single=update((real,noise,labels))
    c.setup()
    for v in (real,noise,labels):dist.broadcast(v,src=0)
    sl=slice(rank*16,(rank+1)*16)
    dual=update((real[sl],noise[sl],labels[sl]))
    signatures=[None]*2
    sig=dict(joint=fingerprint(joint),critic=fingerprint(critic))
    dist.all_gather_object(signatures,sig);assert signatures[0]==signatures[1]
    assert fingerprint(adapter.model)==frozen and all(p.grad is None for p in adapter.model.parameters())
    if rank==0:
        comparisons={}
        for key in single['gradients']:
            a,b=dual['gradients'][key],single['gradients'][key]
            comparisons[key]=dict(max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/b.norm().clamp_min(1e-12)).item())
        for key in ('joint','critic'):
            comparisons[key+'_max_abs']=max((dual[key][n]-v).abs().max().item() for n,v in single[key].items())
        passed=all(comparisons[k]['relative_l2']<.01 for k in ('weak_gradient','coefficient_gradient'))
        passed=passed and comparisons['joint_max_abs']<2e-6 and comparisons['critic_max_abs']<1e-5
        result=dict(passed=passed,checkpoint=str(args.checkpoint),step=state['step'],
            single={k:v for k,v in single.items() if k not in ('gradients','joint','critic')},
            dual={k:v for k,v in dual.items() if k not in ('gradients','joint','critic')},
            comparisons=comparisons,replicas_identical=True,frozen_unchanged=True,
            global_batch=32,microbatch=8,shared_global_probe_batch=8)
        c.atomic(args.output/'result.json',result)
        torch.save(dict(single=single,dual=dual),args.output/'comparison.pt')
        print(result,flush=True)
        assert passed,comparisons
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--head-checkpoint',type=Path,required=True)
    main(p.parse_args())
