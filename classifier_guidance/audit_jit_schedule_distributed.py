"""Compare one- and two-GPU updates from the same real JiT checkpoint and batch."""
import argparse
import copy
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from .data import ImageNetData
from .jit_schedule import HEAD,load_runtime,constant_schedule,optimize,sampler,signature,decoder
from .probe_jit_schedule import difference
from .training_accumulation import step


def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==2 and torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    for path,digest in c.read(args.output/'request.json')['sources'].items():assert c.sha(path)==digest
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    runtime,provenance=load_runtime(args.head_checkpoint);optimize(runtime,precast=True)
    frozen=signature(runtime.net)
    assert frozen==state['frozen'] and provenance==state['provenance']
    schedule=constant_schedule(.5);schedule.load_state_dict(state['schedule'])
    critic=BinaryCritic(classes=1000).cuda();critic.load_state_dict(state['critic'])
    feature=DifferentiableInception2048().cuda().eval().requires_grad_(False)
    data=ImageNetData('jit',state['config']['seed']);data.load_state_dict(state['data_rng'])
    real,noise,labels=data.draw(32)
    sample=sampler(runtime,schedule,noise[:8],labels[:8])

    def update(batch):
        schedule.load_state_dict(state['schedule']);critic.load_state_dict(state['critic'])
        ow=torch.optim.Adam(schedule.parameters(),lr=state['config']['lr_a'],betas=(.9,.99))
        od=torch.optim.Adam(critic.parameters(),lr=state['config']['lr_d'],betas=(0.,.99))
        ow.load_state_dict(copy.deepcopy(state['optimizer']));od.load_state_dict(copy.deepcopy(state['optimizer_d']))
        diagnostics={}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        result=step(head=schedule,critic=critic,optimizer_w=ow,optimizer_d=od,sample=sample,
                    decode=decoder,feature=feature,real=batch[0],noise=batch[1],labels=batch[2],
                    microbatch=8,r1=state['config']['r1'],diagnostics=diagnostics)
        if dist.is_initialized():
            result[-1].square_();dist.all_reduce(result);result/=world;result[-1].sqrt_()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        return result.detach(),diagnostics,elapsed,torch.cuda.max_memory_reserved()/1024**3

    if rank==0:
        times=[]
        for repeat in range(args.repeats+1):
            expected,diagnostic,seconds,memory=update((real,noise,labels))
            if repeat:times.append(seconds)
        reference=dict(metrics=expected.clone(),gradient=diagnostic['head_gradient'].clone(),
                       schedule=schedule.coefficients.detach().clone(),
                       critic={key:value.detach().clone() for key,value in critic.state_dict().items()})
        baseline=dict(median_seconds=statistics.median(times),seconds=times,peak_reserved_gib=memory)
        c.atomic(args.output/'single_gpu.json',baseline)
    # The reference above has no initialized process group, hence no hidden
    # gradient averaging or feature gathering. The same captured microbatch is
    # reused below after restoring both optimizers and all trainable weights.
    c.setup()
    for value in (real,noise,labels):dist.broadcast(value,src=0)
    sl=slice(rank*16,(rank+1)*16)
    times=[]
    for repeat in range(args.repeats+1):
        result,diagnostic,seconds,memory=update((real[sl],noise[sl],labels[sl]))
        perf=torch.tensor([seconds,memory],device='cuda',dtype=torch.float64)
        dist.all_reduce(perf,op=dist.ReduceOp.MAX)
        seconds,memory=perf.cpu().tolist()
        if repeat:times.append(seconds)
    replicas=[None]*world
    replica=dict(schedule=signature(schedule),critic=signature(critic))
    dist.all_gather_object(replicas,replica)
    assert all(value==replica for value in replicas)
    assert signature(runtime.net)==frozen and all(p.grad is None for p in runtime.net.parameters())
    if rank==0:
        checks=dict(metrics=difference(result,reference['metrics']),
                    gradient=difference(diagnostic['head_gradient'],reference['gradient']),
                    coefficients=difference(schedule.coefficients,reference['schedule']))
        critic_max=max(float((value-reference['critic'][key]).abs().max()) for key,value in critic.state_dict().items())
        # Report actual errors; reject large discrepancies instead of assuming
        # that mathematically equivalent averaging is bitwise identical.
        passed=checks['gradient']['relative']<.01 and checks['coefficients']['max_abs']<1e-5 and critic_max<1e-5
        output=dict(passed=passed,checkpoint=str(args.checkpoint),step=state['step'],
                    single_gpu=baseline,two_gpu=dict(median_seconds=statistics.median(times),seconds=times,
                                                    peak_reserved_gib_per_gpu=memory),
                    comparison=checks,critic_state_max_abs=critic_max,replicas_identical=True,
                    frozen_unchanged=True,global_batch=32,microbatch=8,
                    noise_sha256=c.original.array_sha(noise.cpu().numpy()),
                    real_sha256=c.original.array_sha(real.cpu().numpy()))
        c.atomic(args.output/'result.json',output);print(output,flush=True)
        torch.save(dict(single=reference,two_gpu=dict(metrics=result,gradient=diagnostic['head_gradient'],
                   schedule=schedule.state_dict(),critic=critic.state_dict())),args.output/'comparison.pt')
    dist.barrier();dist.destroy_process_group()
    if rank==0:assert passed,checks


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('jit',),default='jit')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--head-checkpoint',type=Path,default=HEAD)
    p.add_argument('--repeats',type=int,default=3)
    main(p.parse_args())
