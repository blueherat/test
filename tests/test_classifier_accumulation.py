import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from classifier_guidance.training import step as full_step
from classifier_guidance.training_accumulation import step as accumulated_step
from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic


@pytest.mark.parametrize('update_head',[False,True])
def test_global_gan_update_equals_microbatch_accumulation_including_spectral_norm(update_head):
    torch.manual_seed(20260922)
    head=torch.nn.Linear(3,3).double()
    micro_head=copy.deepcopy(head)
    critic=BinaryCritic(dimension=4,hidden=7,classes=2).double()
    micro_critic=copy.deepcopy(critic)
    decode=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh()).double().requires_grad_(False)
    feature=torch.nn.Sequential(torch.nn.Linear(5,4),torch.nn.Sigmoid()).double().requires_grad_(False)
    noise=torch.randn(32,3,dtype=torch.double)
    real=torch.randn(32,5,dtype=torch.double)
    labels=torch.arange(32)%2
    expected_parameters=[]
    results=[]
    for model,d,operation in [(head,critic,full_step),(micro_head,micro_critic,accumulated_step)]:
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.99))
        optimizer_d=torch.optim.Adam(d.parameters(),lr=1e-4,betas=(0.,.99))
        diagnostics={}
        kwargs=dict(microbatch=8) if operation is accumulated_step else {}
        result=operation(head=model,critic=d,optimizer_w=optimizer,optimizer_d=optimizer_d,
                         sample=lambda z,y:model(z).tanh(),decode=decode,feature=feature,
                         real=real,noise=noise,labels=labels,update_head=update_head,diagnostics=diagnostics,**kwargs)
        results.append((result,diagnostics))
        expected_parameters.append([p.grad.clone() for p in model.parameters()] if update_head else [])
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-11,atol=1e-12)
    for key in results[0][1]:
        torch.testing.assert_close(results[0][1][key],results[1][1][key],rtol=1e-10,atol=1e-12)
    for name,tensor in critic.state_dict().items():
        # Includes the spectral-norm power-iteration buffers, not just weights.
        torch.testing.assert_close(tensor,micro_critic.state_dict()[name],rtol=1e-11,atol=1e-12)
    for a,b in zip(head.parameters(),micro_head.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
    for a,b in zip(*expected_parameters):
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)


def _distributed_worker(rank, rendezvous, microbatch):
    torch.set_num_threads(1)
    torch.manual_seed(20260922)
    head=torch.nn.Linear(3,3).double()
    critic=BinaryCritic(dimension=4,hidden=7,classes=2).double()
    feature=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Sigmoid()).double().requires_grad_(False)
    initial_head=copy.deepcopy(head.state_dict());initial_critic=copy.deepcopy(critic.state_dict())
    batches=[(torch.randn(32,3,dtype=torch.double),torch.randn(32,3,dtype=torch.double),torch.arange(32)%2)
             for _ in range(3)]

    def optimizers():
        return torch.optim.Adam(head.parameters(),lr=1e-3,betas=(.9,.99)),torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.99))

    def run_batch(batch, update_head, ow, od):
        real,noise,labels=batch
        result=accumulated_step(head=head,critic=critic,optimizer_w=ow,optimizer_d=od,
            sample=lambda z,y:head(z).tanh(),decode=lambda x:x,feature=feature,
            real=real,noise=noise,labels=labels,microbatch=microbatch,update_head=update_head)
        return result

    ow,od=optimizers();expected=[]
    for index,batch in enumerate(batches):
        metrics=run_batch(batch,index>0,ow,od)
        expected.append((metrics,copy.deepcopy(head.state_dict()),copy.deepcopy(critic.state_dict()),
                         [p.grad.clone() for p in head.parameters()] if index>0 else []))
    head.load_state_dict(initial_head);critic.load_state_dict(initial_critic);ow,od=optimizers()
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        for index,batch in enumerate(batches):
            local=tuple(value[rank*16:(rank+1)*16] for value in batch)
            metrics=run_batch(local,index>0,ow,od)
            metrics[-1].square_();dist.all_reduce(metrics);metrics/=2;metrics[-1].sqrt_()
            torch.testing.assert_close(metrics,expected[index][0],rtol=1e-10,atol=1e-12)
            for actual,target in [(head.state_dict(),expected[index][1]),(critic.state_dict(),expected[index][2])]:
                for key in actual:torch.testing.assert_close(actual[key],target[key],rtol=1e-10,atol=1e-12)
            for p,gradient in zip(head.parameters(),expected[index][3]):
                torch.testing.assert_close(p.grad,gradient,rtol=1e-10,atol=1e-12)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('microbatch',[8,16])
def test_two_rank_gan_matches_single_rank_across_warmup_and_adam_updates(tmp_path,microbatch):
    mp.spawn(_distributed_worker,args=(str(tmp_path/'rendezvous'),microbatch),nprocs=2,join=True)
