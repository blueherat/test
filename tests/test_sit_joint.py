import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from classifier_guidance.sampler import Sampler
from classifier_guidance.sit_joint import JointGuidance, gap_anchor, joint_step


def test_joint_replay_matches_full_autograd_for_both_parameter_groups():
    torch.manual_seed(7)
    joint = JointGuidance(torch.nn.Linear(2,2),4,.75).double().eval()
    joint.schedule.coefficients.data.copy_(torch.tensor([.75,0.,-.3,.1]))
    strong = torch.nn.Linear(2,2).double().eval().requires_grad_(False)
    z = torch.randn(2,2,dtype=torch.double,requires_grad=True)
    y = torch.tensor([0,1])
    grid = torch.tensor([0.,.2,.5,.6,1.],dtype=torch.double)
    indices = torch.arange(4,dtype=torch.double)
    def field(x,t,y,index,active):
        s = torch.tanh(strong(x)+t)
        w = torch.sin(joint.weak(x)-t+y[:,None]*.1)
        return s+joint.schedule(index)*(s-w)
    engine = Sampler(field,joint,z,y,grid,indices,[True]*4,heun=True,graphs=False)
    x = z
    for i in range(4):
        h = grid[i+1]-grid[i]
        first = field(x,grid[i],y,indices[i],True)
        second = field(x+h*first,grid[i+1],y,indices[i],True)
        x = x+h/2*(first+second)
    actual = engine(z,y)
    torch.testing.assert_close(actual,x,rtol=0,atol=0)
    variables = (z,*joint.parameters())
    expected_grad = torch.autograd.grad(x.square().mean(),variables)
    actual_grad = torch.autograd.grad(actual.square().mean(),variables)
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert actual_grad[-1][1].abs()>1e-8, 'A zero scale must still receive endpoint gradient'
    assert all(p.grad is None for p in strong.parameters())


def test_soft_anchor_penalizes_radial_rescaling_but_not_sign_and_stops_reference_grad():
    gap = torch.tensor([[1.,2.],[-2.,1.]],dtype=torch.double,requires_grad=True)
    reference = gap.detach().clone().requires_grad_(True)
    assert gap_anchor(gap,reference).item()==0
    assert gap_anchor(-gap,reference).item()==0  # Explicit remaining sign ambiguity.
    loss = gap_anchor(gap*2,reference)
    loss.backward()
    assert loss.item()>0 and gap.grad.abs().sum()>0
    assert reference.grad is None


def test_joint_microbatch_matches_global_gan_update_and_separate_clipping():
    from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
    torch.manual_seed(9)
    joint0 = JointGuidance(torch.nn.Linear(3,3),4,.75).double().eval()
    critic0 = BinaryCritic(dimension=3,hidden=8,classes=2).double()
    real,noise = torch.randn(8,3,dtype=torch.double),torch.randn(8,3,dtype=torch.double)
    labels = torch.tensor([0,1]*4)
    results = []
    for micro in (8,2):
        joint,critic = copy.deepcopy(joint0),copy.deepcopy(critic0)
        optimizer = torch.optim.Adam([
            dict(params=joint.weak.parameters(),lr=1e-4),
            dict(params=joint.schedule.parameters(),lr=2e-4)],betas=(.9,.99))
        optimizer_d = torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.99))
        def sample(x,y):
            return x+joint.schedule.coefficients.mean()*(x-joint.weak(x))
        def probe(real,y):
            # Same independent regularizer for either grouping.
            return (joint.weak.weight.square().mean()-1).square(),{}
        metrics = joint_step(joint=joint,critic=critic,optimizer=optimizer,optimizer_d=optimizer_d,
            sample=sample,decode=lambda x:x,feature=lambda x:x,real=real,noise=noise,
            labels=labels,microbatch=micro,probe=probe)
        assert metrics['weak_update_norm']>0 and metrics['coefficient_update_norm']>0
        results.append((joint,critic,metrics))
    for modules in zip(results[0][:2],results[1][:2]):
        for a,b in zip(modules[0].state_dict().values(),modules[1].state_dict().values()):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    for key in ('d_ce','r1','g_loss','weak_gradient_norm','coefficient_gradient_norm'):
        assert results[0][2][key] == pytest.approx(results[1][2][key],rel=1e-11,abs=1e-12)


def _joint_two_rank_worker(rank,rendezvous):
    from experiments.adversarial_weak_training_20260915.binary_critic import BinaryCritic
    torch.set_num_threads(1);torch.manual_seed(19)
    joint=JointGuidance(torch.nn.Linear(3,3),4,.75).double().eval()
    critic=BinaryCritic(dimension=3,hidden=8,classes=2).double()
    initial_j=copy.deepcopy(joint.state_dict());initial_d=copy.deepcopy(critic.state_dict())
    batches=[(torch.randn(16,3,dtype=torch.double),torch.randn(16,3,dtype=torch.double),torch.arange(16)%2)
             for _ in range(3)]
    def optimizers():
        return (torch.optim.Adam([dict(params=joint.weak.parameters(),lr=1e-4),
            dict(params=joint.schedule.parameters(),lr=2e-4)],betas=(.9,.99)),
            torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.99)))
    def update(batch,global_batch,enabled,ow,od):
        def probe(real,y):
            gap=real-joint.weak(real)
            return gap_anchor(gap,real*.5),{}
        return joint_step(joint=joint,critic=critic,optimizer=ow,optimizer_d=od,
            sample=lambda x,y:x+joint.schedule.coefficients.mean()*(x-joint.weak(x)),
            decode=lambda x:x,feature=lambda x:x,real=batch[0],noise=batch[1],labels=batch[2],
            microbatch=4,probe=probe,probe_real=global_batch[0][:4],probe_labels=global_batch[2][:4],update=enabled)
    ow,od=optimizers();expected=[]
    for i,b in enumerate(batches):
        result=update(b,b,i>0,ow,od)
        expected.append((result,copy.deepcopy(joint.state_dict()),copy.deepcopy(critic.state_dict())))
    joint.load_state_dict(initial_j);critic.load_state_dict(initial_d);ow,od=optimizers()
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        for i,b in enumerate(batches):
            local=tuple(v[rank*8:(rank+1)*8] for v in b)
            result=update(local,b,i>0,ow,od)
            for key in result:
                torch.testing.assert_close(torch.as_tensor(result[key]),torch.as_tensor(expected[i][0][key]),
                    rtol=1e-6,atol=1e-8)
            for actual,target in ((joint.state_dict(),expected[i][1]),(critic.state_dict(),expected[i][2])):
                for key in actual:torch.testing.assert_close(actual[key],target[key],rtol=1e-11,atol=1e-12)
    finally:dist.destroy_process_group()


def test_joint_two_ranks_preserve_global_probe_nonlinearity_and_adam_updates(tmp_path):
    mp.spawn(_joint_two_rank_worker,args=(str(tmp_path/'joint_rendezvous'),),nprocs=2,join=True)
