import copy
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from . import config as k
from .data import Stream,loader
from .models import Adapter,fingerprint
from . import training as train
from .sampling import integrate
from experiments import train_imagenet100_sit_flow as base


def cpu():
    from .sampling import batch_starts
    request=k.verify()
    for model in k.MODELS:
        assert request['real_data'][model]['train_images']>(100000 if model=='sit_small' else 1000000)
        for n in (1000,5000):
            starts=[start for rank in range(4) for start in batch_starts(model,n,rank)]
            coverage=[i for start in starts for i in range(start,start+k.settings(model)['sample_batch'])]
            assert sorted(coverage)==list(range(n)) and len(set(coverage))==n
        old=np.load(k.model_root(model)/'data/train_ids.npy')
        held=np.load(k.model_root(model)/'data/validation_ids.npy')
        if model=='jit':assert not np.intersect1d(old,held).size
    choices,ticks=k.refine([dict(tick=t,fid=f,valid=True) for t,f in ((0,10),(16,11),(32,100),(48,11),(64,10),(80,100))],16,8)
    assert len(choices)==2 and ticks==list(range(0,65,8))
    k.atomic(k.ROOT/'cpu_checks.json',dict(passed=True,full_training_sets=True,
        no_missing_or_duplicate_quality_samples=True,connected_refinement=True,
        request_sha256=k.sha(k.ROOT/'request.json')))


def gpu(model,context):
    root=k.model_root(model)/'checks';root.mkdir(parents=True,exist_ok=True)
    adapter=Adapter(model);initial_hash=fingerprint(adapter.model)
    stream=Stream(model,'real',context)
    g=torch.Generator(device='cuda').manual_seed(adapter.cfg['seed']+121+context.rank)
    first=stream.draw(g)
    if model=='sit_small':
        native_loader,_=base.create_loader(cache_dir=k.SIT_DATA,split='train',local_batch_size=64,
            context=context,seed=adapter.cfg['seed'],shuffle=True,num_workers=0,prefetch_factor=4,drop_last=True)
        native_moments,native_labels=next(iter(native_loader))
        np.testing.assert_array_equal(native_labels.numpy(),first['true_labels'].cpu().numpy())
        # Re-read the same image IDs: posterior draws must differ, while the native helper is unchanged.
        moments=native_moments.cuda();a=base.sample_sdvae_posterior(moments,torch.randn_like(first['positive']))
        b=base.sample_sdvae_posterior(moments,torch.randn_like(first['positive']))
        assert not torch.equal(a,b)
    head=train.initialized_head(adapter);ema=copy.deepcopy(head).eval().requires_grad_(False)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=0.,betas=(.9,.999),fused=True)
    ddp=DDP(head,device_ids=[0],broadcast_buffers=False,static_graph=True)
    state=train.State(root/'resume_test',head,ema,opt,g,context)
    # This disposable checkpoint does not contribute training steps to any candidate.
    assert state.step==0
    opt.zero_grad(set_to_none=True);loss=train.compute_loss(adapter,ddp,first,k.ARMS['real']);loss.backward();opt.step()
    state.step=1;state.save()
    expected_batch=stream.draw(g)
    opt.zero_grad(set_to_none=True);loss=train.compute_loss(adapter,ddp,expected_batch,k.ARMS['real']);loss.backward();opt.step()
    expected={name:value.detach().clone() for name,value in head.state_dict().items()}
    restored=train.State(root/'resume_test',head,ema,opt,g,context)
    repeated=Stream(model,'real',context,start_step=1).draw(g)
    for key in ('positive','labels','pos_id','noise','t'):assert torch.equal(expected_batch[key],repeated[key]),key
    opt.zero_grad(set_to_none=True);loss=train.compute_loss(adapter,ddp,repeated,k.ARMS['real']);loss.backward();opt.step()
    assert all(torch.equal(expected[name],value) for name,value in head.state_dict().items()),'Optimizer resume differs'
    # DDP replicas must contain identical parameters after the actual production loss update.
    flat=torch.cat([p.detach().flatten() for p in head.parameters()]);reference=flat.clone()
    dist.broadcast(reference,0);assert torch.equal(flat,reference)
    bank=k.model_root(model)/'quality_inputs';n=adapter.cfg['sample_batch']
    noise=torch.from_numpy(np.array(np.load(bank/'noise.npy',mmap_mode='r')[:n])).cuda()
    labels=torch.from_numpy(np.load(bank/'labels.npy')[:n]).cuda()
    first_zero,counts=integrate(adapter,None,'strong',0.,noise,labels)
    second_zero,other=integrate(adapter,head,'real',0.,noise,labels)
    assert torch.equal(first_zero,second_zero) and counts==other and counts['head']==0
    assert fingerprint(adapter.model)==initial_hash and all(p.grad is None for p in adapter.model.parameters())
    k.atomic(root/f'rank{context.rank}.json',dict(passed=True,rank=context.rank,
        gpu_uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES'],source_unchanged=True,
        original_full_data_loader=True,data_and_optimizer_resume_exact=True,ddp_parameters_identical=True,
        zero_guidance_exact=True,zero_counts=counts,request_sha256=k.sha(k.ROOT/'request.json')))
    dist.barrier()
    if context.is_main:
        rows=[k.read(root/f'rank{r}.json') for r in range(4)]
        assert len({row['gpu_uuid'] for row in rows})==4
        k.atomic(root/'complete.json',dict(complete=True,passed=True,world_size=4,ranks=rows,
            request_sha256=k.sha(k.ROOT/'request.json')))
    dist.barrier()
