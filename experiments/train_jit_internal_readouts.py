"""Train depth4/depth8 readouts on frozen official pixel JiT-B/16; four-rank DDP."""
import os
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import hashlib,json,time,copy
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset,DistributedSampler
from experiments import jit_internal_guidance as jig
from experiments.raev2_training_core import DeterministicImageNetPacked

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/jit_internal_readouts_20260908')
DATA=Path('/data/shared/imagenet-1k/random_access_v1')
STEPS=50000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def atomic_torch(p,x):
    q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)

def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.manual_seed(202609822);torch.cuda.manual_seed_all(202609822)
    ROOT.mkdir(parents=True,exist_ok=True)
    model=jig.load_source('cuda');heads=jig.Readouts().cuda();ema=copy.deepcopy(heads).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(heads.parameters(),lr=1.e-4,betas=(.9,.999),weight_decay=0.)
    train=DDP(heads,device_ids=[local],broadcast_buffers=False)
    dataset=DeterministicImageNetPacked(DATA,image_size=256,horizontal_flip=False)
    labels=np.concatenate(dataset._labels);val_ids=np.array([np.flatnonzero(labels==c)[0] for c in range(1000)])
    train_ids=np.setdiff1d(np.arange(len(dataset)),val_ids);subset=Subset(dataset,train_ids.tolist())
    sampler=DistributedSampler(subset,num_replicas=world,rank=rank,seed=202609822,drop_last=True)
    loader=DataLoader(subset,batch_size=64,sampler=sampler,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
    rng=torch.Generator(device='cuda').manual_seed(202609823+rank)
    if rank==0:
        request=dict(model='JiT-B/16',state='model_ema1',checkpoint_sha256=sha(jig.CHECKPOINT),
            depths=[4,8],clean_output_velocity_loss=True,steps=STEPS,global_batch=256,lr=1.e-4,ema=.9999,
            P_mean=-.8,P_std=.8,t_eps=.05,label_dropout=.1,seed=202609822,
            dataset_manifest_sha256=sha(DATA/'manifest.json'),validation_ids_sha256=hashlib.sha256(val_ids.tobytes()).hexdigest(),
            sources={str(p):sha(p) for p in [Path(__file__),Path(jig.__file__),jig.REPO/'model_jit.py',jig.REPO/'util/model_util.py']})
        assert request['checkpoint_sha256']=='4ebcf24698748548d13bef1b4c3b26c72c6ec2bc633002b3f558697920cb2695'
        if (ROOT/'request.json').exists():assert json.loads((ROOT/'request.json').read_text())==request
        else:atomic_json(ROOT/'request.json',request);np.save(ROOT/'validation_ids.npy',val_ids)
    dist.barrier();step=0;epoch=0;skip=0
    if (ROOT/'last.pt').exists():
        state=torch.load(ROOT/'last.pt',map_location='cpu',weights_only=False)
        heads.load_state_dict(state['heads']);ema.load_state_dict(state['ema']);optimizer.load_state_dict(state['optimizer'])
        step=state['step'];epoch=state['epoch'];skip=state['batch_in_epoch']
        rng.set_state(state['rng_states'][rank]);del state
    start=time.perf_counter();last_time=start;loss_acc=torch.zeros(2,device='cuda');acc=0
    def save(batch_in_epoch):
        states=[None]*world;dist.all_gather_object(states,rng.get_state().cpu())
        if rank==0:
            state=dict(step=step,epoch=epoch,batch_in_epoch=batch_in_epoch,heads=heads.state_dict(),ema=ema.state_dict(),
                       optimizer=optimizer.state_dict(),rng_states=states,request_sha256=sha(ROOT/'request.json'))
            atomic_torch(ROOT/'last.pt',state)
            if step%10000==0:atomic_torch(ROOT/f'step{step:06d}.pt',state)
        dist.barrier()
    @torch.no_grad()
    def validate():
        total=torch.zeros(4,device='cuda');count=0
        vg=torch.Generator(device='cuda').manual_seed(202609824+rank)
        vd=DataLoader(Subset(dataset,val_ids[rank::world].tolist()),batch_size=16,num_workers=0)
        for imgs,y,_ in vd:
            x=imgs.cuda().mul(2).sub(1);y=y.cuda();t=torch.sigmoid(torch.randn(len(x),device='cuda',generator=vg)*.8-.8)
            eps=torch.randn(x.shape,device='cuda',generator=vg);tt=t[:,None,None,None];z=tt*x+(1-tt)*eps
            with torch.autocast('cuda',dtype=torch.bfloat16):
                feats,c=jig.features(model,z,t,y);pred=ema(feats,c);full=model(z,t,y)
            den=(1-tt).clamp_min(.05)
            vals=[((pred[d].float()-x)/den).square().flatten(1).mean(1).sum() for d in ['4','8']]
            vals += [((full.float()-x)/den).square().flatten(1).mean(1).sum(),torch.tensor(len(x),device='cuda')]
            total+=torch.stack(vals);count+=len(x)
        dist.all_reduce(total)
        if rank==0:
            atomic_json(ROOT/f'validation{step:06d}.json',dict(step=step,samples=int(total[3]),depth4=float(total[0]/total[3]),depth8=float(total[1]/total[3]),full=float(total[2]/total[3]),note='Prediction errors only; quality evaluated by sampling.'))
    while step<STEPS:
        sampler.set_epoch(epoch)
        for batch_index,(imgs,y,_) in enumerate(loader):
            if batch_index<skip:continue
            x=imgs.cuda(non_blocking=True).mul(2).sub(1);y=y.cuda(non_blocking=True)
            flip=torch.rand(len(x),device='cuda',generator=rng)<.5;x=torch.where(flip[:,None,None,None],x.flip(-1),x)
            dropped=torch.where(torch.rand(len(x),device='cuda',generator=rng)<.1,1000,y)
            t=torch.sigmoid(torch.randn(len(x),device='cuda',generator=rng)*.8-.8);tt=t[:,None,None,None]
            eps=torch.randn(x.shape,device='cuda',generator=rng);z=tt*x+(1-tt)*eps
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):feats,c=jig.features(model,z,t,dropped)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=train(feats,c)
            losses=torch.stack([((pred[d].float()-x)/(1-tt).clamp_min(.05)).square().mean() for d in ['4','8']])
            loss=losses.sum();assert torch.isfinite(loss);loss.backward();optimizer.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),heads.parameters()):ep.lerp_(p,.0001)
            step+=1;loss_acc+=losses.detach();acc+=1
            if step==1 or step%50==0:
                dist.all_reduce(loss_acc);stats=(loss_acc/(world*acc)).tolist();now=time.perf_counter()
                if rank==0:
                    record=dict(step=step,steps=STEPS,loss_depth4=stats[0],loss_depth8=stats[1],seconds_per_step=(now-last_time)/acc,elapsed_seconds=now-start,peak_memory_bytes=torch.cuda.max_memory_allocated())
                    atomic_json(ROOT/'progress.json',record);print(json.dumps(record),flush=True)
                loss_acc.zero_();acc=0;last_time=now
            if step%1000==0:save(batch_index+1)
            if step%5000==0:validate()
            if step>=STEPS:break
        epoch+=1;skip=0
    if rank==0:atomic_json(ROOT/'complete.json',dict(complete=True,step=step,checkpoint_sha256=sha(ROOT/'last.pt')))
    dist.destroy_process_group()

if __name__=='__main__':main()
