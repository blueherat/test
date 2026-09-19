from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from . import catalog as c,models
from experiments.sit_measure_guidance_20260912 import data as real_data
from experiments.sit_strong_reference_20260912 import train as parent
from experiments.lifting_scale_sweep_20260909 import Runtime,atomic,read,sha

state_sha=parent.state_sha


def sources():
    return sorted(set([*Path(__file__).resolve().parent.glob('*.py'),c.PROTOCOL,
        Path(models.previous.__file__),Path(models.previous.heads.__file__),Path(real_data.__file__),Path(parent.__file__),
        c.WORK/'experiments/sit_strong_reference_20260912/run.py']))


def prepare():
    c.ROOT.mkdir(parents=True,exist_ok=True)
    if (c.ROOT/'training_request.json').exists():return verify()
    previous=parent.verify()
    hashes=dict(previous['sources']);hashes.update({str(p):sha(p) for p in sources()})
    inputs={str(real_data.ROOT/name):sha(real_data.ROOT/name) for name in
        ('train_moments.npy','validation_moments.npy','class_means.npy','local_partners.npy','random_partners.npy','heat_sigma.npy')}
    stops=dict(previous['old_stop_markers'])
    cancelled=c.ROOT.parent/'sit_reference_aggregation_20260912/STOP_AFTER_CURRENT'
    assert cancelled.exists();stops[str(cancelled)]=sha(cancelled)
    request=dict(sources=hashes,assets=previous['assets'],input_files=inputs,old_stop_markers=stops,
        methods=c.METHODS,steps=c.STEPS,batch=c.BATCH,learning_rate=c.LR,ema=c.EMA,seed=c.TRAIN_SEED,
        target='real-data FM velocity X-epsilon, no model supervision',
        losses={'ig_square':'mean(residual**2)/2','ig_quartic':'mean(residual**4)/4'},
        time_distribution='uniform(.01,.99)',strong_frozen=True,selected_by_validation=False,created_unix=time.time())
    atomic(c.ROOT/'training_request.json',request)
    folder=c.ROOT/'sources';folder.mkdir(exist_ok=True)
    for i,path in enumerate(sorted(hashes)):(folder/f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    return request


def verify():
    request=read(c.ROOT/'training_request.json')
    for key in ('sources','assets','input_files','old_stop_markers'):
        for path,digest in request[key].items():assert sha(path)==digest,(key,path)
    return request


@torch.no_grad()
def batch(rt,pool,generator,count):
    clean,labels=pool.draw('native',generator,count)
    eps=torch.randn(clean.shape,device='cuda',generator=generator)
    times=.01+.98*torch.rand(count,device='cuda',generator=generator)
    z=times[:,None,None,None]*clean+(1-times[:,None,None,None])*eps
    with models.Capture(rt,depths=(4,)) as capture:
        rt.model(z,times,labels)
        return capture.values[4],capture.context,clean-eps


@torch.no_grad()
def validate(rt,head,pool):
    generator=torch.Generator(device='cuda').manual_seed(c.TRAIN_SEED+100)
    second,fourth,number=0.,0.,0
    for _ in range(25):
        f,context,target=batch(rt,pool,generator,32)
        residual=(models.unpatchify(rt,head(f,context))-target).double()
        second+=float(residual.square().sum());fourth+=float(residual.pow(4).sum());number+=residual.numel()
    return dict(images=800,mse=second/number,fourth_moment=fourth/number)


def train(method):
    verify();out=c.ROOT/'training'/method;out.mkdir(parents=True,exist_ok=True)
    with (out/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            done=read(out/'complete.json');assert done['checkpoint_sha256']==sha(out/'model.pt');return
        rt=Runtime('sit_small');torch.set_num_threads(4);torch.manual_seed(c.TRAIN_SEED)
        strong_hash=state_sha(rt.model);head=models.make_head(rt,method).requires_grad_(True)
        initial_hash=state_sha(head);ema=copy.deepcopy(head).requires_grad_(False)
        pool,heldout=real_data.Pool('train'),real_data.Pool('validation');before=validate(rt,ema,heldout)
        optimizer=torch.optim.AdamW(head.parameters(),lr=c.LR,betas=(.9,.999),weight_decay=0.)
        generator=torch.Generator(device='cuda').manual_seed(c.TRAIN_SEED)
        request_hash=sha(c.ROOT/'training_request.json');step0,elapsed0,fingerprints=0,0.,[]
        if (out/'resume.pt').exists():
            saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
            assert saved['request_sha256']==request_hash
            head.load_state_dict(saved['head']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['generator']);step0=saved['step'];elapsed0=saved['elapsed'];fingerprints=saved['fingerprints']
        losses=[];torch.cuda.synchronize();begin=time.perf_counter()
        for step in range(step0+1,c.STEPS+1):
            f,context,target=batch(rt,pool,generator,c.BATCH)
            residual=models.unpatchify(rt,head(f,context))-target
            loss=residual.square().mean()/2 if method=='ig_square' else residual.pow(4).mean()/4
            assert torch.isfinite(loss) and not f.requires_grad
            if step==1 or step%250==0:
                fingerprints.append(dict(step=step,target_sha256=hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest(),
                    generator_sha256=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),head.parameters()):ep.lerp_(p,1-c.EMA)
            losses.append(float(loss.detach()))
            if step==1 or step%100==0:
                torch.cuda.synchronize();elapsed=elapsed0+time.perf_counter()-begin
                status=dict(phase='training',method=method,step=step,total=c.STEPS,loss=float(np.mean(losses[-100:])),gradient_norm=float(norm),elapsed_seconds=elapsed)
                atomic(out/'status.json',status);print(status,flush=True)
            stop=(c.ROOT/'STOP_AFTER_CURRENT').exists()
            if step%500==0 or stop:
                torch.cuda.synchronize();elapsed=elapsed0+time.perf_counter()-begin
                temp=out/'resume.tmp.pt';torch.save(dict(head=head.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                    generator=generator.get_state(),step=step,elapsed=elapsed,fingerprints=fingerprints,request_sha256=request_hash),temp);temp.replace(out/'resume.pt')
            if stop:atomic(out/'status.json',dict(phase='paused',step=step));return
        torch.cuda.synchronize();elapsed=elapsed0+time.perf_counter()-begin
        after=validate(rt,ema,heldout);assert state_sha(rt.model)==strong_hash and all(p.grad is None for p in rt.model.parameters())
        torch.save(dict(ema={k:v.cpu() for k,v in ema.state_dict().items()},method=method,step=c.STEPS,request_sha256=request_hash),out/'model.pt')
        atomic(out/'input_fingerprints.json',fingerprints)
        done=dict(passed=True,method=method,step=c.STEPS,training_seconds=elapsed,trainable_parameters=sum(p.numel() for p in head.parameters()),
            initial_head_sha256=initial_hash,validation_before=before,validation_after=after,strong_unchanged=True,strong_gradients_absent=True,
            strong_state_sha256=strong_hash,checkpoint_sha256=sha(out/'model.pt'),request_sha256=request_hash,
            input_fingerprints_sha256=sha(out/'input_fingerprints.json'),selected_by_validation=False,teacher_supervision=False)
        atomic(out/'complete.json',done);atomic(out/'status.json',dict(phase='complete',**done));print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',choices=c.METHODS,required=True);train(p.parse_args().train)
