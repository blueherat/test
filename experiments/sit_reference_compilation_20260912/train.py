from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import time
import numpy as np
import torch
from . import catalog as c, data, models
from experiments.sit_measure_guidance_20260912 import data as real_data
from experiments.sit_strong_reference_20260912.train import state_sha
from experiments.lifting_scale_sweep_20260909 import Runtime, atomic, read, sha


def prepare():
    data.verify();receipt=read(c.ROOT/'data/complete.json');assert receipt['passed']
    if (c.ROOT/'training_request.json').exists():return verify()
    request=dict(data.verify());request['input_files']=dict(request['input_files'])
    request['input_files'].update(receipt['files'])
    request.update(methods=c.METHODS,steps=c.STEPS,batch=c.BATCH,learning_rate=c.LR,ema=c.EMA,
        seed=c.TRAIN_SEED,strong_frozen=True,selected_by_validation=False)
    atomic(c.ROOT/'training_request.json',request);return request


def verify():
    request=read(c.ROOT/'training_request.json')
    for key in ('sources','assets','input_files','old_stop_markers'):
        for path,digest in request[key].items():assert sha(path)==digest,(key,path)
    return request


class Rollouts:
    def __init__(self,depth):
        keys=['index','f'+str(depth),'context','teacher','strong']
        rows={k:[] for k in keys}
        for rank in range(4):
            record=torch.load(c.ROOT/f'data/rollout_rank{rank}.pt',map_location='cpu',weights_only=False)
            for k in keys:rows[k].append(record[k])
            del record
        rows={k:torch.cat(v).cuda() for k,v in rows.items()}
        self.train={k:v[rows['index']<c.TRAIN_TRAJECTORIES] for k,v in rows.items()}
        self.valid={k:v[rows['index']>=c.TRAIN_TRAJECTORIES] for k,v in rows.items()}
        self.feature='f'+str(depth)


@torch.no_grad()
def validate_ig(rt,head,pool):
    sums=np.zeros(3);count=0
    for start in range(0,len(pool.valid['index']),32):
        row={k:v[start:start+32] for k,v in pool.valid.items()}
        prediction=models.unpatchify(rt,head(row[pool.feature],row['context']))
        target=row['teacher'];strong=row['strong']
        sums+=np.array([float((prediction.double()-target.double()).square().sum()),
            float((strong.double()-target.double()).square().sum()),float(target.double().square().sum())])
        count+=target.numel()
    return dict(states=len(pool.valid['index']),teacher_mse=sums[0]/count,
        teacher_gap_mse=sums[1]/count,error_to_gap_energy=sums[0]/sums[1],teacher_energy=sums[2]/count)


def nearest(patches,centers):
    x=patches.reshape(-1,16)
    distance=x.square().sum(1,keepdim=True)+centers.square().sum(1)[None]-2*x@centers.T
    return distance.argmin(1).reshape(patches.shape[:2])


@torch.no_grad()
def cfg_batch(rt,pool,generator,count):
    clean,labels=pool.draw('native',generator,count)
    eps=torch.randn(clean.shape,device='cuda',generator=generator)
    times=.01+.74*torch.rand(count,device='cuda',generator=generator)
    drop=torch.rand(count,device='cuda',generator=generator)<.5
    labels=torch.where(drop,100,labels)
    z=times[:,None,None,None]*clean+(1-times[:,None,None,None])*eps
    with models.Capture(rt,depths=(12,)) as capture:
        rt.model(z,times,labels)
        f=capture.values[12];context=capture.context
    return f,context,models.patchify(clean),drop


@torch.no_grad()
def validate_cfg(rt,head,pool,centers):
    generator=torch.Generator(device='cuda').manual_seed(c.TRAIN_SEED+100)
    nll,correct,mse,quant,number,elements=0.,0,0.,0.,0,0
    for _ in range(25):
        f,context,target,_=cfg_batch(rt,pool,generator,32)
        ids=nearest(target,centers);logits=head(f,context);probs=logits.softmax(-1)
        nll+=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),ids.flatten(),reduction='sum'))
        correct+=int((logits.argmax(-1)==ids).sum());number+=ids.numel()
        mse+=float((probs@centers-target).double().square().sum())
        quant+=float((centers[ids]-target).double().square().sum());elements+=target.numel()
    return dict(images=800,nll=nll/number,accuracy=correct/number,mean_mse=mse/elements,
        codebook_mse=quant/elements)


def train(method):
    verify();out=c.ROOT/'training'/method;out.mkdir(parents=True,exist_ok=True)
    with (out/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            receipt=read(out/'complete.json');assert receipt['checkpoint_sha256']==sha(out/'model.pt');return
        rt=Runtime('sit_small');torch.set_num_threads(4);torch.manual_seed(c.TRAIN_SEED)
        strong_hash=state_sha(rt.model);head=models.make_head(rt,method,c.CODEWORDS).requires_grad_(True)
        ema=copy.deepcopy(head).requires_grad_(False)
        is_ig=method.startswith('ig_')
        if is_ig:
            pool=Rollouts(4 if method=='ig_shallow' else 12)
            validate=lambda:validate_ig(rt,ema,pool)
        else:
            pool,heldout=real_data.Pool('train'),real_data.Pool('validation')
            centers=torch.load(c.ROOT/'data/codebook.pt',weights_only=False)['centers'].cuda()
            validate=lambda:validate_cfg(rt,ema,heldout,centers)
        before=validate()
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
            if is_ig:
                ids=torch.randint(len(pool.train['index']),(c.BATCH,),device='cuda',generator=generator)
                f=pool.train[pool.feature][ids];context=pool.train['context'][ids];target=pool.train['teacher'][ids]
                value=models.unpatchify(rt,head(f,context));loss=(value-target).square().mean()
                fingerprint=ids
            else:
                f,context,target,_=cfg_batch(rt,pool,generator,c.BATCH)
                ids=nearest(target,centers);value=head(f,context)
                loss=torch.nn.functional.cross_entropy(value.flatten(0,1),ids.flatten())
                fingerprint=generator.get_state()
            assert torch.isfinite(loss) and not f.requires_grad
            if step==1 or step%250==0:
                fingerprints.append(dict(step=step,sha256=hashlib.sha256(fingerprint.cpu().numpy().tobytes()).hexdigest()))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),head.parameters()):ep.lerp_(p,1-c.EMA)
            losses.append(float(loss.detach()))
            if step==1 or step%100==0:
                torch.cuda.synchronize();elapsed=elapsed0+time.perf_counter()-begin
                status=dict(phase='training',method=method,step=step,total=c.STEPS,loss=float(np.mean(losses[-100:])),
                    gradient_norm=float(norm),elapsed_seconds=elapsed)
                atomic(out/'status.json',status);print(status,flush=True)
            stop=(c.ROOT/'STOP_AFTER_CURRENT').exists()
            if step%500==0 or stop:
                torch.cuda.synchronize();elapsed=elapsed0+time.perf_counter()-begin
                temp=out/'resume.tmp.pt';torch.save(dict(head=head.state_dict(),ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(),generator=generator.get_state(),step=step,
                    elapsed=elapsed,fingerprints=fingerprints,request_sha256=request_hash),temp);temp.replace(out/'resume.pt')
            if stop:atomic(out/'status.json',dict(phase='paused',step=step));return
        torch.cuda.synchronize();elapsed=elapsed0+time.perf_counter()-begin
        after=validate();assert state_sha(rt.model)==strong_hash and all(p.grad is None for p in rt.model.parameters())
        torch.save(dict(ema={k:v.cpu() for k,v in ema.state_dict().items()},method=method,step=c.STEPS,
            request_sha256=request_hash),out/'model.pt')
        atomic(out/'input_fingerprints.json',fingerprints)
        done=dict(passed=True,method=method,step=c.STEPS,training_seconds=elapsed,
            trainable_parameters=sum(p.numel() for p in head.parameters()),validation_before=before,validation_after=after,
            strong_unchanged=True,strong_gradients_absent=True,strong_state_sha256=strong_hash,
            checkpoint_sha256=sha(out/'model.pt'),request_sha256=request_hash,
            input_fingerprints_sha256=sha(out/'input_fingerprints.json'),selected_by_validation=False)
        atomic(out/'complete.json',done);atomic(out/'status.json',dict(phase='complete',**done));print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',choices=c.METHODS,required=True)
    train(parser.parse_args().train)
