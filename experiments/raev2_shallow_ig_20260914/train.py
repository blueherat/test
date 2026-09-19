"""Train two fresh depth4 readouts on identical noisy real-latent batches."""
import copy,time
from pathlib import Path
import numpy as np
import torch
from experiments.raev2_shallow_ig_20260914 import core as m
from experiments.raev2_context_20k_20260914.train import save_torch
c=m.c
STEPS=50000
SEED=2026091431


def prepare():
    sources=c.source_manifest([Path(__file__),Path(m.__file__),Path(m.old.__file__),
        c.WORK/'experiments/raev2_context_20k_20260914/train.py',
        c.WORK/'experiments/train_raev2_observable_potential.py',
        c.WORK/'external/RAEv2/src/stage2/models/DDT.py',
        c.WORK/'external/RAEv2/src/stage2/models/model_utils.py',
        c.WORK/'docs/RAEV2_SHALLOW_IG_50K_PROTOCOL_20260914_ZH.md'])
    req=dict(model='raev2',depth=4,steps=STEPS,batch=8,seed=SEED,lr=.0003,weight_decay=.0001,ema=.995,
        time_distribution='Uniform(.01,.99)',target='real clean latent MSE',normalization_batches=32,
        head_architectures={'native':'exact DDTFinalLayer(1440,1,1024), original zero initialization and norm ones',
            'mlp':'repository Context Head(1440,1024,16), train-only feature standardization'},
        initialization='Both fresh, not copied fitted depth8 weights; same data/noise/time sequence',
        validation='1000 disjoint real cases, fixed noise/time; monitor only, final step50K EMA is selected',
        sources=sources,assets={str(p):c.sha(p) for p in c.asset_paths('raev2')},
        data={str(p):c.sha(p) for p in m.old.data_paths('raev2')})
    path=m.TRAIN/'request.json'
    if path.exists():assert c.read(path)==req
    else:c.atomic(path,req)
    return path


def batch(rt,data,g,split='train'):
    clean,y=data.draw(split,g,8);t=.01+.98*torch.rand(8,device='cuda',generator=g)
    eps=torch.randn(clean.shape,device='cuda',generator=g);a=t[:,None,None,None]
    with rt.context():f=m.features(rt,(1-a)*clean+a*eps,t,y)
    return f,m.old.patchify(rt,clean)


@torch.no_grad()
def fixed_cache(rt,data):
    from experiments.train_raev2_observable_potential import batch_from_bank
    bank=data.banks['validation'];labels=bank[1]['labels']
    ids=np.array([np.flatnonzero(labels==k)[0] for k in range(1000)],dtype=np.int64)
    g=torch.Generator(device='cuda').manual_seed(SEED+1);cache=[];times=[];hashes=[]
    for start in range(0,1000,8):
        clean,y=batch_from_bank(bank,ids[start:start+8],'cuda');t=.01+.98*torch.rand(8,device='cuda',generator=g)
        eps=torch.randn(clean.shape,device='cuda',generator=g);a=t[:,None,None,None]
        with rt.context():f=m.features(rt,(1-a)*clean+a*eps,t,y)
        cache.append(({k:v.cpu() for k,v in f.items()},m.old.patchify(rt,clean).cpu()))
        times.extend(t.cpu().tolist());hashes.append(c.array_sha(eps.cpu().numpy()))
    c.atomic(m.TRAIN/'validation_cases.json',dict(seed=SEED+1,indices=ids.tolist(),labels=labels[ids].tolist(),times=times,noise_hashes=hashes))
    return cache


@torch.no_grad()
def evaluate(rt,ema,cache,step):
    values={k:[] for k in ema}
    for feats,target in cache:
        f={k:v.cuda() for k,v in feats.items()};target=target.cuda()
        with rt.context():
            for k,h in ema.items():
                loss=(m.predict(k,h,f).float()-target).square().flatten(1).mean(1)
                values[k].extend(loss.cpu().tolist())
    assert all(len(v)==1000 and np.isfinite(v).all() for v in values.values())
    record=dict(step=step,means={k:float(np.mean(v)) for k,v in values.items()},per_case=values)
    c.atomic(m.TRAIN/'validation'/f'step{step:05d}.json',record)
    print('validation',step,record['means'],flush=True)


def main():
    path=prepare();m.old.verify_request(path)
    if (m.TRAIN/'summary.json').exists():return
    torch.manual_seed(SEED);rt=c.runtime('raev2');data=m.old.RealData('raev2');heads=m.make_heads(rt)
    c.atomic(m.TRAIN/'checks_before.json',m.check(rt,heads));strong_before=m.state_hash(rt.model)
    g=torch.Generator(device='cuda').manual_seed(SEED)
    sums={k:torch.zeros(1440,device='cuda',dtype=torch.float64) for k in ('tokens','condition')}
    squares={k:v.clone() for k,v in sums.items()};counts={k:0 for k in sums}
    for _ in range(32):
        f,_=batch(rt,data,g)
        for k in sums:
            v=f[k].reshape(-1,1440).double();sums[k]+=v.sum(0);squares[k]+=v.square().sum(0);counts[k]+=len(v)
    with torch.no_grad():
        for k,prefix in [('tokens','token'),('condition','condition')]:
            mean=sums[k]/counts[k];std=(squares[k]/counts[k]-mean.square()).clamp_min(1e-8).sqrt()
            getattr(heads['mlp'],prefix+'_mean').copy_(mean.float());getattr(heads['mlp'],prefix+'_std').copy_(std.float())
    ema={k:copy.deepcopy(h).eval().requires_grad_(False) for k,h in heads.items()}
    opts={k:torch.optim.AdamW(h.parameters(),lr=.0003,weight_decay=.0001) for k,h in heads.items()}
    history=[];first=1;elapsed=0.
    latest=m.TRAIN/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False);assert state['request_sha256']==c.sha(path)
        for k in heads:
            heads[k].load_state_dict(state['online'][k]);ema[k].load_state_dict(state['ema'][k]);opts[k].load_state_dict(state['optimizers'][k])
        g.set_state(state['data_rng']);first=state['step']+1;history=state['history'];elapsed=state['elapsed_seconds']
    cache=fixed_cache(rt,data);begin=time.perf_counter()
    if first==1:evaluate(rt,ema,cache,0)
    for step in range(first,STEPS+1):
        f,target=batch(rt,data,g);losses={}
        for k,h in heads.items():
            opts[k].zero_grad(set_to_none=True)
            with rt.context():pred=m.predict(k,h,f)
            loss=(pred.float()-target).square().mean();assert torch.isfinite(loss)
            loss.backward()
            if step==1 or step%100==0:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
            opts[k].step()
            with torch.no_grad():
                for p,q in zip(ema[k].parameters(),h.parameters()):p.lerp_(q,.005)
            losses[k]=float(loss.detach())
        history.append(dict(step=step,**losses))
        if step==1 or step%100==0:
            c.atomic(m.TRAIN/'progress.json',dict(step=step,total=STEPS,seconds=elapsed+time.perf_counter()-begin,**losses))
            print('train',step,losses,flush=True)
        if step==1000 or step%5000==0:evaluate(rt,ema,cache,step)
        if step%1000==0:
            save_torch(latest,dict(step=step,online={k:h.state_dict() for k,h in heads.items()},ema={k:h.state_dict() for k,h in ema.items()},
                optimizers={k:o.state_dict() for k,o in opts.items()},data_rng=g.get_state(),history=history,
                elapsed_seconds=elapsed+time.perf_counter()-begin,request_sha256=c.sha(path)))
    seconds=elapsed+time.perf_counter()-begin
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    strong_after=m.state_hash(rt.model);assert strong_after==strong_before
    checks=m.check(rt,ema);c.atomic(m.TRAIN/'checks_after.json',checks)
    save_torch(m.TRAIN/'head.pt',dict(step=STEPS,ema={k:h.state_dict() for k,h in ema.items()},request_sha256=c.sha(path)))
    c.atomic(m.TRAIN/'summary.json',dict(complete=True,steps=STEPS,batch=8,seed=SEED,depth=4,training_seconds=seconds,
        strong_state_sha256_before=strong_before,strong_state_sha256_after=strong_after,strong_frozen=True,
        parameters=checks['parameters'],request_sha256=c.sha(path),head_sha256=c.sha(m.TRAIN/'head.pt'),
        history=history,validation=c.read(m.TRAIN/'validation/step50000.json')['means']))
    print('Training complete',seconds,flush=True)

if __name__=='__main__':main()
