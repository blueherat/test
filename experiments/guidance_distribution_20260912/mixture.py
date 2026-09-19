"""Sample the shared-contamination inverse, with one explicit finite gain cap."""
import argparse
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c,ig

ROOT=c.EXPS/'guidance_distribution_20260912'
OLD=c.EXPS/'guidance_pasted_20260912'
PROTOCOL=c.WORK/'docs/SHARED_CONTAMINATION_SAMPLING_PROTOCOL_20260912_ZH.md'
KINDS=('inverse','time_mean','moment_constant','native_base','native_half','constant_cap2')
N=400
CAP=2.


def configure():c.ROOT=ROOT
def stage(track):return track+'_contamination_screen_400'
def source_root(model,track):
    return OLD/model/'ig_source_data' if track=='ig' else ROOT/model/'cfg_source_data'
def head_path(model,track):
    return OLD/model/'ig_calibrated_source/head.pt' if track=='ig' else source_root(model,track)/'head.pt'


def coefficient(model,track):
    path=ROOT/model/'moment_contamination/results.json'
    rows=c.read(path)['rows'];row=next(r for r in rows if r['track'].lower()==track)
    assert row['admissible_global_range']
    return row['kappa_unconstrained']


def load_head(model,track,device='cpu'):
    state=torch.load(head_path(model,track),map_location='cpu',weights_only=True)
    h=ig.Head(state['state']['mean'],state['state']['std'])
    h.load_state_dict(state['state'],strict=True)
    return h.to(device).eval().requires_grad_(False)


def gain(logit,kappa):
    log_rho=math.log(kappa)-logit
    rho=log_rho.clamp(max=math.log(CAP/(1+CAP))).exp()
    return rho/(1-rho),log_rho>=0,log_rho>=math.log(CAP/(1+CAP))


def prepare_profile(model,track):
    configure();root=ROOT/model/(track+'_contamination_profile')
    if (root/'summary.json').exists():return
    kappa=coefficient(model,track);head=load_head(model,track)
    minimum=1100 if track=='ig' else 700
    arrays={0:[],1:[]};ids={0:[],1:[]};inputs={};times=stages=None
    with torch.no_grad():
        for p in sorted((source_root(model,track)/'features').glob('*.npz')):
            with np.load(p) as d:
                keep=d['ids']>=minimum
                if not keep.any():continue
                inputs[str(p)]=c.sha(p)
                x=d['features'][keep].astype(np.float32);n,q,dim=x.shape
                if times is None:times=d['times'];stages=d['substages']
                else:
                    np.testing.assert_array_equal(times,d['times']);np.testing.assert_array_equal(stages,d['substages'])
                xx=torch.from_numpy(x.reshape(-1,dim))
                extra=torch.from_numpy(np.column_stack((np.tile(times,n),np.tile(stages,n))).astype(np.float32))
                logits=head.net(torch.cat(((xx-head.mean)/head.std,extra),1)).reshape(n,q)
                logits[:,0]=0.
                arrays[int(d['source'])].append(logits.numpy());ids[int(d['source'])].extend(d['ids'][keep])
    for label in arrays:
        order=np.argsort(ids[label])
        arrays[label]=np.concatenate(arrays[label])[order]
        ids[label]=np.asarray(ids[label])[order]
    np.testing.assert_array_equal(ids[0],ids[1]);assert len(ids[0])==100
    stats={}
    for label,name in ((1,'strong'),(0,'reference')):
        gg,invalid,capped=gain(torch.from_numpy(arrays[label]),kappa)
        stats[name]=dict(mean_gain=float(gg.mean()),invalid_probability_fraction=float(invalid.float().mean()),
                        capped_fraction=float(capped.float().mean()))
        if label==1:profile=gg.mean(0).numpy()
    # An event-frequency test remains meaningful even for a miscalibrated classifier.
    ev_s=(arrays[1]<math.log(kappa)).astype(float)
    ev_w=(arrays[0]<math.log(kappa)).astype(float)
    per_seed=(ev_s-kappa*ev_w).mean(1)
    rng=np.random.default_rng(2026121337)
    boot=per_seed[rng.integers(0,len(per_seed),(2000,len(per_seed)))].mean(1)
    root.mkdir(parents=True,exist_ok=True)
    np.savez(root/'profile.npz',gain=profile,times=times,substages=stages,ids=ids[0],
        strong_logits=arrays[1],reference_logits=arrays[0],signed_event_per_seed=per_seed)
    c.atomic(root/'summary.json',dict(complete=True,model=model,track=track,kappa=kappa,
        cap=CAP,source_audit_seeds=100,stats=stats,signed_event_mean=float(per_seed.mean()),
        signed_event_resampling_2p5_97p5=np.quantile(boot,[.025,.975]).tolist(),
        event_frequency_does_not_assume_classifier_calibration=True,
        initial_log_ratio_exact_zero=True,inputs=inputs,head_sha256=c.sha(head_path(model,track)),
        profile_sha256=c.sha(root/'profile.npz')))
    print(model,track,'profile',stats,'signed event',float(per_seed.mean()),flush=True)


def prepare(model,track):
    configure();prepare_profile(model,track)
    kappa=coefficient(model,track)
    bank=c.prepare_bank(model,stage(track),N,2026121331 if track=='ig' else 2026121341)
    hp=head_path(model,track);pp=ROOT/model/(track+'_contamination_profile')/'profile.npz'
    extra=[Path(__file__).resolve(),PROTOCOL,c.WORK/'experiments/guidance_pasted_20260912/ig.py']
    request=dict(model=model,track=track,samples=N,kappa=kappa,cap=CAP,kinds=list(KINDS),
        sources=c.source_manifest(extra),assets={str(p):c.sha(p) for p in c.asset_paths(model)},
        inputs={str(p):c.sha(p) for p in bank.glob('*.npy')},
        fitted_artifacts={str(p):c.sha(p) for p in (hp,pp,ROOT/model/'moment_contamination/results.json')},
        initial_log_ratio_zero=True,extra_prefix_at_deployment=False,
        candidate='min(rho,2/3)/(1-min(rho,2/3)), rho=kappa*exp(-source_logit)',
        source_ratio_is_feature_based=True,generated_paths_per_sample=1)
    rp=ROOT/model/stage(track)/'request.json'
    if rp.exists():assert c.read(rp)==request
    else:c.atomic(rp,request)
    return rp


def verify(path):
    request=c.read(path)
    for group in ('sources','assets','inputs','fitted_artifacts'):
        for p,h in request[group].items():assert c.sha(p)==h,p
    return request


def query(rt,capture,head,kind,track,kappa,profile,z,t,left,step,substage,trace):
    native=c.amount(rt,left,track)
    if not native:return rt.field(z,t,'full')
    if track=='ig':s,w=rt.pair(z,t);features=capture.value
    else:
        s=rt.field(z,t,'full');features=capture.value.clone();labels=rt.labels
        rt.labels=torch.full_like(labels,c.CLASSES[rt.name])
        try:w=rt.field(z,t,'full')
        finally:rt.labels=labels
    with torch.autocast('cuda',enabled=False):
        logits=head(features,t,substage)
    if step==0 and substage==0:logits=torch.zeros_like(logits)
    predicted,invalid,capped=gain(logits,kappa)
    if kind=='inverse':amount=predicted
    elif kind=='time_mean':
        index=2*step+substage if rt.name=='sit_small' else step
        amount=predicted.new_full(predicted.shape,float(profile[index]))
    elif kind=='moment_constant':amount=predicted.new_full(predicted.shape,min(CAP,kappa/(1-kappa)))
    elif kind=='native_base':amount=predicted.new_full(predicted.shape,native)
    elif kind=='native_half':amount=predicted.new_full(predicted.shape,native/2)
    elif kind=='constant_cap2':amount=predicted.new_full(predicted.shape,CAP)
    else:raise ValueError(kind)
    trace.append(torch.stack((logits,amount,invalid.float(),capped.float()),-1))
    return s+amount[:,None,None,None]*(s-w)


@torch.inference_mode()
def worker(model,track,rank=0,world=1,parent=0):
    configure();rp=ROOT/model/stage(track)/'request.json';request=verify(rp)
    rt=c.runtime(model);capture=ig.Capture(rt);head=load_head(model,track,'cuda')
    with np.load(ROOT/model/(track+'_contamination_profile')/'profile.npz') as d:profile=d['gain']
    kappa=request['kappa'];noise,_,labels=c.bank(model,stage(track))
    expected=dict(full=(128 if model=='sit_small' else 100)+
        (96 if model=='sit_small' else 100)*(track=='cfg'),prefix=0)
    try:
        for kind in KINDS:
            root=ROOT/model/stage(track)/kind/f'rank{rank}'
            for start in range(rank*rt.batch,N,world*rt.batch):
                c.check_parent(parent);p=root/f'batch{start:04d}.npz'
                if p.exists():
                    receipt=c.read(p.with_suffix('.json'));assert c.sha(p)==receipt['sha256']
                    assert receipt['request_sha256']==c.sha(rp);continue
                z=c.cuda(noise[start:start+rt.batch]);y=c.cuda(labels[start:start+rt.batch]);trace=[]
                torch.cuda.synchronize();begin=time.perf_counter()
                with rt.context():
                    latent,counts=c.integrate(rt,z,y,lambda x,t,left,i,j:
                        query(rt,capture,head,kind,track,kappa,profile,x,t,left,i,j,trace))
                pixels=rt.decode(latent);torch.cuda.synchronize();seconds=time.perf_counter()-begin
                assert counts==expected,(counts,expected)
                c.save_batch(p,pixels,latent.float().cpu().numpy(),np.array(y.cpu()),dict(seconds=seconds,
                    full_calls=counts['full'],prefix_calls=counts['prefix'],
                    trace=torch.stack(trace,1).cpu().numpy()),c.sha(rp),start,c.array_sha(noise[start:start+len(y)]))
                if start%(25*rt.batch)==rank*rt.batch:
                    c.atomic(ROOT/model/stage(track)/f'progress{rank}.json',dict(pid=os.getpid(),arm=kind,start=start))
            c.atomic(root/'complete.json',dict(complete=True,rank=rank,world=world,request_sha256=c.sha(rp)))
            print(model,track,rank,kind,'complete',flush=True)
    finally:capture.close()


def collect(model,track,kind):
    configure();root=ROOT/model/stage(track)/kind
    noise,_,labels=c.bank(model,stage(track));rp=ROOT/model/stage(track)/'request.json'
    files=sorted(root.glob('rank*/batch*.npz'),key=lambda p:int(p.stem[5:]))
    if sum(len(np.load(p)['arr_0']) for p in files)!=N:return False
    images=[];coverage=[];seconds=0.;records=[];traces=[]
    for p in files:
        receipt=c.read(p.with_suffix('.json'));assert c.sha(p)==receipt['sha256']
        with np.load(p) as d:
            start=int(d['start']);n=len(d['arr_0']);coverage.extend(range(start,start+n))
            assert str(d['request_sha256'])==c.sha(rp)
            assert str(d['noise_sha256'])==c.array_sha(noise[start:start+n])
            np.testing.assert_array_equal(d['labels'],labels[start:start+n])
            assert np.isfinite(d['latents']).all() and np.isfinite(d['trace']).all()
            assert int(d['prefix_calls'])==0
            images.append(d['arr_0']);seconds+=float(d['seconds']);traces.append(d['trace'])
            calls=int(d['full_calls'])
        records.append(dict(file=str(p),sha256=receipt['sha256']))
    assert coverage==list(range(N))
    trace=np.concatenate(traces)
    np.savez(root/'samples.npz',arr_0=np.concatenate(images))
    np.savez(root/'traces.npz',values=trace,columns=np.array(['source_logit','used_gain','invalid_rho','cap_triggered']))
    c.atomic(root/'summary.json',dict(complete=True,model=model,track=track,stage=stage(track),arm=kind,
        primary_samples=N,generated_paths=N,seconds=seconds,full_calls_per_output=calls,
        prefix_calls_at_inference=0,samples_sha256=c.sha(root/'samples.npz'),records=records,
        request_sha256=c.sha(rp),mean_gain=float(trace[:,:,1].mean()),
        invalid_rho_fraction=float(trace[:,:,2].mean()),cap_triggered_fraction=float(trace[:,:,3].mean()),
        traces_sha256=c.sha(root/'traces.npz')))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True)
    p.add_argument('--track',choices=('cfg','ig'),required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world',type=int,default=1);p.add_argument('--parent',type=int,default=0);a=p.parse_args()
    if a.prepare:torch.set_num_threads(4);prepare(a.model,a.track)
    elif a.worker:worker(a.model,a.track,a.rank,a.world,a.parent)
    else:p.error('choose --prepare or --worker')
