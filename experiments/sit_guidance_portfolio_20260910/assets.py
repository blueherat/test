"""Training-only small linear fits; no generated-image metric enters fitting."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments import small_sit_predictable_gap_20260909 as old
from experiments.lifting_scale_sweep_20260909 import EXPS,WORK,atomic,read,sha

ROOT=EXPS/'sit_guidance_portfolio_20260910'
PYTHON='/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
HEAD_ROOT=EXPS.parent/'imagenet_sit_flow/multiscale_guidance_study_v1/runs'
EXTRA_HEADS={f'depth{depth}_v':HEAD_ROOT/f'depth{depth}_v/checkpoints/step_00050000.pt' for depth in (6,10)}


def ridge(x,y,penalty=.001):
    g=x.T@x/len(x);b=x.T@y/len(x)
    p=torch.eye(x.shape[1],dtype=x.dtype,device=x.device)*penalty;p[-1,-1]=0
    coefficient=torch.linalg.solve(g+p,b)
    error=torch.linalg.vector_norm((g+p)@coefficient-b)/torch.linalg.vector_norm(b).clamp_min(1e-12)
    assert float(error)<1e-9
    return coefficient,g+p,float(error)


def covariance(x):
    x=x-x.mean(0)
    return x.T@x/len(x)


def matrix_power(x,p,floor=1e-5):
    v,q=torch.linalg.eigh((x+x.T)/2)
    assert v.min()>-1e-8
    scale=v.mean().clamp_min(1e-12)
    return (q*v.clamp_min(floor*scale).pow(p))@q.T


def symmetric_regression(z,e,regularization=.001):
    zm,em=z.mean(0),e.mean(0);zc,ec=z-zm,e-em
    sigma=zc.T@zc/len(z)
    cross=zc.T@ec/len(z)
    vals,vec=torch.linalg.eigh((sigma+sigma.T)/2)
    rhs=vec.T@(cross+cross.T)@vec
    a=vec@(rhs/(vals[:,None]+vals[None,:]+2*regularization))@vec.T
    b=em-zm@a
    residual=sigma@a+a@sigma+2*regularization*a-cross-cross.T
    error=float(residual.abs().max())
    assert error<1e-9 and float((a-a.T).abs().max())<1e-10
    return a,b,error


def prepare():
    old.verify_fit();old.cache_source.verify_training()
    ROOT.mkdir(parents=True,exist_ok=True)
    output=ROOT/'assets';output.mkdir(exist_ok=True)
    assert not (output/'fitted.pt').exists(), 'Existing fits are immutable; use a new run directory for a revision'
    torch.set_num_threads(8)
    sources={str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    data_files={str(old.CACHE_ROOT/f'{split}_cache.pt'):sha(old.CACHE_ROOT/f'{split}_cache.pt') for split in ('train','validation')}
    for name in ('train_indices.npy','validation_indices.npy','training_request.json','collection.json'):
        path=old.CACHE_ROOT/name;data_files[str(path)]=sha(path)
    assets={str(path):sha(path) for path in (old.ROOT/'projection.pt',*EXTRA_HEADS.values())}
    request=dict(sources=sources,data_files=data_files,existing_assets=assets,penalty=.001,
        training_only=True,fid_used=False,time_bins=[2,4,8],conservative_time_bins=4,
        parameters_selected_on_validation=False)
    atomic(output/'fit_request.json',request)
    begin=time.perf_counter()
    data=torch.load(old.CACHE_ROOT/'train_cache.pt',map_location='cpu',weights_only=True)
    projection=torch.load(old.ROOT/'projection.pt',map_location='cpu',weights_only=True)
    h=data['hw'].double()
    norm=(h-projection['mean'].double())/projection['std'].double()
    x=torch.cat((norm,torch.ones((len(norm),1),dtype=torch.float64)),1)
    d=(data['s']-data['w']).double()
    c=x@projection['coefficient'].double()
    residual=d-c
    target=(data['target']-data['s']).double()
    coefficient,regularized,normal_error=ridge(x,target)
    inv_gram=torch.linalg.inv(regularized)
    leverage=((x@inv_gram)*x).sum(1)
    assert leverage.min()>=0
    sd,sc,sr=map(covariance,(d,c,residual))
    whitening={}
    for penalty in (.1,.5,2.):
        regular=sr+penalty*sr.trace()/16*torch.eye(16,dtype=torch.float64)
        whitening[str(penalty)]=matrix_power(regular,-.5).float()
    root,invroot=matrix_power(sd,.5),matrix_power(sd,-.5)
    explained=invroot@sc@invroot
    eigenfilter={str(penalty):(root@torch.linalg.solve(torch.eye(16,dtype=torch.float64)+penalty*explained,
        invroot)).float() for penalty in (.5,1.,2.)}
    time_models={};fit_errors=[normal_error];counts={}
    times=data['time'][:,0].double()
    for bins in (2,4,8):
        chunks=[];counts[str(bins)]=[]
        indices=(times*2*bins).long().clamp(0,bins-1)
        for k in range(bins):
            mask=indices==k
            coef,_,error=ridge(x[mask],d[mask]);chunks.append(coef.float())
            fit_errors.append(error);counts[str(bins)].append(int(mask.sum()))
        time_models[str(bins)]=torch.stack(chunks)
    conservative_a,conservative_b=[],[]
    indices=(times*8).long().clamp(0,3)
    for k in range(4):
        mask=indices==k
        a,b,error=symmetric_regression(data['z'][mask].double(),target[mask])
        conservative_a.append(a.float());conservative_b.append(b.float());fit_errors.append(error)
    state=dict(mean=projection['mean'],std=projection['std'],original_coefficient=projection['coefficient'],
        target_coefficient=coefficient.float(),inverse_gram=inv_gram.float(),leverage_median=float(leverage.median()),
        whitening=whitening,eigenfilter=eigenfilter,time_models=time_models,
        conservative_a=torch.stack(conservative_a),conservative_b=torch.stack(conservative_b),
        gap_covariance=sd.float(),predictable_covariance=sc.float(),residual_covariance=sr.float(),
        fit_request_sha256=sha(output/'fit_request.json'))
    torch.save(state,output/'fitted.pt')
    del h,norm,x,c,coefficient,residual,target,data
    validation=[]
    for split in ('train','validation'):
        data=torch.load(old.CACHE_ROOT/f'{split}_cache.pt',map_location='cpu',weights_only=True)
        x=torch.cat(((data['hw'].double()-state['mean'].double())/state['std'].double(),
            torch.ones((len(data['hw']),1),dtype=torch.float64)),1)
        d=(data['s']-data['w']).double();e=(data['target']-data['s']).double()
        row=dict(split=split,tokens=len(x),native_strong_mse=float(e.square().mean()),
            target_reader_mse=float((e-x@state['target_coefficient'].double()).square().mean()),
            native_gap_mse=float(d.square().mean()),
            original_residual_mse=float((d-x@state['original_coefficient'].double()).square().mean()))
        for bins in (2,4,8):
            pred=torch.empty_like(d);idx=(data['time'][:,0]*2*bins).long().clamp(0,bins-1)
            for k in range(bins):
                mask=idx==k;pred[mask]=x[mask]@state['time_models'][str(bins)][k].double()
            row[f'time{bins}_residual_mse']=float((d-pred).square().mean())
        validation.append(row)
        del data,x,d,e
    result=dict(complete=True,fit_request_sha256=sha(output/'fit_request.json'),
        fitted_sha256=sha(output/'fitted.pt'),fitting_and_diagnostic_seconds=time.perf_counter()-begin,
        maximum_normal_equation_error=max(fit_errors),time_bin_counts=counts,validation=validation,
        no_new_backbone_training=True,no_fid_used=True,training_sources_unchanged=True)
    atomic(output/'fit.json',result)
    print(json.dumps(result),flush=True)


def verify():
    directory=ROOT/'assets';request=read(directory/'fit_request.json');result=read(directory/'fit.json')
    for category in ('sources','data_files','existing_assets'):
        for path,digest in request[category].items():assert sha(path)==digest,path
    assert result['complete'] and result['fit_request_sha256']==sha(directory/'fit_request.json')
    assert result['fitted_sha256']==sha(directory/'fitted.pt')
    return request,result


if __name__=='__main__':prepare()
