"""Bounded distribution test: error-only Gaussian convolution and unfitted residuals."""
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from experiments.guidance_pasted_20260912 import common as c
from . import mixture as m

PROTOCOL=c.WORK/'docs/STRUCTURED_MISMATCH_ENDPOINT_PROTOCOL_20260912_ZH.md'
ROOT=c.EXPS/'structured_mismatch_20260912'
SEED=2026121371


def project(model):
    root=ROOT/model;root.mkdir(parents=True,exist_ok=True)
    dest=root/'projections.npz'
    if dest.exists():
        req=c.read(root/'projection_manifest.json')
        assert c.sha(dest)==req['output_sha256']
        assert c.sha(Path(__file__))==req['source_sha256']
        return dest
    dim=4096 if model=='sit_small' else 262144
    rng=np.random.default_rng(SEED)
    directions=(2*rng.integers(0,2,size=(dim,16),dtype=np.int8)-1).astype(np.float32)/np.sqrt(dim)
    directions=directions.astype(np.float32)
    matrix=torch.from_numpy(directions).cuda()
    labels=np.load(m.ROOT/model/'endpoint_sources/inputs/labels.npy')
    out={};records=[];started=time.perf_counter()
    def apply(x):
        z=torch.from_numpy(np.asarray(x,dtype=np.float32).copy()).cuda()
        return (z.flatten(1)@matrix).cpu().numpy()
    for source in ('strong','weak','null'):
        values=[];all_labels=[]
        for p in sorted((m.ROOT/model/'endpoint_sources'/source/'batches').glob('*.npz')):
            with np.load(p) as d:
                values.append(apply(d['latents']));all_labels.append(d['labels'])
            records.append(dict(path=str(p),sha256=c.sha(p)))
        out[source]=np.concatenate(values)
        np.testing.assert_array_equal(np.concatenate(all_labels),labels)
        print(model,source,'projected',flush=True)
    actual=[];indices=[]
    if model=='sit_small':
        base=c.EXPS/'sit_measure_guidance_20260912'
        for split,subset in (('train',labels[:500]),('validation',labels[500:])):
            p=base/(split+'_moments.npy');bank=np.load(p,mmap_mode='r')
            idx=rng.integers(0,bank.shape[1],size=500);indices.extend(idx.tolist())
            v=[]
            for start in range(0,500,20):
                mm=np.asarray(bank[subset[start:start+20],idx[start:start+20]],dtype=np.float32)
                noise=rng.standard_normal(mm[:,:4].shape).astype(np.float32)
                v.append(apply((mm[:,:4]+mm[:,4:]*noise)*.18215))
            actual.append(np.concatenate(v));records.append(dict(path=str(p),sha256=c.sha(p)))
    else:
        base=c.EXPS/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'
        summary=c.read(base/'summary.json')
        for split,subset in (('train',labels[:500]),('validation',labels[500:])):
            p=base/split/'latents.npy';mp=base/split/'metadata.npz'
            assert c.sha(p)==summary['banks'][split]['latents']['sha256']
            assert c.sha(mp)==summary['banks'][split]['metadata']['sha256']
            bank=np.load(p,mmap_mode='r');meta=np.load(mp)
            choices={int(l):np.where(meta['labels']==l)[0] for l in np.unique(meta['labels'])}
            idx=np.asarray([rng.choice(choices[int(l)]) for l in subset]);indices.extend(idx.tolist())
            actual.append(np.concatenate([apply(bank[idx[start:start+20]]) for start in range(0,500,20)]))
            records.extend([dict(path=str(p),sha256=c.sha(p)),dict(path=str(mp),sha256=c.sha(mp))])
    out['real']=np.concatenate(actual)
    np.savez(dest,**out,directions=directions,labels=labels,real_indices=np.asarray(indices))
    c.atomic(root/'projection_manifest.json',dict(model=model,complete=True,records=records,
        seed=SEED,source_sha256=c.sha(Path(__file__)),protocol_sha256=c.sha(PROTOCOL),
        output_sha256=c.sha(dest),seconds=time.perf_counter()-started,
        extra_generation_paths=0,extra_inference_networks=0))
    return dest


def features(x,center,std,signs):
    phase=((x-center)/std)[:,:,None]*np.asarray([.5,1.,2.])[None,None,:]
    base=np.exp(1j*phase)
    return np.stack((base,base*signs[:,:,None]),axis=-1)


def residual(fs,fw,fp,theta,frequency2):
    k,b,tau=theta
    g=np.exp(-.5*tau*frequency2)[None,:,:,None]
    return fw-g/k*fs-((1-b)-(1/k-b)*g)*fp


def fit(fs,fw,fp,frequency2,tau_scale):
    mean=[v[:,:8].mean(0,keepdims=True) for v in (fs,fw,fp)]
    def objective(p,strict=False):
        theta=[p[0],.5,0] if strict else [p[0],p[1],p[2]*tau_scale]
        r=residual(*mean,theta,frequency2[:8]).ravel()
        return np.r_[r.real,r.imag]
    strict=least_squares(lambda p:objective(p,True),[.5],bounds=([.02],[.98]),
        ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=1000)
    records=[];best=None
    for initial in ([strict.x[0],.5,0.000001],[strict.x[0],.5,.1],[.8,.8,1.]):
        r=least_squares(objective,initial,bounds=([.02,.02,0.],[.98,1.,4.]),
            ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=2000)
        records.append(dict(initial=initial,theta_normalized=r.x.tolist(),cost=float(r.cost),
            success=bool(r.success),message=str(r.message),nfev=r.nfev))
        if best is None or r.cost<best.cost:best=r
    return np.array([strict.x[0],.5,0.]),np.array([best.x[0],best.x[1],best.x[2]*tau_scale]),records


def analyze(model,path):
    root=ROOT/model
    with np.load(path) as z:data={k:z[k] for k in z.files if k!='directions'}
    center=data['real'][:500].mean(0);std=data['real'][:500].std(0).clip(1e-6)
    rng=np.random.default_rng(SEED+1)
    class_signs=2*rng.integers(0,2,size=(100 if model=='sit_small' else 1000,16))-1
    signs=class_signs[data['labels']]
    f={k:features(data[k],center,std,signs) for k in ('real','strong','weak','null')}
    frequency2=(np.asarray([.5,1.,2.])[None,:]/std[:,None])**2
    tau_scale=float(np.median(std**2));rows=[];fits=[];arrays={}
    for source,track in (('weak','ig'),('null','cfg')):
        train=[f[k][:500] for k in ('strong',source,'real')]
        old,new,optimization=fit(*train,frequency2,tau_scale)
        arrays[track+'_strict_parameters']=old;arrays[track+'_convolution_parameters']=new
        fits.append(dict(model=model,track=track,strict_kappa=float(old[0]),kappa=float(new[0]),
            b=float(new[1]),a=float(new[0]*new[1]),tau=float(new[2]),
            tau_normalized=float(new[2]/tau_scale),optimization=optimization,
            boundary=bool(new[0]<.02001 or new[0]>.97999 or new[1]<.02001 or new[1]>.99999 or new[2]/tau_scale>3.99999)))
        test=[f[k][500:] for k in ('strong',source,'real')]
        ro=residual(*test,old,frequency2);rn=residual(*test,new,frequency2)
        for name,sl in (('same_directions',slice(0,8)),('new_directions',slice(8,16))):
            a=ro[:,sl].reshape(500,-1);b=rn[:,sl].reshape(500,-1)
            mse=lambda r:float(np.abs(r.mean(0)).dot(np.abs(r.mean(0)))/r.shape[1])
            before,after=mse(a),mse(b);boot=[]
            for _ in range(300):
                ids=rng.integers(0,500,500);boot.append(mse(b[ids])-mse(a[ids]))
            arrays[track+'_'+name+'_strict_residuals']=a
            arrays[track+'_'+name+'_convolution_residuals']=b
            arrays[track+'_'+name+'_bootstrap_delta']=np.array(boot)
            rows.append(dict(model=model,track=track,heldout=name,strict_residual_mse=before,
                convolution_residual_mse=after,delta=after-before,
                descriptive_2p5_97p5=np.quantile(boot,[.025,.975]).tolist()))
    np.savez(root/'analysis_arrays.npz',**arrays,center=center,std=std,frequency2=frequency2,class_signs=class_signs)
    c.atomic(root/'results.json',dict(complete=True,model=model,rows=rows,fits=fits,
        protocol_sha256=c.sha(PROTOCOL),source_sha256=c.sha(Path(__file__)),
        projection_sha256=c.sha(path),arrays_sha256=c.sha(root/'analysis_arrays.npz'),
        interpretation='Finite joint/marginal CF necessary relations only; no density identification or generation claim.'))
    print(model,rows,'parameters',fits,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    for model in ('sit_small','raev2'):analyze(model,project(model))
