"""Frozen SiT training cache with exact gate-dependent quadratic statistics."""
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.audit_sit_subspace_gate import load_official_sit_module,DEFAULT_OFFICIAL_SIT_REPO,create_dual_output_sit,PROTOCOL,sample_sdvae_posterior,dual_output_velocities,sha256_file


def quadratic(ex,ee):
    delta=(ex-ee).double();ee=ee.double()
    return torch.stack([delta.square().sum(1),(delta*ee).sum(1),ee.square().sum(1)],dim=1)


@torch.no_grad()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['protocol']==PROTOCOL
    module,meta=load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    assert ck['official_sit']==meta
    cfg=ck['config'];model=create_dual_output_sit(module,model_name=cfg['model_name'],cfg_dropout=cfg['cfg_dropout'])
    model.load_state_dict(ck['ema'],strict=True);model.to(a.device).eval().requires_grad_(False)
    del ck;gc.collect()
    basis=torch.load(a.basis,map_location='cpu',weights_only=False);p=basis['basis'][:,:32].to(a.device)
    data=np.load(a.cache/'train_moments.npy',mmap_mode='r');labels=np.load(a.cache/'train_labels.npy',mmap_mode='r')
    prior=np.load(a.previous_audit/'t0.05.npz')['indices']
    used=np.concatenate([basis['fit_indices'],basis['holdout_indices'],prior])
    available=np.setdiff1d(np.arange(len(data)),used)
    count=a.fit_count+a.holdout_count
    indices=np.random.default_rng(202609311).choice(available,count,replace=False)
    gen=torch.Generator(device=a.device).manual_seed(202609311)
    zcache=np.lib.format.open_memmap(a.output/'states.npy',mode='w+',dtype='float32',shape=(count,4096))
    qcache=np.lib.format.open_memmap(a.output/'quadratics.npy',mode='w+',dtype='float64',shape=(count,2,3))
    ncache=np.lib.format.open_memmap(a.output/'native_risk.npy',mode='w+',dtype='float64',shape=(count,))
    times=[];ys=[]
    for start in range(0,count,32):
        ii=indices[start:start+32];n=len(ii)
        moments=torch.from_numpy(np.array(data[ii])).to(a.device)
        y=torch.from_numpy(np.array(labels[ii])).long().to(a.device)
        clean=sample_sdvae_posterior(moments,torch.randn((n,4,32,32),generator=gen,device=a.device))
        noise=torch.randn(clean.shape,generator=gen,device=a.device)
        t=.001+.998*torch.rand(n,generator=gen,device=a.device)
        z=(1-t[:,None,None,None])*noise+t[:,None,None,None]*clean
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(z,t,y)
        fields=dual_output_velocities(out,state=z,time_value=t,gate_activation=cfg['gate_activation'],denominator_floor=cfg['denominator_floor'])
        ex=(fields['x']-(clean-noise)).flatten(1);ee=(fields['epsilon']-(clean-noise)).flatten(1)
        full=quadratic(ex,ee);top=quadratic(ex@p,ee@p)
        q=torch.stack([top,full-top],dim=1)
        if not torch.isfinite(q).all():raise FloatingPointError('quadratic cache')
        zcache[start:start+n]=z.flatten(1).cpu().numpy();qcache[start:start+n]=q.cpu().numpy()
        ncache[start:start+n]=(fields['dynamic']-(clean-noise)).double().flatten(1).square().sum(1).cpu().numpy()
        times.append(t.cpu());ys.append(y.cpu())
        if start%2048==0:print(json.dumps({'cached':start+n,'total':count}),flush=True)
    zcache.flush();qcache.flush();ncache.flush()
    np.save(a.output/'times.npy',torch.cat(times).numpy());np.save(a.output/'labels.npy',torch.cat(ys).numpy());np.save(a.output/'indices.npy',indices)
    torch.save({'mean':basis['mean'],'basis':basis['basis'][:,:32]},a.output/'basis.pt')
    manifest={'complete':True,'fit_count':a.fit_count,'holdout_count':a.holdout_count,'seed':202609311,
        'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256_file(a.checkpoint),'basis_sha256':sha256_file(a.basis),
        'source_sha256':sha256_file(Path(__file__)),'source_indices_disjoint_from_prior':not bool(np.isin(indices,used).any())}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_dual-output_seed0/checkpoints/step_00450000.pt'))
    p.add_argument('--cache',type=Path,default=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae'))
    p.add_argument('--basis',type=Path,default=Path('/home/zhoushunyu/data/eqvae/experiments/subspace_gate_sdvae_20260908/geometry/basis.pt'))
    p.add_argument('--previous-audit',type=Path,default=Path('/home/zhoushunyu/data/eqvae/experiments/subspace_gate_sdvae_20260908/local_risk'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-count',type=int,default=32768);p.add_argument('--holdout-count',type=int,default=4096)
    p.add_argument('--device',default='cuda:2');run(p.parse_args())
