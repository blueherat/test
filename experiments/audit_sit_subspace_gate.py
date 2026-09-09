"""Train-only fit/holdout local-risk audit; not a generation-quality experiment."""
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.train_imagenet100_sit_flow import load_official_sit_module,DEFAULT_OFFICIAL_SIT_REPO,sample_sdvae_posterior,sha256_file
from experiments.train_imagenet100_sit_dual_output import create_dual_output_sit,PROTOCOL
from experiments.imagenet100_sit_dual_output import dual_output_velocities


def sums(ex,ee):
    d=(ex-ee).double(); e=ee.double()
    return torch.stack([d.square().sum(),(e*d).sum(),e.square().sum()])


@torch.no_grad()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['protocol']==PROTOCOL
    module,meta=load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO)
    assert ck['official_sit']==meta
    cfg=ck['config']
    model=create_dual_output_sit(module,model_name=cfg['model_name'],cfg_dropout=cfg['cfg_dropout'])
    model.load_state_dict(ck['ema'],strict=True)
    model.to(a.device).eval().requires_grad_(False)
    del ck;gc.collect()
    basis=torch.load(a.basis,map_location='cpu',weights_only=False)
    p=basis['basis'][:,:32].to(a.device)
    data=np.load(a.cache/'train_moments.npy',mmap_mode='r')
    labels=np.load(a.cache/'train_labels.npy',mmap_mode='r')
    used=np.concatenate([basis['fit_indices'],basis['holdout_indices']])
    available=np.setdiff1d(np.arange(len(data)),used)
    indices=np.random.default_rng(202609241).choice(available,2048,replace=False)
    gen=torch.Generator(device=a.device).manual_seed(202609241)
    rows=[]
    for t in [.05,.15,.3,.5,.7,.85,.95]:
        stats=[]
        for start in range(0,2048,32):
            ii=indices[start:start+32]
            moments=torch.from_numpy(np.array(data[ii])).to(a.device)
            y=torch.from_numpy(np.array(labels[ii])).long().to(a.device)
            clean=sample_sdvae_posterior(moments,torch.randn((32,4,32,32),generator=gen,device=a.device))
            noise=torch.randn(clean.shape,generator=gen,device=a.device)
            z=(1-t)*noise+t*clean
            times=torch.full((32,),t,device=a.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(z,times,y)
            fields=dual_output_velocities(out,state=z,time_value=times,
                gate_activation=cfg['gate_activation'],denominator_floor=cfg['denominator_floor'])
            target=clean-noise
            ex=(fields['x']-target).flatten(1)
            ee=(fields['epsilon']-target).flatten(1)
            full=sums(ex,ee)
            top=sums(ex@p,ee@p)
            native=(fields['dynamic']-target).double().square().sum()
            stats.append(torch.cat([full,top,full-top,native[None]]).cpu().numpy())
        stats=np.stack(stats)
        fit=stats[:32].sum(0);hold=stats[32:].sum(0)
        gates=[float(np.clip(-fit[i+1]/fit[i],0,1)) for i in [0,3,6]]
        def risk(offset,g):return hold[offset+2]+2*g*hold[offset+1]+g*g*hold[offset]
        denom=1024*4096
        row={'time':t,'scalar_gate':gates[0],'top32_gate':gates[1],'complement_gate':gates[2],
            'scalar_holdout_mse':risk(0,gates[0])/denom,
            'split_holdout_mse':(risk(3,gates[1])+risk(6,gates[2]))/denom,
            'native_spatial_holdout_mse':hold[-1]/denom}
        rows.append(row)
        np.savez(a.output/f't{t:.2f}.npz',batch_stats=stats,indices=indices)
        print(json.dumps(row),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'rows':rows,'complete':True,
        'checkpoint_sha256':sha256_file(a.checkpoint),'basis_sha256':sha256_file(a.basis),
        'rank':32,'note':'time-specific constant gates; fit1024 and hold1024 train-only; not FID'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_dual-output_seed0/checkpoints/step_00450000.pt'))
    p.add_argument('--cache',type=Path,default=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae'))
    p.add_argument('--basis',type=Path,default=Path('/home/zhoushunyu/data/eqvae/experiments/subspace_gate_sdvae_20260908/geometry/basis.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:3')
    run(p.parse_args())
