"""Train-only holdout check of linear subspace concentration in real SD-VAE latents."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


@torch.no_grad()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.manual_seed(202609221)
    torch.backends.cuda.matmul.allow_tf32=False
    data=np.load(a.cache/'train_moments.npy',mmap_mode='r')
    indices=np.random.default_rng(202609221).choice(len(data),8192,replace=False)
    fit=torch.from_numpy(np.array(data[indices[:4096],:4])).to(a.device).flatten(1)*.18215
    hold=torch.from_numpy(np.array(data[indices[4096:],:4])).to(a.device).flatten(1)*.18215
    mean=fit.mean(0)
    fit,hold=fit-mean,hold-mean
    _,_,p=torch.pca_lowrank(fit,q=256,center=False,niter=5)
    rand=torch.linalg.qr(torch.randn(fit.shape[1],256,device=a.device))[0]
    rows=[]
    for rank in [8,32,128,256]:
        row={'rank':rank,'dimension':fit.shape[1]}
        for name,x in [('fit',fit),('holdout',hold)]:
            total=x.square().sum()
            row[name+'_pca_energy_fraction']=float((x@p[:,:rank]).square().sum()/total)
            row[name+'_random_energy_fraction']=float((x@rand[:,:rank]).square().sum()/total)
        rows.append(row)
    torch.save({'mean':mean.cpu(),'basis':p.cpu(),'fit_indices':indices[:4096],
                'holdout_indices':indices[4096:]},a.output/'basis.pt')
    result={'rows':rows,'source':str(a.cache/'train_moments.npy'),
        'note':'posterior means only; train-only disjoint split; energy concentration not generation quality'}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:3')
    run(p.parse_args())
