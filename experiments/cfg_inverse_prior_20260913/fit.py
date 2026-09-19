"""Fit small real-inverse Gaussian priors; select on heldout noise NLL, not FID.

Inverse noise correction is existing prior art. This diagonal/mean shrinkage
screen does not claim a new principle or that likelihood implies lower FID.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
from experiments.fm_common_inverse_copy_20260913.run import atomic, sha

ROOT = Path(os.environ.get('CFG_INVERSE_PRIOR_ROOT',
    '/home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913'))
KAPPAS = (0., .25, .5, 1.)
RHOS = (0., .25, .5, .75, 1.)


def losses(x, labels, means, std):
    return (.5 * ((x-means[labels])/std)**2 + np.log(std) + .5*np.log(2*np.pi)).mean(1)


def fit(args):
    root, out = args.root, args.root/'fit'
    if out.exists():
        raise FileExistsError('Fit is immutable; use a fresh experiment directory.')
    with np.load(root/'bank.npz', allow_pickle=False) as bank:
        x = bank['noise'].astype(np.float64).reshape(len(bank['noise']), -1)
        labels = bank['labels'].astype(np.int64)
        split = bank['split'].astype(np.int8)
        idkey = 'source_ids' if 'source_ids' in bank else 'source_id'
        ids = bank[idkey]
    assert len(x)==3000 and x.shape[1]==4096 and np.isfinite(x).all()
    assert len(set(ids.tolist()))==3000 and set(split.tolist())=={0,1}
    train, val = x[split==0], x[split==1]
    yt, yv = labels[split==0], labels[split==1]
    assert np.array_equal(np.bincount(yt,minlength=100),np.full(100,20))
    assert np.array_equal(np.bincount(yv,minlength=100),np.full(100,10))
    zeros=np.zeros((100,4096)); ones=np.ones(4096)
    baseline=losses(val,yv,zeros,ones)
    global_mean=train.mean(0)
    raw_means=np.stack([train[yt==c].mean(0) for c in range(100)])
    entries=[]; params=[]
    def add(family,kappa,rho,mean,std):
        loss=losses(val,yv,mean,std);delta=loss-baseline
        # Sources, not latent coordinates, are independent evaluation units.
        row=dict(family=family,kappa=kappa,rho=rho,heldout_nll_per_dim=float(loss.mean()),
            delta_nll_per_dim=float(delta.mean()),source_delta_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
            mean_rms=float(np.sqrt((mean*mean).mean())),std_min=float(std.min()),
            std_max=float(std.max()),std_mean=float(std.mean()))
        entries.append(row);params.append((mean,std,loss))
    add('identity',0.,0.,zeros,ones)
    for kappa in KAPPAS:
        means=global_mean[None]+kappa*(raw_means-global_mean[None])
        variance=((train-means[yt])**2).mean(0)
        assert np.all(variance>0) and np.isfinite(variance).all()
        std=np.sqrt(variance)
        for rho in RHOS[1:]:
            add('diagonal',kappa,rho,rho*means,1+rho*(std-1))
    # Spherical variance is fitted with zero mean. This is the noise-temperature control.
    spherical_std=np.sqrt((train*train).mean())
    for rho in RHOS[1:]:
        add('isotropic',0.,rho,zeros,np.full(4096,1+rho*(spherical_std-1)))
    def winner(indices):
        return min(indices,key=lambda i:(entries[i]['heldout_nll_per_dim'],i))
    picks={
        'primary':winner([0]+[i for i,r in enumerate(entries) if r['family']=='diagonal']),
        'global_diagonal':winner([0]+[i for i,r in enumerate(entries) if r['family']=='diagonal' and r['kappa']==0]),
        'isotropic':winner([0]+[i for i,r in enumerate(entries) if r['family']=='isotropic']),
    }
    out.mkdir(parents=True,exist_ok=False)
    selected={};unique=[]; configs=[dict(arm='cfg_native',kind='cfg',alpha=1.25,steps=64,cutoff=.75),
                                 dict(arm='apg_native',kind='apg',alpha=2.,steps=64,cutoff=.75,beta=-.5)]
    # Recheck neighboring strengths on the same new screening bank. The prior
    # remains tied to CFG1.25; it is never transferred between maps by NLL.
    configs += [dict(arm='cfg_a'+str(a),kind='cfg',alpha=a,steps=64,cutoff=.75) for a in (1.,1.5)]
    configs += [dict(arm='apg_a'+str(a),kind='apg',alpha=a,steps=64,cutoff=.75,beta=-.5) for a in (1.5,2.5)]
    for name,index in picks.items():
        mean,std,loss=params[index]
        selected[name]=dict(index=index,**entries[index])
        # Deduplicate the actual deployed FP32 affine maps, including distinct
        # grid/family winners that become identical after parameter casting.
        deployed_mean=mean.astype(np.float32).reshape(100,4,32,32)
        deployed_std=std.astype(np.float32).reshape(4,32,32)
        if np.all(deployed_mean==0) and np.all(deployed_std==1):
            selected[name]['same_as']='cfg_native'
            continue
        same_as=next((old_name for old_mean,old_std,old_name in unique
                      if np.array_equal(deployed_mean,old_mean)
                      and np.array_equal(deployed_std,old_std)),None)
        if same_as is not None:
            selected[name]['same_as']=same_as
            continue
        unique.append((deployed_mean,deployed_std,name))
        target=out/(name+'.npz')
        np.savez(target,means=deployed_mean,std=deployed_std,
                 heldout_nll_per_dim=loss,heldout_source_ids=ids[split==1])
        selected[name].update(path=str(target),sha256=sha(target))
        configs.append(dict(arm='cfg_'+name,kind='inverse_prior',alpha=1.25,steps=64,cutoff=.75,
                            prior_path=str(target),prior_sha256=sha(target)))
    summary=dict(complete=True,fit_sources=2000,selection_sources=1000,
        split='real training images only; source-disjoint; no FID reference fitting',
        criterion='heldout Gaussian NLL per dimension; identity included; no FID parameter selection',
        kappas=KAPPAS,rhos=RHOS,nominal_diagonal_grid=20,unique_diagonal_including_identity=17,
        baseline_nll_per_dim=float(baseline.mean()),rows=entries,selected=selected,
        bank_sha256=sha(root/'bank.npz'),source_sha256=sha(__file__),
        interpretation='NLL transfer is exact only for the actual invertible deployed map and its true inverse. Numerical inverse is approximate; FID improvement remains untested.')
    atomic(out/'summary.json',summary)
    atomic(out/'configs.json',configs)
    np.savez(out/'heldout_scores.npz',losses=np.stack([p[2] for p in params]),
             source_ids=ids[split==1],labels=yv)
    print(json.dumps(dict(selected=selected,configs=configs)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    fit(parser.parse_args())
