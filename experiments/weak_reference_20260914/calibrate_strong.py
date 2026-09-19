"""Small frozen-model calibration; no FID or generated images select weights."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.weak_reference_20260914 import calibrated as method

ROOT=c.EXPS/'weak_reference_20260914'/'calibration_strong'
DATA=c.EXPS.parent/'imagenet_sit_flow/imagenet100_cmc_sdvae'
KERNELS=('white','lowpass','band')


@torch.inference_mode()
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),Path(method.__file__),Path(method.plain.__file__),
        DATA/'manifest.json',DATA/'train_labels.npy',DATA/'validation_labels.npy',*c.asset_paths('sit_small')]
    # Record cache manifests and the hashes of the actual selected moments below.
    request=dict(seed=2026091411,samples_per_split=1000,time_bin_centers=[(i+.5)/8 for i in range(8)],
        kernels=KERNELS,clip=[0,3],fit='scalar least squares on each time bin, raw conditional baseline',
        selection='lowest mean validation DSM loss over kernels; no image metrics',
        hashes={str(p):c.sha(p) for p in paths})
    c.atomic(ROOT/'request.json',request)
    rt=c.runtime('sit_small');rng=np.random.default_rng(request['seed']);begin=time.perf_counter()
    arrays={};identities={}
    for split in ('train','val'):
        file_split='validation' if split=='val' else split
        labels=np.load(DATA/f'{file_split}_labels.npy');moments=np.load(DATA/f'{file_split}_moments.npy',mmap_mode='r')
        ids=np.concatenate([rng.choice(np.flatnonzero(labels==label),10,replace=False) for label in range(100)])
        picked=np.asarray(moments[ids]).copy();ys=labels[ids].astype(np.int64)
        identities[split]=dict(ids=ids.tolist(),moments_sha256=c.array_sha(picked),labels_sha256=c.array_sha(ys))
        arrays[split]=np.empty((8,3,3,1000),np.float64) # time,kernel,[base loss,e dot g,g squared],image
        for timebin in range(8):
            t=(timebin+.5)/8
            gen=torch.Generator(device='cuda').manual_seed(request['seed']+(0 if split=='train' else 100)+timebin)
            for start in range(0,1000,20):
                m=torch.from_numpy(picked[start:start+20]).cuda();y=torch.from_numpy(ys[start:start+20]).cuda()
                draw=lambda:torch.randn((len(m),4,32,32),device='cuda',generator=gen)
                clean=(m[:,:4]+m[:,4:]*draw())*.18215;eps=draw();probe=draw()
                z=t*clean+(1-t)*eps;target=clean-eps
                with rt.context():
                    v,full,_,_=method.plain.sg.velocity(rt,z,t,y,dict(kind='baseline',omega=0,alpha=0))
                    error=(v-target).double()
                    for j,kernel in enumerate(KERNELS):
                        g=method.residual(rt,z,t,y,full,probe,kernel).double()
                        stats=torch.stack((error.square().flatten(1).mean(1),
                            (error*g).flatten(1).mean(1),g.square().flatten(1).mean(1)),0)
                        arrays[split][timebin,j,:,start:start+len(m)]=stats.cpu().numpy()
        print('evaluated',split,flush=True)
    weights=np.clip(-arrays['train'][:,:,1].mean(-1)/arrays['train'][:,:,2].mean(-1).clip(1e-20),0,3)
    rows=[]
    for j,kernel in enumerate(KERNELS):
        val=arrays['val'][:,j];w=weights[:,j,None]
        delta=2*w*val[:,1]+w*w*val[:,2]
        # All 8 times share the same held-out clean images: cluster by image.
        per_image=delta.mean(0)
        rows.append(dict(kernel=kernel,weights=weights[:,j].tolist(),
            baseline_validation_mse=float(val[:,0].mean()),validation_mse=float((val[:,0]+delta).mean()),
            delta=float(per_image.mean()),cluster_image_se=float(per_image.std(ddof=1)/np.sqrt(1000))))
    selected=min(rows,key=lambda row:row['validation_mse'])
    result=dict(rows=rows,selected_kernel=selected['kernel'],seconds=time.perf_counter()-begin,
        image_metrics_used=False,request_sha256=c.sha(ROOT/'request.json'))
    c.atomic(ROOT/'identities.json',identities);c.atomic(ROOT/'results.json',result)
    np.savez(ROOT/'per_image_stats.npz',**arrays,weights=weights)
    configs=[dict(arm='strong_e64',kind='baseline',alpha=0,omega=0,steps=64)]
    for row in rows:
        configs.append(dict(arm='strong_cal_'+row['kernel'],kind='calibrated',alpha=0,
            kernel=row['kernel'],weights=row['weights'],steps=64,calibration_sha256=c.sha(ROOT/'results.json')))
    c.atomic(ROOT/'configs.json',configs)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
