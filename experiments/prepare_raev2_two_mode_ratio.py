"""Two Gaussian variances from source samples, without FID or time fitting."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

DATA=Path('/home/zhoushunyu/data/eqvae/experiments')
OUT=DATA/'raev2_guidance_20260907/two_mode_ratio'


def moments(files):
    dc=ac=0.;count=0;identities=[]
    for path in files:
        z=np.load(path,mmap_mode='r')
        for i in range(0,len(z),16):
            x=np.array(z[i:i+16],dtype=np.float64)
            avg=x.mean((2,3),keepdims=True)
            dc+=float(np.square(avg).sum())
            ac+=float(np.square(x-avg).sum())/255
            count+=len(x)
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        identities.append({'path':str(path),'sha256':h.hexdigest(),'shape':list(z.shape)})
    return {'samples':count,'dc_variance':256*dc/(count*1024),'ac_variance':ac/(count*1024),'sources':identities}


def main():
    OUT.mkdir(exist_ok=False)
    start=time.perf_counter()
    real=[DATA/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1/train/latents.npy']
    generated=[DATA/f'raev2_ig_scale_response/n5000_seed{seed}_scales7_v1/latents/scale_s1p780000_rank{rank:02d}.npy'
               for seed in [20260801,20260802] for rank in range(4)]
    for p in real+generated:
        if not p.is_file():raise FileNotFoundError(p)
    manifests=[]
    for seed in [20260801,20260802]:
        p=DATA/f'raev2_ig_scale_response/n5000_seed{seed}_scales7_v1/manifest.json'
        d=json.loads(p.read_text())
        if d['ig_interval']!=[.1,1.]:raise ValueError('source IG interval mismatch')
        manifests.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'content':d})
    result={'complete':True,'real':moments(real),'generated':moments(generated),'source_manifests':manifests,
        'fid_used_for_estimation':False,'centering':'none; maximum entropy subject to two raw second-moment constraints',
        'approximation':'unconditional zero-mean Gaussian with DC and orthogonal AC eigenspaces',
        'seconds':time.perf_counter()-start}
    (OUT/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_manifests'},indent=2))


if __name__=='__main__':main()
