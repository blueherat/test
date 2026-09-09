"""Reconstruct best historical SiT RNG banks and FID from retained features."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch

root=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/pfr_ou_strong_direction_magnitude_v1')
rows=[];sets=[];torch.set_num_threads(4)
for seed in [5,6]:
 d=root/f'fid5k_seed{seed}_balanced/pfr_ou_d1_strong_common_direction_raw_norm_first_heun_n22'
 m=json.loads((d/'sampling_manifest.json').read_text());metrics=json.loads((d/'adm_metrics.json').read_text());assert m['batch_rng']['schema']=='namespaced_v2'
 h=hashlib.sha256();individual=set();started=time.perf_counter()
 for b in range(1250):
  gen=torch.Generator(device='cuda').manual_seed((seed<<32)|b)
  x=torch.randn(4,4,32,32,generator=gen,device='cuda').cpu().numpy();h.update(x.tobytes())
  for item in x:individual.add(hashlib.sha256(item.tobytes()).hexdigest())
 assert len(individual)==5000 and h.hexdigest()==m['noise_sha256'];sets.append(individual)
 labels=torch.arange(5000)%100;labels=labels[torch.randperm(5000,generator=torch.Generator().manual_seed(seed^0x5A17))].numpy()
 assert hashlib.sha256(labels.tobytes()).hexdigest()==m['label_sha256'];assert np.array_equal(labels,np.load(m['labels']))
 feat=np.load(d/'adm_activations.npz');ref=np.load(metrics['reference']);print(seed,feat.files,ref.files,flush=True)
 # Official retained ADM pool3 features, independent symmetric covariance-root FID.
 x=feat['pool_3'].astype(np.float64);assert x.shape==(5000,2048) and np.isfinite(x).all()
 mu=ref['mu'].astype(np.float64);cov=ref['sigma'].astype(np.float64)
 xm=x.mean(0);xc=x-xm;sc=xc.T@xc/4999
 eig,q=np.linalg.eigh((cov+cov.T)/2);assert eig.min()>-1e-8
 rootcov=(q*np.sqrt(np.maximum(eig,0)))@q.T;a=rootcov@sc@rootcov
 vals=np.linalg.eigvalsh((a+a.T)/2);assert vals.min()>-1e-7
 fid=float(((xm-mu)**2).sum()+np.trace(sc)+np.trace(cov)-2*np.sqrt(np.maximum(vals,0)).sum())
 assert abs(fid-metrics['fid'])<2e-4,(fid,metrics['fid'])
 rows.append(dict(seed=seed,noise_sha256=h.hexdigest(),unique_noises=len(individual),labels_reproduced=True,reported_fid=metrics['fid'],independent_fid=fid,raw_samples_available=Path(m['samples']).is_file(),seconds=time.perf_counter()-started,manifest_sha256=hashlib.sha256((d/'sampling_manifest.json').read_bytes()).hexdigest()))
assert not sets[0].intersection(sets[1])
out=dict(complete=True,cross_seed_identical_noise_count=0,rows=rows,scope='RNG reconstruction and retained-feature FID, not pixel-to-feature reproduction')
Path('experiments/results/terminal_defect_20260908/pfr_best_5k_assets_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
