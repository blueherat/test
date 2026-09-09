"""Independent source/input/cost/FP64 FID audit of the official-XL OU pilot."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/zhoushunyu/data/eqvae/experiments')
REF=Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert sha(REF)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
 ref=np.load(REF);mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
 old=DATA/'official_sit_pfr_20260908';om=json.loads((old/'pfr/quality/summary.json').read_text());rows=[];sources=set()
 for arm in ['raw','ou']:
  p=DATA/'official_sit_pfr_ou_20260908'/arm;q=p/'quality';sm=p/'smoke'
  c=json.loads((p/'complete.json').read_text());assert all(c[k] for k in ['complete','native_parity','raw_parity','paired_inputs'])
  m=json.loads((q/'summary.json').read_text());s=json.loads((sm/'summary.json').read_text())
  assert m['complete'] and m['samples']==1000 and m['batch_size']==4 and m['arm']==arm
  assert (m['full_calls'],m['prefix_calls'],m['ou_full_calls'],m['parity_full_calls'])==(31250,12500,6250,0)
  assert s['prefix_parity'] and s['parity_full_calls']==1
  assert m['scale']==1.35 and m['horizon']==1/32
  for k in ['noise_sha256','label_sha256','checkpoint_sha256','vae_state_sha256']:assert m[k]==om[k]
  assert m['sources']==s['sources'];sources.add(tuple(sorted(m['sources'].items())))
  for path,digest in m['sources'].items():assert sha(q/Path(path).name)==sha(sm/Path(path).name)==digest
  for part,refdir in [('native',old/'ordinary100/smoke'),('raw',old/'pfr/smoke')]:
   assert np.array_equal(np.load(p/part/'samples.npz')['arr_0'],np.load(refdir/'samples.npz')['arr_0'])
  x=np.load(q/'samples.npz')['arr_0'];assert x.shape==(1000,256,256,3) and x.dtype==np.uint8
  assert np.array_equal(x[:8],np.load(sm/'samples.npz')['arr_0'])
  if arm=='raw':assert np.array_equal(x,np.load(old/'pfr/quality/samples.npz')['arr_0'])
  del x
  metric=json.loads((q/'fid.json').read_text())[0];assert sha(q/'samples.npz')==m['pixel_sha256']==metric['sample_sha256']
  assert metric['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b' and metric['fid_reference']=='imagenet_256_fid_stats'
  f,=(q/'features').glob('*.features.pt');x=torch.load(f,map_location='cpu',weights_only=True).numpy().astype(float)
  assert x.shape==(1000,2048) and np.isfinite(x).all();mean=x.mean(0);x-=mean
  g=x@cov@x.T/999;ev=np.linalg.eigvalsh((g+g.T)/2);assert ev.min()>-1e-7
  mt=float(np.square(mean-mu).sum());ct=float(np.square(x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum())
  assert abs(mt+ct-metric['fid'])<2e-4
  rows.append(dict(arm=arm,fid=metric['fid'],independent_fid=mt+ct,mean_term=mt,covariance_term=ct,inception_score=metric['inception_score'],sampling_seconds=m['seconds'],full_per_image=125,prefix_per_image=50,pixel_sha256=m['pixel_sha256'],feature_sha256=sha(f)))
 assert len(sources)==1
 with (ROOT/'experiments/results/terminal_defect_20260908/official_sit_pfr_ou.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
