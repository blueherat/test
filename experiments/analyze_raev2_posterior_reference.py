"""Audit the fixed PFR query comparison, including independent feature-Gram FID."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_posterior_reference_20260908')
 old=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908')
 ref=np.load('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz');mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
 rows=[];inputs=set();checkpoints=set();sources=set();evaluators=set();prefix_counts=set()
 for arm,d in [('ordinary',old/'quality/ordinary100'),('posterior',root/'posterior/quality'),('ordinary150',root/'ordinary150/quality')]:
  m=json.loads((d/'summary.json').read_text());assert m['complete'] and m['samples']==1000 and m['seed']==202609413 and m['batch_size']==4 and m['full_calls']==(37500 if arm=='ordinary150' else 25000)
  inputs.add((m['noise_sha256'],m['label_sha256']));checkpoints.add(m['checkpoint_sha256'])
  for path,digest in m['sources'].items():assert hashlib.sha256((d/Path(path).name).read_bytes()).hexdigest()==digest
  if arm!='ordinary':
   c=json.loads((d.parent/'complete.json').read_text());assert all(c[k] for k in ['complete','paired_inputs','native_parity'])
   assert np.array_equal(np.load(d.parent/'native/samples.npz')['arr_0'],np.load(old/'smoke/ordinary100/samples.npz')['arr_0'])
   smoke=json.loads((d.parent/'smoke/summary.json').read_text())
   assert smoke['sources']==m['sources'] and smoke['query_rng_initial']==m['query_rng_initial']
   for path,digest in smoke['sources'].items():assert hashlib.sha256((d.parent/'smoke'/Path(path).name).read_bytes()).hexdigest()==digest
   assert np.array_equal(np.load(d/'samples.npz')['arr_0'][:8],np.load(d.parent/'smoke/samples.npz')['arr_0'])
   if arm=='posterior':assert c['prefix_parity'] and smoke['prefix_parity'] and smoke['parity_full_calls']==2
   else:assert smoke['parity_full_calls']==0 and m['query_rng_initial']==m['query_rng_final']
   assert m['arm']==arm and m['h']==1/32 and m['rho']==1 and m['revision_noise_time_min']==.5 and m['parity_full_calls']==0
   assert m['steps']==(150 if arm=='ordinary150' else 100) and m['query_seed']==202609434 and m['antithetic_pairs']==1
   assert m['prefix_calls']==(44500 if arm=='posterior' else 0)
   if arm=='posterior':assert m['active_steps']==89
   sources.add(tuple(sorted(m['sources'].items())))
  x=np.load(d/'samples.npz')['arr_0'];assert x.shape==(1000,256,256,3) and x.dtype==np.uint8;del x
  pixel=hashlib.sha256((d/'samples.npz').read_bytes()).hexdigest();assert pixel==m['pixel_sha256']
  f=json.loads((d/'fid.json').read_text())[0];assert pixel==f['sample_sha256'];evaluators.add((f['fid_reference'],f['evaluator_commit']))
  fs=list((d/'features').glob('*.features.pt'));assert len(fs)==1
  x=torch.load(fs[0],map_location='cpu',weights_only=True).numpy().astype(float);assert x.shape==(1000,2048) and np.isfinite(x).all()
  sm=x.mean(0);x-=sm;gram=x@cov@x.T/999;w=np.linalg.eigvalsh((gram+gram.T)/2);assert w.min()>-1e-7
  fid=float(((sm-mu)**2).sum()+(x*x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(w,0)).sum());assert abs(fid-f['fid'])<2e-4
  row=dict(arm=arm,fid=f['fid'],independent_fid=fid,inception_score=f['inception_score'],sampling_seconds=m['seconds'],full_forwards_per_sample=m['full_calls']/250,prefix_forwards_per_sample=m.get('prefix_calls',0)/250,noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256'],pixel_sha256=pixel)
  rows.append(row);print(row,flush=True)
 assert len(inputs)==len(checkpoints)==len(sources)==len(evaluators)==1
 with Path('experiments/results/terminal_defect_20260908/raev2_posterior_reference.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
