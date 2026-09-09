"""Verify fixed component pilots and independently recompute official1K FID."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/pfr_condition_component_20260908')
REF=Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert sha(REF)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
 ref=np.load(REF);mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float);rows=[]
 for model in ['sit','raev2']:
  old=Path('/home/zhoushunyu/data/eqvae/experiments')/('official_sit_pfr_20260908/ordinary100' if model=='sit' else 'raev2_fsg_clock_transfer_20260908')
  native=old/'smoke' if model=='sit' else old/'smoke/ordinary100'
  ordinary=old/'quality' if model=='sit' else old/'quality/ordinary100'
  om=json.loads((ordinary/'summary.json').read_text());sources=set()
  for arm in ['interaction','unconditional']:
   parent=DATA/model/arm;q=parent/'quality';smoke=parent/'smoke'
   complete=json.loads((parent/'complete.json').read_text());assert all(complete[k] for k in ['complete','native_parity','prefix_parity','paired_inputs'])
   m=json.loads((q/'summary.json').read_text());s=json.loads((smoke/'summary.json').read_text())
   assert m['complete'] and m['samples']==1000 and m['batch_size']==4 and m['arm']==arm and m['full_calls']==25000
   assert m['prefix_calls']==(37500 if model=='sit' else 66750) and m['parity_full_calls']==0
   assert s['prefix_parity'] and s['parity_full_calls']==3
   for key in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[key]==om[key]
   if model=='sit':assert m['vae_state_sha256']==om['vae_state_sha256'] and m['scale']==1.35 and m['horizon']==1/32
   else:assert m['rho']==1 and m['h']==1/32 and m['revision_noise_time_min']==.5
   assert m['sources']==s['sources'];sources.add(tuple(sorted(m['sources'].items())))
   for path,digest in m['sources'].items():assert sha(q/Path(path).name)==sha(smoke/Path(path).name)==digest
   assert np.array_equal(np.load(parent/'native/samples.npz')['arr_0'],np.load(native/'samples.npz')['arr_0'])
   x=np.load(q/'samples.npz')['arr_0'];assert x.shape==(1000,256,256,3) and x.dtype==np.uint8
   assert np.array_equal(x[:8],np.load(smoke/'samples.npz')['arr_0']);del x
   metric=json.loads((q/'fid.json').read_text())[0];assert sha(q/'samples.npz')==m['pixel_sha256']==metric['sample_sha256']
   assert metric['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b' and metric['fid_reference']=='imagenet_256_fid_stats'
   f,=(q/'features').glob('*.features.pt');x=torch.load(f,map_location='cpu',weights_only=True).numpy().astype(float)
   assert x.shape==(1000,2048) and np.isfinite(x).all();mean=x.mean(0);x-=mean
   gram=x@cov@x.T/999;ev=np.linalg.eigvalsh((gram+gram.T)/2);assert ev.min()>-1e-7
   mt=float(np.square(mean-mu).sum());ct=float(np.square(x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum());fid=mt+ct
   assert abs(fid-metric['fid'])<2e-4
   row=dict(model=model,arm=arm,fid=metric['fid'],independent_fid=fid,mean_term=mt,covariance_term=ct,inception_score=metric['inception_score'],sampling_seconds=m['seconds'],full_per_image=100,prefix_per_image=m['prefix_calls']/250,pixel_sha256=m['pixel_sha256'])
   rows.append(row);print(row,flush=True)
  assert len(sources)==1
 with (ROOT/'experiments/results/terminal_defect_20260908/pfr_condition_component.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
