"""Verify paired official SiT PFR assets and independently reconstruct FID."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_20260908')
 refpath=Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
 assert sha(refpath)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
 ref=np.load(refpath);mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
 inputs=set();models=set();vaes=set();sources=set();evaluators=set();rows=[]
 interface=json.loads(Path('experiments/results/terminal_defect_20260908/official_sit_pfr_interface.json').read_text())
 assert interface['complete'] and interface['official_full_latent_parity'] and interface['prefix_parity']
 for arm in ['ordinary100','pfr','ordinary115']:
  parent=root/arm;d=parent/'quality';smoke=parent/'smoke'
  assert json.loads((parent/'complete.json').read_text())['complete']
  m=json.loads((d/'summary.json').read_text());s=json.loads((smoke/'summary.json').read_text())
  steps=115 if arm=='ordinary115' else 100
  for meta,n in [(m,1000),(s,8)]:
   assert meta['complete'] and meta['arm']==arm and meta['samples']==n
   assert meta['batch_size']==4 and meta['seed']==202609428 and meta['steps']==steps
   assert meta['scale']==1.35 and meta['horizon']==1/32
   assert meta['full_calls']==steps*(n//4)
   assert meta['prefix_calls']==(50*(n//4) if arm=='pfr' else 0)
  assert m['sources']==s['sources'] and m['vae_state_sha256']==s['vae_state_sha256']
  for file,digest in m['sources'].items():
   assert sha(d/Path(file).name)==sha(smoke/Path(file).name)==digest
  for file,digest in interface['sources'].items():
   assert m['sources'][file]==digest
  assert m['checkpoint_sha256']==s['checkpoint_sha256']==interface['checkpoint_sha256']
  inputs.add((m['noise_sha256'],m['label_sha256']));models.add(m['checkpoint_sha256']);vaes.add(m['vae_state_sha256']);sources.add(tuple(sorted(m['sources'].items())))
  x=np.load(d/'samples.npz')['arr_0'];assert x.shape==(1000,256,256,3) and x.dtype==np.uint8
  assert np.array_equal(x[:8],np.load(smoke/'samples.npz')['arr_0']);del x
  pixel=sha(d/'samples.npz');assert pixel==m['pixel_sha256']
  f=json.loads((d/'fid.json').read_text())[0];assert f['sample_sha256']==pixel
  evaluators.add((f['fid_reference'],f['evaluator_commit']))
  feature,=(d/'features').glob('*.features.pt')
  x=torch.load(feature,map_location='cpu',weights_only=True).numpy().astype(float)
  assert x.shape==(1000,2048) and np.isfinite(x).all()
  mean=x.mean(0);x-=mean;gram=x@cov@x.T/999
  ev=np.linalg.eigvalsh((gram+gram.T)/2);assert ev.min()>-1e-7
  meanterm=float(np.square(mean-mu).sum())
  covterm=float(np.square(x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum())
  fid=meanterm+covterm;assert abs(fid-f['fid'])<2e-4
  rows.append(dict(arm=arm,fid=f['fid'],independent_fid=fid,mean_term=meanterm,covariance_term=covterm,
                   inception_score=f['inception_score'],sampling_seconds=m['seconds'],full_per_image=steps,
                   prefix_per_image=m['prefix_calls']/250,noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256'],pixel_sha256=pixel))
  print(rows[-1],flush=True)
 assert len(inputs)==len(models)==len(vaes)==len(sources)==len(evaluators)==1
 with Path('experiments/results/terminal_defect_20260908/official_sit_pfr.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
