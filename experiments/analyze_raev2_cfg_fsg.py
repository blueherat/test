"""Audit paired true-CFG RAE clock controls and independently reconstruct FID."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_cfg_fsg_20260908')
 ref=np.load('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
 mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
 rows=[];inputs=set();models=set();sources=set();evaluators=set()
 for arm in ('ordinary110','asynchronous'):
  parent=root/arm;d=parent/'quality'
  assert json.loads((parent/'complete.json').read_text())['complete']
  native=json.loads((parent/'native/summary.json').read_text())
  assert native['native_pixel_parity'] and native['cfg_scale']==1
  assert np.array_equal(np.load(parent/'native/samples.npz')['arr_0'],np.load(parent/'native/native/images-rank00.npy'))
  smoke=json.loads((parent/'smoke/summary.json').read_text())
  assert smoke['complete'] and smoke['full_calls']==440 and smoke['cfg_scale']==2
  m=json.loads((d/'summary.json').read_text())
  assert m['complete'] and m['arm']==arm and m['samples']==1000 and m['batch_size']==4 and m['seed']==202609413
  assert m['cfg_scale']==2 and m['null_label']==1000 and m['full_calls']==55000
  assert m['steps']==(110 if arm=='ordinary110' else 100)
  # calibration_calls counts field evaluations, each with two Full branches.
  assert m['calibration_calls']==(0 if arm=='ordinary110' else 2500)
  if arm=='asynchronous':assert [x['index'] for x in m['events']]==[0,54,83] and [x['iterations'] for x in m['events']]==[2,2,1] and m['h']==.025 and m['H']==.125
  for path,digest in m['sources'].items():assert hashlib.sha256((d/Path(path).name).read_bytes()).hexdigest()==digest
  assert m['sources']==smoke['sources']==native['sources']
  inputs.add((m['noise_sha256'],m['label_sha256']));models.add(m['checkpoint_sha256']);sources.add(tuple(sorted(m['sources'].items())))
  x=np.load(d/'samples.npz')['arr_0'];assert x.shape==(1000,256,256,3) and x.dtype==np.uint8
  assert np.array_equal(x[:8],np.load(parent/'smoke/samples.npz')['arr_0']);del x
  pixel=hashlib.sha256((d/'samples.npz').read_bytes()).hexdigest();assert pixel==m['pixel_sha256']
  f=json.loads((d/'fid.json').read_text())[0];assert f['sample_sha256']==pixel
  evaluators.add((f['fid_reference'],f['evaluator_commit']))
  files=list((d/'features').glob('*.features.pt'));assert len(files)==1
  x=torch.load(files[0],map_location='cpu',weights_only=True).numpy().astype(float);assert x.shape==(1000,2048) and np.isfinite(x).all()
  mean=x.mean(0);x-=mean;gram=x@cov@x.T/999;ev=np.linalg.eigvalsh((gram+gram.T)/2);assert ev.min()>-1e-7
  fid=float(((mean-mu)**2).sum()+(x*x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum())
  assert abs(fid-f['fid'])<2e-4
  rows.append(dict(arm=arm,fid=f['fid'],independent_fid=fid,inception_score=f['inception_score'],sampling_seconds=m['seconds'],full_branches_per_sample=m['full_calls']/250,noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256'],pixel_sha256=pixel))
  print(rows[-1],flush=True)
 assert len(inputs)==len(models)==len(sources)==len(evaluators)==1
 with Path('experiments/results/terminal_defect_20260908/raev2_cfg_fsg.csv').open('x') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
