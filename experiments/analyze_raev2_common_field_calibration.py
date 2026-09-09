"""Audit same-field RAE calibration, paired inputs, parity and independent feature FID."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_common_field_calibration_20260908')
 original=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908/quality')
 ref=np.load('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
 mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
 paths={'none':original/'ordinary100','all':original/'asynchronous','common':root/'quality'}
 rows=[];inputs=set();checkpoints=set();evaluators=set();intervention_sources=set();helper_sources=set()
 for arm,d in paths.items():
  if arm=='common':
   c=json.loads((d.parent/'complete.json').read_text());assert all(c[k] for k in ['complete','paired_inputs','native_parity','original_parity','matched_calls'])
   for name,refarm in [('native_parity','ordinary100'),('original_parity','asynchronous')]:
    x=np.load(d.parent/name/'samples.npz')['arr_0'];y=np.load(original.parent/'smoke'/refarm/'samples.npz')['arr_0'];assert np.array_equal(x,y)
  m=json.loads((d/'summary.json').read_text());assert m['complete'] and m['samples']==1000
  assert m['seed']==202609413 and m['batch_size']==4 and m['steps']==100
  assert m['calibration_calls']==(0 if arm=='none' else 2500)
  for path,digest in m['sources'].items():
   assert hashlib.sha256((d/Path(path).name).read_bytes()).hexdigest()==digest
  helper_sources.add(tuple(sorted((Path(p).name,h) for p,h in m['sources'].items() if Path(p).name in ['sample_raev2_pfr_retiming.py','raev2_pfr_retiming.py'])))
  assert m['full_calls']==250*(100 if arm=='none' else 110)
  if arm=='common':
   assert m['event_subset']=='all' and m['reference']=='common' and m['h']==.025 and m['H']==.125
   assert [e['index'] for e in m['events']]==[0,54,83]
   assert [e['iterations'] for e in m['events']]==[2,2,1]
   intervention_sources.add(tuple(sorted(m['sources'].items())))
  inputs.add((m['noise_sha256'],m['label_sha256']));checkpoints.add(m['checkpoint_sha256'])
  pixels=np.load(d/'samples.npz')['arr_0'];assert pixels.shape==(1000,256,256,3) and pixels.dtype==np.uint8;del pixels
  pixelhash=hashlib.sha256((d/'samples.npz').read_bytes()).hexdigest();assert pixelhash==m['pixel_sha256']
  f=json.loads((d/'fid.json').read_text())[0];assert f['sample_sha256']==pixelhash
  evaluators.add((f['fid_reference'],f['evaluator_commit']))
  fs=list((d/'features').glob('*.features.pt'));assert len(fs)==1
  x=torch.load(fs[0],weights_only=True,map_location='cpu').numpy().astype(float);assert x.shape==(1000,2048) and np.isfinite(x).all()
  sm=x.mean(0);x-=sm;gram=x@cov@x.T/999;v=np.linalg.eigvalsh((gram+gram.T)/2);assert v.min()>-1e-7
  fd=float(((sm-mu)**2).sum()+(x*x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(v,0)).sum());assert abs(fd-f['fid'])<2e-4
  row=dict(arm=arm,fid=f['fid'],independent_fid=fd,inception_score=f['inception_score'],sampling_seconds=m['seconds'],full_forwards_per_sample=m['full_calls']/250,noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256'],pixel_sha256=pixelhash)
  rows.append(row);print(row,flush=True)
 assert len(inputs)==len(checkpoints)==len(evaluators)==len(intervention_sources)==len(helper_sources)==1
 out=Path('experiments/results/terminal_defect_20260908/raev2_common_field_calibration.csv')
 with out.open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
