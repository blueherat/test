import csv,json,hashlib
from pathlib import Path
import numpy as np
r=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908')
ref=np.load('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz');mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
rows=[]
for family in ['cfg','ig']:
 assert json.loads((r/'common_field'/family/'complete.json').read_text())['complete']
 for arm in ['short','asynchronous']:
  d=r/'common_field'/family/arm;m=json.loads((d/'sampling_manifest.json').read_text());old=json.loads((r/family/arm/'sampling_manifest.json').read_text())
  assert all(m[k]==old[k] for k in ['noise_sha256','label_sha256','global_seed','batch_size','requested_samples'])
  assert sum(m['model_forward_totals'].values())==sum(old['model_forward_totals'].values())
  cm=json.loads((d/'common_field_manifest.json').read_text());assert cm['complete']
  assert cm['source_sha256']==hashlib.sha256(Path('experiments/sample_sit_fsg_common_field.py').read_bytes()).hexdigest()
  pix=np.load(d/'samples_n1000.npz')['arr_0'];assert pix.shape==(1000,256,256,3) and pix.dtype==np.uint8;del pix
  x=np.load(d/'activations.npz')['pool_3'].astype(float);assert x.shape==(1000,2048) and np.isfinite(x).all()
  sm=x.mean(0);x-=sm;gram=x@cov@x.T/999;v=np.linalg.eigvalsh((gram+gram.T)/2);assert v.min()>-1e-7
  f=float(((sm-mu)**2).sum()+(x*x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(v,0)).sum())
  fd=json.loads((d/'fid.json').read_text());assert abs(f-fd['fid'])<2e-4
  rows.append(dict(family=family,arm=arm,fid=fd['fid'],independent_fid=f,inception_score=fd['inception_score'],sampling_seconds=m['elapsed_seconds'],model_forwards_per_sample=sum(m['model_forward_totals'].values())*8/1000,noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256']))
  print(rows[-1],flush=True)
p=Path('experiments/results/terminal_defect_20260908/fsg_common_field_controls.csv')
with p.open('x') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
