"""Audit four fixed SiT output controls and independently reconstruct ADM FID."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/sit_clock_compensator_20260908')
 ref=np.load('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz');mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
 rows=[];inputs=set()
 for model in ['v']:
  complete=json.loads((root/model/'complete.json').read_text());assert all(complete[k] for k in ['complete','paired_inputs','native_parity','prefix_parity'])
  assert np.array_equal(np.load(root/model/'native/samples_n8.npz')['arr_0'],np.load(root/model/'smoke_ordinary/samples_n8.npz')['arr_0'])
  for smoke in ['smoke_smooth','smoke_compensated']:
   assert json.loads((root/model/smoke/'pfr_output_manifest.json').read_text())['prefix_parity']
  models=set()
  for arm in ['ordinary','smooth','compensated']:
   d=(Path('/home/zhoushunyu/data/eqvae/experiments/sit_pfr_output_control_20260908')/model/arm if arm=='ordinary' else root/model/arm);m=json.loads((d/'sampling_manifest.json').read_text());p=json.loads((d/'pfr_output_manifest.json').read_text())
   assert p['complete'] and p['arm']==arm and p['parity_full_calls']==0
   assert p['source_sha256']==hashlib.sha256((d/'wrapper_source.py').read_bytes()).hexdigest()
   assert m['requested_samples']==1000 and m['global_seed']==202609417 and m['num_steps']==100 and m['batch_size']==8 and m['precision']=='fp32'
   assert m['field_definition']['gamma']==.35
   inputs.add((m['noise_sha256'],m['label_sha256']));models.add(json.dumps([m['strong'],m['weak']],sort_keys=True))
   counts=m['model_forward_totals'];prefix=counts.get('pfr_prefix_calls',0)
   full=sum(v for k,v in counts.items() if k!='pfr_prefix_calls');assert full==12500 and prefix==(0 if arm=='ordinary' else 6250)
   assert counts.get('ou_full_calls',0)==0
   pix=np.load(d/'samples_n1000.npz')['arr_0'];assert pix.shape==(1000,256,256,3) and pix.dtype==np.uint8;del pix
   x=np.load(d/'activations.npz')['pool_3'].astype(float);assert x.shape==(1000,2048) and np.isfinite(x).all()
   sm=x.mean(0);x-=sm;gram=x@cov@x.T/999;w=np.linalg.eigvalsh((gram+gram.T)/2);assert w.min()>-1e-7
   fid=float(((sm-mu)**2).sum()+(x*x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(w,0)).sum());f=json.loads((d/'fid.json').read_text());assert abs(fid-f['fid'])<2e-4
   row=dict(model=model,arm=arm,fid=f['fid'],independent_fid=fid,inception_score=f['inception_score'],sampling_seconds=m['elapsed_seconds'],full_forwards_per_sample=full/125,prefix_forwards_per_sample=prefix/125,noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256'],strong_sha256=m['strong']['checkpoint_sha256'],head_sha256=m['weak']['heads']['depth4']['checkpoint_sha256'])
   rows.append(row);print(row,flush=True)
  assert len(models)==1
 assert len(inputs)==1
 with Path('experiments/results/terminal_defect_20260908/sit_clock_compensator.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
