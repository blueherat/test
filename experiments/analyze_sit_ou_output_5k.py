"""Independent covariance-root FID and spatial FID on fixed four-arm 5K."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/sit_ou_output_5k_20260908');oldroot=Path('/home/zhoushunyu/data/eqvae/experiments/sit_ou_output_control_20260908')
 refpath=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz');ref=np.load(refpath);refs={}
 for kind,muk,covk in [('fid','mu','sigma'),('sfid','mu_s','sigma_s')]:
  mu=ref[muk].astype(float);cov=ref[covk].astype(float);eig,q=np.linalg.eigh((cov+cov.T)/2);assert eig.min()>-1e-7
  refs[kind]=(mu,cov,(q*np.sqrt(np.maximum(eig,0)))@q.T)
 rows=[];inputs=set()
 for model in ['v','x']:
  checkpoints=set()
  for arm in ['time_only','ou']:
   r=root/model/arm;done=json.loads((r/'complete.json').read_text());assert all(done.values())
   assert np.array_equal(np.load(r/'native/samples_n8.npz')['arr_0'],np.load(r/'ordinary/samples_n8.npz')['arr_0'])
   assert json.loads((r/'smoke/pfr_output_manifest.json').read_text())['prefix_parity']
   d=r/'quality';m=json.loads((d/'sampling_manifest.json').read_text());w=json.loads((d/'pfr_output_manifest.json').read_text());prior=json.loads((oldroot/model/arm/'pfr_output_manifest.json').read_text())
   assert w['complete'] and w['arm']==arm and w['parity_full_calls']==0
   assert w['source_sha256']==prior['source_sha256']==hashlib.sha256((d/'wrapper_source.py').read_bytes()).hexdigest()
   assert m['requested_samples']==5000 and m['global_seed']==202609423 and m['num_steps']==100 and m['batch_size']==8 and m['precision']=='fp32'
   assert m['field_definition']['gamma']==.35
   ct=m['model_forward_totals'];assert ct['pfr_prefix_calls']==31250 and ct['ou_full_calls']==15625 and sum(v for k,v in ct.items() if k!='pfr_prefix_calls')==78125
   inputs.add((m['noise_sha256'],m['label_sha256']));checkpoints.add(json.dumps([m['strong'],m['weak']],sort_keys=True))
   old=json.loads((oldroot/model/arm/'sampling_manifest.json').read_text());assert m['noise_sha256']!=old['noise_sha256']
   pix=np.load(d/'samples_n5000.npz')['arr_0'];assert pix.shape==(5000,256,256,3) and pix.dtype==np.uint8
   assert np.array_equal(pix[:8],np.load(r/'smoke/samples_n8.npz')['arr_0']);del pix
   metrics=json.loads((d/'fid.json').read_text());features=np.load(d/'activations.npz');row=dict(model=model,arm=arm,samples=5000,sampling_seconds=m['elapsed_seconds'],full_forwards_per_sample=125,prefix_forwards_per_sample=50,noise_sha256=m['noise_sha256'],label_sha256=m['label_sha256'])
   for kind,key in [('fid','pool_3'),('sfid','spatial')]:
    x=features[key].astype(float);mu,cov,sq=refs[kind];assert x.shape==(5000,len(mu)) and np.isfinite(x).all()
    xm=x.mean(0);x-=xm;sc=x.T@x/4999;a=sq@sc@sq;eig=np.linalg.eigvalsh((a+a.T)/2);assert eig.min()>-1e-6
    metric=float(((xm-mu)**2).sum()+np.trace(sc)+np.trace(cov)-2*np.sqrt(np.maximum(eig,0)).sum());assert abs(metric-metrics[kind])<.002,(kind,metric,metrics[kind])
    row[kind]=metrics[kind];row['independent_'+kind]=metric
   row['inception_score']=metrics['inception_score'];rows.append(row);print(row,flush=True)
  assert len(checkpoints)==1
 assert len(inputs)==1
 with Path('experiments/results/terminal_defect_20260908/sit_ou_output_5k.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
