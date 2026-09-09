"""Fixed RAE common-field control, gated on source-pixel reproduction."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_future_ig_calibration_20260908');root.mkdir(exist_ok=False)
 original=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for reference,subset,n,name in [('original','all',8,'original_parity'),('contrast','none',8,'native_parity'),('contrast','all',8,'smoke'),('contrast','all',1000,'quality')]:
  out=root/name;cmd=[sys.executable,'experiments/sample_raev2_future_ig_calibration.py','--arm','asynchronous','--reference',reference,'--event-subset',subset,'--samples',str(n),'--output',str(out)]
  (root/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(name+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  m=json.loads((out/'summary.json').read_text());ref=original/('smoke' if n==8 else 'quality')/('ordinary100' if subset=='none' else 'asynchronous');r=json.loads((ref/'summary.json').read_text())
  assert m['complete'] and m['full_calls']==110*(n//4) and m['calibration_calls']==10*(n//4)
  for k in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[k]==r[k],k
  if name in ['original_parity','native_parity']:assert np.array_equal(np.load(out/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
  print(name,'checks passed',flush=True)
 out=root/'quality';cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch','future_ig='+str(out/'samples.npz'),'--output',str(out/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')]
 with (out/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,paired_inputs=True,native_parity=True,original_parity=True,matched_calls=True)))
 print((out/'fid.json').read_text(),flush=True)
if __name__=='__main__':main()
