"""Run fixed radius/direction interventions after exact native and early-only parity."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['radial','angular'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_initial_radius_direction_20260908')/a.arm;root.mkdir(parents=True,exist_ok=False)
 original=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908')
 early=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_clock_event_intervention_20260908/early_only')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for subset,component,n,name in [('none','full',8,'none'),('early_only','full',8,'full'),('early_only',a.arm,8,'smoke'),('early_only',a.arm,1000,'quality')]:
  out=root/name;cmd=[sys.executable,'experiments/sample_raev2_initial_radius_direction.py','--arm','asynchronous','--event-subset',subset,'--initial-component',component,'--samples',str(n),'--output',str(out)]
  (root/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(name+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  m=json.loads((out/'summary.json').read_text());ref=original/'smoke/ordinary100' if name=='none' else early/('smoke' if n==8 else 'quality');r=json.loads((ref/'summary.json').read_text())
  assert m['complete'] and m['full_calls']==110*(n//4) and m['calibration_calls']==10*(n//4)
  for k in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[k]==r[k],k
  if name in ['none','full']:assert np.array_equal(np.load(out/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
  else:assert m['initial_component']==a.arm and m['constraint_max_error']<2e-7
  print(name,'checks passed',flush=True)
 out=root/'quality';cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(out/'samples.npz'),'--output',str(out/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')]
 with (out/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,paired_inputs=True,native_parity=True,early_parity=True,matched_calls=True)))
 print((out/'fid.json').read_text(),flush=True)
if __name__=='__main__':main()
