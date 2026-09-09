"""Native8 parity, actual-query prefix parity, fixed1K component comparison."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['sit','raev2'],required=True);p.add_argument('--arm',choices=['interaction','unconditional'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 root=Path('/home/zhoushunyu/data/eqvae/experiments/pfr_condition_component_20260908')/a.model/a.arm;root.mkdir(parents=True,exist_ok=False)
 if a.model=='sit':
  old=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_20260908/ordinary100');ordinary='ordinary100'
  refs=dict(native=old/'smoke',smoke=old/'smoke',quality=old/'quality')
 else:
  old=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908');ordinary='ordinary'
  refs=dict(native=old/'smoke/ordinary100',smoke=old/'smoke/ordinary100',quality=old/'quality/ordinary100')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for arm,n,part in [(ordinary,8,'native'),(a.arm,8,'smoke'),(a.arm,1000,'quality')]:
  out=root/part;cmd=[sys.executable,f'experiments/sample_{a.model}_pfr_condition_component.py','--arm',arm,'--samples',str(n),'--output',str(out)]
  (root/(part+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(part+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  m=json.loads((out/'summary.json').read_text());ref=refs[part];r=json.loads((ref/'summary.json').read_text());assert m['complete']
  for key in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[key]==r[key],key
  if a.model=='sit':assert m['vae_state_sha256']==r['vae_state_sha256']
  if part=='native':assert np.array_equal(np.load(out/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
  if part=='smoke':assert m['prefix_parity'] and m['parity_full_calls']==3
  if part=='quality':assert np.array_equal(np.load(out/'samples.npz')['arr_0'][:8],np.load(root/'smoke/samples.npz')['arr_0'])
  print(a.model,a.arm,part,'passed',flush=True)
 q=root/'quality';cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(q/'samples.npz'),'--output',str(q/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(q/'features')]
 with (root/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,native_parity=True,prefix_parity=True,paired_inputs=True)))
 print((q/'fid.json').read_text(),flush=True)
if __name__=='__main__':main()
