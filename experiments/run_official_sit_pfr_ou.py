"""Native/raw pixel parity before the official-XL fixed OU comparison."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['raw','ou'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 data=Path('/home/zhoushunyu/data/eqvae/experiments');root=data/'official_sit_pfr_ou_20260908'/a.arm;root.mkdir(parents=True,exist_ok=False)
 old=data/'official_sit_pfr_20260908';env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for arm,n,part,ref in [('ordinary100',8,'native',old/'ordinary100/smoke'),('raw',8,'raw',old/'pfr/smoke'),
                       (a.arm,8,'smoke',old/'pfr/smoke'),(a.arm,1000,'quality',old/'pfr/quality')]:
  out=root/part;cmd=[sys.executable,'experiments/sample_official_sit_pfr_ou.py','--arm',arm,'--samples',str(n),'--output',str(out)]
  (root/(part+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(part+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  m=json.loads((out/'summary.json').read_text());r=json.loads((ref/'summary.json').read_text());assert m['complete']
  for k in ['noise_sha256','label_sha256','checkpoint_sha256','vae_state_sha256']:assert m[k]==r[k],k
  if part in ['native','raw'] or arm=='raw':assert np.array_equal(np.load(out/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
  if n==8 and arm!='ordinary100':assert m['prefix_parity'] and m['parity_full_calls']==1
  if part=='quality':assert np.array_equal(np.load(out/'samples.npz')['arr_0'][:8],np.load(root/'smoke/samples.npz')['arr_0'])
  print(part,'passed',flush=True)
 q=root/'quality';cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(q/'samples.npz'),
                      '--output',str(q/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(q/'features')]
 with (root/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,native_parity=True,raw_parity=True,paired_inputs=True)))
 print((q/'fid.json').read_text(),flush=True)


if __name__=='__main__':main()
