"""Fixed posterior-reference pilot with native pixel and actual-query parity."""
import argparse,json,os,subprocess,sys,shutil
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['posterior','ordinary150'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_posterior_reference_20260908')/a.arm;root.mkdir(parents=True,exist_ok=False)
 for source in [Path(__file__),Path('docs/RAEV2_POSTERIOR_REFERENCE_1K_PROTOCOL_20260908_ZH.md')]:shutil.copy2(source,root/source.name)
 original=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for arm,n,name in [('ordinary',8,'native'),(a.arm,8,'smoke'),(a.arm,1000,'quality')]:
  out=root/name;cmd=[sys.executable,'experiments/sample_raev2_posterior_reference.py','--arm',arm,'--samples',str(n),'--output',str(out)]
  (root/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(name+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  m=json.loads((out/'summary.json').read_text());ref=original/('smoke' if n==8 else 'quality')/'ordinary100';r=json.loads((ref/'summary.json').read_text());assert m['complete']
  for k in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[k]==r[k],k
  if name=='native':assert np.array_equal(np.load(out/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
  if name=='smoke' and a.arm=='posterior':assert m['prefix_parity'] and m['parity_full_calls']==2
  print(name,'checks passed',flush=True)
 out=root/'quality';cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(out/'samples.npz'),'--output',str(out/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')]
 with (out/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,paired_inputs=True,native_parity=True,prefix_parity=True if a.arm=='posterior' else None)))
 print((out/'fid.json').read_text(),flush=True)
if __name__=='__main__':main()
