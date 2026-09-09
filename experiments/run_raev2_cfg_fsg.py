"""Fixed true-CFG versus CFG-FSG RAE comparison; no parameter search."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['ordinary110','asynchronous'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_cfg_fsg_20260908')/a.arm;root.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for part,n,arm in [('native',8,'ordinary100'),('smoke',8,a.arm),('quality',1000,a.arm)]:
  out=root/part
  cmd=[sys.executable,'experiments/sample_raev2_cfg_fsg.py','--arm',arm,'--samples',str(n),'--output',str(out)]
  if part=='native':cmd+=['--cfg-scale','1','--native-parity']
  (root/(part+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(part+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  m=json.loads((out/'summary.json').read_text());assert m['complete']
  if part=='native':assert m['native_pixel_parity']
  else:
   assert m['cfg_scale']==2 and m['null_label']==1000
   assert m['full_calls']==220*((n+3)//4)
  print(part,'passed',flush=True)
 out=root/'quality'
 cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(out/'samples.npz'),'--output',str(out/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')]
 with (root/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,native_full_parity=True)))
 print((out/'fid.json').read_text(),flush=True)
if __name__=='__main__':main()
