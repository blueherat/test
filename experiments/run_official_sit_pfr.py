"""Smoke then fixed 1K sampling and official ImageNet FID."""
import argparse,json,os,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['ordinary100','ordinary115','pfr'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 root=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_20260908')/a.arm;root.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 for part,n in [('smoke',8),('quality',1000)]:
  cmd=[sys.executable,'experiments/sample_official_sit_pfr.py','--arm',a.arm,'--samples',str(n),'--output',str(root/part)]
  (root/(part+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(part+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print(part,'passed',flush=True)
 q=root/'quality'
 cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(q/'samples.npz'),'--output',str(q/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(q/'features')]
 with (root/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True)))
 print((q/'fid.json').read_text(),flush=True)
if __name__=='__main__':main()
