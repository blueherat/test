import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['raw','balanced'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    data=Path('/home/zhoushunyu/data/eqvae/experiments');root=data/'raev2_ig_balanced_write_20260908'/a.arm
    root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    reference=data/'raev2_pfr_working_point_20260908/full'
    for part,n,arm in [('zero',8,'balanced'),('smoke',8,a.arm),('quality',1000,a.arm)]:
        out=root/part
        cmd=[sys.executable,'-m','experiments.sample_raev2_ig_balanced_write','--arm',arm,'--samples',str(n),'--output',str(out)]
        if part=='zero':cmd+=['--zero-write']
        (root/f'{part}_command.json').write_text(json.dumps(cmd,indent=2))
        with (root/f'{part}.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        m=json.loads((out/'summary.json').read_text());assert m['complete']
        pixels=np.load(out/'samples.npz')['arr_0']
        if part=='zero':np.testing.assert_array_equal(pixels,np.load(reference/'smoke/samples.npz')['arr_0'])
        if part=='quality':
            np.testing.assert_array_equal(pixels[:8],np.load(root/'smoke/samples.npz')['arr_0'])
            ref=json.loads((reference/'quality/summary.json').read_text())
            for k in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[k]==ref[k]
        print(part,'passed',flush=True)
    q=root/'quality';cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(q/'samples.npz'),
                           '--output',str(q/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(q/'features')]
    with (root/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    (root/'complete.json').write_text(json.dumps(dict(complete=True,zero_parity=True,smoke_parity=True,paired_inputs=True)))
    print((q/'fid.json').read_text(),flush=True)

if __name__=='__main__':main()
