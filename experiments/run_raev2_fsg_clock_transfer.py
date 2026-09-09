"""Execute frozen 8-image checks and 1K RAEv2 clock transfer arms."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--arms',required=True)
    a=p.parse_args()
    root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_fsg_clock_transfer_20260908')
    parity=json.loads((root/'smoke/ordinary100/summary.json').read_text())
    assert parity['complete'] and parity['native_pixel_parity']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for arm in a.arms.split(','):
        assert arm in ['ordinary100','ordinary110','short','asynchronous','time_only']
        for n,part in [(8,'smoke'),(1000,'quality')]:
            out=root/part/arm
            if arm=='ordinary100' and n==8:continue
            out.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'experiments/sample_raev2_fsg_clock_transfer.py','--arm',arm,
                 '--samples',str(n),'--output',str(out)]
            (out.parent/(arm+'_command.json')).write_text(json.dumps(cmd,indent=2))
            with (out.parent/(arm+'.log')).open('x') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            m=json.loads((out/'summary.json').read_text());assert m['complete']
            for k in ['sources','checkpoint_sha256']:assert m[k]==parity[k],k
            if n==8:
                for k in ['noise_sha256','label_sha256']:assert m[k]==parity[k],k
            print(arm,n,'sampling checks passed',flush=True)
        out=root/'quality'/arm
        cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py',
             '--branch',arm+'='+str(out/'samples.npz'),'--output',str(out/'fid.csv'),
             '--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')]
        with (out/'fid.log').open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        (out/'evaluation_complete.json').write_text(json.dumps({'complete':True,'command':cmd}))
        print(arm,(out/'fid.json').read_text(),flush=True)


if __name__=='__main__':main()
