"""Check old native/raw pixels, then fixed Full-field PFR pilot and evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['full','full_pfr'],required=True)
    p.add_argument('--gpu',required=True)
    a=p.parse_args()
    data=Path('/home/zhoushunyu/data/eqvae/experiments')
    root=data/'raev2_pfr_working_point_20260908'/a.arm
    root.mkdir(parents=True,exist_ok=False)
    native=data/'raev2_fsg_clock_transfer_20260908/smoke/ordinary100'
    raw=data/'raev2_canonical_pfr_query_20260908/time_only/smoke'
    ordinary=data/'raev2_fsg_clock_transfer_20260908/quality/ordinary100'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for arm,n,part,ref in [('native_ig',8,'native',native),('raw_ig',8,'raw',raw),
                          (a.arm,8,'smoke',native),(a.arm,1000,'quality',ordinary)]:
        out=root/part
        cmd=[sys.executable,'experiments/sample_raev2_pfr_working_point.py','--arm',arm,'--samples',str(n),'--output',str(out)]
        (root/(part+'_command.json')).write_text(json.dumps(cmd,indent=2))
        with (root/(part+'.log')).open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        m=json.loads((out/'summary.json').read_text());r=json.loads((ref/'summary.json').read_text())
        assert m['complete']
        for k in ['noise_sha256','label_sha256','checkpoint_sha256']:
            assert m[k]==r[k],k
        if part in ['native','raw']:
            assert np.array_equal(np.load(out/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
        if n==8 and arm in ['raw_ig','full_pfr']:
            assert m['prefix_parity'] and m['parity_full_calls']==1
        if part=='quality':
            assert np.array_equal(np.load(out/'samples.npz')['arr_0'][:8],np.load(root/'smoke/samples.npz')['arr_0'])
        print(part,'passed',flush=True)
    q=root/'quality'
    cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',a.arm+'='+str(q/'samples.npz'),
         '--output',str(q/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(q/'features')]
    with (root/'fid.log').open('x') as f:
        subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    (root/'complete.json').write_text(json.dumps(dict(complete=True,native_parity=True,raw_parity=True,paired_inputs=True)))
    print((q/'fid.json').read_text(),flush=True)


if __name__=='__main__':main()
