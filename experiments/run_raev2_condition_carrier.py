"""Run fixed paired generation only after native production parity passes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_condition_carrier_20260908')


def main():
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    reference = ROOT.parent/'raev2_fsg_clock_transfer_20260908'
    for phase,count in [('smoke',8),('quality',1000)]:
        command = [sys.executable,'-m','experiments.sample_raev2_condition_carrier','--samples',str(count),'--output',str(ROOT/phase)]
        with (ROOT/f'{phase}.log').open('x') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        ref = reference/phase/'ordinary100'
        native = ROOT/phase/'native_ig'
        np.testing.assert_array_equal(np.load(native/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
        baseline = json.loads((ref/'summary.json').read_text())
        for arm in ['native_ig','output_control','condition_carrier']:
            path = ROOT/phase/arm
            result = json.loads((path/'summary.json').read_text())
            assert result['complete'] and result['samples']==count
            for key in ['noise_sha256','label_sha256']:
                assert result[key]==baseline[key]
            if phase == 'quality':
                np.testing.assert_array_equal(np.load(path/'samples.npz')['arr_0'][:8],np.load(ROOT/'smoke'/arm/'samples.npz')['arr_0'])
        print(phase,'native and paired input checks passed',flush=True)
    command = [sys.executable,'experiments/evaluate_raev2_official_samples.py','--output',str(ROOT/'quality/fid.csv'),
               '--batch-size','32','--device','cuda','--feature-cache-dir',str(ROOT/'quality/features')]
    for arm in ['native_ig','output_control','condition_carrier']:
        command += ['--branch',arm+'='+str(ROOT/'quality'/arm/'samples.npz')]
    with (ROOT/'fid.log').open('x') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (ROOT/'complete.json').write_text(json.dumps(dict(complete=True,native_production_parity=True,paired_inputs=True)))
    print((ROOT/'quality/fid.json').read_text(),flush=True)


if __name__ == '__main__':
    main()
