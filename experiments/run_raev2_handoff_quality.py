"""Parity gates before a single fixed1K handoff evaluation."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np

def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments')
    out=root/'raev2_handoff_quality_20260908';out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    refs={'native':root/'raev2_fsg_clock_transfer_20260908/smoke/ordinary100',
          'full':root/'raev2_pfr_working_point_20260908/full/smoke'}
    for part,arm,n in [('pilot','write_drop',8),('native','native_ig',8),('full','full',8),('smoke','write_drop',8),('quality','write_drop',1000)]:
        dest=out/part
        cmd=[sys.executable,'-m','experiments.sample_raev2_handoff_quality','--arm',arm,'--samples',str(n),'--output',str(dest)]
        if part=='pilot':cmd+=['--pilot']
        (out/f'{part}_command.json').write_text(json.dumps(cmd,indent=2))
        with (out/f'{part}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        m=json.loads((dest/'summary.json').read_text());assert m['complete']
        pix=np.load(dest/'samples.npz')['arr_0']
        if part=='pilot':
            old=np.load(root/'fsg_condition_handoff_20260908/calibrate_drop.npz')
            np.testing.assert_array_equal(pix,old['pixels'][:8])
            np.testing.assert_array_equal(np.load(dest/'written.npy'),old['written'][:8])
        if part in refs:
            ref=refs[part];previous=json.loads((ref/'summary.json').read_text())
            np.testing.assert_array_equal(pix,np.load(ref/'samples.npz')['arr_0'])
            for key in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[key]==previous[key]
        if part=='quality':
            np.testing.assert_array_equal(pix[:8],np.load(out/'smoke/samples.npz')['arr_0'])
            old=json.loads((root/'raev2_pfr_working_point_20260908/full/quality/summary.json').read_text())
            for key in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[key]==old[key]
        print(part,'passed',flush=True)
    q=out/'quality'
    cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch','write_drop='+str(q/'samples.npz'),
         '--output',str(q/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(q/'features')]
    with (out/'fid.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'complete.json').write_text(json.dumps(dict(complete=True,pilot_parity=True,native_parity=True,full_parity=True,paired_inputs=True)))
    print((q/'fid.json').read_text(),flush=True)

if __name__=='__main__':main()
