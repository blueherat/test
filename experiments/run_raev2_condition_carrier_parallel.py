"""Finish the live output control and run the candidate on a second GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_condition_carrier_20260908')
OLD_SAMPLER_PID = 388368


def completed(path):
    try:
        record = json.loads(path.read_text())
        return record if record.get('complete') else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    source = Path('experiments/sample_raev2_condition_carrier.py').read_text()
    copy = Path('experiments/sample_raev2_condition_carrier_parallel.py').read_text()
    assert copy == source.replace("arms=['native_ig','output_control','condition_carrier']", "arms=['condition_carrier']")
    cmdline = Path(f'/proc/{OLD_SAMPLER_PID}/cmdline').read_bytes()
    assert b'experiments.sample_raev2_condition_carrier\0' in cmdline
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    command = [sys.executable,'-m','experiments.sample_raev2_condition_carrier_parallel','--samples','1000','--output',str(ROOT/'parallel')]
    (ROOT/'parallel_execution.json').write_text(json.dumps(dict(candidate_gpu=1,original_gpu=0,old_sampler_pid=OLD_SAMPLER_PID,
          command=command,reason='User authorized multiple GPUs; preserve completed arms and stop the old sequential sampler after output_control commits its summary.',
          source_difference='Only arms list; sampling equations unchanged.'),indent=2)+'\n')
    with (ROOT/'parallel.log').open('x') as log:
        worker = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(candidate_pid=worker.pid,gpu=1)),flush=True)
        output = ROOT/'quality/output_control/summary.json'
        while not completed(output):
            if worker.poll() is not None and worker.returncode != 0:
                raise RuntimeError('Parallel candidate failed; original sampler has not been interrupted.')
            if not Path(f'/proc/{OLD_SAMPLER_PID}').exists():
                raise RuntimeError('Original sampler exited before output control completion.')
            time.sleep(2)
        record = completed(output)
        assert record['samples'] == 1000
        assert np.load(output.parent/'samples.npz')['arr_0'].shape == (1000,256,256,3)
        # Stop only after its full output-control archive and summary are present.
        if Path(f'/proc/{OLD_SAMPLER_PID}/cmdline').exists():
            assert b'experiments.sample_raev2_condition_carrier\0' in Path(f'/proc/{OLD_SAMPLER_PID}/cmdline').read_bytes()
            os.kill(OLD_SAMPLER_PID,signal.SIGTERM)
        print('Output control complete; stopped sequential worker before duplicate candidate generation.',flush=True)
        while worker.poll() is None:
            time.sleep(2)
        assert worker.returncode == 0
    destination = ROOT/'quality/condition_carrier'
    if destination.exists():
        destination.rmdir()  # Only an empty, unused directory may be removed.
    (ROOT/'parallel/condition_carrier').rename(destination)
    baseline = json.loads((ROOT/'quality/native_ig/summary.json').read_text())
    for arm in ['native_ig','output_control','condition_carrier']:
        path = ROOT/'quality'/arm
        summary = completed(path/'summary.json')
        assert summary and summary['samples'] == 1000
        for key in ['noise_sha256','label_sha256']:
            assert summary[key] == baseline[key]
        np.testing.assert_array_equal(np.load(path/'samples.npz')['arr_0'][:8],np.load(ROOT/'smoke'/arm/'samples.npz')['arr_0'])
    reference = ROOT.parent/'raev2_fsg_clock_transfer_20260908/quality/ordinary100/samples.npz'
    np.testing.assert_array_equal(np.load(ROOT/'quality/native_ig/samples.npz')['arr_0'],np.load(reference)['arr_0'])
    command = [sys.executable,'experiments/evaluate_raev2_official_samples.py','--output',str(ROOT/'quality/fid.csv'),
               '--batch-size','32','--device','cuda','--feature-cache-dir',str(ROOT/'quality/features')]
    for arm in ['native_ig','output_control','condition_carrier']:
        command += ['--branch',arm+'='+str(ROOT/'quality'/arm/'samples.npz')]
    with (ROOT/'fid.log').open('x') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (ROOT/'complete.json').write_text(json.dumps(dict(complete=True,native_production_parity=True,paired_inputs=True,execution='parallel_candidate'))+'\n')
    print((ROOT/'quality/fid.json').read_text(),flush=True)


if __name__ == '__main__':
    main()
