"""Independent fixed 5K clock confirmation; no parameter selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--arms',required=True,help='Ordered family:arm entries, comma separated')
    a=p.parse_args()
    root=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908/confirmation5k')
    root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
    for entry in a.arms.split(','):
        family,arm=entry.split(':')
        assert family in ['cfg','ig'] and arm in ['closed50','short','asynchronous']
        out=root/family/arm
        out.parent.mkdir(parents=True,exist_ok=True)
        assert not out.exists()
        common=['--family',family,'--global-seed','202609412','--sample-rng-mode','continuous',
                '--num-samples','5000','--batch-size','8','--diagnostic-samples','0',
                '--precision','fp32','--device','cuda:0','--output-dir',str(out)]
        if family=='ig':
            common+=['--internal-head','depth4=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt',
                     '--ig-depths','4','--ig-gamma-segments','.25:.6,.5:.7,1:0']
        closed=arm=='closed50'
        cmd=[sys.executable,'experiments/sample_sit_fsg_clock_ablation.py',*common,
             '--clock-mode','long' if closed else arm,'--method','closed' if closed else 'foresight',
             '--num-steps','50' if closed else '40','--foresight-schedule','' if closed else '0:5:2,5:5:2,15:5:1']
        with (out.parent/(arm+'.log')).open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(__file__,out/'confirmation_driver.py')
        sampling=json.loads((out/'sampling_manifest.json').read_text())
        fidcmd=['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py',
             '--reference','/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz',
             '--samples',str(out/'samples_n5000.npz'),'--output',str(out/'fid.json'),
             '--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30']
        with (out.parent/(arm+'_fid.log')).open('x') as f:
            subprocess.run(fidcmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        record={'complete':True,'sampling_command':cmd,'fid_command':fidcmd,
                'pixels_sha256':hashlib.sha256((out/'samples_n5000.npz').read_bytes()).hexdigest(),
                'noise_sha256':sampling['noise_sha256'],'label_sha256':sampling['label_sha256']}
        (out/'confirmation_complete.json').write_text(json.dumps(record,indent=2))
        print(entry,(out/'fid.json').read_text(),flush=True)


if __name__=='__main__':main()
