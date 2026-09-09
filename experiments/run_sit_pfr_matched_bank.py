"""Fixed same-bank reference comparison with pixel parity and ADM evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['cfg','ig'],required=True)
    p.add_argument('--method',choices=['ordinary','pfr'],required=True)
    p.add_argument('--gpu',required=True)
    a=p.parse_args();assert a.method!='pfr' or a.family=='ig'
    base=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908')
    root=base/'matched_references'/(a.family+'_'+a.method)
    root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
    common=['--family',a.family,'--method','closed','--foresight-schedule','',
            '--global-seed','202609412','--sample-rng-mode','continuous','--batch-size','8',
            '--diagnostic-samples','0','--precision','fp32','--num-steps','50']
    if a.family=='ig':
        common+=['--internal-head','depth4=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt',
                 '--ig-depths','4','--ig-gamma-segments','.25:.6,.5:.7,1:0']
    commands=[]
    for name,method,solver,n in [('parity','ordinary','euler',8),('smoke',a.method,'dopri5',8),('quality',a.method,'dopri5',5000)]:
        out=root/name
        cmd=[sys.executable,'experiments/sample_sit_pfr_matched_bank.py',*common,
             '--reference-method',method,'--reference-solver',solver,'--num-samples',str(n),'--output-dir',str(out)]
        commands.append(cmd);(root/'commands.json').write_text(json.dumps(commands,indent=2))
        with (root/(name+'.log')).open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        if name=='parity':
            original=np.load(base/'confirmation5k'/a.family/'closed50/samples_n5000.npz')['arr_0'][:8]
            assert np.array_equal(original,np.load(out/'samples_n8.npz')['arr_0'])
            print(a.family,a.method,'Euler50 pixel parity passed',flush=True)
        if n==5000:
            m=json.loads((out/'sampling_manifest.json').read_text())
            r=json.loads((base/'confirmation5k'/a.family/'closed50/sampling_manifest.json').read_text())
            for k in ['noise_sha256','label_sha256']:assert m[k]==r[k]
    cmd=['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py',
         '--reference','/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz',
         '--samples',str(out/'samples_n5000.npz'),'--output',str(out/'fid.json'),
         '--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30']
    with (root/'fid.log').open('x') as f:
        subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'pixel_parity':True,'paired_inputs':True,'fid_command':cmd}))
    print(a.family,a.method,(out/'fid.json').read_text(),flush=True)


if __name__=='__main__':main()
