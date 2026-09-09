"""Two predeclared mechanism controls on the existing paired discovery bank."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--family',choices=['cfg','ig'],required=True)
    a=p.parse_args()
    base=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908')
    root=base/'mechanism_controls'/a.family
    root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
    common=['--family',a.family,'--global-seed','202609411','--sample-rng-mode','continuous',
            '--batch-size','8','--diagnostic-samples','0','--precision','fp32','--device','cuda:0',
            '--method','foresight','--num-steps','40','--foresight-schedule','0:5:2,5:5:2,15:5:1']
    if a.family=='ig':
        common+=['--internal-head','depth4=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt',
                 '--ig-depths','4','--ig-gamma-segments','.25:.6,.5:.7,1:0']
    for mode in ['gap','time_only']:
        for n in [8,1000]:
            name=mode if n==1000 else 'smoke_'+mode
            out=root/name
            cmd=[sys.executable,'experiments/sample_sit_fsg_clock_controls.py',*common,
                 '--control-mode',mode,'--num-samples',str(n),'--output-dir',str(out)]
            with (root/(name+'.log')).open('x') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            m=json.loads((out/'sampling_manifest.json').read_text())
            reference=base/a.family/('asynchronous' if n==1000 else 'smoke_asynchronous')/'sampling_manifest.json'
            r=json.loads(reference.read_text())
            for k in ['noise_sha256','label_sha256','model_forward_totals']:
                assert m[k]==r[k],(a.family,mode,n,k)
            if n==8:continue
            with (root/(name+'_fid.log')).open('x') as f:
                subprocess.run(['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py',
                    '--reference','/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz',
                    '--samples',str(out/'samples_n1000.npz'),'--output',str(out/'fid.json'),
                    '--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30'],
                    env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(a.family,mode,(out/'fid.json').read_text(),flush=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'paired_inputs_and_calls':True}))


if __name__=='__main__':main()
