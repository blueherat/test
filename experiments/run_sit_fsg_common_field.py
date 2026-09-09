"""Frozen common-field checks and four-arm discovery-bank comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['cfg','ig'],required=True);p.add_argument('--gpu',required=True)
    a=p.parse_args()
    base=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908')
    root=base/'common_field'/a.family;root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
    common=['--family',a.family,'--method','foresight','--global-seed','202609411',
            '--sample-rng-mode','continuous','--batch-size','8','--precision','fp32',
            '--diagnostic-samples','0','--num-steps','40','--foresight-schedule','0:5:2,5:5:2,15:5:1']
    if a.family=='ig':
        common+=['--internal-head','depth4=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt',
                 '--ig-depths','4','--ig-gamma-segments','.25:.6,.5:.7,1:0']
    for mode in ['short','asynchronous']:
        for n in [8,1000]:
            name='smoke_'+mode if n==8 else mode;out=root/name
            cmd=[sys.executable,'experiments/sample_sit_fsg_common_field.py',*common,
                 '--clock-mode',mode,'--num-samples',str(n),'--output-dir',str(out)]
            (root/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
            with (root/(name+'.log')).open('x') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            m=json.loads((out/'sampling_manifest.json').read_text())
            r=json.loads((base/a.family/name/'sampling_manifest.json').read_text())
            for k in ['noise_sha256','label_sha256']:assert m[k]==r[k]
            assert sum(m['model_forward_totals'].values())==sum(r['model_forward_totals'].values())
            if n==8:continue
            cmd=['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py',
                 '--reference','/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz',
                 '--samples',str(out/'samples_n1000.npz'),'--output',str(out/'fid.json'),
                 '--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30']
            with (root/(mode+'_fid.log')).open('x') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(a.family,mode,(out/'fid.json').read_text(),flush=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'paired_inputs':True,'matched_total_calls':True}))


if __name__=='__main__':main()
