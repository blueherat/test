"""Frozen per-family smoke/parity and five-arm screen with ADM FID."""
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
    p.add_argument('--gpu',required=True)
    a=p.parse_args()
    root=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_clock_quality_20260908')/a.family
    root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
    common=['--family',a.family,'--global-seed','202609411','--sample-rng-mode','continuous',
            '--batch-size','8','--diagnostic-samples','0','--precision','fp32','--device','cuda:0']
    if a.family=='ig':
        common+=['--internal-head','depth4=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt',
                 '--ig-depths','4','--ig-gamma-segments','.25:.6,.5:.7,1:0']
    def run(cmd,log):
        with log.open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    def sample(name,mode,n,steps=40,original=False):
        out=root/name
        method='closed' if name.startswith('closed') else 'foresight'
        schedule='' if method=='closed' else '0:5:2,5:5:2,15:5:1'
        script='sample_imagenet100_sit_foresight_fixed_point.py' if original else 'sample_sit_fsg_clock_ablation.py'
        cmd=[sys.executable,'experiments/'+script,*common,'--method',method,'--foresight-schedule',schedule,
             '--num-samples',str(n),'--num-steps',str(steps),'--output-dir',str(out)]
        if not original:cmd+=['--clock-mode',mode]
        run(cmd,root/(name+'.log'))
        return json.loads((out/'sampling_manifest.json').read_text())
    original=sample('smoke_original','long',8,original=True)
    smoke={mode:sample('smoke_'+mode,mode,8) for mode in ['long','short','asynchronous']}
    x=np.load(root/'smoke_original/samples_n8.npz')['arr_0']
    assert np.array_equal(x,np.load(root/'smoke_long/samples_n8.npz')['arr_0'])
    for m in smoke.values():
        for k in ['noise_sha256','label_sha256','model_forward_totals']:
            assert m[k]==original[k],(k,m[k],original[k])
    (root/'smoke_parity.json').write_text(json.dumps({'original_long_pixel_equal':True,'inputs_and_calls_equal':True},indent=2))
    print(a.family,'smoke/parity passed',flush=True)
    manifests=[]
    for name,mode,steps in [('closed40','long',40),('closed50','long',50),('long','long',40),('short','short',40),('asynchronous','asynchronous',40)]:
        meta=sample(name,mode,1000,steps)
        manifests.append(meta)
        for k in ['noise_sha256','label_sha256']:
            assert meta[k]==manifests[0][k]
        if name in ['long','short','asynchronous']:
            assert sum(meta['model_forward_totals'].values())==sum(manifests[1]['model_forward_totals'].values())
        out=root/name
        run(['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py',
             '--reference','/home/zhoushunyu/data/eqvae/imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz',
             '--samples',str(out/'samples_n1000.npz'),'--output',str(out/'fid.json'),
             '--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30'],root/(name+'_fid.log'))
        print(a.family,name,'complete', (out/'fid.json').read_text(),flush=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'paired_inputs':True,'matched_forward_counts':True}))


if __name__=='__main__':main()
