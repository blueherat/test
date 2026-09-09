"""Run fixed paired SiT output controls with native and prefix parity."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['v','x'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 data=Path('/data/users/zhoushunyu/eqvae/imagenet_sit_flow')
 strong=data/('runs/sit-s-2_seed0/checkpoints/step_00800000.pt' if a.model=='v' else 'runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt')
 head=data/('multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt' if a.model=='v' else 'weak_difference_x800_depth4_readouts_v1/runs/x800_depth4_velocity/checkpoints/step_00050000.pt')
 root=Path('/home/zhoushunyu/data/eqvae/experiments/sit_clock_compensator_20260908')/a.model;root.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
 common=['--family','ig','--method','closed','--foresight-schedule','','--strong-checkpoint',str(strong),'--internal-head','depth4='+str(head),'--ig-depths','4','--ag-gamma','.35','--num-steps','100','--global-seed','202609417','--sample-rng-mode','continuous','--precision','fp32','--batch-size','8','--diagnostic-samples','0']
 for arm,n,name in [('native',8,'native'),('ordinary',8,'smoke_ordinary'),('smooth',8,'smoke_smooth'),('compensated',8,'smoke_compensated'),('smooth',1000,'smooth'),('compensated',1000,'compensated')]:
  out=root/name;script='experiments/sample_imagenet100_sit_foresight_fixed_point.py' if arm=='native' else 'experiments/sample_sit_clock_compensator.py'
  cmd=[sys.executable,script,*common,'--num-samples',str(n),'--output-dir',str(out)]
  if arm!='native':cmd+=['--pfr-arm',arm]
  (root/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(name+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  if name=='smoke_ordinary':assert np.array_equal(np.load(out/'samples_n8.npz')['arr_0'],np.load(root/'native/samples_n8.npz')['arr_0'])
  if name in ['smoke_smooth','smoke_compensated']:assert json.loads((out/'pfr_output_manifest.json').read_text())['prefix_parity']
  print(name,'passed',flush=True)
  if n==1000:
   if arm=='compensated':
    m=json.loads((out/'sampling_manifest.json').read_text());r=json.loads((root/'smooth/sampling_manifest.json').read_text())
    for k in ['noise_sha256','label_sha256']:assert m[k]==r[k]
   cmd=['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py','--reference',str(data/'adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'),'--samples',str(out/'samples_n1000.npz'),'--output',str(out/'fid.json'),'--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30']
   with (root/(arm+'_fid.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   print((out/'fid.json').read_text(),flush=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,paired_inputs=True,native_parity=True,prefix_parity=True)))
if __name__=='__main__':main()
