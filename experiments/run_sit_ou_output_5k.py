"""Fixed independent 5K confirmation, one model/arm per process."""
import argparse,hashlib,json,os,subprocess,sys,shutil
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['v','x'],required=True);p.add_argument('--arm',choices=['time_only','ou'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 data=Path('/data/users/zhoushunyu/eqvae/imagenet_sit_flow')
 strong=data/('runs/sit-s-2_seed0/checkpoints/step_00800000.pt' if a.model=='v' else 'runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt')
 head=data/('multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt' if a.model=='v' else 'weak_difference_x800_depth4_readouts_v1/runs/x800_depth4_velocity/checkpoints/step_00050000.pt')
 root=Path('/home/zhoushunyu/data/eqvae/experiments/sit_ou_output_5k_20260908')/a.model/a.arm;root.mkdir(parents=True,exist_ok=False)
 wrapper=Path('experiments/sample_sit_ou_output_control.py');old=Path('/home/zhoushunyu/data/eqvae/experiments/sit_ou_output_control_20260908')/a.model/a.arm/'pfr_output_manifest.json'
 assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==json.loads(old.read_text())['source_sha256']
 shutil.copy2(__file__,root/'driver_source.py');shutil.copy2('docs/SIT_OU_OUTPUT_5K_PROTOCOL_20260908_ZH.md',root/'protocol.md')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
 common=['--family','ig','--method','closed','--foresight-schedule','','--strong-checkpoint',str(strong),'--internal-head','depth4='+str(head),'--ig-depths','4','--ag-gamma','.35','--num-steps','100','--global-seed','202609423','--sample-rng-mode','continuous','--precision','fp32','--batch-size','8','--diagnostic-samples','0']
 for arm,n,name in [('native',8,'native'),('ordinary',8,'ordinary'),(a.arm,8,'smoke'),(a.arm,5000,'quality')]:
  out=root/name;script='experiments/sample_imagenet100_sit_foresight_fixed_point.py' if arm=='native' else str(wrapper)
  cmd=[sys.executable,script,*common,'--num-samples',str(n),'--output-dir',str(out)]
  if arm!='native':cmd+=['--pfr-arm',arm]
  (root/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (root/(name+'.log')).open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  if name=='ordinary':assert np.array_equal(np.load(out/'samples_n8.npz')['arr_0'],np.load(root/'native/samples_n8.npz')['arr_0'])
  if name=='smoke':assert json.loads((out/'pfr_output_manifest.json').read_text())['prefix_parity']
  if name=='quality':
   assert np.array_equal(np.load(out/'samples_n5000.npz')['arr_0'][:8],np.load(root/'smoke/samples_n8.npz')['arr_0'])
   m=json.loads((out/'sampling_manifest.json').read_text());counts=m['model_forward_totals'];assert counts['pfr_prefix_calls']==31250 and counts['ou_full_calls']==15625
   assert sum(v for k,v in counts.items() if k!='pfr_prefix_calls')==78125
  print(name,'passed',flush=True)
 out=root/'quality';cmd=['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py','--reference',str(data/'adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'),'--samples',str(out/'samples_n5000.npz'),'--output',str(out/'fid.json'),'--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30']
 with (root/'fid.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'complete.json').write_text(json.dumps(dict(complete=True,native_parity=True,prefix_parity=True,quality_prefix_pixel_parity=True,source_matches_1k=True)))
 print((out/'fid.json').read_text(),flush=True)
if __name__=='__main__':main()
