"""Gaussian rotation stencil annihilating every affine certificate field."""
import argparse,gc,json,hashlib,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.pfr_ou_semigroup_spectrum import ou_degree1_retiming_velocity_defect

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['sit','raev2'],required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 device=torch.device('cuda');rows=[]
 if a.model=='sit':
  from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import load_runtime
  runtime,meta=load_runtime(repo=ROOT,data=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow'),adm_python=Path('/data/shared/envs/adm-fid/bin/python'),device=device,allocator_limit_gib=0.)
  shape=(4,32,32)
  def velocity(z,u,label):return runtime.evaluate_pair(torch.tensor(u,device=device),z,torch.full((4,),label,device=device,dtype=torch.long))[0]
 else:
  from experiments import sample_raev2_pfr_retiming as n
  n.install_raev2_decoder_config_compat();cfg=n.load_config(n.DEFAULT_CONFIG);model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
  ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect();shape=tuple(cfg.misc.latent_size)
  def velocity(z,u,label):
   ts=torch.full((4,),1-u,device=device)
   with torch.autocast('cuda',enabled=False):f,_=model(z,ts,context=torch.full((4,),label,device=device,dtype=torch.long),attn_mask=None)
   return -n.clean_to_velocity(f,z,ts,denominator_floor=float(cfg.transport.t_eps))
 gen=torch.Generator(device=device).manual_seed(202609422);c=2**.5-1;started=time.perf_counter();noisehash=hashlib.sha256()
 for j in range(24):
  xy=torch.randn(2,*shape,device=device,generator=gen);x,y=xy;ys=torch.stack([x,y,(x+y)/2**.5,(x-y)/2**.5]);noisehash.update(xy.cpu().numpy().tobytes())
  for u in [.05,.1,.2]:
   h=1/32;z=ys*(u*u+(1-u)**2)**.5;zf=ys*((u+h)**2+(1-u-h)**2)**.5
   f=velocity(z,u,j);ff=velocity(zf,u+h,j);d=ou_degree1_retiming_velocity_defect(f,ff,z,u,u+h).double()
   residual=d[2]+c*d[3]-d[0]-c*d[1]
   difference=d[0]-d[1]
   rows.append(dict(model=a.model,quartet=j,label=j,data_time=u,residual_energy=float(residual.square().mean()),pair_difference_energy=float(difference.square().mean()),certificate_energy=float(d.square().mean())))
 frame=pd.DataFrame(rows);out=ROOT/'experiments/results/terminal_defect_20260908';frame.to_csv(out/f'ou_affine_response_{a.model}_fp32.csv',index=False,mode='x')
 summary=frame.groupby('data_time')[['residual_energy','pair_difference_energy','certificate_energy']].mean();summary['residual_over_pair_difference']=summary.residual_energy/summary.pair_difference_energy
 record=dict(complete=True,seconds=time.perf_counter()-started,noise_sha256=noisehash.hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),quartets=24,model=a.model,scope='Gaussian coordinate probe, not native rollout; class IDs not semantically paired across models')
 (out/f'ou_affine_response_{a.model}_fp32.json').write_text(json.dumps(record,indent=2));print(summary.to_string());print(json.dumps(record))
if __name__=='__main__':main()
