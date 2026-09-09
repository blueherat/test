"""Frozen target Euler-map inversion on FP32 native RAE trajectories; no FID."""
import gc,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as n

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 n.install_raev2_decoder_config_compat();cfg=n.load_config(n.DEFAULT_CONFIG)
 model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
 grid=n.shifted_time_grid(100,math.sqrt(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base),torch.device('cuda'))
 indices=[0,int(torch.argmin(abs(grid[:-1]-.75))),int(torch.argmin(abs(grid[:-1]-.5)))]
 gen=torch.Generator(device='cuda').manual_seed(202609425);rows=[];nh=hashlib.sha256();started=time.perf_counter();calls=0
 floor=float(cfg.transport.t_eps)
 def energy(x):return x.double().flatten(1).square().mean(1)
 for start in (0,4):
  z=torch.randn(4,*cfg.misc.latent_size,generator=gen,device='cuda');nh.update(z.cpu().numpy().tobytes());y=torch.arange(start,start+4,device='cuda')
  for i in range(max(indices)+1):
   t=float(grid[i]);ts=torch.full((4,),t,device='cuda');f,b=model(z,ts,context=y,attn_mask=None)
   vf=n.clean_to_velocity(f,z,ts,denominator_floor=floor);vb=n.clean_to_velocity(b,z,ts,denominator_floor=floor)
   drift=n.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
   if i in indices:
    h=.125;target=z-h*drift;initial=energy(z-h*vb-target);x=z.clone()
    # Solve x-h*B(x,t)=target by Picard, target never recomputed.
    for k in range(9):
     bx=n.evaluate_base_head_only(model,x,ts,context=y,attn_mask=None);calls+=1
     bv=n.clean_to_velocity(bx,x,ts,denominator_floor=floor)
     residual=energy(x-h*bv-target);displacement=energy(x-z)/energy(z)
     for j in range(4):rows.append(dict(sample=start+j,step=i,noise_time=t,iteration=k,residual_ratio=float(residual[j]/initial[j].clamp_min(1e-30)),relative_displacement_squared=float(displacement[j])))
     if k<8:x=target+h*bv
    assert torch.isfinite(x).all()
   z=z-(t-float(grid[i+1]))*drift
 out=ROOT/'experiments/results/terminal_defect_20260908/raev2_anchored_inverse.json'
 record=dict(complete=True,rows=rows,seconds=time.perf_counter()-started,prefix_calls=calls,source_sha256=n.file_sha256(Path(__file__)),checkpoint_sha256=n.file_sha256(n.DEFAULT_CHECKPOINT),noise_sha256=nh.hexdigest(),precision='FP32 TF32',h=.125,seed=202609425,scope='8 native FP32 trajectories, frozen single-Euler target, not multi-step inverse or quality evidence')
 with out.open('x') as f:json.dump(record,f,indent=2)
 for step in indices:
  for k in (0,1,2,4,8):
   v=[r['residual_ratio'] for r in rows if r['step']==step and r['iteration']==k]
   print(dict(step=step,iteration=k,mean_residual_ratio=np.mean(v),max_residual_ratio=max(v)),flush=True)
 print('seconds',record['seconds'],flush=True)
if __name__=='__main__':main()
