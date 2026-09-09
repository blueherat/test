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
 gen=torch.Generator(device='cuda').manual_seed(202609425);rows=[];transport=[];nh=hashlib.sha256();started=time.perf_counter();calls=0;full_calls=0
 floor=float(cfg.transport.t_eps)
 def flow(state,t,steps,label,full=False,reverse=False):
  nonlocal calls,full_calls
  state=state.clone();h=(-.125 if reverse else .125)
  for j in range(steps):
   ts=torch.full((4,),t-j*h/steps,device='cuda')
   if full:
    f,b=model(state,ts,context=label,attn_mask=None);full_calls+=1
    vf=n.clean_to_velocity(f,state,ts,denominator_floor=floor);vb=n.clean_to_velocity(b,state,ts,denominator_floor=floor)
    v=n.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
   else:
    b=n.evaluate_base_head_only(model,state,ts,context=label,attn_mask=None);calls+=1
    v=n.clean_to_velocity(b,state,ts,denominator_floor=floor)
   state=state-h/steps*v
  return state
 def energy(x):return x.double().flatten(1).square().mean(1)
 for start in (0,4):
  z=torch.randn(4,*cfg.misc.latent_size,generator=gen,device='cuda');nh.update(z.cpu().numpy().tobytes());y=torch.arange(start,start+4,device='cuda')
  for i in range(max(indices)+1):
   t=float(grid[i]);ts=torch.full((4,),t,device='cuda');f,b=model(z,ts,context=y,attn_mask=None);full_calls+=1
   vf=n.clean_to_velocity(f,z,ts,denominator_floor=floor);vb=n.clean_to_velocity(b,z,ts,denominator_floor=floor)
   drift=n.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
   if i in indices:
    h=.125;target=flow(z,t,4,y,True)
    refined={steps:flow(z,t,steps,y,True) for steps in (16,32)}
    for reverse_steps in (4,16):
     x=flow(target,t-h,reverse_steps,y,reverse=True)
     assert torch.isfinite(x).all()
     for steps in (4,16,32):
      state=flow(x,t,steps,y);baseline=flow(z,t,steps,y)
      fine_target=target if steps==4 else refined[steps]
      err=energy(state-target);before=energy(baseline-target)
      fine_err=energy(state-fine_target);fine_before=energy(baseline-fine_target)
      for j in range(4):rows.append(dict(sample=start+j,step=i,noise_time=t,reverse_steps=reverse_steps,forward_steps=steps,frozen_target_error_ratio=float(err[j]/before[j].clamp_min(1e-30)),refined_target_error_ratio=float(fine_err[j]/fine_before[j].clamp_min(1e-30)),relative_latent_displacement_squared=float(energy(x-z)[j]/energy(z)[j])))
   z=z-(t-float(grid[i+1]))*drift
 out=ROOT/'experiments/results/terminal_defect_20260908/raev2_anchored_reverse_flow.json'
 record=dict(complete=True,rows=rows,seconds=time.perf_counter()-started,prefix_calls=calls,full_calls=full_calls,source_sha256=n.file_sha256(Path(__file__)),checkpoint_sha256=n.file_sha256(n.DEFAULT_CHECKPOINT),noise_sha256=nh.hexdigest(),precision='FP32 TF32',h=.125,seed=202609425,scope='8 native FP32 trajectories, frozen 4-step IG target, direct reverse Base ODE with 4/16 steps checked under 4/16/32-step transport; not quality evidence')
 with out.open('x') as f:json.dump(record,f,indent=2)
 for step in indices:
  for rev in (4,16):
   for sub in (4,16,32):
    rr=[r for r in rows if r['step']==step and r['reverse_steps']==rev and r['forward_steps']==sub]
    print(dict(step=step,reverse=rev,forward=sub,frozen=np.mean([r['frozen_target_error_ratio'] for r in rr]),refined=np.mean([r['refined_target_error_ratio'] for r in rr])),flush=True)
 print('seconds',record['seconds'],flush=True)
if __name__=='__main__':main()
