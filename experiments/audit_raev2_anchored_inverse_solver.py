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
 gen=torch.Generator(device='cuda').manual_seed(202609425);rows=[];transport=[];nh=hashlib.sha256();started=time.perf_counter();calls=0
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
    for arm in ('picard','anderson'):
     x=z.clone();xs=[];fs=[]
     for k in range(9):
      bx=n.evaluate_base_head_only(model,x,ts,context=y,attn_mask=None);calls+=1
      bv=n.clean_to_velocity(bx,x,ts,denominator_floor=floor);fx=target+h*bv
      residual=energy(x-fx);displacement=energy(x-z)/energy(z)
      for j in range(4):rows.append(dict(arm=arm,sample=start+j,step=i,noise_time=t,iteration=k,residual_ratio=float(residual[j]/initial[j].clamp_min(1e-30)),relative_displacement_squared=float(displacement[j])))
      if k<8:
       xs.append(x);fs.append(fx);xs=xs[-4:];fs=fs[-4:]
       if arm=='picard' or len(xs)==1:x=fx
       else:
        residuals=torch.stack([f-v for f,v in zip(fs,xs)],1).double().flatten(2)
        gram=residuals@residuals.transpose(1,2)/residuals.shape[-1]
        size=gram.shape[-1];scale=gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-30)
        regularized=gram+1e-4*scale[:,None,None]*torch.eye(size,device='cuda')
        alpha=torch.linalg.solve(regularized,torch.ones(4,size,1,device='cuda',dtype=torch.float64)).squeeze(-1)
        alpha=alpha/alpha.sum(-1,keepdim=True)
        x=(torch.stack(fs,1)*alpha.to(z.dtype).reshape(4,size,1,1,1)).sum(1)
     assert torch.isfinite(x).all()
     for steps in (1,4,8):
      state=x.clone();baseline=z.clone()
      for j in range(steps):
       stamp=torch.full_like(ts,t-j*h/steps)
       for which in ('state','baseline'):
        v=state if which=='state' else baseline
        clean=n.evaluate_base_head_only(model,v,stamp,context=y,attn_mask=None);calls+=1
        v=v-h/steps*n.clean_to_velocity(clean,v,stamp,denominator_floor=floor)
        if which=='state':state=v
        else:baseline=v
      err=energy(state-target);before=energy(baseline-target)
      for j in range(4):transport.append(dict(arm=arm,sample=start+j,step=i,substeps=steps,residual_ratio_to_initial_euler=float(err[j]/initial[j].clamp_min(1e-30)),residual_ratio_to_unmodified_same_solver=float(err[j]/before[j].clamp_min(1e-30))))
   z=z-(t-float(grid[i+1]))*drift
 out=ROOT/'experiments/results/terminal_defect_20260908/raev2_anchored_inverse_solver.json'
 record=dict(complete=True,rows=rows,transport=transport,anderson_history=4,anderson_relative_ridge=1e-4,seconds=time.perf_counter()-started,prefix_calls=calls,source_sha256=n.file_sha256(Path(__file__)),checkpoint_sha256=n.file_sha256(n.DEFAULT_CHECKPOINT),noise_sha256=nh.hexdigest(),precision='FP32 TF32',h=.125,seed=202609425,scope='8 native FP32 trajectories, frozen single-Euler target, single-Euler inverse checked under refined Base transport; not quality evidence')
 with out.open('x') as f:json.dump(record,f,indent=2)
 for step in indices:
  for arm in ('picard','anderson'):
   v=[r['residual_ratio'] for r in rows if r['step']==step and r['iteration']==8 and r['arm']==arm]
   print(dict(step=step,arm=arm,mean_residual_ratio=np.mean(v)),flush=True)
   for sub in (1,4,8):
    v=[r['residual_ratio_to_unmodified_same_solver'] for r in transport if r['step']==step and r['arm']==arm and r['substeps']==sub]
    print(dict(step=step,arm=arm,substeps=sub,mean_transport_ratio=np.mean(v)),flush=True)
 print('seconds',record['seconds'],flush=True)
if __name__=='__main__':main()
