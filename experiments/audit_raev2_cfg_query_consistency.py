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
 indices=[0,54,83]
 gen=torch.Generator(device='cuda').manual_seed(202609425);rows=[];nh=hashlib.sha256();started=time.perf_counter();calls=0
 floor=float(cfg.transport.t_eps)
 def field(state,t,label):
  nonlocal calls
  ts=torch.full((4,),t,device='cuda')
  fc,_=model(state,ts,context=label,attn_mask=None)
  fu,_=model(state,ts,context=torch.full_like(label,1000),attn_mask=None);calls+=2
  vc=n.clean_to_velocity(fc,state,ts,denominator_floor=floor)
  vu=n.clean_to_velocity(fu,state,ts,denominator_floor=floor)
  return 2*vc-vu,vu
 def cosine(x,y):
  xx=x.double().flatten(1);yy=y.double().flatten(1)
  return (xx*yy).sum(1)/(xx.norm(dim=1)*yy.norm(dim=1)).clamp_min(1e-30)
 def energy(x):return x.double().flatten(1).square().mean(1)
 for start in (0,4):
  z=torch.randn(4,*cfg.misc.latent_size,generator=gen,device='cuda');nh.update(z.cpu().numpy().tobytes());y=torch.arange(start,start+4,device='cuda')
  for i in range(max(indices)+1):
   t=float(grid[i]);drift,_=field(z,t,y)
   if i in indices:
    h=.025;H=.125;reference=z.clone()
    for j in range(16):
     g,_=field(reference,t-j*H/16,y);reference=reference-H/16*g
    states={'short_state':z-h*drift,'matched_euler':z-H*drift,'rollout16':reference}
    corrections={}
    for arm,state in states.items():
     _,u=field(state,t-H,y);corrections[arm]=h*(u-drift)
    reference_delta=corrections['rollout16'];travel=energy(reference-z).clamp_min(1e-30)
    for arm,state in states.items():
     diff=energy(state-reference)/travel
     err=energy(corrections[arm]-reference_delta)/energy(reference_delta).clamp_min(1e-30)
     cos=cosine(corrections[arm],reference_delta)
     norm=energy(corrections[arm])/energy(reference_delta).clamp_min(1e-30)
     for j in range(4):rows.append(dict(sample=start+j,step=i,noise_time=t,arm=arm,relative_query_error_squared=float(diff[j]),relative_correction_error_squared=float(err[j]),correction_cosine=float(cos[j]),correction_energy_ratio=float(norm[j])))
   z=z-(t-float(grid[i+1]))*drift
 out=ROOT/'experiments/results/terminal_defect_20260908/raev2_cfg_query_consistency.json'
 record=dict(complete=True,rows=rows,seconds=time.perf_counter()-started,full_calls=calls,source_sha256=n.file_sha256(Path(__file__)),checkpoint_sha256=n.file_sha256(n.DEFAULT_CHECKPOINT),noise_sha256=nh.hexdigest(),precision='FP32 TF32',h=.025,H=.125,seed=202609425,scope='8 native FP32 CFG2 trajectories, 16-step reference is not exact; no quality evidence')
 with out.open('x') as f:json.dump(record,f,indent=2)
 for step in indices:
  for arm in ('short_state','matched_euler','rollout16'):
   rr=[r for r in rows if r['step']==step and r['arm']==arm]
   print(dict(step=step,arm=arm,**{k:np.mean([r[k] for r in rr]) for k in ('relative_query_error_squared','relative_correction_error_squared','correction_cosine','correction_energy_ratio')}),flush=True)
 print('seconds',record['seconds'],flush=True)
if __name__=='__main__':main()
