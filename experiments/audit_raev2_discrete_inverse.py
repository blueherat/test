"""Known-preimage inversion of four native Euler steps, no new generated images."""
import gc,inspect,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as n
from experiments import audit_raev2_endpoint_adjoint_response as c

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.cuda.set_device(0)
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_discrete_inverse_20260908');root.mkdir(exist_ok=False)
 cfg=n.load_config(n.DEFAULT_CONFIG);model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 state=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(state['ema'],strict=True);del state;gc.collect()
 sources={str(p):n.file_sha256(p) for p in [Path(__file__),Path(n.__file__),Path(c.__file__),n.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]}
 for i,p in enumerate(sources):shutil.copy2(p,root/f'{i}_{Path(p).name}')
 checkpoint=n.file_sha256(n.DEFAULT_CHECKPOINT);assert checkpoint=='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
 grid=c.time_grid();indices=[0,47,89,99];rows=[];probes=[];inputs={};calls=0;started=time.perf_counter()
 gen=torch.Generator(device='cuda').manual_seed(202609437)
 def energy(x):return float(x.double().square().mean())
 for label in c.IDS:
  folder=c.RESTART/'endpoint_adjoint_response_v1/collect'/f'id{label:04d}'
  meta=json.loads((folder/'summary.json').read_text());path=c.verify(meta['states']);inputs[str(path)]=meta['states']['sha256'];states=np.load(path,mmap_mode='r')
  labels=torch.tensor([label],device='cuda')
  for k in indices:
   t,s=grid[k:k+2];h=t-s;times=torch.tensor([t],device='cuda')
   z=torch.from_numpy(np.array(states[k:k+1])).cuda();target=torch.from_numpy(np.array(states[k+1:k+2])).cuda()
   assert c.tensor_hash(z)==meta['state_sha256_by_step'][k]
   def velocity(x):
    nonlocal calls
    f,b=model(x,times,context=labels,attn_mask=None);calls+=1
    return (x-c.official_heads(f,b,times))/max(t,.05)
   v=velocity(z);assert torch.equal(z-h*v,target)
   direction=torch.randn(z.shape,device='cuda',generator=gen);direction/=direction.square().mean().sqrt()
   for eps in [2**-7,2**-8]:
    dv=(velocity(z+eps*direction).double()-velocity(z-eps*direction).double())/(2*eps)
    jmap=direction.double()-h*dv
    probes.append(dict(label=label,step=k,noise_time=t,eps=eps,picard_directional_gain_squared=energy(h*dv),forward_directional_gain_squared=energy(jmap)))
   x=target.clone();den=energy(target-z);assert den>0
   for iteration in range(33):
    vx=velocity(x);residual=x-h*vx-target
    finite=bool(torch.isfinite(x).all() and torch.isfinite(residual).all())
    rows.append(dict(label=label,step=k,noise_time=t,dt=h,iteration=iteration,finite=finite,
                     residual_ratio=energy(residual)/den if finite else None,preimage_error_ratio=energy(x-z)/den if finite else None))
    if not finite:break
    if iteration<32:x=target+h*vx
  print(json.dumps(dict(label=label,seconds=time.perf_counter()-started)),flush=True)
 result=dict(complete=True,rows=rows,probes=probes,inputs=inputs,sources=sources,checkpoint_sha256=checkpoint,full_calls=calls,seconds=time.perf_counter()-started,seed=202609437,
             scope='8 cached native FP32/noTF32 B1 trajectories; four single-step inversions with known preimages. No global invertibility, real-data inverse or quality claim.')
 with (ROOT/'experiments/results/terminal_defect_20260908/raev2_discrete_inverse.json').open('x') as f:json.dump(result,f,indent=2)
 for k in indices:
  vals=[r for r in rows if r['step']==k and r['iteration']==32 and r['finite']]
  print(json.dumps(dict(step=k,completed=len(vals),median_residual_ratio=float(np.median([r['residual_ratio'] for r in vals])) if vals else None,median_preimage_error_ratio=float(np.median([r['preimage_error_ratio'] for r in vals])) if vals else None)),flush=True)
 print('seconds',result['seconds'],'calls',calls,flush=True)
if __name__=='__main__':main()
