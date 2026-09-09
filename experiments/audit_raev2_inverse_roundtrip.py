"""Full native discrete inverse and forward roundtrip on known cached trajectories."""
import argparse,gc,inspect,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as n
from experiments import audit_raev2_endpoint_adjoint_response as c

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(4),required=True);args=p.parse_args()
 torch.set_num_threads(4);torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_inverse_roundtrip_20260908')/f'shard{args.shard}'
 root.mkdir(parents=True,exist_ok=False)
 cfg=n.load_config(n.DEFAULT_CONFIG);model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
 sources={str(q):n.file_sha256(q) for q in [Path(__file__),Path(n.__file__),Path(c.__file__),n.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]}
 for i,q in enumerate(sources):shutil.copy2(q,root/f'{i}_{Path(q).name}')
 checkpoint=n.file_sha256(n.DEFAULT_CHECKPOINT);assert checkpoint=='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
 grid=c.time_grid();rows=[];cases=[];inputs={};counts=dict(parity=0,inverse=0,roundtrip=0);started=time.perf_counter()
 def energy(x):return float(x.double().square().mean())
 for label in c.IDS[2*args.shard:2*args.shard+2]:
  folder=c.RESTART/'endpoint_adjoint_response_v1/collect'/f'id{label:04d}';meta=json.loads((folder/'summary.json').read_text())
  path=c.verify(meta['states']);inputs[str(path)]=meta['states']['sha256'];original=np.load(path,mmap_mode='r')
  out=root/f'id{label:04d}';out.mkdir();labels=torch.tensor([label],device='cuda')
  def velocity(x,t,kind):
   ts=torch.tensor([t],device='cuda');f,b=model(x,ts,context=labels,attn_mask=None);counts[kind]+=1
   return (x-c.official_heads(f,b,ts))/max(t,.05)
  for k in range(100):
   z=torch.from_numpy(np.array(original[k:k+1])).cuda();t,s=grid[k:k+2]
   assert c.tensor_hash(z)==meta['state_sha256_by_step'][k]
   assert c.tensor_hash(z-(t-s)*velocity(z,t,'parity'))==meta['state_sha256_by_step'][k+1]
  inverse=np.lib.format.open_memmap(out/'inverse_states.npy',mode='w+',dtype=np.float32,shape=original.shape)
  target=torch.from_numpy(np.array(original[100:101])).cuda();inverse[100]=target[0].cpu().numpy()
  for k in reversed(range(100)):
   t,s=grid[k:k+2];h=t-s;x=target.clone()
   # Fixed32 Picard updates; original preimages are never used by the solver.
   for _ in range(32):x=target+h*velocity(x,t,'inverse')
   residual=x-h*velocity(x,t,'inverse')-target
   if not torch.isfinite(x).all() or not torch.isfinite(residual).all():raise RuntimeError(f'nonfinite inverse {label=} {k=}')
   ref=torch.from_numpy(np.array(original[k:k+1])).cuda()
   rows.append(dict(label=label,step=k,noise_time=t,local_residual_mse=energy(residual),
                    trajectory_error_mse=energy(x-ref),reference_state_mse=energy(ref),
                    inverse_state_sha256=c.tensor_hash(x),target_sha256=c.tensor_hash(target)))
   inverse[k]=x[0].cpu().numpy();target=x
  inverse.flush();noise=target.clone();z=noise.clone()
  for k in range(100):
   t,s=grid[k:k+2];z=z-(t-s)*velocity(z,t,'roundtrip')
  assert torch.isfinite(z).all();np.save(out/'roundtrip_endpoint.npy',z[0].cpu().numpy())
  z0=torch.from_numpy(np.array(original[:1])).cuda();end=torch.from_numpy(np.array(original[100:101])).cuda()
  result=dict(label=label,noise_mse=energy(noise-z0),noise_relative_mse=energy(noise-z0)/energy(z0),
              endpoint_mse=energy(z-end),endpoint_relative_mse=energy(z-end)/energy(end),
              inverse_states_sha256=n.file_sha256(out/'inverse_states.npy'),roundtrip_endpoint_sha256=n.file_sha256(out/'roundtrip_endpoint.npy'))
  cases.append(result);print(json.dumps(dict(**result,seconds=time.perf_counter()-started)),flush=True)
 expected=dict(parity=200,inverse=6600,roundtrip=200);assert counts==expected
 record=dict(complete=True,shard=args.shard,cases=cases,rows=rows,inputs=inputs,sources=sources,
             checkpoint_sha256=checkpoint,counts=counts,seconds=time.perf_counter()-started,
             scope='Known native cached FP32/noTF32 B1 trajectories; all100 discrete steps,32 fixed updates/step. No real-data inverse or quality claim.')
 with (root/'summary.json').open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(dict(complete=True,shard=args.shard,counts=counts,seconds=record['seconds'])),flush=True)
if __name__=='__main__':main()
