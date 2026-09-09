"""Exact temporal odd/even decomposition on fixed Gaussian probes, not quality."""
import argparse,gc,json,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['sit','raev2'],required=True);p.add_argument('--suffix',default='');a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda');rows=[]
 if a.model=='sit':
  from experiments.run_imagenet100_sit_path_evidence_pfr_bridge import load_runtime
  runtime,meta=load_runtime(repo=ROOT,data=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow'),adm_python=Path('/data/shared/envs/adm-fid/bin/python'),device=device,allocator_limit_gib=0.)
  shape=(4,32,32)
  def velocity(z,u,label):return runtime.evaluate_pair(torch.tensor(u,device=device),z,torch.full((4,),label,device=device,dtype=torch.long))
 else:
  from experiments import sample_raev2_pfr_retiming as n
  cfg=n.load_config(n.DEFAULT_CONFIG);model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
  ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect();shape=tuple(cfg.misc.latent_size)
  meta={'checkpoint_sha256':n.file_sha256(n.DEFAULT_CHECKPOINT)}
  def velocity(z,u,label):
   ts=torch.full((4,),1-u,device=device)
   f,b=model(z,ts,context=torch.full((4,),label,device=device,dtype=torch.long),attn_mask=None)
   return tuple(-n.clean_to_velocity(v,z,ts,denominator_floor=float(cfg.transport.t_eps)) for v in [f,b])
 # SiT load_runtime changes these flags; enforce precision after loading.
 torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
 gen=torch.Generator(device=device).manual_seed(202609426);started=time.perf_counter();noisehash=hashlib.sha256();calls=0
 flat=lambda x:x.double().flatten(1)
 def energy(x):return flat(x).square().mean(1)
 def cosine(x,y):
  xx,yy=flat(x),flat(y);return (xx*yy).sum(1)/(xx.norm(dim=1)*yy.norm(dim=1)).clamp_min(1e-30)
 for j in range(24):
  ys=torch.randn(4,*shape,device=device,generator=gen);noisehash.update(ys.cpu().numpy().tobytes())
  for u in [.05,.1,.2]:
   h=1/32;z=ys*(u*u+(1-u)**2)**.5
   now=velocity(z,u,j);past=velocity(z,u-h,j);future=velocity(z,u+h,j);calls+=3
   for head,k in [('full',0),('base',1)]:
    # Form decomposition in FP64 to separate identity checking from rounding.
    current,previous,following=[x[k].double() for x in [now,past,future]]
    raw=current-following;odd=(previous-following)/2;even=current-(previous+following)/2
    assert torch.allclose(raw,odd+even,atol=1e-12,rtol=1e-12)
    values=dict(raw_energy=energy(raw),odd_energy=energy(odd),even_energy=energy(even),
                even_to_raw_energy=energy(even)/energy(raw).clamp_min(1e-30),
                raw_odd_cosine=cosine(raw,odd),raw_even_cosine=cosine(raw,even),odd_even_cosine=cosine(odd,even))
    values={k:v.cpu().tolist() for k,v in values.items()}
    for i in range(4):rows.append(dict(head=head,sample=4*j+i,label=j,data_time=u,**{k:v[i] for k,v in values.items()}))
 out=ROOT/'experiments/results/terminal_defect_20260908'/f'pfr_temporal_parity_{a.model}{a.suffix}.json'
 result=dict(rows=rows,model=a.model,seconds=time.perf_counter()-started,batch4_pair_calls=calls,noise_sha256=noisehash.hexdigest(),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),metadata=meta,
             scope='96 Gaussian-coordinate probes, FP32/no TF32, three interior times; no native trajectory, endpoint or quality claim.')
 with out.open('x') as f:json.dump(result,f,indent=2,default=str)
 for head in ['full','base']:
  for u in [.05,.1,.2]:
   selected=[r for r in rows if r['head']==head and r['data_time']==u]
   print(json.dumps(dict(head=head,data_time=u,**{k:float(np.mean([r[k] for r in selected])) for k in ['even_to_raw_energy','raw_odd_cosine','raw_even_cosine']})),flush=True)
 print(json.dumps(dict(complete=True,seconds=result['seconds'],calls=calls)),flush=True)
if __name__=='__main__':main()
