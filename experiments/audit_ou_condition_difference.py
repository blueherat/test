"""Conditional/unconditional OU certificate decomposition geometry."""
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
  def velocity(z,u,label):return runtime.evaluate_pair(torch.tensor(u,device=device),z,torch.full((4,),label,device=device,dtype=torch.long))
 else:
  from experiments import sample_raev2_pfr_retiming as n
  n.install_raev2_decoder_config_compat();cfg=n.load_config(n.DEFAULT_CONFIG);model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
  ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect();shape=tuple(cfg.misc.latent_size)
  def velocity(z,u,label):
   ts=torch.full((4,),1-u,device=device)
   with torch.autocast('cuda',dtype=torch.bfloat16):f,b=model(z,ts,context=torch.full((4,),label,device=device,dtype=torch.long),attn_mask=None)
   return tuple(-n.clean_to_velocity(v,z,ts,denominator_floor=float(cfg.transport.t_eps)) for v in [f,b])
 gen=torch.Generator(device=device).manual_seed(202609424);started=time.perf_counter();noisehash=hashlib.sha256()
 null=100 if a.model=='sit' else 1000
 flat=lambda x:x.double().flatten(1)
 def energy(x):return flat(x).square().mean(1)
 def cosine(x,y):
  xx,yy=flat(x),flat(y);return (xx*yy).sum(1)/(xx.norm(dim=1)*yy.norm(dim=1)).clamp_min(1e-30)
 for j in range(24):
  ys=torch.randn(4,*shape,device=device,generator=gen);noisehash.update(ys.cpu().numpy().tobytes())
  for u in [.05,.1,.2]:
   h=1/32;z=ys*(u*u+(1-u)**2)**.5;zf=ys*((u+h)**2+(1-u-h)**2)**.5
   fc,bc=velocity(z,u,j);fcf,_=velocity(zf,u+h,j);fu,_=velocity(z,u,null);fuf,_=velocity(zf,u+h,null);_,bf=velocity(z,u+h,j)
   dc=ou_degree1_retiming_velocity_defect(fc,fcf,z,u,u+h);du=ou_degree1_retiming_velocity_defect(fu,fuf,z,u,u+h);dp=dc-du;raw=bc-bf
   values=dict(conditional_energy=energy(dc),unconditional_energy=energy(du),posterior_difference_energy=energy(dp),raw_revision_energy=energy(raw),conditional_unconditional_cosine=cosine(dc,du),conditional_posterior_cosine=cosine(dc,dp),raw_conditional_cosine=cosine(raw,dc),raw_posterior_cosine=cosine(raw,dp),raw_unconditional_cosine=cosine(raw,du))
   values={k:v.cpu().tolist() for k,v in values.items()}
   for i in range(4):rows.append(dict(model=a.model,sample=4*j+i,label=j,data_time=u,**{k:v[i] for k,v in values.items()}))
 frame=pd.DataFrame(rows);out=ROOT/'experiments/results/terminal_defect_20260908';frame.to_csv(out/f'ou_condition_difference_{a.model}.csv',index=False,mode='x')
 record=dict(complete=True,seconds=time.perf_counter()-started,noise_sha256=noisehash.hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),samples=96,model=a.model,null_label=null,scope='Gaussian coordinate probes, classes not semantically paired across models, no native rollout or quality test')
 (out/f'ou_condition_difference_{a.model}.json').write_text(json.dumps(record,indent=2));print(frame.groupby('data_time').mean(numeric_only=True).to_string());print(json.dumps(record))
if __name__=='__main__':main()
