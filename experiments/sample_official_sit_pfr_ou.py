"""Official SiT: raw PFR versus inherited norm-preserving strong OU direction."""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.audit_official_sit_pfr_interface import prefix
from experiments.raev2_training_core import file_sha256
from experiments.raev2_pfr_retiming import transport_raev2_state_at_fixed_ou_coordinate, raev2_ou_degree1_velocity_defect
from experiments.pfr_ou_semigroup_controls import split_raw_revision_against_ou_degree1
from experiments.pfr_retiming_controls import rms_match_per_sample

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['ordinary100','raw','ou'],required=True);p.add_argument('--samples',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4);torch.cuda.set_device(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 repo=ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT'
 ck=Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')
 source_paths=[Path(__file__),ROOT/'experiments/audit_official_sit_pfr_interface.py',ROOT/'experiments/run_internal_guidance_sit_audit.py',repo/'models/sit.py',repo/'samplers.py']
 source_paths += [ROOT/'experiments'/name for name in ['raev2_pfr_retiming.py','pfr_ou_semigroup_spectrum.py','pfr_ou_semigroup_controls.py','pfr_retiming_controls.py','internal_guidance_path_extrapolation.py']]
 sources={str(p):file_sha256(p) for p in source_paths}
 for p in source_paths:shutil.copy2(p,a.output/p.name)
 checkpoint_sha=file_sha256(ck);assert checkpoint_sha=='a7f4eb9f417295a14e5063b70020ab3f55b60a4905ccda7b244343deaf9840cd'
 model,meta=load_model(repo=repo,checkpoint_path=ck,model_name='SiT-XL/2',encoder_depth=8,state_key='ema',device=torch.device('cuda'))
 from diffusers import AutoencoderKL
 vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).cuda().eval().requires_grad_(False)
 vh=hashlib.sha256()
 for name,value in sorted(vae.state_dict().items()):vh.update(name.encode());vh.update(value.cpu().contiguous().numpy().tobytes())
 steps=100
 grid=torch.linspace(1,0,steps+1,dtype=torch.float64)
 gen=torch.Generator(device='cuda').manual_seed(202609428)
 nh=hashlib.sha256();lh=hashlib.sha256();images=[];calls=0;pcalls=0;ou_calls=0;parity_calls=0;prefix_parity=None;started=time.perf_counter()
 for start in range(0,a.samples,4):
  n=min(4,a.samples-start);z=torch.randn(n,4,32,32,generator=gen,device='cuda');ys=torch.arange(start,start+n,device='cuda')%1000
  nh.update(z.cpu().numpy().tobytes());lh.update(ys.cpu().numpy().tobytes());z=z.double()
  for t,s in zip(grid[:-1],grid[1:]):
   ts=(torch.ones(n,device='cuda',dtype=torch.float64)*t).float()
   full,base,_=model(z.float(),ts,ys);calls+=1
   drift=base+1.35*(full-base)
   if a.arm!='ordinary100' and float(t)>.5:
    tf=torch.full((n,),max(.5,float(t)-1/32),device='cuda')
    future=prefix(model,z.float(),tf,ys);pcalls+=1
    if a.samples==8 and start==0 and float(t)==1.:
     _,check,_=model(z.float(),tf,ys);parity_calls+=1
     prefix_parity=torch.equal(future,check);assert prefix_parity
    revision=base-future
    if float(t)>.75:
     q=transport_raev2_state_at_fixed_ou_coordinate(z.float(),ts,tf)
     sf,_,_=model(q,tf,ys);calls+=1;ou_calls+=1
     if a.arm=='ou':
      axis=raev2_ou_degree1_velocity_defect(full,sf,z.float(),ts,tf)
      revision=rms_match_per_sample(split_raw_revision_against_ou_degree1(revision,axis).common,revision)
    drift=drift+1.35*revision
   z=z+(s-t)*drift.double()
  assert torch.isfinite(z).all()
  decoded=vae.decode(z.float()/0.18215).sample
  pixels=(255.*((decoded+1)/2.)).clamp(0,255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
  images.append(pixels)
  if (start+n)%100==0 or start==0:print(json.dumps(dict(done=start+n,seconds=time.perf_counter()-started)),flush=True)
 np.savez(a.output/'samples.npz',arr_0=np.concatenate(images))
 assert calls==(100 if a.arm=='ordinary100' else 125)*((a.samples+3)//4)
 assert pcalls==(50*((a.samples+3)//4) if a.arm!='ordinary100' else 0)
 result=dict(complete=True,arm=a.arm,samples=a.samples,batch_size=4,seed=202609428,steps=steps,scale=1.35,horizon=1/32,
             full_calls=calls,prefix_calls=pcalls,ou_full_calls=ou_calls,parity_full_calls=parity_calls,prefix_parity=prefix_parity,seconds=time.perf_counter()-started,noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),
             vae_state_sha256=vh.hexdigest(),checkpoint_sha256=checkpoint_sha,sources=sources,metadata=meta,pixel_sha256=file_sha256(a.output/'samples.npz'))
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
