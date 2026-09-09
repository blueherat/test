"""Direct temporal conditional interaction and unconditional PFR components."""
import argparse,gc,json,math,shutil,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as native
from experiments.raev2_pfr_retiming import transport_raev2_state_at_fixed_ou_coordinate

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['ordinary','interaction','unconditional'],required=True);p.add_argument('--samples',type=int,default=1000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);sources={}
 for source in [Path(__file__),Path(native.__file__),ROOT/'experiments/pfr_ou_semigroup_spectrum.py',ROOT/'experiments/raev2_pfr_retiming.py']:
  sources[str(source)]=native.file_sha256(source);shutil.copy2(source,a.output/source.name)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 native.install_raev2_decoder_config_compat();cfg=native.load_config(native.DEFAULT_CONFIG)
 decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False);del decoder.encoder;torch.cuda.empty_cache()
 model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
 grid=native.shifted_time_grid(100,math.sqrt(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base),torch.device('cuda'))
 active_count=int((grid[:-1]>.5).sum());floor=float(cfg.transport.t_eps);lo=float(cfg.guidance.ig.t_min);hi=float(cfg.guidance.ig.t_max)
 gen=torch.Generator(device='cuda').manual_seed(202609413);nh=hashlib.sha256();lh=hashlib.sha256();images=[];full_calls=0;prefix_calls=0;parity_calls=0;prefix_parity=None;started=time.perf_counter()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for start in range(0,a.samples,4):
   count=min(4,a.samples-start);z=torch.randn(count,*cfg.misc.latent_size,generator=gen,device='cuda');y=torch.arange(start,start+count,device='cuda')%1000
   nh.update(z.cpu().numpy().tobytes());lh.update(y.cpu().numpy().tobytes())
   for i in range(100):
    t=float(grid[i]);dt=t-float(grid[i+1]);ts=torch.full((count,),t,device='cuda')
    full,base=model(z,ts,context=y,attn_mask=None);full_calls+=1
    vf=native.clean_to_velocity(full,z,ts,denominator_floor=floor)
    if lo<=t<=hi:
     vb=native.clean_to_velocity(base,z,ts,denominator_floor=floor)
     drift=native.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
    else:drift=vf
    if a.arm!='ordinary' and t>.5:
     h=min(1/32,t-.5);q=z
     tf=torch.full_like(ts,t-h)
     null=torch.full_like(y,1000)
     future=native.evaluate_base_head_only(model,q,tf,context=y,attn_mask=None)
     u=native.evaluate_base_head_only(model,z,ts,context=null,attn_mask=None)
     uf=native.evaluate_base_head_only(model,q,tf,context=null,attn_mask=None);prefix_calls+=3
     if a.samples==8 and start==0 and i==0:
      prefix_parity=True
      for state_,time_,label_,value_ in [(q,tf,y,future),(z,ts,null,u),(q,tf,null,uf)]:
       _,check=model(state_,time_,context=label_,attn_mask=None);parity_calls+=1
       prefix_parity=prefix_parity and torch.equal(value_,check)
      assert prefix_parity
     futurev=native.clean_to_velocity(future,q,tf,denominator_floor=floor)
     uv=native.clean_to_velocity(u,z,ts,denominator_floor=floor)
     ufv=native.clean_to_velocity(uf,q,tf,denominator_floor=floor)
     raw=vb-futurev;unconditional=uv-ufv;interaction=raw-unconditional
     revision=interaction if a.arm=='interaction' else unconditional
     drift=drift+1.78*revision
    z=z-dt*drift
   assert torch.isfinite(z).all()
   images.append(decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy())
   if start==0 or (start+count)%100==0:print(json.dumps({'arm':a.arm,'done':start+count,'seconds':time.perf_counter()-started}),flush=True)
 np.savez(a.output/'samples.npz',arr_0=np.concatenate(images));batches=math.ceil(a.samples/4)
 assert full_calls==100*batches and prefix_calls==(0 if a.arm=='ordinary' else 3*active_count*batches)
 meta=dict(complete=True,arm=a.arm,samples=a.samples,seed=202609413,batch_size=4,h=1/32,rho=1.,revision_noise_time_min=.5,full_calls=full_calls,prefix_calls=prefix_calls,active_steps=active_count,parity_full_calls=parity_calls,prefix_parity=prefix_parity,seconds=time.perf_counter()-started,noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),checkpoint_sha256=native.file_sha256(native.DEFAULT_CHECKPOINT),pixel_sha256=native.file_sha256(a.output/'samples.npz'),sources=sources)
 (a.output/'summary.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
