"""Fixed ancestral posterior-mean weak reference, one antithetic pair."""
import argparse,gc,json,math,shutil,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as native

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['ordinary','ordinary150','posterior'],required=True);p.add_argument('--samples',type=int,default=1000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);sources={}
 for source in [Path(__file__),Path(native.__file__),ROOT/'experiments/raev2_pfr_retiming.py']:
  sources[str(source)]=native.file_sha256(source);shutil.copy2(source,a.output/source.name)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 native.install_raev2_decoder_config_compat();cfg=native.load_config(native.DEFAULT_CONFIG)
 decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False);del decoder.encoder;torch.cuda.empty_cache()
 model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
 steps=150 if a.arm=='ordinary150' else 100
 grid=native.shifted_time_grid(steps,math.sqrt(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base),torch.device('cuda'))
 active_count=int((grid[:-1]>.5).sum());floor=float(cfg.transport.t_eps);lo=float(cfg.guidance.ig.t_min);hi=float(cfg.guidance.ig.t_max)
 query_gen=torch.Generator(device='cuda').manual_seed(202609434)
 query_start=native.tensor_sha256(query_gen.get_state())
 gen=torch.Generator(device='cuda').manual_seed(202609413);nh=hashlib.sha256();lh=hashlib.sha256();images=[];full_calls=0;prefix_calls=0;parity_calls=0;prefix_parity=None;started=time.perf_counter()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for start in range(0,a.samples,4):
   count=min(4,a.samples-start);z=torch.randn(count,*cfg.misc.latent_size,generator=gen,device='cuda');y=torch.arange(start,start+count,device='cuda')%1000
   nh.update(z.cpu().numpy().tobytes());lh.update(y.cpu().numpy().tobytes())
   for i in range(steps):
    t=float(grid[i]);dt=t-float(grid[i+1]);ts=torch.full((count,),t,device='cuda')
    full,base=model(z,ts,context=y,attn_mask=None);full_calls+=1
    vf=native.clean_to_velocity(full,z,ts,denominator_floor=floor)
    if lo<=t<=hi:
     vb=native.clean_to_velocity(base,z,ts,denominator_floor=floor)
     drift=native.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
    else:drift=vf
    if a.arm=='posterior' and t>.5:
     future_time=max(.5,t-1/32)
     aa=(1-t)/(1-future_time)*future_time**2/t**2
     bb=(1-future_time)-aa*(1-t)
     vv=future_time**2*(1-(1-t)**2*future_time**2/((1-future_time)**2*t**2))
     assert vv>=0
     center=aa*z+bb*base.float()
     eta=torch.randn(count,*cfg.misc.latent_size,generator=query_gen,device='cuda')
     tf=torch.full_like(ts,future_time)
     futures=[]
     for sign in [1.,-1.]:
      q=center+sign*math.sqrt(vv)*eta
      value=native.evaluate_base_head_only(model,q,tf,context=y,attn_mask=None);prefix_calls+=1
      if a.samples==8 and start==0 and i==0:
       _,check=model(q,tf,context=y,attn_mask=None);parity_calls+=1
       prefix_parity=torch.equal(value,check) and (prefix_parity is not False);assert prefix_parity
      futures.append(value.float())
     future_mean=.5*(futures[0]+futures[1])
     drift=drift-(1.78/t)*(base.float()-future_mean)
    z=z-dt*drift
   assert torch.isfinite(z).all()
   images.append(decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy())
   if start==0 or (start+count)%100==0:print(json.dumps({'arm':a.arm,'done':start+count,'seconds':time.perf_counter()-started}),flush=True)
 np.savez(a.output/'samples.npz',arr_0=np.concatenate(images));batches=math.ceil(a.samples/4)
 assert full_calls==steps*batches and prefix_calls==(2*active_count*batches if a.arm=='posterior' else 0)
 meta=dict(complete=True,steps=steps,query_seed=202609434,query_rng_initial=query_start,query_rng_final=native.tensor_sha256(query_gen.get_state()),antithetic_pairs=1,arm=a.arm,samples=a.samples,seed=202609413,batch_size=4,h=1/32,rho=1.,revision_noise_time_min=.5,full_calls=full_calls,prefix_calls=prefix_calls,active_steps=active_count,parity_full_calls=parity_calls,prefix_parity=prefix_parity,seconds=time.perf_counter()-started,noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),checkpoint_sha256=native.file_sha256(native.DEFAULT_CHECKPOINT),pixel_sha256=native.file_sha256(a.output/'samples.npz'),sources=sources)
 (a.output/'summary.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
