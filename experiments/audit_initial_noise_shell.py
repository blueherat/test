"""Paired radial statistics after frozen two-iteration initial calibration."""
import argparse,csv,gc,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['sit','raev2'],required=True);a=p.parse_args()
 out=Path('/home/zhoushunyu/data/eqvae/experiments/initial_noise_shell_20260908')/a.family;out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 if a.family=='raev2':
  from experiments import sample_raev2_pfr_retiming as n
  n.install_raev2_decoder_config_compat();cfg=n.load_config(n.DEFAULT_CONFIG)
  model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
  ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
  shape=tuple(cfg.misc.latent_size);hashes={'strong':sha(n.DEFAULT_CHECKPOINT)}
  def fields(z,t,y):
   ts=torch.full((len(z),),t,device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    full,base=model(z,ts,context=y,attn_mask=None)
    vf=n.clean_to_velocity(full,z,ts,denominator_floor=float(cfg.transport.t_eps));vb=n.clean_to_velocity(base,z,ts,denominator_floor=float(cfg.transport.t_eps))
    g=-n.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
   return g,-2*vf-g
 else:
  from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import load_repo_modules,runtime_paths
  paths=runtime_paths(ROOT,Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow'),Path('/data/shared/envs/adm-fid/bin/python'));m=load_repo_modules(ROOT)
  sit,metadata=m['load_official_sit_module'](m['DEFAULT_OFFICIAL_SIT_REPO'],verify_source=True)
  model,semantics,_=m['load_sit_field_model'](checkpoint_path=paths['strong'],weights='ema',sit_module=sit,source_metadata=metadata,device=torch.device('cuda'))
  head=m['load_internal_head_for_source'](checkpoint_path=paths['depth4'],name='depth4_v',head_weights='ema',model=model,sit_module=sit,source_checkpoint_path=paths['strong'],source_metadata=metadata,device=torch.device('cuda'))
  shape=(4,32,32);hashes={'strong':sha(paths['strong']),'weak':sha(paths['depth4'])}
  def fields(z,t,y):
   full,hs,_=m['evaluate_source_with_heads'](model,z,torch.full((len(z),),t,device='cuda'),y,heads={'depth4_v':head});gap=.6*(full-hs['depth4_v']);return full+gap,full-gap
 gen=torch.Generator(device='cuda').manual_seed(202609416);noisehash=hashlib.sha256();rows=[];started=time.perf_counter();calls=0
 for start in range(0,32,4):
  z0=torch.randn(4,*shape,generator=gen,device='cuda');y=torch.arange(start,start+4,device='cuda');noisehash.update(z0.cpu().numpy().tobytes())
  d=z0[0].numel();n0=z0.double().flatten(1).square().sum(1);t0=1. if a.family=='raev2' else 0.
  for mode in ['none','short','asynchronous','time_only','common_async']:
   z=z0.clone();H=.025 if mode=='short' else .125
   if mode!='none':
    for _ in range(2):
     g,_=fields(z,t0,y);q=z if mode=='time_only' else z+.025*g
     gf,r=fields(q,t0-H if a.family=='raev2' else t0+H,y);calls+=2
     if mode=='common_async':r=gf
     z=z+.025*(g-r) if mode=='time_only' else q-.025*r
   delta=(z-z0).double().flatten(1);norm=z.double().flatten(1).square().sum(1)
   vals={'shell_before':(n0-d)/(2*d)**.5,'shell_after':(norm-d)/(2*d)**.5,'shell_change':(norm-n0)/(2*d)**.5,'norm_ratio':(norm/n0).sqrt(),'delta_rms':delta.square().mean(1).sqrt(),'radial_displacement':(delta*z0.double().flatten(1)).sum(1)/n0.sqrt()}
   assert all(torch.isfinite(v).all() for v in vals.values())
   if mode=='none':assert torch.equal(norm,n0)
   vals={k:v.cpu().tolist() for k,v in vals.items()}
   for j in range(4):rows.append(dict(family=a.family,mode=mode,sample=start+j,dimension=d,**{k:v[j] for k,v in vals.items()}))
 with (out/'per_sample.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'complete.json').write_text(json.dumps(dict(complete=True,samples=32,seed=202609416,batched_model_calls=calls,seconds=time.perf_counter()-started,noise_sha256=noisehash.hexdigest(),models=hashes,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'docs/INITIAL_NOISE_SHELL_PROBE_20260908_ZH.md')),indent=2))
 print((out/'complete.json').read_text(),flush=True)
if __name__=='__main__':main()
